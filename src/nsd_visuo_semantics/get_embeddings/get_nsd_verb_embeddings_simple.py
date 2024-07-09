import os
import pickle
import h5py
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy.spatial
from nsd_visuo_semantics.get_embeddings.word_lists import verb_adjustments
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import load_word_vectors, get_word_embedding


def get_nsd_verb_embeddings_simple(EMBEDDING_TYPE, h5_dataset_path, 
                                   fasttext_embeddings_path, glove_embeddings_path, 
                                   nsd_captions_path, SAVE_PATH, OVERWRITE):
    
    
    print(f"GATHERING VERB EMBEDDINGS \n "
        f"EMBEDDING_TYPE: {EMBEDDING_TYPE} \n "
        f"ON: {nsd_captions_path} \n ") 
    
    CHECK_EMBEDDINGS = 1
    GET_VERB_EMBEDDINGS = 1
    DO_SANITY_CHECK = 1

    save_embeddings_to = SAVE_PATH
    save_test_imgs_to = f"{save_embeddings_to}/_check_imgs"
    os.makedirs(save_test_imgs_to, exist_ok=1)

    save_name = f"nsd_{EMBEDDING_TYPE}_VERBS_embeddings"
    if not OVERWRITE and os.path.exists(f"{save_embeddings_to}/{save_name}.pkl"):
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    
    else:

        if EMBEDDING_TYPE == 'fasttext':
            embeddings = load_word_vectors(fasttext_embeddings_path, 'fasttext')
        elif EMBEDDING_TYPE == 'glove':
            embeddings = load_word_vectors(glove_embeddings_path, 'glove')
        else:
            try:
                # check if EMBEDDING_TYPE is a sentence transformer. If so, load it.
                from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model
                embeddings = get_embedding_model(EMBEDDING_TYPE)
            except Exception as e:
                raise Exception('EMBEDDING_TYPE not understood')


        if CHECK_EMBEDDINGS:
            # sanity checks
            print("Sanity check for embedding relationships")
            print("cosine_dist(runs, run)", scipy.spatial.distance.cosine(get_word_embedding("runs", embeddings, EMBEDDING_TYPE), get_word_embedding("run", embeddings, EMBEDDING_TYPE)))
            print("cosine_dist(runs, eats)", scipy.spatial.distance.cosine(get_word_embedding("runs", embeddings, EMBEDDING_TYPE), get_word_embedding("eats", embeddings, EMBEDDING_TYPE)))
            print("cosine_dist(eats, eating)", scipy.spatial.distance.cosine(get_word_embedding("eats", embeddings, EMBEDDING_TYPE), get_word_embedding("eating", embeddings, EMBEDDING_TYPE)))
            print("cosine_dist(eats, (is+eating)/2)", scipy.spatial.distance.cosine(get_word_embedding("eats", embeddings, EMBEDDING_TYPE),(get_word_embedding("is", embeddings, EMBEDDING_TYPE)+get_word_embedding("eating", embeddings, EMBEDDING_TYPE))/2,))


        if GET_VERB_EMBEDDINGS:

            def get_verbs_from_string(s):
                tokens = nltk.word_tokenize(s)
                tagged = nltk.pos_tag(tokens)
                return [x[0] for x in tagged if x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]  # tags for verbs

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            n_nsd_elements = len(loaded_captions)
            img_verbs = [[] for _ in range(n_nsd_elements)]  # we will also save all verbs for each image
            final_verb_embeddings = np.empty((n_nsd_elements, get_word_embedding("runs", embeddings, EMBEDDING_TYPE).shape[0]))  # fastext embeddings have 300 dimensions
            no_verbs_counter = 0  # we will count the images for which NO verbs were found in ANY of the captions
            skipped_candidates_not_verbs = 0  # we will count the number of candidates classified as verbs, but that are not verbs, or whose meaning is unknown
            skipped_candidates_no_embedding = 0  # we will count the number of verbs do not have a fasttext embedding
            final_skipped_verbs = []  # finally, we will catch any left over "mistakes" after screening as explained above

            for i in range(n_nsd_elements):

                if i % 100 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                for j in range(len(loaded_captions[i])):
                    this_sentence = loaded_captions[i][j]
                    sentence_verbs = get_verbs_from_string(this_sentence)
                    for n, s in enumerate(sentence_verbs):
                        if s in verb_adjustments.keys():
                            # some spelling mistakes are made in the captions. Here, we fix them. In addition, some crap is
                            # miscalssified as verbs. We discard these. We also discard verbs that have no embedding (e.g. waterskiing).
                            # at the bottom of the script, we print out how many are rejected in this way, etc.
                            if verb_adjustments[s] == "_____not_verb_/_unknown_____":
                                skipped_candidates_not_verbs += 1
                            elif verb_adjustments[s] == "_____no_embedding_____":
                                skipped_candidates_no_embedding += 1
                            else:
                                sentence_verbs[n] = verb_adjustments[s]
                    [img_verbs[i].append(v) for v in sentence_verbs]

                img_verb_embeddings = []
                for v in img_verbs[i]:
                    try:
                        img_verb_embeddings.append(get_word_embedding(v, embeddings, EMBEDDING_TYPE))
                    except KeyError:
                        # if the verb does not exist in fasttext (e.g. "unpealed"), skip.
                        final_skipped_verbs.append(v)

                if not img_verb_embeddings:
                    # usually, sentences without verbs are like "a pot on a table". So, for images with NO verbs in ANY of the
                    # captions, we use "is" as the mean embedding.
                    final_verb_embeddings[i] = get_word_embedding("is", embeddings, EMBEDDING_TYPE)
                    no_verbs_counter += 1
                else:
                    final_verb_embeddings[i] = np.mean(np.asarray(img_verb_embeddings), axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb") as fp:  # Pickling
                pickle.dump(final_verb_embeddings, fp)
            with open(f"{save_embeddings_to}/nsd_verbs_per_image.pkl", "wb") as fp:  # Pickling
                pickle.dump(img_verbs, fp)

            print(f"skipped verbs after screening for spelling mistakes and removing unknown words as described on line 329: {final_skipped_verbs}")
            print(f"words classified as verbs but that are not in fact verbs or are unknown words: {skipped_candidates_not_verbs}")
            print(f"verbs that do not have an embedding in {EMBEDDING_TYPE}: {skipped_candidates_no_embedding}")
            print(f"n_imgs with NO verb for ANY caption: {no_verbs_counter}")

        del embeddings, final_verb_embeddings, img_verbs


    if DO_SANITY_CHECK:
        if not os.path.exists(h5_dataset_path):
            raise Exception(f"{h5_dataset_path} not found: cannot get images for sanity check. The embedding creation may still be correct, but we cannot plot embeddings along with images.")

        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/nsd_verbs_per_image.pkl", "rb") as fp:  # Pickling
                loaded_verbs = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb") as fp:  # Pickling
                loaded_verb_mean_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_verbs[i]}\n"
                    f"Emb shape, min, max, mean: {loaded_verb_mean_embeddings[i].shape, loaded_verb_mean_embeddings[i].min(), loaded_verb_mean_embeddings[i].max(), loaded_verb_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
