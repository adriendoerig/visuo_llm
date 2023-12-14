import os
import pickle
import h5py
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nsd_visuo_semantics.get_embeddings.word_lists import verb_adjustments
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import load_word_vectors, get_word_embedding


def get_nsd_allWord_embeddings(EMBEDDING_TYPE, CONCATENATE_EMBEDDINGS,
                               h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE):
    
    print(f"GATHERING ALL WORD EMBEDDINGS \n "
        f"EMBEDDING_TYPE: {EMBEDDING_TYPE} \n "
        f"ON: {nsd_captions_path} \n "
        f"WITH CONCATENATE_EMBEDDINGS: {CONCATENATE_EMBEDDINGS}") 
    
    GET_WORD_EMBEDDINGS = 1
    DO_SANITY_CHECK = 1

    save_test_imgs_to = "../results_dir/_check_imgs"
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_test_imgs_to, exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    save_name = f"nsd_{EMBEDDING_TYPE}_ALLWORDS_{'concat' if CONCATENATE_EMBEDDINGS else 'mean'}_embeddings"

    if OVERWRITE and os.path.exists(f"{save_embeddings_to}/{save_name}.pkl"):
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    
    else:
        
        if GET_WORD_EMBEDDINGS:
            # get all word embeddings
            if EMBEDDING_TYPE == 'fasttext':
                embeddings = load_word_vectors(fasttext_embeddings_path, 'fasttext')
            elif EMBEDDING_TYPE == 'glove':
                embeddings = load_word_vectors(glove_embeddings_path, 'glove')
            else:
                try:
                    from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model
                    embeddings = get_embedding_model(EMBEDDING_TYPE)
                except Exception as e:
                    raise Exception('EMBEDDING_TYPE not understood')

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            n_nsd_elements = len(loaded_captions)
            img_words = [[] for _ in range(n_nsd_elements)]  # we will also save all verbs for each image
            final_allWord_embeddings = np.empty((n_nsd_elements, get_word_embedding("runs", embeddings, EMBEDDING_TYPE).shape[0]))  # fastext embeddings have 300 dimensions

            for i in range(n_nsd_elements):

                if i % 100 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                for j in range(len(loaded_captions[i])):
                    this_sentence = loaded_captions[i][j]
                    sentence_words = nltk.word_tokenize(this_sentence)
                    for n, s in enumerate(sentence_words):
                        if s in verb_adjustments.keys():
                            # some spelling mistakes are made in the captions. Here, we fix them. In addition, some crap is
                            # miscalssified as verbs. We discard these. We also discard verbs that have no embedding (e.g. waterskiing).
                            # at the bottom of the script, we prinit out how many are rejected in this way, etc.
                            if verb_adjustments[s] == "_____not_verb_/_unknown_____":
                                pass
                            elif verb_adjustments[s] == "_____no_embedding_____":
                                pass
                            else:
                                sentence_words[n] = verb_adjustments[s]
                    [img_words[i].append(w) for w in sentence_words]

                img_allWord_embeddings = []
                for w in img_words[i]:
                    try:
                        # if the word exists in fasttext, use the embedding
                        img_allWord_embeddings.append(get_word_embedding(w, embeddings, EMBEDDING_TYPE))
                    except KeyError:
                        # if the word does not exist in fasttext (e.g. "unpealed"), skip.
                        pass
                
                if CONCATENATE_EMBEDDINGS:
                    final_allWord_embeddings[i] = np.concatenate(img_allWord_embeddings)
                else:
                    final_allWord_embeddings[i] = np.mean(np.asarray(img_allWord_embeddings), axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb") as fp:  # Pickling
                pickle.dump(final_allWord_embeddings, fp)
            with open(f"{save_embeddings_to}/nsd_allWords_per_image.pkl", "wb") as fp:  # Pickling
                pickle.dump(img_words, fp)


    if DO_SANITY_CHECK:
        if not os.path.exists(h5_dataset_path):
            raise Exception(
                f"{h5_dataset_path} not found: cannot get images for sanity check. The embedding creation"
                "may still be correct, but we cannot plot embeddings along with images."
            )

        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/nsd_allWords_per_image.pkl", "rb") as fp:  # Pickling
                loaded_allWords = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb") as fp:  # Pickling
                loaded_allWord_mean_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_allWords[i]}\n"
                    f"Emb shape, min, max, mean: {loaded_allWord_mean_embeddings[i].shape, loaded_allWord_mean_embeddings[i].min(), loaded_allWord_mean_embeddings[i].max(), loaded_allWord_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")