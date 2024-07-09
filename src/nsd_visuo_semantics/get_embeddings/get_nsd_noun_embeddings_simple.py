import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import correlation
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import load_word_vectors, get_word_embedding
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import get_word_type_from_string


def get_nsd_noun_embeddings_simple(EMBEDDING_TYPE, h5_dataset_path, 
                                   fasttext_embeddings_path, glove_embeddings_path, 
                                   nsd_captions_path, SAVE_PATH, OVERWRITE):
    '''
    Retrieves the embeddings for nouns in the nsd dataset.
    EMBEDDING_TYPE: 'glove' or 'fasttext' or a sentence model
    h5_dataset_path: path to the h5 dataset with the images
    fasttext_embeddings_path: path to the fasttext embeddings
    glove_embeddings_path: path to the glove embeddings
    nasd_captions_path: path to the nsd captions
    OVERWRITE: if True, we overwrite the existing embeddings'''
    
    print(f"GATHERING NOUN EMBEDDINGS \n "
        f"EMBEDDING_TYPE: {EMBEDDING_TYPE} \n "
        f"ON: {nsd_captions_path} \n ") 
    
    CHECK_EMBEDDINGS = 1
    GET_NOUN_EMBEDDINGS = 1
    DO_SANITY_CHECK = 1

    save_embeddings_to = SAVE_PATH
    save_test_imgs_to = f"{save_embeddings_to}/_check_imgs"
    os.makedirs(save_test_imgs_to, exist_ok=1)

    SAVE_SUFFIX = ""

    save_name = f"nsd_{EMBEDDING_TYPE}_NOUNS_embeddings{SAVE_SUFFIX}"

    if not OVERWRITE and os.path.exists(f"{save_embeddings_to}/{save_name}.pkl"):
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    
    else:

        if CHECK_EMBEDDINGS or GET_NOUN_EMBEDDINGS:
            # get all word embeddings
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
            print("Sanity check for embedding relationships (correlation distance)")
            print("correlation_dist(cat, dog)", correlation(get_word_embedding("cat", embeddings, EMBEDDING_TYPE), get_word_embedding("dog", embeddings, EMBEDDING_TYPE)))
            print("correlation_dist(cat, table)", correlation(get_word_embedding("cat", embeddings, EMBEDDING_TYPE), get_word_embedding("table", embeddings, EMBEDDING_TYPE)))
            print("correlation_dist(table, chair)", correlation(get_word_embedding("table", embeddings, EMBEDDING_TYPE), get_word_embedding("chair", embeddings, EMBEDDING_TYPE)))
            print("correlation_dist(table, sky)", correlation(get_word_embedding("table", embeddings, EMBEDDING_TYPE), get_word_embedding("sky", embeddings, EMBEDDING_TYPE)))


        if GET_NOUN_EMBEDDINGS:

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            n_elements = len(loaded_captions)

            img_nouns = [[] for _ in range(n_elements)]  # we will also save all nouns for each image
            final_noun_embeddings = np.empty((n_elements, get_word_embedding("runs", embeddings, EMBEDDING_TYPE).shape[0]))  # fastext embeddings have 300 dimensions
            no_nouns_counter = 0  # we will count the images for which NO nouns were found in ANY of the captions
            skipped_candidates_not_nouns = 0  # we will count the number of candidates classified as nouns, but that are not nouns, or whose meaning is unknown
            skipped_candidates_no_embedding = 0  # we will count the number of nouns do not have a fasttext embedding
            final_skipped_nouns = []  # finally, we will catch any left over "mistakes" after screening as explained above

            for i in range(n_elements):

                if i % 100 == 0:
                    print(f"\rRunning... {i/n_elements*100:.2f}%", end="")

                # get all caption nouns
                for j in range(len(loaded_captions[i])):
                    this_sentence = loaded_captions[i][j]
                    sentence_nouns = get_word_type_from_string(this_sentence, 'noun')
                    [img_nouns[i].append(v) for v in sentence_nouns]

                # for each caption noun, get the embedding
                img_noun_embeddings = []
                for v in img_nouns[i]:
                    try:
                        # if the noun vector exists, use the embedding
                        img_noun_embeddings.append(get_word_embedding(v, embeddings, EMBEDDING_TYPE))
                    except KeyError:
                        # if the noun vector does not exist (e.g. "unpealed"), skip.
                        final_skipped_nouns.append(v)

                if not img_noun_embeddings:
                    # deal images without a noun. We use the embedding for "something"
                    final_noun_embeddings[i] = get_word_embedding("something", embeddings, EMBEDDING_TYPE)
                    no_nouns_counter += 1
                else:
                    final_noun_embeddings[i] = np.mean(np.asarray(img_noun_embeddings), axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb",) as fp:  # Pickling
                pickle.dump(final_noun_embeddings, fp)
            with open(f"{save_embeddings_to}/nsd_nouns_per_image.pkl", "wb") as fp:  # Pickling
                pickle.dump(img_nouns, fp)

            print(f"skipped nouns after screening for spelling mistakes and removing unknown words as described on line 329: {final_skipped_nouns}")
            print(f"words classified as nouns but that are not in fact nouns or are unknown words: {skipped_candidates_not_nouns}")
            print(f"nouns that do not have an embedding in fasttext: {skipped_candidates_no_embedding}")
            print(f"n_imgs with NO noun for ANY caption: {no_nouns_counter}")

        del embeddings, loaded_captions, img_nouns, final_noun_embeddings


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
            with open(f"{save_embeddings_to}/nsd_nouns_per_image{SAVE_SUFFIX}.pkl", "rb") as fp:  # Pickling
                loaded_nouns = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb") as fp:  # Pickling
                loaded_noun_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_nouns[i]}\n"
                    f"Emb shape, min, max, mean: {loaded_noun_embeddings[i].shape, loaded_noun_embeddings[i].min(), loaded_noun_embeddings[i].max(), loaded_noun_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
