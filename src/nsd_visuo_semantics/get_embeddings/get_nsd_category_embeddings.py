import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, correlation
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91, noun_adjustments
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import load_word_vectors, get_word_embedding
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import get_words_from_multihot


def get_nsd_category_embeddings(EMBEDDING_TYPE, h5_dataset_path, 
                                fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE):
    '''
    Retrieves the embeddings for nouns in the nsd dataset.
    EMBEDDING_TYPE: 'glove' or 'fasttext'
    MATCH_TO_COCO_CATEGORY_NOUNS: if 'positive', we only use nouns that have an embedding CLOSE to COCO category nouns. 
                                  ff 'negative', we only use nouns that have an embedding FAR to COCO category nouns.
                                  if None, we use all nouns.
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

    save_test_imgs_to = "../results_dir/_check_imgs"
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_test_imgs_to, exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)


    save_name = f"nsd_{EMBEDDING_TYPE}_CATEGORY_embeddings"

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
            n_nsd_elements = len(loaded_captions)

            with h5py.File(h5_dataset_path,'r') as f:
                loaded_multihot_labels = f['test']['img_multi_hot'][:]

            coco_cat_embeds = {}
            for c, cat in enumerate(coco_categories_91):
                if cat == "baseball-bat":
                    baseball_vector = get_word_embedding('baseball', embeddings, EMBEDDING_TYPE)
                    bat_vector = get_word_embedding('bat', embeddings, EMBEDDING_TYPE)
                    coco_cat_embeds[cat] = (baseball_vector + bat_vector) / 2
                elif cat == "baseball-glove":
                    baseball_vector = get_word_embedding('baseball', embeddings, EMBEDDING_TYPE)
                    glove_vector = get_word_embedding('glove', embeddings, EMBEDDING_TYPE)
                    coco_cat_embeds[cat] = (baseball_vector + glove_vector) / 2
                elif cat == "tennis-racket":
                    tennis_vector = get_word_embedding('tennis', embeddings, EMBEDDING_TYPE)
                    racket_vector = get_word_embedding('racket', embeddings, EMBEDDING_TYPE)
                    coco_cat_embeds[cat] = (tennis_vector + racket_vector) / 2
                else:
                    coco_cat_embeds[cat] = get_word_embedding(cat, embeddings, EMBEDDING_TYPE)

            final_categ_embeddings = np.empty((n_nsd_elements, get_word_embedding("runs", embeddings, EMBEDDING_TYPE).shape[0]))
            final_categ_words = []

            for i in range(n_nsd_elements):

                if i % 100 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                img_category_labels = loaded_multihot_labels[i]
                these_embeds = []
                these_words = get_words_from_multihot(img_category_labels, coco_categories_91)
                these_embeds = [coco_cat_embeds[w] for w in these_words]
                final_categ_words.append(these_words)
                final_categ_embeddings[i] = np.mean(np.asarray(these_embeds), axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb",) as fp:  # Pickling
                pickle.dump(final_categ_embeddings, fp)
            with open(f"{save_embeddings_to}/nsd_categ_words_per_image.pkl", "wb") as fp:  # Pickling
                pickle.dump(final_categ_words, fp)

    if DO_SANITY_CHECK:
        print("Sanity check for embeddings")
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb",) as fp:
                loaded_mean_embeddings = pickle.load(fp)
            with open(f"{save_embeddings_to}/nsd_categ_words_per_image.pkl", "rb") as fp:
                loaded_cats_per_image = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_cats_per_image[i]}\n"
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
