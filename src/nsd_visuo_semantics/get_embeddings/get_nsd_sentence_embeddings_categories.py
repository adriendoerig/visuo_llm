import os, pickle, nltk, h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.spatial.distance import cdist
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import sentence_embeddings_sanity_check, get_words_from_multihot, get_all_word_embeds_from_wordlist, get_all_word_embeds_from_caption, filter_categs_to_keep
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings


def get_nsd_sentence_embeddings_categories(embedding_model_type, captions_to_embed_path,
                                           h5_dataset_path, OVERWRITE):
    '''
    Concatenates the coco categories into a string, and throws that into a sentence embedder.
    There is the option to only keep the coco categories that are also present/absent in the captions.
    embedding_model_type: str, the model to use. See embedding_models_zoo.py for options.
    captions_to_embed_path: str, path to the pickle file containing the captions of nsd.
    h5_dataset_path: str, path to the h5 dataset containing the images and categories of ms-coco/nsd.
    OVERWRITE: bool, if True, overwrite existing embeddings.
    '''

    print(f"GATHERING CATEGORY EMBEDDINGS FOR: {embedding_model_type}\n "
          f"ON: {captions_to_embed_path}") 

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 1

    save_test_imgs_to = "../results_dir/_check_imgs"
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_test_imgs_to, exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    METRIC = 'correlation'
    CUTOFF = 0.33

    save_name = f"nsd_{embedding_model_type}_mean_CATEGORY_embeddings"


    with h5py.File(h5_dataset_path,'r') as f:
        loaded_multihot_labels = f['test']['img_multi_hot'][:]


    if os.path.exists(f"{save_embeddings_to}/{save_name}_allCats.pkl") and not OVERWRITE:
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            sentence_embeddings_sanity_check(embedding_model_type, embedding_model, METRIC, save_test_imgs_to)

        if GET_EMBEDDINGS:
            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            n_nsd_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)

            mean_embeddings_all = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))
            mean_embeddings_captionMatchPositive = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))
            mean_embeddings_captionMatchNegative = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))

            cats_per_image = {'ALL CATEGORIES': [], 'CATEGORIES CLOSE TO CAPTION WORD': [], 'CATEGORIES FAR FROM CAPTION WORD': []}

            for i in range(n_nsd_elements):
                if i % 1000 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                img_category_words = get_words_from_multihot(loaded_multihot_labels[i], coco_categories_91)

                # first, we keep all categories
                if len(img_category_words) == 0:
                    all_cat_word_string = "something"
                else:
                    all_cat_word_string = " ".join(img_category_words)
                these_all_cat_embeds = get_embeddings(all_cat_word_string, embedding_model, embedding_model_type)
                mean_embeddings_all[i] = these_all_cat_embeds
                cats_per_image["ALL CATEGORIES"].append(all_cat_word_string)

                # then, we only keep the categories that are also present in the caption
                img_category_word_embeds = get_all_word_embeds_from_wordlist(img_category_words, embedding_model, embedding_model_type)
                img_caption_word_embeds = get_all_word_embeds_from_caption(loaded_captions[i], embedding_model, embedding_model_type)
                words_to_keep_positive = filter_categs_to_keep(img_category_words, img_category_word_embeds, img_caption_word_embeds, 'positive', CUTOFF, METRIC)
                if len(words_to_keep_positive) == 0:
                    word_string_positive = "something"
                else:
                    word_string_positive = " ".join(words_to_keep_positive)
                these_capMatchPos_embeds = get_embeddings(word_string_positive, embedding_model, embedding_model_type)
                mean_embeddings_captionMatchPositive[i] = these_capMatchPos_embeds
                cats_per_image["CATEGORIES CLOSE TO CAPTION WORD"].append(word_string_positive)

                # then, we only keep the categories that are NOT present in the caption.
                words_to_keep_negative = [w for w in img_category_words if w not in words_to_keep_positive]
                if len(words_to_keep_negative) == 0:
                    word_string_negative = "something"
                else:
                    word_string_negative = " ".join(words_to_keep_negative)
                these_capMatchNeg_embeds = get_embeddings(word_string_negative, embedding_model, embedding_model_type)
                mean_embeddings_captionMatchNegative[i] = these_capMatchNeg_embeds
                cats_per_image["CATEGORIES FAR FROM CAPTION WORD"].append(word_string_negative)

            with open(f"{save_embeddings_to}/{save_name}_allCats.pkl", "wb") as fp:
                pickle.dump(mean_embeddings_all, fp)
            with open(f"{save_embeddings_to}/{save_name}_captionMatchPositive.pkl", "wb") as fp:
                pickle.dump(mean_embeddings_captionMatchPositive, fp)
            with open(f"{save_embeddings_to}/{save_name}_captionMatchNegative.pkl", "wb") as fp:
                pickle.dump(mean_embeddings_captionMatchNegative, fp)
            with open(f"{save_embeddings_to}/{save_name}_categs_per_image.pkl", "wb") as fp:
                pickle.dump(cats_per_image, fp)
            

    if FINAL_CHECK:
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}_allCats.pkl", "rb") as fp:
                loaded_mean_embeddings = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}_categs_per_image.pkl", "rb") as fp:
                loaded_cats_per_image = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_cats_per_image['ALL CATEGORIES'][i]}\n"
                    f"{loaded_cats_per_image['CATEGORIES CLOSE TO CAPTION WORD'][i]}\n"
                    f"{loaded_cats_per_image['CATEGORIES FAR FROM CAPTION WORD'][i]}\n"
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
