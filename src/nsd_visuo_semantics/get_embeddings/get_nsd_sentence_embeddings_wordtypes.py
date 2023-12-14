import os, pickle, h5py, random
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import sentence_embeddings_sanity_check, get_word_type_from_string
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings


def get_nsd_sentence_embeddings_wordtypes(embedding_model_type, captions_to_embed_path,
                                          word_types_to_use, concat_five_captions, max_n_words_per_caption,
                                          h5_dataset_path, OVERWRITE):
    '''
    Concatenates the coco categories into a string, and throws that into a sentence embedder.
    There is the option to only keep the coco categories that are also present/absent in the captions.
    embedding_model_type: str, the model to use. See embedding_models_zoo.py for options.
    captions_to_embed_path: str, path to the pickle file containing the captions of nsd.
    h5_dataset_path: str, path to the h5 dataset containing the images and categories of ms-coco/nsd.
    concat_five_captions: bool, if True, get a single embedding for all words from the 5 captions. Otherwise, one embed per caption, and we then take the mean.
    OVERWRITE: bool, if True, overwrite existing embeddings.
    '''

    print(f"GATHERING CATEGORY EMBEDDINGS FOR: {embedding_model_type}\n "
          f"ON: {captions_to_embed_path}\n "
          f" FOR WORD TYPES: {word_types_to_use}\n "
          f" WITH MAX_N_WORDS_PER_CAPTION: {max_n_words_per_caption}\n "
          f" CONCATENATE CAPTIONS: {concat_five_captions}\n") 

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 0

    METRIC = 'correlation'

    save_test_imgs_to = "../results_dir/_check_imgs"
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_test_imgs_to, exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    if 'ms_coco_nsd_captions_test.pkl' in captions_to_embed_path:
        prefix = 'nsd'
    else:
        FINAL_CHECK = 0  # not implemented yet
        prefix = captions_to_embed_path.split('/')[-1].split('.')[0]

    save_name = f"{prefix}_{embedding_model_type}_{f'max{max_n_words_per_caption}words_' if max_n_words_per_caption else ''}{'concat5caps_' if concat_five_captions else 'mean_'}WORDTYPE_embeddings"

    if os.path.exists(f"{save_embeddings_to}/{save_name}_{word_types_to_use[-1]}s.pkl") and not OVERWRITE:
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}_{word_types_to_use[-1]}s.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            sentence_embeddings_sanity_check(embedding_model_type, embedding_model, METRIC, save_test_imgs_to)

        if GET_EMBEDDINGS:
            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            n_nsd_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)
            mean_embeddings_all = {wt: np.empty((n_nsd_elements, dummy_embeddings.shape[-1])) for wt in word_types_to_use}
            img_words_all = {wt: [] for wt in word_types_to_use}
            
            for i in range(n_nsd_elements):
                if i % 100 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                these_captions = loaded_captions[i]

                if not isinstance(these_captions, list):
                    # needed if we are using a single caption per image
                    # in that case, we have a string and convert it to a list
                    # with a single element
                    these_captions = [these_captions]

                for wt in word_types_to_use:
                    these_embeddings = np.empty((len(these_captions), dummy_embeddings.shape[-1]))
                    if concat_five_captions:
                        these_words = []
                        for c, cap in enumerate(these_captions):
                            these_words += get_word_type_from_string(cap, wt)
                        if max_n_words_per_caption:
                            if len(these_words) > max_n_words_per_caption:
                                these_words = random.sample(these_words, max_n_words_per_caption)
                        these_words_string = " ".join(these_words)
                        img_words_all[wt].append(these_words_string)
                        this_embedding = get_embeddings(these_words_string, embedding_model, embedding_model_type)
                        mean_embeddings_all[wt][i] = this_embedding
                    else:
                        for c, cap in enumerate(these_captions):
                            these_words = get_word_type_from_string(cap, wt)
                            if max_n_words_per_caption:
                                if len(these_words) > max_n_words_per_caption:
                                    these_words = random.sample(these_words, max_n_words_per_caption)
                            these_words_string = " ".join(these_words)
                            these_embeddings[c] = get_embeddings(these_words_string, embedding_model, embedding_model_type)
                            img_words_all[wt].append(these_words_string)
                        mean_embeddings_all[wt][i] = these_embeddings.mean(axis=0)

            for wt in word_types_to_use:
                with open(f"{save_embeddings_to}/{save_name}_{wt}s.pkl", "wb") as fp:
                    pickle.dump(mean_embeddings_all[wt], fp)
                with open(f"{save_embeddings_to}/{save_name}_{wt}s_per_image.pkl", "wb") as fp:
                    pickle.dump(img_words_all[wt], fp)
            

    if FINAL_CHECK:
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            for wt in word_types_to_use:
                with open(f"{save_embeddings_to}/{save_name}_{wt}.pkl", "rb") as fp:
                    loaded_mean_embeddings = pickle.load(fp)
                with open(f"{save_embeddings_to}/{save_name}_{wt}_per_image.pkl", "rb") as fp:
                    loaded_words_per_image = pickle.load(fp)

                for i in range(0, total_n_stims, step_size):
                    plt.imshow(h5_dataset["test"]["data"][i])
                    plt.title(
                        f"{loaded_captions[i][0]}\n"
                        f"{loaded_words_per_image[i]}\n"
                        f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                    )
                    plt.savefig(f"{save_test_imgs_to}/{save_name}_{wt}_check_{i}.png")
                    plt.close()
