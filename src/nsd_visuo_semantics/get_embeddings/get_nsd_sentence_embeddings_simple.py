import os, pickle, h5py
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import sentence_embeddings_sanity_check


def get_nsd_sentence_embeddings_simple(embedding_model_type, captions_to_embed_path, 
                                       h5_dataset_path, SAVE_PATH, OVERWRITE):

    print(f"GATHERING EMBEDDINGS FOR: {embedding_model_type}\n "
          f"ON: {captions_to_embed_path}") 

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 1

    safety_check_metric = 'correlation'

    save_embeddings_to = SAVE_PATH
    save_test_imgs_to = f"{save_embeddings_to}/_check_imgs"
    os.makedirs(save_test_imgs_to, exist_ok=1)

    if 'ms_coco_nsd_captions_test.pkl' in captions_to_embed_path:
        prefix = 'nsd'
    else:
        FINAL_CHECK = 0  # not implemented yet for non-NSD captions
        prefix = captions_to_embed_path.split('/')[-1].split('.')[0]

    save_name = f"{prefix}_{embedding_model_type}_mean_embeddings"

    if os.path.exists(f"{save_embeddings_to}/{save_name}.pkl") and not OVERWRITE:
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            sentence_embeddings_sanity_check(embedding_model_type, embedding_model, safety_check_metric, save_test_imgs_to)

        if GET_EMBEDDINGS:
            if '.pkl' in captions_to_embed_path:
                with open(captions_to_embed_path, "rb") as fp:
                    loaded_captions = pickle.load(fp)
            elif '.npy' in captions_to_embed_path:
                loaded_captions = np.load(captions_to_embed_path, allow_pickle=True)
            else:
                raise ValueError("Captions file format not recognized.")

            n_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)

            mean_embeddings = np.empty((n_elements, dummy_embeddings.shape[-1]))

            for i in range(n_elements):
                if i % 100 == 0:
                    print(f"\rRunning... {i/n_elements*100:.2f}%", end="")

                these_captions = loaded_captions[i]

                if not isinstance(these_captions, list):
                    # needed if we are using a single caption per image
                    # in that case, we have a string and convert it to a list
                    # with a single element
                    these_captions = [these_captions]

                img_embeddings = get_embeddings(these_captions, embedding_model, embedding_model_type)
                mean_embeddings[i] = np.mean(img_embeddings, axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb") as fp:
                pickle.dump(mean_embeddings, fp)


    if FINAL_CHECK:
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb") as fp:
                loaded_mean_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
