import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91


GET_MULTIHOT = 1
DO_SANITY_CHECK = 1

h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"
nsd_captions_path = "./ms_coco_nsd_captions_test.pkl"
save_test_imgs_to = "./_check_imgs"
os.makedirs(save_test_imgs_to, exist_ok=1)
save_to = "../results_dir/saved_embeddings"
os.makedirs("../results_dir", exist_ok=1)
os.makedirs(save_to, exist_ok=1)


if GET_MULTIHOT:

    with h5py.File(h5_dataset_path, "r") as h5_dataset:
        
        n_nsd_elements = h5_dataset["test"]["img_multi_hot"][:].shape[0]
        nsd_multihot = np.empty((n_nsd_elements, h5_dataset["test"]["img_multi_hot"][0].shape[0]))

        for i in range(n_nsd_elements):
            nsd_multihot[i] = h5_dataset["test"]["img_multi_hot"][i]

        with open(f"{save_to}/nsd_multihot.pkl", "wb") as fp:  # Pickling
            pickle.dump(nsd_multihot, fp)


if DO_SANITY_CHECK:

    with h5py.File(h5_dataset_path, "r") as h5_dataset:
        total_n_stims = h5_dataset["test"]["img_multi_hot"][:].shape[-1]
        plot_n_imgs = 10
        step_size = total_n_stims // plot_n_imgs

        with open("./ms_coco_nsd_captions_test.pkl", "rb") as fp:
            loaded_captions = pickle.load(fp)
        with open(f"{save_to}/nsd_multihot.pkl", "rb") as fp:
            loaded_multihot = pickle.load(fp)

        for i in range(0, total_n_stims, step_size):
            plt.imshow(h5_dataset["test"]["data"][i])
            plt.title(
                f"{loaded_captions[i][0]}\n"
                f"{[coco_categories_91[j-1] for j in range(loaded_multihot[i].shape[-1]) if loaded_multihot[i,j] == 1]}\n"
                f"multihot shape, min, max, mean: {loaded_multihot[i].shape, loaded_multihot[i].min(), loaded_multihot[i].max(), loaded_multihot[i].mean()}"
            )
            plt.savefig(
                f"{save_test_imgs_to}/NSD_multihot_check_{i}.png"
            )
            plt.close()
