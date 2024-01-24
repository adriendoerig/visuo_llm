import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy import stats

from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import (
    get_embedding_model,
    get_embeddings,
)
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91

MAKE_DATASET = 1  # loads ms_coco_guse dataset, and copies it in a new dataset, adding embeddings
CHECK_DATASET = 1  # loads newly made dataset and runs a few sanity checks
MAKE_EMBEDDING_RDM_FOR_NSD = 0  # make RDM of cosine distances between embeddings for the whole NSD (here: test set of our version of ms-coco, which contains all nsd images)
LOAD_AND_CHECK_EMBEDDING_RDM_FOR_NSD = (
    0  # load the saved RDM to check it is alright
)

EMBEDDING_MODEL_NAME = "all_mpnet_base_v2"  # , 'USE_CMLM_Base',  'openai_ada2', 'GUSE_transformer',  'GUSE_DAN', 'T5'
precomputed_embeddings_path = "./saved_embeddings"  # if not None, precomputed embeddings should be in precomputed_embeddings_path/{train/val/test}_{MODEL_NAME}_mean_embeddings.pkl

original_dataset_path = (
    "/home/student/a/adoerig/code/datasets/ms_coco_GUSE_fasttext_square256.h5"
)
new_dataset_path = f"/home/student/a/adoerig/code/datasets/ms_coco_square256_GUSE_fasttext_{EMBEDDING_MODEL_NAME}.h5"  # where to save new dataset
save_test_imgs_to = "./_tmp_playground"
os.makedirs(save_test_imgs_to, exist_ok=1)

# declare embedding model and get dummy embeddings (used later to get embedding size)
embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
dummy_sentences = ["Good day how do you do?", "I am at the gates of hell."]
dummy_embeddings = get_embeddings(
    dummy_sentences, embedding_model, EMBEDDING_MODEL_NAME
)

if MAKE_DATASET:
    with h5py.File(original_dataset_path, "r") as orig_dataset:
        # Quick checks that everything is alright with coco classes
        fig, ax = plt.subplots(5, 1, figsize=(7, 7 * 5))
        for i in range(5):
            ax[i].imshow(orig_dataset["train"]["data"][i])
            these_categories = [
                coco_categories_91[lin - 1]
                for lin in np.where(
                    orig_dataset["train"]["img_multi_hot"][i] == 1
                )[0]
            ]
            ax[i].set_title(these_categories)
        plt.savefig(
            f"{save_test_imgs_to}/cocoAdd{EMBEDDING_MODEL_NAME}__check_categs_match_image.png"
        )

        with h5py.File(new_dataset_path, "w") as new_dataset:
            new_dataset.create_dataset("categories", data=coco_categories_91)

            for dataset in ["train", "val", "test"]:
                with open(f"ms_coco_nsd_captions_{dataset}.pkl", "rb") as fp:
                    loaded_captions = pickle.load(fp)
                if precomputed_embeddings_path is not None:
                    with open(
                        f"{precomputed_embeddings_path}/"
                        f'{"nsd" if dataset == "test" else dataset}_{EMBEDDING_MODEL_NAME}_mean_embeddings.pkl',
                        "rb",
                    ) as fp:
                        loaded_embeddings = pickle.load(fp)

                group = new_dataset.create_group(dataset)
                n_samples = orig_dataset[dataset]["data"].shape[0]

                # create structures to copy over original data
                orig_subgroups = [s for s in orig_dataset[dataset]]
                for subgroup in orig_subgroups:
                    group.create_dataset(
                        subgroup,
                        shape=orig_dataset[dataset][subgroup].shape,
                        dtype=orig_dataset[dataset][subgroup].dtype,
                        chunks=True,
                    )
                # for new embedding
                group.create_dataset(
                    EMBEDDING_MODEL_NAME + "_mean_embeddings",
                    shape=(n_samples, dummy_embeddings.shape[-1]),
                    dtype=orig_dataset[dataset][subgroup].dtype,
                    chunks=True,
                )

                for this_idx in range(n_samples):
                    if this_idx % 1000 == 0:
                        print(
                            f"\rCreating {dataset}: {this_idx/n_samples*100}%",
                            end="",
                        )

                    # copy over stuff from original dataset
                    for subgroup in orig_subgroups:
                        new_dataset[dataset][subgroup][
                            this_idx
                        ] = orig_dataset[dataset][subgroup][this_idx]

                    # get embeddings for the current sample and insert into dataset
                    if precomputed_embeddings_path is None:
                        these_embeddings = get_embeddings(
                            loaded_captions[this_idx],
                            embedding_model,
                            EMBEDDING_MODEL_NAME,
                        )
                        new_dataset[dataset][
                            EMBEDDING_MODEL_NAME + "_mean_embeddings"
                        ][this_idx] = np.mean(these_embeddings, axis=0)
                    else:
                        new_dataset[dataset][
                            EMBEDDING_MODEL_NAME + "_mean_embeddings"
                        ][this_idx] = loaded_embeddings[this_idx]

                    if this_idx % 5000 == 0:
                        print(
                            f"\rCreating {dataset}: {this_idx/n_samples*100}%",
                            end="",
                        )
                        these_categories = [
                            coco_categories_91[lin - 1]
                            for lin in np.where(
                                new_dataset[dataset]["img_multi_hot"][this_idx]
                                == 1
                            )[0]
                        ]
                        plt.figure()
                        plt.imshow(new_dataset[dataset]["data"][this_idx])
                        this_mean_emb = new_dataset[dataset][
                            EMBEDDING_MODEL_NAME + "_mean_embeddings"
                        ][this_idx]
                        plt.title(
                            f"{loaded_captions[this_idx][0]}\n"
                            f"categs: {these_categories}\n"
                            f"emb shape, min, max, mean:\n"
                            f"{this_mean_emb.shape}, {np.min(this_mean_emb)}, {np.max(this_mean_emb)}, {np.mean(this_mean_emb)}"
                        )
                        plt.tight_layout()
                        plt.savefig(
                            f"{save_test_imgs_to}/cocoAdd{EMBEDDING_MODEL_NAME}__dataset_{dataset}_{this_idx}"
                        )
                        plt.close()


if CHECK_DATASET:
    with h5py.File(new_dataset_path, "r") as new_dataset:
        plot_datasets = ["train", "val", "test"]
        plot_n_image_pairs = 33
        scatterplot_n_image_pairs = 10000
        corr_metric = "pearson"  # corr metric to correlate distances between multihot/fasttext/guse
        if corr_metric.lower() == "pearson":
            corr_func = stats.pearsonr
        elif corr_metric.lower() == "spearman":
            corr_func = stats.spearmanr

        for dataset in plot_datasets:
            with open(f"ms_coco_nsd_captions_{dataset}.pkl", "rb") as fp:
                loaded_captions = pickle.load(fp)

            n_dataset_idx = new_dataset[dataset]["data"].shape[0]

            print(f"Plotting pairs of images for {dataset} set...")
            for i in range(plot_n_image_pairs):
                idx1, idx2 = np.random.randint(n_dataset_idx), np.random.randint(n_dataset_idx)
                im1, im2 = new_dataset[dataset]["data"][idx1], \
                    new_dataset[dataset]["data"][idx2],
                hot1, hot2 = new_dataset[dataset]["img_multi_hot"][idx1], \
                    new_dataset[dataset]["img_multi_hot"][idx2]
                cat1, cat2 = [coco_categories_91[lin - 1] for lin in np.where(hot1 == 1)[0]], \
                    [coco_categories_91[lin - 1] for lin in np.where(hot2 == 1)[0]]
                em1, em2 = new_dataset[dataset][EMBEDDING_MODEL_NAME + "_mean_embeddings"][idx1], \
                    new_dataset[dataset][EMBEDDING_MODEL_NAME + "_mean_embeddings"][idx2],
                cap1, cap2 = loaded_captions[idx1], loaded_captions[idx2]
                print(f"\ncategories 1: {cat1}")
                print(f"\ncategories 2: {cat2}")
                print(f"\ncaption 1: {cap1}")
                print(f"\ncaption 2: {cap2}")
                print(
                    f"embedding 1 - shape, min, max, mean: {em1.shape, np.min(em1), np.max(em1), np.mean(em1)}"
                )
                print(
                    f"embedding 2 - shape, min, max, mean: {em2.shape, np.min(em2), np.max(em2), np.mean(em2)}"
                )
                cos_dist = scipy.spatial.distance.cosine(em1, em2)
                print(f"cosine_dist(em1, em2): {cos_dist}")
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(im1)
                ax[0].set_title(cap1[0])
                ax[1].imshow(im2)
                ax[1].set_title(cap2[0])
                fig.suptitle(f"cos_dist = {cos_dist}")
                plt.tight_layout()
                plt.savefig(
                    f"{save_test_imgs_to}/cocoAdd{EMBEDDING_MODEL_NAME}__check_cosdist_matches_semSimilarity_{dataset}_{i}"
                )
                plt.close()


if MAKE_EMBEDDING_RDM_FOR_NSD:
    with h5py.File(new_dataset_path, "r") as new_dataset:
        dataset = "test"  # we only use test, because that contains all the NSD images

        embeddings = new_dataset[dataset][
            EMBEDDING_MODEL_NAME + "_mean_embeddings"
        ][:]
        rdm = scipy.spatial.distance.pdist(embeddings, "cosine")

        plt.figure()
        plt.matshow(scipy.spatial.distance.squareform(rdm))
        plt.savefig(f"{save_test_imgs_to}/test__fasttext_rdm_nsd")
        plt.close()

        np.save("./NSD_fasttext_RDM.npy", rdm)


if LOAD_AND_CHECK_EMBEDDING_RDM_FOR_NSD:
    print("Loading RDM and plotting to check it looks alright")
    rdm = np.load("./NSD_fasttext_RDM.npy")

    plt.figure()
    plt.matshow(scipy.spatial.distance.squareform(rdm))
    plt.savefig(
        f"{save_test_imgs_to}/test__fasttext_rdm_nsd_loaded_from_saved_npy"
    )
    plt.close()
