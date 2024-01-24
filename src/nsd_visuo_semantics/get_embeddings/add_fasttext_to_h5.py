import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy import stats
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91, load_fasttext_vectors


CHECK_FASTTEXT = 0  # loads fasttext embeddings and runs a few checks
MAKE_DATASET = 0  # loads ms_coco_guse dataset, and copies it in a new dataset, adding fasttext embeddings
CHECK_DATASET = 1  # loads newly made dataset and runs a few sanity checks
MAKE_FASTTEXT_RDM_FOR_NSD = 0  # make RDM of cosine distances between fasttext embeddings for the whole NSD (here: test set of our version of ms-coco, which contains all nsd images)
LOAD_AND_CHECK_FASTTEXT_RDM_FOR_NSD = (
    0  # load the saved RDM to check it is alright
)

orig_dataset_path = "/home/student/a/adoerig/code/datasets/ms_coco_embeddings_nocategembeds_square256.h5"
new_dataset_path = (
    "/home/student/a/adoerig/code/datasets/ms_coco_embeddings_square256.h5"
)
fasttext_embeddings_path = "./crawl-300d-2M.vec"
ecoset_fasttext_path = "/home/student/a/adoerig/code/semantic_scene_descriptions_code/blt_vNet_ecoset_semanticloss/dataset_loader/fasttext_embeddings.h5"
save_test_imgs_to = "./_tmp_playground"
os.makedirs(save_test_imgs_to, exist_ok=1)

if CHECK_FASTTEXT:
    embeddings = load_fasttext_vectors(fasttext_embeddings_path)

    # retrieve embeddings for coco categories
    # how to get one vector: v = np.array([i for i in embeddings['tree']])
    coco_word_embeddings = {}
    baseball_vector = None  # each embedding can only be accessed once (due to the map() method). This trick is needed to access this vector twice, once for baseball-bat and once for baseball-glove
    for w in coco_categories_91:
        if w == "baseball-bat":
            baseball_vector = (
                np.array([i for i in embeddings["baseball"]])
                if baseball_vector is None
                else baseball_vector
            )
            coco_word_embeddings[w] = (
                baseball_vector + np.array([i for i in embeddings["bat"]])
            ) / 2
        elif w == "baseball-glove":
            baseball_vector = (
                np.array([i for i in embeddings["baseball"]])
                if baseball_vector is None
                else baseball_vector
            )
            coco_word_embeddings[w] = (
                baseball_vector + np.array([i for i in embeddings["glove"]])
            ) / 2
        elif w == "tennis-racket":
            coco_word_embeddings[w] = (
                np.array([i for i in embeddings["tennis"]])
                + np.array([i for i in embeddings["racket"]])
            ) / 2
        else:
            coco_word_embeddings[w] = np.array([i for i in embeddings[w]])

    # sanity checks
    print("Sanity check for embedding relationships")
    print(
        "cosine_dist(baseball-bat, baseball-glove)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["baseball-bat"],
            coco_word_embeddings["baseball-glove"],
        ),
    )
    print(
        "cosine_dist(baseball-bat, sink)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["baseball-bat"], coco_word_embeddings["sink"]
        ),
    )
    print(
        "cosine_dist(baseball-bat, tennis-racket)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["baseball-bat"],
            coco_word_embeddings["tennis-racket"],
        ),
    )
    print(
        "cosine_dist(hairdryer, giraffe)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["hairdryer"], coco_word_embeddings["giraffe"]
        ),
    )
    print(
        "cosine_dist(hairdryer, person)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["hairdryer"], coco_word_embeddings["person"]
        ),
    )
    print(
        "cosine_dist(hairdryer, donut)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["hairdryer"], coco_word_embeddings["donut"]
        ),
    )
    print(
        "cosine_dist(cake, donut)",
        scipy.spatial.distance.cosine(
            coco_word_embeddings["cake"], coco_word_embeddings["donut"]
        ),
    )

    print(
        "Sanity check that these embeddings are the same as used for ecoset."
    )
    with h5py.File(ecoset_fasttext_path, "r") as ecoset_fasttext:
        assert np.allclose(
            ecoset_fasttext["embeddings"][0],
            np.array([i for i in embeddings["man"]]),
        )
        assert np.allclose(
            ecoset_fasttext["embeddings"][1],
            np.array([i for i in embeddings["house"]]),
        )
        assert np.allclose(
            ecoset_fasttext["embeddings"][2], coco_word_embeddings["car"]
        )


if MAKE_DATASET:
    assert CHECK_FASTTEXT, "CHECK_FASTTEXT must be True to make the dataset"

    with h5py.File(orig_dataset_path, "r") as orig_dataset:
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
            f"{save_test_imgs_to}/test__check_that_categories_match_image.png"
        )

        with h5py.File(new_dataset_path, "w") as new_dataset:
            new_dataset.create_dataset("categories", data=coco_categories_91)

            for dataset in ["train", "val", "test"]:
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
                    )  # and here comes the new entry for fasttext embeddings!
                fasttext_category_mean_embeddings = group.create_dataset(
                    "fasttext_category_mean_embeddings",
                    shape=(n_samples, 300),
                    dtype="float32",
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

                    # get word embeddings for the current sample...
                    these_categories = [
                        coco_categories_91[lin - 1]
                        for lin in np.where(
                            orig_dataset[dataset]["img_multi_hot"][this_idx]
                            == 1
                        )[0]
                    ]
                    these_embeddings = [
                        coco_word_embeddings[c] for c in these_categories
                    ]
                    this_mean_embedding = np.mean(these_embeddings, axis=0)
                    if np.any(np.isnan(this_mean_embedding)):
                        # this may happen when no categories are present
                        this_mean_embedding = (
                            1 / 300 * np.ones_like(this_mean_embedding)
                        )

                    # ...and insert in new dataset
                    fasttext_category_mean_embeddings[
                        this_idx
                    ] = this_mean_embedding

                    if this_idx % 5000 == 0:
                        print(
                            f"\rCreating {dataset}: {this_idx/n_samples*100}%",
                            end="",
                        )
                        plt.figure()
                        plt.imshow(new_dataset[dataset]["data"][this_idx])
                        plt.title(
                            f"{these_categories}\n"
                            f"emb shape, min, max, mean: "
                            f"{fasttext_category_mean_embeddings[this_idx].shape}, "
                            f"{np.min(fasttext_category_mean_embeddings[this_idx])}, "
                            f"{np.max(fasttext_category_mean_embeddings[this_idx])}, "
                            f"{np.mean(fasttext_category_mean_embeddings[this_idx])}"
                        )
                        plt.savefig(
                            f"{save_test_imgs_to}/test__for_dataset_{dataset}_{this_idx}"
                        )


if CHECK_DATASET:
    with h5py.File(new_dataset_path, "r") as fasttext_coco_guse:
        plot_datasets = ["train", "val", "test"]
        plot_n_image_pairs = 33
        scatterplot_n_image_pairs = 10000
        corr_metric = "pearson"  # corr metric to correlate distances between multihot/fasttext/guse
        if corr_metric.lower() == "pearson":
            corr_func = stats.pearsonr
        elif corr_metric.lower() == "spearman":
            corr_func = stats.spearmanr

        for dataset in plot_datasets:
            n_dataset_idx = fasttext_coco_guse[dataset]["data"].shape[0]
            print(f"Plotting pairs of images for {dataset} set...")
            for i in range(plot_n_image_pairs):
                idx1, idx2 = np.random.randint(
                    n_dataset_idx
                ), np.random.randint(n_dataset_idx)
                im1, im2 = (
                    fasttext_coco_guse[dataset]["data"][idx1],
                    fasttext_coco_guse[dataset]["data"][idx2],
                )
                hot1, hot2 = (
                    fasttext_coco_guse[dataset]["img_multi_hot"][idx1],
                    fasttext_coco_guse[dataset]["img_multi_hot"][idx2],
                )
                cat1, cat2 = [
                    coco_categories_91[lin - 1]
                    for lin in np.where(hot1 == 1)[0]
                ], [
                    coco_categories_91[lin - 1]
                    for lin in np.where(hot2 == 1)[0]
                ]
                em1, em2 = (
                    fasttext_coco_guse[dataset][
                        "fasttext_category_mean_embeddings"
                    ][idx1],
                    fasttext_coco_guse[dataset][
                        "fasttext_category_mean_embeddings"
                    ][idx2],
                )
                print(f"\ncategories 1: {cat1}")
                print(f"\ncategories 2: {cat2}")
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
                ax[0].set_title(cat1)
                ax[1].imshow(im2)
                ax[1].set_title(cat2)
                fig.suptitle(f"cos_dist = {cos_dist}")
                plt.savefig(
                    f"{save_test_imgs_to}/test__dataset_check_cosdist_matches_categSimilarity_{dataset}_{i}"
                )
                plt.close()

                if cos_dist == 0:
                    import pdb

                    pdb.set_trace()

            print(
                f"Plotting scatterplots + correlations between [multihot, fasttext, guse] distances for {scatterplot_n_image_pairs} {dataset} set..."
            )
            multihot_dists = np.zeros(scatterplot_n_image_pairs)
            fasttext_dists = np.zeros(scatterplot_n_image_pairs)
            guse_dists = np.zeros(scatterplot_n_image_pairs)
            for i in range(scatterplot_n_image_pairs):
                idx1, idx2 = np.random.randint(
                    n_dataset_idx
                ), np.random.randint(n_dataset_idx)
                multihot_dists[i] = scipy.spatial.distance.cosine(
                    fasttext_coco_guse[dataset]["img_multi_hot"][idx1],
                    fasttext_coco_guse[dataset]["img_multi_hot"][idx2],
                )
                fasttext_dists[i] = scipy.spatial.distance.cosine(
                    fasttext_coco_guse[dataset][
                        "fasttext_category_mean_embeddings"
                    ][idx1],
                    fasttext_coco_guse[dataset][
                        "fasttext_category_mean_embeddings"
                    ][idx2],
                )
                guse_dists[i] = scipy.spatial.distance.cosine(
                    fasttext_coco_guse[dataset]["labels"][idx1],
                    fasttext_coco_guse[dataset]["labels"][idx2],
                )

            multihot_fasttext_corr = corr_func(
                multihot_dists, fasttext_dists
            )  # [0] is r, [1] is p-val
            multihot_guse_corr = corr_func(multihot_dists, guse_dists)
            fasttext_guse_corr = corr_func(fasttext_dists, guse_dists)

            fig, ax = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5))
            ax[0].scatter(multihot_dists, fasttext_dists)
            ax[0].set_xlabel("multihot_dists"), ax[0].set_ylabel(
                "fasttext_dists"
            )
            ax[0].set_title(
                f"{corr_metric}={multihot_fasttext_corr[0]:.2f}, p={multihot_fasttext_corr[1]:.4f}"
            )
            ax[1].scatter(multihot_dists, guse_dists)
            ax[1].set_xlabel("multihot_dists"), ax[1].set_ylabel("guse_dists")
            ax[1].set_title(
                f"{corr_metric}={multihot_guse_corr[0]:.2f}, p={multihot_guse_corr[1]:.4f}"
            )
            ax[2].scatter(fasttext_dists, guse_dists)
            ax[2].set_xlabel("fasttext_dists"), ax[2].set_ylabel("guse_dists")
            ax[2].set_title(
                f"{corr_metric}={fasttext_guse_corr[0]:.2f}, p={fasttext_guse_corr[1]:.4f}"
            )
            # [ax[i].axis('off') for i in range(3)]
            [ax[i].set_aspect("equal") for i in range(3)]
            plt.savefig(
                f"{save_test_imgs_to}/test__dataset_multihotVsFasttextVsGuse_{dataset}Set"
            )
            plt.close()


if MAKE_FASTTEXT_RDM_FOR_NSD:
    with h5py.File(new_dataset_path, "r") as fasttext_coco_guse:
        dataset = "test"  # we only use test, because that contains all the NSD images

        embeddings = fasttext_coco_guse[dataset][
            "fasttext_category_mean_embeddings"
        ][:]
        rdm = scipy.spatial.distance.pdist(embeddings, "cosine")

        plt.figure()
        plt.matshow(scipy.spatial.distance.squareform(rdm))
        plt.savefig(f"{save_test_imgs_to}/test__fasttext_rdm_nsd")
        plt.close()

        np.save("./NSD_fasttext_RDM.npy", rdm)


if LOAD_AND_CHECK_FASTTEXT_RDM_FOR_NSD:
    print("Loading RDM and plotting to check it looks alright")
    rdm = np.load("./NSD_fasttext_RDM.npy")

    import pdb

    pdb.set_trace()
    plt.figure()
    plt.matshow(scipy.spatial.distance.squareform(rdm))
    plt.savefig(
        f"{save_test_imgs_to}/test__fasttext_rdm_nsd_loaded_from_saved_npy"
    )
    plt.close()
