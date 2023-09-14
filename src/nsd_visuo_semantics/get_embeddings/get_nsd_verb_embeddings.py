import os
import pickle

import h5py
import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy.spatial

from nsd_visuo_semantics.get_embeddings.word_lists import verb_adjustments, load_fasttext_vectors

CHECK_FASTTEXT = 1
GET_VERB_EMBEDDINGS = 1
DO_SANITY_CHECK = 1

h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"
fasttext_embeddings_path = "./crawl-300d-2M.vec"
nsd_captions_path = "./ms_coco_nsd_captions_test.pkl"
save_test_imgs_to = "./_check_imgs"
os.makedirs(save_test_imgs_to, exist_ok=1)
save_embeddings_to = "../results_dir/saved_embeddings"
os.makedirs("../results_dir", exist_ok=1)
os.makedirs(save_embeddings_to, exist_ok=1)


if CHECK_FASTTEXT or GET_VERB_EMBEDDINGS:
    # get all word embeddings
    embeddings = load_fasttext_vectors(fasttext_embeddings_path)


if CHECK_FASTTEXT:
    # retrieve embeddings for coco categories
    # how to get one vector: v = np.array([i for i in embeddings['tree']])
    ms_coco_verb_embeddings = {}
    ms_coco_verb_embeddings["runs"] = np.array([i for i in embeddings["runs"]])
    ms_coco_verb_embeddings["run"] = np.array([i for i in embeddings["run"]])
    ms_coco_verb_embeddings["eats"] = np.array([i for i in embeddings["eats"]])
    ms_coco_verb_embeddings["eating"] = np.array(
        [i for i in embeddings["eating"]]
    )
    ms_coco_verb_embeddings["is"] = np.array([i for i in embeddings["is"]])

    # sanity checks
    print("Sanity check for embedding relationships")
    print(
        "cosine_dist(runs, run)",
        scipy.spatial.distance.cosine(
            ms_coco_verb_embeddings["runs"], ms_coco_verb_embeddings["run"]
        ),
    )
    print(
        "cosine_dist(runs, eats)",
        scipy.spatial.distance.cosine(
            ms_coco_verb_embeddings["runs"], ms_coco_verb_embeddings["eats"]
        ),
    )
    print(
        "cosine_dist(eats, eating)",
        scipy.spatial.distance.cosine(
            ms_coco_verb_embeddings["eats"], ms_coco_verb_embeddings["eating"]
        ),
    )
    print(
        "cosine_dist(eats, (is+eating)/2)",
        scipy.spatial.distance.cosine(
            ms_coco_verb_embeddings["eats"],
            (ms_coco_verb_embeddings["is"] + ms_coco_verb_embeddings["eating"])
            / 2,
        ),
    )
else:
    ms_coco_verb_embeddings = {}


if GET_VERB_EMBEDDINGS:

    def get_verbs_from_string(s):
        tokens = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        return [
            x[0] for x in tagged if "VB" in x[1]
        ]  # VB is the tag for verbs

    with open(nsd_captions_path, "rb") as fp:
        loaded_captions = pickle.load(fp)

    n_nsd_elements = len(loaded_captions)
    img_verbs = [
        [] for _ in range(n_nsd_elements)
    ]  # we will also save all verbs for each image
    mean_verb_embeddings = np.empty(
        (n_nsd_elements, 300)
    )  # fastext embeddings have 300 dimensions
    no_verbs_counter = 0  # we will count the images for which NO verbs were found in ANY of the captions
    skipped_candidates_not_verbs = 0  # we will count the number of candidates classified as verbs, but that are not verbs, or whose meaning is unknown
    skipped_candidates_no_embedding = (
        0  # we will count the number of verbs do not have a fasttext embedding
    )
    final_skipped_verbs = (
        []
    )  # finally, we will catch any left over "mistakes" after screening as explained above

    for i in range(n_nsd_elements):
        for j in range(len(loaded_captions[i])):
            this_sentence = loaded_captions[i][j]
            sentence_verbs = get_verbs_from_string(this_sentence)
            for n, s in enumerate(sentence_verbs):
                if s in verb_adjustments.keys():
                    # some spelling mistakes are made in the captions. Here, we fix them. In addition, some crap is
                    # miscalssified as verbs. We discard these. We also discard verbs that have no embedding (e.g. waterskiing).
                    # at the bottom of the script, we prinit out how many are rejected in this way, etc.
                    if verb_adjustments[s] == "_____not_verb_/_unknown_____":
                        skipped_candidates_not_verbs += 1
                    elif verb_adjustments[s] == "_____no_embedding_____":
                        skipped_candidates_no_embedding += 1
                    else:
                        sentence_verbs[n] = verb_adjustments[s]
            [img_verbs[i].append(v) for v in sentence_verbs]

        img_verb_embeddings = []
        for v in img_verbs[i]:
            if v in ms_coco_verb_embeddings.keys():
                img_verb_embeddings.append(
                    ms_coco_verb_embeddings[v]
                )  # we can't read the same embedding twice from the main embeddings dict, so we need to reload pre-existing ones
            else:
                try:
                    # if the verb exists in fasttext, use the embedding
                    new_embedding = np.array([i for i in embeddings[v]])
                    img_verb_embeddings.append(new_embedding)
                    ms_coco_verb_embeddings[v] = new_embedding
                except ValueError:
                    # if the verb does not exist in fasttext (e.g. "unpealed"), skip.
                    final_skipped_verbs.append(v)

        if not img_verb_embeddings:
            # usually, sentences without verbs are like "a pot on a table". So, for images with NO verbs in ANY of the
            # captions, we use "is" as the mean embedding.
            mean_verb_embeddings[i] = ms_coco_verb_embeddings["is"]
            no_verbs_counter += 1
        else:
            mean_verb_embeddings[i] = np.mean(
                np.asarray(img_verb_embeddings), axis=0
            )

    with open(
        f"{save_embeddings_to}/nsd_fasttext_VERB_mean_embeddings.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(mean_verb_embeddings, fp)
    with open(
        f"{save_embeddings_to}/nsd_verbs_per_image.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(img_verbs, fp)

    print(
        f"skipped verbs after screening for spelling mistakes and removing unknown words as described on line 329: {final_skipped_verbs}"
    )
    print(
        f"words classified as verbs but that are not in fact verbs or are unknown words: {skipped_candidates_not_verbs}"
    )
    print(
        f"verbs that do not have an embedding in fasttext: {skipped_candidates_no_embedding}"
    )
    print(f"n_imgs with NO verb for ANY caption: {no_verbs_counter}")


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

        with open("./ms_coco_nsd_captions_test.pkl", "rb") as fp:
            loaded_captions = pickle.load(fp)
        with open("./nsd_verbs_per_image.pkl", "rb") as fp:  # Pickling
            loaded_verbs = pickle.load(fp)
        with open(
            "./nsd_fasttext_VERB_mean_embeddings.pkl", "rb"
        ) as fp:  # Pickling
            loaded_verb_mean_embeddings = pickle.load(fp)

        for i in range(0, total_n_stims, step_size):
            plt.imshow(h5_dataset["test"]["data"][i])
            plt.title(
                f"{loaded_captions[i][0]}\n"
                f"{loaded_verbs[i]}\n"
                f"Emb shape, min, max, mean: {loaded_verb_mean_embeddings[i].shape, loaded_verb_mean_embeddings[i].min(), loaded_verb_mean_embeddings[i].max(), loaded_verb_mean_embeddings[i].mean()}"
            )
            plt.savefig(
                f"{save_test_imgs_to}/NSD_verb_embeddings_check_{i}.png"
            )
            plt.close()
