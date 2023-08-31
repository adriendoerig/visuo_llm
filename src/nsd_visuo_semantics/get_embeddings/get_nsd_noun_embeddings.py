import os
import pdb
import pickle

import h5py
import matplotlib.pyplot as plt
import nltk
import numpy as np
from scipy.spatial.distance import cdist, correlation

from nsd_visuo_semantics.get_embeddings.word_lists import (
    coco_categories_91,
    noun_adjustments,
)

CHECK_FASTTEXT = 1
GET_NOUN_EMBEDDINGS = 1
DO_SANITY_CHECK = 1

# MATCH_TO_COCO_CATEGORY_NOUNS = True  # if True, use the nearest coco object category embedding for each noun

for MATCH_TO_COCO_CATEGORY_NOUNS in [True, False]:
    if MATCH_TO_COCO_CATEGORY_NOUNS:
        METRIC = "correlation"  # use this distance to get nearest coco cat neighbour
        CUTOFF = 0.33  # if the closest coco categ is at a larger distanc than this, it is ignored (1->no cutoff)
        SAVE_SUFFIX = f"_closest_cocoCats_cut{CUTOFF}"
    else:
        SAVE_SUFFIX = ""
    h5_dataset_path = "/rds/projects/c/charesti-start/projects/NSD/ms_coco_GUSE_square256.h5"  # '../ms_coco_GUSE_square256.h5'
    fasttext_embeddings_path = "./crawl-300d-2M.vec"
    nsd_captions_path = "./ms_coco_nsd_captions_test.pkl"
    save_test_imgs_to = "./_check_imgs"
    os.makedirs(save_test_imgs_to, exist_ok=1)
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    if CHECK_FASTTEXT or GET_NOUN_EMBEDDINGS:
        # get all word embeddings
        def load_vectors(fname):
            try:
                fin = open(
                    fname, encoding="utf-8", newline="\n", errors="ignore"
                )
            except ValueError:
                raise Exception(
                    f"{fname} not found. Localize the .vec containing the embeddings, or download "
                    '"wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"'
                )
            n, d = map(int, fin.readline().split())
            data = {}
            for line in fin:
                tokens = line.rstrip().split(" ")
                data[tokens[0]] = map(float, tokens[1:])
            return data

        embeddings = load_vectors(fasttext_embeddings_path)

    if CHECK_FASTTEXT:
        # retrieve embeddings for coco categories
        # how to get one vector: v = np.array([i for i in embeddings['tree']])
        ms_coco_noun_embeddings = {}
        ms_coco_noun_embeddings["cat"] = np.array(
            [i for i in embeddings["cat"]]
        )
        ms_coco_noun_embeddings["dog"] = np.array(
            [i for i in embeddings["dog"]]
        )
        ms_coco_noun_embeddings["table"] = np.array(
            [i for i in embeddings["table"]]
        )
        ms_coco_noun_embeddings["chair"] = np.array(
            [i for i in embeddings["chair"]]
        )
        ms_coco_noun_embeddings["sky"] = np.array(
            [i for i in embeddings["sky"]]
        )

        # sanity checks
        print(
            "Sanity check for embedding relationships (correlation distance)"
        )
        print(
            "cosine_dist(cat, dog)",
            correlation(
                ms_coco_noun_embeddings["cat"], ms_coco_noun_embeddings["dog"]
            ),
        )
        print(
            "cosine_dist(cat, table)",
            correlation(
                ms_coco_noun_embeddings["cat"],
                ms_coco_noun_embeddings["table"],
            ),
        )
        print(
            "cosine_dist(table, chair)",
            correlation(
                ms_coco_noun_embeddings["table"],
                ms_coco_noun_embeddings["chair"],
            ),
        )
        print(
            "cosine_dist(table, sky)",
            correlation(
                ms_coco_noun_embeddings["table"],
                ms_coco_noun_embeddings["sky"],
            ),
        )
    else:
        ms_coco_noun_embeddings = {}

    if GET_NOUN_EMBEDDINGS:

        def get_nouns_from_string(s):
            tokens = nltk.word_tokenize(s)
            tagged = nltk.pos_tag(tokens)
            return [
                x[0] for x in tagged if "NN" in x[1]
            ]  # NN is the tag for nouns

        with open(nsd_captions_path, "rb") as fp:
            loaded_captions = pickle.load(fp)
        n_nsd_elements = len(loaded_captions)

        if MATCH_TO_COCO_CATEGORY_NOUNS:
            # get the embeddings for coco object categories
            img_nouns_coco_cats = [
                [] for _ in range(n_nsd_elements)
            ]  # we will also save cat names corresponding to all nouns for each image
            coco_cat_embeds = np.empty((len(coco_categories_91), 300))
            (
                baseball_vector,
                bat_vector,
                glove_vector,
                tennis_vector,
                racket_vector,
            ) = (
                None,
                None,
                None,
                None,
                None,
            )  # needed to avoid loading it twice (you can't lead the fasttext embs twice)
            for c, cat in enumerate(coco_categories_91):
                if cat in ms_coco_noun_embeddings.keys():
                    new_embedding = ms_coco_noun_embeddings[cat]
                else:
                    if cat == "baseball-bat":
                        baseball_vector = (
                            np.array([i for i in embeddings["baseball"]])
                            if baseball_vector is None
                            else baseball_vector
                        )
                        bat_vector = (
                            np.array([i for i in embeddings["bat"]])
                            if bat_vector is None
                            else bat_vector
                        )
                        new_embedding = (baseball_vector + bat_vector) / 2
                    elif cat == "baseball-glove":
                        baseball_vector = (
                            np.array([i for i in embeddings["baseball"]])
                            if baseball_vector is None
                            else baseball_vector
                        )
                        glove_vector = (
                            np.array([i for i in embeddings["glove"]])
                            if glove_vector is None
                            else glove_vector
                        )
                        new_embedding = (baseball_vector + glove_vector) / 2
                    elif cat == "tennis-racket":
                        tennis_vector = (
                            np.array([i for i in embeddings["tennis"]])
                            if tennis_vector is None
                            else tennis_vector
                        )
                        racket_vector = (
                            np.array([i for i in embeddings["racket"]])
                            if racket_vector is None
                            else racket_vector
                        )
                        new_embedding = (tennis_vector + racket_vector) / 2
                    else:
                        new_embedding = np.array([i for i in embeddings[cat]])
                    ms_coco_noun_embeddings["baseball"] = baseball_vector
                    ms_coco_noun_embeddings["bat"] = bat_vector
                    ms_coco_noun_embeddings["glove"] = glove_vector
                    ms_coco_noun_embeddings["tennis"] = tennis_vector
                    ms_coco_noun_embeddings["racket"] = racket_vector

                ms_coco_noun_embeddings[cat] = new_embedding
                coco_cat_embeds[c, :] = new_embedding

        img_nouns = [
            [] for _ in range(n_nsd_elements)
        ]  # we will also save all nouns for each image
        mean_noun_embeddings = np.empty(
            (n_nsd_elements, 300)
        )  # fastext embeddings have 300 dimensions
        no_nouns_counter = 0  # we will count the images for which NO nouns were found in ANY of the captions
        skipped_candidates_not_nouns = 0  # we will count the number of candidates classified as nouns, but that are not nouns, or whose meaning is unknown
        skipped_candidates_no_embedding = 0  # we will count the number of nouns do not have a fasttext embedding
        distances_to_closest_cat = [
            [] for _ in range(n_nsd_elements)
        ]  # we will also save all nouns for each image
        final_skipped_nouns = (
            []
        )  # finally, we will catch any left over "mistakes" after screening as explained above
        something_vector = np.array(
            [i for i in embeddings["something"]]
        )  # what we use in case an image has no nouns
        ms_coco_noun_embeddings["something"] = something_vector

        for i in range(n_nsd_elements):
            # get all caption nouns
            for j in range(len(loaded_captions[i])):
                this_sentence = loaded_captions[i][j]
                sentence_nouns = get_nouns_from_string(this_sentence)
                for n, s in enumerate(sentence_nouns):
                    if s in noun_adjustments.keys():
                        # some spelling mistakes are made in the captions. Here, we fix them. In addition, some crap is
                        # miscalssified as nouns. We discard these. We also discard nouns that have no embedding (e.g. waterskiing).
                        # at the bottom of the script, we prinit out how many are rejected in this way, etc.
                        # THERE ARE OVER 2000 OF THESE, SO I DID NOT DO IT, AND RELY ON THE 5 COCO CAPTIONS TO GET IT RIGHT
                        # ON AVERAGE. THERE ARE NO IMAGES WITHOUT ANY NOUNS, SO WE SHOULD BE SAFE. STILL, SHOULD YOU WANT TO
                        # DO THIS, YOU NEED TO UPDATE word_lists.py TO ACCOUNT FOR THE BAD WORDS
                        # COLLECTED IN final_skipped_nouns
                        if (
                            noun_adjustments[s]
                            == "_____not_noun_/_unknown_____"
                        ):
                            skipped_candidates_not_nouns += 1
                        elif noun_adjustments[s] == "_____no_embedding_____":
                            skipped_candidates_no_embedding += 1
                        else:
                            sentence_nouns[n] = noun_adjustments[s]
                [img_nouns[i].append(v) for v in sentence_nouns]

            # for each caption noun, get the embedding
            img_noun_embeddings = []
            for v in img_nouns[i]:
                if MATCH_TO_COCO_CATEGORY_NOUNS:
                    # woman, etc, have a quite big distance to "person", which is the closest coco cat.
                    # but this is important, so we hard code it
                    if (
                        v == "woman"
                        or v == "man"
                        or v == "kid"
                        or v == "child"
                        or v == "women"
                        or v == "men"
                        or v == "kids"
                        or v == "children"
                    ):
                        v = "person"
                if v in ms_coco_noun_embeddings.keys():
                    img_noun_embeddings.append(
                        ms_coco_noun_embeddings[v]
                    )  # we can't read the same embedding twice from the main embeddings dict, so we need to reload pre-existing ones
                else:
                    try:
                        # if the noun exists in fasttext, use the embedding
                        new_embedding = np.array([i for i in embeddings[v]])
                        img_noun_embeddings.append(new_embedding)
                        ms_coco_noun_embeddings[v] = new_embedding
                    except ValueError:
                        # if the noun does not exist in fasttext (e.g. "unpealed"), skip.
                        final_skipped_nouns.append(v)

            if not img_noun_embeddings:
                # deal images without a noun. We use the embedding for "something"
                mean_noun_embeddings[i] = something_vector
                no_nouns_counter += 1
            else:
                if MATCH_TO_COCO_CATEGORY_NOUNS:
                    cococat_matched_img_noun_embeddings = []
                    for e, this_e in enumerate(img_noun_embeddings):
                        try:
                            lookup_distances = cdist(
                                this_e[None, :], coco_cat_embeds, metric=METRIC
                            )
                        except ValueError:
                            pdb.set_trace()
                        winner = np.argmin(lookup_distances)
                        this_dist = lookup_distances[0, winner]
                        closest_categ = coco_categories_91[winner]
                        distances_to_closest_cat[i].append(this_dist)
                        if this_dist > CUTOFF:
                            img_nouns_coco_cats[i].append(
                                f"{closest_categ}: skipped, dist={this_dist:.2f}"
                            )
                        else:
                            img_nouns_coco_cats[i].append(closest_categ)
                            cococat_matched_img_noun_embeddings.append(
                                coco_cat_embeds[winner]
                            )
                    if cococat_matched_img_noun_embeddings == []:
                        cococat_matched_img_noun_embeddings = something_vector
                        no_nouns_counter += 1
                    mean_noun_embeddings[i] = np.mean(
                        np.asarray(cococat_matched_img_noun_embeddings), axis=0
                    )
                else:
                    mean_noun_embeddings[i] = np.mean(
                        np.asarray(img_noun_embeddings), axis=0
                    )

        with open(
            f"{save_embeddings_to}/nsd_fasttext_NOUNS_mean_embeddings{SAVE_SUFFIX}.pkl",
            "wb",
        ) as fp:  # Pickling
            pickle.dump(mean_noun_embeddings, fp)
        with open(
            f"{save_embeddings_to}/nsd_nouns_per_image.pkl", "wb"
        ) as fp:  # Pickling
            pickle.dump(img_nouns, fp)
        if MATCH_TO_COCO_CATEGORY_NOUNS:
            with open(
                f"{save_embeddings_to}/nsd_nouns_per_image{SAVE_SUFFIX}.pkl",
                "wb",
            ) as fp:  # Pickling
                pickle.dump(img_nouns_coco_cats, fp)

        print(
            f"skipped nouns after screening for spelling mistakes and removing unknown words as described on line 329: {final_skipped_nouns}"
        )
        print(
            f"words classified as nouns but that are not in fact nouns or are unknown words: {skipped_candidates_not_nouns}"
        )
        print(
            f"nouns that do not have an embedding in fasttext: {skipped_candidates_no_embedding}"
        )
        print(f"n_imgs with NO noun for ANY caption: {no_nouns_counter}")

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
            with open(
                f"{save_embeddings_to}/nsd_nouns_per_image{SAVE_SUFFIX}.pkl",
                "rb",
            ) as fp:  # Pickling
                loaded_nouns = pickle.load(fp)
            with open(
                f"{save_embeddings_to}/nsd_fasttext_NOUNS_mean_embeddings{SAVE_SUFFIX}.pkl",
                "rb",
            ) as fp:  # Pickling
                loaded_noun_mean_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_nouns[i]}\n"
                    f"Emb shape, min, max, mean: {loaded_noun_mean_embeddings[i].shape, loaded_noun_mean_embeddings[i].min(), loaded_noun_mean_embeddings[i].max(), loaded_noun_mean_embeddings[i].mean()}"
                )
                plt.savefig(
                    f"{save_test_imgs_to}/NSD_noun_embeddings_check_{i}{SAVE_SUFFIX}.png"
                )
                plt.close()


pdb.set_trace()
