import os
import pickle

import h5py
import matplotlib.pyplot as plt
import nltk
import numpy as np

from .word_lists import verb_adjustments

GET_WORD_EMBEDDINGS = 1
DO_SANITY_CHECK = 1

h5_dataset_path = "../ms_coco_GUSE_square256.h5"
fasttext_embeddings_path = "./crawl-300d-2M.vec"
nsd_captions_path = "./ms_coco_nsd_captions_test.pkl"
save_test_imgs_to = "./_check_imgs"
os.makedirs(save_test_imgs_to, exist_ok=1)
save_embeddings_to = "../results_dir/saved_embeddings"
os.makedirs("../results_dir", exist_ok=1)
os.makedirs(save_embeddings_to, exist_ok=1)


if GET_WORD_EMBEDDINGS:
    # get all word embeddings
    def load_vectors(fname):
        try:
            fin = open(fname, encoding="utf-8", newline="\n", errors="ignore")
        except ValueError:
            raise Exception(
                f"{fname} not found. Localize the .vec containing the embeddings, or download "
                f'"wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"'
            )
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = map(float, tokens[1:])
        return data

    embeddings = load_vectors(fasttext_embeddings_path)

    ms_coco_mean_allWord_embeddings = {}

    with open(nsd_captions_path, "rb") as fp:
        loaded_captions = pickle.load(fp)

    n_nsd_elements = len(loaded_captions)
    img_words = [
        [] for _ in range(n_nsd_elements)
    ]  # we will also save all verbs for each image
    mean_allWord_embeddings = np.empty(
        (n_nsd_elements, 300)
    )  # fastext embeddings have 300 dimensions

    for i in range(n_nsd_elements):
        for j in range(len(loaded_captions[i])):
            this_sentence = loaded_captions[i][j]
            sentence_words = nltk.word_tokenize(this_sentence)
            for n, s in enumerate(sentence_words):
                if s in verb_adjustments.keys():
                    # some spelling mistakes are made in the captions. Here, we fix them. In addition, some crap is
                    # miscalssified as verbs. We discard these. We also discard verbs that have no embedding (e.g. waterskiing).
                    # at the bottom of the script, we prinit out how many are rejected in this way, etc.
                    if verb_adjustments[s] == "_____not_verb_/_unknown_____":
                        pass
                    elif verb_adjustments[s] == "_____no_embedding_____":
                        pass
                    else:
                        sentence_words[n] = verb_adjustments[s]
            [img_words[i].append(w) for w in sentence_words]

        img_allWord_embeddings = []
        for w in img_words[i]:
            if w in ms_coco_mean_allWord_embeddings.keys():
                img_allWord_embeddings.append(
                    ms_coco_mean_allWord_embeddings[w]
                )  # we can't read the same embedding twice from the main embeddings dict, so we need to reload pre-existing ones
            else:
                try:
                    # if the word exists in fasttext, use the embedding
                    new_embedding = np.array([i for i in embeddings[w]])
                    img_allWord_embeddings.append(new_embedding)
                    ms_coco_mean_allWord_embeddings[w] = new_embedding
                except ValueError:
                    # if the word does not exist in fasttext (e.g. "unpealed"), skip.
                    pass

        mean_allWord_embeddings[i] = np.mean(
            np.asarray(img_allWord_embeddings), axis=0
        )

    with open(
        f"{save_embeddings_to}/nsd_fasttext_allWord_mean_embeddings.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(mean_allWord_embeddings, fp)
    with open(
        f"{save_embeddings_to}/nsd_allWords_per_image.pkl", "wb"
    ) as fp:  # Pickling
        pickle.dump(img_words, fp)


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
        with open("./nsd_allWords_per_image.pkl", "rb") as fp:  # Pickling
            loaded_allWords = pickle.load(fp)
        with open(
            "./nsd_fasttext_allWord_mean_embeddings.pkl", "rb"
        ) as fp:  # Pickling
            loaded_allWord_mean_embeddings = pickle.load(fp)

        for i in range(0, total_n_stims, step_size):
            plt.imshow(h5_dataset["test"]["data"][i])
            plt.title(
                f"{loaded_captions[i][0]}\n"
                f"{loaded_allWords[i]}\n"
                f"Emb shape, min, max, mean: {loaded_allWord_mean_embeddings[i].shape, loaded_allWord_mean_embeddings[i].min(), loaded_allWord_mean_embeddings[i].max(), loaded_allWord_mean_embeddings[i].mean()}"
            )
            plt.savefig(
                f"{save_test_imgs_to}/NSD_allWord_embeddings_check_{i}.png"
            )
