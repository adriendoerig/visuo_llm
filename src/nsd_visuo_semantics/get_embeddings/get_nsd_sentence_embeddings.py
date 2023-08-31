import os
import pickle
from random import shuffle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

from .embedding_models_zoo import get_embedding_model, get_embeddings

SANITY_CHECK = 1
GET_EMBEDDINGS = 1
FINAL_CHECK = 1

RANDOMIZE_WORD_ORDER = (
    False  # If True, word order will be randomized in each sentence.
)

save_every_n = 0  # if >0, save a checkpoint after every 10000 embeddings
load_intermediate_result = 0  # if >0, load checkpoitn from f"./nsd_{embedding_model_type}_mean_embeddings_intermediate_{i}.pkl"

ms_coco_GUSE_path = "../ms_coco_GUSE_square256.h5"
ms_coco_nsd_train_captions = "./ms_coco_nsd_captions_train.pkl"
ms_coco_nsd_val_captions = "./ms_coco_nsd_captions_val.pkl"
nsd_captions_path = "./ms_coco_nsd_captions_test.pkl"  # (nsd this is the ms_coco_nsd_test set))

save_test_imgs_to = "./_check_imgs"
os.makedirs(save_test_imgs_to, exist_ok=1)
save_embeddings_to = "../results_dir/saved_embeddings"
os.makedirs("../results_dir", exist_ok=1)
os.makedirs(save_embeddings_to, exist_ok=1)

captions_to_embed_path = nsd_captions_path

for embedding_model_type in [
    "GUSE_DAN",
    "all_mpnet_base_v2",
]:  # ['all_mpnet_base_v2', 'USE_CMLM_Base', 'openai_ada2', 'GUSE_transformer',  'GUSE_DAN', 'T5']:
    embedding_model = get_embedding_model(embedding_model_type)

    if SANITY_CHECK:
        print("running sanity check")
        # inspired from tutorial https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder?fbclid=IwAR1hlPezVtDLZCF4f4Nr2JxXZmF8WcQ5FA-PBtYnuBIXlpzWlISRCHse4WM

        def plt_rdm(sentences, embeddings):
            distances = scipy.spatial.distance.pdist(
                embeddings, metric="cosine"
            )
            plt.imshow(
                scipy.spatial.distance.squareform(distances), cmap="magma"
            )
            plt.colorbar()
            plt.savefig(
                f"{save_test_imgs_to}/semantic_similarity_check_{embedding_model_type}.png"
            )
            plt.close()

        def run_and_plot(sentences):
            sentence_embeddings = get_embeddings(
                sentences, embedding_model, embedding_model_type
            )
            plt_rdm(sentences, sentence_embeddings)

        sentences = [
            # Smartphones
            "I like my phone.",
            "My phone is not good.",
            "Your cellphone looks great.",
            # Weather
            "Will it snow tomorrow?",
            "Recently a lot of hurricanes have hit the US.",
            "Global warming is real,",
            # Food and health
            "An apple a day, keeps the doctors away,",
            "Eating strawberries is healthy.",
            "Is paleo better than keto?",
            # Asking about age
            "How old are you?",
            "what is your age?",
            # Word order
            "The child walks the dog.",
            "The dog walks the child.",
            "The dog devours the child.",
            "The the child devours the dog.",
        ]

        run_and_plot(sentences)

    if GET_EMBEDDINGS:

        def scramble(sentence):
            # helper function to randomize word order in sentences
            split = sentence.split()  # Split the string into a list of words
            shuffle(split)  # This shuffles the list in-place.
            return " ".join(split)  # Turn the list back into a string

        with open(captions_to_embed_path, "rb") as fp:
            loaded_captions = pickle.load(fp)

        n_nsd_elements = len(loaded_captions)
        dummy_embeddings = get_embeddings(
            loaded_captions[0], embedding_model, embedding_model_type
        )

        if load_intermediate_result:
            init_i = load_intermediate_result
            with open(
                "./nsd_{embedding_model_type}_mean_embeddings_intermediate_{i}.pkl",
                "rb",
            ) as fp:
                mean_embeddings = pickle.load(fp)
        else:
            init_i = 0
            mean_embeddings = np.empty(
                (n_nsd_elements, dummy_embeddings.shape[-1])
            )

        for i in range(init_i, n_nsd_elements):
            if i % 1000 == 0:
                print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

            if save_every_n and i % 10000 == 0 and i > 0:
                print(f"Saving intermediate result {i}")
                with open(
                    f"{save_embeddings_to}/nsd_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}_intermediate_{i}.pkl",
                    "wb",
                ) as fp:
                    pickle.dump(mean_embeddings, fp)

            if RANDOMIZE_WORD_ORDER:
                these_captions = [scramble(cap) for cap in loaded_captions[i]]
            else:
                these_captions = loaded_captions[i]

            img_embeddings = get_embeddings(
                these_captions, embedding_model, embedding_model_type
            )
            mean_embeddings[i] = np.mean(img_embeddings, axis=0)

        with open(
            f"{save_embeddings_to}/nsd_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}.pkl",
            "wb",
        ) as fp:
            pickle.dump(mean_embeddings, fp)

    if FINAL_CHECK:
        with h5py.File(ms_coco_GUSE_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open("./ms_coco_nsd_captions_test.pkl", "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(
                f"./nsd_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}.pkl",
                "rb",
            ) as fp:
                loaded_mean_embeddings = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(
                    f"{save_test_imgs_to}/NSD_{embedding_model_type}_mean_embeddings_check_{i}.png"
                )
                plt.close()
