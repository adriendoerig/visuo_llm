import os, pickle, h5py
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import sentence_embeddings_sanity_check, scramble_word_order, randomize_by_word_type


def get_nsd_sentence_embeddings(embedding_model_type, captions_to_embed_path, RANDOMIZE_BY_WORD_TYPE, RANDOMIZE_WORD_ORDER,
                                h5_dataset_path, OVERWRITE):

    print(f"GATHERING EMBEDDINGS FOR: {embedding_model_type}\n "
          f"ON: {captions_to_embed_path} \n "
          f"WITH RANDOMIZE_BY_WORD_TYPE: {RANDOMIZE_BY_WORD_TYPE}\n "
          f"AND RANDOMIZE_WORD_ORDER: {RANDOMIZE_WORD_ORDER}") 
    
    if not type(RANDOMIZE_BY_WORD_TYPE) == list:
        RANDOMIZE_BY_WORD_TYPE = [RANDOMIZE_BY_WORD_TYPE]

    METRIC = 'correlation'

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 1

    save_every_n = 0  # if >0, save a checkpoint after every 10000 embeddings
    load_intermediate_result = 0  # if >0, load checkpoitn from f"./nsd_{embedding_model_type}_mean_embeddings_intermediate_{i}.pkl"

    save_test_imgs_to = "../results_dir/_check_imgs"
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_test_imgs_to, exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    save_name = f"nsd_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"

    if os.path.exists(f"{save_embeddings_to}/{save_name}.pkl") and OVERWRITE:
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            sentence_embeddings_sanity_check(embedding_model_type, embedding_model, METRIC, save_test_imgs_to)

        if GET_EMBEDDINGS:
            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            randomized_captions = []

            n_nsd_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)

            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                mean_n_changes = 0

            if load_intermediate_result:
                init_i = load_intermediate_result
                with open(f"{save_embeddings_to}/{save_name}_intermediate_{i}.pkl", "rb") as fp:
                    mean_embeddings = pickle.load(fp)
            else:
                init_i = 0
                mean_embeddings = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))

            for i in range(init_i, n_nsd_elements):
                if i % 1000 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                if save_every_n and i % 10000 == 0 and i > 0:
                    print(f"Saving intermediate result {i}")
                    with open(f"{save_embeddings_to}/{save_name}_intermediate_{i}.pkl", "wb",) as fp:
                        pickle.dump(mean_embeddings, fp)

                if RANDOMIZE_WORD_ORDER:
                    these_captions = [scramble_word_order(cap) for cap in loaded_captions[i]]
                else:
                    these_captions_not_rand = loaded_captions[i]

                if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                    these_n_changes = np.zeros(len(these_captions_not_rand))
                    these_captions = []
                    for c, cap in enumerate(these_captions_not_rand):
                        rand_cap, these_n_changes[c] = randomize_by_word_type(cap, RANDOMIZE_BY_WORD_TYPE, loaded_captions)
                        these_captions.append(rand_cap)
                    randomized_captions.append(these_captions)
                    mean_n_changes += these_n_changes.mean()/n_nsd_elements
                else:
                    these_captions = these_captions_not_rand

                img_embeddings = get_embeddings(these_captions, embedding_model, embedding_model_type)
                mean_embeddings[i] = np.mean(img_embeddings, axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb") as fp:
                pickle.dump(mean_embeddings, fp)
            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                with open(f"{save_embeddings_to}/{save_name}_randomized_captions.pkl", "wb") as fp:
                    pickle.dump(randomized_captions, fp)
                np.save(f"{save_embeddings_to}/{save_name}_mean_n_changes.npy", mean_n_changes)
                print(f"Done. Mean number of changesin word typerandomization: {mean_n_changes}")


    if FINAL_CHECK:
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_embeddings_to}/{save_name}.pkl", "rb") as fp:
                loaded_mean_embeddings = pickle.load(fp)
            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                with open(f"{save_embeddings_to}/{save_name}_randomized_captions.pkl", "rb") as fp:
                    loaded_randomized_captions = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_randomized_captions[i][0]}\n" if len(RANDOMIZE_BY_WORD_TYPE) > 0 else ""
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
