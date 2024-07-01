import os, pickle, h5py
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import sentence_embeddings_sanity_check, scramble_word_order, randomize_by_word_type


def get_nsd_sentence_embeddings(embedding_model_type, captions_to_embed_path, 
                                RANDOMIZE_BY_WORD_TYPE, RANDOMIZE_WORD_ORDER, MIN_DIST_CUTOFF,
                                h5_dataset_path, use_saved_randomized_sentences_from_other_model, OVERWRITE):
    '''MIN_DIST_CUTOFF: if >0, when scrambling ensure that the word we are using for scrambling has an embedding at least MIN_DIST_CUTOFF away from the word to replace.'''


    print(f"GATHERING EMBEDDINGS FOR: {embedding_model_type}\n "
          f"ON: {captions_to_embed_path} \n "
          f"WITH RANDOMIZE_BY_WORD_TYPE: {RANDOMIZE_BY_WORD_TYPE}\n "
          f"AND RANDOMIZE_WORD_ORDER: {RANDOMIZE_WORD_ORDER}\n " 
          f"AND MIN_DIST_CUTOFF: {MIN_DIST_CUTOFF}") 

    METRIC = 'correlation'

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 1

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

    if RANDOMIZE_BY_WORD_TYPE is None:
        save_name = f"{prefix}_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}"
    else:
        if not type(RANDOMIZE_BY_WORD_TYPE) == list:
            RANDOMIZE_BY_WORD_TYPE = [RANDOMIZE_BY_WORD_TYPE]
        if MIN_DIST_CUTOFF == 0:
            save_name = f"{prefix}_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"
        else:
            save_name = f"{prefix}_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}_cutoffDist{MIN_DIST_CUTOFF}"

        if use_saved_randomized_sentences_from_other_model is not None:
            try:
                randomized_captions_path = save_name.replace(embedding_model_type, use_saved_randomized_sentences_from_other_model)
                randomized_captions_path += "_randomized_captions.pkl"
                print(f"Loading randomized captions from {randomized_captions_path}")
                with open(f'{save_embeddings_to}/{randomized_captions_path}', "rb") as fp:
                    loaded_randomized_captions = pickle.load(fp)
            except FileNotFoundError:
                print(f"Could not find {randomized_captions_path}. Will generate new randomized captions.")
                use_saved_randomized_sentences_from_other_model = None

    if os.path.exists(f"{save_embeddings_to}/{save_name}.pkl") and not OVERWRITE:
        print(f"Embeddings already exist at {save_embeddings_to}/{save_name}.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            sentence_embeddings_sanity_check(embedding_model_type, embedding_model, METRIC, save_test_imgs_to)

        if GET_EMBEDDINGS:
            if '.pkl' in captions_to_embed_path:
                with open(captions_to_embed_path, "rb") as fp:
                    loaded_captions = pickle.load(fp)
            elif '.npy' in captions_to_embed_path:
                loaded_captions = np.load(captions_to_embed_path, allow_pickle=True)
            else:
                raise ValueError("Captions file format not recognized.")

            randomized_captions = []

            n_nsd_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)

            if RANDOMIZE_BY_WORD_TYPE is not None:
                mean_n_changes = 0

            mean_embeddings = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))

            for i in range(n_nsd_elements):
                if i % 100 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                if RANDOMIZE_WORD_ORDER:
                    these_captions = [scramble_word_order(cap) for cap in loaded_captions[i]]
                else:
                    these_captions = loaded_captions[i]

                if not isinstance(these_captions, list):
                    # needed if we are using a single caption per image
                    # in that case, we have a string and convert it to a list
                    # with a single element
                    these_captions = [these_captions]

                if RANDOMIZE_BY_WORD_TYPE is not None:
                    if use_saved_randomized_sentences_from_other_model is not None:
                        # overwrites these_captions
                        these_captions = loaded_randomized_captions[i]
                    else:
                        # randomizes these_captions according to the requested word type(s)
                        these_n_changes = np.zeros(len(these_captions))
                        these_captions_rand = []
                        for c, cap in enumerate(these_captions):
                            rand_cap, these_n_changes[c] = randomize_by_word_type(cap, RANDOMIZE_BY_WORD_TYPE, loaded_captions, MIN_DIST_CUTOFF, 
                                                                                embedding_model, embedding_model_type, metric=METRIC)
                            these_captions_rand.append(rand_cap)
                        randomized_captions.append(these_captions_rand)
                        mean_n_changes += these_n_changes.mean()/n_nsd_elements
                        these_captions = these_captions_rand

                img_embeddings = get_embeddings(these_captions, embedding_model, embedding_model_type)
                mean_embeddings[i] = np.mean(img_embeddings, axis=0)

            with open(f"{save_embeddings_to}/{save_name}.pkl", "wb") as fp:
                pickle.dump(mean_embeddings, fp)
            if RANDOMIZE_BY_WORD_TYPE is not None:
                if use_saved_randomized_sentences_from_other_model is None:
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
            if RANDOMIZE_BY_WORD_TYPE is not None:
                if use_saved_randomized_sentences_from_other_model is None:
                    with open(f"{save_embeddings_to}/{save_name}_randomized_captions.pkl", "rb") as fp:
                        loaded_randomized_captions = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_randomized_captions[i][0]}\n" if RANDOMIZE_BY_WORD_TYPE is not None else ""
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_test_imgs_to}/{save_name}_check_{i}.png")
                plt.close()
