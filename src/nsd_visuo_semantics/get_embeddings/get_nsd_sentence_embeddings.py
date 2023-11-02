import os, pickle, nltk, h5py, random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings


def get_nsd_sentence_embeddings(embedding_model_type, captions_to_embed_path, RANDOMIZE_BY_WORD_TYPE, RANDOMIZE_WORD_ORDER,
                                h5_dataset_path, OVERWRITE):

    if not type(RANDOMIZE_BY_WORD_TYPE) == list:
        RANDOMIZE_BY_WORD_TYPE = [RANDOMIZE_BY_WORD_TYPE]

    print(f"GATHERING EMBEDDINGS FOR: {embedding_model}\n "
          f"ON: {captions_to_embed_path} \n "
          f"WITH RANDOMIZE_BY_WORD_TYPE: {RANDOMIZE_BY_WORD_TYPE}"
          f"AND RANDOMIZE_WORD_ORDER: {RANDOMIZE_WORD_ORDER}") 

    SANITY_CHECK = 1
    GET_EMBEDDINGS = 1
    FINAL_CHECK = 1

    save_every_n = 0  # if >0, save a checkpoint after every 10000 embeddings
    load_intermediate_result = 0  # if >0, load checkpoitn from f"./nsd_{embedding_model_type}_mean_embeddings_intermediate_{i}.pkl"

    save_test_imgs_to = "./_check_imgs"
    os.makedirs(save_test_imgs_to, exist_ok=1)
    save_embeddings_to = "../results_dir/saved_embeddings"
    os.makedirs("../results_dir", exist_ok=1)
    os.makedirs(save_embeddings_to, exist_ok=1)

    word_type_dict = {'noun': ['NN', 'NNS'], 'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'adjective': ['JJ', 'JJR', 'JJS'], 'adverb': ['RB', 'RBR', 'RBS'], 'preposition': ['IN']}

    save_name = f"{save_embeddings_to}/nsd_{embedding_model_type}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}_{'' if RANDOMIZE_BY_WORD_TYPE is None else 'RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"
    
    if os.path.exists(f"{save_name}.pkl") and not OVERWRITE:
        print(f"Embeddings already exist at {save_name}.pkl. Set OVERWRITE=True to overwrite.")
    else:
        embedding_model = get_embedding_model(embedding_model_type)

        if SANITY_CHECK:
            print("running sanity check")
            # inspired from tutorial https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder?fbclid=IwAR1hlPezVtDLZCF4f4Nr2JxXZmF8WcQ5FA-PBtYnuBIXlpzWlISRCHse4WM

            def plt_rdm(sentences, embeddings):
                distances = scipy.spatial.distance.pdist(embeddings, metric="cosine")
                plt.imshow(scipy.spatial.distance.squareform(distances), cmap="magma")
                plt.colorbar()
                plt.savefig(f"{save_test_imgs_to}/semantic_similarity_check_{embedding_model_type}.png")
                plt.close()

            def run_and_plot(sentences):
                sentence_embeddings = get_embeddings(sentences, embedding_model, embedding_model_type)
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

            ### HELPER FUNCTIONS
            def scramble_word_order(sentence):
                # helper function to randomize word order in sentences
                split = sentence.split()  # Split the string into a list of words
                shuffle(split)  # This shuffles the list in-place.
                return " ".join(split)  # Turn the list back into a string
            

            def randomize_by_word_type(sentence, types_to_randomize, loaded_captions, max_n_changes=0):
                # helper function to randomize by word type (e.g. randomize verbs)
                # can randomize several word types at once
                # max_n_changes: if >0, will only randomize up to this number of words

                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)

                if max_n_changes > 0:
                    change_ids = np.random.choice(len(tagged), max_n_changes, replace=False)
                
                change_dict = {}
                for this_type_to_rnd in types_to_randomize:
                    for this_subtype in word_type_dict[this_type_to_rnd]:
                        for tagged_word in tagged:
                            if tagged_word[1] == this_subtype:
                                # get a random word of the same type
                                random_word = tagged_word[0]
                                while random_word == tagged_word[0]:
                                    same_type = []
                                    while same_type == []:
                                        random_id = np.random.randint(len(loaded_captions))
                                        random_cap_id = np.random.randint(len(loaded_captions[random_id]))
                                        random_cap = loaded_captions[random_id][random_cap_id]  # gets a random caption
                                        random_tagged_word = nltk.pos_tag(nltk.word_tokenize(random_cap))
                                        same_type = [w[0] for w in random_tagged_word if w[1] == this_subtype]
                                    random_word = random.sample(same_type, 1)[0]
                                change_dict[tagged_word[0]] = random_word

                # replace the word in the sentence
                n_changes = 0
                split = sentence.split() # Split the string into a list of words
                for n_s, s in enumerate(split):  # change words
                    if max_n_changes == 0 or n_s in change_ids:
                        if s in change_dict.keys():
                            n_changes += 1
                            split[n_s] = change_dict[s]
                sentence = " ".join(split)   # put the sentence back together again

                return sentence, n_changes


            ### LET'S GO
            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)

            randomized_captions = []

            n_nsd_elements = len(loaded_captions)
            dummy_embeddings = get_embeddings(loaded_captions[0], embedding_model, embedding_model_type)

            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                mean_n_changes = 0

            if load_intermediate_result:
                init_i = load_intermediate_result
                with open(f"{save_name}_intermediate_{i}.pkl", "rb") as fp:
                    mean_embeddings = pickle.load(fp)
            else:
                init_i = 0
                mean_embeddings = np.empty((n_nsd_elements, dummy_embeddings.shape[-1]))

            for i in range(init_i, n_nsd_elements):
                if i % 1000 == 0:
                    print(f"\rRunning... {i/n_nsd_elements*100:.2f}%", end="")

                if save_every_n and i % 10000 == 0 and i > 0:
                    print(f"Saving intermediate result {i}")
                    with open(f"{save_name}_intermediate_{i}.pkl", "wb",) as fp:
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

                img_embeddings = get_embeddings(these_captions, embedding_model, embedding_model_type)
                mean_embeddings[i] = np.mean(img_embeddings, axis=0)

            with open(f"{save_name}.pkl", "wb") as fp:
                pickle.dump(mean_embeddings, fp)
            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                with open(f"{save_name}_randomized_captions.pkl", "wb") as fp:
                    pickle.dump(randomized_captions, fp)
                np.save(f"{save_name}_mean_n_changes.npy", mean_n_changes)
                print(f"Done. Mean number of changesin word typerandomization: {mean_n_changes}")

    if FINAL_CHECK:
        with h5py.File(h5_dataset_path, "r") as h5_dataset:
            total_n_stims = h5_dataset["test"]["labels"][:].shape[0]
            plot_n_imgs = 10
            step_size = total_n_stims // plot_n_imgs

            with open(captions_to_embed_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(f"{save_name}.pkl", "rb") as fp:
                loaded_mean_embeddings = pickle.load(fp)
            if len(RANDOMIZE_BY_WORD_TYPE) > 0:
                with open(f"{save_name}_randomized_captions.pkl", "rb") as fp:
                    loaded_randomized_captions = pickle.load(fp)

            for i in range(0, total_n_stims, step_size):
                plt.imshow(h5_dataset["test"]["data"][i])
                plt.title(
                    f"{loaded_captions[i][0]}\n"
                    f"{loaded_randomized_captions[i][0]}\n" if len(RANDOMIZE_BY_WORD_TYPE) > 0 else ""
                    f"Emb shape, min, max, mean: {loaded_mean_embeddings[i].shape, loaded_mean_embeddings[i].min(), loaded_mean_embeddings[i].max(), loaded_mean_embeddings[i].mean()}"
                )
                plt.savefig(f"{save_name}_check_{i}.png")
                plt.close()
