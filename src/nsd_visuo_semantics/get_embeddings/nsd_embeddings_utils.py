import nltk, random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.spatial.distance import cdist
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embeddings


def sentence_embeddings_sanity_check(embedding_model_type, embedding_model, metric, save_test_imgs_to):
    print("running sanity check")
    # inspired from tutorial https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder?fbclid=IwAR1hlPezVtDLZCF4f4Nr2JxXZmF8WcQ5FA-PBtYnuBIXlpzWlISRCHse4WM

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

    sentence_embeddings = get_embeddings(sentences, embedding_model, embedding_model_type)
    distances = scipy.spatial.distance.pdist(sentence_embeddings, metric=metric)
    plt.imshow(scipy.spatial.distance.squareform(distances), cmap="magma")
    plt.colorbar()
    plt.savefig(f"{save_test_imgs_to}/semantic_similarity_check_{embedding_model_type}.png")
    plt.close() 


def get_words_from_multihot(multihot_labels, coco_categories_91):
    return [coco_categories_91[lin - 1] for lin in np.where(multihot_labels == 1)[0]]


def get_all_word_embeds_from_wordlist(wordlist, embedding_model, embedding_model_type):
    all_words_embeds = get_embeddings(wordlist, embedding_model, embedding_model_type)
    return all_words_embeds


def get_all_words_from_captions(captions):
    all_words = []
    for c in captions:
        all_words.append(nltk.word_tokenize(c))
    return all_words


def get_all_word_embeds_from_caption(captions, embedding_model, embedding_model_type):
    all_words_embeds = []
    for c in captions:
        all_words = nltk.word_tokenize(c)
        all_words_embeds.append(get_embeddings(all_words, embedding_model, embedding_model_type))
    return np.concatenate(all_words_embeds, axis=0)


def filter_categs_to_keep(these_categ_words, these_categ_embeds, these_caption_word_embeds, categs_to_keep, CUTOFF, METRIC):
    lookup_distances = cdist(these_categ_embeds, these_caption_word_embeds, metric=METRIC)
    if categs_to_keep == 'positive':
        lookup_distances[lookup_distances>CUTOFF] = 0
    elif categs_to_keep == 'negative':
        lookup_distances[lookup_distances<=CUTOFF] = 0
    match_sum = np.sum(lookup_distances, axis=1)
    match_idx = np.where(match_sum>0)[0]
    return [these_categ_words[i] for i in match_idx]


def get_word_type_dict():
    return {'noun': ['NN', 'NNS'], 'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'adjective': ['JJ', 'JJR', 'JJS'], 'adverb': ['RB', 'RBR', 'RBS'], 'preposition': ['IN'],
            'nouns': ['NN', 'NNS'], 'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'adjectives': ['JJ', 'JJR', 'JJS'], 'adverbs': ['RB', 'RBR', 'RBS'], 'prepositions': ['IN']}


def get_word_type_from_string(s, word_type):
    word_type_dict = get_word_type_dict()
    tokens = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(tokens)
    return [x[0] for x in tagged if x[1] in word_type_dict[word_type]]  # NN and NNS the tags for nouns


def scramble_word_order(sentence):
    # helper function to randomize word order in sentences
    split = sentence.split()  # Split the string into a list of words
    shuffle(split)  # This shuffles the list in-place.
    return " ".join(split)  # Turn the list back into a string


def randomize_by_word_type(sentence, types_to_randomize, loaded_captions, max_n_changes=0):
    # helper function to randomize by word type (e.g. randomize verbs)
    # can randomize several word types at once
    # max_n_changes: if >0, will only randomize up to this number of words

    word_type_dict = get_word_type_dict()
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

