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


# SOME CARE IS NEEDED TO TRANSLATE BETWEEN COCO 1-INDEXING AND PYTHON'S 0-INDEXING
def get_multihot_from_catIDs(catIDs, coco_categories_91):
    multihot = np.zeros(len(coco_categories_91))
    for c in catIDs:
        multihot[c+1] = 1
    return multihot


def get_words_from_multihot(multihot_labels, coco_categories_91):
    return [get_cocoCat_from_hotID(lin, coco_categories_91) for lin in np.where(multihot_labels == 1)[0]]


def get_hotID_from_cocoCat(word, coco_categories_91):
    return np.where(np.array(coco_categories_91) == word)[0][0] + 1


def get_cocoCat_from_hotID(hotID, coco_categories_91):
    return coco_categories_91[hotID - 1]


def get_multihot_from_cocoCatList(wordlist, coco_categories_91):
    multihot = np.zeros(len(coco_categories_91))
    for w in wordlist:
        multihot[get_hotID_from_cocoCat(w, coco_categories_91)] = 1
    return multihot


def get_cocoCat_embed_from_hotID(hotID, multihot_categs_embeddings):
    return multihot_categs_embeddings[hotID - 1]


def get_cocoCat_embed_from_cocoCat(word, coco_categories_91, multihot_categs_embeddings):
    return get_cocoCat_embed_from_hotID(get_hotID_from_cocoCat(word, coco_categories_91), multihot_categs_embeddings)


def get_closest_cocoCatAndID(word_embedding, coco_categories_91, multihot_categs_embeddings, METRIC):
    lookup_distances = cdist(multihot_categs_embeddings, word_embedding[None,:], metric=METRIC)
    min_dist = np.min(lookup_distances)
    cat_ID = np.argmin(lookup_distances)
    cat_word = coco_categories_91[cat_ID]
    return cat_word, cat_ID, min_dist


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


def get_sentence_tags(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return tagged


def custom_join(words):
    modified_sentence = ''
    for i, word in enumerate(words):
        # Add space unless the current word is a punctuation mark or starts with an apostrophe
        if i > 0 and word not in ",.?!;:" and not word.startswith("'"):
            modified_sentence += ' '
        modified_sentence += word
    return modified_sentence


def get_word_type_from_string(s, word_type):
    word_type_dict = get_word_type_dict()
    tagged = get_sentence_tags(s)
    return [x[0] for x in tagged if x[1] in word_type_dict[word_type]]  # NN and NNS the tags for nouns


def scramble_word_order(sentence):
    # helper function to randomize word order in sentences
    split = nltk.word_tokenize(sentence)  # Split the string into a list of words
    shuffle(split)  # This shuffles the list in-place.
    return custom_join(split)  # Turn the list back into a string


def randomize_by_word_type(sentence, types_to_randomize, loaded_captions, min_dist_cutoff, 
                           embedding_model, embedding_model_type,
                           max_n_changes=0, max_n_trials=50, metric='correlation'):
    # helper function to randomize by word type (e.g. randomize verbs)
    # can randomize several word types at once
    # max_n_changes: if >0, will only randomize up to this number of words
    # max_n_trials: if >0, will only try this number of randomizations (to make sure we don't get stuck in an infinite while loop)

    word_type_dict = get_word_type_dict()
    tagged = get_sentence_tags(sentence)

    if max_n_changes > 0:
        change_ids = np.random.choice(len(tagged), max_n_changes, replace=False)
    
    change_dict = {}
    for this_type_to_rnd in types_to_randomize:
        for this_subtype in word_type_dict[this_type_to_rnd]:
            for tagged_word in tagged:
                if tagged_word[1] == this_subtype:
                    # get a random word of the same type
                    max_dist_word = tagged_word[0]
                    while max_dist_word == tagged_word[0]:
                        # same_type = []
                        this_dist, max_dist, n_trials = 0, 0, 0
                        max_dist_word = ''
                        while this_dist <= min_dist_cutoff and n_trials < max_n_trials:
                            same_type = []
                            while same_type == []:
                                random_id = np.random.randint(len(loaded_captions))
                                random_cap_id = np.random.randint(len(loaded_captions[random_id]))
                                random_cap = loaded_captions[random_id][random_cap_id]  # gets a random caption
                                random_tagged_word = get_sentence_tags(random_cap)
                                same_type = [w[0] for w in random_tagged_word if w[1] == this_subtype]
                            random_word = random.sample(same_type, 1)[0]
                            this_dist = cdist(get_embeddings(random_word, embedding_model, embedding_model_type)[None,:], 
                                              get_embeddings(tagged_word[0], embedding_model, embedding_model_type)[None,:], 
                                              metric=metric)
                            if this_dist > max_dist:
                                max_dist = this_dist
                                max_dist_word = random_word
                            # if n_trials > 1:
                            #     print(f"\nn_trials... {n_trials}, orig_word: {tagged_word[0]}, random_word: {random_word}, max_dist_word: {max_dist_word} max_dist: {max_dist}", end="")
                            n_trials += 1
                            
                    if tagged_word[0] not in change_dict.keys():
                        # if statement: e.g. avoids "unchanging" a word changed with NN tag, when looking at "NNS"
                        change_dict[tagged_word[0]] = max_dist_word

    # replace the word in the sentence
    n_changes = 0
    split = nltk.word_tokenize(sentence) # Split the string into a list of words
    for n_s, s in enumerate(split):  # change words
        if max_n_changes == 0 or n_s in change_ids:
            if s in change_dict.keys():
                n_changes += 1
                split[n_s] = change_dict[s]
    sentence = custom_join(split)   # put the sentence back together again

    return sentence, n_changes

