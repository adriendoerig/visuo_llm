import itertools
from nsd_visuo_semantics.get_embeddings.get_nsd_noun_embeddings import get_nsd_noun_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_category_embeddings import get_nsd_category_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_verb_embeddings import get_nsd_verb_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings import get_nsd_sentence_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_categories import get_nsd_sentence_embeddings_categories
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_wordtypes import get_nsd_sentence_embeddings_wordtypes
from nsd_visuo_semantics.get_embeddings.get_nsd_allWord_embeddings import get_nsd_allWord_embeddings


OVERWRITE = 1

# GENERAL PATHS, ETC
h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"
base_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings"
fasttext_embeddings_path = f"{base_path}/crawl-300d-2M.vec"
glove_embeddings_path = f"{base_path}/glove.840B.300d.txt"

ms_coco_nsd_train_captions = f"{base_path}/ms_coco_nsd_captions_train.pkl"
ms_coco_nsd_val_captions = f"{base_path}/ms_coco_nsd_captions_val.pkl"
nsd_captions_path = f"{base_path}/ms_coco_nsd_captions_test.pkl"
nsd_special100_gpt4Captions_path = f"{base_path}/nsd_special100_gpt4Captions.pkl"
nsd_special100_cocoCaptions_path = f"{base_path}/nsd_special100_cocoCaptions.pkl"
nsd_all_gpt4CaptionsLong = f"/share/klab/adoerig/adoerig/nsd_all_gpt4CaptionsLong.npy"

captions_to_embed_path = nsd_all_gpt4CaptionsLong

# GENERAL SENTENCE EMBEDDING PARAMETERS
# SENTENCE_EMBEDDING_MODEL_TYPES = ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
#                                   'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
#                                   'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
#                                   'GUSE_transformer', 'GUSE_DAN', 'USE_CMLM_Base', 'T5']  
SENTENCE_EMBEDDING_MODEL_TYPES = ['all-mpnet-base-v2']
for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
    # JUST STANDARD EMBEDDINGS
    get_nsd_sentence_embeddings(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path, 
                                None, None, 0, 
                                h5_dataset_path, 
                                use_saved_randomized_sentences_from_other_model=None,
                                OVERWRITE=OVERWRITE)
# WORD_TYPES = ['noun', 'verb']  # , 'adjective', 'adverb', 'preposition']

# RANDOMIZATION VERSIONS OF NSD SENTENCE EMBEDDINGS
# RANDOMIZE_WORD_ORDER = False  # If True, word order will be randomized in each sentence.
# RANDOMIZE_BY_WORD_TYPES = [None] #+ WORD_TYPES  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
# for i in range(1, len(WORD_TYPES) + 1):
#     RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])

# for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
    # for MIN_DIST_CUTOFF in [0.7]:
    #     for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
    #         get_nsd_sentence_embeddings(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path, 
    #                                     RANDOMIZE_BY_WORD_TYPE, RANDOMIZE_WORD_ORDER, MIN_DIST_CUTOFF, 
    #                                     h5_dataset_path, 
    #                                     use_saved_randomized_sentences_from_other_model='all_mpnet_base_v2',
    #                                     OVERWRITE=OVERWRITE)
    
    # for CUTOFF in [0.5, 0.3]:
    #     get_nsd_sentence_embeddings_categories(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path,
    #                                            h5_dataset_path, CUTOFF, OVERWRITE=OVERWRITE)

    # for concat_five_captions in [False, True]:
    #     for max_n_words_per_caption in [0]:  # range(1,6):
    #         get_nsd_sentence_embeddings_wordtypes(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path,
    #                                             WORD_TYPES, concat_five_captions, max_n_words_per_caption,
    #                                             h5_dataset_path, OVERWRITE)


# WORD_EMBEDDING_TYPES = ['all-mpnet-base-v2']  # or 'fasttext', 'glove'
# WORD_CONCATENATE_EMBEDDINGS = [False]  # if True, we concatenate the embeddings. If false, we mean them.

# WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS = [None]  # if 'positive', we only use nouns that have an embedding CLOSE to COCO category nouns. 
                                                                          # if 'negative', we only use nouns that have an embedding FAR to COCO category nouns.
                                                                          # if None, we use all nouns.

# for WORD_EMBEDDING_TYPE in WORD_EMBEDDING_TYPES:
#     for WORD_CONCATENATE_EMBEDDINGS in WORD_CONCATENATE_EMBEDDINGS:

#         get_nsd_verb_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS,
#                                 h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, captions_to_embed_path, OVERWRITE=OVERWRITE)
        
#         get_nsd_allWord_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS,
#                                    h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, captions_to_embed_path, OVERWRITE=OVERWRITE)

#         for WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS in WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS:
#             get_nsd_noun_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS, WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS,
#                                     h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, captions_to_embed_path, OVERWRITE=OVERWRITE)

# get_nsd_category_embeddings('glove', h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE)