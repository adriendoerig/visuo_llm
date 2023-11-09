import itertools
from nsd_visuo_semantics.get_embeddings.get_nsd_noun_embeddings import get_nsd_noun_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_verb_embeddings import get_nsd_verb_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings import get_nsd_sentence_embeddings
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_categories import get_nsd_sentence_embeddings_categories
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_wordtypes import get_nsd_sentence_embeddings_wordtypes
from nsd_visuo_semantics.get_embeddings.get_nsd_allWord_embeddings import get_nsd_allWord_embeddings


OVERWRITE = True


h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"
fasttext_embeddings_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/crawl-300d-2M.vec"
glove_embeddings_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/glove.840B.300d.txt"
ms_coco_nsd_train_captions = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_train.pkl"
ms_coco_nsd_val_captions = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_val.pkl"
nsd_captions_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"

SENTENCE_EMBEDDING_MODEL_TYPE = 'all_mpnet_base_v2'  #'all_mpnet_base_v2', 'USE_CMLM_Base', 'openai_ada2', 'GUSE_transformer',  'GUSE_DAN', 'T5'
WORD_TYPES = ['noun', 'verb', 'adjective', 'adverb', 'preposition']

# RANDOMIZE_WORD_ORDER = False  # If True, word order will be randomized in each sentence.
# RANDOMIZE_BY_WORD_TYPES = []  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
# for i in range(1, len(WORD_TYPES) + 1):
#     RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])
# RANDOMIZE_BY_WORD_TYPES = [None] + RANDOMIZE_BY_WORD_TYPES  # add no randomization to the list

# for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
#     get_nsd_sentence_embeddings(SENTENCE_EMBEDDING_MODEL_TYPE, nsd_captions_path, RANDOMIZE_BY_WORD_TYPE, RANDOMIZE_WORD_ORDER,
#                                 h5_dataset_path, OVERWRITE=OVERWRITE)
    

# get_nsd_sentence_embeddings_categories(SENTENCE_EMBEDDING_MODEL_TYPE, nsd_captions_path,
#                                        h5_dataset_path, OVERWRITE=OVERWRITE)

for concat_five_captions in [True]:#, False]:
        get_nsd_sentence_embeddings_wordtypes(SENTENCE_EMBEDDING_MODEL_TYPE, nsd_captions_path,
                                        WORD_TYPES, concat_five_captions,
                                        h5_dataset_path, OVERWRITE)


WORD_EMBEDDING_TYPES = ['glove', 'fasttext']  # or 'glove'
WORD_CONCATENATE_EMBEDDINGS = [False]  # if True, we concatenate the embeddings. If false, we mean them.

WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS = ['positive', 'negative', None]  # if 'positive', we only use nouns that have an embedding CLOSE to COCO category nouns. 
                                                                          # if 'negative', we only use nouns that have an embedding FAR to COCO category nouns.
                                                                          # if None, we use all nouns.

# for WORD_EMBEDDING_TYPE in WORD_EMBEDDING_TYPES:
#     for WORD_CONCATENATE_EMBEDDINGS in WORD_CONCATENATE_EMBEDDINGS:

        # get_nsd_verb_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS,
        #                         h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE=OVERWRITE)
        
        # get_nsd_allWord_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS,
        #                            h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE=OVERWRITE)

        # for WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS in WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS:
        #     get_nsd_noun_embeddings(WORD_EMBEDDING_TYPE, WORD_CONCATENATE_EMBEDDINGS, WORD_NOUNS_MATCH_TO_COCO_CATEGORY_NOUNS,
        #                             h5_dataset_path, fasttext_embeddings_path, glove_embeddings_path, nsd_captions_path, OVERWRITE=OVERWRITE)
