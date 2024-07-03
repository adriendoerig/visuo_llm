import h5py, pickle, os
from nsd_visuo_semantics.get_embeddings.get_nsd_noun_embeddings_simple import get_nsd_noun_embeddings_simple
from nsd_visuo_semantics.get_embeddings.get_nsd_category_embeddings_simple import get_nsd_category_embeddings_simple
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_categories_simple import get_nsd_sentence_embeddings_categories_simple
from nsd_visuo_semantics.get_embeddings.get_nsd_verb_embeddings_simple import get_nsd_verb_embeddings_simple
from nsd_visuo_semantics.get_embeddings.get_nsd_sentence_embeddings_simple import get_nsd_sentence_embeddings_simple
from nsd_visuo_semantics.get_embeddings.get_nsd_allWord_embeddings_simple import get_nsd_allWord_embeddings_simple


OVERWRITE = False

# GENERAL PATHS, ETC
h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"
base_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings"
fasttext_embeddings_path = f"{base_path}/crawl-300d-2M.vec"
glove_embeddings_path = f"{base_path}/glove.840B.300d.txt"
captions_to_embed_path = f"{base_path}/ms_coco_nsd_captions_test.pkl"

SAVE_PATH = "../results_dir/saved_embeddings"
os.makedirs(SAVE_PATH, exist_ok=1)

# GENERAL SENTENCE EMBEDDING PARAMETERS
# SENTENCE_EMBEDDING_MODEL_TYPES = ['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
#                                   'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
#                                   'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
#                                   'GUSE_transformer', 'GUSE_DAN', 'USE_CMLM_Base', 'T5']  # these are all possibilities
SENTENCE_EMBEDDING_MODEL_TYPES = ['all-mpnet-base-v2']

for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:

    get_nsd_sentence_embeddings_simple(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path, 
                                       h5_dataset_path, SAVE_PATH, OVERWRITE)
    
    get_nsd_sentence_embeddings_categories_simple(SENTENCE_EMBEDDING_MODEL_TYPE, captions_to_embed_path, 
                                                  h5_dataset_path, SAVE_PATH, OVERWRITE=OVERWRITE)

WORD_EMBEDDING_TYPES = ['all-mpnet-base-v2', 'fasttext', 'glove']

for WORD_EMBEDDING_TYPE in WORD_EMBEDDING_TYPES:

    get_nsd_noun_embeddings_simple(WORD_EMBEDDING_TYPE, h5_dataset_path, 
                                   fasttext_embeddings_path, glove_embeddings_path, 
                                   captions_to_embed_path, SAVE_PATH, OVERWRITE=OVERWRITE)

    get_nsd_verb_embeddings_simple(WORD_EMBEDDING_TYPE, h5_dataset_path, 
                                   fasttext_embeddings_path, glove_embeddings_path, 
                                   captions_to_embed_path, SAVE_PATH, OVERWRITE=OVERWRITE)
    
    get_nsd_allWord_embeddings_simple(WORD_EMBEDDING_TYPE, h5_dataset_path, 
                                      fasttext_embeddings_path, glove_embeddings_path, 
                                      captions_to_embed_path, SAVE_PATH, OVERWRITE=OVERWRITE)

    get_nsd_category_embeddings_simple(WORD_EMBEDDING_TYPE, h5_dataset_path, 
                                       fasttext_embeddings_path, glove_embeddings_path, 
                                       captions_to_embed_path, SAVE_PATH, OVERWRITE)
    
# FInally, this gets the multihot embeddings from the .h5 dataset provided with the code
with h5py.File(h5_dataset_path,'r') as f:
        
        multihot = f['test']['img_multi_hot'][:]
        
        with open(f"{SAVE_PATH}/nsd_multihot.pkl", "wb") as fp:  # Pickling
                pickle.dump(multihot, fp)