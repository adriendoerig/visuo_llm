import itertools

def get_name2file_list(saved_embeddings_dir,
                       ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir):

    # specify where each set of nsd embeddings is saved
    modelname2file = {
        # basic models
        "multihot": f"{saved_embeddings_dir}/nsd_multihot.pkl",
        "fasttext_categories": f"{saved_embeddings_dir}/nsd_fasttext_CATEGORY_mean_embeddings.pkl",
        "fasttext_nouns": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_mean_embeddings.pkl",
        "fasttext_nouns_closestCocoCats": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_mean_embeddings_cocoCatsMatch_positive_cut0.33.pkl",
        "fasttext_verbs": f"{saved_embeddings_dir}/nsd_fasttext_VERB_mean_embeddings.pkl",
        "fasttext_all": f"{saved_embeddings_dir}/nsd_fasttext_ALLWORDS_mean_embeddings.pkl",
        "glove_categories": f"{saved_embeddings_dir}/nsd_glove_CATEGORY_mean_embeddings.pkl",
        "glove_nouns": f"{saved_embeddings_dir}/nsd_glove_NOUNS_mean_embeddings.pkl",
        "glove_nouns_closestCocoCats": f"{saved_embeddings_dir}/nsd_glove_NOUNS_mean_embeddings_cocoCatsMatch_positive_cut0.33.pkl",
        "glove_verbs": f"{saved_embeddings_dir}/nsd_glove_VERB_mean_embeddings.pkl",
        "glove_all": f"{saved_embeddings_dir}/nsd_glove_ALLWORDS_mean_embeddings.pkl",
        "guse": f"{saved_embeddings_dir}/nsd_guse_mean_embeddings.pkl",
        "mpnet": f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_embeddings.pkl",

        # DNN activities
        "dnn_multihot_ff": f"{ms_coco_saved_dnn_activities_dir}/multihot_ff_nsd_activations_epoch200.h5",
        "dnn_multihot_rec": f"{ms_coco_saved_dnn_activities_dir}/multihot_rec_nsd_activations_epoch200.h5",
        "dnn_guse_ff": f"{ms_coco_saved_dnn_activities_dir}/guse_ff_nsd_activations_epoch200.h5",
        "dnn_guse_rec": f"{ms_coco_saved_dnn_activities_dir}/guse_rec_nsd_activations_epoch200.h5",
        "dnn_mpnet_ff": f"{ms_coco_saved_dnn_activities_dir}/mpnet_ff_nsd_activations_epoch200.h5",
        "dnn_mpnet_rec": f"{ms_coco_saved_dnn_activities_dir}/mpnet_rec_nsd_activations_epoch200.h5",

        # DNNs trained on ecoset activities
        "dnn_ecoset_category": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_category_post_gn_epoch80.h5",
        "dnn_ecoset_fasttext": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_fasttext_post_gn_epoch80.h5",
    }

    # sentence embeddings on (lists of) words
    modelname2file['mpnet_category_all'] = f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_CATEGORY_embeddings_allCats.pkl"
    modelname2file['mpnet_category_captionMatchPositive'] = f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_CATEGORY_embeddings_captionMatchPositive.pkl"
    modelname2file['mpnet_category_captionMatchNegative'] = f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_CATEGORY_embeddings_captionMatchNegative.pkl"

    # shuffled, and otherwise more complex models
    SENTENCE_EMBEDDING_MODEL_TYPES = ['all_mpnet_base_v2']  #'all_mpnet_base_v2', 'USE_CMLM_Base', 'openai_ada2', 'GUSE_transformer',  'GUSE_DAN', 'T5'
    RANDOMIZE_WORD_ORDER = False  # If True, word order will be randomized in each sentence.
    WORD_TYPES = ['noun', 'verb', 'adjective', 'adverb', 'preposition']
    RANDOMIZE_BY_WORD_TYPES = []  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
    for i in range(1, len(WORD_TYPES) + 1):
        RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])
    RANDOMIZE_BY_WORD_TYPES = [None] + RANDOMIZE_BY_WORD_TYPES  # add no randomization to the list

    for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
        for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
            this_save_name = f"nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"
            this_short_name = this_save_name.replace("nsd_", "").replace("_mean_embeddings", "").replace("all_mpnet_base_v2", "mpnet")
            modelname2file[this_short_name] = f"{saved_embeddings_dir}/{this_save_name}.pkl"

    return modelname2file
