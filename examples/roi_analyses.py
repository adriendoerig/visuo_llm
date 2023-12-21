import os, itertools
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses import nsd_roi_analyses
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses_figure import nsd_roi_analyses_figure
from nsd_visuo_semantics.get_embeddings.correlate_model_rdms_figure import correlate_model_rdms_figure


### DECLARE PARAMS

for roi_analysis_dnn_layer_to_use in [-1]:#, -5]:#, -2, -3, -4, -5]:

    plt_suffix = f"_dnns"

    OVERWRITE = False

    MODEL_INPUT_DATA = ['full_nsd']  # ['nsd_special100_gpt4Captions', 'nsd_special100_cocoCaptions']  # 'full_nsd', 'nsd_special100_gpt4Captions' or 'nsd_special100_cocoCaptions'

    WORD_TYPES = ['noun']  #, 'verb']#, 'adjective', 'adverb', 'preposition']

    MODEL_NAMES = []

    # custom selection
    MODEL_NAMES += []

    ### BELOW IS CODE TO GET ALL MODELS
    # models from original paper
    MODEL_NAMES += [
        # "multihot",
        "mpnet",
        # "fasttext_categories",
        # "fasttext_verbs",
        # "fasttext_all",
        # "guse",
        # "dnn_ecoset_category",
        # "dnn_ecoset_fasttext",
    ]

    # dnn 10 seeds
    # for epoch in [200]:
    #     for target in ['multihot', 'mpnet']:
    #         for seed in range(1, 11):
    #             MODEL_NAMES += [f"dnn_{target}_rec_seed{seed}_ep{epoch}"]    

    # dnn extensive check
    MODEL_NAMES += [
        # "dnn_multihot_rec_ep0",
        # "dnn_multihot_rec_ep100",
        # "dnn_multihot_rec_ep200",
        # "dnn_multihot_rec_ep300",
        # "dnn_multihot_rec_ep400",
        # 'dnn_multihot_rec_seed1_softmax_ep100',
        # 'dnn_multihot_rec_seed1_softmax_ep200',

        # "dnn_mpnet_rec_ep0",
        # "dnn_mpnet_rec_ep100",
        # "dnn_mpnet_rec_ep200",
        # "dnn_mpnet_rec_ep300",
        # "dnn_mpnet_rec_ep400",
        # "dnn_mpnet_rec_seed1_ep100",
        # "dnn_mpnet_rec_seed1_ep200",

        # "dnn_multihot_rec_old_ep200",
        # "dnn_mpnet_rec_old_ep200",

        # "dnn_simclr_rec_ep200",
    ]

    ### SENTENCE EMBEDDINGS
    # SENTENCE_EMBEDDING_MODEL_TYPES = ['mpnet', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
    #                                   'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
    #                                   'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
    #                                   'GUSE_transformer', 'GUSE_DAN']#, 'USE_CMLM_Base', 'T5']
    # SENTENCE_EMBEDDING_MODEL_TYPES = ['all_mpnet_base_v2']
    # RANDOMIZE_WORD_ORDER = False  # If True, word order will be randomized in each sentence.

    # sentence embeddings with various randomized words
    # RANDOMIZE_BY_WORD_TYPES = []  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
    # for i in range(1, len(WORD_TYPES) + 1):
    #     # randomize all combinations of word types
    #     RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])
    # RANDOMIZE_BY_WORD_TYPES = [[w] for w in WORD_TYPES]  # Just randomize single word types
    # RANDOMIZE_BY_WORD_TYPES = [None, ['noun'], ['verb']]  # add no randomization to the list

    # for RANDOMIZE_WORD_ORDER in [False]:
    #     for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
    #         for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
                
    #             this_save_name = f"nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"     
    #             this_short_name = this_save_name.replace("nsd_", "").replace("_mean_embeddings", "")#.replace("all_mpnet_base_v2", "mpnet")
                
    #             # add model without cutoff distance in the randomization
    #             # MODEL_NAMES.append(this_short_name)

    #             # add model with cutoff dist = 0.7
    #             if RANDOMIZE_BY_WORD_TYPE is None:
    #                 # there is no cutoff when there is no randomization
    #                 MODEL_NAMES.append(this_short_name)
    #             else:
    #                 this_short_name_cutoff07 = this_short_name + "_cutoffDist0.7"
    #                 MODEL_NAMES.append(this_short_name_cutoff07)



    # sentence embeddings on (lists of) words of a single type (e.g. nouns)
    # for wt in WORD_TYPES:
    #     MODEL_NAMES += [f"mpnet_{wt}s"]
        # MODEL_NAMES += [f"mpnet_{wt}s_concat5caps"]
        # for max_n_words in range(1,6):
        #     MODEL_NAMES += [f'mpnet_{wt}_max{max_n_words}words']
            # MODEL_NAMES += [f'mpnet_{wt}_max{max_n_words}words_concat5caps']

    # sentence embeddings with words matched to coco categories in various ways
    # for cutoff in [0.3, 0.5]:
    #     MODEL_NAMES += [
    #         # f"mpnet_category_all",
    #         # f"mpnet_category_catNamesCaptionMatchPositive_cutoff{cutoff}",
    #         # f"mpnet_category_catNamesCaptionMatchNegative_cutoff{cutoff}",
    #         # f"mpnet_category_captionNounsCatMatchPositive_cutoff{cutoff}",
    #         # f"mpnet_category_captionNounsCatMatchNegative_cutoff{cutoff}",
    #         # f"mpnet_category_captionNounsCatNameMap_cutoff{cutoff}",
    #         # f"mpnet_category_captionNounsCatNameMapMultihot_cutoff{cutoff}"
    #     ]
        
    # other new models
    MODEL_NAMES += [
        # "glove_verbs",
        "glove_all",
        # "glove_nouns_cocoCatsMatchPositive",
        # "glove_nouns_cocoCatsMatchNegative",
        # "glove_nouns",
        # "fasttext_nouns_cocoCatsMatchPositive",
        # "fasttext_nouns_cocoCatsMatchNegative",
        # "fasttext_nouns",
        # "mpnetWordAvg_all",
        # "mpnetWordAvg_nouns",
        # "mpnetWordAvg_verbs",
        # "CLIP_RN50_text",
        # "CLIP_RN50_images",
        # "CLIP_ViT_text",
        # "CLIP_ViT_images",
    ]

    MODEL_NAMES = list(set(MODEL_NAMES))
    dummy = MODEL_NAMES.copy()
    for mn in dummy:
        for mi in MODEL_INPUT_DATA:
            if mi == 'full_nsd':
                pass
            else:
                MODEL_NAMES.append(mn + '_' + mi)
        if 'full_nsd' not in MODEL_INPUT_DATA:
            MODEL_NAMES.remove(mn)


    # if true, the 515 stimuli seen by all subjects are removed (so they can be used in the test set of other experiments
    # based on searchlight maps while avoiding double-dipping)
    remove_shared_515 = False

    # RDM distance measure for models NOTE: BRAIN RDMS ARE DONE WITH CORRELATION DISTANCE
    models_rdm_distance = "correlation"

    # if we are using a DNN, use last layer (and last timestep if recurrent). If you want another layer,
    # find its index (between in [0 ,n_layers*n_timesteps-1]) and apply it here. Ignored if "dnn_" not in MODEL_NAME
    # roi_analysis_dnn_layer_to_use = -1

    which_rois = "streams"  # streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...

    plot_noise_ceiling = True  # if True, plot noise-ceiling corrected corrs. If false, do not use noise ceiling

    ### PATHS
    base_save_dir = "../results_dir"  # base dir from which to load model RDMs and in which to save results
    nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
    saved_embeddings_dir = f"{base_save_dir}/saved_embeddings"
    base_networks_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets'
    # ms_coco_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ms_coco_nets/extracted_activities"
    ms_coco_saved_dnn_activities_dir = f"/share/klab/adoerig/adoerig/nsd_visuo_semantics/examples/dnn_extracted_activities"
    ecoset_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ecoset_nets/extracted_activities"
    rdms_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{models_rdm_distance}'
    betas_dir = os.path.join(nsd_dir, '..', "NSD_for_visuo_semantics_derivatives", "betas")
    rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')


    ### PREPARE RDMs FOR EACH REQUESTED MODEL

    # nsd_prepare_modelrdms(MODEL_NAMES, models_rdm_distance,
    #                       saved_embeddings_dir, rdms_dir, nsd_dir,
    #                       ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
    #                       remove_shared_515, OVERWRITE)


    # ### RUN ROI ANALYSES

    for i, m in enumerate(MODEL_NAMES):
        if 'dnn' in m:
            MODEL_NAMES[i] = f'{m}_layer{roi_analysis_dnn_layer_to_use}'

    # nsd_roi_analyses(MODEL_NAMES, models_rdm_distance, roi_analysis_dnn_layer_to_use, which_rois,
    #                 nsd_dir, betas_dir, rois_dir, base_save_dir,
    #                 remove_shared_515, OVERWRITE_NEURO_RDMs=False, OVERWRITE_RDM_CORRs=OVERWRITE)


    nsd_roi_analyses_figure(base_save_dir, which_rois, models_rdm_distance, plot_noise_ceiling, 
                            fig_id=0, custom_model_keys=MODEL_NAMES, plt_suffix=plt_suffix,
                            custom_model_labels=None, average_seeds=True)


    # correlate_model_rdms_figure(MODEL_NAMES, nsd_dir, base_save_dir, models_rdm_distance, 
    #                             remove_shared_515, roi_analysis_dnn_layer_to_use, 
    #                             plt_suffix=plt_suffix, 
    #                             COMPUTE=True, OVERWRITE=OVERWRITE)