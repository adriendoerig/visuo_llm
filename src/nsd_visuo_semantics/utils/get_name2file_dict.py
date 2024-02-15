import itertools

def get_name2file_dict(saved_embeddings_dir, saved_dnn_activities_dir,
                       ecoset_saved_dnn_activities_dir):

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
        "glove_nouns_cocoCatsMatchPositive": f"{saved_embeddings_dir}/nsd_glove_NOUNS_mean_embeddings_cocoCatsMatch_positive_cut0.33.pkl",
        "glove_nouns_cocoCatsMatchNegative": f"{saved_embeddings_dir}/nsd_glove_NOUNS_mean_embeddings_cocoCatsMatch_negative_cut0.33.pkl",
        "glove_verbs": f"{saved_embeddings_dir}/nsd_glove_VERB_mean_embeddings.pkl",
        "glove_all": f"{saved_embeddings_dir}/nsd_glove_ALLWORDS_mean_embeddings.pkl",
        "CLIP_ViT_text": f"{saved_embeddings_dir}/nsd_CLIP-vit_mean_embeddings.pkl",
        "CLIP_ViT_images": f"{saved_dnn_activities_dir}/CLIP-vit_nsd_image_features.pkl",
        "CLIP_RN50_text": f"{saved_embeddings_dir}/nsd_CLIP-rn50_mean_embeddings.pkl",
        "CLIP_RN50_images": f"{saved_dnn_activities_dir}/CLIP-rn50_nsd_image_features.pkl",
        "thingsvision_cornet-s": f"{saved_dnn_activities_dir}/thingsvision_cornet-s_nsd_image_features.pkl",
        "thingsvision_simclr-rn50": f"{saved_dnn_activities_dir}/thingsvision_simclr-rn50_nsd_image_features.pkl",
        "thingsvision_barlowtwins-rn50": f"{saved_dnn_activities_dir}/thingsvision_barlowtwins-rn50_nsd_image_features.pkl",
        "brainscore_alexnet": f"{saved_dnn_activities_dir}/brainscore_alexnet_nsd_image_features.pkl",
        "brainscore_regnet_y_400mf": f"{saved_dnn_activities_dir}/brainscore_regnet_y_400mf_nsd_image_features.pkl",
        "brainscore_resnet50_julios": f"{saved_dnn_activities_dir}/brainscore_resnet50_julios_nsd_image_features.pkl",
        "brainscore_tv_efficientnet-b1": f"{saved_dnn_activities_dir}/brainscore_tv_efficientnet-b1_nsd_image_features.pkl",
        "resnext101_32x8d_wsl": f"{saved_dnn_activities_dir}/resnext101_32x8d_wsl_nsd_image_features.pkl",
        "google_simclrv1_rn50": f"{saved_dnn_activities_dir}/google_simclrv1_rn50_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_nsd_image_features.pkl",  # these are with inputs in [0,255] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01_01inputs": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_01inputs_nsd_image_features.pkl",  # these are with inputs in [0,1] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x_01inputs": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_01inputs_nsd_image_features.pkl",
        "mpnetWordAvg_all": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_ALLWORDS_mean_embeddings.pkl",
        "mpnetWordAvg_nouns": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_NOUNS_mean_embeddings.pkl",
        "mpnetWordAvg_verbs": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_VERB_mean_embeddings.pkl",
        "guse": f"{saved_embeddings_dir}/nsd_guse_mean_embeddings.pkl",
        "mpnet": f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_embeddings.pkl",
        "all-mpnet-base-v2": f"{saved_embeddings_dir}/nsd_all_mpnet_base_v2_mean_embeddings.pkl",  # this is a duplicate of the line above, both names work

        # DNNs trained on ecoset activities
        "dnn_ecoset_category": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_category_post_gn_epoch80.h5",
        "dnn_ecoset_fasttext": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_fasttext_post_gn_epoch80.h5",
    }

    # DNN activities
    old_dnn_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets/semantics_paper_ms_coco_nets/extracted_activities'
    for epoch in [0, 100, 200, 300, 400]:
        for modelname in ["multihot_rec", "multihot_ff", "mpnet_rec", "mpnet_ff", 'simclr_rec', 'simclr_ff', 'guse_rec', 'guse_ff', 'mpnet_rec_small', 'multihot_rec_small', 'mpnet_rec_old', 'multihot_rec_old']:
            if '_old' in modelname:
                modelname2file[f"dnn_{modelname}_ep{epoch}"] = f"{old_dnn_dir}/{modelname.replace('_old', '')}_nsd_activations_epoch{epoch}.h5"
                for seed in range(1,11):
                    modelname2file[f"dnn_{modelname}_seed{seed}_ep{epoch}"] = f"{old_dnn_dir}/{modelname.replace('_old', '')}_seed{seed}_nsd_activations_epoch{epoch}.h5"
            else:    
                modelname2file[f"dnn_{modelname}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_nsd_activations_epoch{epoch}.h5"
                for seed in range(1,11):
                    modelname2file[f"dnn_{modelname}_seed{seed}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_seed{seed}_nsd_activations_epoch{epoch}.h5"

                    # weirder non-standard ones
                    modelname2file[f'dnn_multihot_rec_seed{seed}_softmax_ep{epoch}'] = f"{saved_dnn_activities_dir}/multihot_rec_seed{seed}_softmax_nsd_activations_epoch{epoch}.h5"

    # word types embeddings
    WORD_TYPES = ['noun', 'verb', 'adjective', 'adverb', 'preposition']

    # sentence embeddings on (lists of) words
    for mpnet_moniker in ["mpnet", "all-mpnet-base-v2"]:
        mpnet_full_name = "all-mpnet-base-v2" if mpnet_moniker == "all-mpnet-base-v2" else "all_mpnet_base_v2"
        for cutoff in [0.3, 0.5, 0.7]:
            modelname2file[f"{mpnet_moniker}_category_all"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF0.5_allCats.pkl"
            modelname2file[f"{mpnet_moniker}_category_catNamesCaptionMatchPositive_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_catNamesCaptionMatchPositive.pkl"
            modelname2file[f"{mpnet_moniker}_category_catNamesCaptionMatchNegative_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_catNamesCaptionMatchNegative.pkl"
            modelname2file[f"{mpnet_moniker}_category_captionNounsCatMatchPositive_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_captionNounsCatMatchPositive.pkl"
            modelname2file[f"{mpnet_moniker}_category_captionNounsCatMatchNegative_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_captionNounsCatMatchNegative.pkl"
            modelname2file[f"{mpnet_moniker}_category_captionNounsCatNameMap_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_captionNounsCatNameMap.pkl"
            modelname2file[f"{mpnet_moniker}_category_captionNounsCatNameMapMultihot_cutoff{cutoff}"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_CATEGORY_embeddings_CUTOFF{cutoff}_captionNounsCatNameMapMultihot.pkl"
        
        for wt in WORD_TYPES:
            modelname2file[f'{mpnet_moniker}_{wt}'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_WORDTYPE_embeddings_{wt}s.pkl"  # same as below, but want to catch both singular or plural in the short name
            modelname2file[f'{mpnet_moniker}_{wt}s'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_WORDTYPE_embeddings_{wt}s.pkl"
            modelname2file[f'{mpnet_moniker}_{wt}_concat5caps'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_concat5caps_WORDTYPE_embeddings_{wt}s.pkl"  # same as below, but want to catch both singular or plural in the short name
            modelname2file[f'{mpnet_moniker}_{wt}s_concat5caps'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_concat5caps_WORDTYPE_embeddings_{wt}s.pkl"

            for max_n_words in range(10):
                modelname2file[f'{mpnet_moniker}_{wt}_max{max_n_words}words'] = f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_max{max_n_words}words_mean_WORDTYPE_embeddings_{wt}s.pkl"  # same as below, but want to catch both singular or plural in the short name
                modelname2file[f'{mpnet_moniker}_{wt}s_max{max_n_words}words'] = f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_max{max_n_words}words_mean_WORDTYPE_embeddings_{wt}s.pkl"
                modelname2file[f'{mpnet_moniker}_{wt}_max{max_n_words}words_concat5caps'] = f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_max{max_n_words}words_concat5caps_WORDTYPE_embeddings_{wt}s.pkl"  # same as below, but want to catch both singular or plural in the short name
                modelname2file[f'{mpnet_moniker}_{wt}s_max{max_n_words}words_concat5caps'] = f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_max{max_n_words}words_concat5caps_WORDTYPE_embeddings_{wt}s.pkl"
        

    # shuffled, and otherwise more complex models
    SENTENCE_EMBEDDING_MODEL_TYPES = ['mpnet', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
                                      'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
                                      'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
                                      'GUSE_transformer', 'GUSE_DAN', 'USE_CMLM_Base', 'T5']    
    RANDOMIZE_BY_WORD_TYPES = []  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
    for i in range(1, len(WORD_TYPES) + 1):
        RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])
    RANDOMIZE_BY_WORD_TYPES = [None] + RANDOMIZE_BY_WORD_TYPES  # add no randomization to the list

    for RANDOMIZE_WORD_ORDER in [False, True]:
        for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
            for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
                this_save_name = f"nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"
                this_short_name = this_save_name.replace("nsd_", "").replace("_mean_embeddings", "").replace("all_mpnet_base_v2", "mpnet")
                modelname2file[this_short_name] = f"{saved_embeddings_dir}/{this_save_name}.pkl"
                
                for cutoff in [0.5, 0.7]:
                    this_save_name_cutoff = this_save_name + f"_cutoffDist{cutoff}"
                    this_short_name_cutoff = this_short_name + f"_cutoffDist{cutoff}"
                    modelname2file[this_short_name_cutoff] = f"{saved_embeddings_dir}/{this_save_name_cutoff}.pkl"


    # add all the same keys and values for the other input data possible for the models
    # (i.e. models on the special100 with coco or gpt4-generated captions)
    additional_prefixes = ['nsd_special100_gpt4Captions', 'nsd_special100_cocoCaptions']
    dummy_dict = modelname2file.copy()
    for this_prefix in additional_prefixes:
        for k, v in dummy_dict.items():
            modelname2file[f"{k}_{this_prefix}"] = v.replace("nsd_", f"{this_prefix}_")

    return modelname2file
