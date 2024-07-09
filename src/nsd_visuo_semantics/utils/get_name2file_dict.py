'''Helper that returns a dictionary with the paths to the embeddings and DNN activities files'''

import itertools

def get_name2file_dict(saved_embeddings_dir, saved_dnn_activities_dir,
                       ecoset_saved_dnn_activities_dir):

    # specify where each set of nsd embeddings is saved
    modelname2file = {
        # basic models
        "mpnet": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings.pkl",
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
        "timm_nf_resnet50": f"{saved_dnn_activities_dir}/timm_nf_resnet50_nsd_image_features.pkl",
        "timm_efficientnet_b1": f"{saved_dnn_activities_dir}/timm_efficientnet_b1_nsd_image_features.pkl",
        "timm_hardcorenas_a": f"{saved_dnn_activities_dir}/timm_hardcorenas_a_nsd_image_features.pkl",
        "timm_xcit_nano_12_p8_224": f"{saved_dnn_activities_dir}/timm_xcit_nano_12_p8_224_nsd_image_features.pkl",
        "timm_levit_128": f"{saved_dnn_activities_dir}/timm_levit_128_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_nsd_image_features.pkl",  # these are with inputs in [0,255] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01_01inputs": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_01inputs_nsd_image_features.pkl",  # these are with inputs in [0,1] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x_01inputs": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_01inputs_nsd_image_features.pkl",
        "mpnetWordAvg_all": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_ALLWORDS_mean_embeddings.pkl",
        "mpnetWordAvg_nouns": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_NOUNS_mean_embeddings.pkl",
        "mpnetWordAvg_verbs": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_VERB_mean_embeddings.pkl",
        "taskonomy_scenecat_resnet50": f"{saved_dnn_activities_dir}/taskonomy_scenecat_resnet50_nsd_image_features.pkl",
        "guse": f"{saved_embeddings_dir}/nsd_guse_mean_embeddings.pkl",
        "all-mpnet-base-v2": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings.pkl",  # this is a duplicate of the line above, both names work
        "mpnet_resnet50_finalLayer": f"{saved_dnn_activities_dir}/mpnet_resnet50_finalLayer_nsd_image_features.pkl",
        "multihot_resnet50_finalLayer": f"{saved_dnn_activities_dir}/multihot_resnet50_finalLayer_nsd_image_features.pkl",
        "sceneCateg_resnet50_finalLayer": f"{saved_dnn_activities_dir}/sceneCateg_resnet50_finalLayer_nsd_image_features.pkl",
        
        # DNNs trained on ecoset activities
        "dnn_ecoset_category": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_category_post_gn_epoch80.h5",
        "dnn_ecoset_fasttext": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_fasttext_post_gn_epoch80.h5",
    }

    # DNN activities
    old_dnn_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets/semantics_paper_ms_coco_nets/extracted_activities'
    for epoch in [0, 100, 200]:
        for modelname in ["multihot_rec", "mpnet_rec"]: 
            modelname2file[f"dnn_{modelname}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_nsd_activations_epoch{epoch}.h5"
            for seed in range(1,11):
                modelname2file[f"dnn_{modelname}_seed{seed}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_seed{seed}_nsd_activations_epoch{epoch}.h5"

    # word types embeddings
    WORD_TYPES = ['noun', 'verb']

    # sentence embeddings on (lists of) words
    for mpnet_moniker in ["mpnet", "all-mpnet-base-v2"]:
        mpnet_full_name = "all-mpnet-base-v2"
        modelname2file[f"{mpnet_moniker}_category_all"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_CATEGORY_concatString_mean_embeddings_allCats.pkl"
        for wt in WORD_TYPES:
            modelname2file[f'{mpnet_moniker}_{wt}'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_WORDTYPE_embeddings_{wt}s.pkl"  # same as below, but want to catch both singular or plural in the short name
            modelname2file[f'{mpnet_moniker}_{wt}s'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_mean_WORDTYPE_embeddings_{wt}s.pkl"

    return modelname2file
