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
        "fasttext_nouns": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_embeddings.pkl",
        "fasttext_verbs": f"{saved_embeddings_dir}/nsd_fasttext_VERBS_embeddings.pkl",
        "fasttext_all": f"{saved_embeddings_dir}/nsd_fasttext_ALLWORDS_embeddings.pkl",
        "glove_categories": f"{saved_embeddings_dir}/nsd_glove_CATEGORY_mean_embeddings.pkl",
        "glove_nouns": f"{saved_embeddings_dir}/nsd_glove_NOUNS_embeddings.pkl",
        "glove_verbs": f"{saved_embeddings_dir}/nsd_glove_VERBS_embeddings.pkl",
        "glove_all": f"{saved_embeddings_dir}/nsd_glove_ALLWORDS_embeddings.pkl",
        "CLIP_ViT_text": f"{saved_embeddings_dir}/nsd_CLIP-vit_mean_embeddings.pkl",
        "CLIP_ViT_images": f"{saved_dnn_activities_dir}/CLIP-vit_nsd_image_features.pkl",
        "CLIP_RN50_text": f"{saved_embeddings_dir}/nsd_CLIP-rn50_mean_embeddings.pkl",
        "CLIP_RN50_images": f"{saved_dnn_activities_dir}/CLIP-rn50_nsd_image_features.pkl",
        "thingsvision_cornet-s": f"{saved_dnn_activities_dir}/thingsvision_cornet-s_nsd_image_features.pkl",
        "brainscore_alexnet": f"{saved_dnn_activities_dir}/brainscore_alexnet_nsd_image_features.pkl",
        "brainscore_resnet50_julios": f"{saved_dnn_activities_dir}/brainscore_resnet50_julios_nsd_image_features.pkl",
        "resnext101_32x8d_wsl": f"{saved_dnn_activities_dir}/resnext101_32x8d_wsl_nsd_image_features.pkl",
        "google_simclrv1_rn50": f"{saved_dnn_activities_dir}/google_simclrv1_rn50_nsd_image_features.pkl",
        "timm_nf_resnet50": f"{saved_dnn_activities_dir}/timm_nf_resnet50_nsd_image_features.pkl",
        "konkle_alexnetgn_ipcl_ref01": f"{saved_dnn_activities_dir}/konkle_alexnetgn_ipcl_ref01_nsd_image_features.pkl",  # these are with inputs in [0,255] before the transform (I was not sure which to use)
        "konkle_alexnetgn_supervised_ref12_augset1_5x": f"{saved_dnn_activities_dir}/konkle_alexnetgn_supervised_ref12_augset1_5x_nsd_image_features.pkl",
        "mpnetWordAvg_all": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_ALLWORDS_embeddings.pkl",
        "taskonomy_scenecat_resnet50": f"{saved_dnn_activities_dir}/taskonomy_scenecat_resnet50_nsd_image_features.pkl",
        "guse": f"{saved_embeddings_dir}/nsd_guse_mean_embeddings.pkl",
        "all-mpnet-base-v2": f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings.pkl",  # this is a duplicate of the line above, both names work
        "mpnet_resnet50_finalLayer": f"{saved_dnn_activities_dir}/mpnet_resnet50_finalLayer_nsd_image_features.pkl",
        "multihot_resnet50_finalLayer": f"{saved_dnn_activities_dir}/multihot_resnet50_finalLayer_nsd_image_features.pkl",
        "sceneCateg_resnet50_finalLayer": f"{saved_dnn_activities_dir}/sceneCateg_resnet50_finalLayer_nsd_image_features.pkl",
        'mpnet_scrambled': f"{saved_embeddings_dir}/nsd_all-mpnet-base-v2_mean_embeddings_scrambled.pkl",

        # DNNs trained on ecoset activities
        "dnn_ecoset_category": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_category_post_gn_epoch80.h5",
        "dnn_ecoset_fasttext": f"{ecoset_saved_dnn_activities_dir}/blt_vnet_fasttext_post_gn_epoch80.h5",
    }

    # DNN activities
    for epoch in [0, 100, 200]:
        for modelname in ["multihot_rec", "mpnet_rec"]: 
            modelname2file[f"dnn_{modelname}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_nsd_activations_epoch{epoch}.h5"
            for seed in range(1,11):
                modelname2file[f"dnn_{modelname}_seed{seed}_ep{epoch}"] = f"{saved_dnn_activities_dir}/{modelname}_seed{seed}_nsd_activations_epoch{epoch}.h5"

    # word types embeddings
    WORD_TYPES = ['nouns', 'verbs', 'noun', 'verb', 'prepositions', 'adjectives', 'adverbs']

    # sentence embeddings on (lists of) words
    for mpnet_moniker in ["mpnet", "all-mpnet-base-v2"]:
        mpnet_full_name = "all-mpnet-base-v2"
        modelname2file[f"{mpnet_moniker}_category_all"] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_CATEGORY_concatString_mean_embeddings_allCats.pkl"
        for wt in WORD_TYPES:
            modelname2file[f'{mpnet_moniker}_{wt}'] = f"{saved_embeddings_dir}/nsd_{mpnet_full_name}_{wt.upper()}_embeddings.pkl"

    SENTENCE_EMBEDDING_MODEL_TYPES = ['multi-qa-mpnet-base-dot-v1', 'all-distilroberta-v1', 'all-MiniLM-L12-v2', 
                                      'paraphrase-multilingual-mpnet-base-v2', 'paraphrase-albert-small-v2', 
                                      'paraphrase-MiniLM-L3-v2', 'distiluse-base-multilingual-cased-v2',
                                      'GUSE_transformer', 'GUSE_DAN', 'USE_CMLM_Base', 'T5']
    for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
        modelname2file[SENTENCE_EMBEDDING_MODEL_TYPE] = f"{saved_embeddings_dir}/nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings.pkl"

    return modelname2file
