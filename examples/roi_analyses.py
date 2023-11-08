import os, itertools
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses import nsd_roi_analyses
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses_figure import nsd_roi_analyses_figure


### DECLARE PARAMS

OVERWRITE = False

MODEL_NAMES = []

# custom selection
MODEL_NAMES += ["mpnet", "glove_all", "mpnet_category_all",
                "mpnet_category_captionMatchPositive",
                "mpnet_category_captionMatchNegative"]

# ### BELOW IS CODE TO GET ALL MODELS
# # models from original paper
# MODEL_NAMES += [
#     "multihot",
#     "mpnet",
#     "fasttext_categories",
#     "fasttext_verbs",
#     "fasttext_all",
#     "guse",
#     "dnn_multihot_rec",
#     "dnn_mpnet_rec"
# ]

# # sentence embeddings with various randomized words
# SENTENCE_EMBEDDING_MODEL_TYPES = ['all_mpnet_base_v2']  #'all_mpnet_base_v2', 'USE_CMLM_Base', 'openai_ada2', 'GUSE_transformer',  'GUSE_DAN', 'T5'
# RANDOMIZE_WORD_ORDER = False  # If True, word order will be randomized in each sentence.
# WORD_TYPES = ['noun', 'verb', 'adjective', 'adverb', 'preposition']
# RANDOMIZE_BY_WORD_TYPES = []  # randomize within word type (e.g. use a random other verb instead of the sentence verb). Ignored if empty list.
# for i in range(1, len(WORD_TYPES) + 1):
#     RANDOMIZE_BY_WORD_TYPES.extend([list(elem) for elem in itertools.combinations(WORD_TYPES, i)])
# RANDOMIZE_BY_WORD_TYPES = [None] + RANDOMIZE_BY_WORD_TYPES  # add no randomization to the list

# for SENTENCE_EMBEDDING_MODEL_TYPE in SENTENCE_EMBEDDING_MODEL_TYPES:
#     for RANDOMIZE_BY_WORD_TYPE in RANDOMIZE_BY_WORD_TYPES:
#         this_save_name = f"nsd_{SENTENCE_EMBEDDING_MODEL_TYPE}_mean_embeddings{'_SCRAMBLED_WORD_ORDER' if RANDOMIZE_WORD_ORDER else ''}{'' if RANDOMIZE_BY_WORD_TYPE is None else '_RND_BY_' + '_'.join(RANDOMIZE_BY_WORD_TYPE)}"
#         MODEL_NAMES.append(this_save_name.replace("nsd_", "").replace("_mean_embeddings", "").replace("all_mpnet_base_v2", "mpnet"))

# # other new models
# MODEL_NAMES += [
#     "mpnet_category_all",
#     "mpnet_category_captionMatchPositive",
#     "mpnet_category_captionMatchNegative",
#     "glove_verbs",
#     "glove_all",
#     # "glove_nouns_closestCocoCats"
#     # "glove_nouns",
# ]
# MODEL_NAMES = list(set(MODEL_NAMES))

# if true, the 515 stimuli seen by all subjects are removed (so they can be used in the test set of other experiments
# based on searchlight maps while avoiding double-dipping)
remove_shared_515 = False

# RDM distance measure for models NOTE: BRAIN RDMS ARE DONE WITH CORRELATION DISTANCE
models_rdm_distance = "correlation"

# if we are using a DNN, use last layer (and last timestep if recurrent). If you want another layer,
# find its index (between in [0 ,n_layers*n_timesteps-1]) and apply it here. Ignored if "dnn_" not in MODEL_NAME
roi_analysis_dnn_layer_to_use = -1

which_rois = "streams"  # streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...

plot_noise_ceiling = True  # if True, plot noise-ceiling corrected corrs. If false, do not use noise ceiling

### PATHS
base_save_dir = "../results_dir"  # base dir from which to load model RDMs and in which to save results
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
saved_embeddings_dir = f"{base_save_dir}/saved_embeddings"
base_networks_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets'
ms_coco_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ms_coco_nets/extracted_activities"
ecoset_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ecoset_nets/extracted_activities"
rdms_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{models_rdm_distance}'
betas_dir = os.path.join(nsd_dir, '..', "NSD_for_visuo_semantics_derivatives", "betas")
rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')


# ### PREPARE RDMs FOR EACH REQUESTED MODEL

# nsd_prepare_modelrdms(MODEL_NAMES, models_rdm_distance,
#                       saved_embeddings_dir, rdms_dir, nsd_dir,
#                       ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
#                       remove_shared_515, OVERWRITE)


### RUN ROI ANALYSES

# nsd_roi_analyses(MODEL_NAMES, models_rdm_distance, roi_analysis_dnn_layer_to_use, which_rois,
#                  nsd_dir, betas_dir, rois_dir, base_save_dir,
#                  remove_shared_515, OVERWRITE_NEURO_RDMs=False, OVERWRITE_RDM_CORRs=True)

nsd_roi_analyses_figure(base_save_dir, which_rois, models_rdm_distance, plot_noise_ceiling, 
                        fig_id=0, custom_model_keys=MODEL_NAMES, plt_suffix="_mpnet_categs")
