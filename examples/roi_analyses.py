import os, itertools
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses import nsd_roi_analyses
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses_figure import nsd_roi_analyses_figure
from nsd_visuo_semantics.get_embeddings.correlate_model_rdms_figure import correlate_model_rdms_figure

PAPER_FIG = 2  # do the models for paper figure 2 or 3

### DECLARE PARAMS
OVERWRITE = False
RCNN_LAYER = -1  # layer to use for the RCNNs
PLT_SUFFIX = f"_test"  # gets added after the figure we will make
MODEL_NAMES = []  # we will fill this in with the models we want.

MODELS_RDM_DIST = "correlation"  # RDM distance measure for models NOTE: SEARCHLIGHTS USE CORRELATION
WHICH_ROIS = "streams"  # streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...
DO_NOISE_CEILING = True  # if True, plot noise-ceiling corrected corrs. If false, do not use noise ceiling

if PAPER_FIG == 2:
    ### MODELS FOR PAPER FIGURE 2
    # models from original paper
    MODEL_NAMES += [
        "mpnet",
        "mpnet_category_all",
        # "multihot"
        # "fasttext_categories",
        # "mpnet_nouns",
        # "mpnet_verbs",
        # "mpnetWordAvg_all",
        # "fasttext_all",
        # "glove_all"
        ]
elif PAPER_FIG == 3:
    ### MODELS FOR PAPER FIGURE 3 (YOU WILL FIRST NEED TO EXTRACT THE DNN ACTIVATIONS USING get_dnn_activities.py)
    # dnn 10 seeds
    for target in ['mpnet', 'multihot']:
        for seed in range(1, 11):
            MODEL_NAMES += [f"dnn_{target}_rec_seed{seed}_ep200"]    
    MODEL_NAMES += ['thingsvision_cornet-s', 'dnn_ecoset_category', 'konkle_alexnetgn_supervised_ref12_augset1_5x', 
                    'timm_nf_resnet50', 'brainscore_resnet50_julios', 'brainscore_alexnet', 
                    'sceneCateg_resnet50_finalLayer', 'taskonomy_scenecat_resnet50',
                    'CLIP_RN50_images', 'resnext101_32x8d_wsl', 'CLIP_ViT_images',
                    'google_simclrv1_rn50', 'konkle_alexnetgn_ipcl_ref01']
else:
    raise ValueError(f"Unknown paper figure {PAPER_FIG}")

### PATHS
base_save_dir = "../results_dir"  # base dir from which to load model RDMs and in which to save results
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
saved_embeddings_dir = f"{base_save_dir}/saved_embeddings"
base_networks_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets'
# ms_coco_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ms_coco_nets/extracted_activities"
ms_coco_saved_dnn_activities_dir = f"/share/klab/adoerig/adoerig/nsd_visuo_semantics/examples/dnn_extracted_activities"
ecoset_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ecoset_nets/extracted_activities"
rdms_dir = f'{base_save_dir}/serialised_models_{MODELS_RDM_DIST}'
betas_dir = os.path.join(nsd_dir, '..', "NSD_for_visuo_semantics_derivatives", "betas")
rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')


### PREPARE RDMs FOR EACH REQUESTED MODEL

# nsd_prepare_modelrdms(MODEL_NAMES, MODELS_RDM_DIST,
#                       saved_embeddings_dir, rdms_dir, nsd_dir,
#                       ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
#                       OVERWRITE, RCNN_LAYER=RCNN_LAYER)

### RUN ROI ANALYSES

for i, m in enumerate(MODEL_NAMES):
    if 'dnn' in m:
        MODEL_NAMES[i] = f'{m}_layer{RCNN_LAYER}'

nsd_roi_analyses(MODEL_NAMES, MODELS_RDM_DIST, RCNN_LAYER, WHICH_ROIS,
                 nsd_dir, betas_dir, rois_dir, base_save_dir,
                 OVERWRITE_NEURO_RDMs=False, OVERWRITE_RDM_CORRs=OVERWRITE)

nsd_roi_analyses_figure(base_save_dir, WHICH_ROIS, MODELS_RDM_DIST, DO_NOISE_CEILING, 
                        custom_model_keys=MODEL_NAMES, plt_suffix=PLT_SUFFIX,
                        custom_model_labels=None, average_seeds=True,
                        plot_pval_tables=False)



