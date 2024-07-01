import os, itertools
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses import nsd_roi_analyses
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses_figure import nsd_roi_analyses_figure
from nsd_visuo_semantics.roi_analyses.nsd_roi_analyses_dnnAllLayers_figure import nsd_roi_analyses_dnnAllLayers_figure
from nsd_visuo_semantics.get_embeddings.correlate_model_rdms_figure import correlate_model_rdms_figure
from nsd_visuo_semantics.utils.get_name2file_dict import get_name2file_dict


### DECLARE PARAMS

plt_suffix = f"_bihemNets"

OVERWRITE = False

MODEL_AND_LAYER_NAMES = [
    'dnn_bihem_singleStreamApr24_Layer_gap',
    'dnn_bihem_bottleneckNoCCApr24_Layer_gap_left',
    'dnn_bihem_bottleneckCCApr24_Layer_gap_left',
    'dnn_bihem_bottleneckBlurNoDelayCCApr24_Layer_gap_left',
    'dnn_bihem_bottleneckBlurLDelayCCApr24_Layer_gap_left',
                ]

MODEL_NAMES = [m.split("_Layer")[0] for m in MODEL_AND_LAYER_NAMES]

MODEL_LAYER_NAMES = {
    'dnn_bihem_singleStreamApr24': 'gap',
    'dnn_bihem_bottleneckNoCCApr24': 'gap_left',
    'dnn_bihem_bottleneckCCApr24': 'gap_left',
    'dnn_bihem_bottleneckBlurNoDelayCCApr24': 'gap_left',
    'dnn_bihem_bottleneckBlurLDelayCCApr24': 'gap_left',
}

import pdb; pdb.set_trace()

# BELOW IS UNFINISHED CODE
# BELOW IS UNFINISHED CODE
# BELOW IS UNFINISHED CODE
# BELOW IS UNFINISHED CODE
# BELOW IS UNFINISHED CODE







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

nsd_prepare_modelrdms(MODEL_NAMES, models_rdm_distance,
                        saved_embeddings_dir, rdms_dir, nsd_dir,
                        ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
                        remove_shared_515, OVERWRITE)

# ### RUN ROI ANALYSES

for i, m in enumerate(MODEL_NAMES):
    if 'dnn' in m:
        MODEL_NAMES[i] = f'{m}_layer{roi_analysis_dnn_layer_to_use}'

nsd_roi_analyses(MODEL_NAMES, models_rdm_distance, roi_analysis_dnn_layer_to_use, which_rois,
                nsd_dir, betas_dir, rois_dir, base_save_dir,
                remove_shared_515, OVERWRITE_NEURO_RDMs=False, OVERWRITE_RDM_CORRs=OVERWRITE)

nsd_roi_analyses_figure(base_save_dir, which_rois, models_rdm_distance, plot_noise_ceiling, 
                        fig_id=0, custom_model_keys=MODEL_NAMES, plt_suffix=plt_suffix,
                        custom_model_labels=None, average_seeds=True)

# correlate_model_rdms_figure(MODEL_NAMES, nsd_dir, base_save_dir, models_rdm_distance, 
#                             remove_shared_515, roi_analysis_dnn_layer_to_use, 
#                             plt_suffix=plt_suffix, 
#                             COMPUTE=True, OVERWRITE=OVERWRITE)

# nsd_roi_analyses_dnnAllLayers_figure(base_save_dir, which_rois, models_rdm_distance)