import os
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.searchlight_analyses.nsd_searchlight_main_tf import nsd_searchlight_main_tf
from nsd_visuo_semantics.searchlight_analyses.nsd_project_fsaverage import nsd_project_fsaverage
from nsd_visuo_semantics.utils.py_plot_brain_utils import pyplot_brains_from_models_list
### DECLARE PARAMS
OVERWRITE = False
MODELS_RDM_DIST = "correlation"  # RDM distance measure for models NOTE: BRAIN RDMS ARE DONE WITH CORRELATION DISTANCE

# models to test
MODEL_NAMES = [
    "test1k_gs_lstm_n_1024_tm_3_loc_1_reg_1_indp_0.25_rnndp_0.25_gaze_dg3_gcpc_1_semc_0_scc_0_locmse_0_tr_train_plus_lr_0.0001_num_1",
    "test1k_gs_lstm_n_256_tm_5_loc_1_reg_1_indp_0.25_rnndp_0.1_gaze_dg3_gcpc_0_semc_1_scc_0_locmse_0_tr_train_plus_lr_0.0001_num_1",
]


### PATHS
base_save_dir = "../results_dir"  # base dir from which to load model RDMs and in which to save results
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = f"{nsd_derivatives_dir}/betas"
precompsl_dir = f"{nsd_derivatives_dir}/searchlights"
figures_dir = f"{base_save_dir}/searchlight_respectedsampling_correlation/figures"

saved_embeddings_dir = '/share/klab/sthorat/sthorat/gaze-stitch/gs_analyse/save_actvs'
base_networks_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets'
ms_coco_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ms_coco_nets/extracted_activities"
ecoset_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ecoset_nets/extracted_activities"
rdms_dir = f'{base_save_dir}/serialised_models_{MODELS_RDM_DIST}'

# ### PREPARE RDMs FOR EACH REQUESTED MODEL
nsd_prepare_modelrdms(MODEL_NAMES, MODELS_RDM_DIST,
                      saved_embeddings_dir, rdms_dir, nsd_dir,
                      ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
                      OVERWRITE)


# ### RUN SEARCHLIGHT
nsd_searchlight_main_tf(MODEL_NAMES, MODELS_RDM_DIST, 
                        nsd_dir, nsd_derivatives_dir, betas_dir, base_save_dir, 
                        OVERWRITE)


# ### PROJECT SEARCHLIGHT MAPS TO FSAVERAGE
nsd_project_fsaverage(MODEL_NAMES, MODELS_RDM_DIST, nsd_dir, base_save_dir)

### PLOT SEARCHLIGHT MAPS
pyplot_brains_from_models_list(MODEL_NAMES, MODEL_NAMES, f"{base_save_dir}/searchlight_respectedsampling_correlation",
                               layer='last', contrast_layer='same', 
                               contrast_same_model=False, max_cmap_val=None,
                               save_type='png', figpath=figures_dir, 
                               plot_indiv_sub=True, plot_subj_avg=True)