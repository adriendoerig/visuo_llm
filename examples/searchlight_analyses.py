import os
from nsd_visuo_semantics.utils.nsd_prepare_modelrdms import nsd_prepare_modelrdms
from nsd_visuo_semantics.searchlight_analyses.nsd_searchlight_main_tf import nsd_searchlight_main_tf
from nsd_visuo_semantics.searchlight_analyses.nsd_project_fsaverage import nsd_project_fsaverage


'''run with gpu on osna hpc:
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX 
export TMPDIR='/share/klab/adoerig'
conda activate tensorflowGPU'''


### DECLARE PARAMS

OVERWRITE = False

# models to test
MODEL_NAMES = [
    # "mpnet",
    # "multihot",``
    # "fasttext_categories",
    # "fasttext_verbs",
    # "fasttext_all",
    # "guse",    
    # "dnn_mpnet_rec_seed1_ep200",
    # "dnn_multihot_rec_seed1_ep200",
    # "dnn_multihot_rec_old_ep200",
    # "dnn_mpnet_rec_old_ep200",
    # "dnn_multihot_rec_seed1_ep200"
]

# MODEL_NAMES += [f"dnn_mpnet_rec_seed{s}_ep200" for s in [10]]
MODEL_NAMES += [f"dnn_multihot_rec_seed{s}_ep200" for s in [9,10]]

# if true, the 515 stimuli seen by all subjects are removed (so they can be used in the test set of other experiments
# based on searchlight maps while avoiding double-dipping)
remove_shared_515 = False

# RDM distance measure for models NOTE: BRAIN RDMS ARE DONE WITH CORRELATION DISTANCE
models_rdm_distance = "correlation"

### PATHS
base_save_dir = "../results_dir"  # base dir from which to load model RDMs and in which to save results
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = f"{nsd_derivatives_dir}/betas"
precompsl_dir = f"{nsd_derivatives_dir}/searchlights"

saved_embeddings_dir = f"{base_save_dir}/saved_embeddings"
base_networks_dir = '/share/klab/adoerig/adoerig/semantics_paper_nets'
ms_coco_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ms_coco_nets/extracted_activities"
ecoset_saved_dnn_activities_dir = f"{base_networks_dir}/semantics_paper_ecoset_nets/extracted_activities"
rdms_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{models_rdm_distance}'

### PREPARE RDMs FOR EACH REQUESTED MODEL
# nsd_prepare_modelrdms(MODEL_NAMES, models_rdm_distance,
#                       saved_embeddings_dir, rdms_dir, nsd_dir,
#                       ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir,
#                       remove_shared_515, OVERWRITE)


### RUN SEARCHLIGHT
nsd_searchlight_main_tf(MODEL_NAMES, models_rdm_distance, 
                        nsd_dir, nsd_derivatives_dir, betas_dir, base_save_dir, 
                        remove_shared_515, OVERWRITE)


### PROJECT SEARCHLIGHT MAPS TO FSAVERAGE
nsd_project_fsaverage(MODEL_NAMES, models_rdm_distance, 
                      nsd_dir, base_save_dir, 
                      remove_shared_515)