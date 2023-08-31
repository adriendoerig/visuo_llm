"""[nsd_prepare_vnet_rdms]

    module to gather the subject's full model RDMs (MPNet, multihot, etc, model)
    Need quite some RAM, but no need for GPU.

    Run on bluebear:
    module load slurm-interactive
    fisbatch_screen --nodes 1-1 --ntasks 10 --mem=96G --time 240:0:0 --qos=bbdefault --account=charesti-start
    module load SciPy-bundle/2020.03-foss-2020a-Python-3.8.2
    module load TensorFlow/2.3.1-foss-2020a-Python-3.8.2
"""
import os
import pickle

import h5py
import numpy as np
from scipy.spatial.distance import pdist

from nsd_visuo_semantics.utils.nsd_get_conditions import (
    get_conditions,
    get_conditions_515,
)

# initialise parameters
n_sessions = 40
n_subjects = 8
subs = [f"subj0{x+1}" for x in range(n_subjects)]

# if true, the 515 stimuli seen by all subjects are removed (so they can be used in the test set of other experiments
# based on searchlight maps while avoiding double-dipping)
remove_shared_515 = False

# RDM distance measure NOTE: BRAIN RDMS ARE DONE WITH PEARSON CORR
rdm_distance = "correlation"

# set up directories
base_dir = os.path.join("/rds", "projects", "c", "charesti-start")
nsd_dir = os.path.join(base_dir, "data", "NSD")
base_save_dir = "./results_dir"
saved_embeddings_dir = f"{base_save_dir}/saved_embeddings"
ms_coco_saved_dnn_activities_dir = (
    f"{base_dir}/projects/NSD/paper_ms_coco_networks/extracted_activities"
)
ecoset_saved_dnn_activities_dir = (
    f"{base_dir}/projects/NSD/paper_ecoset_networks/extracted_activities"
)
rdms_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}'
os.makedirs(rdms_dir, exist_ok=True)

# specify where each set of nsd embeddings is saved
modelname2file = {
    "multihot": f"{saved_embeddings_dir}/nsd_multihot.pkl",
    "fasttext_categories": f"{saved_embeddings_dir}/NSD_fasttext_embeddings.npy",
    "fasttext_nouns": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_mean_embeddings.pkl",
    "nsd_fasttext_nouns_closest_cocoCats_cut0.33": f"{saved_embeddings_dir}/nsd_fasttext_NOUNS_mean_embeddings_closest_cocoCats_cut0.33.pkl",
    "fasttext_verbs": f"{saved_embeddings_dir}/nsd_fasttext_VERB_mean_embeddings.pkl",
    "fasttext_all": f"{saved_embeddings_dir}/nsd_fasttext_allWord_mean_embeddings.pkl",
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

for MODEL_NAME in [
    "fasttext_nouns",
    "nsd_fasttext_nouns_closest_cocoCats_cut0.33",
]:
    save_dir = os.path.join(rdms_dir, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # get embeddings from saved file. MUST BE 73000 images x n_embedding_features in NSD order.
    if modelname2file[MODEL_NAME][-4:] == ".pkl":
        with open(modelname2file[MODEL_NAME], "rb") as fp:  # Pickling
            embeddings = pickle.load(fp)
    elif modelname2file[MODEL_NAME][-4:] == ".npy":
        embeddings = np.load(modelname2file[MODEL_NAME], allow_pickle=True)
    elif "dnn" in MODEL_NAME:
        print(
            "You requested rdm for DNN activities, creating one rdm per layer & timestep"
        )
    else:
        raise Exception(
            f"Embeddings file type not understood. "
            f"Found: {modelname2file[MODEL_NAME]}. Please sue .pkl or.npy."
        )

    # loop over subjects
    for sub in subs:
        # extract conditions data (see nsd_searchlight_main_tf.py for a detailed explanation of how this works)
        conditions = get_conditions(nsd_dir, sub, n_sessions)
        # we also need to reshape conditions to be ntrials x 1
        conditions = np.asarray(conditions).ravel()
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions
        ]
        if remove_shared_515:
            conditions_515 = get_conditions_515(
                nsd_dir
            )  # [515,]  (nsd_indices for the 515 shared images)
            conditions_515_bool = [
                True if x in conditions_515 else False for x in conditions
            ]  # [n_subj_stims,] boolean array with True if this idx is a 515 shared img
            conditions_bool = [
                True if x and not y else False
                for x, y in zip(conditions_bool, conditions_515_bool)
            ]  # [n_subj_stims-515,] array of nsd_indices
        conditions_sampled = conditions[conditions_bool]
        # find the subject's condition list (sample pool)
        sample = np.unique(conditions[conditions_bool])

        if "dnn" in MODEL_NAME:
            with h5py.File(
                modelname2file[MODEL_NAME], "r"
            ) as activations_file:
                layer_names = [x for x in activations_file.keys()]
                for layer_name in layer_names:
                    save_name = os.path.join(
                        save_dir,
                        f"{sub}_{MODEL_NAME}_{layer_name}_fullrdm.npy",
                    )
                    if os.path.exists(save_name):
                        print(f"Found file at {save_name}. Skipping...")
                    else:
                        print(f"Creating {MODEL_NAME} rdm for {sub}")
                        this_embedding = activations_file[layer_name][
                            sample - 1, :
                        ]  # 10'000xn_features (other subjects have fewer images) - NOTE: from NSD's 1-based indexing pipeline, so we move back to 0-based
                        this_rdm = pdist(this_embedding, rdm_distance).astype(
                            np.float32
                        )  # subject based RDM for 10000 items
                        print(f"Saving in {save_name}")
                        np.save(save_name, this_rdm)

        else:
            save_name = os.path.join(
                save_dir, f"{sub}_{MODEL_NAME}_fullrdm.npy"
            )
            if os.path.exists(save_name):
                print(f"Found file at {save_name}. Skipping...")
            else:
                print(f"Creating {MODEL_NAME} rdm for {sub}")
                this_embedding = embeddings[
                    sample - 1, :
                ]  # 10'000xn_features (other subjects have fewer images) - NOTE: from NSD's 1-based indexing pipeline, so we move back to 0-based
                this_rdm = pdist(this_embedding, rdm_distance).astype(
                    np.float32
                )  # subject based RDM for 10000 items
                print(f"Saving in {save_name}")
                np.save(save_name, this_rdm)
