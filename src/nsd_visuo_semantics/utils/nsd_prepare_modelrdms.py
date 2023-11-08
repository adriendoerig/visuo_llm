"""
    module to gather the full model RDMs for different models (MPNet, multihot, DNNs, etc) 
    correspoding to each subject's images.
    Need quite some RAM, but no need for GPU.

"""
import os
import pickle
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515
from nsd_visuo_semantics.utils.model_name2file_list import get_name2file_list

def nsd_prepare_modelrdms(MODEL_NAMES, rdm_distance,
                          saved_embeddings_dir, rdms_dir, nsd_dir,
                          ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir, 
                          remove_shared_515, OVERWRITE):

    # initialise parameters
    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]

    # specify where each set of nsd embeddings is saved
    modelname2file = get_name2file_list(saved_embeddings_dir,
                                        ms_coco_saved_dnn_activities_dir, 
                                        ecoset_saved_dnn_activities_dir)

    for MODEL_NAME in MODEL_NAMES:
        
        save_dir = os.path.join(rdms_dir, MODEL_NAME)
        os.makedirs(save_dir, exist_ok=True)

        # get embeddings from saved file. MUST BE 73000 images x n_embedding_features in NSD order.
        if modelname2file[MODEL_NAME][-4:] == ".pkl":
            with open(modelname2file[MODEL_NAME], "rb") as fp:  # Pickling
                embeddings = pickle.load(fp)
        elif modelname2file[MODEL_NAME][-4:] == ".npy":
            embeddings = np.load(modelname2file[MODEL_NAME], allow_pickle=True)
        elif "dnn" in MODEL_NAME:
            print("You requested rdm for DNN activities, creating one rdm per layer & timestep")
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
            conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
            if remove_shared_515:
                conditions_515 = get_conditions_515(nsd_dir)  # [515,]  (nsd_indices for the 515 shared images)
                conditions_515_bool = [True if x in conditions_515 else False for x in conditions]  # [n_subj_stims,] boolean array with True if this idx is a 515 shared img
                conditions_bool = [True if x and not y else False for x, y in zip(conditions_bool, conditions_515_bool)]  # [n_subj_stims-515,] array of nsd_indices
            conditions_sampled = conditions[conditions_bool]
            # find the subject's condition list (sample pool)
            sample = np.unique(conditions[conditions_bool])

            if "dnn" in MODEL_NAME:
                with h5py.File(modelname2file[MODEL_NAME], "r") as activations_file:
                    layer_names = [x for x in activations_file.keys()]
                    for layer_name in layer_names:
                        save_name = os.path.join(save_dir, f"{sub}_{MODEL_NAME}_{layer_name}_fullrdm.npy")
                        if os.path.exists(save_name) and not OVERWRITE:
                            print(f"Found file at {save_name}. Skipping...")
                        else:
                            print(f"Creating {MODEL_NAME} rdm for {sub}")
                            this_embedding = activations_file[layer_name][sample-1, :]  # 10'000xn_features (other subjects have fewer images) - NOTE: from NSD's 1-based indexing pipeline, so we move back to 0-based
                            this_rdm = pdist(this_embedding, rdm_distance).astype(np.float32)  # subject based RDM for 10000 items
                            print(f"Saving in {save_name}")
                            np.save(save_name, this_rdm)

            else:
                save_name = os.path.join(save_dir, f"{sub}_{MODEL_NAME}_fullrdm.npy")
                if os.path.exists(save_name) and not OVERWRITE:
                    print(f"Found file at {save_name}. Skipping...")
                else:
                    print(f"Creating {MODEL_NAME} rdm for {sub}")
                    this_embedding = embeddings[sample-1, :]  # 10'000xn_features (other subjects have fewer images) - NOTE: from NSD's 1-based indexing pipeline, so we move back to 0-based
                    this_rdm = pdist(this_embedding, rdm_distance).astype(np.float32)  # subject based RDM for 10000 items
                    print(f"Saving in {save_name}")
                    np.save(save_name, this_rdm)
