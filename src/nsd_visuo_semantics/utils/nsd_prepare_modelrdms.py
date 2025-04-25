"""
    module to gather the full model RDMs for different models (MPNet, multihot, DNNs, etc) 
    correspoding to each subject's images.
    Need quite some RAM, but no need for GPU.
"""
import os, pickle, h5py, re
import numpy as np
from scipy.spatial.distance import pdist
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions
from nsd_visuo_semantics.utils.get_name2file_dict import get_name2file_dict

def nsd_prepare_modelrdms(MODEL_NAMES, rdm_distance,
                               saved_embeddings_dir, rdms_dir, nsd_dir,
                               ms_coco_saved_dnn_activities_dir, ecoset_saved_dnn_activities_dir, 
                               OVERWRITE, RCNN_LAYER=None):
    
    if not isinstance(MODEL_NAMES, list):
        MODEL_NAMES = [MODEL_NAMES]

    # initialise parameters
    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]

    # specify where each set of nsd embeddings is saved
    modelname2file = get_name2file_dict(saved_embeddings_dir,
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
            if RCNN_LAYER is not None:
                print(f"You requested rdm for DNN activities, and layer {RCNN_LAYER} only")
            else:
                print("You requested rdm for DNN activities, creating one rdm per layer & timestep")
        else:
            raise Exception(f"Embeddings file type not understood. "
                            f"Found: {modelname2file[MODEL_NAME]}. Please use .pkl or.npy.")
        
        # loop over subjects
        for sub in subs:

            # extract conditions data (see nsd_searchlight_main_tf.py for a detailed explanation of how this works)
            conditions = get_conditions(nsd_dir, sub, n_sessions)
            # we also need to reshape conditions to be ntrials x 1
            conditions = np.asarray(conditions).ravel()
            # then we find the valid trials for which we do have 3 repetitions.
            conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
            conditions_sampled = conditions[conditions_bool]
            # find the subject's condition list (sample pool)
            sample = np.unique(conditions_sampled)

            if "dnn" in MODEL_NAME:
                with h5py.File(modelname2file[MODEL_NAME], "r") as activations_file:
                    if RCNN_LAYER is not None:
                        if RCNN_LAYER == -1:
                            RCNN_LAYER = len(activations_file.keys()) - 1
                        l, t = RCNN_LAYER // 6, RCNN_LAYER % 6
                        if 'ecoset' in MODEL_NAME:
                            layer_names = [f'groupnorm_layer_{l}_time_{t}']
                        else:
                            layer_names = [f'layernorm_layer_{l}_time_{t}']
                    else:
                        # do all layers
                        layer_names = [x for x in activations_file.keys()]

                    for layer_name in layer_names:
                        save_name = os.path.join(save_dir, f"{sub}_{MODEL_NAME}_{layer_name}_fullrdm.npy")
                        if os.path.exists(save_name) and not OVERWRITE:
                            print(f"Found file at {save_name}. Skipping...")
                        else:
                            print(f"Creating {MODEL_NAME} {layer_name} rdm for {sub}")
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
