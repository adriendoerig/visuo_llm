"""Train a frac ridge regression between NSD voxels and embeddings.
"""

import os, pickle, time, h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressorCV
from nsd_access import NSDAccess
from scipy.spatial.distance import cdist, correlation, cosine, pdist
from nsd_visuo_semantics.decoding_analyses.decoding_utils import remove_inert_embedding_dims, restore_inert_embedding_dims, restore_nan_dims, pairwise_corr
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_rois,get_sentence_lists, load_or_compute_betas_average
from sklearnex import patch_sklearn
patch_sklearn()



np.random.seed(0)

ENCODING_MODEL_NAME = "all-mpnet-base-v2"

IS_DNN = False  # if true, we expect to be using DNN activities instead of e.g. mpnet embedding (useful for getting DNN layer encoding models, etc)
DNN_LAYER =  ''  # -1  # dnn layer to use. only used if IS_DNN is True

USE_ROIS = None  # "mpnet_noShared515_sig0.005_fsaverage"  # None, or 'mpnet_noShared515_sig0.005_fsaverage' or streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...

SPLIT_DATA = 'full'  # 'halves' or 'full'  # if 'halves', we train two encoders: one on the first half, and the other on the second half of the training set
if SPLIT_DATA == 'halves':
    SPLITS = [0,1]
else:
    SPLITS = [0]

SHUFFLE_DATA = 'voxelWisePredictorShuffle'
SHUFFLE_DATA_SUFFIX = f'_{SHUFFLE_DATA}' if SHUFFLE_DATA else ''
# setup
total_time = time.time()
n_jobs = 4

# params from nsd
n_sessions = 40
n_subjects = 8
subs = [f"subj0{x + 1}" for x in range(n_subjects)]
targetspace = "fsaverage"

# fractional ridge regression parameters
n_alphas = 20
fracs = np.linspace(1/n_alphas, 1 + 1/n_alphas, n_alphas)  # from https://github.com/nrdg/fracridge/blob/master/examples/plot_alpha_vs_gamma.py

# set up directories
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses"
os.makedirs(base_save_dir, exist_ok=True)
nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
os.makedirs(nsd_embeddings_path, exist_ok=True)
nsd_dnn_activities_path = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/examples/dnn_extracted_activities"

nsda = NSDAccess(nsd_dir)

# get the condition list for the special 515
# these will be used as testing set for the guse predictions
conditions_515 = get_conditions_515(nsd_dir)
images_515 = nsda.read_images(np.asarray(conditions_515) - 1)
sentences_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)

# prepare the test set embeddings
if not IS_DNN:
    embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings.npy"
    if not os.path.exists(embeddings_test_path):
        embedding_model = get_embedding_model(ENCODING_MODEL_NAME)
        captions_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)
        dummy_embedding = get_embeddings(captions_515[0], embedding_model, ENCODING_MODEL_NAME)
        embedding_dim = dummy_embedding.shape[-1]
        embeddings_test = np.empty((515, embedding_dim))
        for i in range(len(captions_515)):
            embeddings_test[i] = np.mean(get_embeddings(captions_515[i], embedding_model, ENCODING_MODEL_NAME), axis=0)
        np.save(embeddings_test_path, embeddings_test)
    else:
        embeddings_test = np.load(embeddings_test_path)
        embedding_dim = embeddings_test.shape[-1]
else:
    embeddings_test_path = f"{nsd_dnn_activities_path}/{ENCODING_MODEL_NAME}.h5"
    with h5py.File(embeddings_test_path, "r") as f:
        layer_names = [x for x in f.keys()]
        embeddings_test = f[layer_names[DNN_LAYER]][np.asarray(conditions_515)-1, :]
    embedding_dim = embeddings_test.shape[-1]

# ROI STUFF
if USE_ROIS is not None:
    maskdata, roi_id2name = get_rois(USE_ROIS, "./save_dir/roi_analyses/roi_defs")
    roi_name2id = {v: k for k, v in roi_id2name.items()}
    ROI_NAMES = list(roi_name2id.keys())
    ROI_NAMES.remove("Unknown")
    print(f"Using ROI_NAMES = {ROI_NAMES}")
else:
    ROI_NAMES = ["fullbrain"]


for SPLIT in SPLITS:

    if SPLIT_DATA == 'halves':
        SPLIT_SUFFIX= f"_split{SPLIT}"
    else:
        SPLIT_SUFFIX = ''

    for USE_N_STIMULI in [1000]:  # None means use all stimuli
        for ROI_NAME in ROI_NAMES:
            if IS_DNN:
                this_results_dir = os.path.join(
                    base_save_dir,
                    f'{ENCODING_MODEL_NAME}_layer{DNN_LAYER}_results_ROI{ROI_NAME}{"" if USE_N_STIMULI is None else f"_{USE_N_STIMULI}stims"}_encodingModel{SPLIT_SUFFIX}{SHUFFLE_DATA_SUFFIX}',
                )
            else:
                this_results_dir = os.path.join(
                    base_save_dir,
                    f'{ENCODING_MODEL_NAME}_results_ROI{ROI_NAME}{"" if USE_N_STIMULI is None else f"_{USE_N_STIMULI}stims"}_encodingModel{SPLIT_SUFFIX}{SHUFFLE_DATA_SUFFIX}',
                )

            os.makedirs(this_results_dir, exist_ok=True)
            fitted_models_dir = os.path.join(this_results_dir, "fitted_models")
            os.makedirs(fitted_models_dir, exist_ok=True)

            for s_n, subj in enumerate(subs):
                # prepare the train/val set embeddings

                # find indices that are NOT in the 515 spacial test images
                # extract conditions data
                conditions = get_conditions(nsd_dir, subj, n_sessions)
                # we also need to reshape conditions to be ntrials x 1
                conditions = np.asarray(conditions).ravel()
                # then we find the valid trials for which we do have 3 repetitions.
                conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
                # and identify those.
                conditions_sampled = conditions[conditions_bool]
                # find the subject's condition list (sample pool)
                # this sample is the same order as the betas
                sample = np.unique(conditions[conditions_bool])

                # identify which image in the sample is a conditions_515
                sample_515_bool = [True if x in conditions_515 else False for x in sample]
                # and identify which sample image isn't in conditions_515
                sample_train_bool = [False if x in conditions_515 else True for x in sample]
                # also identify the training set (i.e. not the special 515)
                sample_train = sample[sample_train_bool]

                # get the guse embeddings for the training sample
                if not IS_DNN:
                    embeddings_train_path = (f"{nsd_embeddings_path}/captions_not515_embeddings_{subj}.npy")
                    if not os.path.exists(embeddings_train_path):
                        embedding_model = get_embedding_model(ENCODING_MODEL_NAME)
                        captions_not515 = get_sentence_lists(nsda, sample_train - 1)
                        embeddings_train = np.empty((len(captions_not515), embedding_dim))
                        for i in range(len(captions_not515)):
                            embeddings_train[i] = np.mean(
                                get_embeddings(
                                    captions_not515[i],
                                    embedding_model,
                                    ENCODING_MODEL_NAME,
                                ),
                                axis=0,
                            )
                        np.save(embeddings_train_path, embeddings_train)
                    else:
                        embeddings_train = np.load(embeddings_train_path)
                else:
                    embeddings_train_path = f"{nsd_dnn_activities_path}/{ENCODING_MODEL_NAME}.h5"
                    with h5py.File(embeddings_train_path, "r") as f:
                        layer_names = [x for x in f.keys()]
                        embeddings_train = f[layer_names[DNN_LAYER]][np.asarray(sample_train)-1, :]

                # Betas per subject
                print(f"loading betas for {subj}")
                betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
                betas_mean = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)

                if USE_ROIS is not None:
                    # load the lh mask
                    orig_n_voxels = betas_mean.shape[0]
                    vs_mask = maskdata == roi_name2id[ROI_NAME]
                    betas_mean = betas_mean[vs_mask, :]
                    print(f"Applied ROI mask. Went from {orig_n_voxels} to {betas_mean.shape[0]}")

                good_vertex = [True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]
                if np.sum(good_vertex) != len(good_vertex):
                    print(f"found some NaN for {subj}")
                betas_mean = betas_mean[good_vertex, :]

                # now we further split the brain data according to the 515 test set or the training set for that subject
                betas_test = betas_mean[:, sample_515_bool].T  # sub1: (515, 327673) (n_voxels may vary from subj to subj because of nans)
                betas_train = betas_mean[:, sample_train_bool].T  # sub1: (9485, 327673) (may vary from subj to subj)
                
                del betas_mean  # make space

                # format to float32,
                embeddings_train, embeddings_test, betas_train, betas_test = (
                    embeddings_train.astype(np.float32),
                    embeddings_test.astype(np.float32),
                    betas_train.astype(np.float32),
                    betas_test.astype(np.float32),
                )


                if SPLIT_DATA == 'halves':
                    if SPLIT == 0:
                        embeddings_train = embeddings_train[:embeddings_train.shape[0]//2]
                        betas_train = betas_train[:betas_train.shape[0]//2]
                    else:
                        embeddings_train = embeddings_train[embeddings_train.shape[0]//2:]
                        betas_train = betas_train[betas_train.shape[0]//2:]

                
                # In this case, we use embeddings as an encoding model to linearly predict voxels
                corrs_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingCorrMap_{ROI_NAME}_{ENCODING_MODEL_NAME}.npy"
                coefs_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingCoefs_{ROI_NAME}_{ENCODING_MODEL_NAME}.npy"

                if not os.path.exists(corrs_save_path):

                    # Initialize an empty array to store coefficients
                    coeffs_list = []
                    preds_list = []
                    corrs_list = []

                    # Loop over each column of betas_train
                    for i in range(betas_train.shape[1]):

                        if i % 100 == 0:
                            print(f"Processing voxel {i} of {betas_train.shape[1]} for {subj}", end="\r")

                        # Extract the ith column of betas_train
                        y = betas_train[:, i]
                        y_test = betas_test[:, i]
                        
                        # Fit linear regression model
                        frr = FracRidgeRegressorCV(fit_intercept=True, jit=True)
                        fitted_fracridge = frr.fit(
                            embeddings_train[:USE_N_STIMULI],
                            y[:USE_N_STIMULI],
                            frac_grid=fracs,
                        )
                        
                        coeff = frr.coef_  # [n_embed_dims,]
                        coeffs_list.append(coeff)

                        test_preds = fitted_fracridge.predict(embeddings_test)  # [n_test_items,]
                        preds_list.append(test_preds)

                        fitted_test_corrs = pairwise_corr(test_preds, y_test)  # [,]
                        corrs_list.append(fitted_test_corrs)

                    coeffs = np.hstack([c[:,None] for c  in coeffs_list])  # [n_embed_dims,n_voxels]
                    test_preds = np.hstack([p[:,None] for p  in preds_list])  # [n_test_items,n_voxels]
                    fitted_test_corrs = np.array(corrs_list)  # [n_voxels,]

                    # we removed NaNs in data before doing the fracridge. But we need all voxels to plot the brain maps,
                    # so we add them back at the right places here.
                    nan_idx_to_restore = np.array([i for i, x in enumerate(good_vertex) if not x])
                    nan_idx_to_restore_path = f"{fitted_models_dir}/{subj}_NanIdxToRestore.npy"
                    np.save(nan_idx_to_restore_path, nan_idx_to_restore)
                    
                    fitted_test_corrs = restore_nan_dims(fitted_test_corrs, nan_idx_to_restore, axis=0)
                    np.save(corrs_save_path, fitted_test_corrs)
                    print(f"... Encoding model predictions saved for {subj}")

                    restored_coefs = restore_nan_dims(coeffs, nan_idx_to_restore, axis=1)
                    np.save(coefs_save_path, restored_coefs)  # [n_embedding_dims, n_voxels]
                    print(f"... Encoding model coeffs saved for {subj}")

                else:
                    print("Found saved encoding model predictions, skipping...")

                    model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel_{ROI_NAME}_{ENCODING_MODEL_NAME}.pkl"
                    with open(model_save_path, "rb") as f:
                        fitted_fracridge = pickle.load(f)
                    nan_idx_to_restore = np.array([i for i, x in enumerate(good_vertex) if not x])
                    restored_coefs = restore_nan_dims(fitted_fracridge.coef_, nan_idx_to_restore, axis=1)
                    np.save(coefs_save_path, restored_coefs)  # [n_embedding_dims, n_voxels]
                    print(f"... Encoding model coeffs saved for {subj}")