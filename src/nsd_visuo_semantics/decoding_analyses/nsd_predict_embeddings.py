"""Train a frac ridge regression between NSD voxels and embeddings.
"""

import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressorCV
from nsd_access import NSDAccess
from scipy.spatial.distance import cdist, correlation, cosine, pdist
from nsd_visuo_semantics.decoding_analyses.decoding_utils import remove_inert_embedding_dims, restore_inert_embedding_dims, nsd_parallelize_fracridge_fit, restore_nan_dims
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_rois,get_sentence_lists


EMBEDDING_MODEL_NAME = "all_mpnet_base_v2"
USE_ROIS = None  # "mpnet_noShared515_sig0.005_fsaverage"  # None, or 'mpnet_noShared515_sig0.005_fsaverage' or streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...
METRIC = "correlation"  # 'correlation', 'cosine'
PREDICT_X_FROM_Y = "embeddings_from_voxels"  # 'embeddings_from_voxels' or 'voxels_from_embeddings'
USE_GCC_LOOKUP = True
USE_LAION_LOOKUP = False  # if True, use LAION lookup in addition to Google Conceptual Captions lookup (makes small difference)

# setup
total_time = time.time()
n_jobs = 4
save_to_matlab = 0

# params from nsd
n_sessions = 40
n_subjects = 8
subs = [f"subj0{x + 1}" for x in range(n_subjects)]
targetspace = "fsaverage"

# fractional ridge regression parameters
n_alphas = 20
fracs = np.linspace(1 / n_alphas, 1 + 1 / n_alphas, n_alphas)  # from https://github.com/nrdg/fracridge/blob/master/examples/plot_alpha_vs_gamma.py

# set up directories
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "../results_dir"
nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
os.makedirs(nsd_embeddings_path, exist_ok=True)

nsda = NSDAccess(nsd_dir)

# get the condition list for the special 515
# these will be used as testing set for the guse predictions
conditions_515 = get_conditions_515(nsd_dir)
images_515 = nsda.read_images(np.asarray(conditions_515) - 1)
sentences_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)

# prepare the test set embeddings
embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings.npy"
if not os.path.exists(embeddings_test_path):
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    captions_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)
    dummy_embedding = get_embeddings(captions_515[0], embedding_model, EMBEDDING_MODEL_NAME)
    embedding_dim = dummy_embedding.shape[-1]
    embeddings_test = np.empty((515, embedding_dim))
    for i in range(len(captions_515)):
        embeddings_test[i] = np.mean(get_embeddings(captions_515[i], embedding_model, EMBEDDING_MODEL_NAME), axis=0)
    np.save(embeddings_test_path, embeddings_test)
else:
    embeddings_test = np.load(embeddings_test_path)
    embedding_dim = embeddings_test.shape[-1]

# ROI STUFF
if USE_ROIS is not None:
    maskdata, roi_id2name = get_rois(
        USE_ROIS, "./save_dir/roi_analyses/roi_defs"
    )
    roi_name2id = {v: k for k, v in roi_id2name.items()}
    ROI_NAMES = list(roi_name2id.keys())
    ROI_NAMES.remove("Unknown")
    print(f"Using ROI_NAMES = {ROI_NAMES}")
else:
    ROI_NAMES = ["fullbrain"]


for USE_N_STIMULI in [None]:  # None means use all stimuli
    for ROI_NAME in ROI_NAMES:
        this_results_dir = os.path.join(
            base_save_dir,
            f'{EMBEDDING_MODEL_NAME}_results_ROI{ROI_NAME}{"" if USE_N_STIMULI is None else f"_{USE_N_STIMULI}stims"}{"" if PREDICT_X_FROM_Y == "embeddings_from_voxels" else "_encodingModel"}',
        )
        os.makedirs(this_results_dir, exist_ok=True)
        fitted_models_dir = os.path.join(this_results_dir, "fitted_models")
        os.makedirs(fitted_models_dir, exist_ok=True)
        laion_lookup_dir = os.path.join(this_results_dir, "laion600_lookup")

        if USE_GCC_LOOKUP:
            lookup_sentences_path = os.path.join(
                base_save_dir,
                "google_conceptual_captions_embeddings",
                "conceptual_captions_{}.tsv",
            )
            lookup_embeddings_path = os.path.join(
                base_save_dir,
                "google_conceptual_captions_embeddings",
                "conceptual_captions_mpnet_{}.npy",
            )
            lookup_datasets = [
                "train",
                "val",
            ]  # we can choose to use either the gcc train, val, or both for the lookup
        if USE_LAION_LOOKUP:
            laion_overall_winner_sentences = np.load(
                f"{laion_lookup_dir}/all_subjects_decoded_sentences.npy"
            )  # [n_subj, 515]

        for s_n, subj in enumerate(subs):
            # prepare the train/val set embeddings

            # find indices that are NOT in the 515 spacial test images
            # extract conditions data
            conditions = get_conditions(nsd_dir, subj, n_sessions)
            # we also need to reshape conditions to be ntrials x 1
            conditions = np.asarray(conditions).ravel()
            # then we find the valid trials for which we do have 3 repetitions.
            conditions_bool = [
                True if np.sum(conditions == x) == 3 else False
                for x in conditions
            ]
            # and identify those.
            conditions_sampled = conditions[conditions_bool]
            # find the subject's condition list (sample pool)
            # this sample is the same order as the betas
            sample = np.unique(conditions[conditions_bool])

            # identify which image in the sample is a conditions_515
            sample_515_bool = [
                True if x in conditions_515 else False for x in sample
            ]
            # and identify which sample image isn't in conditions_515
            sample_train_bool = [
                False if x in conditions_515 else True for x in sample
            ]
            # also identify the training set (i.e. not the special 515)
            sample_train = sample[sample_train_bool]

            # get the guse embeddings for the training sample
            train_embeddings_path = (
                f"{nsd_embeddings_path}/captions_not515_embeddings_{subj}.npy"
            )
            if not os.path.exists(train_embeddings_path):
                embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
                captions_not515 = get_sentence_lists(nsda, sample_train - 1)
                embeddings_train = np.empty(
                    (len(captions_not515), embedding_dim)
                )
                for i in range(len(captions_not515)):
                    embeddings_train[i] = np.mean(
                        get_embeddings(
                            captions_not515[i],
                            embedding_model,
                            EMBEDDING_MODEL_NAME,
                        ),
                        axis=0,
                    )
                np.save(train_embeddings_path, embeddings_train)
            else:
                embeddings_train = np.load(train_embeddings_path)

            # Betas per subject
            print(f"loading betas for {subj}")
            betas_file = os.path.join(
                betas_dir, f"{subj}_betas_average_{targetspace}.npy"
            )
            betas_mean = np.load(betas_file, allow_pickle=True)

            if USE_ROIS is not None:
                # load the lh mask
                orig_n_voxels = betas_mean.shape[0]
                vs_mask = maskdata == roi_name2id[ROI_NAME]
                betas_mean = betas_mean[vs_mask, :]
                print(
                    f"Applied ROI mask. Went from {orig_n_voxels} to {betas_mean.shape[0]}"
                )

            good_vertex = [True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]
            if np.sum(good_vertex) != len(good_vertex):
                print(f"found some NaN for {subj}")
            betas_mean = betas_mean[good_vertex, :]

            # now we further split the brain data according to the 515 test set or the training set for that subject
            betas_test = betas_mean[:, sample_515_bool].T  # sub1: (515, 327673) (n_voxels may vary from subj to subj because of nans)
            betas_train = betas_mean[:, sample_train_bool].T  # sub1: (9485, 327673) (may vary from subj to subj)
            del betas_mean  # make space

            # format to float32, remove "inert" dimensions.
            embeddings_train, embeddings_test, betas_train, betas_test = (
                embeddings_train.astype(np.float32),
                embeddings_test.astype(np.float32),
                betas_train.astype(np.float32),
                betas_test.astype(np.float32),
            )

            if PREDICT_X_FROM_Y == "embeddings_from_voxels":
                # In this case, we use voxels to linearly predict voxels
                embeddings_train, drop_dims_idx, drop_dims_avgs = remove_inert_embedding_dims(embeddings_train)

                model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridge_{ROI_NAME}.pkl"

                if not os.path.exists(model_save_path):
                    print("Fitting fractional ridge regression...")
                    frr = FracRidgeRegressorCV(jit=True, fit_intercept=True)#, n_jobs=n_jobs)
                    fitted_fracridge = frr.fit(
                        betas_train[:USE_N_STIMULI],
                        embeddings_train[:USE_N_STIMULI],
                        frac_grid=fracs,
                    )
                    with open(model_save_path, "wb") as f:
                        pickle.dump(fitted_fracridge, f)
                else:
                    print("Found saved fractional ridge regression, loading...")
                    with open(model_save_path, "rb") as f:
                        fitted_fracridge = pickle.load(f)

                print("Gathering predictions...")
                # NOTE: fitted_fracridge.predict(X_test) = X_test@fitted_fracridge.coef_ where @ is matmul
                predicted_embeddings_save_path = f"{this_results_dir}/{subj}_fittedFracridge_{ROI_NAME}_testPredictions.pkl"
                if not os.path.exists(predicted_embeddings_save_path):
                    test_preds = fitted_fracridge.predict(betas_test)
                    test_preds = restore_inert_embedding_dims(
                        test_preds, drop_dims_idx, drop_dims_avgs
                    )
                    with open(predicted_embeddings_save_path, "wb") as f:
                        pickle.dump(test_preds, f)
                else:
                    with open(predicted_embeddings_save_path, "rb") as f:
                        test_preds = pickle.load(f)

                # prediction-target similarities will be used to visualize examples with different prediction accuracies
                print("Getting prediction-target similarities on test set...")
                image_corrs = []
                for image_i in range(embeddings_test.shape[0]):
                    this_pred = test_preds[image_i, :]
                    this_target = embeddings_test[image_i, :]
                    if METRIC == "correlation":
                        this_distance = correlation(this_pred, this_target)
                    elif METRIC == "cosine":
                        this_distance = cosine(this_pred, this_target)
                    else:
                        raise Exception(f"METRIC not understood. You entered {METRIC}.")
                    image_corrs.append(this_distance)
                image_corrs_argsort = np.argsort(image_corrs)

                print("Loading lookup embeddings and captions...")
                if USE_GCC_LOOKUP:
                    lookup_embeddings = None
                    for d in lookup_datasets:
                        # there is a train and val set in the gcc captions, we load the ones chosen by the user (concatenating them)
                        if lookup_embeddings is None:
                            lookup_embeddings = np.load(
                                lookup_embeddings_path.format(d)
                            )
                            df = pd.read_csv(
                                lookup_sentences_path.format(d),
                                sep="\t",
                                header=None,
                                names=["sent", "url"],
                            )
                            lookup_sentences = df["sent"].to_list()
                        else:
                            lookup_embeddings = np.vstack(
                                [
                                    lookup_embeddings,
                                    np.load(lookup_embeddings_path.format(d)),
                                ]
                            )
                            df = pd.read_csv(
                                lookup_sentences_path.format(d),
                                sep="\t",
                                header=None,
                                names=["sent", "url"],
                            )
                            lookup_sentences += df["sent"].to_list()

                print("Plotting predicted sentences...")
                plot_n_examples = 10
                for i in range(0, 515, 515 // 10):
                    this_image = images_515[image_corrs_argsort[i]]
                    this_target_sentence = sentences_515[
                        image_corrs_argsort[i]
                    ][0]
                    this_pred = test_preds[image_corrs_argsort[i]]

                    if USE_GCC_LOOKUP:
                        lookup_distances = cdist(
                            this_pred[None, :],
                            lookup_embeddings,
                            metric=METRIC,
                        )
                        winner = np.argmin(lookup_distances)
                        this_pred_sentence = lookup_sentences[winner]

                        plt.figure(figsize=(8, 8))
                        plt.imshow(this_image)
                        plt.title(f"{METRIC} rank {i}\ntarget: {this_target_sentence}\npred: {this_pred_sentence}")
                        plt.axis("off")
                        plt.savefig(f"{this_results_dir}/predicted_sentence_{subj}_{METRIC}_rank{i}.svg")
                        plt.close()

                print("Computing mean distance between test embeddings...")
                distances = pdist(embeddings_test, metric=METRIC)
                print(f"\tmean_distance = {np.mean(distances)}")

                print("Making predicted vs. target embeddings RDM...")
                pred_target_rdm = 1 - cdist(test_preds, embeddings_test, metric=METRIC)
                im = plt.matshow(pred_target_rdm, cmap="magma")
                plt.colorbar(im, shrink=0.8)
                plt.axis("off")
                plt.title(f"subject {subj}\npredicted (rows) vs. target (cols) embeddings")
                plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.svg")
                plt.close()
                np.save(
                    f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.npy",
                    pred_target_rdm,
                )

                print(f"Mean correlation between predicted and target embeddings = {np.mean(np.diag(pred_target_rdm))}")
                np.save(
                    f"{this_results_dir}/mean_corr(predtarget)_{METRIC}_{subj}.npy",
                    np.mean(np.diag(pred_target_rdm)),
                )

                hist_data = np.empty((pred_target_rdm.shape[0],))
                plt.figure(figsize=(14, 7))  # Make it 14x7 inch
                plt.style.use("seaborn-whitegrid")  # nice and clean grid
                for i in range(pred_target_rdm.shape[0]):
                    off_diag_bool = [
                        False if x == i else True
                        for x in range(pred_target_rdm.shape[0])
                    ]
                    hist_data[i] = pred_target_rdm[i, i] - np.mean(pred_target_rdm[i, off_diag_bool])
                plt.hist(
                    hist_data,
                    bins=hist_data.shape[0] // 10,
                    facecolor="#2ab0ff",
                    edgecolor="#169acf",
                    linewidth=0.5,
                )
                plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_{METRIC}_{subj}.svg")
                plt.close()
                np.save(
                    f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_{METRIC}_{subj}.npy",
                    hist_data,
                )

            elif PREDICT_X_FROM_Y == "voxels_from_embeddings":
            # In this case, we use embeddings as an encoding model to linearly predict voxels

                corrs_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeCorrMap_{ROI_NAME}.pkl"

                if not os.path.exists(corrs_save_path):
                    print("Fitting fractional ridge regression...")
                    frr = FracRidgeRegressorCV(jit=True, fit_intercept=True)#, n_jobs=n_jobs)
                    fitted_fracridge = frr.fit(
                        embeddings_train[:USE_N_STIMULI],
                        betas_train[:USE_N_STIMULI],
                        frac_grid=fracs,
                    )

                    # we removed NaNs in data before doing the fracridge. But we need all voxels to plot the brain maps,
                    # so we add them back at the right places here.
                    nan_idx_to_restore = np.array([i for i, x in enumerate(good_vertex) if not x])
                    fitted_model_corrs = restore_nan_dims(fitted_model_corrs, nan_idx_to_restore)

                    nan_idx_to_restore = [i for i, x in enumerate(good_vertex) if not x]
                    with open(corrs_save_path, "wb") as f:
                        pickle.dump(fitted_model_corrs, f)
                        print(f"... Encoding model predictions saved for {subj}")
                else:
                    print("Found saved encoding model predictions, loading...")
                    with open(corrs_save_path, "rb") as f:
                        fitted_model_corrs = pickle.load(f)

                    with open(corrs_save_path, "wb") as f:
                        pickle.dump(fitted_model_corrs, f)
                        print(f"... Encoding model predictions saved for {subj}")
            else:
                raise Exception(
                    f"Please use PREDICT_X_FROM_Y = 'voxels_from_embeddings' or embeddings_from_voxels."
                    f"Found {PREDICT_X_FROM_Y}"
                )

        if PREDICT_X_FROM_Y == "embeddings_from_voxels":
            print("Making plots aggregated across subjects...")
            loaded_rdms = []
            for subj in subs:
                loaded_rdms.append(
                    np.load(
                        f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.npy"
                    )
                )
            loaded_rdms = np.stack(loaded_rdms)  # np.array of shape n_subjx515x515

            mean_rdm = np.mean(loaded_rdms, axis=0)
            im = plt.matshow(mean_rdm, cmap="magma")
            plt.colorbar(im, shrink=0.8)
            plt.axis("off")
            plt.title("mean across subjects\npredicted (rows) vs. target (cols) embeddings")
            plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_subjMean.svg")
            plt.close()
            np.save(
                f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_subjMean.npy",
                mean_rdm,
            )

            print(f"Mean correlation between predicted and target embeddings = {np.mean(np.diag(mean_rdm))}")
            np.save(
                f"{this_results_dir}/mean_corr(predtarget)_{METRIC}_subjMean.npy",
                np.mean(np.diag(mean_rdm)),
            )

            n_test_items = loaded_rdms.shape[-1]
            hist_data = np.empty((n_subjects * n_test_items,))
            for s in range(n_subjects):
                for i in range(n_test_items):
                    off_diag_bool = [
                        False if x == i else True for x in range(n_test_items)
                    ]
                    hist_data[s * loaded_rdms.shape[-1] + i] = loaded_rdms[
                        s, i, i
                    ] - np.mean(loaded_rdms[s, i, off_diag_bool])
            plt.figure(figsize=(14, 7))  # Make it 14x7 inch
            plt.style.use("seaborn-whitegrid")  # nice and clean grid
            plt.hist(
                hist_data,
                bins=hist_data.shape[0] // 10,
                facecolor="#2ab0ff",
                edgecolor="#169acf",
                linewidth=0.5,
            )
            plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_{METRIC}_subjMean.svg")
            plt.close()
            np.save(
                f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_{METRIC}_subjMean.npy",
                hist_data,
            )

            # # You can use this to get the histogram data from the loaded rdms
            # METRIC = 'correlation'
            # loaded_rdms = []
            # this_results_dir = 'path_to_saved_results'
            # for subj in subs:
            #     loaded_rdms.append(np.load(f'{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.npy'))
            # loaded_rdms = np.stack(loaded_rdms)  # np.array of shape n_subjx515x515
            # loaded_rdms.shape
            # def decode_hist(loaded_rdms, subjects_list):
            #     n_test_items = loaded_rdms.shape[-1]
            #     hist_data = np.empty((len(subjects_list)*n_test_items,))
            #     for s in subjects_list:
            #         for i in range(n_test_items):
            #             off_diag_bool = [False if x == i else True for x in range(n_test_items)]
            #             hist_data[s * loaded_rdms.shape[-1] + i] = loaded_rdms[s, i, i] - np.mean(loaded_rdms[s, i, off_diag_bool])
            #     return hist_data
