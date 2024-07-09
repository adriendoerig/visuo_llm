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
from nsd_visuo_semantics.encoding_decoding_analyses.encoding_decoding_utils import remove_inert_embedding_dims, restore_inert_embedding_dims, make_conditions_nsd_embeddings, make_subj_conditional_nsd_embeddings, load_gcc_embeddings
from nsd_visuo_semantics.utils.nsd_get_data_light import get_subject_conditions, get_conditions_515, get_rois, get_sentence_lists, load_or_compute_betas_average



def nsd_decode_llm(EMBEDDING_MODEL_NAME, USE_ROIS, WHICH_ROIS, METRIC, nsd_dir, betas_dir, gcc_dir, base_save_dir):

    # params from nsd
    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x + 1}" for x in range(n_subjects)]
    targetspace = "fsaverage"

    # fractional ridge regression parameters
    n_alphas = 20
    fracs = np.linspace(1 / n_alphas, 1 + 1 / n_alphas, n_alphas)  # from https://github.com/nrdg/fracridge/blob/master/examples/plot_alpha_vs_gamma.py

    # set up directories
    rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')
    os.makedirs(base_save_dir, exist_ok=True)
    nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
    os.makedirs(nsd_embeddings_path, exist_ok=True)

    lookup_sentences, lookup_embeddings = None, None  # will be loaded later

    nsda = NSDAccess(nsd_dir)

    # get the condition list for the special 515
    # these will be used as testing set for the guse predictions
    conditions_515 = get_conditions_515(nsd_dir)
    images_515 = nsda.read_images(np.asarray(conditions_515) - 1)
    sentences_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)

    # prepare the test set embeddings
    embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings.npy"
    if not os.path.exists(embeddings_test_path):
        embeddings_test = make_conditions_nsd_embeddings(nsda, EMBEDDING_MODEL_NAME, conditions_515)
        np.save(embeddings_test_path, embeddings_test)
    else:
        embeddings_test = np.load(embeddings_test_path)

    # ROI STUFF
    if USE_ROIS is not None:
        maskdata, roi_id2name = get_rois(USE_ROIS, rois_dir)
        roi_name2id = {v: k for k, v in roi_id2name.items()}
        if WHICH_ROIS == 'allvisROIs':
            ROI_NAMES = ['allvisROIs']
        else:
            ROI_NAMES = list(roi_name2id.keys())
            ROI_NAMES.remove("Unknown")
        print(f"Using ROI_NAMES = {ROI_NAMES}")
    else:
        ROI_NAMES = ["fullbrain"]


    for ROI_NAME in ROI_NAMES:
        this_results_dir = os.path.join(base_save_dir, f'{EMBEDDING_MODEL_NAME}_results_ROI{ROI_NAME}_decoding')
        os.makedirs(this_results_dir, exist_ok=True)
        fitted_models_dir = os.path.join(this_results_dir, "fitted_models")
        os.makedirs(fitted_models_dir, exist_ok=True)

        for s_n, subj in enumerate(subs):
            # prepare the train/val set embeddings

            # get sampled conditions for this subject
            conditions, conditions_sampled, sample = get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True)
            # identify which images in the sample are from conditions_515
            sample_515_bool = [True if x in conditions_515 else False for x in sample]
            # and identify which sample images aren't in conditions_515
            sample_train_bool = [False if x in conditions_515 else True for x in sample]
            # select images that are not in special 515 for training
            sample_train = sample[sample_train_bool]

            # get the guse embeddings for the training sample
            train_embeddings_path = (f"{nsd_embeddings_path}/captions_not515_embeddings_{subj}.npy")
            if not os.path.exists(train_embeddings_path):
                embeddings_train = make_subj_conditional_nsd_embeddings(nsda, sample_train, EMBEDDING_MODEL_NAME, bool_conditional=None)  # none because we are already filtering the conditions above
                np.save(train_embeddings_path, embeddings_train)
            else:
                embeddings_train = np.load(train_embeddings_path)

            # Betas per subject
            print(f"loading betas for {subj}")
            betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
            betas_mean = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)

            if USE_ROIS is not None:
                # load the lh mask
                orig_n_voxels = betas_mean.shape[0]
                if ROI_NAME.lower() == 'allvisrois':
                    vs_mask = maskdata != 0
                else:
                    vs_mask = maskdata == roi_name2id[ROI_NAME]
                betas_mean = betas_mean[vs_mask, :]
                print(f"Applied {ROI_NAME} ROI mask. Went from {orig_n_voxels} to {betas_mean.shape[0]}")

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

            # In this case, we use voxels to linearly predict voxels
            embeddings_train, drop_dims_idx, drop_dims_avgs = remove_inert_embedding_dims(embeddings_train)

            model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridge_{ROI_NAME}.pkl"

            if not os.path.exists(model_save_path):
                print("Fitting fractional ridge regression...")
                frr = FracRidgeRegressorCV(jit=True, fit_intercept=True)
                fitted_fracridge = frr.fit(
                    betas_train,
                    embeddings_train,
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
                test_preds = restore_inert_embedding_dims(test_preds, drop_dims_idx, drop_dims_avgs)
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
            if lookup_sentences is None or lookup_embeddings is None:
                lookup_sentences, lookup_embeddings = load_gcc_embeddings(gcc_dir=gcc_dir)

            print("Plotting predicted sentences...")
            plot_n_examples = 10
            for i in range(0, 515, 515 // plot_n_examples):
                this_image = images_515[image_corrs_argsort[i]]
                this_target_sentence = sentences_515[image_corrs_argsort[i]][0]
                this_pred = test_preds[image_corrs_argsort[i]]

                lookup_distances = cdist(this_pred[None, :], lookup_embeddings, metric=METRIC,)
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
            np.save(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.npy", pred_target_rdm)

            print(f"Mean correlation between predicted and target embeddings = {np.mean(np.diag(pred_target_rdm))}")
            np.save(f"{this_results_dir}/mean_corr(predtarget)_{METRIC}_{subj}.npy", np.mean(np.diag(pred_target_rdm)))

            hist_data = np.empty((pred_target_rdm.shape[0],))
            plt.figure(figsize=(14, 7))  # Make it 14x7 inch
            # plt.style.use("seaborn-whitegrid")  # nice and clean grid
            for i in range(pred_target_rdm.shape[0]):
                off_diag_bool = [False if x == i else True for x in range(pred_target_rdm.shape[0])]
                hist_data[i] = pred_target_rdm[i, i] - np.mean(pred_target_rdm[i, off_diag_bool])
            plt.hist(hist_data, bins=hist_data.shape[0]//10, facecolor="#2ab0ff", edgecolor="#169acf", linewidth=0.5)
            plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_{METRIC}_{subj}.svg")
            plt.close()
            np.save(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_{METRIC}_{subj}.npy", hist_data)

        print("Making plots aggregated across subjects...")
        loaded_rdms = []
        for subj in subs:
            loaded_rdms.append(
                np.load(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_{subj}.npy"))
        loaded_rdms = np.stack(loaded_rdms)  # np.array of shape n_subjx515x515

        mean_rdm = np.mean(loaded_rdms, axis=0)
        im = plt.matshow(mean_rdm, cmap="magma")
        plt.colorbar(im, shrink=0.8)
        plt.axis("off")
        plt.title("mean across subjects\npredicted (rows) vs. target (cols) embeddings")
        plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_subjMean.svg")
        plt.close()
        np.save(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_{METRIC}_subjMean.npy", mean_rdm)

        print(f"Mean correlation between predicted and target embeddings = {np.mean(np.diag(mean_rdm))}")
        np.save(f"{this_results_dir}/mean_corr(predtarget)_{METRIC}_subjMean.npy", np.mean(np.diag(mean_rdm)))

        n_test_items = loaded_rdms.shape[-1]
        hist_data = np.empty((n_subjects * n_test_items,))
        for s in range(n_subjects):
            for i in range(n_test_items):
                off_diag_bool = [False if x == i else True for x in range(n_test_items)]
                hist_data[s * loaded_rdms.shape[-1] + i] = loaded_rdms[s, i, i] - np.mean(loaded_rdms[s, i, off_diag_bool])
        plt.figure(figsize=(14, 7))
        # plt.style.use("seaborn-whitegrid")  # nice and clean grid
        plt.hist(hist_data, bins=hist_data.shape[0]//10, facecolor="#2ab0ff", edgecolor="#169acf", linewidth=0.5)
        plt.savefig(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_{METRIC}_subjMean.svg")
        plt.close()
        np.save(f"{this_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_{METRIC}_subjMean.npy", hist_data)