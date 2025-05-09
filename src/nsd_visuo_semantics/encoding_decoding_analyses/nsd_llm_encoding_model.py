"""Train a frac ridge regression between NSD voxels and embeddings.
"""

import os, pickle
import numpy as np
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    pass
from fracridge import FracRidgeRegressorCV
from nsd_access import NSDAccess
from nsd_visuo_semantics.encoding_decoding_analyses.encoding_decoding_utils import restore_nan_dims, pairwise_corr, make_conditions_nsd_embeddings, make_subj_conditional_nsd_embeddings
from nsd_visuo_semantics.utils.nsd_get_data_light import get_subject_conditions, get_conditions_515, load_or_compute_betas_average

def nsd_llm_encoding_model(EMBEDDING_MODEL_NAME, nsd_dir, betas_dir, base_save_dir):

    # params from nsd
    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x + 1}" for x in range(n_subjects)]
    targetspace = "fsaverage"

    # fractional ridge regression parameters
    n_alphas = 20
    fracs = np.linspace(1/n_alphas, 1+1/n_alphas, n_alphas)  # from https://github.com/nrdg/fracridge/blob/master/examples/plot_alpha_vs_gamma.py

    # paths
    nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
    os.makedirs(nsd_embeddings_path, exist_ok=True)
    this_results_dir = os.path.join(base_save_dir, f'{EMBEDDING_MODEL_NAME}_encodingModel')
    os.makedirs(this_results_dir, exist_ok=True)
    fitted_models_dir = os.path.join(this_results_dir, "fitted_models")
    os.makedirs(fitted_models_dir, exist_ok=True)

    # nsd access is used to get the captions etc
    nsda = NSDAccess(nsd_dir)

    # get the condition list for the special 515
    # these will be used as testing set
    conditions_515 = get_conditions_515(nsd_dir)

    # prepare the test set embeddings
    embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings.npy"
    if not os.path.exists(embeddings_test_path):
        embeddings_test = make_conditions_nsd_embeddings(nsda, EMBEDDING_MODEL_NAME, conditions_515)
        np.save(embeddings_test_path, embeddings_test)
    else:
        embeddings_test = np.load(embeddings_test_path)

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

        # get the embeddings for the training sample
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

        corrs_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingCorrMap.npy"
        coefs_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingCoefs.npy"
        model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel.pkl"
        nan_idx_to_restore_save_path = f"{fitted_models_dir}/{subj}_NanIdxToRestore.npy"
        

        if not os.path.exists(corrs_save_path):

            if not os.path.exists(model_save_path):
                print("Fitting fractional ridge regression...")
                frr = FracRidgeRegressorCV(jit=True, fit_intercept=True)
                fitted_fracridge = frr.fit(
                    embeddings_train,
                    betas_train,
                    frac_grid=fracs,
                )
                with open(model_save_path, "wb") as f:
                    pickle.dump(fitted_fracridge, f)
            else:
                print("Found saved fractional ridge regression, loading...")
                with open(model_save_path, "rb") as f:
                    fitted_fracridge = pickle.load(f)

            test_preds = fitted_fracridge.predict(embeddings_test)  # [n_test_items, n_voxels]
            fitted_test_corrs = pairwise_corr(test_preds, betas_test)  # [n_voxels,]

            # we removed NaNs in data before doing the fracridge. But we need all voxels to plot the brain maps,
            # so we add them back at the right places here.
            nan_idx_to_restore = np.array([i for i, x in enumerate(good_vertex) if not x])
            fitted_test_corrs = restore_nan_dims(fitted_test_corrs, nan_idx_to_restore, axis=0)

            np.save(nan_idx_to_restore_save_path, nan_idx_to_restore)
            print(f"... NaN indices to restore saved for {subj}")
            
            np.save(corrs_save_path, fitted_test_corrs)
            print(f"... Encoding model predictions saved for {subj}")

            restored_coefs = restore_nan_dims(fitted_fracridge.coef_, nan_idx_to_restore, axis=1)
            np.save(coefs_save_path, restored_coefs)  # [n_embedding_dims, n_voxels]
            print(f"... Encoding model coeffs saved for {subj}")

        else:
            print("Found saved encoding model predictions, skipping...")


