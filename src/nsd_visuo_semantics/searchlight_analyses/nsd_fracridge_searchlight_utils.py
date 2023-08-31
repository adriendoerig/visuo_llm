import warnings

import numpy as np
import tensorflow_probability as tfp
from fracridge import FracRidgeRegressorCV
from joblib import Parallel, cpu_count, delayed
from sklearn.exceptions import ConvergenceWarning


class GroupIterator:
    """Group iterator. cf. nilearn.
    Provides group of features for search_light loop
    that may be used with Parallel.
    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s
    """

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        yield from split


def nsd_fit_rdms_fracridge(
    sl_train,
    sl_test,
    model_train,
    model_test,
    fracs,
    verbose=0,
    n_xval_folds=5,
    n_jobs=1,
):
    """
    Fits fracridge to predict brain rdms on a batch of batch_sl_locations from the rdms of n_model_layers.
    Model and brain rdms are split into smaller train (usually rdm_cells_train = upper_tri_70x70) and test rdms
    Training input shapes: model_train=[rdm_cells_train, n_model_layers], sl_train=[rdm_cells_train, batch_sl_locations]
    Returns: RDM fracridge predictions from model_test rdms (shape: [rdm_cells_test, batch_sl_locations])
    fracs: list of the fractions to test using fracridge (see fracridge docs)
    verbose: as used in sklearn standard pipelines
    n_xval_folds: how many crossval folds to use
    n_jobs: how many xfolds to run in parallel. NOTE: in the current setup, we are parallelizing at the level of
        searchlight location chunks, so this n_jobs is better left to 1, unless you can maximize chunk paralellization
        and still have space for parallelization here.
    """

    if np.any(np.isnan(sl_train)):
        print("Found NaNs in a searchlight")
    else:
        frr = FracRidgeRegressorCV(
            jit=True,
            fit_intercept=True,
            cv=n_xval_folds,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        fitted_fracridge = frr.fit(model_train, sl_train, frac_grid=fracs)
        these_preds = fitted_fracridge.predict(model_test).astype(
            np.float32
        )  # [n_rdm_dims_test, n_sl_locations]
        return these_preds


def pairwise_corr(x, y, batch_size=-1):
    """
    :param x: [n_pairs_to_correlate, n_dims_to_correlate]
    :param y:  [n_pairs_to_correlate, n_dims_to_correlate]
    :param batch_size: compute correlations in batches of this size. if -1 -> do all in one go
    :return:
    """

    if batch_size == -1:
        return tfp.stats.correlation(
            x, y, sample_axis=0, event_axis=None
        ).numpy()
    else:
        corrs_out = []
        batch_indices = np.array_split(np.arange(x.shape[0]), batch_size)
        for this_batch in batch_indices:
            corrs_out.append(
                tfp.stats.correlation(
                    x[this_batch],
                    y[this_batch],
                    sample_axis=1,
                    event_axis=None,
                ).numpy()
            )
        return np.hstack(corrs_out)


def remove_nan_dims(x):
    """Remove rows with nans from array."""

    drop_rows_idx = np.unique(np.where(np.isnan(x))[0])
    keep_rows_idx = np.setdiff1d(np.arange(x.shape[0]), drop_rows_idx)
    nanless_x = x[keep_rows_idx]

    return nanless_x, drop_rows_idx


def restore_nan_dims(x, drop_rows_idx):
    """Add back the removed dimensions (if any were removed, see remove_nan_dims())."""

    nan_idx_offset = 0
    prev = -1
    for i in range(drop_rows_idx.shape[0]):
        x = np.insert(x, drop_rows_idx[i - nan_idx_offset], np.nan)
        if i == prev:
            nan_idx_offset += 1
        else:
            nan_idx_offset = 0
        prev += 1

    return x


def nsd_parallelize_fracridge_fit(
    brain_sl_rdms_sample_train,
    brain_sl_rdms_sample_test,
    model_rdms_sample_train,
    model_rdms_sample_test,
    fracs,
    n_jobs,
    verbose,
):
    """
    Parallelizes fracridge computation across several jobs of searchlight location chunks.
        brain_sl_rdms_sample_train: [n_voxels, n_rdm_dims_train]
        brain_sl_rdms_sample_test: [n_voxels, n_rdm_dims_test]
        model_rdms_sample_train: [n_model_layers, n_rdm_dims_train]
        model_rdms_sample_test: [n_model_layers, n_rdm_dims_test]
        fracs: list of the fractions to test using fracridge (see fracridge docs)
        n_jobs: how many jobs to use (-1 returns 1 job per CPU)
        verbose: how much stuff to print while computing
        :return: vector of correlations between fracridge-predicted model rdms and brain rdms for each sl location
    """

    (
        brain_sl_rdms_sample_train,
        brain_sl_rdms_sample_test,
        model_rdms_sample_train,
        model_rdms_sample_test,
    ) = (
        brain_sl_rdms_sample_train.astype(np.float32),
        brain_sl_rdms_sample_test.astype(np.float32),
        model_rdms_sample_train.astype(np.float32),
        model_rdms_sample_test.astype(np.float32),
    )

    if np.any(np.isnan(brain_sl_rdms_sample_train)):
        # subject 8 has some NaNs. This creates errors in fracridge. So we remove them from the arrays passed to
        # fracridge. We remember the indices where we removed them, and we will add NaNs to the correlations vector
        # at the end of this function. This is a data problem which also occurs for the non-fracridge versions of
        # this script, but only fracridge throws errors when NaNs are in the brain data, so we only need to go through
        # this trouble here.
        print(
            "Found some NaNs in brain data, these voxels will be skipped, and NaNs inserted in the corr_map."
        )
        skipping_nans = True
        brain_sl_rdms_sample_train, nan_idx = remove_nan_dims(
            brain_sl_rdms_sample_train
        )
    else:
        skipping_nans = False

    group_iter = GroupIterator(brain_sl_rdms_sample_train.shape[0], n_jobs)

    # just a test of a single fracridge fit: remove for real runs
    # list_i = [x for x in group_iter]
    # preds = nsd_fit_rdms_fracridge(sl_train=brain_sl_rdms_sample_train[list_i[0]].T,  # [n_rdm_dims_train, job_n_sl_locations]
    #             sl_test=brain_sl_rdms_sample_test[list_i[0]].T,  # [n_rdm_dims_test, job_n_sl_locations]
    #             model_train=model_rdms_sample_train.T,  # [n_rdm_dims_train, n_model_layers]
    #             model_test=model_rdms_sample_test.T,  # [n_rdm_dims_test, n_model_layers]
    #             fracs=fracs,
    #             n_jobs=n_jobs,
    #             verbose=verbose)
    # import pdb; pdb.set_trace()
    # end of test

    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        # corrs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        all_test_preds = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(nsd_fit_rdms_fracridge)(
                sl_train=brain_sl_rdms_sample_train[
                    list_i
                ].T,  # [n_rdm_dims_train, job_n_sl_locations]
                sl_test=brain_sl_rdms_sample_test[
                    list_i
                ].T,  # [n_rdm_dims_test, job_n_sl_locations]
                model_train=model_rdms_sample_train.T,  # [n_rdm_dims_train, n_model_layers]
                model_test=model_rdms_sample_test.T,  # [n_rdm_dims_test, n_model_layers]
                fracs=fracs,
                n_xval_folds=4,
                n_jobs=2,
                verbose=1,
            )
            for list_i in group_iter
        )

    print(
        "Done fitting fracridge for all sl_locations, computing correlations..."
    )
    all_test_preds = np.hstack(all_test_preds)
    corrs = pairwise_corr(
        all_test_preds.T, brain_sl_rdms_sample_test, batch_size=1000
    )

    if skipping_nans:
        # add back nan dimensions
        corrs = restore_nan_dims(corrs, nan_idx)
        print("sanity cheeck for NaN handling:")
        print("nan_idx", nan_idx)
        try:
            print("nan in corrs", np.unique(np.where(np.isnan(corrs))[0]))
        except ValueError:
            print("you fucked up in printing the nan locations")

    return corrs
