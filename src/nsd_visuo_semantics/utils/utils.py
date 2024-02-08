from math import ceil, floor

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import cdist, squareform

def corr_rdms(X, Y):
    """corr_rdms useful for correlation of RDMs (e.g. in multimodal RSA fusion)
        where you correlate two ndim RDM stacks.

    Args:
        X [array]: e.g. the fmri searchlight RDMs (ncenters x npairs)
        Y [array]: e.g. the EEG time RDMs (ntimes x npairs)

    Returns:
        [type]: correlations between X and Y, of shape dim0 of X by dim0 of Y
    """
    # from scipy.stats.stats import pearsonr 
    # X[np.isnan(X)] = 0
    # Y[np.isnan(Y)] = 0
    # return pearsonr(X.squeeze(), Y.squeeze())[0]
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]
    return np.einsum("ik,jk", X, Y)


def load_mat_save_npy(mat_path, mat_array_name, npy_path=None):
    """load_mat_save_npy

    Args:
        mat_path (str): path to .mat file
        npy_path (str): path to save .npy file
    """
    import scipy.io as sio

    if npy_path is None:
        npy_path = mat_path.replace(".mat", ".npy")

    data = sio.loadmat(mat_path)[mat_array_name]
    np.save(npy_path, data)


def reorder_rdm(utv, newOrder):
    ax, bx = np.ix_(newOrder, newOrder)
    newOrderRDM = squareform(utv)[ax, bx]
    return squareform(newOrderRDM, "tovector", 0)
