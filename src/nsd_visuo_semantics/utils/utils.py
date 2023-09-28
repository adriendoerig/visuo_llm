from math import ceil, floor

import numpy as np
from fracridge import FracRidgeRegressorCV
from scipy.optimize import nnls
from scipy.spatial.distance import cdist, squareform


# def compute_rdm(measurements):
#     """compute_rdm computes a correlation distance based rdm from
#        a measurements array

#     Args:
#         measurements (2D array): conditions by features

#     Returns:
#         correlation distance vector
#     """
#     row, col = np.triu_indices(measurements.shape[0], 1)
#     measurements = measurements - measurements.mean(axis=1, keepdims=True)
#     measurements /= np.sqrt(np.einsum("ij,ij->i", measurements, measurements))[
#         :, None
#     ]

#     return 1 - np.einsum("ik,jk", measurements, measurements)[row, col]


# def compute_rdm_cosine(measurements):
#     """compute_rdm computes a cosine distance based rdm from
#        a measurements array

#     Args:
#         measurements (2D array): conditions by features

#     Returns:
#         correlation distance vector
#     """
#     row, col = np.triu_indices(measurements.shape[0], 1)
#     measurements = measurements / np.linalg.norm(measurements, axis=1)[:, None]
#     return 1 - np.einsum("ik,jk->ij", measurements, measurements)[row, col]


# def make_project_matrix(X):
#     """Calculates a projection matrix

#     Args:
#         X (array): design matrix

#     Returns:
#         array: Projection matrix size of X.shape[0] x X.shape[0]
#     """
#     n_items = X.shape[0]
#     X = np.c_[X, np.ones(n_items)]
#     X = np.mat(X)
#     return np.eye(X.shape[0]) - (X * (np.linalg.inv(X.T * X) * X.T))


# def mask_vec_condition(condition, n_conditions):
#     """find utv indices for a subset of conditions

#     Args:
#         condition ([type]): [description]
#         n_conditions ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     ax, bx = np.ix_(condition, condition)
#     mask_rdm = np.zeros((n_conditions, n_conditions))
#     mask_rdm[ax, bx] = 1
#     mask_rdm[np.eye(n_conditions) == 1] = 0
#     return squareform(mask_rdm, "tovector") == 1


# def upper_tri_indexing(RDM):
#     """upper_tri_indexing returns the upper triangular index of an RDM

#     Args:
#         RDM 2Darray: squareform RDM

#     Returns:
#         1D array: upper triangular vector of the RDM
#     """
#     # returns the upper triangle
#     m = RDM.shape[0]
#     r, c = np.triu_indices(m, 1)
#     return RDM[r, c]


# def get_distance_cdist(data, ind):
#     """get_distance_cdist returns the correlation distance
#        across condition patterns in X
#        get_distance_cdist uses scipy's cdist method

#     Args:
#         data (array): conditions x all channels.
#         ind (vector): subspace of voxels for that sphere.

#     Returns:
#         UTV: pairwise distances between condition patterns in X
#              (in upper triangular vector form)
#     """
#     ind = np.array(ind)
#     X = data[ind, :].T
#     return upper_tri_indexing(cdist(X, X, "correlation"))


# def get_distance(data, ind):
#     """get_distance returns the correlation distance
#        across condition patterns in X
#        get_distance uses numpy's einsum

#     Args:
#         data (array): conditions x all channels.
#         ind (vector): subspace of voxels for that sphere.

#     Returns:
#         UTV: pairwise distances between condition patterns in X
#              (in upper triangular vector form)
#     """
#     ind = np.array(ind)
#     X = np.array(data[ind, :]).T
#     X = X - X.mean(axis=1, keepdims=True)
#     X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]

#     return 1 - upper_tri_indexing(np.einsum("ik,jk", X, X))

from scipy.stats.stats import pearsonr 
def corr_rdms(X, Y):
    """corr_rdms useful for correlation of RDMs (e.g. in multimodal RSA fusion)
        where you correlate two ndim RDM stacks.

    Args:
        X [array]: e.g. the fmri searchlight RDMs (ncenters x npairs)
        Y [array]: e.g. the EEG time RDMs (ntimes x npairs)

    Returns:
        [type]: correlations between X and Y, of shape dim0 of X by dim0 of Y
    """
    X[np.isnan(X)] = 0
    Y[np.isnan(Y)] = 0
    return pearsonr(X.squeeze(), Y.squeeze())[0]
    # X = X - X.mean(axis=1, keepdims=True)
    # X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    # Y = Y - Y.mean(axis=1, keepdims=True)
    # Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]

    # return np.einsum("ik,jk", X, Y)


# def chunking(vect, num, chunknum=None):
#     """chunking
#     Input:
#         <vect> is a array
#         <num> is desired length of a chunk
#         <chunknum> is chunk number desired (here we use a 1-based
#               indexing, i.e. you may want the frist chunk, or the second
#               chunk, but not the zeroth chunk)
#     Returns:
#         [numpy array object]:

#         return a numpy array object of chunks.  the last vector
#         may have fewer than <num> elements.

#         also return the beginning and ending indices associated with
#         this chunk in <xbegin> and <xend>.

#     Examples:

#         a = np.empty((2,), dtype=np.object)
#         a[0] = [1, 2, 3]
#         a[1] = [4, 5]
#         assert(np.all(chunking(list(np.arange(5)+1),3)==a))

#         assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))


#         # do in chunks
#         chunks = chunking(
#             list(range(mflat.shape[1])), int(np.ceil(mflat.shape[1]/numchunks)))

#     """
#     if chunknum is None:
#         nchunk = int(np.ceil(len(vect) / num))
#         f = []
#         for point in range(nchunk):
#             f.append(
#                 vect[point * num : np.min((len(vect), int((point + 1) * num)))]
#             )

#         return np.asarray(f)
#     else:
#         f = chunking(vect, num)
#         # double check that these behave like in matlab (xbegin)
#         xbegin = (chunknum - 1) * num + 1
#         return np.asarray(f[num - 1]), xbegin, xbegin + len(f[num - 1]) - 1


# def isnotfinite(arr):
#     res = np.isfinite(arr)
#     np.bitwise_not(res, out=res)  # in-place
#     return res


# def sample_conditions(conditions, n_samples, replace=False):
#     unique_conditions = np.unique(conditions)

#     choices = np.random.choice(unique_conditions, n_samples, replace=replace)

#     conditions_bool = np.any(
#         np.array([conditions == v for v in choices]), axis=0
#     )

#     return conditions_bool


# def makeimagestack(m):
#     """
#     def makeimagestack(m)

#     <m> is a 3D matrix.  if more than 3D, we reshape to be 3D.
#     we automatically convert to double format for the purposes of this method.
#     try to make as square as possible
#     (e.g. for 16 images, we would use [4 4]).
#     find the minimum possible to fit all the images in.
#     """

#     bordersize = 1

#     # calc
#     nrows, ncols, numim = m.shape
#     mx = np.nanmax(m.ravel())

#     # calculate csize

#     rows = floor(np.sqrt(numim))
#     cols = ceil(numim / rows)
#     csize = [rows, cols]

#     # calc
#     chunksize = csize[0] * csize[1]
#     # total cols and rows for adding border to slices
#     tnrows = nrows + bordersize
#     tncols = ncols + bordersize

#     # make a zero array of chunksize
#     # add border
#     mchunk = np.zeros((tnrows, tncols, chunksize))
#     mchunk[:, :, :numim] = mx
#     mchunk[:-1, :-1, :numim] = m

#     # combine images

#     flatmap = np.zeros((tnrows * rows, tncols * cols))
#     ci = 0
#     ri = 0
#     for plane in range(chunksize):
#         flatmap[ri : ri + tnrows, ci : ci + tncols] = mchunk[:, :, plane]
#         ri += tnrows
#         # if we have filled rows rows, change column
#         # and reset r
#         if plane != 0 and ri == tnrows * rows:
#             ci += tncols
#             ri = 0

#     return flatmap


# def reorder_rdm(utv, newOrder):
#     ax, bx = np.ix_(newOrder, newOrder)
#     newOrderRDM = squareform(utv)[ax, bx]
#     return squareform(newOrderRDM, "tovector", 0)
