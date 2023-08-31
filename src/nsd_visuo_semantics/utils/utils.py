from math import ceil, floor

import numpy as np
from fracridge import FracRidgeRegressorCV
from scipy.optimize import nnls
from scipy.spatial.distance import cdist, squareform


class NSD_parsing:
    def __init__(self):
        self.subject_string = (
            "the subject number you want to analyse "
            + "[valid options 1:8]"
            + "e.g: python nsd_searchlight.py --subject=1"
        )
        self.boot_string = (
            "the number of bootstraps to perform."
            + "e.g: python nsd_searchlight.py --subject=1 --nboot=100"
        )
        self.rad_string = (
            "the searchlight radius to use. e.g:"
            + "python nsd_searchlight.py --subject=1"
            + "--nboot=100 --nsamples=100 --radius=3"
        )
        self.session_string = (
            "the number of sessions completed for"
            + "the current subject. e.g: python nsd_searchlight.py"
            + "--subject=1 --nboot=100 --nsamples=100 --radius=3"
            + "--nsessions=40"
        )
        self.job_string = (
            "the number of cpus available. we use parallel computing"
            + "when we can. e.g: python nsd_searchlight.py --subject=1"
            + "--nboot=100 --nsamples=100 --radius=3 --nsessions=40"
            + "--njobs=2"
        )
        self.chunk_string = (
            "the number of centers in a chunk. e.g: python"
            + "nsd_searchlight.py"
            + "--subject=1 --nboot=100 --nsamples=100 --radius=3"
            + "--nsessions=40"
            + "--njobs=2 --ninchunks=250"
        )
        self.nsamples_string = (
            "the number of conditions to sample. e.g:"
            + "python nsd_searchlight.py --subject=1"
            + "--nboot=100 --nsamples=10000"
        )
        self.layer_string = (
            "the rcnn ecoset relu layer number (0-based):"
            + "python nsd_searchlight.py --subject=1"
            + "--nboot=100 --nsamples=10000 --layer=0"
        )


# def mds(utv, pos=None, n_jobs=1):

#     rdm = squareform(utv)
#     seed = np.random.RandomState(seed=3)
#     mds = MDS(n_components=2, max_iter=100, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=n_jobs)
#     pos = mds.fit_transform(rdm, init=pos)

#     return pos

category_dict = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 1,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 1,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 1,
    "21": 0,
    "22": 0,
    "23": 0,
    "24": 0,
    "25": 1,
    "26": 0,
    "27": 0,
    "28": 1,
    "29": 0,
    "30": 1,
    "31": 0,
    "32": 0,
    "33": 0,
    "34": 1,
    "35": 0,
    "36": 0,
    "37": 1,
    "38": 0,
    "39": 0,
    "40": 0,
    "41": 0,
    "42": 0,
    "43": 0,
    "44": 0,
    "45": 1,
    "46": 0,
    "47": 0,
    "48": 0,
    "49": 1,
    "50": 0,
    "51": 0,
    "52": 0,
    "53": 0,
    "54": 0,
    "55": 0,
    "56": 1,
    "57": 0,
    "58": 0,
    "59": 0,
    "60": 1,
    "61": 0,
    "62": 0,
    "63": 0,
    "64": 0,
    "65": 0,
    "66": 0,
    "67": 0,
    "68": 0,
    "69": 0,
    "70": 0,
    "71": 0,
    "72": 0,
    "73": 0,
    "74": 0,
    "75": 0,
    "76": 0,
    "77": 0,
    "78": 0,
    "79": 1,
}


def fit_rdms(
    model_rdms, brain_rdms, intercept=True, fit_method="fracridge", n_jobs=8
):
    weights = []

    # find problematic RDMs with nans
    nanrdms = np.flatnonzero(np.isnan(np.sum(brain_rdms, axis=1)))

    if fit_method == "nnls":
        if intercept:
            model_rdms = np.c_[model_rdms, np.ones(model_rdms.shape[0])]
        n_weights = model_rdms.shape[1]
        for rdm_i, rdm in enumerate(brain_rdms):
            if rdm_i in nanrdms:
                weights.append(np.zeros(n_weights))
            else:
                weights.append(nnls(model_rdms, rdm)[0])

        return np.asarray(weights).astype(np.float32)

    elif fit_method == "fracridge":
        frr = FracRidgeRegressorCV(jit=True, fit_intercept=True, n_jobs=n_jobs)
        frr.fit(model_rdms, brain_rdms)
        raise Exception("Not finished implementing")


def compute_rdm(measurements):
    """compute_rdm computes a correlation distance based rdm from
       a measurements array

    Args:
        measurements (2D array): conditions by features

    Returns:
        correlation distance vector
    """
    row, col = np.triu_indices(measurements.shape[0], 1)
    measurements = measurements - measurements.mean(axis=1, keepdims=True)
    measurements /= np.sqrt(np.einsum("ij,ij->i", measurements, measurements))[
        :, None
    ]

    return 1 - np.einsum("ik,jk", measurements, measurements)[row, col]


def compute_rdm_cosine(measurements):
    """compute_rdm computes a cosine distance based rdm from
       a measurements array

    Args:
        measurements (2D array): conditions by features

    Returns:
        correlation distance vector
    """
    row, col = np.triu_indices(measurements.shape[0], 1)
    measurements = measurements / np.linalg.norm(measurements, axis=1)[:, None]
    return 1 - np.einsum("ik,jk->ij", measurements, measurements)[row, col]


def make_project_matrix(X):
    """Calculates a projection matrix

    Args:
        X (array): design matrix

    Returns:
        array: Projection matrix size of X.shape[0] x X.shape[0]
    """
    n_items = X.shape[0]
    X = np.c_[X, np.ones(n_items)]
    X = np.mat(X)
    return np.eye(X.shape[0]) - (X * (np.linalg.inv(X.T * X) * X.T))


def mask_vec_condition(condition, n_conditions):
    """find utv indices for a subset of conditions

    Args:
        condition ([type]): [description]
        n_conditions ([type]): [description]

    Returns:
        [type]: [description]
    """
    ax, bx = np.ix_(condition, condition)
    mask_rdm = np.zeros((n_conditions, n_conditions))
    mask_rdm[ax, bx] = 1
    mask_rdm[np.eye(n_conditions) == 1] = 0
    return squareform(mask_rdm, "tovector") == 1


def upper_tri_indexing(RDM):
    """upper_tri_indexing returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


def get_distance_cdist(data, ind):
    """get_distance_cdist returns the correlation distance
       across condition patterns in X
       get_distance_cdist uses scipy's cdist method

    Args:
        data (array): conditions x all channels.
        ind (vector): subspace of voxels for that sphere.

    Returns:
        UTV: pairwise distances between condition patterns in X
             (in upper triangular vector form)
    """
    ind = np.array(ind)
    X = data[ind, :].T
    return upper_tri_indexing(cdist(X, X, "correlation"))


def get_distance(data, ind):
    """get_distance returns the correlation distance
       across condition patterns in X
       get_distance uses numpy's einsum

    Args:
        data (array): conditions x all channels.
        ind (vector): subspace of voxels for that sphere.

    Returns:
        UTV: pairwise distances between condition patterns in X
             (in upper triangular vector form)
    """
    ind = np.array(ind)
    X = np.array(data[ind, :]).T
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]

    return 1 - upper_tri_indexing(np.einsum("ik,jk", X, X))


def corr_rdms(X, Y):
    """corr_rdms useful for correlation of RDMs (e.g. in multimodal RSA fusion)
        where you correlate two ndim RDM stacks.

    Args:
        X [array]: e.g. the fmri searchlight RDMs (ncenters x npairs)
        Y [array]: e.g. the EEG time RDMs (ntimes x npairs)

    Returns:
        [type]: correlations between X and Y, of shape dim0 of X by dim0 of Y
    """
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.sqrt(np.einsum("ij,ij->i", Y, Y))[:, None]

    return np.einsum("ik,jk", X, Y)


# def run_per_center(data, c, labels):
#     svc = svm.LinearSVC()
#     clf = make_pipeline(StandardScaler(), svc)

#     # Get indices from center
#     ind = np.array(c)
#     X = np.array(data[ind, :]).T
#     # pdb.set_trace()
#     score = np.mean(cross_val_score(clf, X, labels, cv=9, n_jobs=1))
#     return score


def chunking(vect, num, chunknum=None):
    """chunking
    Input:
        <vect> is a array
        <num> is desired length of a chunk
        <chunknum> is chunk number desired (here we use a 1-based
              indexing, i.e. you may want the frist chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning and ending indices associated with
        this chunk in <xbegin> and <xend>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))


        # do in chunks
        chunks = chunking(
            list(range(mflat.shape[1])), int(np.ceil(mflat.shape[1]/numchunks)))

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect) / num))
        f = []
        for point in range(nchunk):
            f.append(
                vect[point * num : np.min((len(vect), int((point + 1) * num)))]
            )

        return np.asarray(f)
    else:
        f = chunking(vect, num)
        # double check that these behave like in matlab (xbegin)
        xbegin = (chunknum - 1) * num + 1
        return np.asarray(f[num - 1]), xbegin, xbegin + len(f[num - 1]) - 1


def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res


def sample_conditions(conditions, n_samples, replace=False):
    unique_conditions = np.unique(conditions)

    choices = np.random.choice(unique_conditions, n_samples, replace=replace)

    conditions_bool = np.any(
        np.array([conditions == v for v in choices]), axis=0
    )

    return conditions_bool


def average_over_conditions(data, conditions, conditions_to_avg):
    lookup = np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims == 2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape
        avg_data = np.empty((x, y, z, n_conds))

    for j, x in enumerate(lookup):
        conditions_bool = conditions == x
        if n_dims == 2:
            if np.sum(conditions_bool) == 0:
                break
            # print((j, np.sum(conditions_bool)))
            sliced = data[:, conditions_bool]

            avg_data[:, j] = np.nanmean(sliced, axis=1)
        else:
            avg_data[:, :, :, j] = np.nanmean(
                data[:, :, :, conditions_bool], axis=3
            )

    return avg_data


def makeimagestack(m):
    """
    def makeimagestack(m)

    <m> is a 3D matrix.  if more than 3D, we reshape to be 3D.
    we automatically convert to double format for the purposes of this method.
    try to make as square as possible
    (e.g. for 16 images, we would use [4 4]).
    find the minimum possible to fit all the images in.
    """

    bordersize = 1

    # calc
    nrows, ncols, numim = m.shape
    mx = np.nanmax(m.ravel())

    # calculate csize

    rows = floor(np.sqrt(numim))
    cols = ceil(numim / rows)
    csize = [rows, cols]

    # calc
    chunksize = csize[0] * csize[1]
    # total cols and rows for adding border to slices
    tnrows = nrows + bordersize
    tncols = ncols + bordersize

    # make a zero array of chunksize
    # add border
    mchunk = np.zeros((tnrows, tncols, chunksize))
    mchunk[:, :, :numim] = mx
    mchunk[:-1, :-1, :numim] = m

    # combine images

    flatmap = np.zeros((tnrows * rows, tncols * cols))
    ci = 0
    ri = 0
    for plane in range(chunksize):
        flatmap[ri : ri + tnrows, ci : ci + tncols] = mchunk[:, :, plane]
        ri += tnrows
        # if we have filled rows rows, change column
        # and reset r
        if plane != 0 and ri == tnrows * rows:
            ci += tncols
            ri = 0

    return flatmap


def reorder_rdm(utv, newOrder):
    ax, bx = np.ix_(newOrder, newOrder)
    newOrderRDM = squareform(utv)[ax, bx]
    return squareform(newOrderRDM, "tovector", 0)
