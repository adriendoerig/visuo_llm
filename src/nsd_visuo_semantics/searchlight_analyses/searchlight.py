import os

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm


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
    return pdist(X, metric="correlation")


def fit_rsa(data, allIndices, n_jobs=1, verbose=True, save_file=None):
    """
    Fit Searchlight for RDM
    Parameters:
        data:       4D numpy array - (x, y, z, condition vols)
        metric :    str or callable, optional
                    The distance metric to use.
                    If a string, the distance function can be
                    'braycurtis', 'canberra', 'chebyshev',
                    'cityblock', 'correlation', 'cosine', 'dice',
                    'euclidean', 'hamming', 'jaccard', 'kulsinski',
                    'mahalanobis', 'matching', 'minkowski',
                    'rogerstanimoto', 'russellrao',
                    'seuclidean', 'sokalmichener', 'sokalsneath',
                    'sqeuclidean', 'wminkowski', 'yule'.
    """
    print("Running searchlight RSA")
    # reshape the data to squish the first three dimensions
    x, y, z, nobjects = data.shape

    # now the first dimension of data is directly indexable by
    # subspace index of the searchlight centers
    data = data.reshape((x * y * z, nobjects))

    if save_file is not None:
        if os.path.exists(save_file):
            # load pre-computed rdms
            print(f"load pre-computed rdms from file:\n\t\t... {save_file}")
            rdms = np.load(save_file, allow_pickle=True)
        else:
            # compute
            if n_jobs == 1:
                # single core
                if verbose is True:
                    rdms = np.asarray(
                        [
                            get_distance(data, x)
                            for x in tqdm(
                                allIndices,
                                desc="spheres",
                                ascii=True,
                                ncols=60,
                            )
                        ]
                    )
                else:
                    rdms = np.asarray(
                        [get_distance(data, x) for x in allIndices]
                    )
            else:
                # parallel
                if verbose is True:
                    rdms = Parallel(n_jobs=n_jobs)(
                        delayed(get_distance)(data, x)
                        for x in tqdm(
                            allIndices, desc="spheres", ascii=True, ncols=60
                        )
                    )
                else:
                    rdms = Parallel(n_jobs=n_jobs)(
                        delayed(get_distance)(data, x) for x in allIndices
                    )
                    rdms = np.asarray(rdms)

            print(f"saving searchlight rdms to file:\n\t\t... {save_file}")
            np.save(save_file, rdms)
    else:
        # run but don't save
        if n_jobs == 1:
            # single core
            if verbose is True:
                rdms = np.asarray(
                    [
                        get_distance(data, x)
                        for x in tqdm(
                            allIndices, desc="spheres", ascii=True, ncols=60
                        )
                    ]
                )
            else:
                rdms = np.asarray([get_distance(data, x) for x in allIndices])
        else:
            # parallel
            if verbose is True:
                rdms = Parallel(n_jobs=n_jobs)(
                    delayed(get_distance)(data, x)
                    for x in tqdm(
                        allIndices, desc="spheres", ascii=True, ncols=60
                    )
                )
            else:
                rdms = Parallel(n_jobs=n_jobs)(
                    delayed(get_distance)(data, x) for x in allIndices
                )
                rdms = np.asarray(rdms)

    return rdms


class RSASearchLight:
    def __init__(self, mask, radius=1, thr=0.7, njobs=1, verbose=False):
        """
        Parameters:
            mask:    3d spatial mask (of usable voxels set to 1)
            radius:  radius around each center (in voxels)
            thr :    proportion of usable voxels necessary
                     thr = 1 means we don't accept centers with voxels outside
                     the brain
        """
        self.verbose = verbose
        self.mask = mask
        self.njobs = njobs
        self.radius = radius
        self.thr = thr
        print("finding centers")
        self.centers = self._findCenters()
        print("finding center indices")
        self.centerIndices = self._findCenterIndices()
        print("finding all sphere indices")
        self.allIndices = self._allSphereIndices()
        self.NaNs = []

    def _findCenters(self):
        """
        Find all indices from centers with usable voxels over threshold.
        """
        # make centers a list of 3-tuple coords
        centers = zip(*np.nonzero(self.mask))
        good_center = []
        for center in centers:
            ind = self.searchlightInd(center)
            if self.mask[ind].mean() >= self.thr:
                good_center.append(center)
        return np.array(good_center)

    def _findCenterIndices(self):
        """
        Find all subspace indices from centers
        """
        centerIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i / len(self.centers) * 100
            if i % 50 == 0 and self.verbose is True:
                print(
                    "Converting voxel coordinates of centers to subspace"
                    f"indices {n_done:.0f}% done!",
                    end="\r",
                )
            centerIndices.append(np.ravel_multi_index(np.array(cen), dims))
        print("\n")
        return np.array(centerIndices)

    def _allSphereIndices(self):
        allIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i / len(self.centers) * 100
            if i % 50 == 0 and self.verbose is True:
                print(f"Finding SearchLights {n_done:.0f}% done!", end="\r")

            # Get indices from center
            ind = np.array(self.searchlightInd(cen))
            allIndices.append(np.ravel_multi_index(np.array(ind), dims))
        print("\n")
        return allIndices

    def searchlightInd(self, center):
        """Return indices for searchlight where distance < radius

        Parameters:
            center: point around which to make searchlight sphere
        Sets RDM variable to:
            numpy array of shape (3, N_comparisons) for subsetting data
        """
        center = np.array(center)
        shape = self.mask.shape
        cx, cy, cz = np.array(center)
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])

        # First mask the obvious points
        # - may actually slow down your calculation depending.
        x = x[abs(x - cx) < self.radius]
        y = y[abs(y - cy) < self.radius]
        z = z[abs(z - cz) < self.radius]

        # Generate grid of points
        X, Y, Z = np.meshgrid(x, y, z)
        data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        distance = cdist(data, center.reshape(1, -1), "euclidean").ravel()

        return data[distance < self.radius].T.tolist()

    def checkNaNs(X):
        """
        TODO - this function
        """
        # nans = np.all(np.isnan(X), axis=0)[0]
        # return X[:,~nans]
