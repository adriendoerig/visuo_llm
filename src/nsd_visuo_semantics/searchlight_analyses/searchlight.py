import numpy as np
from scipy.spatial.distance import cdist


class RSASearchLight:
    def __init__(self, mask, radius=1, thr=0.7, verbose=False):
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
        self.radius = radius
        self.thr = thr
        print("finding centers")
        self.centers = self._findCenters()
        print("finding center indices")
        self.centerIndices = self._findCenterIndices()
        print("finding all sphere indices")
        self.allIndices = self._allSphereIndices()

    def _findCenters(self):
        """
        Find all indices from centers with usable voxels over threshold.
        """
        # make centers a list of 3-tuple coords
        centers = zip(*np.nonzero(self.mask))
        good_center = []
        for center in centers:
            ind = self.searchlightInd(center)
            this_mask_mean = 0
            for i in range(ind.shape[0]):
                this_mask_mean += self.mask[tuple(ind[i,:])]/ind.shape[0]
            if this_mask_mean >= self.thr:
                good_center.append(center)
            # ind = self.searchlightInd(center)
            # try:
            #     if self.mask[ind].mean() >= self.thr:
            #         good_center.append(center)
            # except IndexError:
            #     import pdb; pdb.set_trace()
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
            # allIndices.append(np.ravel_multi_index(np.array(ind), dims))
            allIndices.append(np.ravel_multi_index(np.array(ind.T), dims))
        print("\n")
        return np.array(allIndices, dtype=object)

    def searchlightInd(self, center):
        """Return indices for searchlight where distance < radius

        Parameters:
            center: point around which to make searchlight sphere.
        """
        center = np.array(center)
        shape = self.mask.shape
        cx, cy, cz = center
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

        # return data[distance < self.radius].T.tolist()
        return data[distance < self.radius]
