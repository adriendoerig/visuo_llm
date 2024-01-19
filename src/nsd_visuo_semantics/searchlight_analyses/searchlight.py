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
        n_centers = len(list(zip(*np.nonzero(self.mask))))
        good_center = []
        for i, center in enumerate(centers):
            n_done = i/n_centers * 100
            if i % 50 == 0 and self.verbose is True:
                print(f"Finding centers with at least {self.thr*100}% voxels in brain {n_done:.0f}%", end="\r")
            # get all voxel indices for this center (ind, which is a list of
            # n_voxels_in_brain_mask_for_this_center x 3(x,y,z)), and also get
            # the fraction of voxels in the brain mask for this center
            ind, frac_good_ind = self.searchlightInd(center)
            if frac_good_ind >= self.thr:
                good_center.append(center)

        return np.array(good_center)

    def _findCenterIndices(self):
        """
        Find all subspace indices from centers
        """
        centerIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i/len(self.centers) * 100
            if i % 50 == 0 and self.verbose is True:
                print(f"Converting voxel coordinates of centers to 1d indices {n_done:.0f}%", end="\r")
            centerIndices.append(np.ravel_multi_index(np.array(cen), dims))
        return np.array(centerIndices)  # [n_good_centers x 1(ravelled_index))]

    def _allSphereIndices(self):
        allIndices = []
        dims = self.mask.shape
        for i, cen in enumerate(self.centers):
            n_done = i / len(self.centers) * 100
            if i % 50 == 0 and self.verbose is True:
                print(f"Finding SearchLights {n_done:.0f}% done!", end="\r")
            # Get indices from center
            ind, frac_good_ind = self.searchlightInd(cen)
            # we append a list of n_voxels_in_brain_mask_for_this_center x 1(ravelled_index)
            allIndices.append(np.ravel_multi_index(np.array(ind).T, dims))        
        return allIndices

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
        data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # [n_vox, y,y,z]
        distance = cdist(data, center.reshape(1, -1), "euclidean").ravel()

        # we get a list of n_voxels x 3 indices(x,y,z)
        all_indices = data[distance < self.radius]
        # we only keep the ones that are in the mask (i.e. we throw away those outside the brain)
        # (so we get an n_voxels_in_brain x 3 indices(x,y,z))
        valid_indices = [i for i in all_indices if self.mask[tuple(i)] > 0]

        # count the fraction of indices we have inside the brain (will be used in
        # the findCenters function to decide whether to keep a center or not)
        fraction_good_indices = len(valid_indices) / all_indices.shape[0]

        return valid_indices, fraction_good_indices
