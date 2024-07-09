from itertools import permutations
import numpy as np


def give_vector_pos(list_index, m):
    # takes a pair of indices in the upper tri RDM, and returns the 1D pdist vector index.

    # permutation of length 2
    perm = permutations(list_index, 2)
    upper_list_index = []
    # select valid purmuatations
    for i in perm:
        if i[0] < i[1]:
            upper_list_index.append(i)
    vector_pos = []
    # Get the vector positions of the tuples
    for j in upper_list_index:
        if j[0] == 0:
            vector = j[1] - j[0] - 1
        else:
            vector = (int((m*j[0]) - (j[0]*(j[0]+1)/2) + j[1]-j[0]) - 1)
        vector_pos.append(vector)
    # Return the sorted vector positions of all possible tuples
    vector_pos.sort()
    return vector_pos


class BatchGen:

    """
    Each instance is defined by batch size, RDM and list of image indices.
    nextBatch() returns a new batch , associated images and associated labels indefinately as a tuple (RDM_list, img_list, lbl_list)
    If RDM provided for class args is np array, will return np array as RDM_list.

    Given:
    batch_size = int #say : 100
    image_index_list = [] #say : range(1000,2000)
    RDM = vector (Upper Triangular numpy array) #say : np.random.rand(int((dim*(dim - 1))/2)) for dim = 2000
    epoch = int #say : 5

    Example: To get the next batch.
    B = BatchGen(batch_size, image_index_list, RDM)
    B.nextBatch()

    Example: To get all the batches for a certain number of epochs
    B = BatchGen(batch_size, image_index_list, RDM)
    total_calls = num_batches_needed(len(image_index_list),batch_size) * epoch
    for i in range(total_calls):
        print(B.nextBatch())

    Sanity Check(Batch - wise): Returns "ok" if everything is alright and "Error" otherwise.

    # betas_all = np.load(op.join(betas_dir, betas_template.format(roi)), allow_pickle=True)
    from restrict_rdms_for_test.py (To check only first 1000)

    B = BatchGen(batch_size, image_index_list, RDM)
    next_batch = B.nextBatch()
    # Calculate the RDMs only for the image indices used to compute next_batch
    image_index_list = B.batch_list[B.current_batch - 1]  # Image indices
    image_index_list.sort()  # Sort for easy comparison
    beta_list = betas_all[:, image_index_list].T  # Get betas only for the selected Image indices
    rdm_list = upper_tri_indexing(cdist(beta_list, beta_list)).astype(np.float32)  # Calculate the Distance Matrix
    if (np.array_equal(rdm_list, next_batch)):
        print("ok")
    else:
        print("Error")

    """

    def __init__(self, RDM, all_conditions):
        self.RDM = RDM
        self.all_conditions = all_conditions

    def index_rdms(self, condition_subset):
        """No args, returns list of RDMs, images, and labels for processing. Each sequential call returns the next batch.
        If first batch in an epoch, will call listIndex to shuffle the population before generating the batch data.
        If RDM provided for class args is np array, will return np array.

        Returns list of RDMs. If given single RDM, returns list with single index, otherwise returns list in
        order of RDMs input.
        """

        # if given list, for each segmented rdm in the list, return the rdm as a list
        # if given single RDM, return single segmented rdm in a list with one index

        RDM_list = []

        if type(self.RDM) is list:  # we have RDMs from many models.
            RDM_array = np.asarray(self.RDM)
            RDM_index = give_vector_pos(condition_subset, len(self.all_conditions))

            RDM_array = RDM_array[:, RDM_index]
            RDM_list = [rdm for rdm in RDM_array]
        else:
            RDM_index = self.RDM[give_vector_pos(condition_subset, len(self.all_conditions))]
            RDM_list.append(RDM_index)

        # Return a subsetted RDM vector
        return RDM_list
