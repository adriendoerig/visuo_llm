import numpy as np
import tensorflow as tf

from nsd_visuo_semantics.utils.tf_utils import chunking as ch
from nsd_visuo_semantics.utils.tf_utils import compute_rdm_batch


def tf_searchlight(betas_sampled, indices, sorted_indices, batch_size=50):
    """[volumetric searchlight on GPU]

    Args:
        betas_sampled ([array]): volume of beta conditions
                                 (x, y, z, n_conditions).
        indices ([array]):       list of searchlight sphere indices
        sorted_indices ([type]): list of searchlight spheres,
                                 sorted by n_voxels.
        batch_size (int, opt):   how large a chunk to send to GPU.
                                 Defaults to 50.

    Returns:
        rdms [array]:
    """
    x, y, z, n_conditions = betas_sampled.shape

    # unroll the betas
    betas_unrolled = tf.convert_to_tensor(betas_sampled.reshape((-1, n_conditions)))

    # now chunk the betas given the sphere sizes and batch_size
    # and for each chunk, compute a batch of rdm computations.
    # start_time = time.time()
    rdms = []
    for i, ind in enumerate(sorted_indices):
        chunks = ch(ind, batch_size)
        for c, chunk in enumerate(chunks):
            # line below is from a previous version where we were not using sorted indices
            # here, we DO use them. i.e., we get a list of sorted_indices, where all
            # indices have the same n_voxels (this occurs because we remove NaN voxels).
            # we go over these lsits one by one, and compute their RDMs
            # sl = tf.gather(betas_unrolled, np.stack(indices[chunk.astype(np.int32)]))
            these_indices = [indices[i] for i in chunk]
            sl = tf.gather(betas_unrolled, np.stack(these_indices))
            t = tf.transpose(sl, perm=[0, 2, 1])
            rdm = np.asarray(compute_rdm_batch(t))
            rdms.append(rdm)

    # elapsed_time = time.time() - start_time
    # print(
    #     'elapsedtime: ',
    #     f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    # )

    # return to CPU
    rdms_np = np.vstack(rdms)

    return rdms_np
