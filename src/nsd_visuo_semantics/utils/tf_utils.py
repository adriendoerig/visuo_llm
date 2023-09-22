"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
import tensorflow as tf
from scipy.stats import rankdata


def upper_tri_indexing_tf(rdm):
    # rdm is shape [b,n,n]
    batch_dim = rdm.shape[0]
    ones = tf.ones_like(rdm)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    #  Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    upper_triangular_flat = tf.boolean_mask(rdm, mask)
    return tf.reshape(upper_triangular_flat, (batch_dim, -1))


def compute_rdm(x):
    """compute rdm from a single pattern

    Args:
        x ([matrix]): conditions x features

    Returns:
        rdm [matrix]: conditions x conditions
    """
    ma = x - tf.reduce_mean(x, axis=1, keepdims=True)
    ma = ma / tf.sqrt(tf.einsum("ij,ij->i", ma, ma))[:, None]
    rdm = 1 - tf.einsum("ik,jk", ma, ma)
    return upper_tri_indexing_tf(rdm)


def compute_rdm_batch(x):
    """compute RDMs over a batch of patterns

    Args:
        x ([matrix]): samples by conditions by features
    Returns:
        rdm ([matrix]): samples x pairwise comparisons of conditions

    say we have 50 patterns: e.g. 50 searchlight centers. each
    searchlight center has 100 conditions by 300 voxels. For every
    sphere what we want to obtain is the 50 different 4950x4950 distance
    matrices.

    e.g.

    a = np.random.rand(50, 100, 300)
    b = a[0, :, :]

    rdms = compute_rdm_batch(a)
    rdm = compute_rdm(b)
    np.testing.assert_array_almost_equal(rdm, rdms[0,:,:])

    """
    ma = x - tf.reduce_mean(x, axis=2, keepdims=True)
    ma = ma / tf.sqrt(tf.einsum("...ij,...ij->...i", ma, ma))[:, :, None]
    rdms = 1 - tf.einsum("bik,bjk->bij", ma, ma)

    return upper_tri_indexing_tf(rdms)


def sort_spheres(sphere_indices):
    """sorts searchilght spheres by n_voxels in the sphere

    Args:
        sphere_indices ([type]): [description]

    Returns:
        [type]: [description]
    """
    all_sizes = np.asarray([len(indices) for indices in sphere_indices])
    unique_sizes = np.unique(all_sizes)

    if len(unique_sizes) > 1:
        # sort and find the sorting indices
        sized_indices = [np.where(all_sizes == size)[0] for size in unique_sizes]
    else:
        sized_indices = list(range(len(all_sizes)))

    return sized_indices


def chunking(vect, num, chunknum=None):
    """chunking
    Input:
        <vect> is a array
        <num> is desired length of a chunk
        <chunknum> is chunk number desired (here we use a 1-based
              indexing, i.e. you may want the first chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning indices associated with
        this chunk in <xbegin>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect) / num))
        f = []
        for point in range(nchunk):
            f.append(vect[point * num : np.min((len(vect), int((point + 1) * num)))])

        return np.asarray(f, dtype=object)
    else:
        f = chunking(vect, num)
        # double check that these behave like in matlab (xbegin)
        xbegin = (chunknum - 1) * num + 1
        return np.asarray(f[num - 1]), xbegin


# def rank(x, axis=1):
#     """[summary]

#     Args:
#         x ([tensor]): [description]
#         axis (int, optional): [description]. Defaults to 0.
#     """
#     return rankdata(x, axis=axis)


def get_rank(y_pred):
    rank = (
        tf.argsort(tf.argsort(y_pred, axis=-1, direction="ASCENDING"), axis=-1)
        + 1
    )  # +1 to get the rank starting in 1 instead of 0
    rank = tf.dtypes.cast(rank, tf.float32)
    return rank


def rank_slow(x):
    """[summary]

    Args:
        x ([tensor]): [description]
        axis (int, optional): [description]. Defaults to 1.
    """
    return np.apply_along_axis(rankdata, 1, x)


def corr_rdms_rank_slow(x, y, n_inchunks=100000):
    y = tf.convert_to_tensor(y)
    y = get_rank(y)
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)
    y /= tf.sqrt(tf.einsum("ij,ij->i", y, y))[:, None]

    chunks = chunking(x, n_inchunks)
    corrs = []
    for chunki in chunks:
        chunk = tf.convert_to_tensor(chunki)
        chunk = get_rank(chunk)
        chunk = chunk - tf.reduce_mean(chunk, axis=1, keepdims=True)
        chunk /= tf.sqrt(tf.einsum("ij,ij->i", chunk, chunk))[:, None]
        corrs.append(tf.einsum("ik,jk", chunk, y))

    return np.asarray(tf.concat(corrs, axis=0))


# def corr_rdms_rank(x, y, n_inchunks=100000):
#     y = tf.convert_to_tensor(y)
#     y = tf.py_function(func=rank, inp=[y], Tout=[tf.float32])[0]
#     y = y - tf.reduce_mean(y, axis=1, keepdims=True)
#     y /= tf.sqrt(tf.einsum('ij,ij->i', y, y))[:, None]

#     chunks = chunking(x, n_inchunks)
#     corrs = []
#     for chunki in chunks:
#         chunk = tf.convert_to_tensor(chunki)
#         chunk = tf.py_function(func=rank, inp=[chunk], Tout=[tf.float32])[0]
#         chunk = chunk - tf.reduce_mean(chunk, axis=1, keepdims=True)
#         chunk /= tf.sqrt(tf.einsum('ij,ij->i', chunk, chunk))[:, None]
#         corrs.append(tf.einsum('ik,jk', chunk, y))

#     return np.asarray(tf.concat(corrs, axis=0))


def corr_rdms(x, y, n_inchunks=100000):
    """corr_rdms useful for correlation of RDMs (e.g. in multimodal RSA fusion)
        where you correlate two ndim RDM stacks.

    Args:
        x [array]: e.g. the fmri searchlight RDMs (ncenters x npairs)
        y [array]: e.g. the EEG time RDMs (ntimes x npairs)

    Returns:
        [type]: correlations between X and Y, of shape dim0 of X by dim0 of Y
    """
    y = tf.convert_to_tensor(y)
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)
    y /= tf.sqrt(tf.einsum("ij,ij->i", y, y))[:, None]

    chunks = chunking(x, n_inchunks)
    corrs = []
    for chunki in chunks:
        chunk = tf.convert_to_tensor(chunki)
        chunk = chunk - tf.reduce_mean(chunk, axis=1, keepdims=True)
        chunk /= tf.sqrt(tf.einsum("ij,ij->i", chunk, chunk))[:, None]
        corrs.append(tf.einsum("ik,jk", chunk, y))

    return np.asarray(tf.concat(corrs, axis=0))
