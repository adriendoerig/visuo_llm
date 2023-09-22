import numpy as np

def remove_inert_embedding_dims(embeddings, cutoff=1e-20):
    """Remove embedding dimensions whose mean is < embedding_mean*cutoff. This is done because some dims may contain
    only minuscule numbers (e.g. all values smaller than 1e-33), which are pointless and lead to NaNs in fracridge.
    We keep the mean value of these dims, so we can then use this to get the full embedding back.
    """

    mean = embeddings.mean()
    mean_cols = embeddings.mean(axis=0)
    drop_dims_bool = abs(mean_cols) < abs(mean) * cutoff
    keep_dims_bool = abs(mean_cols) > abs(mean) * cutoff

    drop_dims_idx = np.where(drop_dims_bool)[0]
    drop_dims_avgs = [np.mean(embeddings[:, i]) for i in drop_dims_idx]
    embeddings = embeddings[:, keep_dims_bool]

    return embeddings, drop_dims_idx, drop_dims_avgs


def restore_inert_embedding_dims(embeddings, drop_dims_idx, drop_dims_avgs):
    """Add back the removed embedding dimensions (if any were removed, see remove_inert_embedding_dims())."""

    out = None
    prev_idx = 0
    for i, idx in enumerate(drop_dims_idx):
        insert_dim = np.expand_dims(
            drop_dims_avgs[i] * np.ones_like(embeddings[:, idx]), axis=-1
        )
        if out is None:
            if idx == 0:
                out = insert_dim
            else:
                out = np.hstack([embeddings[:, prev_idx:idx], insert_dim])
        else:
            out = np.hstack([out, embeddings[:, prev_idx:idx], insert_dim])
        if idx == drop_dims_idx[-1]:
            out = np.hstack([out, embeddings[:, idx:]])
        prev_idx = idx

    return out


def restore_nan_dims(x, drop_rows_idx):
    '''Add back the removed dimensions (if any were removed, see remove_nan_dims()).'''

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