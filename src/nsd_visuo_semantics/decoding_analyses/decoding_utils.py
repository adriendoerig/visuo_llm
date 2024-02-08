import os
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import tensorflow_probability as tfp


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


def restore_nan_dims(x, drop_rows_idx, axis=0):
    '''Add back the removed dimensions (if any were removed, see remove_nan_dims()).'''

    nan_idx_offset = 0
    prev = -1
    for i in range(drop_rows_idx.shape[0]):
        x = np.insert(x, drop_rows_idx[i - nan_idx_offset], np.nan, axis=axis)
        if i == prev:
            nan_idx_offset += 1
        else:
            nan_idx_offset = 0
        prev += 1
        
    return x


def pairwise_corr(x, y, batch_size=-1):
    '''
    :param x: [n_pairs_to_correlate, n_dims_to_correlate]
    :param y:  [n_pairs_to_correlate, n_dims_to_correlate]
    :param batch_size: compute correlations in batches of this size. if -1 -> do all in one go
    :return: [n_pairs_to_correlate] correlation values
    '''
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if batch_size == -1:
        return tfp.stats.correlation(x, y, sample_axis=0, event_axis=None).numpy()
    else:
        corrs_out = []
        batch_indices = np.array_split(np.arange(x.shape[0]), batch_size)
        for this_batch in batch_indices:
            corrs_out.append(tfp.stats.correlation(x[this_batch], y[this_batch], sample_axis=1, event_axis=None).numpy())
        return np.hstack(corrs_out)
    

def get_gcc_nearest_neighbour(embedding, n_neighbours=1, gcc_dir='/share/klab/datasets/google_conceptual_captions', 
                              METRIC='correlatiuon', norm_mu_sigma=[0.0, 1.0]):

    print(f"Getting nearest neighbour (distance = {METRIC}) in GCC...")

    if embedding.ndim == 1:
        embedding = embedding[np.newaxis, :]

    assert embedding.shape == (1, 768), "Embedding should be a 1x768 vector"

    lookup_sentences_path = os.path.join(
        gcc_dir,
        "conceptual_captions_{}.tsv",
    )
    lookup_embeddings_path = os.path.join(
        gcc_dir,
        "conceptual_captions_mpnet_{}.npy",
    )
    lookup_datasets = [
        "train",
        "val",
    ]  # we can choose to use either the gcc train, val, or both for the lookup

    lookup_embeddings = None
    for d in lookup_datasets:
        # there is a train and val set in the gcc captions, we load the ones chosen by the user (concatenating them)
        if lookup_embeddings is None:
            lookup_embeddings = np.load(lookup_embeddings_path.format(d))
            df = pd.read_csv(lookup_sentences_path.format(d), sep="\t", header=None, names=["sent", "url"])
            lookup_sentences = df["sent"].to_list()
        else:
            lookup_embeddings = np.vstack([lookup_embeddings, np.load(lookup_embeddings_path.format(d))])
            df = pd.read_csv(lookup_sentences_path.format(d), sep="\t", header=None, names=["sent", "url"])
            lookup_sentences += df["sent"].to_list()

    if norm_mu_sigma[0] == 'zscore':
        norm_mu_sigma = [lookup_embeddings.mean(axis=0), lookup_embeddings.std(axis=0)]
    lookup_embeddings = (lookup_embeddings - norm_mu_sigma[0]) / norm_mu_sigma[1]

    lookup_distances = cdist(embedding, lookup_embeddings, metric=METRIC).squeeze()
    indices_of_min_values = np.argsort(lookup_distances)[:n_neighbours]
    pred_sentences = [lookup_sentences[i] for i in indices_of_min_values]
    return pred_sentences