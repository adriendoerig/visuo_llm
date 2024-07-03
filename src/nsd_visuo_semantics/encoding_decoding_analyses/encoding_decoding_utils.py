import os
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
try:
    import tensorflow_probability as tfp
except:
    print("PRINTED FROM DECODING_UTILS: Tensorflow probability cannot be imported, some functions may not work.")


def make_515_embeddings(nsda, conditions_515, EMBEDDING_MODEL_NAME):
    '''get embeddings for the 515 special images. 
    This is used as the test set for the encoding/decoding models.'''
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    captions_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)
    dummy_embedding = get_embeddings(captions_515[0], embedding_model, EMBEDDING_MODEL_NAME)
    embedding_dim = dummy_embedding.shape[-1]
    embeddings_test = np.empty((515, embedding_dim))
    for i in range(len(captions_515)):
        embeddings_test[i] = np.mean(get_embeddings(captions_515[i], embedding_model, EMBEDDING_MODEL_NAME), axis=0)
    return embeddings_test


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
    

def load_gcc_embeddings(gcc_dir='/share/klab/datasets/google_conceptual_captions'):

    print("Loading GCC embeddings...")

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
    return lookup_sentences, lookup_embeddings


def get_gcc_nearest_neighbour(embedding, n_neighbours=1, 
                              lookup_sentences=None, lookup_embeddings=None,
                              gcc_dir='/share/klab/datasets/google_conceptual_captions', 
                              METRIC='cosine', norm_mu_sigma=[0.0, 1.0]):

    # print(f"Getting nearest neighbour (distance = {METRIC}) in GCC...")

    if embedding.ndim == 1:
        embedding = embedding[np.newaxis, :]

    assert embedding.shape[-1] == 768, "Embedding should be a 1x768 vector"

    if lookup_sentences is None or lookup_embeddings is None:
        lookup_sentences, lookup_embeddings = load_gcc_embeddings(gcc_dir=gcc_dir)

    if norm_mu_sigma == ['zscore']:
        print("Normalizing lookup embeddings with z-score...")
        norm_mu_sigma = [lookup_embeddings.mean(axis=0), lookup_embeddings.std(axis=0)]
    elif norm_mu_sigma == ['center']:
        print("Normalizing lookup embeddings by centering...")
        norm_mu_sigma = [lookup_embeddings.mean(axis=0), np.ones_like(lookup_embeddings.mean(axis=0))]
    lookup_embeddings = (lookup_embeddings - norm_mu_sigma[0]) / norm_mu_sigma[1]

    lookup_distances = cdist(embedding, lookup_embeddings, metric=METRIC)
    # print(f'Mean(lookup_distances) = {np.mean(lookup_distances)}, Std(lookup_distances) = {np.std(lookup_distances)}')
    pred_sentences = []
    pred_distances = []
    for i in range(lookup_distances.shape[0]):
        indices_of_min_values = np.argsort(lookup_distances[i])[:n_neighbours]
        pred_sentences.append([lookup_sentences[j] for j in indices_of_min_values])
        pred_distances.append([lookup_distances[i,j] for j in indices_of_min_values])
    
    return pred_sentences, pred_distances


sentences_zoo = {
    'people':
            ['Man with a beard smiling at the camera.',
             'Some children playing.',
             'Her face was beautiful.',
             'Woman and her daughter playing.',
             'Close up of a face of young boy.'],
    'places':
            ['A view of a beautiful landscape.',
              'Houses along a street.',
              'City skyline with blue sky.',
              'Woodlands in the morning.',
              'A park with bushes and trees in the distance.'],
    'food':
            ['A plate of food with vegetables.',
             'A hamburger with fries.',
             'A bowl of fruit.',
             'A plate of spaghetti.',
             'A bowl of soup.'],
    'unique_sentence_people': 
            ['An average-looking person.'],
    'unique_sentence_places': 
            ['A typical house in the suburbs.'],
    'unique_sentence_food': 
            ['A sandwich with ham and cheese.'],
}