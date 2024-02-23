import os, h5py
import pandas as pd
import numpy as np

DEBUG = False

gcc_dir = '/share/klab/datasets/google_conceptual_captions'

sentences_path = os.path.join(gcc_dir, "conceptual_captions_{}.tsv")
embeddings_path = os.path.join(gcc_dir, "conceptual_captions_mpnet_{}.npy")

# there is a train and val set in the gcc captions, we load the ones chosen by the user (concatenating them)
train_embeddings = np.load(embeddings_path.format('train'))
df = pd.read_csv(sentences_path.format('train'), sep="\t", header=None, names=["sent", "url"],)
train_sentences = df["sent"].to_list()

val_embeddings = np.load(embeddings_path.format('val'))
df = pd.read_csv(sentences_path.format('val'), sep="\t", header=None, names=["sent", "url"],)
val_sentences = df["sent"].to_list()

if DEBUG:
    train_embeddings = train_embeddings[:100]
    train_sentences = train_sentences[:100]
    val_embeddings = val_embeddings[:100]
    val_sentences = val_sentences[:100]

# print average sentence length
print(np.mean([len(s.split()) for s in train_sentences]))
print(np.max([len(s.split()) for s in train_sentences]))
print(np.min([len(s.split()) for s in train_sentences]))

# Create an HDF5 file
h5_file = f'{gcc_dir}/gcc_mpnet_embeddings{"DEBUG" if DEBUG else ""}.h5'
dt = h5py.string_dtype(encoding='utf-8')
with h5py.File(h5_file, 'w') as f:
    # Create 'train' group
    train_group = f.create_group('train')
    train_group.create_dataset('captions', data=train_sentences, shape=len(train_sentences), dtype=dt)
    train_group.create_dataset('mpnet_embeddings', data=train_embeddings)

    # Create 'val' group
    val_group = f.create_group('val')
    val_group.create_dataset('captions', data=val_sentences, shape=len(val_sentences), dtype=dt)
    val_group.create_dataset('mpnet_embeddings', data=val_embeddings)
