'''
The following embeddings are pre-computed as part opf the ms_coco h5 dataset provided with the code: 
mpnet, guse, category_embeddings, and multihot.

This script extracts them and saves them.'''

import h5py, pickle, os

h5_dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"

save_to = "../results_dir/saved_embeddings"
os.makedirs(save_to, exist_ok=1)

with h5py.File(h5_dataset_path,'r') as f:
    
    guse_mean_embeds = f['test']['guse_mean_embeddings'][:]
    categ_mean_word_embeds = f['test']['fasttext_category_mean_embeddings'][:]
    multihot = f['test']['img_multi_hot'][:]

    with open(f"{save_to}/nsd_guse_mean_embeddings.pkl", "wb") as fp:  # Pickling
            pickle.dump(guse_mean_embeds, fp)
    with open(f"{save_to}/nsd_fasttext_CATEGORY_mean_embeddings.pkl", "wb") as fp:  # Pickling
            pickle.dump(categ_mean_word_embeds, fp)
    with open(f"{save_to}/nsd_multihot.pkl", "wb") as fp:  # Pickling
            pickle.dump(multihot, fp)
