import pickle, os
import numpy as np
import pandas as pd
from nsd_visuo_semantics.decoding_analyses.decoding_utils import restore_nan_dims, load_gcc_embeddings
from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois

# NOTE: Looking at maps, cluster 4 should be people, 8 might be food be food, 9 should be places

# use_n_lookup = None  # if None, use all embeddings, otherwise, use only the use_n_lookup random ones (good for debugging)

for use_n_lookup in [20000]:

    base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir"
    results_dir = os.path.join(base_save_dir, f"voxel_wise_analyses")
    cache_dir = os.path.join(results_dir, "cache")

    nsd_captions_path = f"/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"
    nsd_embeddings_path = f"/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/saved_embeddings/nsd_mpnet_mean_embeddings.pkl"

    which_rois = 'streams'
    nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
    rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')
    maskdata, ROIS = get_rois(which_rois, rois_dir)

    n_clusters = 11
    cluster_file = f'{cache_dir}/semantic_clusters/cluster_labels_all_visROIs_4_12_1_zscored.csv'
    cluster_assignments = pd.read_csv(cluster_file)
    cluster_vector = np.array(cluster_assignments[str(n_clusters)])+1  # 0 is kept for no cluster (e.g. for non-significant voxels)

    fitted_models_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/fitted_models"
    save_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/project_embeddings/cache'
    os.makedirs(save_dir, exist_ok=True)

with open(nsd_captions_path, "rb") as fp:
    lookup_sentences = pickle.load(fp)
with open(nsd_embeddings_path, "rb") as fp:
    lookup_embeddings = pickle.load(fp)    
if use_n_lookup is not None:
    print(f"Using only {use_n_lookup} random embeddings")
    np.random.seed(0)
    random_idx = np.random.choice(len(lookup_sentences), use_n_lookup, replace=False)
    lookup_sentences = [lookup_sentences[i] for i in random_idx]
    lookup_embeddings = np.array([lookup_embeddings[i] for i in random_idx], dtype=np.float32)

    full_brain_map = np.zeros((maskdata.shape[0],))
    if 'highervisrois' in cluster_file.lower():
        full_brain_map[maskdata >= 5] = cluster_vector
    elif 'visrois' in cluster_file.lower():
        full_brain_map[maskdata != 0] = cluster_vector
    else:
        full_brain_map = cluster_vector

    subs = [f'subj0{s}' for s in range(1,9)]

    pred_voxels_per_cluster_save_name = f"{save_dir}/pred_voxels_per_cluster_{'fullLookup' if use_n_lookup is None else f'{use_n_lookup}lookups'}.pickle"
    if os.path.exists(pred_voxels_per_cluster_save_name):
        print(f"Loading pred_voxels_per_cluster from {pred_voxels_per_cluster_save_name}")
        with open(pred_voxels_per_cluster_save_name, "rb") as f:
            pred_voxels_per_cluster = pickle.load(f)

    else:
        print(f"Computing pred_voxels_per_cluster and saving to {pred_voxels_per_cluster_save_name}")
        
        pred_voxels_per_cluster = {}
        for subj in subs:
            print(f"\tProcessing {subj}")

            pred_voxels_per_cluster[subj] = {}

            model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel_fullbrain.pkl"
            with open(model_save_path, "rb") as f:
                fitted_fracridge = pickle.load(f)

            pred_voxels = fitted_fracridge.predict(lookup_embeddings)

            nan_idx_to_restore = np.load(f"{fitted_models_dir}/{subj}_NanIdxToRestore.npy")
            pred_voxels = restore_nan_dims(pred_voxels, nan_idx_to_restore, axis=1)

            for c in range(1, n_clusters+1):
                pred_voxels_per_cluster[subj][f'cluster{c}'] = pred_voxels[:, full_brain_map == c].astype(np.float32)

            del fitted_fracridge, pred_voxels  # make space for the next subject

        with open(pred_voxels_per_cluster_save_name, "wb") as f:
            print(f"Done. Saving pred_voxels_per_cluster to {pred_voxels_per_cluster_save_name}")
            pickle.dump(pred_voxels_per_cluster, f)

    # for each cluster, get the sentence that maximally voxels in the cluser (taking the sum of the predictions)
    max_sentences_per_cluster_save_name = f"{save_dir}/max_sentences_per_cluster_{'fullLookup' if use_n_lookup is None else f'{use_n_lookup}lookups'}.pickle"

    means_across_vis_ROIs = {}
    for subj in subs:
        means_across_vis_ROIs[subj] = np.zeros((len(lookup_sentences),))
        for c in range(1, n_clusters+1):
            means_across_vis_ROIs[subj] += np.nanmean(pred_voxels_per_cluster[subj][f'cluster{c}'], axis=1)/n_clusters


    max_sentences_per_cluster = {}
    subj_avg_pred_voxels_per_cluster = {}
    for c in range(1, n_clusters+1):
        max_sentences_per_cluster[f'cluster{c}'] = {}
        subj_avg_pred_voxels_per_cluster[f'cluster{c}'] = np.zeros((len(lookup_sentences),))
        for subj in subs:
            these_cluser_sums = np.nanmean(pred_voxels_per_cluster[subj][f'cluster{c}'], axis=1) - means_across_vis_ROIs[subj]
            subj_avg_pred_voxels_per_cluster[f'cluster{c}'] += these_cluser_sums/np.nanmax(these_cluser_sums)
            max_sentences_per_cluster[f'cluster{c}'][subj] = lookup_sentences[np.argmax(these_cluser_sums)][0]
        max_sentences_per_cluster[f'cluster{c}']['subjAvg'] = lookup_sentences[np.argmax(subj_avg_pred_voxels_per_cluster[f'cluster{c}'])][0]

        with open(max_sentences_per_cluster_save_name, "wb") as f:
            pickle.dump(max_sentences_per_cluster, f)

# import pprint
# pprint.pprint(max_sentences_per_cluster[f'cluster1'])
print('\ncluster4 (faces)\n', max_sentences_per_cluster[f'cluster4'])
print('\ncluster8 (food?)\n', max_sentences_per_cluster[f'cluster8'])
print('\ncluster9 (places)\n', max_sentences_per_cluster[f'cluster9'])

# subjAvg_sentences = [(f'cluster{c}', max_sentences_per_cluster[f'cluster{c}']['subjAvg']) for c in range(1,11)]
# [print(f"{s}\n") for s in subjAvg_sentences]

# find which index in lookup_sentences corresponds to the sentence that maximally activates each cluster
# subjAvg_sentence_idx = [lookup_sentences.index(s) for c, s in subjAvg_sentences]

import pdb; pdb.set_trace()