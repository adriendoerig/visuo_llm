import h5py, pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

with open('transfer_results_places365.pkl', 'rb') as f:
    results_dict = pickle.load(f)

MPNet_transfer_perfs = [v*100 for v in results_dict['mpnet_rec_places365']['places365'].values()]
MPNet_transfer_perf = np.mean(MPNet_transfer_perfs)
MPNet_transfer_std = np.std(MPNet_transfer_perfs)

multihot_transfer_perfs = [v*100 for v in results_dict['multihot_rec_places365']['places365'].values()]
multihot_transfer_perf = np.mean(multihot_transfer_perfs)
multihot_transfer_std = np.std(multihot_transfer_perfs)

# baselines
with h5py.File('/share/klab/datasets/ms_coco_embeddings_square256_proper_chunks.h5', 'r') as f:
    mpnet_train = f['train']['all_mpnet_base_v2_mean_embeddings'][:]
    mpnet_train_avg = mpnet_train.mean(axis=0)

    multihot_train = f['train']['img_multi_hot'][:]
    multihot_train_avg = multihot_train.mean(axis=0)

    # avg distance between avg_train and each vector in the val set
    mpnet_val = f['val']['all_mpnet_base_v2_mean_embeddings'][:]
    mpnet_distances = cdist(mpnet_train_avg[np.newaxis, :], mpnet_val, metric='cosine')
    multihot_val = f['val']['img_multi_hot'][:]
    multihot_distances = cdist(multihot_train_avg[np.newaxis, :], multihot_val, metric='cosine')

    mpnet_avg_sim = np.nanmean(1 - mpnet_distances)
    multihot_avg_sim = np.nanmean(1 - multihot_distances)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Plotting bars
bar_labels = ['MPNet_transfer_perfs', 'multihot_transfer_perf']
bar_values = [MPNet_transfer_perf, multihot_transfer_perf]
bar_errors = [MPNet_transfer_std, multihot_transfer_std]
bar_colors = ['skyblue', 'skyblue']#, 'salmon', 'salmon']
bar_positions = np.array([0, 0.6])

ax.bar(bar_positions, bar_values, yerr=bar_errors, color=bar_colors, width=0.4)

# Set title and labels
ax.set_xticks([0, 0.6])
ax.set_xticklabels(bar_labels, fontsize=12, rotation=45)
ax.set_ylabel('Validation performance\n[%correct]', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# Plotting horizontal dashed bars
# ax.axhline(y=multihot_avg_sim, color='b', linestyle='--', label='avg(train_multihot)', xmin=0.02, xmax=0.4)
# ax.axhline(y=mpnet_avg_sim, color='r', linestyle='--', label='avg(train_MPNet_embeds)', xmin=0.6, xmax=0.98)
# ax.legend()

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y axis limit
# y_min = min(mpnet_avg_sim, multihot_avg_sim) - 0.02
# ax.set_ylim(y_min, None)

plt.tight_layout()
plt.savefig('transfer_perf_barplot_scens.png', dpi=300)  # Save the plot with high resolution
