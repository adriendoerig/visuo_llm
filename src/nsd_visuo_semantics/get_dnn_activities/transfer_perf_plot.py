import h5py, pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

with open('transfer_results.pkl', 'rb') as f:
    results_dict = pickle.load(f)

# MPNet_to_MPNet_perfs = [0.6852, 0.6903, 0.6888, 0.6867, 0.6852, 0.6822, 0.6926, 0.6842, 0.6914, 0.6905]
MPNet_to_MPNet_perfs = [v for v in results_dict['mpnet_rec']['mpnet'].values()]
MPNet_to_MPNet_perf = np.mean(MPNet_to_MPNet_perfs)
MPNet_to_MPNet_std = np.std(MPNet_to_MPNet_perfs)

# multihot_to_multihot_perfs = [0.6269, 0.6361, 0.6307, 0.6255, 0.6217, 0.6271, 0.6313, 0.6247, 0.6292, 0.6324]
multihot_to_multihot_perfs = [v for v in results_dict['multihot_rec']['multihot'].values()]
multihot_to_multihot_perf = np.mean(multihot_to_multihot_perfs)
multihot_to_multihot_std = np.std(multihot_to_multihot_perfs)

# MPNet_to_multihot_transfer_perfs = [0.638]
MPNet_to_multihot_transfer_perfs = [v for v in results_dict['mpnet_rec']['multihot'].values()]
MPNet_to_multihot_transfer_perf = np.mean(MPNet_to_multihot_transfer_perfs)
MPNet_to_multihot_transfer_std = np.std(MPNet_to_multihot_transfer_perfs)

# multihot_to_MPNet_transfer_perfs = [0.621]
multihot_to_MPNet_transfer_perfs = [v for v in results_dict['multihot_rec']['mpnet'].values()]
multihot_to_MPNet_transfer_perf = np.mean(multihot_to_MPNet_transfer_perfs)
multihot_to_MPNet_transfer_std = np.std(multihot_to_MPNet_transfer_perfs)

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
bar_labels = ['multihot', 'MPNet_to_multihot_transfer', 'MPNet', 'multihot_to_MPNet_transfer']
bar_values = [multihot_to_multihot_perf, MPNet_to_multihot_transfer_perf, MPNet_to_MPNet_perf, multihot_to_MPNet_transfer_perf]
bar_errors = [multihot_to_multihot_std, MPNet_to_multihot_transfer_std, MPNet_to_MPNet_std, multihot_to_MPNet_transfer_std]
bar_colors = ['skyblue', 'skyblue', 'salmon', 'salmon']
bar_positions = np.array([0, 0.6, 1.8, 2.4])

ax.bar(bar_positions, bar_values, yerr=bar_errors, color=bar_colors, width=0.4)

# Set title and labels
ax.set_xticks([0, 0.6, 1.8, 2.4])
ax.set_xticklabels(bar_labels, fontsize=12, rotation=45)
ax.set_ylabel('Validation performance\n[cosine similarity]', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# Plotting horizontal dashed bars
ax.axhline(y=multihot_avg_sim, color='b', linestyle='--', label='avg(train_multihot)', xmin=0.02, xmax=0.4)
ax.axhline(y=mpnet_avg_sim, color='r', linestyle='--', label='avg(train_MPNet_embeds)', xmin=0.6, xmax=0.98)
ax.legend()

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y axis limit
y_min = min(mpnet_avg_sim, multihot_avg_sim) - 0.02
ax.set_ylim(y_min, None)

plt.tight_layout()
plt.savefig('transfer_perf_barplot.svg', dpi=300)  # Save the plot with high resolution
