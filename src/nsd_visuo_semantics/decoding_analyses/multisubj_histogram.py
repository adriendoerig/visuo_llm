import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n_subj = 8
base_results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIallvisROIs/'
dummy = np.load(f'{base_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_correlation_subj01.npy')

subj_scores = np.zeros(((n_subj,) + dummy.shape))

for i in range(n_subj):
    subj_scores[i] = np.load(f'{base_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_correlation_subj{i+1:02d}.npy')

# Create a color palette of different shades of blue
palette = sns.color_palette("Blues", n_subj)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each subject's distribution with a slightly transparent shade of blue
for i in range(n_subj):
    sns.kdeplot(subj_scores[i], color=palette[-1], fill=True, alpha=0.3, ax=ax)

# Set labels and title
ax.set_xlabel('Prediction accuracy gain\n[row-wise spearman correlation difference]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

# Show plot
plt.savefig('multisubj_kde_plot.png')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each subject's distribution with a slightly transparent shade of blue
for i in range(n_subj):
    sns.histplot(subj_scores[i], color=palette[-1], bins=100, fill=True, alpha=0.3, ax=ax)

# Set labels and title
ax.set_xlabel('Prediction accuracy gain\n[row-wise spearman correlation difference]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

# Show plot
plt.savefig('multisubj_hist_plot.png')