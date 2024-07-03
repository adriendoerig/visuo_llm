'''Plots histogram of decoding model performance (i.e., correlation between predicted and target embeddings) 
for each subject, as well as the noise ceiling between subjects.
NOTE: the data used in this function are computed in decoding_extra_analyses.py.'''

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cap_perf_metric = 'correlation'# 'correlation', or an entry of the f'{base_results_dir}/subj01_capEval_scores.npy' file (e.g. cider_d)

n_subj = 8
base_results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIallvisROIs/predicted_sentences_scores'

if cap_perf_metric == 'correlation':
    dummy = np.load(f'{base_results_dir}/subj01_avg_cap_corrs.npy')
else:
    with open(f"{base_results_dir}/subj01_capEval_scores.pkl", "rb") as f:
        dummy = pickle.load(f)[cap_perf_metric]

subj_scores = np.zeros(((n_subj,) + dummy.shape))

for i in range(n_subj):
    if cap_perf_metric == 'correlation':
        # subj_scores[i] = np.load(f'{base_results_dir}/predicted_vs_target_embeddings_RDM_diag-offdiag_data_correlation_subj{i+1:02d}.npy')
        subj_scores[i] = np.load(f'{base_results_dir}/subj{i+1:02d}_avg_cap_corrs.npy')
        cap_consists = np.load(f'{base_results_dir}/crossSubj_capCorr_consists.npy')
    else:
        with open(f"{base_results_dir}/subj{i+1:02d}_capEval_scores.pkl", "rb") as f:
            subj_scores[i] = pickle.load(f)[cap_perf_metric]
        with open(f"{base_results_dir}/crossSubj_capEval_scores.pkl", "rb") as f:
            cap_consists = pickle.load(f)[cap_perf_metric]


# palette = sns.color_palette("colorblind", n_subj)
palette = sns.color_palette("husl", n_subj)
custom_labels = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7', 'Subject 8']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each subject's distribution with a slightly transparent shade of blue
for i in range(n_subj):
    sns.kdeplot(subj_scores[i], color=palette[i], fill=True, alpha=0.3, ax=ax, label=custom_labels[i])

ax.axvline(x=np.mean(cap_consists), linestyle='-', linewidth=1)

# Set labels and title
# ax.set_xlabel('Prediction accuracy gain\n[row-wise spearman correlation difference]', fontsize=12)
ax.set_xlabel(f'{cap_perf_metric}(pred,target)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

plt.legend()

# Show plot
plt.savefig(f'multisubj_kde_plot_means_{cap_perf_metric}.svg')
plt.close()
