import pickle, os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.multitest as multi
from itertools import combinations as cb
from scipy import stats


def nsd_roi_analyses_dnnAllLayers_figure(base_save_dir, which_rois, rdm_distance, 
                            models_to_contrast=['dnn_mpnet_rec', 'dnn_multihot_rec'],
                            layers_to_contrast=np.arange(10),
                            timesteps_to_contrast=np.arange(6),
                            USE_NOISE_CEIL=True, 
                            average_seeds=np.arange(1, 11),
                            epoch=200):
    '''Plots a [n_layersxn_timesteps] matrix per ROI, where each cell is the mean correlation difference between two models'''

    roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
    results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")

    model_keys = []
    if average_seeds is not None:
        for m in models_to_contrast:
            for s in average_seeds:
                for l in range(len(layers_to_contrast)*len(timesteps_to_contrast)):
                    model_keys.append(f'{m}_seed{s}_ep{epoch}_layer{l}')
    else:
        raise Exception('Not implemented yet - please use ' \
                        'average_seeds=[np.array of seeds to average], ' \
                        'or implement a version without seed averaging')

    roi_keys = [
        "earlyROI",
        "midventralROI",
        "ventralROI",
        "midlateralROI",
        "lateralROI",
        "midparietalROI",
        "parietalROI",
    ]
    roi_labels = [
        "EVC",
        "midventral",
        "ventral",
        "midlateral",
        "lateral",
        "midparietal",
        "parietal",
    ]
    roi_colors = [
        "mediumaquamarine",
        "khaki",
        "yellow",
        "lightskyblue",
        "royalblue",
        "lightcoral",
        "red",
    ]
    roi_specs = {
        k: {"label": l, "color": c}
        for k, l, c in zip(roi_keys, roi_labels, roi_colors)
    }

    # load dicts with all correlations
    if USE_NOISE_CEIL:
        with open(f'{results_dir}/subj_roi_rdms/subjWiseNoiseCeiling_group_corrs.pkl', "rb") as g1, \
            open(f'{results_dir}/subj_roi_rdms/subjWiseNoiseCeiling_group_mean_corrs.pkl', "rb") as g2, \
            open(f'{results_dir}/subj_roi_rdms/subjWiseNoiseCeiling_group_std_corrs.pkl', "rb") as g3:
            group_corrs = pickle.load(g1)
            group_mean_corrs = pickle.load(g2)
            group_std_corrs = pickle.load(g3)
    else:
        with open(f'{results_dir}/subj_roi_rdms/group_corrs_no_noise_ceiling.pkl', "rb") as g1, \
            open(f'{results_dir}/subj_roi_rdms/group_mean_corrs_no_noise_ceiling.pkl', "rb") as g2, \
            open(f'{results_dir}/subj_roi_rdms/group_std_corrs_no_noise_ceiling.pkl', "rb") as g3:
            group_corrs = pickle.load(g1)
            group_mean_corrs = pickle.load(g2)
            group_std_corrs = pickle.load(g3)

    group_mean_corrs = {k: group_mean_corrs[k] for k in model_keys}
    means = {roi_key: {model_key: group_mean_corrs[model_key][roi_key] 
                    for model_key in model_keys} 
                    for roi_key in roi_keys}
    stds = {roi_key: {model_key: group_std_corrs[model_key][roi_key]/np.sqrt(8) 
                    for model_key in model_keys }
                    for roi_key in roi_keys }
    corr_samples = {
        roi_key: {model_key: group_corrs[model_key][roi_key]
                for model_key in model_keys}
                for roi_key in roi_keys}
    
    if average_seeds is not None:
        # in this case, expects '_seedN_epM' in model names, and averages across seeds
        seed_avg_corrs = {roi_key: {} for roi_key in roi_keys}

        import re
        def remove_seed_from_strings(string_list):
            # gets rid of '_seedN' in model names
            pattern = re.compile(r'_seed\d+')
            result_list = [pattern.sub('', s) for s in string_list]
            return result_list

        # seed_base_model_names = list(set([m.split('_seed')[0] for m in seed_models]))
        # seed_base_model_suffix = list(set([m.split('_ep')[1] for m in seed_models]))[0]
        seed_base_model_names = remove_seed_from_strings(model_keys)
        seed_base_model_keys = {seed_base_model_name: [] for seed_base_model_name in seed_base_model_names}
        for seed_base_model_name in seed_base_model_names:
            for m in model_keys:
                if m.split('_seed')[0] == seed_base_model_name.split('_ep')[0]:
                    if m.split('_ep')[1] == seed_base_model_name.split('_ep')[1]:
                         seed_base_model_keys[seed_base_model_name].append(m)
                for roi_key in roi_keys:
                    seed_avg_corrs[roi_key][seed_base_model_name] = np.mean(np.asarray([corr_samples[roi_key][mk] for mk in seed_base_model_keys[seed_base_model_name]]), axis=0)
        means = {roi_key: {model_key: np.mean(seed_avg_corrs[roi_key][model_key]) for model_key in seed_base_model_names} for roi_key in roi_keys}
        stds = {roi_key: {model_key: np.std(seed_avg_corrs[roi_key][model_key])/np.sqrt(8) for model_key in seed_base_model_names} for roi_key in roi_keys}

    my_stats = {
        "uncorrected": {
            "single_model_ttest_1samp_ttest_2sided": {},
            "model_comparisons_ttest_ind_2sided": {},
        },
        "corrected": {
            "single_model_ttest_1samp_ttest_2sided": {},
            "model_comparisons_ttest_ind_2sided": {},
        },
    }

    # plot!
    fig, ax = plt.subplots(len(roi_keys), 3, figsize=(3*4, len(roi_keys)*4))
    from matplotlib import colors
    model_matrices0 = {k: np.zeros((len(layers_to_contrast), len(timesteps_to_contrast))) for k in roi_keys}
    model_matrices1 = {k: np.zeros((len(layers_to_contrast), len(timesteps_to_contrast))) for k in roi_keys}
    contrast_matrices = {k: np.zeros((len(layers_to_contrast), len(timesteps_to_contrast))) for k in roi_keys}
    # Fill in the matrices
    for i, roi_key in enumerate(roi_keys):
        for l in layers_to_contrast:
            for t in timesteps_to_contrast:
                l_idx = np.ravel_multi_index((l, t), (len(layers_to_contrast), len(timesteps_to_contrast)))
                model_matrices0[roi_key][l, t] = means[roi_key][f'{models_to_contrast[0]}_ep{epoch}_layer{l_idx}']
                model_matrices1[roi_key][l, t] = means[roi_key][f'{models_to_contrast[1]}_ep{epoch}_layer{l_idx}']
                contrast_matrices[roi_key][l, t] = model_matrices0[roi_key][l, t] - model_matrices1[roi_key][l, t]
        p = ax[i][0].imshow(model_matrices0[roi_key], cmap='Blues')
        plt.colorbar(p,ax=ax[i][0])
        p = ax[i][1].imshow(model_matrices1[roi_key], cmap='Reds')
        plt.colorbar(p,ax=ax[i][1])
        divnorm=colors.TwoSlopeNorm(vmin=-np.abs(contrast_matrices[roi_key]).max(), vcenter=0., vmax=np.abs(contrast_matrices[roi_key]).max())
        p = ax[i][2].imshow(contrast_matrices[roi_key], cmap='RdBu', norm=divnorm)
        plt.colorbar(p,ax=ax[i][2])

    # Labeling rows & cols
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for i, row_name in enumerate(roi_keys):
        ax[i, 0].set_ylabel(row_name, rotation=0, labelpad=15, ha='right')
    for j, col_name in enumerate(models_to_contrast + [models_to_contrast[0] + ' - ' + models_to_contrast[1]]):
        ax[0, j].set_xlabel(col_name, rotation=0, labelpad=15, ha='center')
        ax[0, j].xaxis.set_label_position('top') 
    plt.tight_layout()
    plt.savefig(f'{results_dir}/DNN_MULTILAYER_ROIwiseModelCorrs.png')
    plt.close()

    #         # statistics
            # print('WARNING: PLEASE CHECK STATISTICS CODE IN NSD_ROI_ANALYSIS_FIGURE.PY')
            # print('WARNING: PLEASE CHECK STATISTICS CODE IN NSD_ROI_ANALYSIS_FIGURE.PY')
            # print('WARNING: PLEASE CHECK STATISTICS CODE IN NSD_ROI_ANALYSIS_FIGURE.PY')
            # print('WARNING: PLEASE CHECK STATISTICS CODE IN NSD_ROI_ANALYSIS_FIGURE.PY')
    #         s = stats.ttest_1samp(corr_samples[roi_key][model_key], 0, alternative="two-sided")
    #         my_stats["uncorrected"]["single_model_ttest_1samp_ttest_2sided"][roi_labels[i]][model_key] = s.pvalue

    #         for contrast_model_key in model_keys:
    #             if contrast_model_key != model_key:
    #                 this_comparison = f"{model_key}_vs_{contrast_model_key}"
    #                 this_comparison_inverse = (f"{contrast_model_key}_vs_{model_key}")
    #                 if this_comparison not in model_comparisons_done:
    #                     s = stats.ttest_ind(
    #                         corr_samples[roi_key][model_key],
    #                         corr_samples[roi_key][contrast_model_key],
    #                         axis=0,
    #                         alternative="two-sided",
    #                     )
    #                     my_stats["uncorrected"]["model_comparisons_ttest_ind_2sided"][roi_labels[i]][this_comparison] = s.pvalue
    #                     model_comparisons_done.append(this_comparison)
    #                     model_comparisons_done.append(this_comparison_inverse)

    #     for k1 in my_stats["uncorrected"].keys():  # 'single_model_ttest_1samp_ttest_2sided', ...
    #         print(f"\t\tTest: {k1}")
    #         these_pvals = []
    #         for k2 in my_stats["uncorrected"][k1][roi_labels[i]].keys():  # models names, or comparison names, ...
    #             these_pvals.append(my_stats["uncorrected"][k1][roi_labels[i]][k2])
    #         out = multi.multipletests(these_pvals, alpha=0.05, method="fdr_bh")
    #         these_corrected_reject = out[0]
    #         these_corrected_pvals = out[1]
    #         for k_i, k2 in enumerate(my_stats["uncorrected"][k1][roi_labels[i]].keys()):
    #             my_stats["corrected"][k1][roi_labels[i]][k2] = these_corrected_pvals[k_i]
    #             if not these_corrected_reject[k_i]:
    #                 print(f"\t\t\t{k2}: pval: {these_corrected_pvals[k_i]} - reject: {these_corrected_reject[k_i]}")

    # # save np arrays for each ROI with corrected pvals of each model comparison
    # model_comps = list(cb(model_keys, 2))
    # n_comparisons = len(my_stats["corrected"]["model_comparisons_ttest_ind_2sided"][roi_labels[0]].keys())
    # ROI_wise_pvals = {k: np.empty(n_comparisons) for k in roi_labels}
    # for this_roi in roi_labels:
    #     for idx, model_comp in enumerate(model_comps):
    #         ROI_wise_pvals[this_roi][idx] = my_stats["corrected"][
    #             "model_comparisons_ttest_ind_2sided"
    #         ][this_roi][f"{model_comp[0]}_vs_{model_comp[1]}"]
    # np.save(f"{results_dir}/PAPER_FIG{fig_id}_ROIwisePvals",  ROI_wise_pvals, allow_pickle=True)


    # ### FINAL COSMETICS
    # # Add axis labels
    # ax.set_ylabel("Noise-ceiling corrected\nPearson correlation\n(mean across subjects)")
    # ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])#, fontsize="smaller")
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels(model_labels * len(roi_keys), rotation=45, ha="right", fontsize="small")

    # # Add ROI names above each group of bars
    # for i, roi_key in enumerate(roi_keys):
    #     center_position = i + 0.5 * (len(model_keys) - 1) * bar_width
    #     roi_label = roi_specs[roi_key]["label"]
    #     ax.text(center_position, 1.02, roi_label, ha="center", transform=ax.get_xaxis_transform())

    # # Save figure
    # plt.tight_layout()
    # # plt.savefig(f"{results_dir}/PAPER_FIG{fig_id}{'_SubjWiseNoiseCeiling' if USE_NOISE_CEIL else ''}{plt_suffix}.svg")  # , dpi=300)
    # plt.savefig(f"{results_dir}/PAPER_FIG{fig_id}{'_SubjWiseNoiseCeiling' if USE_NOISE_CEIL else ''}{plt_suffix}.png")  # , dpi=300)

