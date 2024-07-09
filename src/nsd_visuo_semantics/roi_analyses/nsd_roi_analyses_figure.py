'''Plotting functions for ROI analyses.
'''

import pickle, os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.multitest as multi
from itertools import combinations as cb
from scipy import stats


def nsd_roi_analyses_figure(base_save_dir, which_rois, rdm_distance, USE_NOISE_CEIL, custom_model_keys=None, plt_suffix='', 
                            alphabetical_order=False, best_to_worst_order=False,
                            custom_model_labels=None, average_seeds=False,
                            plot_pval_tables=False):
   
    roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
    results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")

    if alphabetical_order:
        custom_model_keys = sorted(custom_model_keys)
    model_keys = custom_model_keys

    roi_keys = [
        "early",
        # "midventral",
        "ventral",
        # "midlateral",
        "lateral",
        # "midparietal",
        "parietal",
    ]
    roi_labels = [
        "EVC",
        # "midventral",
        "ventral",
        # "midlateral",
        "lateral",
        # "midparietal",
        "parietal",
    ]
    roi_colors = [
        "mediumaquamarine",
        # "khaki",
        "yellow",
        # "lightskyblue",
        "royalblue",
        # "lightcoral",
        "red",
    ]  # tab:cyan
    roi_specs = {
        k: {"label": l, "color": c}
        for k, l, c in zip(roi_keys, roi_labels, roi_colors)
    }

    if USE_NOISE_CEIL:
        with open(f'{results_dir}/cache/subjWiseNoiseCeiling_group_corrs.pkl', "rb") as g1, \
            open(f'{results_dir}/cache/subjWiseNoiseCeiling_group_mean_corrs.pkl', "rb") as g2, \
            open(f'{results_dir}/cache/subjWiseNoiseCeiling_group_std_corrs.pkl', "rb") as g3:
            group_corrs = pickle.load(g1)
            group_mean_corrs = pickle.load(g2)
            group_std_corrs = pickle.load(g3)
    else:
        with open(f'{results_dir}/cache/group_corrs_no_noise_ceiling.pkl', "rb") as g1, \
            open(f'{results_dir}/cache/group_mean_corrs_no_noise_ceiling.pkl', "rb") as g2, \
            open(f'{results_dir}/cache/group_std_corrs_no_noise_ceiling.pkl', "rb") as g3:
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
        
    if average_seeds:

        seed_models = []
        non_seed_models = []
        for item in model_keys:
            if '_seed' in item:
                seed_models.append(item)
            else:
                non_seed_models.append(item)

        seed_avg_corrs = {roi_key: {} for roi_key in roi_keys}
        if len(seed_models) > 0:
            seed_base_model_names = list(set([m.split('_seed')[0] for m in seed_models]))
            seed_base_model_suffix = list(set([m.split('_ep')[1] for m in seed_models]))[0]
            for seed_base_model_name in seed_base_model_names:
                seed_base_model_keys = [m for m in model_keys if seed_base_model_name in m]
                for roi_key in roi_keys:
                    these_samples = np.asarray([corr_samples[roi_key][m] for m in seed_base_model_keys])
                    seed_avg_corrs[roi_key][seed_base_model_name+'_seedAVG_ep'+seed_base_model_suffix] = np.mean(these_samples, axis=0)
                    corr_samples[roi_key][seed_base_model_name+'_seedAVG_ep'+seed_base_model_suffix] = np.mean(these_samples, axis=0)
        
        model_keys = list(seed_avg_corrs[roi_keys[0]].keys()) + non_seed_models
        means = {roi_key: {model_key: means[roi_key][model_key] if model_key in non_seed_models else np.mean(seed_avg_corrs[roi_key][model_key]) for model_key in model_keys} for roi_key in roi_keys}
        stds = {roi_key: {model_key: stds[roi_key][model_key] if model_key in non_seed_models else np.std(seed_avg_corrs[roi_key][model_key])/np.sqrt(8) for model_key in model_keys} for roi_key in roi_keys}


    n_models = len(model_keys)
    model_labels = model_keys
    model_alphas = np.linspace(0.1, 1, len(model_keys))
    bar_specs = {
        "width": 0.00975*(44/n_models),  # rough estimate of what will look good
        "edgecolor": "black",
        "linewidth": 0.7*(0.00975/0.021),
        "zorder": 10,
    }
    fig, ax = plt.subplots(figsize=((n_models+5)*2, 5))  # rough estimate of what will look good

    if custom_model_labels is not None:
        model_labels = custom_model_labels
        
    # order models from best to worst perforamnce if desired
    if best_to_worst_order:
        model_sums_across_rois = np.zeros(len(model_keys))
        for roi_key in roi_keys:
            model_sums_across_rois += np.asarray([means[roi_key][model_key] for model_key in model_keys])
        ordered_model_indices = np.argsort(model_sums_across_rois)[::-1]
        model_labels = [model_labels[i] for i in ordered_model_indices]
    else:
        ordered_model_indices = range(len(model_keys))

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

    # Set a few helpers
    bar_width = bar_specs["width"]
    x_positions = []
    x = np.arange(len(roi_keys))

    # Plot the bars
    for i, roi_key in enumerate(roi_keys):
        print(f"\tROI: {roi_key}")

        model_comparisons_done = []

        roi_label = roi_specs[roi_key]["label"]
        roi_color = roi_specs[roi_key]["color"]

        for k1 in my_stats.keys():
            for k2 in my_stats[k1].keys():
                my_stats[k1][k2][roi_labels[i]] = {}

        for j, model_idx in enumerate(ordered_model_indices):
            model_key = model_keys[model_idx]
            model_alpha = model_alphas[j]
            facecolor = mcolors.to_rgb(roi_color) + (model_alpha,)
            perf = means[roi_key][model_key]
            std = stds[roi_key][model_key]
            bar_pos = i + j * bar_width
            ax.bar(
                bar_pos,
                perf,
                **bar_specs,
                facecolor=facecolor,
                label="_no_legend_",
            )  # roi_label)
            ax.errorbar(bar_pos, perf, yerr=std, color="black", capsize=1, capthick=(0.00975/0.021), lw=(0.00975/0.021), zorder=11)
            x_positions.append(bar_pos)

            # statistics
            s = stats.ttest_1samp(corr_samples[roi_key][model_key], 0, alternative="two-sided")
            my_stats["uncorrected"]["single_model_ttest_1samp_ttest_2sided"][roi_labels[i]][model_key] = s.pvalue

            for contrast_model_key in model_keys:
                if contrast_model_key != model_key:
                    this_comparison = f"{model_key}_vs_{contrast_model_key}"
                    this_comparison_inverse = (f"{contrast_model_key}_vs_{model_key}")
                    if this_comparison not in model_comparisons_done:
                        s = stats.ttest_ind(corr_samples[roi_key][model_key], corr_samples[roi_key][contrast_model_key],
                                            axis=0, alternative="two-sided")
                        my_stats["uncorrected"]["model_comparisons_ttest_ind_2sided"][roi_labels[i]][this_comparison] = s.pvalue
                        model_comparisons_done.append(this_comparison)
                        model_comparisons_done.append(this_comparison_inverse)

        for k1 in ['model_comparisons_ttest_ind_2sided']:  # my_stats["uncorrected"].keys():  # 'single_model_ttest_1samp_ttest_2sided', ...
            print(f"\t\tTest: {k1}")
            these_pvals = []
            for k2 in my_stats["uncorrected"][k1][roi_labels[i]].keys():  # models names, or comparison names, ...
                these_pvals.append(my_stats["uncorrected"][k1][roi_labels[i]][k2])
            out = multi.multipletests(these_pvals, alpha=0.05, method="fdr_bh")
            these_corrected_reject = out[0]
            these_corrected_pvals = out[1]
            for k_i, k2 in enumerate(my_stats["uncorrected"][k1][roi_labels[i]].keys()):
                my_stats["corrected"][k1][roi_labels[i]][k2] = these_corrected_pvals[k_i]
                if not these_corrected_reject[k_i]:
                    # if 'dnn_mpnet_rec_seedAVG_ep200_layer-1' in k2:
                    print(f"\t\t\t{k2}: pval: {these_corrected_pvals[k_i]:.4f} - reject: {these_corrected_reject[k_i]}")

    ### FINAL COSMETICS
    # Add axis labels
    ax.set_ylabel("Noise-ceiling corrected\nPearson correlation\n(mean across subjects)")
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])#, fontsize="smaller")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels * len(roi_keys), rotation=45, ha="right", fontsize="small")

    # Add ROI names above each group of bars
    for i, roi_key in enumerate(roi_keys):
        center_position = i + 0.5 * (len(model_keys) - 1) * bar_width
        roi_label = roi_specs[roi_key]["label"]
        ax.text(center_position, 1.02, roi_label, ha="center", transform=ax.get_xaxis_transform())

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ROIFIG{'_SubjWiseNoiseCeiling' if USE_NOISE_CEIL else ''}{plt_suffix}.png")  # , dpi=300)

    if plot_pval_tables:
        import pandas as pd
        from matplotlib.colors import Normalize
        import seaborn as sns
        from matplotlib.patches import Rectangle


        model_names_to_plot = {
            'multihot': 'multihot categ',
            'fasttext_categories': 'fasttext categ',
            'glove_categories': 'glove categ',
            'mpnet': 'LLM caption',
            'mpnet_verbs': 'LLM verbs',
            'mpnet_nouns': 'LLM nouns',
            'mpnet_category_all': 'LLM categ',
            'mpnetWordAvg_all': 'LLM wordavg',
            'fasttext_all': 'fasttext wordavg',
            'glove_all': 'glove wordavg',

            'dnn_mpnet_rec_seedAVG_ep200_layer-1': 'LLM trained (ours)',
            'thingsvision_cornet-s': 'cornet-s', 
            'dnn_ecoset_category_layer-1': 'rcnn_ecoset', 
            'konkle_alexnetgn_supervised_ref12_augset1_5x': 'alexnet-gn-sv', 
            'timm_nf_resnet50': 'nf_resnet50', 
            'brainscore_resnet50_julios': 'resnet50', 
            'brainscore_alexnet': 'alexnet', 
            'sceneCateg_resnet50_finalLayer': 'resnet50_Places365', 
            'taskonomy_scenecat_resnet50': 'taskonomy_scene_cat',
            'CLIP_RN50_images': 'CLIP_RN50_imgs', 
            'resnext101_32x8d_wsl': 'resnext101_32x8d', 
            'CLIP_ViT_images': 'CLIP_ViT_imgs', 
            'google_simclrv1_rn50': 'google_simclr_rn50', 
            'konkle_alexnetgn_ipcl_ref01': 'alexnetgn_ipcl',
        }

        roi_labels_to_plot = ['ventral', 'lateral', 'parietal']
        num_plots = len(roi_labels_to_plot)
        num_cols = 2  # Number of columns for subplots
        num_rows = -(-num_plots // num_cols)  # Ceiling division to ensure enough rows
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 10 * num_rows))  # Adjust the figsize as needed
        axes = [item for sublist in axes for item in sublist]

        for i, (this_roi, this_ax) in enumerate(zip(roi_labels_to_plot, axes)):
            stats_dict = my_stats['corrected']['model_comparisons_ttest_ind_2sided'][this_roi]
            # model_names = set()
            # for comparison in stats_dict.keys():
            #     model_names.update(comparison.split('_vs_'))
            # model_names = list(model_names)
            # model_names = [model_names_to_plot[m] for m in model_names]
            # df = pd.DataFrame(index=model_names, columns=model_names)
            model_names = [model_names_to_plot[m] for m in model_keys]
            df = pd.DataFrame(index=model_names, columns=model_names)
            for comparison in stats_dict.keys():
                model1, model2 = [model_names_to_plot[c] for c in comparison.split('_vs_')]
                if model1 == model2:
                    df.loc[model1, model2] = None
                else:
                    df.loc[model1, model2] = stats_dict[comparison]
                    df.loc[model2, model1] = stats_dict[comparison]

            df_reordered = df.reindex(index=model_names, columns=model_names)

            p_values = np.around(df.values.astype(np.float), 20)
            tick_labels = model_names
            mask = p_values > 0.05
            cmap = sns.color_palette("Blues_r", as_cmap=True)
            norm = Normalize(vmin=0, vmax=0.05)

            # Plot the heatmap on the current subplot
            ax = sns.heatmap(
                p_values, mask=None, cmap=cmap, norm=norm, annot=True, cbar=True,
                linewidths=1, linecolor="gray", ax=this_ax
            )
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=14, fontweight='bold')
            ax.set_yticklabels(tick_labels, rotation=45, fontsize=14, fontweight='bold')
            cbar = ax.collections[0].colorbar
            cbar.set_label("p-value", fontsize=20)
            cbar.ax.tick_params(labelsize=16)
            # Add a thick border around the entire heatmap
            ax.add_patch(Rectangle((0, 0), len(p_values[0]), len(p_values), fill=False, edgecolor="black", lw=2, clip_on=False))

            # Add subtitle
            this_ax.set_title(f'{this_roi}', fontsize=24, fontweight='bold')

        # Adjust the layout
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{plt_suffix}_stats.png', dpi=300)
        plt.savefig(f'{results_dir}/{plt_suffix}_stats.svg', dpi=300)
        plt.show()

    