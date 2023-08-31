import os
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.multitest as multi
from scipy import stats

results_dir = "./save_dir/roi_analyses/streams_roi_results_correlation"

for fig_id in [5]:  # [2, 5]:
    print(f"Fig: {fig_id}")

    if fig_id == 2:
        model_keys = [
            "multihot",
            "fasttext_categories",
            "fasttext_verbs",
            "fasttext_all",
            "guse",
            "mpnet",
        ]
        model_labels = [
            "categ multihot",
            "categ word embeds",
            "verb word embeds",
            "all word embeds",
            "GUSE",
            "MPNet",
        ]
        model_alphas = [0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    elif fig_id == 5:
        # model_keys = ['multihot', 'dnn_multihot_ff', 'dnn_multihot_rec', 'mpnet', 'dnn_mpnet_ff', 'dnn_mpnet_rec']
        # model_labels = ['multihot', 'multihot_trained_cnn', 'multihot_trained_rcnn', 'MPNet', 'MPNet_trained_cnn', 'MPNet_trained_rcnn']
        # model_alphas = [0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
        model_keys = ["multihot", "dnn_multihot_rec", "mpnet", "dnn_mpnet_rec"]
        model_labels = [
            "multihot",
            "multihot-trained RCNN",
            "MPNet",
            "MPNet-trained RCNN",
        ]
        model_alphas = [0.25, 0.50, 0.75, 1.0]

    bar_specs = {
        "width": 0.13,
        "edgecolor": "black",
        "linewidth": 0.7,
        "zorder": 10,
    }
    model_specs = {
        k: {"label": l, "alpha": a}
        for k, l, a in zip(model_keys, model_labels, model_alphas)
    }
    # roi_keys = ['earlyROI', 'midventralROI', 'midlateralROI', 'midparietalROI', 'ventralROI', 'lateralROI', 'parietalROI']
    # roi_labels = ['EVC', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal']
    # roi_colors = ['tab:%s' % c for c in ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'olive']]  # tab:cyan
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
    ]  # tab:cyan
    roi_specs = {
        k: {"label": l, "color": c}
        for k, l, c in zip(roi_keys, roi_labels, roi_colors)
    }

    with open(
        os.path.join(
            results_dir,
            "subj_roi_rdms",
            "subjWiseNoiseCeiling_group_corrs.pkl",
        ),
        "rb",
    ) as g1, open(
        os.path.join(
            results_dir,
            "subj_roi_rdms",
            "subjWiseNoiseCeiling_group_mean_corrs.pkl",
        ),
        "rb",
    ) as g2, open(
        os.path.join(
            results_dir,
            "subj_roi_rdms",
            "subjWiseNoiseCeiling_group_std_corrs.pkl",
        ),
        "rb",
    ) as g3:
        group_corrs = pickle.load(g1)
        group_mean_corrs = pickle.load(g2)
        group_std_corrs = pickle.load(g3)

    group_mean_corrs = {k: group_mean_corrs[k] for k in model_keys}
    means = {
        roi_key: {
            model_key: group_mean_corrs[model_key][roi_key]
            for model_key in model_keys
        }
        for roi_key in roi_keys
    }
    stds = {
        roi_key: {
            model_key: group_std_corrs[model_key][roi_key]
            / np.sqrt(8)  # because we have 8 subjects
            for model_key in model_keys
        }
        for roi_key in roi_keys
    }
    corr_samples = {
        roi_key: {
            model_key: group_corrs[model_key][roi_key]
            / np.sqrt(8)  # because we have 8 subjects
            for model_key in model_keys
        }
        for roi_key in roi_keys
    }

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
    fig, ax = plt.subplots(figsize=(10, 3))
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

        for j, model_key in enumerate(model_keys):
            model_label = model_specs[model_key]["label"]
            model_alpha = model_specs[model_key]["alpha"]
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
            ax.errorbar(
                bar_pos, perf, yerr=std, color="black", capsize=3, zorder=11
            )
            x_positions.append(bar_pos)

            # statistics
            s = stats.ttest_1samp(
                corr_samples[roi_key][model_key], 0, alternative="two-sided"
            )
            my_stats["uncorrected"]["single_model_ttest_1samp_ttest_2sided"][
                roi_labels[i]
            ][model_key] = s.pvalue

            for contrast_model_key in model_keys:
                if contrast_model_key != model_key:
                    this_comparison = f"{model_key}_vs_{contrast_model_key}"
                    this_comparison_inverse = (
                        f"{contrast_model_key}_vs_{model_key}"
                    )
                    if this_comparison not in model_comparisons_done:
                        s = stats.ttest_ind(
                            corr_samples[roi_key][model_key],
                            corr_samples[roi_key][contrast_model_key],
                            axis=0,
                            alternative="two-sided",
                        )
                        my_stats["uncorrected"][
                            "model_comparisons_ttest_ind_2sided"
                        ][roi_labels[i]][this_comparison] = s.pvalue
                        model_comparisons_done.append(this_comparison)
                        model_comparisons_done.append(this_comparison_inverse)

        for k1 in my_stats[
            "uncorrected"
        ].keys():  # 'single_model_ttest_1samp_ttest_2sided', ...
            print(f"\t\tTest: {k1}")
            these_pvals = []
            for k2 in my_stats["uncorrected"][k1][
                roi_labels[i]
            ].keys():  # models names, or comparison names, ...
                these_pvals.append(
                    my_stats["uncorrected"][k1][roi_labels[i]][k2]
                )
            out = multi.multipletests(these_pvals, alpha=0.05, method="fdr_bh")
            these_corrected_reject = out[0]
            these_corrected_pvals = out[1]
            for k_i, k2 in enumerate(
                my_stats["uncorrected"][k1][roi_labels[i]].keys()
            ):
                my_stats["corrected"][k1][roi_labels[i]][
                    k2
                ] = these_corrected_pvals[k_i]
                if not these_corrected_reject[k_i]:
                    print(
                        f"\t\t\t{k2}: pval: {these_corrected_pvals[k_i]} - reject: {these_corrected_reject[k_i]}"
                    )

    # save np arrays for each ROI with corrected pvals of each model comparison
    from itertools import combinations as cb

    model_comps = list(cb(model_keys, 2))
    n_comparisons = len(
        my_stats["corrected"]["model_comparisons_ttest_ind_2sided"][
            roi_labels[0]
        ].keys()
    )
    ROI_wise_pvals = {k: np.empty(n_comparisons) for k in roi_labels}
    for this_roi in roi_labels:
        for idx, model_comp in enumerate(model_comps):
            ROI_wise_pvals[this_roi][idx] = my_stats["corrected"][
                "model_comparisons_ttest_ind_2sided"
            ][this_roi][f"{model_comp[0]}_vs_{model_comp[1]}"]
    np.save(
        f"{results_dir}/PAPER_FIG{fig_id}_ROIwisePvals",
        ROI_wise_pvals,
        allow_pickle=True,
    )

    # Add axis labels
    ax.set_ylabel(
        "Noise-ceiling corrected\nPearson correlation\n(mean across subjects)"
    )
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize="smaller")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        model_labels * len(roi_keys), rotation=45, ha="right", fontsize="small"
    )

    # Add ROI names above each group of bars
    for i, roi_key in enumerate(roi_keys):
        center_position = i + 0.5 * (len(model_keys) - 1) * bar_width
        roi_label = roi_specs[roi_key]["label"]
        ax.text(
            center_position,
            1.02,
            roi_label,
            ha="center",
            transform=ax.get_xaxis_transform(),
        )

    # Save figure
    plt.tight_layout()
    plt.savefig(
        f"{results_dir}/PAPER_FIG{fig_id}_SubjWiseNoiseCeiling.svg"
    )  # , dpi=300)
