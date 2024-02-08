import os, pickle
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_model_rdms
from nsd_visuo_semantics.utils.utils import corr_rdms
from nsd_visuo_semantics.utils.batch_gen import BatchGen
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def correlate_model_rdms_figure(MODEL_NAMES, nsd_dir, base_save_dir, rdm_distance, remove_shared_515, 
                                dnn_layer_to_use, plt_suffix, 
                                OVERWRITE=False, COMPUTE=True):
    """
    Correlate RDMs and plot results. 
    """

    if len(MODEL_NAMES) < 2:
        raise ValueError("Need at least two models to correlate.")
    else:
        pairs = []
        for i in range(len(MODEL_NAMES)):
            for j in range(i + 1, len(MODEL_NAMES)):
                pair = [MODEL_NAMES[i], MODEL_NAMES[j]]
                # Check if the reversed pair already exists
                if pair[::-1] not in pairs:
                    pairs.append(pair)
        pair_strings = ["_vs_".join(p) for p in pairs]

    results_dir = os.path.join(base_save_dir, "model_rdm_correlations")
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(base_save_dir,f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}')

    if os.path.exists(f"{results_dir}/model_rdms_corrs.pkl"):
        with open(f"{results_dir}/model_rdms_corrs.pkl", "rb") as fp:
            model_rdms_corrs = pickle.load(fp)
    else:
        model_rdms_corrs = {}

    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]

    if COMPUTE:
        for subj in subs:

            if subj not in model_rdms_corrs.keys():
                model_rdms_corrs[subj] = {}

            # extract conditions data.
            # NOTES ABOUT HOW THIS WORKS:
            # get_conditions returns a list with one item for each session the subject attended. Each of these items contains
            # the NSD_ids for the images presented in that session. Then, we reshape all this into a single array, which now
            # contains all the NSD_ids for the subject, in the order in which they were shown. Next, we create a boolean list of
            # the same size as the conditions array, which assigns True to NSD_ids that are present 3x in the condition array.
            # We use this boolean to create conditions_sampled, which now contains all NSD_indices for stimuli the subject has
            # seen 3x. This list still contains the 3 repetitions of each stimulus, and is still in the stimulus presentation
            # order. For example: [46003, 61883,   829, ...]
            # Hence, we need to only keep each NSD_id once (since we compute everything on the average fMRI data over
            # the 3 presentations), and we also need to order them in increasing NSD_id order (so that we can then easily
            # for all subjects/models). Both of these desiderata are addressed by using np.unique (which sorts the unique idx).
            # So sample contains the unique NSD_ids for that subject, in increasing order (e.g. [ 14,  28,  72, ...]).
            # Importantly, the average betas loaded above are arranged in the same way, so that if we want to find the betas
            # for NSD_id=72, we just need to find the idx of 72 in sample (in the present example: 2). Using this method, we can
            # find the avg_betas corresponding to the shared 515 images as done below with subj_indices_515 (hint: the trick to
            # go from an ordered list of nsd_ids to finding the idx as described above is to use enumerate).
            # For example sample[subj_indices_515[0]] = conditions_515[0].
            conditions = get_conditions(nsd_dir, subj, n_sessions)  # list of len=N_sessions. Each item contains 750_nsd_ids
            conditions = np.asarray(conditions).ravel()  # reshape to [N_images_seen,] (30000 for subjects who did all conditions)
            conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]  # get valid trials for which we do have 3 repetitions.
            conditions_sampled = conditions[conditions_bool]  # shape=[N_images_seen,] (30000 for subjects who did all conditions 3x)
            sample = np.unique(conditions[conditions_bool])  # shape=[N_ordered_unique_nsd_ids,] (10000) for thorough subjects.
            all_conditions = range(sample.shape[0])
            n_samples = int(np.round(sample.shape[0] / 100))

            
            subj_model_rdms = {}

            for MODEL_NAME in MODEL_NAMES:

                # fetch the model RDMs
                # (filt should be a wildcard to catch correct model rdms, careful not to catch other models)
                these_rdms, _ = get_model_rdms(f"{models_dir}/{MODEL_NAME}", subj, filt=MODEL_NAME)

                if "dnn_" in MODEL_NAME.lower():
                    model_rdm_idx = dnn_layer_to_use
                else:
                    # otherwise, there is just one rdm anyway, so we use it
                    model_rdm_idx = 0

                subj_model_rdms[MODEL_NAME] = these_rdms[model_rdm_idx]  # [1, model_rdm_size]


            for this_model_pair in pairs:

                this_model_pair_str = "_vs_".join(this_model_pair)

                if this_model_pair_str in model_rdms_corrs[subj].keys() and not OVERWRITE:
                    # if we are completing an existing file, no need to create the subj dict
                    print(f"Correlation for {this_model_pair_str} already computed for {subj}. Skipping.")
                    pass
                else:

                    # initialise batch generator
                    batchg_model1 = BatchGen(subj_model_rdms[this_model_pair[0]], all_conditions)
                    # initialise batch generator
                    batchg_model2 = BatchGen(subj_model_rdms[this_model_pair[1]], all_conditions)

                    # path the sample_ids used in searchlight analysis for fair comparison
                    saved_samples_file = os.path.join(base_save_dir, f"searchlight_respectedsampling_{rdm_distance}",
                                                    f"{subj}", "saved_sampling", f"{subj}_nsd-allsubstim_sampling.npy")
                    sample_pool = np.load(saved_samples_file, allow_pickle=True)

                    these_corrs = []
                    for j in range(n_samples):
                        print(f"\rworking on usual case: boot {j}", end="")

                        # sample 100 stimuli from the subject's sample.
                        choices = sample_pool[j]

                        # now get the sampled 100x100 rdms for model and roi and correlate
                        # this returns 1_modelx(upper_tri_sampled_model_rdm)
                        model_rdm_sample = np.asarray(batchg_model1.index_rdms(choices), dtype=np.float32)
                        roi_rdm_sample = np.asarray(batchg_model2.index_rdms(choices), dtype=np.float32)

                        these_corrs.append(corr_rdms(model_rdm_sample, roi_rdm_sample))
                    model_rdms_corrs[subj][this_model_pair_str] = np.mean(these_corrs)

        model_rdms_corrs['subj_mean'] = {}
        model_rdms_corrs['subj_std'] = {}
        for pair_str in pair_strings:
            model_rdms_corrs['subj_mean'][pair_str] = np.mean([model_rdms_corrs[subj][pair_str] for subj in subs])
            model_rdms_corrs['subj_std'][pair_str] = np.std([model_rdms_corrs[subj][pair_str] for subj in subs])


        with open(f"{results_dir}/model_rdms_corrs.pkl", "wb") as fp:
            pickle.dump(model_rdms_corrs, fp)


    n_bars = len(pair_strings)
    bar_alphas = np.linspace(0.1, 1, len(pair_strings))
    bar_specs = {
        "width": 0.1,  # rough estimate of what will look good
        "edgecolor": "black",
        "linewidth": 0.7,
        "zorder": 10,
    }
    plot_color = "blue"

    fig, ax = plt.subplots(figsize=((n_bars)*2, 5))  # rough estimate of what will look good
    bar_width = bar_specs["width"]
    x_positions = []

    for i, bar_name in enumerate(pair_strings):
        bar_alpha = bar_alphas[i]
        facecolor = mcolors.to_rgb(plot_color) + (bar_alpha,)
        perf = model_rdms_corrs['subj_mean'][bar_name]
        std = model_rdms_corrs['subj_std'][bar_name]/np.sqrt(n_subjects)
        bar_pos = i * bar_width
        ax.bar(
            bar_pos,
            perf,
            **bar_specs,
            facecolor=facecolor,
            label="_no_legend_",
        )  # roi_label)
        ax.errorbar(bar_pos, perf, yerr=std, color="black", capsize=3, zorder=11)
        x_positions.append(bar_pos)

    # Add axis labels
    ax.set_ylabel("Model-model pearson correlation\n(mean across samples seen by each subject)")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])#, fontsize="smaller")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pair_strings, rotation=45, ha="right", fontsize="small")
    plt.tight_layout()

    plt.savefig(f"{results_dir}/model_rdms_corrs_{plt_suffix}.png")
