import os
import pickle
import time
import numpy as np
from scipy.spatial.distance import pdist
from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import (
    get_conditions,
    get_conditions_515,
    get_model_rdms,
    get_rois,
)
from nsd_visuo_semantics.utils.tf_utils import corr_rdms

COMPUTE = False  # if False, load rdm correlations and skip directly to postprocessing/plotting
overwrite = (
    False  # if True, overwrite existing stuff. If False, load existing stuff
)

# model names for which to compute ROI-wise correlations with brain activities
MODEL_NAMES = [
    "multihot",
    "mpnet",
    "fasttext_nouns",
    "nsd_fasttext_nouns_closest_cocoCats_cut0.33",
    "dnn_multihot_rec",
    "dnn_mpnet_rec"
]

# if we are using a DNN, use last layer (and last timestep if recurrent). If you want another layer,
# find its index (between in [0 ,n_layers*n_timesteps-1]) and apply it here. Ignored if "dnn_" not in MODEL_NAME
dnn_layer_to_use = -1

remove_shared_515 = False  # if True, use RDMs computed without the 515 stimuli seen by all subjects
rdm_distance = "correlation"  # 'cosine', 'correlation', etc. NOTE: ONLY USED FOR HUMANS. FOR MODELS, WE ARE USING PRECOMPUTED RDMS WITH CORRELATION DIST

n_jobs = 38
n_sessions = 40
n_subjects = 8
subs = [f"subj0{x+1}" for x in range(n_subjects)]
which_rois = (
    "streams"  # streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...
)

# set up directories
base_dir = os.path.join("/rds", "projects", "c", "charesti-start")
nsd_dir = os.path.join(base_dir, "data", "NSD")
betas_dir = os.path.join(base_dir, "projects", "NSD", "derivatives", "betas")

base_save_dir = "./save_dir"
models_dir = os.path.join(
    base_save_dir,
    f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}',
)
roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
os.makedirs(roi_analyses_dir, exist_ok=True)
results_dir = os.path.join(
    roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}"
)
os.makedirs(results_dir, exist_ok=True)
subj_roi_rdms_path = os.path.join(results_dir, "subj_roi_rdms")
os.makedirs(subj_roi_rdms_path, exist_ok=True)

# we use the fsaverage space.
targetspace = "fsaverage"

# Get roi info
maskdata, ROIS = get_rois(
    which_rois, os.path.join(roi_analyses_dir, "roi_defs")
)

if COMPUTE:
    # we'll also compute correlations between subjects on the 515 images that all subjects saw 3 times (for noise ceiling).
    conditions_515 = get_conditions_515(nsd_dir)
    conditions_515 = np.asarray(conditions_515).ravel()

    # here we go
    all_corrs = {}
    for subj in subs:
        # Betas per subject
        betas_file = os.path.join(
            betas_dir, f"{subj}_betas_average_{targetspace}.npy"
        )
        print(f"loading betas for {subj}")
        betas = np.load(betas_file, allow_pickle=True)

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
        conditions = get_conditions(
            nsd_dir, subj, n_sessions
        )  # list of len=N_sessions. Each item contains 750_nsd_ids
        conditions = np.asarray(
            conditions
        ).ravel()  # reshape to [N_images_seen,] (30000 for subjects who did all conditions)
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions
        ]  # get valid trials for which we do have 3 repetitions.
        conditions_sampled = conditions[
            conditions_bool
        ]  # shape=[N_images_seen,] (30000 for subjects who did all conditions 3x)
        sample = np.unique(
            conditions[conditions_bool]
        )  # shape=[N_ordered_unique_nsd_ids,] (10000) for thorough subjects.
        all_conditions = range(sample.shape[0])
        n_samples = int(np.round(sample.shape[0] / 100))
        subj_indices_515 = [
            x for x, j in enumerate(sample) if j in conditions_515
        ]

        # save the subject's full ROI RDMs
        roi_rdms = []
        for roi in range(1, len(ROIS)):
            mask_name = ROIS[roi]
            rdm_full_file = os.path.join(
                subj_roi_rdms_path,
                f"{subj}_{mask_name}_fullrdm_{rdm_distance}.npy",
            )
            rdm_515_file = os.path.join(
                subj_roi_rdms_path,
                f"{subj}_{mask_name}_515rdm_{rdm_distance}.npy",
            )

            if (
                overwrite
                or not os.path.exists(rdm_full_file)
                or not os.path.exists(rdm_515_file)
            ):
                print(f"Gathering betas for ROI: {rdm_515_file}")
                # maskdata is an array of shape [n_voxels,], with a number corresponding to the
                # ROI of each voxel (e.g. 0 means no ROI is associated with this voxel, 1 means voxel
                # is in ROIS[1] (EVC for example), etcetc).
                # so vs_mask is a logical array of mask vertices, with True in ROI vertices
                vs_mask = maskdata == roi

                # betas is [n_voxels, n_subj_conditions] (ordered by nsd_id, see thorough comments above).
                # masked betas is [n_roi_betas, n_conditions]
                masked_betas = betas[vs_mask, :]
                # remove vertices with a nan
                good_vox = [
                    True if np.sum(np.isnan(x)) == 0 else False
                    for x in masked_betas
                ]
                if np.sum(good_vox) != len(good_vox):
                    print(f"found some NaN for ROI: {mask_name} - {subj}")
                masked_betas = masked_betas[good_vox, :]

            if overwrite or not os.path.exists(rdm_full_file):
                # prepare for cosine distance
                X = (
                    masked_betas.T
                )  # [n_conditions, n_roi_betas], i.e., we make an [n_conditionsxn_conditions] rdm

                print(f"computing RDM for roi: {mask_name}")
                start_time = time.time()
                rdm = pdist(X, metric=rdm_distance)
                if np.any(np.isnan(rdm)):
                    raise ValueError(f"nan found in RDM for ROI {mask_name}")

                elapsed_time = time.time() - start_time
                print(
                    "elapsedtime: ",
                    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}',
                )

                print(f"saving full rdm for {mask_name} : {subj}")
                np.save(rdm_full_file, rdm)
                roi_rdms.append(rdm.astype(np.float32))

            if overwrite or not os.path.exists(rdm_515_file):
                # rdm on the 515 conditions seen by all subjects (used later to compute the noise ceiling)
                print(f"computing {subj} 515_RDM for roi: {mask_name}")

                masked_betas_515 = masked_betas[
                    :, subj_indices_515
                ]  # [n_roi_betas, 515]

                # prepare for cosine distance
                X = masked_betas_515.T  # [515, n_roi_betas]
                rdm_515 = pdist(X, metric=rdm_distance)  # [515, 515]
                if np.any(np.isnan(rdm_515)):
                    raise ValueError(f"nan found in RDM for ROI {mask_name}")

                print(f"saving 515 shared images rdm for {mask_name} : {subj}")
                np.save(rdm_515_file, rdm_515)

            else:
                print(f"loading full rdm for {mask_name} : {subj}")
                # rdm_515 will be loaded later when we compute the noise ceiling, for now we're just correlating subj&model
                rdm = np.load(rdm_full_file, allow_pickle=True)
                roi_rdms.append(rdm.astype(np.float32))

        # # make some rdm figures
        rdm_figures = os.path.join(results_dir, "rdm_figures")
        os.makedirs(rdm_figures, exist_ok=True)

        # This concludes the brain data rdm computations. Now, we move to model rdms
        all_corrs[subj] = {}
        for MODEL_NAME in MODEL_NAMES:
            # fetch the model RDMs
            model_rdms, model_names = get_model_rdms(
                f"{models_dir}/{MODEL_NAME}", subj, filt=MODEL_NAME
            )  # (filt should be a wildcard to catch correct model rdms, careful not to catch other models)

            if "dnn_" in MODEL_NAME.lower():
                model_rdm_idx = dnn_layer_to_use
            else:
                # otherwise, there is just one rdm anyway, so we use it
                model_rdm_idx = 0

            model_rdm = model_rdms[
                model_rdm_idx
            ]  # always shape [1, model_rdm_size] (as expected by batchg, etc)

            rdm_fig_file = os.path.join(
                rdm_figures, f"{subj}_{MODEL_NAME}_rdm_norank.svg"
            )
            # if overwrite or not os.path.exists(rdm_fig_file):
            #     # make a figure of the RDM.
            #     plt.imshow(squareform(rankdata(model_rdm)), cmap='magma')  # rdm_colormap())
            #     plt.colorbar()
            #     plt.savefig(rdm_fig_file)
            #     plt.close('all')
            #     plt.imshow(squareform(model_rdm), cmap='magma')  # rdm_colormap())
            #     plt.colorbar()
            #     rdm_fig_file = os.path.join(rdm_figures, f'{subj}_{MODEL_NAME}_rdm_norank.svg')
            #     plt.savefig(rdm_fig_file)

            # initialise batch generator
            batchg_model = BatchGen(model_rdm, all_conditions)

            all_corrs[subj][MODEL_NAME] = {}
            # Compute correlations between model and all ROI rdms
            for roi in range(1, len(ROIS)):
                # compute & save, or find and load, the data for that subject
                save_rdmcorrs = os.path.join(
                    subj_roi_rdms_path,
                    f"{subj}_{MODEL_NAME}_{ROIS[roi]}ROI_corrs.npy",
                )  # nr for new roi

                if overwrite or not os.path.exists(save_rdmcorrs):
                    batchg_roi = BatchGen(
                        roi_rdms[roi - 1], all_conditions
                    )  # -1 because roi indexing starts at 1

                    these_corrs = []

                    # path the sample_ids used in searchlight analysis for fair comparison
                    saved_samples_file = os.path.join(
                        base_save_dir,
                        f"searchlight_respectedsampling_{rdm_distance}",
                        f"{subj}",
                        "saved_sampling",
                        f"{subj}_nsd-allsubstim_sampling.npy",
                    )
                    sample_pool = np.load(
                        saved_samples_file, allow_pickle=True
                    )

                    start_time = time.time()
                    for j in range(n_samples):
                        print(f"\rworking on usual case: boot {j}", end="")

                        # sample 100 stimuli from the subject's sample.
                        choices = sample_pool[j]

                        # now get the sampled 100x100 rdms for model and roi and correlate
                        # this returns 1_modelx(upper_tri_sampled_model_rdm)
                        model_rdm_sample = np.asarray(
                            batchg_model.index_rdms(choices)
                        )
                        roi_rdm_sample = np.asarray(
                            batchg_roi.index_rdms(choices)
                        )

                        these_corrs.append(
                            corr_rdms(model_rdm_sample, roi_rdm_sample)
                        )

                    elapsed_time = time.time() - start_time
                    print(
                        f' - elapsedtime: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
                    )
                    np.save(save_rdmcorrs, these_corrs)

                else:
                    these_corrs = np.load(save_rdmcorrs, allow_pickle=True)

                all_corrs[subj][MODEL_NAME][f"{ROIS[roi]}ROI"] = np.squeeze(
                    these_corrs
                )

    with open(f"{results_dir}/all_{rdm_distance}_rdm_corrs.pkl", "wb") as fp:
        pickle.dump(all_corrs, fp)

else:
    with open(f"{results_dir}/all_{rdm_distance}_rdm_corrs.pkl", "rb") as fp:
        all_corrs = pickle.load(fp)

# get ceiling of explainable variance (i.e., Compute the average RDM for n-1 people and correlate with
# the left out one. Do this N times for all participants, average the correlation values).
roi_noise_ceilings_per_sub_path = os.path.join(
    subj_roi_rdms_path, f"noise_ceilings_rois_515_{rdm_distance}_per_sub.pkl"
)

roi_noise_ceilings_per_sub = {
    f"{roi_name}ROI": {}
    for roi_name in ROIS.values()
    if roi_name.lower() != "unknown"
}

for roi in range(1, len(ROIS)):
    mask_name = ROIS[roi]
    per_left_out_corrs = np.zeros(n_subjects)
    for s, subj in enumerate(subs):
        left_out_subj_rdm = np.load(
            os.path.join(
                subj_roi_rdms_path,
                f"{subj}_{mask_name}_515rdm_{rdm_distance}.npy",
            ),
            allow_pickle=True,
        )

        mean_of_others_rdm = np.zeros_like(left_out_subj_rdm)
        for o, other in enumerate(subs):
            if other != subj:
                mean_of_others_rdm += np.load(
                    os.path.join(
                        subj_roi_rdms_path,
                        f"{other}_{mask_name}_515rdm_{rdm_distance}.npy",
                    ),
                    allow_pickle=True,
                )
        mean_of_others_rdm /= n_subjects - 1

        per_left_out_corrs[s] = corr_rdms(
            left_out_subj_rdm[None, :], mean_of_others_rdm[None, :]
        )  # [None,:] adds batch dim

        roi_noise_ceilings_per_sub[f"{ROIS[roi]}ROI"][
            subj
        ] = per_left_out_corrs[s]

with open(roi_noise_ceilings_per_sub_path, "wb") as f:
    pickle.dump(roi_noise_ceilings_per_sub, f)

# get mean corrs
mean_corrs = {}
for subj in all_corrs.keys():  # ['subj01', ..., 'subj08']
    mean_corrs[subj] = {}
    for model in all_corrs[subj].keys():  # [model_name_1, ...]
        mean_corrs[subj][model] = {}
        for roi in all_corrs[subj][model].keys():  # ['earlyROI', ...]
            # each all_corrs[k1][k2][k3] is 100 numbers (1 corr per 100 split). We mavg and divide by subj's noiseCeil
            mean_corrs[subj][model][roi] = (
                np.mean(all_corrs[subj][model][roi])
                / roi_noise_ceilings_per_sub[roi][subj]
            )

group_corrs = {}
group_mean_corrs = {}
group_std_corrs = {}
for model_name in mean_corrs[list(mean_corrs.keys())[0]].keys():
    group_corrs[model_name] = {}
    group_mean_corrs[model_name] = {}
    group_std_corrs[model_name] = {}
    for roi in mean_corrs[list(mean_corrs.keys())[0]][model_name].keys():
        group_corrs[model_name][roi] = []
        group_mean_corrs[model_name][roi] = 0
        group_std_corrs[model_name][roi] = 0
        for subj in mean_corrs.keys():
            group_corrs[model_name][roi].append(
                mean_corrs[subj][model_name][roi]
            )
        group_mean_corrs[model_name][roi] = np.mean(
            group_corrs[model_name][roi]
        )
        group_std_corrs[model_name][roi] = np.std(group_corrs[model_name][roi])

with open(
    f"{subj_roi_rdms_path}/subjWiseNoiseCeiling_group_corrs.pkl", "wb"
) as fp:
    pickle.dump(group_corrs, fp)
with open(
    f"{subj_roi_rdms_path}/subjWiseNoiseCeiling_group_mean_corrs.pkl", "wb"
) as fp:
    pickle.dump(group_mean_corrs, fp)
with open(
    f"{subj_roi_rdms_path}/subjWiseNoiseCeiling_group_std_corrs.pkl", "wb"
) as fp:
    pickle.dump(group_std_corrs, fp)
