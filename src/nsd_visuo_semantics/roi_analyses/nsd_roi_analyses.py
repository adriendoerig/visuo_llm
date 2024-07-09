'''Compute ROI-wise correlations between model RDMs and brain RDMs for NSD data.'''

import os, pickle
import numpy as np
from scipy.spatial.distance import pdist
from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_subject_conditions, get_conditions_515, get_conditions_100, get_model_rdms, get_rois, load_or_compute_betas_average
from nsd_visuo_semantics.utils.utils import corr_rdms
from nsd_visuo_semantics.roi_analyses.roi_utils import get_all_subj_all_roi_neural_rdms


def nsd_roi_analyses(MODEL_NAMES, rdm_distance, dnn_layer_to_use, which_rois,
                     nsd_dir, betas_dir, rois_dir, base_save_dir,
                     OVERWRITE_NEURO_RDMs, OVERWRITE_RDM_CORRs):
    
    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]
    targetspace = "fsaverage"

    maskdata, ROIS = get_rois(which_rois, rois_dir)

    # set up directories
    model_rdm_dir = os.path.join(base_save_dir,f'serialised_models_{rdm_distance}')
    roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
    os.makedirs(roi_analyses_dir, exist_ok=True)
    results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")
    os.makedirs(results_dir, exist_ok=True)
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # load neural RDMs if they exist, otherwise compute them
    all_neural_rdms = {}
    for which_data in ['special515', 'fullnsd']:
        rdms_path = f'{cache_dir}/{which_rois}_all_neural_rdms_{rdm_distance}_{which_data}.pkl'
        if os.path.exists(rdms_path) and not OVERWRITE_NEURO_RDMs:
            with open(rdms_path, "rb") as pickle_file:
                all_neural_rdms[which_data] = pickle.load(pickle_file)
        else:
            all_neural_rdms[which_data] = get_all_subj_all_roi_neural_rdms(nsd_dir, betas_dir, which_rois, rois_dir, which_data, rdm_distance, cache_dir, targetspace)

    # we will save all model-brain corrs in a dict. Load it if it exists, otherwise create it.
    all_corrs_path = f"{cache_dir}/all_{rdm_distance}_rdm_corrs.pkl"
    if os.path.exists(all_corrs_path):
        with open(all_corrs_path, "rb") as pickle_file:
            all_corrs = pickle.load(pickle_file)
    else:
        all_corrs = {}

    # here we go
    for subj in subs:

        # we will need to know the stimuli that each subject saw
        conditions, conditions_sampled, sample = get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True)
        all_conditions = range(sample.shape[0])

        if subj in all_corrs.keys():
            # if we are completing an existing file, no need to create the subj dict
            pass
        else:
            all_corrs[subj] = {}

        for MODEL_NAME in MODEL_NAMES:

            if MODEL_NAME in all_corrs[subj].keys():
                # if we are completing an existing file, no need to create the subj dict
                pass
            else:
                all_corrs[subj][MODEL_NAME] = {}

            # find name where the model RDM was saved based on MODEL_NAME
            # (filt should be a wildcard to catch correct model rdms, careful not to catch other models)
            if '_layer' in MODEL_NAME:
                MODEL_NAME_NO_LAYER = MODEL_NAME.split('_layer')[0]
            else:
                MODEL_NAME_NO_LAYER = MODEL_NAME
            if "dnn_" in MODEL_NAME.lower():
                model_rdm_idx = dnn_layer_to_use
            else:
                # otherwise, there is just one rdm anyway, so we use it
                model_rdm_idx = 0
            
            # get model RDM for full NSD
            model_rdms, model_names = get_model_rdms(f"{model_rdm_dir}/{MODEL_NAME_NO_LAYER}", subj, filt=MODEL_NAME_NO_LAYER)
            model_rdm = model_rdms[model_rdm_idx]  # always shape [1, model_rdm_size] (as expected by batchg, etc)
            
            # initialise model_rdm batch generator (we will sample the same 100x100 sub rdms 
            # as in the searchlight analyses, to ensure consistency)
            # NOTE: this model RDM is subject-specific. It only contains the stimuli that the subject saw. 
            batchg_model = BatchGen(model_rdm, all_conditions)

            # Compute correlations between model and all ROI rdms
            for roi in range(1, len(ROIS)):

                if f"{ROIS[roi]}" in all_corrs[subj][MODEL_NAME].keys() and not OVERWRITE_RDM_CORRs:
                    print(f'Found model corrs for {MODEL_NAME} {ROIS[roi]}. Skipping.')
                    continue
                else:
                    print(f'Computing model-brain corr for {MODEL_NAME} {ROIS[roi]}.')

                    # initialise roi_rdm batch generator (we will sample the same 100x100 sub rdms 
                    # as in the searchlight analyses, to ensure consistency)
                    this_neural_data = all_neural_rdms['fullnsd'][subj][ROIS[roi]]
                    batchg_roi = BatchGen(this_neural_data, all_conditions)

                    # use the sample_ids used in searchlight analysis for fair comparison
                    saved_samples_file = os.path.join(base_save_dir, f"searchlight_respectedsampling_{rdm_distance}", f"{subj}", "saved_sampling", f"{subj}_nsd-allsubstim_sampling.npy")
                    sample_pool = np.load(saved_samples_file, allow_pickle=True)
                    these_corrs = []
                    for j in range(len(sample_pool)):
                        print(f"\r\Correlating subsampled rdms: {j}", end="")
                        # sample 100 stimuli from the subject's sample.
                        choices = sample_pool[j]
                        # now get the sampled 100x100 rdms for model and roi and correlate
                        # this returns 1_modelx(upper_tri_sampled_model_rdm)
                        model_rdm_sample = np.asarray(batchg_model.index_rdms(choices), dtype=np.float32)
                        roi_rdm_sample = np.asarray(batchg_roi.index_rdms(choices), dtype=np.float32)
                        these_corrs.append(corr_rdms(model_rdm_sample, roi_rdm_sample))

                    all_corrs[subj][MODEL_NAME][f"{ROIS[roi]}"] = np.mean(these_corrs)

                    # we save to file after each roi to avoid losing data
                    with open(f"{cache_dir}/all_{rdm_distance}_rdm_corrs.pkl", 'wb') as pickle_file:
                        pickle.dump(all_corrs, pickle_file)
    


    # get sibject-wise noise ceiling of explainable variance (i.e., Compute the average RDM 
    # for n-1 other subject and correlate with the subject's rdm for each roi.
    roi_noise_ceilings_per_sub_path = f"{cache_dir}/noise_ceilings_rois_515_{rdm_distance}_per_sub.pkl"
    roi_noise_ceilings_per_sub = {ROIS[roi]: {} for roi in range(1,len(ROIS))}

    for roi in range(1, len(ROIS)):

        roi_name = ROIS[roi]
        per_left_out_corrs = np.zeros(n_subjects)

        for s, subj in enumerate(subs):

            left_out_subj_rdm = all_neural_rdms['special515'][subj][roi_name]

            mean_of_others_rdm = np.zeros_like(left_out_subj_rdm)
            for o, other in enumerate(subs):
                if other != subj:
                    mean_of_others_rdm += all_neural_rdms['special515'][other][roi_name]
            mean_of_others_rdm /= n_subjects - 1

            per_left_out_corrs[s] = corr_rdms(left_out_subj_rdm[None, :], mean_of_others_rdm[None, :])  # [None,:] adds batch dim

            roi_noise_ceilings_per_sub[ROIS[roi]][subj] = per_left_out_corrs[s]

    with open(roi_noise_ceilings_per_sub_path, 'wb') as pickle_file:
        pickle.dump(roi_noise_ceilings_per_sub, pickle_file)

    # apply the subject-wise noise ceilings
    all_corrs_noise_ceiling_corrected = {}
    for subj in all_corrs.keys():  # ['subj01', ..., 'subj08']
        all_corrs_noise_ceiling_corrected[subj] = {}
        for model in all_corrs[subj].keys():  # [model_name_1, ...]
            all_corrs_noise_ceiling_corrected[subj][model] = {}
            for roi in all_corrs[subj][model].keys():  # ['earlyROI', ...]
                all_corrs_noise_ceiling_corrected[subj][model][roi] = all_corrs[subj][model][roi]/roi_noise_ceilings_per_sub[roi][subj]


    # get subject corrs, as well as mean and std across subjects 
    # and format each as dict[model][roi][subj] (expected later by plotting functions)
    group_corrs = {}
    group_mean_corrs = {}
    group_std_corrs = {}
    group_corrs_noise_ceiling_corrected = {}
    group_mean_corrs_noise_ceiling_corrected = {}
    group_std_corrs_noise_ceiling_corrected = {}
    for model_name in all_corrs[list(all_corrs.keys())[0]].keys():
        group_corrs[model_name] = {}
        group_mean_corrs[model_name] = {}
        group_std_corrs[model_name] = {}
        group_corrs_noise_ceiling_corrected[model_name] = {}
        group_mean_corrs_noise_ceiling_corrected[model_name] = {}
        group_std_corrs_noise_ceiling_corrected[model_name] = {}
        for roi in all_corrs[list(all_corrs.keys())[0]][model_name].keys():
            group_corrs[model_name][roi] = []
            group_mean_corrs[model_name][roi] = 0
            group_std_corrs[model_name][roi] = 0
            group_corrs_noise_ceiling_corrected[model_name][roi] = []
            group_mean_corrs_noise_ceiling_corrected[model_name][roi] = 0
            group_std_corrs_noise_ceiling_corrected[model_name][roi] = 0
            for subj in all_corrs.keys():
                group_corrs_noise_ceiling_corrected[model_name][roi].append(all_corrs_noise_ceiling_corrected[subj][model_name][roi])
                group_corrs[model_name][roi].append(all_corrs[subj][model_name][roi])
            group_mean_corrs[model_name][roi] = np.mean(group_corrs[model_name][roi])
            group_std_corrs[model_name][roi] = np.std(group_corrs[model_name][roi])
            group_mean_corrs_noise_ceiling_corrected[model_name][roi] = np.mean(group_corrs_noise_ceiling_corrected[model_name][roi])
            group_std_corrs_noise_ceiling_corrected[model_name][roi] = np.std(group_corrs_noise_ceiling_corrected[model_name][roi])

    with open(f"{cache_dir}/group_corrs_no_noise_ceiling.pkl", 'wb') as pickle_file:
        pickle.dump(group_corrs, pickle_file)
    with open(f"{cache_dir}/group_mean_corrs_no_noise_ceiling.pkl", 'wb') as pickle_file:
        pickle.dump(group_mean_corrs, pickle_file)
    with open(f"{cache_dir}/group_std_corrs_no_noise_ceiling.pkl", 'wb') as pickle_file:
        pickle.dump(group_std_corrs, pickle_file)
    with open(f"{cache_dir}/subjWiseNoiseCeiling_group_corrs.pkl", 'wb') as pickle_file:
        pickle.dump(group_corrs_noise_ceiling_corrected, pickle_file)
    with open(f"{cache_dir}/subjWiseNoiseCeiling_group_mean_corrs.pkl", 'wb') as pickle_file:
        pickle.dump(group_mean_corrs_noise_ceiling_corrected, pickle_file)
    with open(f"{cache_dir}/subjWiseNoiseCeiling_group_std_corrs.pkl", 'wb') as pickle_file:
        pickle.dump(group_std_corrs_noise_ceiling_corrected, pickle_file)
