import os
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_rois,get_sentence_lists, load_or_compute_betas_average


nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/special515_betas'
os.makedirs(results_dir, exist_ok=True)

n_sessions = 40
n_subjects = 8
subs = [f"subj0{x + 1}" for x in range(n_subjects)]
targetspace = "fsaverage"
n_voxels = 327684

conditions_515 = get_conditions_515(nsd_dir)

subj_test_betas_path = 

subj_test_betas = np.empty((n_subjects, 515, n_voxels))
for s, subj in enumerate(subs):

    # find indices that are in the 515 spacial test images
    # extract conditions data
    conditions = get_conditions(nsd_dir, subj, n_sessions)
    # we also need to reshape conditions to be ntrials x 1
    conditions = np.asarray(conditions).ravel()
    # then we find the valid trials for which we do have 3 repetitions.
    conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
    # and identify those.
    conditions_sampled = conditions[conditions_bool]
    # find the subject's condition list (sample pool)
    # this sample is the same order as the betas
    sample = np.unique(conditions[conditions_bool])
    # identify which image in the sample is a conditions_515
    sample_515_bool = [True if x in conditions_515 else False for x in sample]

    # Betas per subject
    print(f"loading betas for {subj}")
    betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
    betas_mean = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)

    good_vertex = [True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]
    if np.sum(good_vertex) != len(good_vertex):
        print(f"found some NaN for {subj}")
    betas_mean = betas_mean[good_vertex, :]

    # now we further split the brain data according to the 515 test set or the training set for that subject
    subj_test_betas[s] = betas_mean[:, sample_515_bool].T  # sub1: (515, 327673) (n_voxels may vary from subj to subj because of nans)

    del betas_mean  # make space                    
