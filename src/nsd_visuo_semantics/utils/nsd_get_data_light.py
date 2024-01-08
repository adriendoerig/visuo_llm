"""[nds_get_data]
    utilies for nsd
"""

import glob
import json
import os
import re
import nibabel as nb
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import zscore
# from nsd_visuo_semantics.utils.utils import average_over_conditions


def get_model_rdms(models_dir, subj, filt=None, only_names=False):
    # filt is model name (e.g. fasttext_verbs_mean) - careful, uses a wildcard, so the wildcard must be specific
    if filt is not None:
        model_files = glob.glob(os.path.join(models_dir, f"{subj}_{filt}*_fullrdm.npy"))
    else:
        model_files = glob.glob(os.path.join(models_dir, f"{subj}*_fullrdm.npy"))
    model_files.sort()

    model_names = [re.split(f"{subj}_", re.split("_fullrdm.npy", os.path.basename(model_file))[0])[1]
        for model_file in model_files]

    if not only_names:
        all_rdms = [np.load(model_file).astype(np.float32) for model_file in model_files]
        if len(all_rdms) == 0:
            raise Exception(f"No rdm files found in {models_dir}.")
        return all_rdms, model_names
    else:
        return model_names


def get_masks(nsd_dir, sub, targetspace="func1pt8mm"):
    """[summary]

    Args:
        nsd_dir ([type]): [description]
        sub ([type]): [description]
        targetspace (str, optional): [description]. Defaults to 'func1pt8mm'.

    Returns:
        [type]: [description]
    """
    # initiate nsda
    ppdata_folder = os.path.join(nsd_dir, "nsddata", "ppdata")

    full_path = os.path.join(
        ppdata_folder, sub, targetspace, "brainmask.nii.gz"
    )

    brainmask = nb.load(full_path).get_fdata()

    return brainmask


def read_behavior(nsd_dir, subject, session_index, trial_index=[]):
    """read_behavior [summary]

    Parameters
    ----------
    subject : str
        subject identifier, such as 'subj01'
    session_index : int
        which session, counting from 0
    trial_index : list, optional
        which trials from this session's behavior to return, by default [], which returns all trials

    Returns
    -------
    pandas DataFrame
        DataFrame containing the behavioral information for the requested trials
    """
    nsd_folder = nsd_dir
    ppdata_folder = os.path.join(nsd_folder, "nsddata", "ppdata")

    behavior_file = os.path.join(
        ppdata_folder, f"{subject}", "behav", "responses.tsv"
    )

    behavior = pd.read_csv(behavior_file, delimiter="\t")

    # the behavior is encoded per run.
    # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
    # i.e. using indexing per session, and not using the 'run' information.
    session_behavior = behavior[behavior["SESSION"] == session_index]

    if len(trial_index) == 0:
        trial_index = slice(0, len(session_behavior))

    return session_behavior.iloc[trial_index]


def average_over_conditions(data, conditions, conditions_to_avg):
    lookup = np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims == 2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape
        avg_data = np.empty((x, y, z, n_conds))

    for j, x in enumerate(lookup):
        conditions_bool = conditions == x
        if n_dims == 2:
            if np.sum(conditions_bool) == 0:
                break
            # print((j, np.sum(conditions_bool)))
            sliced = data[:, conditions_bool]

            avg_data[:, j] = np.nanmean(sliced, axis=1)
        else:
            avg_data[:, :, :, j] = np.nanmean(
                data[:, :, :, conditions_bool], axis=3
            )

    return avg_data


def load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace):
    
    if not os.path.exists(betas_file):
        print('betas average not found, computing..')
        print('\tloading betas')

        # get betas
        betas = get_betas(nsd_dir, subj, n_sessions, targetspace=targetspace)

        # concatenate trials
        print('\tconcatenating betas across runs..')
        betas = np.concatenate(betas, axis=-1)

        # average betas across three repeats
        print(f'\taveraging betas for {subj}')
        betas = average_over_conditions(betas, conditions, conditions_sampled)

        # saving betas
        print(f'saving betas for {subj}')
        np.save(betas_file, betas, allow_pickle=True)
        
    else:
        print(f'loading betas for {subj}')
        betas = np.load(betas_file, allow_pickle=True)

    return betas


def get_betas(nsd_dir, sub, n_sessions, mask=None, targetspace="func1pt8mm"):
    
    nsddata_betas_folder = os.path.join(nsd_dir, "nsddata_betas", "ppdata")
    data_folder = os.path.join(nsddata_betas_folder, sub, targetspace, "betas_fithrf_GLMdenoise_RR")

    betas = []
    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses + 1
        si_str = str(ses_i).zfill(2)  # e.g. '01'

        print(f"\t\tsub: {sub} fetching betas for trials in session: {ses_i}")
        this_ses = read_behavior(nsd_dir, subject=sub, session_index=ses_i)
        # these are the 73K ids.
        ses_conditions = np.asarray(this_ses["73KID"])
        valid_trials = [j for j, x in enumerate(ses_conditions)]

        # this skips if say session 39 doesn't exist for subject x
        if valid_trials:
            if targetspace == "fsaverage":
                # no need to divide by 300 in this case
                cond_axis = -1
                # load lh
                img_lh = nb.load(os.path.join(data_folder, f"lh.betas_session{si_str}.mgh")).get_fdata().squeeze()
                # load rh
                img_rh = nb.load(os.path.join(data_folder, f"rh.betas_session{si_str}.mgh")).get_fdata().squeeze()
                # concatenate
                all_verts = np.vstack((img_lh, img_rh))
                # mask
                if mask is not None:
                    betas.append((zscore(all_verts, axis=cond_axis)[mask, :]).astype(np.float32))
                else:
                    betas.append((zscore(all_verts, axis=cond_axis)).astype(np.float32))

            elif targetspace == "func1pt8mm":
                # we will need to divide the loaded data by 300 in this case
                cond_axis = -1
                img = nb.load(os.path.join(data_folder, f"betas_session{si_str}.nii.gz")).get_fdata().squeeze()
                # img = nb.load(os.path.join(data_folder, f"betas_session{si_str}.nii.gz"))
                # re-hash the betas to save memory
                if mask is not None:
                    betas.append((zscore(img/300., axis=cond_axis)[mask, :]).astype(np.float32))
                else:
                    betas.append((zscore(img/300., axis=cond_axis)).astype(np.float32))

            else:
                raise Exception("targetspace not recognized")

    return betas


def get_conditions(nsd_dir, sub, n_sessions):
    """[summary]

    Args:
        nsd_dir ([type]): [description]
        sub ([type]): [description]
        n_sessions ([type]): [description]

    Returns:
        [type]: [description]
    """

    # read behaviour files for current subj
    conditions = []

    # loop over sessions
    for ses in range(n_sessions):
        ses_i = ses + 1
        print(f"\t\tsub: {sub} fetching condition trials in session: {ses_i}")

        # we only want to keep the shared_1000
        this_ses = np.asarray(read_behavior(nsd_dir, subject=sub, session_index=ses_i)["73KID"])

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return conditions


def get_1000(nsd_dir):
    """[get condition indices for the special 1000 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the 1000 special
                          stimuli in a coco format]
    """
    stim1000_dir = os.path.join(
        nsd_dir, "nsddata", "stimuli", "nsd", "shared1000", "*.png"
    )

    stim1000 = [os.path.basename(x)[:-4] for x in glob.glob(stim1000_dir)]
    stim1000.sort()
    stim_ids = [
        int(re.split("nsd", stim1000[x])[1]) for x, n in enumerate(stim1000)
    ]

    stim_ids = list(np.asarray(stim_ids))
    return stim_ids


def get_conditions_100(nsd_dir):
    """[get condition indices for the special chosen 100 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the chosen 100 special stimuli in a coco format]
    """

    stim_ids = get_1000(nsd_dir)
    # kendrick's chosen 100
    chosen_100 = [
        4,
        8,
        22,
        30,
        33,
        52,
        64,
        69,
        73,
        137,
        139,
        140,
        145,
        157,
        159,
        163,
        186,
        194,
        197,
        211,
        234,
        267,
        287,
        300,
        307,
        310,
        318,
        326,
        334,
        350,
        358,
        362,
        369,
        378,
        382,
        404,
        405,
        425,
        463,
        474,
        487,
        488,
        491,
        498,
        507,
        520,
        530,
        535,
        568,
        570,
        579,
        588,
        589,
        591,
        610,
        614,
        616,
        623,
        634,
        646,
        650,
        689,
        694,
        695,
        700,
        727,
        730,
        733,
        745,
        746,
        754,
        764,
        768,
        786,
        789,
        790,
        797,
        811,
        825,
        853,
        857,
        869,
        876,
        882,
        896,
        905,
        910,
        925,
        936,
        941,
        944,
        948,
        960,
        962,
        968,
        969,
        974,
        986,
        991,
        999,
    ]
    chosen_100 = np.asarray(chosen_100) - 1

    chosen_ids = list(np.asarray(stim_ids)[chosen_100])

    return chosen_ids


def get_conditions_515(nsd_dir, n_sessions=40):
    """[get condition indices for the special 515 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the special 515
                          stimuli in a coco format]
    """
    stim_1000 = get_1000(nsd_dir)

    sub_conditions = []
    # loop over sessions
    for sub in range(8):
        subix = f"subj0{sub+1}"
        # extract conditions data and reshape conditions to be ntrials x 1
        conditions = np.asarray(get_conditions(nsd_dir, subix, n_sessions)).ravel()

        # find the 3 repeats
        conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]

        conditions = conditions[conditions_bool]

        conditions_1000 = [x for x in stim_1000 if x in conditions]
        print(f"{subix} saw {len(conditions_1000)} of the 1000")

        if sub == 0:
            sub_conditions = conditions_1000
        else:
            sub_conditions = [x for x in conditions_1000 if x in sub_conditions]

    return sub_conditions


def get_sentence_lists(nsda, image_indices):
    """gets a list of captions from nsd given indices
    nsda must be an instance of NSDAccess: nsda = NSDAccess(nsd_dir)"""

    print('Careful with the indices! You may need to subtract 1 from them.')

    # Read in captions
    # print('reading coco captions for the requested images')
    captions = nsda.read_image_coco_info(image_indices, info_type="captions", show_annot=False)

    sentence_lists = []
    for caption in captions:
        image_capt = []
        for j, cap in enumerate(caption):
            image_capt.append(cap["caption"])
        sentence_lists.append(image_capt)

    return sentence_lists


def get_rois(which_rois, roi_defs_dir):
    roi_names_file = os.path.join(roi_defs_dir, f"{which_rois}.mgz.ctab")
    try:
        with open(roi_names_file) as f:
            # get ROI names automatically. If you don't have the .ctab file
            # you can also enter them by hand. 0 is always "Unknown")
            roi_id2name = {int(x[0]): x[2:-1] for x in f}
    except ValueError:
        print(
            f"roi_names_file not found. Requested {roi_names_file}. Using {which_rois} as single ROI name."
        )
        roi_id2name = {0: "Unknown"}
        roi_id2name[1] = which_rois

    # load the roi masks
    try:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.mgz")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.mgz")
        maskdata_lh = nb.load(lh_file).get_fdata().squeeze()
        maskdata_rh = nb.load(rh_file).get_fdata().squeeze()
    except ValueError:
        lh_file = os.path.join(roi_defs_dir, f"lh.{which_rois}.npy")
        rh_file = os.path.join(roi_defs_dir, f"rh.{which_rois}.npy")
        maskdata_lh = np.load(lh_file, allow_pickle=True)
        maskdata_rh = np.load(rh_file, allow_pickle=True)

    maskdata = np.hstack((maskdata_lh, maskdata_rh))

    return maskdata, roi_id2name
