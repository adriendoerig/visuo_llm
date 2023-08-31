"""[nds_get_data]

    utilies for nsd
"""
import glob
import os
import re

import numpy as np
import pandas as pd

ppdata_folder = os.path.join(
    "/rds",
    "projects",
    "c",
    "charesti-start",
    "data",
    "NSD",
    "nsddata",
    "ppdata",
)

behaviour_file = os.path.join(
    ppdata_folder, "{subject}", "behav", "responses.tsv"
)


def read_behavior(behaviour_file, subject, session_index, trial_index=[]):
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

    behavior = pd.read_csv(
        behaviour_file.format(subject=subject), delimiter="\t"
    )

    # the behavior is encoded per run.
    # I'm now setting this function up so that it aligns with the timepoints in the fmri files,
    # i.e. using indexing per session, and not using the 'run' information.
    session_behavior = behavior[behavior["SESSION"] == session_index]

    if len(trial_index) == 0:
        trial_index = slice(0, len(session_behavior))

    return session_behavior.iloc[trial_index]


def get_stim_ids(data_store, subject):
    """[return sorted stim ids]
    Args:
        data_store ([json]): [ma task json data_store]
        subject ([string]): [meadows subject name]
    Returns:
        indcs ([list]): [indices for stim sorting]
        stim_ids ([list]): [names of the sorted stimuli]
    """
    stimuli = data_store[subject]["tasks"][1]["stimuli"]
    # get the stimulus names from the data_store dictionary

    stim_ids = [x["name"] for x in stimuli]
    # get the 73K ids (used later for reading in the images)
    # get the nsd integer value from the stimuli variable

    stim_ids_np = np.asarray(
        [int(re.split("nsd", x["name"])[1]) for x in stimuli]
    )
    indcs = np.argsort(stim_ids_np)
    stim_ids = [stim_ids_np[i] for i in indcs]
    # sort the ids (nsa.read_images needs sorted indices)
    # by index sorting then listing using these indexes

    return stim_ids, indcs


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


def get_100(nsd_dir):
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
    # here we remove 1 to account for the difference
    # between matlab's 1-based indexing and python's 0-based
    # indexing.
    chosen_100 = np.asarray(chosen_100) - 1

    chosen_ids = list(np.asarray(stim_ids)[chosen_100])

    return chosen_ids


def get_conditions_515(nsd_dir, n_sessions=40):
    """[get condition indices for the special 515 images.]

    Arguments:
        nsd_dir {[os.path]} -- [where is the nsd data?]

    Returns:
        [lit of inds] -- [indices related to the special
                          515 stimuli in a coco format]
    """

    stim_1000 = get_1000(nsd_dir)

    sub_conditions = []
    # loop over sessions
    for sub in range(8):
        subix = f"subj0{sub+1}"
        # extract conditions data and reshape conditions to be ntrials x 1
        conditions = np.asarray(
            get_conditions(nsd_dir, subix, n_sessions)
        ).ravel()

        # find the 3 repeats
        conditions_bool = [
            True if np.sum(conditions == x) == 3 else False for x in conditions
        ]
        conditions = conditions[conditions_bool]

        conditions_1000 = [x for x in stim_1000 if x in conditions]
        print(f"{subix} saw {len(conditions_1000)} of the 1000")

        if sub == 0:
            sub_conditions = conditions_1000
        else:
            sub_conditions = [
                x for x in conditions_1000 if x in sub_conditions
            ]

    return sub_conditions


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
        this_ses = np.asarray(
            read_behavior(behaviour_file, subject=sub, session_index=ses_i)[
                "73KID"
            ]
        )

        # these are the 73K ids.
        valid_trials = [j for j, x in enumerate(this_ses)]

        # this skips if say session 39 doesn't exist for subject x
        # (see n_sessions comment above)
        if valid_trials:
            conditions.append(this_ses)

    return conditions
