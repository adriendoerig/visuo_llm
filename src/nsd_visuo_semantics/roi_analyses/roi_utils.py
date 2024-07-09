import os, pickle
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois, get_conditions_1000, get_conditions_515, get_conditions_100, get_subject_conditions, load_or_compute_betas_average
from nsd_visuo_semantics.utils.batch_gen import give_vector_pos
from scipy.spatial.distance import pdist

def get_neural_rdm(data, rdm_distance, roi_name='name_not_given'):
    
    print(f"\tcomputing neural RDM for roi: {roi_name}")
    rdm = pdist(data, metric=rdm_distance)
    if np.any(np.isnan(rdm)):
        raise ValueError(f"nan found in RDM for ROI {roi_name}")
    return rdm


def get_subj_all_roi_rdms(subj, betas, ROIS, maskdata, rdm_distance):

    print(f"computing neural RDMs for {subj}")
    
    subj_rdms = {}
    
    for roi in range(1, len(ROIS)):
        
        roi_name = ROIS[roi]

        # maskdata is an array of shape [n_voxels,], with a number corresponding to the
        # ROI of each voxel (e.g. 0 means no ROI is associated with this voxel, 1 means voxel
        # is in ROIS[1] (EVC for example), etcetc).
        # so vs_mask is a logical array of mask vertices, with True in ROI vertices
        vs_mask = maskdata == roi

        # betas is [n_voxels, n_subj_conditions] (ordered by nsd_id, see thorough comments above).
        # masked betas is [n_roi_betas, n_conditions]
        masked_betas = betas[vs_mask, :]
        # remove vertices with a nan
        good_vox = [True if np.sum(np.isnan(x)) == 0 else False for x in masked_betas]

        if np.sum(good_vox) != len(good_vox):
            print(f"found some NaN for ROI: {roi_name} - {subj}")
        masked_betas = masked_betas[good_vox, :]

        X = masked_betas.T  # [n_conditions, n_roi_betas], i.e., we make an [n_conditionsxn_conditions] rdm

        subj_rdms[roi_name] = get_neural_rdm(X, rdm_distance, roi_name)

    return subj_rdms


def get_all_subj_all_roi_neural_rdms(nsd_dir, betas_dir, which_rois, rois_dir, which_data, 
                                     rdm_distance, save_dir, targetspace):
    '''Get all neural RDMs for all ROIs for all subjects.
    which_rois should refer to a NSD ROI definition (e.g. 'streams') 
    rois_dir should be the path to the directory containing the ROIs.
    which_data should be either 'fullnsd', 'special1000', 'special515' or 'special100'.
    rdm_distance: 'correlation' or 'cosine', ...
    '''

    all_rdms = {}

    n_sessions = 40
    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]
    maskdata, ROIS = get_rois(which_rois, rois_dir)
    targetspace = "fsaverage"

    print(f'Computing neural RDMs for all subjects for the "{which_rois}" ROIs')

    if 'special1000' in which_data:
        print('\tUsing special 1000 conditions')
        subconds = np.asarray(get_conditions_1000(nsd_dir)).ravel()
    elif 'special515' in which_data:
        print('\tUsing special 515 conditions')
        subconds = np.asarray(get_conditions_515(nsd_dir)).ravel()
    elif 'special100' in which_data:
        print('\tUsing special 100 conditions')
        subconds = np.asarray(get_conditions_100(nsd_dir)).ravel()
    elif 'fullnsd' in which_data:
        print('\tUsing all conditions')
    else:
        raise ValueError('which_data should be either all, special1000, special515 or special100')

    for s_n, subj in enumerate(subs):
        
        conditions, conditions_sampled, sample = get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True)

        # Betas per subject
        betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
        betas = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)
        if which_data != 'fullnsd':
            subinds = [x for x, j in enumerate(sample) if j in subconds]
            betas = betas[:, subinds]

        all_rdms[subj] = get_subj_all_roi_rdms(subj, betas, ROIS, maskdata, rdm_distance)

        del betas

    with open(f'{save_dir}/{which_rois}_all_neural_rdms_{rdm_distance}_{which_data}.pkl', 'wb') as pickle_file:
        pickle.dump(all_rdms, pickle_file)

    return all_rdms
