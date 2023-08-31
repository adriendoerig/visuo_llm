import os
import time

import nibabel as nib
import numpy as np
from nsdcode.nsd_mapdata import NSDmapdata
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

MODEL_SUFFIX = "_noShared515"  # Default: '', or '_noShared515' or '_fracridgeFit' - append to the saved folder name, useful for distinguishing different runs.
sig_alpha = 0.005

# various paths
base_dir = os.path.join("/rds", "projects", "c", "charesti-start")
nsd_path = os.path.join(base_dir, "data", "NSD")
base_save_dir = "./save_dir"
roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses", "roi_defs")

# initiate NSDmapdata
nsd = NSDmapdata(
    nsd_path
)  # ian and kendrick made this. Takes subject data in mni and project to freesurfer, etcetc. All the transformations that we can do to the data can be done with this

# NSD fsaverage stuff
fs_dir = os.path.join(nsd.base_dir, "nsddata", "freesurfer", "fsaverage")

# fixed parameters
n_subjects = 8

# per subject vox sizes
voxelsizes = [
    [81, 104, 83],
    [82, 106, 84],
    [81, 106, 82],
    [85, 99, 80],
    [79, 97, 78],
    [85, 113, 83],
    [78, 95, 81],
    [80, 103, 78],
]

# we do it for the two henispheres (always left right order)
hemis = ["lh", "rh"]

initial_time = time.time()
# ['dnn_multihot_ff', 'dnn_multihot_rec', 'dnn_guse_ff', 'dnn_guse_rec', 'dnn_mpnet_ff', 'dnn_mpnet_rec',
# 'guse', 'multihot', 'mpnet', 'fasttext_categories', 'fasttext_all', 'fasttext_verbs', 'openai_ada2',
# # 'dnn_ecoset_category', 'dnn_ecoset_fasttext', 'guse_SCRAMBLED_WORD_ORDER', 'mpnet_SCRAMBLED_WORD_ORDER']:
for MODEL_NAME in ["mpnet"]:
    # define where the fsaverage maps will be saved
    data_dir_fsav = os.path.join(
        base_save_dir,
        "respectedsampling",
        "{}",
        MODEL_NAME,
        f"{MODEL_NAME}{MODEL_SUFFIX}_fsaverage",
    )

    for MODEL_ID in [
        1
    ]:  # if a model has eg multiple layers, this allows you to pick one. If not, use 1.
        all_sub_data = {}
        for subjix in range(n_subjects):
            # specify subject full name
            this_sub = f"subj0{subjix+1}"
            load_dir = data_dir_fsav.format(this_sub)

            # load data. we will get one [n_subj, n_fsaverage_voxels] array per hemispheres (for mean and tval of corrs)
            print(f"Loading fsaverage maps from {load_dir}")
            for h in hemis:
                # mean and tval for corr values accross the 100 100x100 samples
                loaded_means = (
                    nib.load(
                        f"{load_dir}/{h}.{this_sub}-model-{MODEL_ID}-surf.mgz"
                    )
                    .get_fdata()
                    .squeeze()
                )
                loaded_tvals = (
                    nib.load(
                        f"{load_dir}/{h}.{this_sub}-model-{MODEL_ID}-surf-tvals.mgz"
                    )
                    .get_fdata()
                    .squeeze()
                )
                if f"{h}_means" not in all_sub_data.keys():
                    all_sub_data[f"{h}_means"] = loaded_means
                    all_sub_data[f"{h}_tvals"] = loaded_tvals
                else:
                    all_sub_data[f"{h}_means"] = np.vstack(
                        [all_sub_data[f"{h}_means"], loaded_means]
                    )
                    all_sub_data[f"{h}_tvals"] = np.vstack(
                        [all_sub_data[f"{h}_tvals"], loaded_tvals]
                    )

        all_sub_data["both_hemis_means"] = np.hstack(
            [all_sub_data[f"{h}_means"] for h in hemis]
        )

        # NOT FINISHED
        t_stat, p_vals = ttest_1samp(
            all_sub_data["both_hemis_means"], 0, axis=0
        )
        reject_hyp_bool, p_vals_adj = fdrcorrection(p_vals, alpha=sig_alpha)
        n_lh_voxels = all_sub_data["lh_means"].shape[1]
        # Masks are hemisphere-wise, with 1 where sig, 0 otherwise
        lh_mask, rh_mask = reject_hyp_bool[:n_lh_voxels].astype(
            np.uint8
        ), reject_hyp_bool[n_lh_voxels:].astype(np.uint8)
        print(
            f"Found {lh_mask.sum()} sig voxels at corr_alpha={sig_alpha} in the left hemisphere, "
            f"and {rh_mask.sum()} in the right. "
            f"In total, {reject_hyp_bool.sum()/len(reject_hyp_bool)*100}% of voxels are significant"
        )
        print(f"Saving to {roi_analyses_dir}")
        np.save(
            os.path.join(
                roi_analyses_dir,
                f"lh.{MODEL_NAME}{MODEL_SUFFIX}_sig{sig_alpha}_fsaverage",
            ),
            lh_mask,
        )
        np.save(
            os.path.join(
                roi_analyses_dir,
                f"rh.{MODEL_NAME}{MODEL_SUFFIX}_sig{sig_alpha}_fsaverage",
            ),
            rh_mask,
        )
