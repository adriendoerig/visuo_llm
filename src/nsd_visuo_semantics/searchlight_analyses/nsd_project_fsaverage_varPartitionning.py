import glob, os, pickle
import numpy as np
from nsdcode.nsd_mapdata import NSDmapdata
from tqdm import tqdm
from nsd_visuo_semantics.roi_analyses.variance_partitionning import combination_scores_to_unique_var


def nsd_project_fsaverage_varPartitionning(MODEL_NAMES, models_rdm_distance, nsd_dir, base_save_dir, remove_shared_515):
    
    # initiate NSDmapdata
    nsd = NSDmapdata(nsd_dir)  # Takes subject data in mni and project to freesurfer, etcetc. All the transformations that we can do to the data can be done with this

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

    # define where the searchlights are saved
    data_dir = os.path.join(
        base_save_dir,
        f"searchlight_respectedsampling_{models_rdm_distance}_newTest",
        "{}",
        "var_partition_['mpnet', 'mpnet_nouns', 'mpnet_verbs']"
    )  # '{}' will be subject number.

    # define where the fsaverage maps will be saved
    data_dir_fsav = os.path.join(
        base_save_dir,
        f"searchlight_respectedsampling_{models_rdm_distance}_newTest",
        "{}",
        "var_partition_['mpnet', 'mpnet_nouns', 'mpnet_verbs']",
        f"fsaverage",
    )

    for subjix in range(n_subjects):
        # specify subject full name
        this_sub = f"subj0{subjix+1}"
        output_dir_fsav = data_dir_fsav.format(this_sub)
        os.makedirs(output_dir_fsav, exist_ok=True)
        output_dir = data_dir.format(this_sub)
        os.makedirs(output_dir, exist_ok=True)

        # define the subject directory where the searchlights
        # for this model live.
        subj_dir = data_dir.format(this_sub)

        n_voxels = voxelsizes[subjix]

        # get the sample files and sort them in ascending order (volumes for each of 100 times 100x100 upper tri rdms sampled in sl_main)
        sample_files = glob.glob(os.path.join(glob.escape(subj_dir), "*sample*.npy"))  # escape needed because we have '[' in the path
        sample_files.sort()  # need alphabetical order
        
        # load pickle file with model fit samples
        with open(os.path.join(subj_dir, "model_combinations.pkl"), "rb") as f:
            model_combinations = pickle.load(f)

        brain_vol_scores = []
        brain_vol_unique_vars = []
        print(f"reading model fit samples")
        for sample in tqdm(sample_files, desc="samples", ascii=True):
            brain_vol_score = np.load(sample, allow_pickle=True)
            brain_vol_scores.append(brain_vol_score)
            brain_vol_unique_var = np.empty(brain_vol_score.shape)
            for i in range(brain_vol_unique_var.shape[0]):
                for j in range(brain_vol_unique_var.shape[1]):
                    for k in range(brain_vol_unique_var.shape[2]):
                        brain_vol_unique_var[i,j,k] = combination_scores_to_unique_var(brain_vol_score[i,j,k])
            brain_vol_unique_vars.append(brain_vol_unique_var)

        # stack back into array
        brain_vol_scores = np.stack(brain_vol_scores)
        brain_vol_unique_vars = np.stack(brain_vol_unique_vars)
        if brain_vol_scores.shape[0] == 1:
            brain_vol_scores = np.squeeze(brain_vol_scores)  # 100xbrain_vol_dims
            brain_vol_unique_vars = np.squeeze(brain_vol_unique_vars)  # 100xbrain_vol_dims
        else:
            # average
            brain_vol_scores = np.nanmean(brain_vol_scores, axis=0)  # shape (brain_vol_dims,)
            brain_vol_unique_vars = np.nanmean(brain_vol_unique_vars, axis=0)  # shape (brain_vol_dims,)

        np.save(os.path.join(output_dir, "brain_vol_scores.npy"), brain_vol_scores, allow_pickle=True)
        np.save(os.path.join(output_dir, "brain_vol_unique_vars.npy"), brain_vol_unique_vars, allow_pickle=True)

        # print("projecting to fsaverage\n")
        # for i, mc in enumerate(model_combinations):
        #     print(f"\tprojecting map for {mc}")
        #     data = []
        #     # project the data to three
        #     # cortical depths separately
        #     # for each hemisphere.
        #     for hemi in hemis:
        #         hemi_data = []
        #         for lay in range(3):  # part of NSD pipeline. Take average across 3 cortical depths.
        #             hemi_data.append(
        #                 nsd.fit(  # goes from 'func1pt8' source space and projects to f'{hemi}.layerB{lay+1}' (subj native freesurfer space)
        #                     subjix + 1,
        #                     "func1pt8",
        #                     f"{hemi}.layerB{lay+1}",
        #                     brain_vol_scores[:,:,:,i],
        #                     "cubic",
        #                     badval=0,
        #                 )
        #             )
        #         data.append(np.nanmean(np.stack(hemi_data), axis=0))

        #     # port the maps
        #     for h, d in zip(hemis, data):
        #         output_file = os.path.join(
        #             output_dir_fsav, f"{h}.{this_sub}-scores-{mc}-surf.npy"
        #         )

        #         print(f"\t\tsaving {output_file} to disk")
        #         transformed_data = nsd.fit(  # projects to fsaverage
        #                                     subjix + 1,
        #                                     f"{h}.white",
        #                                     "fsaverage",
        #                                     d,
        #                                     interptype=None,
        #                                     badval=0,
        #                                     fsdir=fs_dir,
        #                                 )
        #         np.save(output_file, transformed_data, allow_pickle=True)

        #     data = []
        #     # project the data to three
        #     # cortical depths separately
        #     # for each hemisphere.
        #     for hemi in hemis:
        #         hemi_data = []
        #         for lay in range(3):  # part of NSD pipeline. Take average across 3 cortical depths.
        #             hemi_data.append(
        #                 nsd.fit(  # goes from 'func1pt8' source space and projects to f'{hemi}.layerB{lay+1}' (subj native freesurfer space)
        #                     subjix + 1,
        #                     "func1pt8",
        #                     f"{hemi}.layerB{lay+1}",
        #                     brain_vol_unique_vars[:,:,:,i],
        #                     "cubic",
        #                     badval=0,
        #                 )
        #             )
        #         data.append(np.nanmean(np.stack(hemi_data), axis=0))

        #     # port the maps
        #     for h, d in zip(hemis, data):
        #         output_file = os.path.join(
        #             output_dir_fsav, f"{h}.{this_sub}-uniquevars-{mc}-surf.npy"
        #         )

        #         print(f"\t\tsaving {output_file} to disk")
        #         transformed_data = nsd.fit(  # projects to fsaverage
        #                                     subjix + 1,
        #                                     f"{h}.white",
        #                                     "fsaverage",
        #                                     d,
        #                                     interptype=None,
        #                                     badval=0,
        #                                     fsdir=fs_dir,
        #                                 )
        #         np.save(output_file, transformed_data, allow_pickle=True)
