import glob
import os
import numpy as np
from nsdcode.nsd_mapdata import NSDmapdata
from tqdm import tqdm


def nsd_project_fsaverage(MODEL_NAMES, models_rdm_distance, nsd_dir, base_save_dir, remove_shared_515):
    
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

    MODEL_SUFFIX = "_noShared515" if remove_shared_515 else ""

    for MODEL_NAME in MODEL_NAMES:
        # define where the searchlights are saved
        data_dir = os.path.join(
            base_save_dir,
            f"searchlight_respectedsampling_{models_rdm_distance}_newTest",
            "{}",
            MODEL_NAME,
            f"corr_vols{MODEL_SUFFIX}_{models_rdm_distance}",
        )  # '{}' will be subject number.

        # define where the fsaverage maps will be saved
        data_dir_fsav = os.path.join(
            base_save_dir,
            f"searchlight_respectedsampling_{models_rdm_distance}_newTest",
            "{}",
            MODEL_NAME,
            f"{MODEL_NAME}{MODEL_SUFFIX}_{models_rdm_distance}_fsaverage",
        )

        for subjix in range(n_subjects):
            # specify subject full name
            this_sub = f"subj0{subjix+1}"
            output_dir = data_dir_fsav.format(this_sub)
            os.makedirs(output_dir, exist_ok=True)

            # define the subject directory where the searchlights
            # for this model live.
            subj_dir = data_dir.format(this_sub)

            n_voxels = voxelsizes[subjix]

            # get the sample files and sort them in ascending order (volumes for each of 100 times 100x100 upper tri rdms sampled in sl_main)
            sample_files = glob.glob(os.path.join(subj_dir, "*sample*.npy"))
            sample_files.sort()  # need alphabetical order

            n_samples = len(sample_files)

            this_sample = sample_files[0]
            print(f"\treading in {this_sample}\n")

            brain_vols = np.load(this_sample, allow_pickle=True)

            n_models_xyz = brain_vols.shape  # checks how many models were run
            n_models = n_models_xyz[0]

            for model_i in range(n_models):
                brain_vols = []
                print(f"reading model fit samples for {MODEL_NAME}")
                for sample in tqdm(sample_files, desc="samples", ascii=True):
                    brain_vol = np.load(sample, allow_pickle=True)
                    brain_vol = np.reshape(brain_vol[model_i], n_voxels)
                    brain_vols.append(brain_vol)

                # stack back into array
                brain_vols = np.stack(brain_vols)  # 100xbrain_vol_dims

                # average
                model_means = np.nanmean(brain_vols, axis=0)  # shape (brain_vol_dims,)

                # also compute T
                t_brain = np.nanmean(brain_vols, axis=0)/(np.nanstd(brain_vols, axis=0)/np.sqrt(n_samples))

                print("projecting to fsaverage\n")
                model = model_means
                model_t = t_brain

                print(f"\tprojecting model {model_i} out of {n_models} models")
                output_file = os.path.join(output_dir, "{}"+f".{this_sub}-model-{model_i+1}-surf.npy")
                if not os.path.exists(output_file.format("lh")) or not os.path.exists(output_file.format("rh")):
                    data = []
                    # project the data to three
                    # cortical depths separately
                    # for each hemisphere.
                    for hemi in hemis:
                        hemi_data = []
                        for lay in range(3):  # part of NSD pipeline. Take average across 3 cortical depths.
                            hemi_data.append(
                                nsd.fit(  # goes from 'func1pt8' source space and projects to f'{hemi}.layerB{lay+1}' (subj native freesurfer space)
                                    subjix + 1,
                                    "func1pt8",
                                    f"{hemi}.layerB{lay+1}",
                                    model,
                                    "cubic",
                                    badval=0,
                                )
                            )
                        data.append(np.nanmean(np.stack(hemi_data), axis=0))

                    # port the model
                    for h, d in zip(hemis, data):
                        print(f"\t\tsaving {output_file} to disk")
                        transformed_data = nsd.fit(  # projects to fsaverage
                                                    subjix + 1,
                                                    f"{h}.white",
                                                    "fsaverage",
                                                    d,
                                                    interptype=None,
                                                    badval=0,
                                                    fsdir=fs_dir,
                                                )
                        np.save(output_file.format(h), transformed_data, allow_pickle=True)
                else:
                    print(f"\t\t{output_file.format('lh')} already exists, skipping")
                
                # exactly the same idea as above but for tvals
                print(f"\tprojecting t-values for model {model_i} out of {n_models} models")
                output_file = os.path.join(output_dir, "{}"+f".{this_sub}-model-{model_i+1}-surf-tvals.npy")
                if not os.path.exists(output_file.format("lh")) or not os.path.exists(output_file.format("rh")):
                    data = []
                    # project the data to three
                    # cortical depths separately
                    # for each hemisphere.
                    for hemi in hemis:
                        hemi_data = []
                        for lay in range(3):
                            hemi_data.append(
                                nsd.fit(
                                    subjix + 1,
                                    "func1pt8",
                                    f"{hemi}.layerB{lay+1}",
                                    model_t,
                                    "cubic",
                                    badval=0,
                                )
                            )
                        data.append(np.nanmean(np.stack(hemi_data), axis=0))

                    # port the model
                    for h, d in zip(hemis, data):
                        print(f"\t\tsaving {output_file} to disk")
                        transformed_data = nsd.fit(
                                                subjix + 1,
                                                f"{h}.white",
                                                "fsaverage",
                                                d,
                                                interptype=None,
                                                badval=0,
                                                fsdir=fs_dir,
                                                )
                        np.save(output_file.format(h), transformed_data, allow_pickle=True)
                else:
                    print(f"\t\t{output_file.format('lh')} already exists, skipping")
