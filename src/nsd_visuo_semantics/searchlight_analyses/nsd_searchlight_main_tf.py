'''Main script to compute searchlight correlations between model RDMs and brain RDMs.
the output of this is 100 correlation volumes, each based on a randomly sampled 100x100 RDM.
This is done for computational efficiency (correlating this amount of 10'000x10'000 RDMs 
takes too long).'''

import os, time, pickle
import numpy as np
from nsd_visuo_semantics.utils.tf_utils import chunking, corr_rdms, sort_spheres
from nsd_visuo_semantics.searchlight_analyses.tf_searchlight import tf_searchlight as tfs
from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_subject_conditions, get_masks, get_model_rdms, load_or_compute_betas_average
from nsd_visuo_semantics.utils.utils import reorder_rdm

def nsd_searchlight_main_tf(MODEL_NAMES, rdm_distance, 
                            nsd_dir, precompsl_dir, betas_dir, base_save_dir, 
                            OVERWRITE):

    initial_time = time.time()

    # general variables
    batch_size = 250

    # fixed parameters
    radius = 6
    n_sessions = 40
    targetspace = "func1pt8mm"

    # set up directories
    os.makedirs(betas_dir, exist_ok=True)
    os.makedirs(precompsl_dir, exist_ok=True)

    for MODEL_NAME in MODEL_NAMES:

        print(f"Starting main searchlight computations for {MODEL_NAME}")
        models_dir = f'{base_save_dir}/serialised_models_{rdm_distance}/{MODEL_NAME}'
        print(f"Loading serialised model rdms from {models_dir}")

        # loop over subjects
        for subject in range(8):
            # define subject
            sub = subject + 1
            # format subject
            subj = f"subj0{sub}"

            # called like this because all models sample the same 100 images every time for fair comparison
            results_dir = f"{base_save_dir}/searchlight_respectedsampling_{rdm_distance}/{subj}"
            os.makedirs(results_dir, exist_ok=True)

            # where to save/load sample ids: all models sample the same 100 images every time for fair comparison.
            # we compute them only once for guse, and then will reload them for others
            samples_dir = f'{results_dir}/saved_sampling'
            os.makedirs(samples_dir, exist_ok=True)

            # where to save searchlight correlations
            searchlight_correlations_dir = f'{results_dir}/{MODEL_NAME}/corr_vols_{rdm_distance}'
            os.makedirs(searchlight_correlations_dir, exist_ok=True)

            print(f"\tthe output files will be stored in {searchlight_correlations_dir}..")
            print(f"\tlooking for saved samples in {samples_dir}..")

            # get model rdms for this subject
            model_rdms, model_names = get_model_rdms(models_dir, subj, filt=MODEL_NAME)  # (filt should be a wildcard to catch correct model rdms, careful not to catch other models)
            n_models = len(model_rdms)  # sometimes, we have many models (e.g. 1 per layer per timestep)

            # get subject brain mask (only used if searchlight indices are not computed yet).
            # We always want the same indices, radius, etcetc. across models
            print("\tloading brain mask")
            mask = get_masks(nsd_dir, subj, targetspace)
            n_voxels = list(mask.shape)

            subj_precompsl_dir = os.path.join(precompsl_dir, subj)
            os.makedirs(subj_precompsl_dir, exist_ok=True)
            sl_indices = f"{subj_precompsl_dir}/{subj}-{targetspace}-{radius}rad-searchlight_indices.npy"
            sl_centers = f"{subj_precompsl_dir}/{subj}-{targetspace}-{radius}rad-searchlight_centers.npy"

            if not os.path.exists(sl_indices):
                print("\tinitialising searchlight")
                # initiate searchlight indices for spheres restrained to valid brain masks
                from nsd_visuo_semantics.searchlight_analyses.searchlight import RSASearchLight
                SL = RSASearchLight(mask, radius=radius, thr=.5, verbose=True)
                # save allIndices
                all_indices = SL.allIndices
                center_indices = SL.centerIndices
                # save centers and indices in pickle files
                with open(sl_indices, "wb") as fp:
                    pickle.dump(all_indices, fp)
                with open(sl_centers, "wb") as fp:
                    pickle.dump(center_indices, fp)
            else:
                print("\tloading pre-computed searchlight")
                with open(sl_indices, "rb") as fp:
                    all_indices = pickle.load(fp)
                with open(sl_centers, "rb") as fp:
                    center_indices = pickle.load(fp)
            
            # sort sphere by n_features. We will make batches where all spheres have the same n_voxels (required to use tf). 
            sorted_indices = sort_spheres(all_indices)

            # pre-compute the final sorting order
            rdms_sort = []
            for i, ind in enumerate(sorted_indices):
                chunks = chunking(ind, batch_size)
                for c, chunk in enumerate(chunks):
                    rdms_sort.append(center_indices[chunk.astype(np.int32)])  # this is where the sorting mentioned above happens
            rdms_sort = np.hstack(rdms_sort).astype(int)

            # extract conditions data and reshape conditions to be ntrials x 1
            conditions, conditions_sampled, subj_sample = get_subject_conditions(nsd_dir, subj, n_sessions, keep_only_3repeats=True)
            subj_n_images = len(subj_sample)
            all_conditions = range(subj_n_images)
            subj_n_samples = int(subj_n_images // 100)

            # Betas per subject
            print(f"loading betas for {subj}")
            betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
            betas = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)
            
            # initialise batch generator. Retrieves 100x100 sampled RDM from upper tri of 10000x10000 full RDM
            batchg = BatchGen(model_rdms, all_conditions)

            # now we start the sampling procedure
            saved_samples_file = os.path.join(samples_dir, f"{subj}_nsd-allsubstim_sampling.npy")

            # compute sampling if we are computing mpnet and it does not exist yet, else load
            if not os.path.exists(saved_samples_file):
                if MODEL_NAME == "mpnet":
                    print("Running MPNET and DID NOT FIND existing saved_samples_file. Computing from scratch.")
                    subj_sample_pool = []
                    for j in range(subj_n_samples):
                        choices = np.random.choice(all_conditions, 100, replace=False)
                        choices.sort()
                        subj_sample_pool.append(choices)
                        all_conditions = np.setdiff1d(all_conditions, choices)
                    np.save(saved_samples_file, subj_sample_pool)
                else:
                    raise FileNotFoundError(
                        "Saved samples not found for MPNET. Raising an error for security."
                        f"\n Looked in {saved_samples_file}"
                        "What happens here is that we try to load the 100x100 samples used for the original"
                        "MPNET sampling procedure, and reapply them for subsequent models, for fair"
                        "comparisons."
                        "\nIf the samples should already be computed, please check saved_samples_file"
                    )
            else:
                print(f"Loading 100x100 sample choices from {saved_samples_file}")
                subj_sample_pool = np.load(saved_samples_file, allow_pickle=True)

            # run the searchlight mappings
            for j in range(subj_n_samples):
                file_save = os.path.join(
                    searchlight_correlations_dir,
                    f'{subj}_nsd-{MODEL_NAME}_{targetspace}_sample-{j}.npy',
                )

                if os.path.exists(file_save) and not OVERWRITE:
                    print(f"\n\n\n\tFound existing file at {file_save}, skipping...")

                else:
                    print(f"\n\n\n\tworking on {subj} - usual case: boot {j}\n")
                    start_time = time.time()

                    # sample 100 stimuli from the subject's sample.
                    choices = subj_sample_pool[j]

                    # simple case without fitting RDMs from all model layers. We simply take our 100 samples and correlate
                    # their brain RDM with the model RDMs of each layer
                    betas_sampled = betas[:, :, :, choices]
                    betas_sampled = betas_sampled.astype(np.float32)

                    # now get the models and correlate
                    # this returns N_modelsx(upper_tri_sampled_model_rdm)
                    model_rdms_sample = np.asarray(batchg.index_rdms(choices))

                    # tfs is tensorflow searchlight, an efficient GPU-powered way to compute the 100x100 brain rdms
                    # returns n_voxelsx(upper_tri_sampled_brain_rdm)
                    brain_sl_rdms_sample = tfs(betas_sampled, all_indices, sorted_indices, batch_size)
                    # computes correlation between ALL searchlight brain rdms and the model rdm for this sampled 100x100 rdm
                    brain_maps = corr_rdms(brain_sl_rdms_sample, model_rdms_sample)

                    # reshape into original volume
                    brain_vols = []
                    for map_i in range(n_models):
                        brain_map = brain_maps[:, map_i]
                        brain_vect = np.zeros(np.prod(n_voxels))
                        brain_vect[rdms_sort] = brain_map.squeeze()  # insert corr map in the right brain locations (each corr ends up in the right voxel)
                        brain_vols.append(np.reshape(brain_vect, n_voxels))  # reshape to original xyz fmrivolume
                    brain_vols = np.asarray(brain_vols)  # vols is plural because there may be more than 1 model. when using 1 model there is just one vol

                    # save correlation vol for that sample
                    np.save(file_save, brain_vols)

            del betas
            
            print("NSD searchlight mapping done.")
            elapsed_time = time.time() - initial_time
            print("elapsedtime: ", f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
