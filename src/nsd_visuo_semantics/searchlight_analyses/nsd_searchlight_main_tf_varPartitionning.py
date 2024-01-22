import os, time, pickle
import numpy as np
from nsd_visuo_semantics.utils.tf_utils import chunking, corr_rdms, sort_spheres
from nsd_visuo_semantics.searchlight_analyses.tf_searchlight import tf_searchlight as tfs
from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_conditions_100, get_masks, get_model_rdms, load_or_compute_betas_average
from nsd_visuo_semantics.utils.utils import reorder_rdm
from nsd_visuo_semantics.roi_analyses.variance_partitionning import variance_partitioning


def nsd_searchlight_main_tf_varPartitionning(MODEL_NAMES, rdm_distance, 
                            nsd_dir, precompsl_dir, betas_dir, base_save_dir, 
                            use_special_100,
                            remove_shared_515, OVERWRITE):

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

    print(f"Starting main searchlight computations for {MODEL_NAMES}")
    models_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}'
    print(f"Loading serialised model rdms from {models_dir}")

    # loop over subjects
    for subject in range(8):
        # define subject
        sub = subject + 1
        # format subject
        subj = f"subj0{sub}"

        # called like this because all models sample the same 100 images every time for fair comparison
        results_dir = f"{base_save_dir}/searchlight_respectedsampling_{rdm_distance}_newTest/{subj}"
        os.makedirs(results_dir, exist_ok=True)

        # where to save/load sample ids: all models sample the same 100 images every time for fair comparison.
        # we compute them only once for guse, and then will reload them for others
        samples_dir = f'{results_dir}/saved_sampling{"_noShared515" if remove_shared_515 else ""}'
        os.makedirs(samples_dir, exist_ok=True)

        # where to save searchlight correlations
        searchlight_variance_partitionning_dir = f'{results_dir}/var_partition_{MODEL_NAMES}_{"_noShared515" if remove_shared_515 else ""}'
        os.makedirs(searchlight_variance_partitionning_dir, exist_ok=True)

        print(f"\tthe output files will be stored in {searchlight_variance_partitionning_dir}..")
        print(f"\tlooking for saved samples in {samples_dir}..")

        # get model rdms for this subject
        all_model_rdms, all_model_names = [], []
        for MODEL_NAME in MODEL_NAMES:
            model_rdms, model_names = get_model_rdms(f"{models_dir}/{MODEL_NAME}", subj, filt=MODEL_NAME)
            all_model_rdms.append(model_rdms[0])
            all_model_names.append(model_names[0].replace('_nsd_special100_cocoCaptions', '').replace('all-mpnet-base-v2', 'mpnet').replace('cutoffDist0.7_', ''))

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
        # So subj_sample contains the unique NSD_ids for that subject, in increasing order (e.g. [ 14,  28,  72, ...]).
        # Importantly, the average betas loaded above are arranged in the same way, so that if we want to find the betas
        # for NSD_id=72, we just need to find the idx of 72 in subj_sample (in the present example: 2). Using this method, we can
        # find the avg_betas corresponding to the shared 515 images as done below with subj_indices_515 (hint: the trick to
        # go from an ordered list of nsd_ids to finding the idx as described above is to use enumerate).
        # For example sample[subj_indices_515[0]] = conditions_515[0].
        conditions = np.asarray(get_conditions(nsd_dir, subj, n_sessions)).ravel()
        # then we find the valid trials for which we do have 3 repetitions.
        conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
        if remove_shared_515:
            conditions_3repeats = np.unique(conditions[conditions_bool])  # save for later n_subj images WITH 515 -> THIS CAN INDEX THE BETAS CORRECTLY
            conditions_515 = get_conditions_515(nsd_dir)  # [515,]  (nsd_indices for the 515 shared images)
            conditions_515_bool = [True if x in conditions_515 else False for x in conditions]  # [n_subj_stims,] boolean array with True if this idx is a 515 shared img
            conditions_bool = [True if x and not y else False for x, y in zip(conditions_bool, conditions_515_bool)]  # [n_subj_stims-515,] array of nsd_indices
        # apply the condition boolean (which conditions had 3 repeats)
        conditions_sampled = conditions[conditions_bool]
        # find the subject's condition list (sample pool)
        subj_sample = np.unique(conditions_sampled)  # ordered nsd_indices for single conditions seen 3x, optionally removing the shared515  (10000 if NSD subject completed all conds 3 times and we are not removing the 515)
        subj_n_images = len(subj_sample)
        all_conditions = range(subj_n_images)
        subj_n_samples = 100 if use_special_100 else int(subj_n_images // 100)

        # Betas per subject
        print(f"loading betas for {subj}")
        betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
        betas = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)
        
        if remove_shared_515:
            # When removing the shared 515, we need to change the indices of the betas in the same way as we changed
            # the indices of the rdms, eetc, so as to keep everything consistent
            subj_sample_no515_bool = [False if x in conditions_515 else True for x in conditions_3repeats]
            betas = betas[:, :, :, subj_sample_no515_bool]  # [voxx, voxy, voxz, n_subj_conditions-515]

        # now we start the sampling procedure
        if use_special_100:
            # in this case, we already have the special100 rdm, so no need to index 
            # to get the right entries from the big 10000 stim RDM. All we need is the 
            # special100_sample_indices, to index the special 100 in the betas file.
            subj_n_samples = 1
            conditions_100 = get_conditions_100(nsd_dir)
            conditions_100_bool = [True if x in conditions_100 else False for x in subj_sample]
            special100_sample_indices = np.where(conditions_100_bool)[0]
            subj_sample_pool = [special100_sample_indices]
            model_rdms_sample = all_model_rdms
        else:
            # here, we need to sample 100 times 100 stimuli from the subject's 10000 stimuli.
            # we will index into the big rdm in the main loop.

             # initialise batch generator. Retrieves 100x100 sampled RDM from upper tri of 10000x10000 full RDM
            batchg = {model_name: BatchGen(model_rdm, all_conditions) for model_name, model_rdm in zip(all_model_names, all_model_rdms)}

            # compute sampling if we are computing mpnet and it does not exist yet, else load
            saved_samples_file = os.path.join(samples_dir, f"{subj}_nsd-allsubstim_sampling.npy")
            if not os.path.exists(saved_samples_file):
                if MODEL_NAME == "mpnet":
                    print("Running MPNET and DID NOT FIND existing saved_samples_file. Computing from scratch.")
                    subj_sample_pool = []
                    subj_shuffle_pool = []
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
                searchlight_variance_partitionning_dir,
                f'{subj}_nsd-{MODEL_NAME}_{targetspace}{"_noShared515" if remove_shared_515 else ""}{"_special100" if use_special_100 else ""}_sample-{j}.npy',
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

                # tfs is tensorflow searchlight, an efficient GPU-powered way to compute the 100x100 brain rdms
                # returns n_voxelsx(upper_tri_sampled_brain_rdm)
                brain_sl_rdms_sample = tfs(betas_sampled, all_indices, sorted_indices, batch_size)
                # computes correlation between ALL searchlight brain rdms and the model rdm for this sampled 100x100 rdm

                # now get the model rdms (not needed if we use the special 100)
                # this returns N_modelsx(upper_tri_sampled_model_rdm)
                if not use_special_100:
                    model_rdms_sample = [np.asarray(batchg[m].index_rdms(choices) for m in all_model_names)]

                variance_components = np.zeros((brain_sl_rdms_sample.shape[0], 7))  # 7 because we get 7 r-squared values per voxel
                for voxel, brain_rdm in enumerate(brain_sl_rdms_sample):
                    variance_components[voxel,:], combination_names = variance_partitioning(brain_rdm, model_rdms_sample, all_model_names, zscore=True, return_np=True)

                # save correlation vol for that sample
                np.save(file_save, variance_components)
                np.save(file_save[:-4]+'model_combination_names.npy', combination_names)

                elapsed_time = time.time() - start_time
                print(f"boot {j} : elapsedtime : ", f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

                import pdb; pdb.set_trace()

        del betas
        
        print("NSD searchlight variance partitionning done.")
        elapsed_time = time.time() - initial_time
        print("elapsedtime: ", f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
