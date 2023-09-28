import os, time
import numpy as np
from nsd_visuo_semantics.utils.tf_utils import chunking, corr_rdms, sort_spheres
from nsd_visuo_semantics.searchlight_analyses.tf_searchlight import tf_searchlight as tfs
from nsd_visuo_semantics.utils.batch_gen import BatchGen
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_masks, get_model_rdms, load_or_compute_betas_average
from nsd_visuo_semantics.utils.utils import reorder_rdm

initial_time = time.time()

# general variables
batch_size = 250

# fixed parameters
radius = 6
n_boot = 100
n_sessions = 40
targetspace = "func1pt8mm"

# if true, the 515 stimuli seen by all subjects are removed (so they can be used in the test set of other experiments
# based on searchlight maps while avoiding double-dipping)
remove_shared_515 = False

# RDM distance measure NOTE: BRAIN RDMS ARE DONE WITH PEARSON CORR (needs to be the same as in nsd_prepare_rdms.py)
rdm_distance = "correlation"

# set up directories
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = f"{nsd_derivatives_dir}/betas"
precompsl_dir = f"{nsd_derivatives_dir}/searchlights"
base_save_dir = "../results_dir"
os.makedirs(nsd_derivatives_dir, exist_ok=True)
os.makedirs(betas_dir, exist_ok=True)
os.makedirs(precompsl_dir, exist_ok=True)

for MODEL_NAME in ["mpnet", "multihot",  "fasttext_nouns", "nsd_fasttext_nouns_closest_cocoCats_cut0.33",
                   "dnn_multihot_rec", "dnn_mpnet_rec"]:

    print(f"Starting main searchlight computations for {MODEL_NAME}")
    models_dir = f'{base_save_dir}/serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}/{MODEL_NAME}'
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
        samples_dir = f'{results_dir}/saved_sampling{"_noShared515" if remove_shared_515 else ""}'
        os.makedirs(samples_dir, exist_ok=True)

        # where to save searchlight correlations
        searchlight_correlations_dir = f'{results_dir}/{MODEL_NAME}/corr_vols{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}'
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
            np.save(sl_indices, all_indices)
            np.save(sl_centers, center_indices)
        else:
            print("\tloading pre-computed searchlight")
            all_indices = np.load(sl_indices, allow_pickle=True)
            center_indices = np.load(sl_centers, allow_pickle=True)
        
        # ort sphere by n_features. We will make batches where all spheres have the same n_voxels (required to use tf). 
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
        if remove_shared_515:
            conditions_515 = get_conditions_515(nsd_dir)  # [515,]  (nsd_indices for the 515 shared images)
            conditions_515_bool = [True if x in conditions_515 else False for x in conditions]  # [n_subj_stims,] boolean array with True if this idx is a 515 shared img
            conditions_bool = [True if x and not y else False for x, y in zip(conditions_bool, conditions_515_bool)]  # [n_subj_stims-515,] array of nsd_indices
        # apply the condition boolean (which conditions had 3 repeats)
        conditions_sampled = conditions[conditions_bool]
        # find the subject's condition list (sample pool)
        subj_sample = np.unique(conditions_sampled)  # ordered nsd_indices for single conditions seen 3x, optionally removing the shared515  (10000 if NSD subject completed all conds 3 times and we are not removing the 515)
        subj_n_images = len(subj_sample)
        all_conditions = range(subj_n_images)
        subj_n_samples = int(subj_n_images // 100)

        # Betas per subject
        print(f"loading betas for {subj}")
        betas_file = os.path.join(betas_dir, f"{subj}_betas_average_{targetspace}.npy")
        betas = load_or_compute_betas_average(betas_file, nsd_dir, subj, n_sessions, conditions, conditions_sampled, targetspace)
        
        if remove_shared_515:
            # When removing the shared 515, we need to change the indices of the betas in the same way as we changed
            # the indices of the rdms, eetc, so as to keep everything consistent
            subj_sample_no515_bool = [False if x in conditions_515 else True for x in conditions_3repeats]
            betas = betas[:, :, :, subj_sample_no515_bool]  # [voxx, voxy, voxz, n_subj_conditions-515]

        # initialise batch generator. Retrieves 100x100 sampled RDM from upper tri of 10000x10000 full RDM
        batchg = BatchGen(model_rdms, all_conditions)

        # now we start the sampling procedure
        saved_samples_file = os.path.join(samples_dir, f"{subj}_nsd-allsubstim_sampling.npy")
        # we also save shuffled indices to shuffle RDM rows and columns to use as a null hypothesis (not used in paper)
        saved_shuffled_samples_file = os.path.join(samples_dir, f"{subj}_nsd-allsubstim_shuffling.npy",)

        # compute sampling if we are computing mpnet and it does not exist yet, else load
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
                    shuffler = np.random.permutation(range(100))
                    subj_shuffle_pool.append(shuffler)
                np.save(saved_samples_file, subj_sample_pool)
                np.save(saved_shuffled_samples_file, subj_shuffle_pool)
            else:
                raise FileNotFoundError(
                    "Saved samples not found for MPNET. Raising an error for security."
                    f"\n Looked in {saved_samples_file}"
                    "What happens here is that we try to load the 100x100 samples used for the original"
                    "MPNET sampling procedure, and reapply them for subsequent models, for fair"
                    "comparisons."
                    "\nIf the samples should already be computed, please check saved_samples_file"
                    "\nIf you are absolutely certain you want to recompute them, comment"
                    "out this error, and uncomment the following. Please make sure you are happy with"
                    "where sample_dir and other paths point to. Have a good day."
                )
        else:
            print(f"Loading 100x100 sample choices from {saved_samples_file}")
            subj_sample_pool = np.load(saved_samples_file, allow_pickle=True)
            subj_shuffle_pool = np.load(saved_shuffled_samples_file, allow_pickle=True)

        # run the searchlight mappings
        for j in range(subj_n_samples):
            file_save = os.path.join(
                searchlight_correlations_dir,
                f'{subj}_nsd-{MODEL_NAME}_{targetspace}{"_noShared515" if remove_shared_515 else ""}_sample-{j}.npy',
            )

            if os.path.exists(file_save):
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

                # now permute models and re-run correlation (used for statistical analysis at the single subject level, see above)
                # exactly same steps as above
                shuffler = subj_shuffle_pool[j]
                shuffled_rdms = np.asarray([reorder_rdm(utv, shuffler) for utv in model_rdms_sample])
                brain_maps_perm = corr_rdms(brain_sl_rdms_sample, shuffled_rdms)

                brain_vols_perm = []
                for map_i in range(n_models):
                    brain_map = brain_maps_perm[:, map_i]
                    brain_vect = np.zeros(np.prod(n_voxels))
                    brain_vect[rdms_sort] = brain_map.squeeze()
                    brain_vols_perm.append(np.reshape(brain_vect, n_voxels))
                brain_vols_perm = np.asarray(brain_vols_perm)

                # save correlation maps for that shuffle
                file_save = os.path.join(
                    searchlight_correlations_dir,
                    f'{subj}_nsd-{MODEL_NAME}_{targetspace}{"_noShared515" if remove_shared_515 else ""}_shuffle-{j}.npy',
                )
                np.save(file_save, brain_vols_perm)

                elapsed_time = time.time() - start_time
                print(f"boot {j} : elapsedtime : ", f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        del betas
        
        print("NSD searchlight mapping done.")
        elapsed_time = time.time() - initial_time
        print("elapsedtime: ", f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
