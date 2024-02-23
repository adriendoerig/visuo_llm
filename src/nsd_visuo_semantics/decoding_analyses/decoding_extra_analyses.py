import pickle, os
import numpy as np
import pandas as pd
from aac_metrics import evaluate
from scipy.spatial.distance import cdist, correlation, cosine, pdist
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_conditions_515, get_rois,get_sentence_lists, load_or_compute_betas_average
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.decoding_analyses.decoding_utils import remove_inert_embedding_dims, restore_inert_embedding_dims, get_gcc_nearest_neighbour
from nsd_access import NSDAccess

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
METRIC = 'correlation'
save_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIallvisROIs/predicted_sentences_scores'

USE_ROIS = 'streams'  # None  # "mpnet_noShared515_sig0.005_fsaverage"  # None, or 'mpnet_noShared515_sig0.005_fsaverage' or streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...
which_rois = 'allvisROIs'  # allvisROIs, or 'independantvisRois'

nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses"
nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
results_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIallvisROIs"
fitted_models_dir = os.path.join(results_dir, "fitted_models")
mean_embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings.npy"
indiv_embeddings_test_path = f"{nsd_embeddings_path}/captions_515_embeddings_indiv.npy"

gcc_dir = '/share/klab/datasets/google_conceptual_captions'
sentences_path = os.path.join(gcc_dir, "conceptual_captions_{}.tsv")
embeddings_path = os.path.join(gcc_dir, "conceptual_captions_mpnet_{}.npy")
train_embeddings = np.load(embeddings_path.format('train'))
df = pd.read_csv(sentences_path.format('train'), sep="\t", header=None, names=["sent", "url"],)
train_sentences = df["sent"].to_list()
val_embeddings = np.load(embeddings_path.format('val'))
df = pd.read_csv(sentences_path.format('val'), sep="\t", header=None, names=["sent", "url"],)
val_sentences = df["sent"].to_list()
lookup_embeddings = np.vstack([train_embeddings, val_embeddings])
lookup_sentences = train_sentences + val_sentences

nsda = NSDAccess(nsd_dir)

if USE_ROIS is not None:
    maskdata, roi_id2name = get_rois(USE_ROIS, rois_dir)
    roi_name2id = {v: k for k, v in roi_id2name.items()}
    if which_rois == 'allvisROIs':
        ROI_NAMES = ['allvisROIs']
    else:
        ROI_NAMES = list(roi_name2id.keys())
        ROI_NAMES.remove("Unknown")
    print(f"Using ROI_NAMES = {ROI_NAMES}")
else:
    ROI_NAMES = ["fullbrain"]

n_sessions = 40
n_subjects = 8
subs = [f"subj0{x + 1}" for x in range(n_subjects)]
targetspace = "fsaverage"
conditions_515 = get_conditions_515(nsd_dir)
images_515 = nsda.read_images(np.asarray(conditions_515) - 1)
sentences_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)

mean_embeddings_test = np.load(mean_embeddings_test_path)
embedding_dim = mean_embeddings_test.shape[-1]

if not os.path.exists(indiv_embeddings_test_path):
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    captions_515 = get_sentence_lists(nsda, np.asarray(conditions_515) - 1)
    dummy_embedding = get_embeddings(captions_515[0], embedding_model, EMBEDDING_MODEL_NAME)
    embedding_dim = dummy_embedding.shape[-1]
    indiv_embeddings_test = np.empty((515, 5, embedding_dim))
    for i in range(len(captions_515)):
        if len(captions_515[i]) != 5:
            these_caps = captions_515[i][:5]
        else:
            these_caps = captions_515[i]
        indiv_embeddings_test[i] = get_embeddings(these_caps, embedding_model, EMBEDDING_MODEL_NAME)
    np.save(indiv_embeddings_test_path, indiv_embeddings_test)
else:
    indiv_embeddings_test = np.load(indiv_embeddings_test_path)

for ROI_NAME in ROI_NAMES:
    for subj in subs:

        # find indices that are NOT in the 515 spacial test images
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

        if USE_ROIS is not None:
            # load the lh mask
            orig_n_voxels = betas_mean.shape[0]
            if ROI_NAME.lower() == 'allvisrois':
                vs_mask = maskdata != 0
            else:
                vs_mask = maskdata == roi_name2id[ROI_NAME]
            betas_mean = betas_mean[vs_mask, :]
            print(f"Applied ROI mask. Went from {orig_n_voxels} to {betas_mean.shape[0]}")

        good_vertex = [True if np.sum(np.isnan(x)) == 0 else False for x in betas_mean]
        if np.sum(good_vertex) != len(good_vertex):
            print(f"found some NaN for {subj}")
        betas_mean = betas_mean[good_vertex, :]

        # now we further split the brain data according to the 515 test set or the training set for that subject
        betas_test = betas_mean[:, sample_515_bool].T  # sub1: (515, 327673) (n_voxels may vary from subj to subj because of nans)
        del betas_mean  # make space

        # format to float32,
        mean_embeddings_test, indiv_embeddings_test, betas_test = \
        mean_embeddings_test.astype(np.float32), indiv_embeddings_test.astype(np.float32), betas_test.astype(np.float32)

        model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridge_{ROI_NAME}.pkl"
        with open(model_save_path, "rb") as f:
            fitted_fracridge = pickle.load(f)

        # inert embedding dimensions are removed during training. We need to reinsert them here.
        mean_train_embeddings_path = (f"{nsd_embeddings_path}/captions_not515_embeddings_{subj}.npy")
        embeddings_train = np.load(mean_train_embeddings_path)
        _, drop_dims_idx, drop_dims_avgs = remove_inert_embedding_dims(embeddings_train)

        test_preds = fitted_fracridge.predict(betas_test)
        test_preds = restore_inert_embedding_dims(test_preds, drop_dims_idx, drop_dims_avgs)

        # h5_for_mpnet2text = '/share/klab/vbosch/mpnet_t5/adrien_pred_embeds'
        # h5_file = f'{h5_for_mpnet2text}/{subj}_predicted_mpnet.h5'
        # import h5py
        # dt = h5py.string_dtype(encoding='utf-8')
        # with h5py.File(h5_file, 'w') as f:
        #     # Create 'val' group
        #     val_group = f.create_group('val')
        #     val_group.create_dataset('captions', data=[s[0] for s in sentences_515], shape=len(sentences_515), dtype=dt)
        #     val_group.create_dataset('mpnet_embeddings', data=test_preds)

        lookup_save_path = f"{fitted_models_dir}/{subj}_predLookup_{ROI_NAME}.pkl"
        if not os.path.exists(lookup_save_path):
            test_preds_NN_lookup = get_gcc_nearest_neighbour(test_preds, lookup_embeddings=lookup_embeddings, lookup_sentences=lookup_sentences, METRIC='cosine')
            # test_preds_NN_lookup = [get_gcc_nearest_neighbour(p, lookup_embeddings=lookup_embeddings, lookup_sentences=lookup_sentences, METRIC='cosine') for p in test_preds[:5]]
            with open(lookup_save_path, "wb") as f:
                pickle.dump(test_preds_NN_lookup, f)
        else:
            with open(lookup_save_path, "rb") as f:
                test_preds_NN_lookup = pickle.load(f)

        if isinstance(test_preds_NN_lookup[0], list):
            print('Found list of lists, converting to list strings by using 0th entry of each list')
            test_preds_NN_lookup = [x[0] for x in test_preds_NN_lookup]

        print("Getting prediction-target CIDER scores on test set NN_lookup [PRED_LOOKUP VS 5 CAPTIONS]...")
        sentences_515 = [[string.replace('\n', '') for string in sublist] for sublist in sentences_515]
        mean_scores, all_scores = evaluate(test_preds_NN_lookup, sentences_515)
        with open(f"{save_dir}/{subj}_capEval_scores.pkl", "wb") as f:
            pickle.dump(all_scores, f)

        # prediction-target similarities will be used to visualize examples with different prediction accuracies
        print("Getting prediction-target scores on test set [PREDICTED EMBED VS AVG EMBED OVER 5 CAPTIONS]...")
        avg_cap_corrs = []
        for image_i in range(mean_embeddings_test.shape[0]):
            this_pred = test_preds[image_i, :]
            this_target = mean_embeddings_test[image_i, :]
            if METRIC == "correlation":
                this_distance = 1 - correlation(this_pred, this_target)
            elif METRIC == "cosine":
                this_distance = 1 - cosine(this_pred, this_target)
            else:
                raise Exception(f"METRIC not understood. You entered {METRIC}.")
            avg_cap_corrs.append(this_distance)
        avg_cap_corrs_argsort = np.argsort(avg_cap_corrs)
        np.save(f"{save_dir}/{subj}_avg_cap_corrs.npy", avg_cap_corrs)

        print("Getting prediction-target scores on test set [MEAN(PREDICTED EMBED VS EACH OF 5 CAPTIONS)]...")
        indiv_cap_corrs = []
        for image_i in range(indiv_embeddings_test.shape[0]):
            this_pred = test_preds[image_i, :]
            these_corrs = []
            for cap_i in range(5):
                this_target = indiv_embeddings_test[image_i, cap_i, :]
                if METRIC == "correlation":
                    these_corrs.append(1 - correlation(this_pred, this_target))
                elif METRIC == "cosine":
                    these_corrs.append(1 - cosine(this_pred, this_target))
                else:
                    raise Exception(f"METRIC not understood. You entered {METRIC}.")
            indiv_cap_corrs.append(np.mean(these_corrs))
        indiv_cap_corrs_argsort = np.argsort(indiv_cap_corrs)
        np.save(f"{save_dir}/{subj}_indiv_cap_corrs.npy", indiv_cap_corrs)


    print("Getting between-caption distances on test set [i.e., how consistent captions are] using CIDER..")
    sentences_515 = [[string.replace('\n', '') for string in sublist] for sublist in sentences_515]
    mean_scores_i = []
    all_scores_i = []
    for cap_i in range(5):
        these_candidates = [s[cap_i] for s in sentences_515]
        these_refs = [s[:cap_i] + s[cap_i + 1:] for s in sentences_515]
        these_mean_scores, these_all_scores = evaluate(these_candidates, these_refs)
        mean_scores_i.append(these_mean_scores)
        all_scores_i.append(these_all_scores)
    mean_scores = {k: np.mean([x[k] for x in mean_scores_i]) for k in mean_scores_i[0].keys()}
    all_scores = {k: np.array([x[k].numpy() for x in all_scores_i]).mean(axis=0) for k in all_scores_i[0].keys()}
    with open(f"{save_dir}/crossSubj_capEval_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)


    print("Getting between-caption distances on test set [i.e., how consistent captions are] using MPNet embed distances ...")
    cap_consists = []
    for image_i in range(indiv_embeddings_test.shape[0]):
        for cap_i in range(5):
            this_target = indiv_embeddings_test[image_i, cap_i, :]
            avg_other_caps = np.mean(np.delete(indiv_embeddings_test[image_i, :, :], cap_i, axis=0), axis=0)
            these_corrs = []
            if METRIC == "correlation":
                these_corrs.append(1 - correlation(avg_other_caps, this_target))
            elif METRIC == "cosine":
                these_corrs.append(1 - cosine(avg_other_caps, this_target))
            else:
                raise Exception(f"METRIC not understood. You entered {METRIC}.")
        cap_consists.append(np.mean(these_corrs))
    cap_consists_argsort = np.argsort(cap_consists)
    np.save(f"{save_dir}/crossSubj_capCorr_consists.npy", cap_consists)


