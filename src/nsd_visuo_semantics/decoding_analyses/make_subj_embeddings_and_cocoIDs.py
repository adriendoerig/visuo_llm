import os
import pandas as pd
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import get_conditions, get_sentence_lists
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_access import NSDAccess


nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses"
os.makedirs(base_save_dir, exist_ok=True)
nsd_embeddings_path = os.path.join(base_save_dir, "nsd_caption_embeddings")
os.makedirs(nsd_embeddings_path, exist_ok=True)

nsda = NSDAccess(nsd_dir)

subj = "subj01"
n_sessions = 40
n_subjects = 8

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_dim = 768

save_dir = '/share/klab/datasets/_for_philip_from_adrien/nsd_tsne'
embeddings_save_path = f"{save_dir}/nsd_embeddings_{subj}.npy"
coco_ids_save_path = f"{save_dir}/nsd_coco_ids_{subj}.npy"

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
nsd_indices = np.unique(conditions[conditions_bool])

embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
captions, coco_IDs = get_sentence_lists(nsda, nsd_indices - 1, return_coco_ids=True)
embeddings = np.empty((len(captions), 5, embedding_dim))  # 5 is the number of captions per image

for i in range(len(captions)):
    print(i)
    try:
        if len(captions[i]) != 5:
            these_caps = captions[i][:5]
        else:
            these_caps = captions[i]
        embeddings[i,:,:] = get_embeddings(these_caps, embedding_model, EMBEDDING_MODEL_NAME)
    except:
        import pdb; pdb.set_trace()
np.save(embeddings_save_path, embeddings)
np.save(coco_ids_save_path, coco_IDs)

import pdb; pdb.set_trace()