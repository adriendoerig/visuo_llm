'''In this script, we preject sentences into LLM embedding space,
and then predict voxel activities based on these embeddings.'''

import pickle, os
import numpy as np
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.decoding_analyses.decoding_utils import restore_nan_dims, sentences_zoo


EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
average_over = 'embeddings'  # embeddings or voxels

fitted_models_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/fitted_models"
save_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/project_embeddings/cache'
os.makedirs(save_dir, exist_ok=True)

embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

for name in ['objects']:  #sentences_zoo.keys():

    sentences = sentences_zoo[name]

    embeds = get_embeddings(sentences, embedding_model, EMBEDDING_MODEL_NAME)

    if average_over == 'embeddings':
        embeds = np.mean(embeds, axis=0)[np.newaxis, :]

    subs = [f'subj0{s}' for s in range(1,9)]
    pred_voxels_all = []
    for subj in subs:

        model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel_fullbrain.pkl"
        with open(model_save_path, "rb") as f:
            fitted_fracridge = pickle.load(f)

        pred_voxelsA = fitted_fracridge.predict(embeds).squeeze()

        nan_idx_to_restore = np.load(f"{fitted_models_dir}/{subj}_NanIdxToRestore.npy")
        pred_voxels = restore_nan_dims(pred_voxelsA, nan_idx_to_restore, axis=0)

        np.save(f"{save_dir}/{subj}_pred_voxels_{name}.npy", pred_voxels)

        pred_voxels_all.append(pred_voxels)

    # average all_voxels, and save
    pred_voxels_all = np.mean(pred_voxels_all, axis=0)

    np.save(f"{save_dir}/subjAvg_pred_voxels_{name}.npy", pred_voxels_all)

