'''In this script, we preject sentence_list into LLM embedding space,
and then predict voxel activities based on these embeddings.'''

import pickle, os
import numpy as np
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.encoding_decoding_analyses.encoding_decoding_utils import restore_nan_dims
from nsd_visuo_semantics.utils.py_plot_brain_utils import pyplot_indiv_subjects, pyplot_subj_avg


def text_to_brain_prediction(EMBEDDING_MODEL_NAME, sentence_list, name,
                             fitted_models_dir, save_dir):
    '''
    predict voxel activities based on LLM embeddings of sentence_list.
    If more than one sentence is passed, we average the embeddings of the sentence_list.
    EMBEDDING_MODEL_NAME: name of the LLM model to use for embeddings
    sentence_list: List of sentence_list to project into brain space.
    name: str to append to predicted activities saved filename.
    fitted_models_dir: directory where the fitted encoding models are saved.
    save_dir: directory to save the predicted voxel activities.
    '''

    if not isinstance(sentence_list, list):
        sentence_list = [sentence_list]

    if 'cache' not in save_dir.split('/')[-1].lower():
        save_dir = os.path.join(save_dir, 'cache')

    print(f"Predicting voxels from {sentence_list}.")

    os.makedirs(save_dir, exist_ok=True)

    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    embeds = get_embeddings(sentence_list, embedding_model, EMBEDDING_MODEL_NAME)
    embeds = np.mean(embeds, axis=0)[np.newaxis, :]

    subs = [f'subj0{s}' for s in range(1,9)]
    pred_voxels_all = []
    for subj in subs:

        model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel.pkl"
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


def plot_predicted_brain_contrast(contrast, preds_save_dir, fig_save_dir, save_type='png', plot_indiv_sub=True, plot_subj_avg=True):
    
    '''plot brain maps for the predicted activities'''

    n_subjects, n_vertices = 8, 327684

    contrast_str = contrast[0] + '_minus_' + contrast[1]

    pred_voxelsA = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    pred_voxelsB = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    contrasts = np.zeros((n_subjects, n_vertices), dtype=np.float32)

    for sub in range(n_subjects):
        pred_voxelsA[sub] = np.load(f"{preds_save_dir}/subj{sub+1:02}_pred_voxels_{contrast[0]}.npy")
        pred_voxelsB[sub] = np.load(f"{preds_save_dir}/subj{sub+1:02}_pred_voxels_{contrast[1]}.npy")
        contrasts[sub] = pred_voxelsA[sub] - pred_voxelsB[sub]

    if plot_indiv_sub:
        pyplot_indiv_subjects(contrasts, f'{contrast_str}_predicted', fig_save_dir, save_type=save_type)
    if plot_subj_avg:
        pyplot_subj_avg(contrasts, f'{contrast_str}_predicted', fig_save_dir, sig_mask='uncorrected', save_type=save_type)
    if not plot_indiv_sub and not plot_subj_avg:
        raise ValueError('At least one of plot_indiv_sub or plot_subj_avg should be True.')