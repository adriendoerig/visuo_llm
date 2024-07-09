import os
import numpy as np
from nsd_visuo_semantics.encoding_decoding_analyses.nsd_llm_encoding_model import nsd_llm_encoding_model
from nsd_visuo_semantics.encoding_decoding_analyses.text_to_brain_prediction import text_to_brain_prediction, plot_predicted_brain_contrast
from nsd_visuo_semantics.encoding_decoding_analyses.encoding_decoding_utils import sentences_zoo
from nsd_visuo_semantics.utils.py_plot_brain_utils import pyplot_indiv_subjects, pyplot_subj_avg

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "../results_dir/encoding_decoding_analyses"
encoding_models_dir = f'{base_save_dir}/{EMBEDDING_MODEL_NAME}_encodingModel/fitted_models'
fig_save_dir = f'{base_save_dir}/{EMBEDDING_MODEL_NAME}_encodingModel/figures'
text_to_brain_preds_dir = f'{base_save_dir}/{EMBEDDING_MODEL_NAME}_encodingModel/text_to_brain_preds'

# train encoding model for each subject and save model, preds, coeffs
# nsd_llm_encoding_model(EMBEDDING_MODEL_NAME, nsd_dir, betas_dir, base_save_dir)

# plot performance brain maps
# n_subjects, n_vertices = 8, 327684
# encoding_preds = np.empty((n_subjects, n_vertices))
# for sub in range(8):
#     encoding_preds[sub, :] = np.load(f'{encoding_models_dir}/subj{sub+1:02d}_fittedFracridgeEncodingCorrMap.npy')
# pyplot_indiv_subjects(encoding_preds, f'{EMBEDDING_MODEL_NAME}_encodingPerf', fig_save_dir, save_type='png')
# pyplot_subj_avg(encoding_preds, f'{EMBEDDING_MODEL_NAME}_encodingPerf', fig_save_dir, sig_mask='fdr_bh', save_type='png')

# predict brain activities from user-derived sentences
# for name in ['people', 'places', 'food']:
#     text_to_brain_prediction(EMBEDDING_MODEL_NAME, sentences_zoo[name], name, 
#                              encoding_models_dir, text_to_brain_preds_dir)

# plot brain maps for the predicted activities
preds_save_dir = f'{text_to_brain_preds_dir}/cache'
for contrast in [['people', 'places'], ['food', 'people']]:
    plot_predicted_brain_contrast(contrast, preds_save_dir, fig_save_dir, save_type='png', plot_indiv_sub=True, plot_subj_avg=True)