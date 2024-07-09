import os
from nsd_visuo_semantics.encoding_decoding_analyses.nsd_decode_llm import nsd_decode_llm

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
USE_ROIS = 'streams'  # None  # "mpnet_noShared515_sig0.005_fsaverage"  # None, or 'mpnet_noShared515_sig0.005_fsaverage' or streams, highlevelvisual, mpnet_sig0.05_fsaverage, ...
WHICH_ROIS = 'allvisROIs'  # allvisROIs, or 'independantvisRois'
METRIC = "correlation"  # Metric for nearest neighbour readout (NN) 'correlation', 'cosine'

nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "../results_dir/encoding_decoding_analyses"
gcc_dir = '/share/klab/datasets/google_conceptual_captions'  # GCC is used for NN readout

nsd_decode_llm(EMBEDDING_MODEL_NAME, USE_ROIS, WHICH_ROIS, METRIC, nsd_dir, betas_dir, gcc_dir, base_save_dir)