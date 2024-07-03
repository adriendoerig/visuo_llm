import os
from nsd_visuo_semantics.encoding_decoding_analyses.nsd_llm_encoding_model import nsd_llm_encoding_model

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
nsd_derivatives_dir = '/share/klab/datasets/NSD_for_visuo_semantics_derivatives/'  # we will put data modified from nsd here
betas_dir = os.path.join(nsd_derivatives_dir, "betas")
base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses"

nsd_llm_encoding_model(EMBEDDING_MODEL_NAME, nsd_dir, betas_dir, base_save_dir)