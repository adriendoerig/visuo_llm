close all; clear all;

% YOU NEED TO DOWNLOAD CVNCODE, FREESURFER, KNKUTILS, AND NPY-MATLAB (see README.md)
% YOU NEED TO CHANGE THE PATHS BELOW TO YOUR OWN PATHS
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/cvncode')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/matlab')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/fsfast/toolbox')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/knkutils')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/npy-matlab/npy-matlab')));
setenv('SUBJECTS_DIR', fullfile('/share/klab/datasets/NSD_for_visuo_semantics/nsddata/freesurfer'));
% Paths within this repository
addpath(genpath(fullfile('../src/nsd_visuo_semantics/utils')));
addpath(genpath(fullfile('../src/nsd_visuo_semantics/searchlight_analyses')));

% parameters for plotting
OVERWRITE = 1;  % if 0, do not redo existing plots
SAVE_TYPE = 'png';  % 'svg' or 'png'
SEARCHLIGHT_SAVE_DIR = '../results_dir/searchlight_respectedsampling_correlation_newTest';
RECTIFY_NEG_CORRS = 0;  % if 1, set all negative correlations to 0 for model comparisons (because neg rdm corrs are not so easy to interpret)

ALL_MODEL_NAMES =  {'dnn_multihot_ff', 'dnn_multihot_rec', 'dnn_guse_ff', 'dnn_guse_rec', 'dnn_mpnet_ff', 'dnn_mpnet_rec', 'guse', 'multihot', 'mpnet', 'fasttext_categories', 'fasttext_all', 'fasttext_verbs', 'dnn_ecoset_category', 'dnn_ecoset_fasttext'};
MODEL_NAMES = {"mpnet"};  % , "multihot", "fasttext_nouns", "nsd_fasttext_nouns_closest_cocoCats_cut0.33", "dnn_multihot_rec", "dnn_mpnet_rec"};
MODEL_SUFFIX =  '';  % default is ''
CONTRAST_MODEL_NAMES = {};  % ALL_MODEL_NAMES

DNN_LAYER = 'all';  % 'all' to do all layers, else an int
DNN_TIMESTEP = 6;  % 'all' % 6  % 'all' to do all timesteps, else an int
DNN_CONTRAST_LAYER = 'first';  % same: compare net1 layer l time t with net2 same lt. 'first': compare with first timestep
CONTRAST_SAME_MODEL = 1;  % if 0, do not contrast model with itself. else, do it. useful for e.g. contrasting t6 vs t0

PLOT_INDIVIDUAL_SUBJECTS = 1;  % if 0, only do group level maps

% this actually does the plotting
plot_brains