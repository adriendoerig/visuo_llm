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
SAVE_TYPE = 'svg';  % 'svg' or 'png'
SEARCHLIGHT_SAVE_DIR = '..
;
RECTIFY_NEG_CORRS = 0;  % if 1, set all negative correlations to 0 for model comparisons (because neg rdm corrs are not so easy to interpret)

ALL_MODEL_NAMES =  {'dnn_bihem_bottleneckCCApr24', 'dnn_bihem_bottleneckNoCCApr24', 'dnn_bihem_singleStreamApr24'};
MODEL_NAME = {"dnn_bihem_bottleneckCCApr24"};  % , 'dnn_mpnet_rec_avgSeed_ep200',  "multihot", "fasttext_nouns", "nsd_fasttext_nouns_closest_cocoCats_cut0.33", "dnn_multihot_rec", "dnn_mpnet_rec", "var_partition_['mpnet', 'mpnet_nouns', 'mpnet_verbs']"};
MODEL_SUFFIX =  '';  % default is ''
CONTRAST_MODEL_NAME = {'dnn_bihem_bottleneckCCApr24'};  % ALL_MODEL_NAMES

DNN_LAYER_NAME = 'gap_left';
DNN_CONTRAST_LAYER_NAME = 'gap_right';

PLOT_INDIVIDUAL_SUBJECTS = 0;  % if 0, only do group level maps

MAX_CMAP_VAL = 0  % CONSTANT CMAP MAX VAL FOR ALL GROUP SIG PLOTS. USE 0 TO AUTOMATICALLY SET THE MAX VALUE WITH EACH MAP

viewz_to_plot = {13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.

% Now, we do some formatting to get the right layers from the layer MODEL_NAMES
layerNames2IDs = struct();
dnn_bihem_bottleneckCCApr24_layersNames = {'gap_left', 'gap_left_ln', 'gap_right', 'gap_right_ln', 'it_left', 'it_left_ln', 'it_right', 'it_right_ln', 'left_postV1_to_right_preV2_wm', 'left_postV2_to_right_preV4_wm', 'left_postV4_to_right_preIT_wm', 'post_gap_left_relu', 'post_gap_right_relu', 'post_it_left_relu', 'post_it_right_relu', 'post_retina_left_relu', 'post_retina_right_relu', 'post_v1_left_relu', 'post_v1_right_relu', 'post_v2_left_relu', 'post_v2_right_relu', 'post_v4_left_relu', 'post_v4_right_relu', 'pre_concat_bottleneck_left', 'pre_concat_bottleneck_right', 'readout', 'retina_left', 'retina_right', 'right_postV1_to_left_preV2_wm', 'right_postV2_to_left_preV4_wm', 'right_postV4_to_left_preIT_wm', 'v1_left', 'v1_left_ln', 'v1_right', 'v1_right_ln', 'v2_left', 'v2_left_ln', 'v2_right', 'v2_right_ln', 'v4_left', 'v4_left_ln', 'v4_right', 'v4_right_ln'};
dnn_bihem_bottleneckCCApr24_layersIDs = 1:numel(dnn_bihem_bottleneckCCApr24_layersNames);
layerNames2IDs.('dnn_bihem_bottleneckCCApr24') = containers.Map(dnn_bihem_bottleneckCCApr24_layersNames, dnn_bihem_bottleneckCCApr24_layersIDs);
dnn_bihem_bottleneckNoCCApr24_layersNames = {'gap_left', 'gap_left_ln', 'gap_right', 'gap_right_ln', 'it_left', 'it_left_ln', 'it_right', 'it_right_ln', 'left_postV1_to_right_preV2_wm', 'left_postV2_to_right_preV4_wm', 'left_postV4_to_right_preIT_wm', 'post_gap_left_relu', 'post_gap_right_relu', 'post_it_left_relu', 'post_it_right_relu', 'post_retina_left_relu', 'post_retina_right_relu', 'post_v1_left_relu', 'post_v1_right_relu', 'post_v2_left_relu', 'post_v2_right_relu', 'post_v4_left_relu', 'post_v4_right_relu', 'pre_concat_bottleneck_left', 'pre_concat_bottleneck_right', 'readout', 'retina_left', 'retina_right', 'right_postV1_to_left_preV2_wm', 'right_postV2_to_left_preV4_wm', 'right_postV4_to_left_preIT_wm', 'v1_left', 'v1_left_ln', 'v1_right', 'v1_right_ln', 'v2_left', 'v2_left_ln', 'v2_right', 'v2_right_ln', 'v4_left', 'v4_left_ln', 'v4_right', 'v4_right_ln'};
dnn_bihem_bottleneckNoCCApr24_layersIDs = 1:numel(dnn_bihem_bottleneckNoCCApr24_layersNames);
layerNames2IDs.('dnn_bihem_bottleneckNoCCApr24') = containers.Map(dnn_bihem_bottleneckNoCCApr24_layersNames, dnn_bihem_bottleneckNoCCApr24_layersIDs);
dnn_bihem_singleStreamApr24_layersNames = {'bottleneck', 'gap', 'gap_ln', 'it', 'it_ln', 'post_gap_relu', 'post_it_relu', 'post_retina_relu', 'post_v1_relu', 'post_v2_relu', 'post_v4_relu', 'readout', 'retina', 'v1', 'v1_ln', 'v2', 'v2_ln', 'v4', 'v4_ln'};
dnn_bihem_singleStreamApr24_layersIDs = 1:numel(dnn_bihem_singleStreamApr24_layersNames);
layerNames2IDs.('dnn_bihem_singleStreamApr24') = containers.Map(dnn_bihem_singleStreamApr24_layersNames, dnn_bihem_singleStreamApr24_layersIDs);

DNN_LAYER = layerNames2IDs.(MODEL_NAME).(DNN_LAYER_NAME);
DNN_CONTRAST_LAYER = layerNames2IDs.(CONTRAST_MODEL_NAME).(DNN_CONTRAST_LAYER_NAME);
dbstop

% this actually does the plotting
plot_brains_bihem
