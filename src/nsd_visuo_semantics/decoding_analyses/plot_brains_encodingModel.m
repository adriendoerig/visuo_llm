close all; clear all;

MODEL_NAME = 'mpnet_encodingModel_split0'
USE_FDR = 1;
OVERWRITE = 0;  % if 0, do not redo existing plots
SAVE_TYPE = 'png';  % 'svg' or 'png'
MAX_CMAP_VAL = 0;

% YOU NEED TO DOWNLOAD CVNCODE, FREESURFER, KNKUTILS, AND NPY-MATLAB (see README.md)
% YOU NEED TO CHANGE THE PATHS BELOW TO YOUR OWN PATHS
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/cvncode')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/matlab')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/fsfast/toolbox')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/knkutils')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/npy-matlab/npy-matlab')));
setenv('SUBJECTS_DIR', fullfile('/share/klab/datasets/NSD_for_visuo_semantics/nsddata/freesurfer'));
% Paths within this repository
addpath(genpath(fullfile('../utils')));
addpath(genpath(fullfile('../src/nsd_visuo_semantics/searchlight_analyses')));
addpath(genpath(fullfile('../src/nsd_visuo_semantics/searchlight_analyses')));

% some parameters
viewz_to_plot = {13};  % {5,6,11,13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.
n_subjects = 8;
% cvn plot params
Lookup = [];
wantfig = 1;
extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1]};

n_vertices = 327684;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% encoding_results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel';
% datapath = fullfile(encoding_results_dir, 'fitted_models', '%s_fittedFracridgeEncodingCorrMap_fullbrain.npy');
encoding_results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel_split0';
datapath = fullfile(encoding_results_dir, 'fitted_models', '%s_fittedFracridgeEncodingCorrMap_fullbrain_all-mpnet-base-v2.npy');
% encoding_results_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/mpnet_rec_seed1_nsd_activations_epoch200_layer0_results_ROIfullbrain_encodingModel';
% datapath = fullfile(encoding_results_dir, 'fitted_models', '%s_fittedFracridgeEncodingCorrMap_fullbrain_mpnet_rec_seed1_nsd_activations_epoch200.npy');

% where to save
figpath  = fullfile(encoding_results_dir, 'Figures');
if ~exist(figpath)
    mkdir(figpath)
end

main_data = single(zeros(n_subjects, n_vertices));
% loop over subjects
for sub = 1:n_subjects
    sub_data = [];
    subj = sprintf('subj%02d', sub)
    datapath
    sprintf(datapath, subj)
    sub_data = cat(1, sub_data, readNPY(sprintf(datapath, subj)));
    main_data(sub, :) = sub_data;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INDIVIDUAL SUBJECT MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot each subject flatmap
for sub = 1:n_subjects
    subj = sprintf('subj%02d', sub);
    this_subj_data = squeeze(main_data(sub,  :));
    for v = 1:length(viewz_to_plot)
        this_view = viewz_to_plot{v};
        if OVERWRITE | ~exist(fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', MODEL_NAME, '.', SAVE_TYPE)))
            [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, this_subj_data', [-max(this_subj_data(:)), max(this_subj_data(:))], cmapsign4(256),[],Lookup,wantfig,extraopts);
            title(sprintf('%s \n max: %3.2f', subj, max(this_subj_data(:))))
            saveas(gcf, fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', MODEL_NAME)), SAVE_TYPE)
        end
        close all;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GROUP MAP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get group results. IF contrasting models, need to load sub_data1 and sub_data2. subtract means. and t-test [h, p] = ttest(sub_data1, subdata2);
% get the mean values
mean_corrs = squeeze(nanmean(main_data,1));
max_corr = max(mean_corrs(:));

% ttest fdr version
mean_corrs_threshold = mean_corrs;
[raw_h, p] = ttest(main_data, 0);%, 'tail', 'right');
if USE_FDR
    [adj_h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p, 0.05);  % https://ch.mathworks.com/matlabcentral/fileexchange/27418-fdr_bh
    %adj_h(isnan(adj_h))=0;  % remove nans (and count as not significant)
    mean_corrs_threshold(~adj_h) = nan;
else
    mean_corrs_threshold(p>0.001) = nan;
end

sig_mask = ones(size(mean_corrs_threshold));
sig_mask(isnan(mean_corrs_threshold)) = 0;
% save the mask
save(fullfile(encoding_results_dir, strcat('encoding_sig_mask_', MODEL_NAME, '.mat')), 'sig_mask')


% plot
for v = 1:length(viewz_to_plot)
    this_view = viewz_to_plot{v};
    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_view', num2str(this_view), '_', MODEL_NAME, '.', SAVE_TYPE)))
        % non-thresholded image
        [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_corrs', [-max(mean_corrs(:)), max(mean_corrs(:))], cmapsign4(256), [], Lookup, wantfig, extraopts);
        title(sprintf('group average max: %3.2f', max_corr))
        saveas(gcf, fullfile(figpath, strcat('group_view', num2str(this_view), '_', MODEL_NAME)), SAVE_TYPE)
    end
    close all;
    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_sig_view', num2str(this_view), '_', MODEL_NAME, '.', SAVE_TYPE)))
        % significant voxels only. Need hack to fix bug in cvnlookup
        cvn_plot_fix(mean_corrs_threshold, this_view, figpath, strcat('group_sig_view', num2str(this_view), '_', MODEL_NAME), MODEL_NAME, SAVE_TYPE, MAX_CMAP_VAL, extraopts)
    
    end
    close all;
end
close all; clear all
