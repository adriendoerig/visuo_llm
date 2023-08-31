% How to run:
% ssh -X username@bluebear.bham.ac.uk
% fisbatch_screen as usual
% module load MATLAB/2020a
% matlab
% then, copy paste stuff from here in the matlab window (or simply type matlab, then plot_brains from matlab terminal)

close all; clear all;

base_dir = '/rds/projects/c/charesti-start/';
MODEL_NAME = 'mpnet_encodingModel'
USE_FDR = 1;
OVERWRITE = 0;  % if 0, do not redo existing plots

% add path to cvncode, utils, and heklper functions
addpath(genpath(fullfile(base_dir,'software','cvncode')));
addpath(genpath(fullfile(base_dir,'software','knkutils')));
addpath(genpath(fullfile(base_dir,'software','npy-matlab')));
addpath(genpath(fullfile(base_dir,'software','npy-matlab')));
%addpath(genpath(fullfile(base_dir,'software','MatlabTFCE')));

% add path to freesurfer tools
addpath(genpath(fullfile(base_dir,'software', 'freesurfer7', 'matlab')));

% we need to point to the subjects in NSD data
setenv('SUBJECTS_DIR', fullfile(base_dir,'data','NSD','nsddata','freesurfer'));

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

encoding_results_dir = './save_dir/decoding/all_mpnet_base_v2_results_ROIfullbrain_encodingModel';
datapath = fullfile(encoding_results_dir, 'fitted_models', '%s_fittedFracridgeCorrMap_fullbrain.pkl');

% where to save
figpath  = fullfile(encoding_results_dir, 'Figures');
if ~exist(figpath)
    mkdir(figpath)
end

main_data = single(zeros(n_subjects, n_vertices));
% loop over subjects
for sub = 1:n_subjects
    subj = sprintf('subj%02d', sub);
    fid = py.open(sprintf(datapath, subj), 'rb');
    sub_data = py.pickle.load(fid).tolist();  % next three lines are to deal with matlab being incapable of simply loading pkl
    sub_data = cell(sub_data);
    sub_data = cell2mat(sub_data);
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
        if OVERWRITE | ~exist(fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', MODEL_NAME, '.svg')))
            [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, this_subj_data', [-max(this_subj_data(:)), max(this_subj_data(:))], cmapsign4(256),[],Lookup,wantfig,extraopts);
            title(sprintf('%s \n max: %3.2f', subj, max(this_subj_data(:))))
            saveas(gcf, fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', MODEL_NAME)), 'svg')
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

% plot
for v = 1:length(viewz_to_plot)
    this_view = viewz_to_plot{v};
    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_view', num2str(this_view), '_', MODEL_NAME, '.svg')))
        % non-thresholded image
        [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_corrs', [-max(mean_corrs(:)), max(mean_corrs(:))], cmapsign4(256), [], Lookup, wantfig, extraopts);
        title(sprintf('group average max: %3.2f', max_corr))
        saveas(gcf, fullfile(figpath, strcat('group_view', num2str(this_view), '_', MODEL_NAME)), 'svg')
    end
    close all;
    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_sig_view', num2str(this_view), '_', MODEL_NAME, '.svg')))
        % significant voxels only. Need hack to fix bug in cvnlookup
        cvn_plot_fix(mean_corrs_threshold, this_view, figpath, strcat('group_sig_view', num2str(this_view), '_', MODEL_NAME), MODEL_NAME)
    end
    close all;
end
close all; clear all
