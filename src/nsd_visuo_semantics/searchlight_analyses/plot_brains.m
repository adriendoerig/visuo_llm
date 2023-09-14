close all; clear all;

base_dir = '/rds/projects/c/charesti-start/';
searchlight_save_dir = '../results_dir/searchlight_respectedsampling_correlation';

RECTIFY_NEG_CORRS = 0;  % if 1, set all negative correlations to 0 for model comparisons (because neg rdm corrs are not so easy to interpret)
USE_FDR = 1;
OVERWRITE = 0;  % if 0, do not redo existing plots

% add path to cvncode, utils, and heklper functions
addpath(genpath(fullfile(base_dir,'software','cvncode')));
addpath(genpath(fullfile(base_dir,'software','knkutils')));
addpath(genpath(fullfile(base_dir,'software','npy-matlab')));
addpath(genpath(fullfile(base_dir,'software','npy-matlab')));

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
extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1], 'text',{'' ''}};

n_vertices = 327684;
hemis = {'lh', 'rh'};

ALL_MODEL_NAMES =  {'dnn_multihot_ff', 'dnn_multihot_rec', 'dnn_guse_ff', 'dnn_guse_rec', 'dnn_mpnet_ff', 'dnn_mpnet_rec', 'guse', 'multihot', 'mpnet', 'fasttext_categories', 'fasttext_all', 'fasttext_verbs', 'openai_ada2', 'dnn_ecoset_category', 'dnn_ecoset_fasttext'};
MODEL_NAMES = {'mpnet'};
MODEL_SUFFIX =  ''  % default is ''
CONTRAST_MODEL_NAMES = {'multihot'}  % ALL_MODEL_NAMES

DNN_LAYER = 10  % 'all' to do all layers, else an int
DNN_TIMESTEP = 6  % 'all' % 6  % 'all' to do all timesteps, else an int
DNN_CONTRAST_LAYER = 'same'  % same: compare net1 layer l time t with net2 same lt. 'first': compare with first timestep
CONTRAST_SAME_MODEL = 1  % if 0, do not contrast model with itself. else, do it. useful for e.g. contrasting t6 vs t0

PLOT_INDIVIDUAL_SUBJECTS = 0  % if 0, only do group level maps

for m1 = 1:length(MODEL_NAMES)

    MODEL_NAME = MODEL_NAMES{m1}

    if ~contains(MODEL_NAME, 'dnn_') || contains(MODEL_SUFFIX, 'fracridgeFit')
        is_dnn = 0
        DNN_LAYERS = 1:1
        DNN_TIMESTEPS = 1:1
    else
        is_dnn = 1
        if strcmp(DNN_LAYER, 'all')
            DNN_LAYERS = 1:10
        else
            DNN_LAYERS = DNN_LAYER
        end

        if contains(MODEL_NAME, '_ff')
            DNN_TIMESTEPS = 1:1
        else
            if strcmp(DNN_TIMESTEP, 'all')
                DNN_TIMESTEPS = 1:6
            else
                DNN_TIMESTEPS = DNN_TIMESTEP
            end
        end
    end

    for layer = DNN_LAYERS

        for timestep = DNN_TIMESTEPS

            if is_dnn
                SAVE_MODEL_NAME = strcat(MODEL_NAME, '_l', string(layer), '_t', string(timestep), MODEL_SUFFIX)
            else
                SAVE_MODEL_NAME = strcat(MODEL_NAME, MODEL_SUFFIX)
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % LOAD DATA
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % where is the data.
            if is_dnn
                map_id = (layer-1)*(DNN_TIMESTEPS(end))+timestep
            else
                map_id = 1
            end
            datapath = fullfile(searchlight_save_dir, '%s', MODEL_NAME, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.mgz');

            % where to save
            figpath  = fullfile(searchlight_save_dir, 'Figures', MODEL_NAME);
            if ~exist(figpath)
                mkdir(figpath)
            end

            main_data = single(zeros(n_subjects, n_vertices));
            % loop over subjects
            for sub = 1:n_subjects
                subj = sprintf('subj%02d', sub);
                sub_data = [];
                for hemi = 1:2
                    this_hemi = hemis{hemi};
                    sub_data = cat(1, sub_data, MRIread(sprintf(datapath, subj, strcat(MODEL_NAME, MODEL_SUFFIX), this_hemi, subj, string(map_id))).vol');
                end
                main_data(sub, :) = sub_data;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % INDIVIDUAL SUBJECT MAPS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if PLOT_INDIVIDUAL_SUBJECTS
                % plot each subject flatmap
                for sub = 1:n_subjects
                    subj = sprintf('subj%02d', sub);
                    this_subj_data = squeeze(main_data(sub,  :));
                    for v = 1:length(viewz_to_plot)
                        this_view = viewz_to_plot{v};
                        if OVERWRITE | ~exist(fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.svg')))
                            [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, this_subj_data', [-max(abs(main_data(:))), max(abs(main_data(:)))], cmapsign4(256),[],Lookup,wantfig,extraopts);
                             title(sprintf('%s \n max: %3.2f', subj, max(abs(main_data(:)))))
                            saveas(gcf, fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME)), 'svg')
                        end
                        close all;
                    end
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

            nanmin(mean_corrs_threshold(mean_corrs_threshold>0))
            nanmax(mean_corrs_threshold(mean_corrs_threshold<0))

            % plot
            for v = 1:length(viewz_to_plot)
                this_view = viewz_to_plot{v};
                if OVERWRITE | ~exist(fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.svg')))
                    % non-thresholded image
                    [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_corrs', [-max(mean_corrs(:)), max(mean_corrs(:))], cmapsign4(256), [], Lookup, wantfig, extraopts);
                    % title(sprintf('group average max: %3.2f', max_corr))
                    saveas(gcf, fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME)), 'svg')
                end
                close all;
                if OVERWRITE | ~exist(fullfile(figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.svg')))
                    % significant voxels only. Need hack to fix bug in cvnlookup
                    cvn_plot_fix(mean_corrs_threshold, this_view, figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME), SAVE_MODEL_NAME, extraopts)
                end
                close all;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CONTRAST WITH OTHER MODELS
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            for m2 = 1:length(CONTRAST_MODEL_NAMES)

                CONTRAST_MODEL_NAME = CONTRAST_MODEL_NAMES{m2}

                % do not compare model with itself
                if strcmp(CONTRAST_MODEL_NAME, MODEL_NAME) && ~CONTRAST_SAME_MODEL
                    continue
                end

                if ~contains(CONTRAST_MODEL_NAME, 'dnn_') || contains(MODEL_SUFFIX, 'fracridgeFit')
                    contrast_map_id = 1
                    CONTRAST_SAVE_MODEL_NAME = strcat(CONTRAST_MODEL_NAME, MODEL_SUFFIX)
                else
                    contrast_n_timesteps = DNN_TIMESTEPS(end)
                    if contains(DNN_CONTRAST_LAYER, 'same')
                        contrast_layer = layer
                        contrast_timestep = timestep
                    elseif contains(DNN_CONTRAST_LAYER, 'first')
                        contrast_layer = layer
                        contrast_timestep = 1
                    end
                    if contains(CONTRAST_MODEL_NAME, '_ff')
                        contrast_timestep = 1
                        contrast_n_timesteps = 1
                    end
                    contrast_map_id = (contrast_layer-1)*(contrast_n_timesteps)+contrast_timestep
                    CONTRAST_SAVE_MODEL_NAME = strcat(CONTRAST_MODEL_NAME, '_l', string(contrast_layer), '_t', string(contrast_timestep), MODEL_SUFFIX)
                end

                contrast_datapath = fullfile(searchlight_save_dir, '%s', CONTRAST_MODEL_NAME, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.mgz');
                contrast_data = single(zeros(n_subjects, n_vertices));

                % loop over subjects
                for sub = 1:n_subjects
                    subj = sprintf('subj%02d', sub);
                    sub_contrast_data = [];
                    for hemi = 1:2
                        this_hemi = hemis{hemi};
                        sub_contrast_data = cat(1, sub_contrast_data, MRIread(sprintf(contrast_datapath, subj, strcat(CONTRAST_MODEL_NAME, MODEL_SUFFIX), this_hemi, subj, string(contrast_map_id))).vol');
                    end
                    contrast_data(sub, :) = sub_contrast_data;
                end

                % ttest diff
                if RECTIFY_NEG_CORRS == 1
                    main_data(main_data<0) = 0;
                    contrast_data(contrast_data<0) = 0;
                    plt_suffix = '_rectNegCorr';
                else
                    plt_suffix = '';
                end

                mean_diff = squeeze(nanmean(main_data, 1))-squeeze(nanmean(contrast_data, 1));

                % ttest + fdr_bh version
                mean_diff_threshold = mean_diff;
                [raw_h, p] = ttest(main_data-contrast_data, 0);
                if USE_FDR
                    [adj_h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p, 0.05);  % https://ch.mathworks.com/matlabcentral/fileexchange/27418-fdr_bh
                    adj_h(isnan(adj_h))=0;  % remove nans (and count as not significant)
                    mean_diff_threshold(~adj_h) = nan;
                else
                    mean_diff_threshold(p>0.001) = nan;  % the thresh param of cvnlookup is buggy with 2tail tests, so we enforce it
                end

                nanmin(mean_diff_threshold(mean_diff_threshold>0))
                nanmax(mean_diff_threshold(mean_diff_threshold<0))

                % plot each subject contrast
                if PLOT_INDIVIDUAL_SUBJECTS
                    difference_data = main_data - contrast_data;
                    for sub = 1:n_subjects
                        subj = sprintf('subj%02d', sub);
                        this_subj_data = squeeze(difference_data(sub, :));
                        for v = 1:length(viewz_to_plot)
                            this_view = viewz_to_plot{v};
                            if OVERWRITE | ~exist(fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix, '.svg')))
                                [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, this_subj_data', [-max(abs(difference_data(:))), max(abs(difference_data(:)))], cmapsign4(256),[],Lookup,wantfig,extraopts);
                                 title(sprintf('%s \n max: %3.2f', subj, max(abs(difference_data(:)))))
                                saveas(gcf, fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix)), 'svg')
                            end
                            close all;
                        end
                    end
                end

                % plot group contrast
                for v = 1:length(viewz_to_plot)
                    this_view = viewz_to_plot{v};
                    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix, '.svg')))
                        % non-thresholded image
                        [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_diff', [-max(mean_diff(:)), max(mean_diff(:))], cmapsign4(256), [], Lookup, wantfig, extraopts);
                        % title(sprintf('%s vs. %s \n group max diff: %3.2f', SAVE_MODEL_NAME, CONTRAST_SAVE_MODEL_NAME, max(mean_diff(:))))
                        saveas(gcf, fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix)), 'svg');
                    end
                    close all;
                    if OVERWRITE | ~exist(fullfile(figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix, '.svg')))
                        % significant voxels only. Need hack to fix bug in cvnlookup
                        cvn_plot_fix(mean_diff_threshold, this_view, figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME, '_minus_', CONTRAST_SAVE_MODEL_NAME, plt_suffix), sprintf('%s vs. %s', SAVE_MODEL_NAME, CONTRAST_SAVE_MODEL_NAME), extraopts)
                    end
                    close all;
                end
            end
        end
        close all;
    end
    close all;
end
close all; clear all
