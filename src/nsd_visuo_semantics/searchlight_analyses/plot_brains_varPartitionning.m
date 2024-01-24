USE_FDR = 1;

% some parameters
viewz_to_plot = {13};  % {5,6,11,13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.
n_subjects = 8;
% cvn plot params
Lookup = [];
wantfig = 1; 
extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1], 'text',{'' ''}};

n_vertices = 327684;
hemis = {'lh', 'rh'};

% FACTORS = {"uniqueA", "uniqueB", "uniqueC", "uniqueAB", "uniqueAC", "uniqueBC", "uniqueABC"};
FACTORS = {('mpnet',), "('mpnet_nouns',)", "('mpnet_verbs',)", "('mpnet', 'mpnet_nouns')", "('mpnet', 'mpnet_verbs')", "('mpnet_nouns', 'mpnet_verbs')", "('mpnet', 'mpnet_nouns', 'mpnet_verbs')"};

for m = 1:7

    FACTOR = FACTORS{m}
    SAVE_MODEL_NAME = strcat(MODEL_NAMES{1}, FACTOR);

    datapath = fullfile(SEARCHLIGHT_SAVE_DIR, '%s', MODEL_NAMES{1}, 'fsaverage', '%s.%s-uniquevars-%s-surf.npy');
    main_data = single(zeros(n_subjects, n_vertices));
    % loop over subjects
    for sub = 1:n_subjects
        subj = sprintf('subj%02d', sub);
        sub_data = [];
        for hemi = 1:2
            this_hemi = hemis{hemi};
            sub_data = cat(1, sub_data, readNPY(sprintf(datapath, subj, this_hemi, subj, FACTOR)));
        end
        main_data(sub, :) = sub_data;
    end

    % where to save
    figpath  = fullfile(SEARCHLIGHT_SAVE_DIR, 'Figures', MODEL_NAMES{1});
    if ~exist(figpath)
        mkdir(figpath)
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
                if OVERWRITE | ~exist(fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.', SAVE_TYPE)))
                    [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, this_subj_data', [-max(abs(main_data(:))), max(abs(main_data(:)))], cmapsign4(256),[],Lookup,wantfig,extraopts);
                        title(sprintf('%s \n max: %3.2f', subj, max(abs(main_data(:)))))
                    saveas(gcf, fullfile(figpath, strcat(subj, '_view', num2str(this_view), '_', SAVE_MODEL_NAME)), SAVE_TYPE)
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
        if OVERWRITE | ~exist(fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.', SAVE_TYPE)))
            % non-thresholded image
            [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_corrs', [-max(mean_corrs(:)), max(mean_corrs(:))], cmapsign4(256), [], Lookup, wantfig, extraopts);
            title(sprintf('group average max: %3.2f', max_corr))
            saveas(gcf, fullfile(figpath, strcat('group_view', num2str(this_view), '_', SAVE_MODEL_NAME)), SAVE_TYPE)
        end
        close all;
        if OVERWRITE | ~exist(fullfile(figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME, '.', SAVE_TYPE)))
            % significant voxels only. Need hack to fix bug in cvnlookup
            cvn_plot_fix(mean_corrs_threshold, this_view, figpath, strcat('group_sig_view', num2str(this_view), '_', SAVE_MODEL_NAME), SAVE_MODEL_NAME, SAVE_TYPE, extraopts)
        end
        close all;
    end
end