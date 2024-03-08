close all; clear all;

name = 'single_word_people_to_single_word_food';
np_map_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/interpolate_project_embeddings/cache';
n_steps = 100
only_sig = 1
FDR = 0
title_nn_sentence = 1 

figpath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/interpolate_project_embeddings/brain_maps';
if ~exist(figpath)
    mkdir(figpath)
end

SAVE_TYPE = 'png';  % 'svg' or 'png'

% YOU NEED TO DOWNLOAD CVNCODE, FREESURFER, KNKUTILS, AND NPY-MATLAB (see README.md)
% YOU NEED TO CHANGE THE PATHS BELOW TO YOUR OWN PATHS
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/cvncode')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/matlab')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/freesurfer/fsfast/toolbox')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/knkutils')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/software/npy-matlab/npy-matlab')));
addpath(genpath(fullfile('/share/klab/adoerig/adoerig/nsd_visuo_semantics/src/nsd_visuo_semantics/utils')));
setenv('SUBJECTS_DIR', fullfile('/share/klab/datasets/NSD_for_visuo_semantics/nsddata/freesurfer'));

% some parameters
colormap = cmapsign4(256);
viewz_to_plot = {13};  % {5,6,11,13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.
% cvn plot params
Lookup = [];
wantfig = 1;

n_vertices = 327684;
n_subjects = 8;

if title_nn_sentence
    NN_sentences_path = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/interpolate_project_embeddings/cache/nn_sentences.txt';
    NN_sentences = textread(NN_sentences_path, '%s', 'delimiter', '\n');
end

for i = 0:n_steps-1

    this_NN_sentence = NN_sentences{i+1};

    rawdata = zeros(8, n_vertices);

    for sub = 1:n_subjects
        load_name = strcat('subj0', num2str(sub), '_pred_voxels_', name, '_interp', num2str(i));
        datapath = [np_map_dir '/' load_name '.npy'];
        rawdata(sub,: ) = readNPY(datapath);
    end
    
    all_data = squeeze(rawdata);
    mean_data = squeeze(nanmean(all_data,1));
    max_data = max(mean_data(:));

    % ttest fdr version
    mean_data_threshold = ones(size(mean_data));
    [raw_h, p] = ttest(all_data, 0);
    [adj_h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(p, 0.05);  % https://ch.mathworks.com/matlabcentral/fileexchange/27418-fdr_bh
    if FDR
        mean_data_threshold(~adj_h) = nan;
        correc_str = 'fdr_correc';
    else
        mean_data_threshold(p>0.05) = nan;
        correc_str = 'p0.05 (no correc)';
    end

    mean_data_threshold(isnan(mean_data_threshold)) = 0;

    % [figname ' sum of significant vertices: ' num2str(sum(mean_data_threshold))]

    % formatting forplotting functions
    mean_data = mean_data'; %'

    if only_sig
        mean_data_threshold = mean_data_threshold';  %'
    else
        mean_data_threshold = ones(size(mean_data_threshold))'; %'
    end

    extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1], 'overlayalpha', mean_data_threshold};

    for v = 1:length(viewz_to_plot)
        this_view = viewz_to_plot{v};
        [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, mean_data, [], colormap, [], Lookup, wantfig, extraopts);
        if title_nn_sentence
            title(sprintf([name '(subjAvg, ' correc_str ')  \n' this_NN_sentence '\ni = ' num2str(i)]))
            fig_pos = get(gcf, 'Position');
            fig_pos(4) = fig_pos(4) + 100; % Increase height by 100 pixels
            set(gcf, 'Position', fig_pos);
        else
            title([name '(subjAvg, ' correc_str '), i = ' num2str(i)])
        end
        saveas(gcf, fullfile(figpath, [name '_interp' num2str(i) '_subjAvg_view' num2str(this_view)]), SAVE_TYPE)
    end
end

close all; clear all
