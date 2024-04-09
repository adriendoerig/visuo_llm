close all; clear all;

all_names = {'tim_people', 'tim_places', 'gpt_people', 'gpt_places', 'gpt_food', 'single_word_people', 'single_word_places', 'single_word_food'};
names_A = {'people'};
names_B = {'places'};

FDR = 1

maps = {'contrast'};

np_map_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/project_embeddings/cache';
figpath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/project_embeddings/brain_maps';
if ~exist(figpath)
    mkdir(figpath)
end

SAVE_TYPE = 'svg';  % 'svg' or 'png'

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
viewz_to_plot = {6,13};  % {5,6,11,13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.
% cvn plot params
Lookup = [];
wantfig = 1;

n_vertices = 327684;
n_subjects = 8;

for i = 1:length(names_A)
    name_A = names_A{i};
    
    for j = i:length(names_B)
        name_B = names_B{j};

        if strcmp(name_A, name_B)
            continue
        end

        for idx = 1:length(maps)
            s = maps{idx};

            rawdata = zeros(8, n_vertices);

            for sub = 1:n_subjects

                datapath_A = strcat(np_map_dir, '/subj0', num2str(sub), '_pred_voxels_', name_A, '.npy');
                datapath_B = strcat(np_map_dir, '/subj0', num2str(sub), '_pred_voxels_', name_B, '.npy');

                if strcmp(s, 'A')
                    rawdata(sub,: ) = readNPY(datapath_A);
                    figname = name_A;
                elseif strcmp(s, 'B')
                    rawdata(sub,: ) = readNPY(datapath_B);
                    figname = name_B;
                elseif strcmp(s, 'contrast')
                    rawdata(sub,: ) = readNPY(datapath_A) - readNPY(datapath_B);
                    figname = [name_A '_minus_' name_B];
                end
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

            % mean_data_threshold(isnan(mean_data_threshold)) = 0;

            [figname ' sum of significant vertices: ' num2str(sum(mean_data_threshold))]

            % formatting forplotting functions
            mean_data = mean_data';
            mean_data_threshold = mean_data_threshold';

            extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1], 'overlayalpha', mean_data_threshold};

            for v = 1:length(viewz_to_plot)
                cvn_plot_fix(mean_data', viewz_to_plot{v}, figpath, [figname '_view' num2str(viewz_to_plot{v}) '_' correc_str '.' SAVE_TYPE], figname, SAVE_TYPE, max_data, extraopts);
            end
        end
    end
end

close all; clear all
