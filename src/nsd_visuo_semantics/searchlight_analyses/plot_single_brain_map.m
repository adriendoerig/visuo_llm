close all; clear all;

% fignames = {'pcaOnZscoreFullBrain', 'pcaOnZscoreVisROIs', 'pcaOnZscoreSigOnly', 'clusterAssigmentsFullBrain', 'clusterAssigmentsVisROIs', 'clusterAssignmentsSigOnly'};
fignames = {'pcaOnZscoreVisROIs'};
% fignames = {'clusterAssigmentsFullBrain', 'clusterAssigmentsVisROIs', 'clusterAssignmentsSigOnly'};
% fignames = {'clusterAssigmentsVisROIs_nclusters11'};

for f = 1:length(fignames)
    figname = fignames{f}

    if contains(lower(figname), 'cluster')
        if contains(lower(figname), 'fullbrain')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_fullBrain_cluster_assignments_subjavg_nclusters11_zscored.npy';
            mask_path = ''  % you can provide a path to mask of 0s and 1s. Not used if ''
        elseif contains(lower(figname), 'visrois')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_visROIs_cluster_assignments_subjavg_nclusters11_zscored.npy';
            mask_path = ''  % you can provide a path to mask of 0s and 1s. Not used if ''
        elseif contains(lower(figname), 'sigonly')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_sigOnly_cluster_assignments_subjavg_nclusters11_zscored.npy';
            mask_path = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/encoding_sig_mask_mpnet_encodingModel.mat';  % you can provide a path to mask of 0s and 1s. Not used if ''
        end
    else
        if contains(lower(figname), 'fullbrain')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_pca3D_fullbrain_average_3componentszscore.npy';
            mask_path = ''  % you can provide a path to mask of 0s and 1s. Not used if ''
        elseif contains(lower(figname), 'visrois')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_pca3D_average_3componentszscore.npy';
            mask_path = ''  % you can provide a path to mask of 0s and 1s. Not used if ''
        elseif contains(lower(figname), 'sigonly')
            datapath = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/encodingModelCoeffs_pca3D_sigOnly_average_3componentszscore.npy';
            mask_path = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/encoding_sig_mask_mpnet_encodingModel.mat';  % you can provide a path to mask of 0s and 1s. Not used if ''
        end
    end

    overlay_alpha_datapath = ''  % '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/fitted_models/subj08_fittedFracridgeEncodingCorrMap_fullbrain.npy';  % transparancy map. Use '' to skip

    normalize_overlay = 0;  % if 1, normalize overlay to have a max of 1

    figpath  = fullfile('/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses', 'brain_maps');
    if ~exist(figpath)
        mkdir(figpath)
    end

    OVERWRITE = 1;  % if 0, do not redo existing plots
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
    colormap = cmapsign4(256);  % cmapsign4(256); 'huth3d'
    viewz_to_plot = {6,13};  % {5,6,11,13};  % determines which angle the brain is seen at. 13 is the standard flatmap. see also 5&6.
    % cvn plot params
    Lookup = [];
    wantfig = 1;

    n_vertices = 327684;
        
    rawdata = readNPY(datapath);

    % dims = {1,2,3,'huth3d','clusters'};
    % dims = {'clusters'};
    dims = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};

    for i = 1:length(dims)

        dim = dims{i};

        % check if colormap is a string
        if ischar(dim)
            if strcmp(dim, 'clusters')
                'Using cluster colormap'

                data = rawdata;

                %colormap = [215 48 39
                %            252 41 89
                %            253 224 144
                %            255 255 191
                %            224 243 248
                %            145 191 219
                %            69 117 180
                %            11 102 35
                %            ] / 255;

                colormap = [
                    0 0 0;
                    0 1 1;          % Cyan
                    1 0.6471 0;     % Orange
                    1 1 0;          % Yellow
                    0.4784 0.0627 1;% Purple
                    0 1 0;          % Green
                    0 0.8039 1;     % Sky Blue
                    1 0 1;          % Magenta
                    1 0.2706 0.5412;% Pink
                    0.1137 0.5098 0.2745;  % Dark Green
                    1 0.6471 0;     % Orange
                    1 0.8431 0.898; % Light Pink
                    0.5176 1 0.898; % Light Green
                        ];

            elseif strcmp(dim, 'huth3d')
                'Using huth3d colormap'

                data = rawdata;

                %data = abs(data);

                %% first, make the color map
                color1 = [190 63 60]/255;
                color2 = [136 172 91]/255;
                color3 = [62 85 155]/255;
                %color1 = [230 17 34]/255;  % PC1 = red
                %color2 = [16 230 34]/255;  % PC2 = green
                %color3 = [16 16 226]/255;  % PC3 = blue

                [xx,yy,zz] = ndgrid(1:5,1:5,1:5);

                colormap = [];
                for p=1:size(xx,1)
                    for q=1:size(xx,2)
                        for r=1:size(xx,3)
                            color0 = (xx(p,q,r)-1)/4*color1 + (yy(p,q,r)-1)/4*color2 + (zz(p,q,r)-1)/4*color3;
                            colormap(end+1,:) = min(color0,[1 1 1]);
                        end
                    end
                end

                %% second, make a magical function

                % define function that takes indices 1-5 along three dimensions
                % to the final colormap index betwen 1 and 125
                cfun = @(x,y,z) (x-1)*5*5 + (y-1)*5 + z;

                %% demonstrate on some data

                % fake data
                xval = data(:,1);
                yval = data(:,2);
                zval = data(:,3);

                % conform to the range [.5,5.499] using the min and max of -2 and 2,
                % and then round so that we get 1 through 5 integers.
                xval2 = round(normalizerange(xval,.5,5.499,-2,2));
                yval2 = round(normalizerange(yval,.5,5.499,-2,2));
                zval2 = round(normalizerange(zval,.5,5.499,-2,2));

                % now, take your data and use the appropriate colormap:
                % use cfun(xval2,yval2,zval2) with min value 0.5 and max value 125.5 and colormap

                data = cfun(xval2,yval2,zval2);
                
                size(data)

            else
                'colormap not understood'
            end
        else
            data = squeeze(rawdata(:,dim));
        end

        if strcmp(overlay_alpha_datapath, '')
            'No overlay alpha map'
            overlay_alpha = ones(n_vertices,1);
        else
            'Loading overlay alpha map'
            overlay_alpha = readNPY(overlay_alpha_datapath);
        end

        if normalize_overlay
            overlay_alpha = overlay_alpha ./ max(overlay_alpha);
        end

        if contains(lower(figname), 'visroi')
            if strcmp(dim, 'clusters')
                overlay_alpha(data == 0) = 0.;
            else
                overlay_alpha(rawdata(:, 2) == 0) = 0.;
            end
        end

        if strcmp(mask_path, '')
            'No mask'
            overlay_alpha = overlay_alpha;
        else
            'Loading mask'
            load(mask_path, 'sig_mask');
            mask = squeeze(sig_mask);
            overlay_alpha(mask==0) = 0;
            % data(mask==0) = 0;
        end

        extraopts = {'rgbnan', 1, 'hemibordercolor', [1 1 1], 'overlayalpha', overlay_alpha};

        for v = 1:length(viewz_to_plot)
            this_view = viewz_to_plot{v};
            if OVERWRITE | ~exist(fullfile(figpath, strcat(figname, 'PCAdim', num2str(dim), 'view', num2str(this_view), '.', SAVE_TYPE)))
                [rawimg, unused, rgbimg] = cvnlookup('fsaverage', this_view, data, [], colormap,[],Lookup,wantfig,extraopts);
                if strcmp(colormap, 'huth3d')
                    title('PC1=red, PC2=green, PC3=blue, PC1+2=yellow, PC2+3=turquoise, PC1+3 = pink, PC1+2+3 = white')
                end
                saveas(gcf, fullfile(figpath, strcat(figname, 'PCAdim', num2str(dim), 'view', num2str(this_view))), SAVE_TYPE)
            end
            close all;
        end
    end
end

close all; clear all
