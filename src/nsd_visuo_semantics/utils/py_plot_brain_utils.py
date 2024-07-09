import cortex, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def pyplot_brain(fsavg_data, savename, figpath, save_type='png', max_cmap_val=None):

    os.makedirs(figpath, exist_ok=True)

    if max_cmap_val is None:
        boundar = np.nanmax(np.abs(fsavg_data))
    else:
        boundar = np.nanmax(np.abs(max_cmap_val))

    vert = cortex.dataset.Vertex(fsavg_data, "fsaverage", cmap='RdBu_r', vmin=-boundar, vmax=boundar)    
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    
    fig = plt.gcf()
    fig.suptitle(f'{savename} - max abs val: {np.nanmax(np.abs(fsavg_data)):.2f}')
    plt.savefig(f'{figpath}/{savename}.{save_type}', dpi=300)
    plt.close()


def pyplot_indiv_subjects(fsavg_data_allsub, savename, figpath, save_type='png', max_cmap_val=None):
    '''
    fsavg_data_allsub: [n_subj, n_fsavg_voxels]'''
    
    for s in range(fsavg_data_allsub.shape[0]):
        fsavg_data = fsavg_data_allsub[s]
        pyplot_brain(fsavg_data, f'{savename}_subj{s+1:02}', figpath, save_type=save_type, max_cmap_val=max_cmap_val)


def pyplot_subj_avg(fsavg_data_allsub, savename, figpath, sig_mask='fdr_bh', save_type='png', max_cmap_val=None):
    '''
    fsavg_data_allsub: [n_subj, n_fsavg_voxels]
    sig_mask: "fdr_bh", "uncorrected" or None (no mask is applied if None)'''
    
    sig_str = '' if sig_mask is None else sig_mask
    
    # get mean
    fsavg_data = np.nanmean(fsavg_data_allsub, axis=0)
    
    # 2-tailed t-test against 0 for each element in fsavg_data
    if sig_mask != None:
        t_stat, p_values = ttest_1samp(fsavg_data_allsub, 0, axis=0, nan_policy='omit')
    
    # Apply significance mask if specified
    if sig_mask == 'fdr_bh':
        # Apply FDR correction
        _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        sig_mask = p_values_corrected < 0.05
    elif sig_mask == 'uncorrected':
        sig_mask = p_values < 0.05
    else:
        sig_mask = np.ones_like(fsavg_data, dtype=bool)  # No masking, all true
    
    # Apply significance mask to data
    fsavg_data_masked = np.where(sig_mask, fsavg_data, np.nan)  # Mask non-significant points with NaN

    pyplot_brain(fsavg_data_masked, f'{savename}_subjavg_{sig_str}', figpath, save_type=save_type, max_cmap_val=max_cmap_val)


def get_fsavg_data_from_path(base_dir, model_name, layer_id, model_suffix='', 
                             n_subjects=8, n_vertices=327684, hemis=['lh', 'rh']):
    '''
    NOTE: we will format the datapath string using the arguments passed here. make sure
    they correspond to the way things are saved in the base_dir.
    RETURNS: an array of shape [n_subjects, n_vertices] where each row is the flattened data for a subject
    '''

    datapath = os.path.join(base_dir, '%s', model_name, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.npy')

    main_data = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    # loop over subjects
    for sub in range(n_subjects):
        subj = f'subj{(1+sub):02d}'
        sub_data = []
        for this_hemi in hemis:
            if 'dnn' in model_name:
                sub_data.append(np.load(datapath % (subj, model_name + model_suffix, this_hemi, subj, str(layer_id))))
            else:
                sub_data.append(np.load(datapath % (subj, model_name + model_suffix, this_hemi, subj, str(1))))
        main_data[sub, :] = np.concatenate(sub_data)

    return main_data


def pyplot_brains_from_models_list(models, contrast_models, base_dir, 
                                   layer=60, contrast_layer='same', 
                                   contrast_same_model=True, max_cmap_val=None,
                                   save_type='png', figpath='', 
                                   plot_indiv_sub=True, plot_subj_avg=True):
    '''models: list of model names to plot
    contrast_models: list of model names to use as contrast
    base_dir: base directory where the data is saved
    layer: layer to plot (only used if "dnn" in model_name)
    contrast_layer: layer to use for contrast (only used if "dnn" in model_name). can be "same" or a layer number
    contrast_same_model: if True, do not plot the same model as contrast
    max_cmap_val: maximum value for the colorbar
    save_type: file type to save the figures
    figpath: path to save the figures
    plot_indiv_sub: if True, plot individual subject maps
    plot_subj_avg: if True, plot the average map across subjects (with sig threshold)'''

    if not isinstance(models, list):
        models = [models]
    if not isinstance(contrast_models, list):
        contrast_models = [contrast_models]

    if contrast_layer == 'same':
        contrast_layer = layer
    elif isinstance(contrast_layer, int):
        pass
    else:
        raise NotImplementedError('Only "same" is implemented for contrast_layer')

    for model_name in models:

        fsavg_data = get_fsavg_data_from_path(base_dir, model_name, layer)
        if plot_indiv_sub:
            pyplot_indiv_subjects(fsavg_data, model_name, figpath, save_type=save_type, max_cmap_val=max_cmap_val)
        if plot_subj_avg:
            pyplot_subj_avg(fsavg_data, model_name, figpath, sig_mask='fdr_bh', save_type=save_type, max_cmap_val=max_cmap_val)

        for contrast_model_name in contrast_models:

            if not contrast_same_model and contrast_model_name == model_name:
                continue

            fsavg_data_contrast = get_fsavg_data_from_path(base_dir, contrast_model_name, contrast_layer)
            
            fsavg_data_diff = fsavg_data - fsavg_data_contrast
            if plot_indiv_sub:
                pyplot_indiv_subjects(fsavg_data_diff, model_name, figpath, save_type=save_type, max_cmap_val=max_cmap_val)
            if plot_subj_avg:
                pyplot_subj_avg(fsavg_data_diff, model_name, figpath, sig_mask='fdr_bh', save_type=save_type, max_cmap_val=max_cmap_val)
