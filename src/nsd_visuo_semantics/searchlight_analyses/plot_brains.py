import cortex, os
import matplotlib.pyplot as plt
import numpy as np

def bihem_layer_id(MODEL_NAME, LAYER_NAME):
    
    if MODEL_NAME == 'dnn_bihem_bottleneckCCApr24':
        dnn_bihem_bottleneckCCApr24_layersNames = ['gap_left', 'gap_left_ln', 'gap_right', 'gap_right_ln', 'it_left', 'it_left_ln', 'it_right', 'it_right_ln', 'left_postV1_to_right_preV2_wm', 'left_postV2_to_right_preV4_wm', 'left_postV4_to_right_preIT_wm', 'post_gap_left_relu', 'post_gap_right_relu', 'post_it_left_relu', 'post_it_right_relu', 'post_retina_left_relu', 'post_retina_right_relu', 'post_v1_left_relu', 'post_v1_right_relu', 'post_v2_left_relu', 'post_v2_right_relu', 'post_v4_left_relu', 'post_v4_right_relu', 'pre_concat_bottleneck_left', 'pre_concat_bottleneck_right', 'readout', 'retina_left', 'retina_right', 'right_postV1_to_left_preV2_wm', 'right_postV2_to_left_preV4_wm', 'right_postV4_to_left_preIT_wm', 'v1_left', 'v1_left_ln', 'v1_right', 'v1_right_ln', 'v2_left', 'v2_left_ln', 'v2_right', 'v2_right_ln', 'v4_left', 'v4_left_ln', 'v4_right', 'v4_right_ln']
        return dnn_bihem_bottleneckCCApr24_layersNames.index(LAYER_NAME) + 1

    elif MODEL_NAME == 'dnn_bihem_bottleneckNoCCApr24':
        dnn_bihem_bottleneckNoCCApr24_layersNames = ['gap_left', 'gap_left_ln', 'gap_right', 'gap_right_ln', 'it_left', 'it_left_ln', 'it_right', 'it_right_ln', 'post_gap_left_relu', 'post_gap_right_relu', 'post_it_left_relu', 'post_it_right_relu', 'post_retina_left_relu', 'post_retina_right_relu', 'post_v1_left_relu', 'post_v1_right_relu', 'post_v2_left_relu', 'post_v2_right_relu', 'post_v4_left_relu', 'post_v4_right_relu', 'pre_concat_bottleneck_left', 'pre_concat_bottleneck_right', 'readout', 'retina_left', 'retina_right', 'v1_left', 'v1_left_ln', 'v1_right', 'v1_right_ln', 'v2_left', 'v2_left_ln', 'v2_right', 'v2_right_ln', 'v4_left', 'v4_left_ln', 'v4_right', 'v4_right_ln']
        return dnn_bihem_bottleneckNoCCApr24_layersNames.index(LAYER_NAME) + 1

    elif MODEL_NAME == 'dnn_bihem_singleStreamApr24':
        dnn_bihem_singleStreamApr24_layersNames = ['bottleneck', 'gap', 'gap_ln', 'it', 'it_ln', 'post_gap_relu', 'post_it_relu', 'post_retina_relu', 'post_v1_relu', 'post_v2_relu', 'post_v4_relu', 'readout', 'retina', 'v1', 'v1_ln', 'v2', 'v2_ln', 'v4', 'v4_ln']
        return dnn_bihem_singleStreamApr24_layersNames.index(LAYER_NAME) + 1
    
    else:
        raise ValueError('MODEL_NAME not recognized')    


def plot_brains_bihem(MODEL_NAME, CONTRAST_MODEL_NAME, LAYER_NAME, CONTRAST_LAYER_NAME, MODEL_SUFFIX,
                      SEARCHLIGHT_SAVE_DIR, RESULTS_DIR):

    USE_FDR = 1

    # some parameters
    n_subjects = 8
    n_vertices = 327684
    hemis = ['lh', 'rh']

    layer_id = bihem_layer_id(MODEL_NAME, LAYER_NAME)

    # load the data
    datapath = os.path.join(SEARCHLIGHT_SAVE_DIR, '%s', MODEL_NAME, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.npy')

    main_data = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    # loop over subjects
    for sub in range(n_subjects):
        subj = f'subj{(1+sub):02d}'
        sub_data = []
        for hemi in range(2):
            this_hemi = hemis[hemi]
            sub_data.append(np.load(datapath % (subj, MODEL_NAME + MODEL_SUFFIX, this_hemi, subj, str(layer_id))))
        main_data[sub, :] = np.concatenate(sub_data)

    main_data_avg = np.mean(main_data, axis=0)

    # where to save
    figpath  = os.path.join(RESULTS_DIR, MODEL_NAME)
    os.makedirs(figpath, exist_ok=True)

    vert = cortex.dataset.Vertex(main_data_avg, "fsaverage", cmap='Reds')
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    plt.title(f'{MODEL_NAME}_{LAYER_NAME}_avg.png')
    plt.savefig(f'{figpath}/{MODEL_NAME}_{LAYER_NAME}_avg.png', dpi=400)
    plt.close()

    
    ### CONTRAST MODEL

    layer_id = bihem_layer_id(CONTRAST_MODEL_NAME, CONTRAST_LAYER_NAME)

    # load the data
    contrast_datapath = os.path.join(SEARCHLIGHT_SAVE_DIR, '%s', MODEL_NAME, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.npy')

    contrast_main_data = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    # loop over subjects
    for sub in range(n_subjects):
        subj = f'subj{(1+sub):02d}'
        contrast_sub_data = []
        for hemi in range(2):
            this_hemi = hemis[hemi]
            contrast_sub_data.append(np.load(contrast_datapath % (subj, MODEL_NAME + MODEL_SUFFIX, this_hemi, subj, str(layer_id))))
        contrast_main_data[sub, :] = np.concatenate(contrast_sub_data)

    contrast_main_data_avg = np.mean(contrast_main_data, axis=0)

    # where to save
    figpath  = os.path.join(RESULTS_DIR, MODEL_NAME)
    os.makedirs(figpath, exist_ok=True)

    vert = cortex.dataset.Vertex(contrast_main_data_avg, "fsaverage", cmap='Reds')
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    plt.title(f'{CONTRAST_MODEL_NAME}_{CONTRAST_LAYER_NAME}_avg.png')
    plt.savefig(f'{figpath}/{CONTRAST_MODEL_NAME}_{CONTRAST_LAYER_NAME}_avg.png', dpi=400)
    plt.close()

    # plot the difference
    diff = main_data_avg - contrast_main_data_avg
    boundar = np.max(np.abs(diff))
    vert = cortex.dataset.Vertex(diff, "fsaverage", cmap='RdBu_r', vmin=-boundar, vmax=boundar)
    flatmap = cortex.quickflat.make_figure(vert, height=480, with_colorbar=1, with_rois=False)
    plt.title(f'{MODEL_NAME}_{LAYER_NAME}_avg - {CONTRAST_MODEL_NAME}_{CONTRAST_LAYER_NAME}_avg.png')
    plt.savefig(f'{figpath}/{MODEL_NAME}_{LAYER_NAME}_avg - {CONTRAST_MODEL_NAME}_{CONTRAST_LAYER_NAME}_avg.png', dpi=400)
    plt.close()
    
    assert np.sum(diff) > 0

if __name__ == '__main__':
    MODEL_NAME = 'dnn_bihem_bottleneckCCApr24'
    CONTRAST_MODEL_NAME = 'dnn_bihem_singleStreamApr24' # 'dnn_bihem_bottleneckNoCCApr24'
    LAYER_NAME = 'pre_concat_bottleneck_right'  # 'v1_left_ln'  # 'pre_concat_bottleneck_left'
    CONTRAST_LAYER_NAME = 'bottleneck'  # 'v1_right_ln'  # 'pre_concat_bottleneck_right'
    MODEL_SUFFIX = ''
    SEARCHLIGHT_SAVE_DIR = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/searchlight_respectedsampling_correlation_newTest'
    RESULTS_DIR = '/share/klab/adoerig/adoerig/bihem_results_dir'
    plot_brains_bihem(MODEL_NAME, CONTRAST_MODEL_NAME, LAYER_NAME, CONTRAST_LAYER_NAME, MODEL_SUFFIX, SEARCHLIGHT_SAVE_DIR, RESULTS_DIR)
    print('done')
