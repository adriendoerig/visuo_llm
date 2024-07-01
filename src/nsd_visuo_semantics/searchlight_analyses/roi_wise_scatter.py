'''Make a scatter plot of the correlation between two models and the brain, 
with each ROI having a different color'''

import os
import numpy as np
from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# model_names = ['dnn_mpnet_rec_ep200_avgSeed_SL', 'dnn_multihot_rec_ep200_avgSeed_SL']
model_names = ['dnn_mpnet_rec_ep200_avgSeed_SL', 'mpnet_SL']
# model_names = ['mpnet_encoding_model', 'betas_test_noise_ceil']
dnn_layer = 60

subsample = 1  # only plot this fraction of voxels (for visibility)

shuffle_order = True  # if true, shuffle order of voxels before plotting (useful so that it is not always the same ROI on top of the plot)

save_type = 'svg'

nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
searchlights_save_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/searchlight_respectedsampling_correlation_newTest'
encoding_model_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/fitted_models'
betas_test_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/special515_betas'

which_rois = 'streams'
rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')
maskdata, ROIS = get_rois(which_rois, rois_dir)
ROIS[1] = "EVC" # replace the 'early' ROI name with "EVC"
ROIdata = [ROIS[int(k)] for k in maskdata]

n_subjects = 8
subs = [f'subj0{i}' for i in range(1, n_subjects + 1)]
n_voxels = 327684

roi_keys = [
        "earlyROI",
        "midventralROI",
        "ventralROI",
        "midlateralROI",
        "lateralROI",
        "midparietalROI",
        "parietalROI",
    ]
roi_labels = [
    "EVC",
    "midventral",
    "ventral",
    "midlateral",
    "lateral",
    "midparietal",
    "parietal",
]
roi_colors = [
    "mediumaquamarine",
    "khaki",
    "yellow",
    "lightskyblue",
    "royalblue",
    "lightcoral",
    "red",
]
# i flipped the key and label. 
roi_specs = {
    l: {"key": k, "color": c} 
    for k, l, c in zip(roi_keys, roi_labels, roi_colors)
}

# gather data, averaging across subjects, and, if needed, across DNN seeds
final_fsaverage_voxels = {}
for model_name in model_names:
    
    if 'mpnet_encoding_model' in model_name:
        print('Loading MPnet encoding model')
        subjwise_fsavg_voxels = np.empty((n_subjects, n_voxels))
        for s, subj in enumerate(subs):
            print(f'Loading encoding model for {subj}')
            subjwise_fsavg_voxels[s] = np.load(f"{encoding_model_dir}/{subj}_fittedFracridgeEncodingCorrMap_fullbrain.npy")
        final_fsaverage_voxels[model_name] = np.mean(subjwise_fsavg_voxels, axis=0)
    
    elif 'betas_test_noise_ceil' in model_name:
        print('Loading betas test noise ceiling')
        final_fsaverage_voxels[model_name] = np.load(f"{betas_test_dir}/noise_ceiling.npy")
    
    elif 'SL' in model_name:
        print('Loading searchlight model')
        subjwise_fsavg_voxels = np.empty((n_subjects, n_voxels))

        for s, subj in enumerate(subs):

            if 'dnn' in model_name:
                map_id = dnn_layer
            else:
                map_id = 1
            
            if 'avgSeed' in model_name:
                print(f'Loading average seed model for {subj}')
                avg_seed_fsavg_voxels = np.zeros((10, n_voxels))
                seedless_model_name = model_name.split('_avgSeed')[0]
                
                for seed in range(10):
                    seed_id = seed + 1  # 1-indexing for this seed ID
                    split_name = seedless_model_name.split('_ep200')[0]
                    seed_model_name = split_name + f'_seed{seed_id}' + '_ep200'
                    seed_model_folder = f"{searchlights_save_dir}/{subj}/{seed_model_name}/{seed_model_name}_correlation_fsaverage"
                    lh = np.load(f"{seed_model_folder}/lh.{subj}-model-{map_id}-surf.npy")
                    rh = np.load(f"{seed_model_folder}/rh.{subj}-model-{map_id}-surf.npy")
                    avg_seed_fsavg_voxels[seed] = np.concatenate((lh, rh))
                
                subjwise_fsavg_voxels[s] = avg_seed_fsavg_voxels.mean(axis=0)

            else:
                print(f'Loading single seed model for {subj}')
                name = model_name.split('_SL')[0]
                model_folder = f"{searchlights_save_dir}/{subj}/{name}/{name}_correlation_fsaverage"
                lh = np.load(f"{model_folder}/lh.{subj}-model-{map_id}-surf.npy")
                rh = np.load(f"{model_folder}/rh.{subj}-model-{map_id}-surf.npy")
                subjwise_fsavg_voxels[s] = np.concatenate((lh, rh))
                
        
        final_fsaverage_voxels[model_name] = subjwise_fsavg_voxels.mean(axis=0)

    else:
        raise NotImplementedError('model name not understood')

roi_colors_data = [roi_specs[k]["color"] if k != 'Unknown' else "lightgray" for k in ROIdata]
labels_data = [roi_specs[k]["key"] if k != 'Unknown' else "Non-visual" for k in ROIdata]

roi_colors_leg = roi_colors.copy()
roi_colors_leg.insert(0, 'lightgray')
roi_labels_leg = roi_labels.copy()
roi_labels_leg.insert(0, 'Non-visual')
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(roi_colors_leg, roi_labels_leg)]

if subsample < 1:
    n_voxels = int(subsample * n_voxels)
    keep = np.random.choice(n_voxels, n_voxels, replace=False)
    for model_name in model_names:
        final_fsaverage_voxels[model_name] = final_fsaverage_voxels[model_name][keep]
    roi_colors_data = [roi_colors_data[k] for k in keep]

if shuffle_order:
    order = np.random.permutation(n_voxels)
    for model_name in model_names:
        final_fsaverage_voxels[model_name] = final_fsaverage_voxels[model_name][order]
    roi_colors_data = [roi_colors_data[k] for k in order]

ax_lim = 0.3 if 'SL' in model_name else 0.8

plt.figure(figsize=(6, 6))
plt.scatter(final_fsaverage_voxels[model_names[0]], final_fsaverage_voxels[model_names[1]], c=roi_colors_data, alpha=0.5, s=0.5)
plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha = 0.8)
plt.xlim([0, ax_lim])
plt.ylim([0, ax_lim])
plt.legend(handles=patches)
if model_names[0] == 'dnn_mpnet_rec_ep200_avgSeed_SL':
    plt.xlabel('RCNN(LLM) to brain correlation\n[pearson, voxel-wise]')
elif model_names[0] == 'mpnet_encoding_model':
    plt.xlabel('encoding model to brain correlation\n[pearson, voxel-wise]')
else:
    plt.xlabel(f'{model_names[0]} to brain correlation\n[pearson, voxel-wise]')
if model_names[1] == 'dnn_multihot_rec_ep200_avgSeed_SL':
    plt.ylabel('RCNN(Category) to brain correlation\n[pearson, voxel-wise]')
elif model_names[1] == 'mpnet_SL':
    plt.ylabel('LLM to brain correlation\n[pearson, voxel-wise]')
elif model_names[1] == 'betas_test_noise_ceil':
    plt.ylabel('Noise ceiling\n[pearson, voxel-wise]')
else:
    plt.ylabel(f'{model_names[1]} to brain correlation\n[pearson, voxel-wise]')
plt.savefig(f"scatter_{model_names[0]}_{model_names[1]}.{save_type}")
plt.close()

# accidental save in: {searchlights_save_dir}/
