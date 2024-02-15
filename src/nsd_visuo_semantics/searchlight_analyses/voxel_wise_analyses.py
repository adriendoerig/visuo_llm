import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois
from nsd_visuo_semantics.decoding_analyses.decoding_utils import restore_nan_dims, get_gcc_nearest_neighbour
from nsd_visuo_semantics.utils.utils import load_mat_save_npy


if __name__ == "__main__":

    # analysis_type determines the type of analysis to be performed.
    # 'tsne2d' or 'tsne3d', 'pca3D'
    # if 'individualrois' is in the analysis_type, the data will be plotted for each ROI separately.
    # if 'xval' is in the analysis_type, the PCA will be trained on an ecoding model trained on one half of the nsd data, and we plot the variance explained on the weights of another encoding model trained on the other half of the nsd data.
    # if 'train' is in the analysis_type, the PCA model will be trained on the whole data and then used to project the data for each ROI.
    # if 'visrois' is in the analysis_type, the data will be plotted for the visual ROIs only.
    # if 'fullbrain' is in the analysis_type, the data will be plotted for the whole brain.
    # if 'sigonly' is in the analysis_type, the data will be plotted for the significant voxels only.
    # if 'makeclustermap' is in the analysis_type, load cluster assignments for each voxel, and save the full voxel map.
    # e.g. 'pca3D_VisROIsTrain_individualROIsTest' will train a PCA model on all visual ROIs and then project the data for each ROI separately.
    for analysis_type in ['pca3D_VisROIs_xval']:  #['makeclustermap_visrois']:#, 'makeclustermap_sigonly', 'makeclustermap_fullbrain']:  # ['tsne2d', 'tsne3d', 'pca3D', 'pca3DTrain', 'pca3D_VisROIsTrain_individualROIsTest', 'pca3D_VisROIsTrain', 'pca3D_VisROIs', 'pca3D_individualROIs', 'pca3D_individualROIsTrain
    # for analysis_type in ['pca3D_VisROIs']:  # ['tsne2d', 'tsne3d', 'pca3D', 'pca3DTrain', 'pca3D_VisROIsTrain_individualROIsTest', 'pca3D_VisROIsTrain', 'pca3D_VisROIs', 'pca3D_individualROIs', 'pca3D_individualROIsTrain

        cluster_n = 5 # 2, 7, 12 or 17  -- only used if 'makeclustermap' in analysis_type

        save_type = 'svg'  # 'png' or 'svg'

        massage_data = 'zscore'  # "zscore" or None

        OVERWRITE = False

        n_subjects = 8
        subs = [f"subj0{x+1}" for x in range(n_subjects)]
        rdm_distance = 'correlation'
        remove_shared_515 = False
        which_rois = 'streams'

        roi_colors = {
            "early": "mediumaquamarine",
            "midventral": "khaki",
            "ventral": "yellow",
            "midlateral": "lightskyblue",
            "lateral": "royalblue",
            "midparietal": "lightcoral",
            "parietal": "red"
            }
            

        # set up directories
        base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir"
        models_dir = os.path.join(base_save_dir,f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}')

        roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
        os.makedirs(roi_analyses_dir, exist_ok=True)
        subj_roi_rdms_path = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}", "subj_roi_rdms")
        os.makedirs(subj_roi_rdms_path, exist_ok=True)
        
        results_dir = os.path.join(base_save_dir, f"voxel_wise_analyses")
        os.makedirs(results_dir, exist_ok=True)
        cache_dir = os.path.join(results_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
        rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')

        encoding_base_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses'

        # we use the fsaverage space.
        targetspace = "fsaverage"

        # Get roi info
        maskdata, ROIS = get_rois(which_rois, rois_dir)

        for data_type in ['encodingModelCoeffs']:  # ['encodingModelCoeffs_mpnetDnnLayer0', 'encodingModelCoeffs', 'var_part_uniquevars', 'var_part_scores', 'clusterAssignments']:

            varThreshVal = 0.7  # if we are doing PCA analyses, we will plot n_components required to explain this fraction of variance
            cumvars = {}
            eigenspectrums = {}
            varThresholds = {}
            if 'individualrois' in analysis_type.lower():
                # this allows to plot the data for each ROI separately
                rois_to_use = [1,2,3,4,5,6,7]
            else:
                rois_to_use = ['all']

            if not 'xval' in analysis_type.lower():
                fig, ax = plt.subplots(2)
            else:
                fig, ax = plt.subplots(2, 2, figsize=(12, 12))

            for roi in rois_to_use:

                data_in_avg = []
                data_in_avg1 = []
                data_in_avg2 = []
                filtered_data_in_avg = []
                filtered_data_in_avg1 = []
                filtered_data_in_avg2 = []

                for subj in subs+['average']:

                    print(f"Processing {subj}, roi {roi}, for {data_type} and {analysis_type} analysis...")

                    if subj != 'average':

                        if data_type == 'encodingModelCoeffs' and not 'xval' in analysis_type.lower():
                            this_encoding_base_dir = os.path.join(encoding_base_dir, 'all-mpnet-base-v2_results_ROIfullbrain_encodingModel')
                            encoding_model_dir = os.path.join(this_encoding_base_dir, 'fitted_models')
                            encoding_model_sig_mask_path = os.path.join(this_encoding_base_dir, "encoding_sig_mask_mpnet_encodingModel.npy")
                            data_in = np.load(encoding_model_dir + f"/{subj}_fittedFracridgeEncodingCoefs_fullbrain.npy", allow_pickle=True)
                            data_in = data_in.T

                        elif data_type == 'encodingModelCoeffs' and 'xval' in analysis_type.lower():
                            this_encoding_base_dir1 = os.path.join(encoding_base_dir, 'all-mpnet-base-v2_results_ROIfullbrain_encodingModel_split0')
                            encoding_model_dir1 = os.path.join(this_encoding_base_dir1, 'fitted_models')
                            data_in1 = np.load(encoding_model_dir1 + f"/{subj}_fittedFracridgeEncodingCoefs_fullbrain_all-mpnet-base-v2.npy", allow_pickle=True)
                            data_in1 = data_in1.T

                            this_encoding_base_dir2 = os.path.join(encoding_base_dir, 'all-mpnet-base-v2_results_ROIfullbrain_encodingModel_split1')
                            encoding_model_dir2 = os.path.join(this_encoding_base_dir2, 'fitted_models')
                            data_in2 = np.load(encoding_model_dir2 + f"/{subj}_fittedFracridgeEncodingCoefs_fullbrain_all-mpnet-base-v2.npy", allow_pickle=True)
                            data_in2 = data_in2.T

                            encoding_model_sig_mask_path = os.path.join(this_encoding_base_dir1, "encoding_sig_mask_mpnet_encodingModel.npy")

                        elif data_type == 'encodingModelCoeffs_mpnetDnnLayer0':
                            this_encoding_base_dir = os.path.join(encoding_base_dir, 'mpnet_rec_seed1_nsd_activations_epoch200_layer0_results_ROIfullbrain_encodingModel')
                            encoding_model_dir = os.path.join(this_encoding_base_dir, 'fitted_models')
                            encoding_model_sig_mask_path = os.path.join(this_encoding_base_dir, "encoding_sig_mask_mpnet_dnn_layer0_encodingModel.npy")
                            data_in = np.load(encoding_model_dir + f"/{subj}_fittedFracridgeEncodingCoefs_fullbrain_mpnet_rec_seed1_nsd_activations_epoch200.npy", allow_pickle=True)
                            data_in = data_in.T

                        elif data_type == 'encodingModelCoeffs_mpnetDnnLayer-1':
                            this_encoding_base_dir = os.path.join(encoding_base_dir, 'mpnet_rec_seed1_nsd_activations_epoch200_layer-1_results_ROIfullbrain_encodingModel')
                            encoding_model_dir = os.path.join(this_encoding_base_dir, 'fitted_models')
                            encoding_model_sig_mask_path = os.path.join(this_encoding_base_dir, "encoding_sig_mask_mpnet_dnn_layer-1_encodingModel.npy")
                            data_in = np.load(encoding_model_dir + f"/{subj}_fittedFracridgeEncodingCoefs_fullbrain_mpnet_rec_seed1_nsd_activations_epoch200.npy", allow_pickle=True)
                            data_in = data_in.T

                        elif data_type == 'var_part_uniquevars' or data_type == 'var_part_scores':

                            data_dir = os.path.join(base_save_dir,
                                f"searchlight_respectedsampling_{rdm_distance}_newTest",  "{}",
                                "var_partition_['mpnet', 'mpnet_nouns', 'mpnet_verbs']")

                            with open(os.path.join(data_dir.format(subj), "model_combinations.pkl"), "rb") as f:
                                model_combinations = pickle.load(f)

                            brain_map_data = []
                            data_name = data_type.replace('var_part_', '')
                            for i, mc in enumerate(model_combinations):
                                brain_map_data_lh = np.load(os.path.join(data_dir.format(subj), 'fsaverage', f'lh.{subj}-{data_name}-{mc}-surf.npy'), allow_pickle=True)
                                brain_map_data_rh = np.load(os.path.join(data_dir.format(subj), 'fsaverage', f'rh.{subj}-{data_name}-{mc}-surf.npy'), allow_pickle=True)
                                brain_map_data.append(np.concatenate([brain_map_data_lh, brain_map_data_rh], axis=0))
                            data_in = np.stack(brain_map_data, axis=-1)

                        else:
                            raise Exception("Data type not recognised.")
                                    
                        # keep only relevant voxels
                        if not 'xval' in analysis_type.lower():
                            if 'fullbrain' in analysis_type.lower():
                                filtered_data_in = data_in
                                filtered_labels = maskdata
                            elif 'sigonly' in analysis_type.lower():
                                if not os.path.exists(encoding_model_sig_mask_path) and os.path.exists(encoding_model_sig_mask_path.replace(".npy", ".mat")):
                                    load_mat_save_npy(encoding_model_sig_mask_path.replace(".npy", ".mat"), 'sig_mask')
                                sig_mask = np.load(encoding_model_sig_mask_path).squeeze()
                                filtered_data_in = data_in[sig_mask !=0, :]
                                filtered_labels = maskdata[sig_mask !=0]
                            elif 'individualrois' in analysis_type.lower():
                                filtered_data_in = data_in[maskdata == roi, :]
                                filtered_labels = maskdata[maskdata == roi]
                            elif 'visrois' in analysis_type.lower():
                                filtered_data_in = data_in[maskdata != 0, :]
                                filtered_labels = maskdata[maskdata != 0]
                            elif 'makeclustermap' in analysis_type.lower():
                                continue
                            else:
                                raise Exception("Analysis type not recognised.")
                            filtered_data_in_avg.append(filtered_data_in)
                            data_in_avg.append(data_in)

                        else:
                            if 'fullbrain' in analysis_type.lower():
                                filtered_data_in1 = data_in1
                                filtered_data_in2 = data_in2
                                filtered_labels = maskdata
                            elif 'sigonly' in analysis_type.lower():
                                if not os.path.exists(encoding_model_sig_mask_path) and os.path.exists(encoding_model_sig_mask_path.replace(".npy", ".mat")):
                                    load_mat_save_npy(encoding_model_sig_mask_path.replace(".npy", ".mat"), 'sig_mask')
                                sig_mask = np.load(encoding_model_sig_mask_path).squeeze()
                                filtered_data_in1 = data_in1[sig_mask !=0, :]
                                filtered_data_in2 = data_in2[sig_mask !=0, :]
                                filtered_labels = maskdata[sig_mask !=0]
                            elif 'individualrois' in analysis_type.lower():
                                filtered_data_in1 = data_in1[maskdata == roi, :]
                                filtered_data_in2 = data_in2[maskdata == roi, :]
                                filtered_labels = maskdata[maskdata == roi]
                            elif 'visrois' in analysis_type.lower():
                                filtered_data_in1 = data_in1[maskdata != 0, :]
                                filtered_data_in2 = data_in2[maskdata != 0, :]
                                filtered_labels = maskdata[maskdata != 0]
                            else:
                                raise Exception("Analysis type not recognised.")
                            filtered_data_in_avg1.append(filtered_data_in1)
                            data_in_avg1.append(data_in1)
                            filtered_data_in_avg2.append(filtered_data_in2)
                            data_in_avg2.append(data_in2)
                            
                    
                    else:

                        if 'makeclustermap' in analysis_type.lower():
                            import pandas as pd
                            if 'visrois' in analysis_type.lower():
                                f = 'visROIs'
                            elif 'fullbrain' in analysis_type.lower():
                                f = 'fullBrain'
                            elif 'sigonly' in analysis_type.lower():
                                f = 'sigOnly'
                            # cluster_assignments = pd.read_csv(f'{cache_dir}/semantic_clusters/cluster_labels_all_{f}_2_22_5{"_zscored" if massage_data == "zscore" else ""}.csv')
                            cluster_assignments = pd.read_csv(f'{cache_dir}/semantic_clusters/cluster_labels_all_{f}_4_12_1{"_zscored" if massage_data == "zscore" else ""}.csv')
                            cluster_vector = np.array(cluster_assignments[str(cluster_n)])+1  # 0 is kept for no cluster (e.g. for non-significant voxels)

                        else:

                            if not 'xval' in analysis_type.lower():
                                data_in = np.nanmean(data_in_avg, axis=0)
                                filtered_data_in = np.nanmean(filtered_data_in_avg, axis=0)
                            else:
                                data_in1 = np.nanmean(data_in_avg1, axis=0)
                                filtered_data_in1 = np.nanmean(filtered_data_in_avg1, axis=0)
                                data_in2 = np.nanmean(data_in_avg2, axis=0)
                                filtered_data_in2 = np.nanmean(filtered_data_in_avg2, axis=0)
    

                    if 'makeclustermap' in analysis_type.lower() and subj == 'average':
                        if 'visrois' in analysis_type.lower():
                            f = 'visROIs'
                        elif 'fullbrain' in analysis_type.lower():
                            f = 'fullBrain'
                        elif 'sigonly' in analysis_type.lower():
                            f = 'sigOnly'
                        d = 'encodingModelCoeffs'
                        # data_in = np.load(f'{cache_dir}/{d}_{f}_data_in_subjavg.npy')
                        maskdata = np.load(f'{cache_dir}/{d}_{f}_maskdata.npy')
                        # filtered_data_in = np.load(f'{cache_dir}/{d}_{f}_filtered_data_in_subjavg.npy')
                        # filtered_labels = np.load(f'{cache_dir}/{d}_{f}_filtered_maskdata.npy')
                        if not 'fullbrain' in analysis_type.lower():
                            full_brain_map = np.zeros((maskdata.shape[0],))
                            if 'sigonly' in analysis_type.lower():
                                full_brain_map[sig_mask != 0] = cluster_vector
                            elif 'individualrois' in analysis_type.lower():
                                full_brain_map[maskdata == roi] = cluster_vector
                            elif 'visrois' in analysis_type.lower():
                                full_brain_map[maskdata != 0] = cluster_vector
                            else:
                                raise Exception("Analysis type not recognised.")
                        else:
                            full_brain_map = cluster_vector
                        print('saving in', f'{cache_dir}/{d}_{f}_cluster_assignments_subjavg_nclusters{cluster_n}{"_zscored" if massage_data == "zscore" else ""}.npy')
                        np.save(f'{cache_dir}/{d}_{f}_cluster_assignments_subjavg_nclusters{cluster_n}{"_zscored" if massage_data == "zscore" else ""}.npy', full_brain_map)
                        continue

                    if 'savefiltereddata' in analysis_type.lower() and subj == 'average':
                        np.save(f'{cache_dir}/{data_type}_{analysis_type}_data_in_subjavg.npy', data_in)
                        np.save(f'{cache_dir}/{data_type}_{analysis_type}_maskdata.npy', maskdata)
                        np.save(f'{cache_dir}/{data_type}_{analysis_type}_filtered_data_in_subjavg.npy', filtered_data_in)
                        np.save(f'{cache_dir}/{data_type}_{analysis_type}_filtered_maskdata.npy', filtered_labels)

                    if not 'xval' in analysis_type.lower():
                        if np.any(np.isnan(filtered_data_in)):
                            print(f"WARNING: {np.sum(np.isnan(filtered_data_in))} NaNs found in {subj} {data_type} data ({np.sum(np.isnan(filtered_data_in))/filtered_data_in.size}%). Removing them.")
                            nan_mask = np.isnan(filtered_data_in).any(axis=1)
                            filtered_data_in = filtered_data_in[~nan_mask]
                            filtered_labels = filtered_labels[~nan_mask]
                        else:
                            nan_mask = None

                        if massage_data == 'zscore':
                            mu, sigma = np.mean(filtered_data_in, axis=0), np.std(filtered_data_in, axis=0)
                            filtered_data_in = (filtered_data_in - mu) / sigma
                        else:
                            mu, sigma = np.zeros(filtered_data_in.shape[1]), np.ones(filtered_data_in.shape[1])
                    else:
                        if np.any(np.isnan(filtered_data_in1)):
                            print(f"WARNING: {np.sum(np.isnan(filtered_data_in1))} NaNs found in {subj} {data_type} data ({np.sum(np.isnan(filtered_data_in1))/filtered_data_in1.size}%). Removing them.")
                            nan_mask1 = np.isnan(filtered_data_in1).any(axis=1)
                            filtered_data_in1 = filtered_data_in1[~nan_mask1]
                            filtered_labels1 = filtered_labels[~nan_mask1]
                        else:
                            nan_mask1 = None
                        if np.any(np.isnan(filtered_data_in2)):
                            print(f"WARNING: {np.sum(np.isnan(filtered_data_in2))} NaNs found in {subj} {data_type} data ({np.sum(np.isnan(filtered_data_in2))/filtered_data_in2.size}%). Removing them.")
                            nan_mask2 = np.isnan(filtered_data_in2).any(axis=1)
                            filtered_data_in2 = filtered_data_in2[~nan_mask2]
                            filtered_labels2 = filtered_labels[~nan_mask2]
                        else:
                            nan_mask2 = None

                        if massage_data == 'zscore':
                            mu1, sigma1 = np.mean(filtered_data_in1, axis=0), np.std(filtered_data_in1, axis=0)
                            filtered_data_in1 = (filtered_data_in1 - mu1) / sigma1
                            mu2, sigma2 = np.mean(filtered_data_in2, axis=0), np.std(filtered_data_in2, axis=0)
                            filtered_data_in2 = (filtered_data_in2 - mu2) / sigma2
                        else:
                            mu1, sigma1 = np.zeros(filtered_data_in1.shape[1]), np.ones(filtered_data_in1.shape[1])
                            mu2, sigma2 = np.zeros(filtered_data_in2.shape[1]), np.ones(filtered_data_in2.shape[1])

                        
                    if analysis_type == 'tsne2d':

                        tsne_dim = 2

                        embedding_path = f'{cache_dir}/{data_type}_{analysis_type}_{subj}.npy'
                        if not OVERWRITE and not os.path.exists(embedding_path):
                            tsne = TSNE(n_components=2, random_state=42)
                            embedding = tsne.fit_transform(filtered_data_in)
                            np.save(embedding_path, embedding)
                        else:
                            embedding = np.load(embedding_path)

                        plt.figure(figsize=(12, 12))
                        for label_value, label_string in ROIS.items():  # Skip label 0
                            if label_value == 0:
                                continue
                            else:
                                indices = filtered_labels == label_value
                                plt.scatter(embedding[indices, 0], embedding[indices, 1], color=roi_colors[ROIS[label_value]],
                                            s=50, alpha=0.6, label=label_string)

                        plt.title('2D t-SNE Plot')
                        plt.legend(loc='upper right')
                        plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_{subj}.{save_type}')
                    
                    elif analysis_type == 'tsne3d':

                        tsne_dim = 3

                        import pandas as pd
                        import plotly.express as px

                        embedding_path = f'{cache_dir}/{data_type}_{analysis_type}_{subj}.npy'
                        if not OVERWRITE and not os.path.exists(embedding_path):
                            tsne = TSNE(n_components=3, random_state=42)
                            embedding = tsne.fit_transform(filtered_data_in)
                            np.save(embedding_path, embedding)
                        else:
                            embedding = np.load(embedding_path)

                        df = pd.DataFrame({'X': embedding[:, 0]*10, 'Y': embedding[:, 1]*10, 'Z': embedding[:, 2]*10, 'label': filtered_labels})
                        # Create an interactive 3D scatter plot with Plotly
                        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='label', title='3D t-SNE', size_max=0.1, opacity=0.7, color_continuous_scale=px.colors.qualitative.Plotly)
                        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), scene_bgcolor='white')
                        # fig.update(layout_coloraxis_showscale=False)
                        fig.write_html(f'{results_dir}/{data_type}_{analysis_type}_{subj}.html')


                    elif 'pca3D' in analysis_type and not 'xval' in analysis_type.lower():
                        # reduce dimensionality to 3D using PCA, then plot in RGB colorspace

                        n_components = 768

                        embedding_path = f'{cache_dir}/{data_type}_{analysis_type}_{subj}_{n_components}components{massage_data}.npy'
                        pca_to_load_name = analysis_type if not 'train' in analysis_type.lower() else analysis_type.split('Train')[0]
                        pca_to_load_path = f'{cache_dir}/pca_model_{pca_to_load_name}_{subj}_{n_components}components{massage_data}.pkl'

                        if 'train' in analysis_type.lower() and not os.path.exists(pca_to_load_path):
                            import pdb; pdb.set_trace()
                            raise Exception(f"PCA model not found for {analysis_type} analysis. Please run the analysis without 'train' in the name first.")

                        if OVERWRITE or not os.path.exists(pca_to_load_path):
                            pca = PCA(n_components=n_components)
                            embedding = pca.fit_transform(filtered_data_in)
                            if nan_mask is not None:
                                drop_rows_idx = np.where(nan_mask)[0]
                                embedding = restore_nan_dims(embedding, drop_rows_idx, axis=0)
                                embedding[np.isnan(embedding)] = 0

                            with open(f'{cache_dir}/pca_model_{analysis_type}_{subj}_{n_components}components{massage_data}.pkl', 'wb') as file:
                                pickle.dump(pca, file)

                            if not 'fullbrain' in analysis_type:
                                all_vox_embedding = np.zeros((maskdata.shape[0], n_components))
                                if 'sigOnly' in analysis_type:
                                    all_vox_embedding[sig_mask != 0, :] = embedding
                                elif 'individualrois' in analysis_type:
                                    all_vox_embedding[maskdata == roi, :] = embedding
                                else:
                                    all_vox_embedding[maskdata != 0, :] = embedding
                                embedding = all_vox_embedding

                            np.save(embedding_path, embedding)
                        else:

                            with open(pca_to_load_path, 'rb') as file:
                                pca = pickle.load(file)
                            # embedding = np.load(embedding_path)


                        if subj == 'average':

                            if 'train' in analysis_type.lower():
                                # this projects the roi data onto the pca fitted on the whole data 
                                # (e.g. the visual rois if 'visroi' in analysis_type)
                                roi_projected = pca.transform(filtered_data_in)
                                explained_variance_ratio = np.var(roi_projected, axis=0)/np.var(roi_projected, axis=0).sum()
                            else:
                                explained_variance_ratio = pca.explained_variance_ratio_
                                cumvars = explained_variance_ratio.cumsum()

                            # Access the components (eigenvectors) if needed
                            components = pca.components_

                            # Plot the cumulative explained variance
                            ax[0].plot(range(1, len(cumvars) + 1), cumvars)
                            ax[0].set_xlabel('Number of Principal Components')
                            ax[0].set_ylabel('Cumulative Explained Variance')
                            # Plot the cumulative explained variance
                            ax[1].loglog(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
                            ax[1].set_xlabel('Principal Component Number')
                            ax[1].set_ylabel('Explained Variance')
                            plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_subjavg_explained_variance.{save_type}')
                            
                            if 'individualrois' in analysis_type.lower():
                                print(f"Components shape: {components.shape}")
                                eigenspectrums[ROIS[roi]] = explained_variance_ratio
                                cumvars[ROIS[roi]] = explained_variance_ratio.cumsum()
                                varThresholds[ROIS[roi]] = np.argmax(cumvars[ROIS[roi]] > varThreshVal*cumvars[ROIS[roi]][-1])+1  # +1 because of 0 indexing
                                
                                
                                # Plot the cumulative explained variance
                                ax[0].plot(range(1, len(cumvars[ROIS[roi]]) + 1), cumvars[ROIS[roi]], color=roi_colors[ROIS[roi]])
                                ax[0].set_xlabel('Number of Principal Components')
                                ax[0].set_ylabel('Cumulative Explained Variance')
                                # Plot the cumulative explained variance
                                ax[1].loglog(range(1, len(eigenspectrums[ROIS[roi]]) + 1), eigenspectrums[ROIS[roi]], color=roi_colors[ROIS[roi]])
                                ax[1].set_xlabel('Principal Component Number')
                                ax[1].set_ylabel('Explained Variance')
                            # plt.title('Explained Variance for Principal Components')
                            # plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_roi{roi}_subjavg_explained_variance_loglog.{save_type}')
                            # plt.close()
                            # for c in range(n_components):
                            #     comp_NN_positive = get_gcc_nearest_neighbour(components[c], n_neighbours=10, METRIC='cosine')
                            #     comp_NN_negative = get_gcc_nearest_neighbour(-components[c], n_neighbours=10, METRIC='cosine')

                            #     print(f"Component {c} NN positive: {comp_NN_positive}")
                            #     print(f"Component {c} NN negative: {comp_NN_negative}")

                    elif 'pca3D' in analysis_type and 'xval' in analysis_type.lower():
            
                        if subj == 'average':
                            # this projects the roi data onto the pca fitted on the whole data 
                            # (e.g. the visual rois if 'visroi' in analysis_type)
                            n_components = 768
                            pca1 = PCA(n_components=n_components)
                            emb1 = pca1.fit_transform(filtered_data_in1)
                            pca2 = PCA(n_components=n_components)
                            emb2 = pca2.fit_transform(filtered_data_in2)

                            xval1 = pca1.transform(filtered_data_in2)
                            xval_explained_variance_ratio1 = np.var(xval1, axis=0)/np.var(xval1, axis=0).sum()
                            xval_cumvar1 = xval_explained_variance_ratio1.cumsum()
                            xval2 = pca2.transform(filtered_data_in1)
                            xval_explained_variance_ratio2 = np.var(xval2, axis=0)/np.var(xval2, axis=0).sum()
                            xval_cumvar2 = xval_explained_variance_ratio2.cumsum()

                            ax[0,0].plot(range(1, len(xval_cumvar1) + 1), xval_cumvar1)
                            ax[0,0].set_title('Train split 1, test split 2 (cumvar explained)')
                            ax[0,0].set_xlabel('Number of Principal Components')
                            ax[0,0].set_ylabel('Cumulative Explained Variance')
                            ax[0,1].loglog(range(1, len(xval_explained_variance_ratio1) + 1), xval_explained_variance_ratio1)
                            ax[0,1].set_title('Train split 1, test split 2 (eigenspectrum)')
                            ax[0,1].set_xlabel('Principal Component Number')
                            ax[0,1].set_ylabel('Explained Variance')
                            ax[1,0].plot(range(1, len(xval_cumvar2) + 1), xval_cumvar2)
                            ax[1,0].set_title('Train split 2, test split 1 (cumvar explained)')
                            ax[1,0].set_xlabel('Number of Principal Components')
                            ax[1,0].set_ylabel('Cumulative Explained Variance')
                            ax[1,1].loglog(range(1, len(xval_explained_variance_ratio2) + 1), xval_explained_variance_ratio2)
                            ax[1,1].set_title('Train split 2, test split 1 (eigenspectrum)')
                            ax[1,1].set_xlabel('Principal Component Number')
                            ax[1,1].set_ylabel('Explained Variance')
                            plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_subjavg_XVAL_varExplained.{save_type}')
                            plt.close()

                            plt.figure(figsize=(6, 6))  # Set figsize to create a square plot
                            plt.loglog(range(1, len(xval_explained_variance_ratio1) + 1), xval_explained_variance_ratio1, 'k')
                            plt.loglog(range(1, len(xval_explained_variance_ratio2) + 1), xval_explained_variance_ratio2, 'gray')
                            plt.legend(['PCA fit on split 1, explained variance on split 2', 'PCA fit on split 2, explained variance on split 1'])
                            plt.xlabel('Principal Component Number')
                            plt.ylabel('Explained Variance')
                            plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_subjavg_XVAL_varExplained_loglog.{save_type}')
                            plt.show()



            # Save the cumulative explained variance for each ROI
            if subj == 'average' and 'pca3D' in analysis_type.lower():
                with open(f'{results_dir}/{data_type}_{analysis_type}_subjavg_cumulative_explained_variances.pkl', 'wb') as file:
                    pickle.dump(cumvars, file)
                with open(f'{results_dir}/{data_type}_{analysis_type}_subjavg_eigenspectrums.pkl', 'wb') as file:
                    pickle.dump(eigenspectrums, file)
                with open(f'{results_dir}/{data_type}_{analysis_type}_subjavg_varThresholds.pkl', 'wb') as file:
                    pickle.dump(varThresholds, file)
                
                # add ROIS.values() as legends to the plots
                ax[0].legend(varThresholds.keys())
                ax[1].legend(varThresholds.keys())
                plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_subjavg_pcaVarPlots.{save_type}')
                plt.close()

                keys = varThresholds.keys()
                values = varThresholds.values()
                plt.bar(keys, values, color=[roi_colors[k] for k in keys])
                plt.xlabel('ROIs')
                plt.ylabel(f'Number of Principal Components needed for {varThreshVal*100}% variance')
                plt.savefig(f'{results_dir}/{data_type}_{analysis_type}_subjavg_varThresholds_barplot_{varThreshVal*100}%.{save_type}')


