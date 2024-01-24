import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois


if __name__ == "__main__":

    for vars_or_scores in ['uniquevars', 'scores']:

        n_subjects = 8
        subs = [f"subj0{x+1}" for x in range(n_subjects)]
        rdm_distance = 'correlation'
        remove_shared_515 = False
        which_rois = 'streams'

        # set up directories
        base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir"
        models_dir = os.path.join(base_save_dir,f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}')
        roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
        os.makedirs(roi_analyses_dir, exist_ok=True)
        results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")
        os.makedirs(results_dir, exist_ok=True)
        subj_roi_rdms_path = os.path.join(results_dir, "subj_roi_rdms")
        os.makedirs(subj_roi_rdms_path, exist_ok=True)
        nsd_dir = '/share/klab/datasets/NSD_for_visuo_semantics'
        rois_dir = os.path.join(nsd_dir, 'nsddata/freesurfer/fsaverage/label')

        # we use the fsaverage space.
        targetspace = "fsaverage"

        # Get roi info
        maskdata, ROIS = get_rois(which_rois, rois_dir)

        data_dir = os.path.join(
            base_save_dir,
            f"searchlight_respectedsampling_{rdm_distance}_newTest",
            "{}",
            "var_partition_['mpnet', 'mpnet_nouns', 'mpnet_verbs']")
        
        brain_vol_scores_avg = []

        for subj in subs+['average']:

            print(f"Processing {subj}, for {vars_or_scores}...")

            if subj != 'average':

                with open(os.path.join(data_dir.format(subj), "model_combinations.pkl"), "rb") as f:
                    model_combinations = pickle.load(f)

                brain_vol_scores = []
                for i, mc in enumerate(model_combinations):
                    brain_vol_scores_lh = np.load(os.path.join(data_dir.format(subj), 'fsaverage', f'lh.{subj}-{vars_or_scores}-{mc}-surf.npy'), allow_pickle=True)
                    brain_vol_scores_rh = np.load(os.path.join(data_dir.format(subj), 'fsaverage', f'rh.{subj}-{vars_or_scores}-{mc}-surf.npy'), allow_pickle=True)
                    brain_vol_scores.append(np.concatenate([brain_vol_scores_lh, brain_vol_scores_rh], axis=0))
                brain_vol_scores = np.stack(brain_vol_scores, axis=-1)

                # remove voxels with ROI label 0 (i.e., non-visual parts)
                filtered_brain_vol_scores = brain_vol_scores[maskdata != 0, :]
                filtered_labels = maskdata[maskdata != 0]

                brain_vol_scores_avg.append(filtered_brain_vol_scores)
            
            else:

                filtered_brain_vol_scores = np.mean(brain_vol_scores_avg, axis=0)
                filtered_labels = maskdata[maskdata != 0]


            # filtered_brain_vol_scores = filtered_brain_vol_scores[:1000]
            # filtered_labels = filtered_labels[:1000]

            # make tsne embedding
            tsne = TSNE(n_components=2, random_state=42)
            embedding = tsne.fit_transform(filtered_brain_vol_scores)

            plt.figure(figsize=(12, 12))
            for label_value, label_string in ROIS.items():  # Skip label 0
                if label_value == 0:
                    continue
                else:
                    indices = filtered_labels == label_value
                    plt.scatter(embedding[indices, 0], embedding[indices, 1], s=50, alpha=0.6, label=label_string)

            # Customize the plot
            plt.title('2D t-SNE Plot')

            # Add legend
            plt.legend(title='Classes', loc='upper right')

            plt.savefig(f'./var_part_model_{vars_or_scores}_tsne_{subj}.png')