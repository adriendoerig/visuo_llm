from nsd_visuo_semantics.get_dnn_activities.get_nsd_activations import get_nsd_activations
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
[tf.config.experimental.set_memory_growth(dev, True) for dev in physical_devices]
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

OVERWRITE = True

# GENERAL PATHS, ETC
base_path = '/share/klab/adoerig/adoerig/'
dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"  # IMPORTANT: must be a dataset with NSD in nsd_id order as the test set
dataset_path_places365 = "/share/klab/datasets/places365_small/places365_small_first73000train.h5"
networks_basedir = f"{base_path}/blt_rdl_pipeline/save_dir"
results_dir = f"./dnn_extracted_activities"
safety_check_plots_dir = "./dnn_activities_safety_check_plots"
nsd_captions_path = f"{base_path}/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"
nsd_embeddings_path = f"{base_path}/nsd_visuo_semantics/results_dir/saved_embeddings/nsd_mpnet_mean_embeddings.pkl"

# ['CLIP-vit', 'CLIP-rn50', 'konkle_alexnetgn_ipcl_ref01', 'konkle_alexnetgn_supervised_ref12_augset1_5x', 'brainscore_alexnet', 'google_simclrv1_rn50']
MODEL_NAMES = ['sceneCateg_resnet50_finalLayer']
# MODEL_NAMES = []
# for seed in range(1,11):
#     # MODEL_NAMES += [f'mpnet_rec_seed{seed}', f'multihot_rec_seed{seed}']
    # MODEL_NAMES += [f'multihot_rec_seed{seed}_places365']#, f'multihot_rec_seed{seed}_places365']

for d in ['nsd']:
    get_nsd_activations(MODEL_NAMES, dataset_path, dataset_path_places365,
                        networks_basedir, results_dir, safety_check_plots_dir,
                        nsd_captions_path=nsd_captions_path, nsd_embeddings_path=nsd_embeddings_path,
                        n_layers=10, epoch=21, train_val_nsd=d, OVERWRITE=OVERWRITE)

