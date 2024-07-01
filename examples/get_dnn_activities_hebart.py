from nsd_visuo_semantics.get_dnn_activities.get_nsd_activations import get_nsd_activations
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
[tf.config.experimental.set_memory_growth(dev, True) for dev in physical_devices]
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

OVERWRITE = True

# GENERAL PATHS, ETC
base_path = '/share/klab/adoerig/adoerig/'
dataset_path = f"{base_path}/charest_special100.h5"
networks_basedir = f"{base_path}/blt_rdl_pipeline/save_dir"
results_dir = f"{base_path}/charest_special100_dnn_extracted_activities"
safety_check_plots_dir = f"{results_dir}/dnn_activities_safety_check_plots"
nsd_captions_path = f"{base_path}/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"
nsd_embeddings_path = f"{base_path}/nsd_visuo_semantics/results_dir/saved_embeddings/nsd_mpnet_mean_embeddings.pkl"

# MODEL_NAMES = ['mpnet_rec_seed2', 'multihot_rec_seed2', 
#                'mpnet_resnet50_finalLayer', 'multihot_resnet50_finalLayer']
MODEL_NAMES = [f'mpnet_rec_seed{n}' for n in range(1,11)]
# MODEL_NAMES = [f'multihot_rec_seed{n}' for n in range(1,11)]

get_nsd_activations(MODEL_NAMES, dataset_path, None,
                    networks_basedir, results_dir, safety_check_plots_dir,
                    nsd_captions_path=nsd_captions_path, nsd_embeddings_path=nsd_embeddings_path,
                    n_layers=10, epoch=200, train_val_nsd='test', OVERWRITE=OVERWRITE, batch_size=1)

