"""Get activities for all layers of a network on the nsd dataset.
We average across space to keep size reasonable
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nsd_visuo_semantics.get_dnn_activities.dataset_loader.make_tf_dataset import (
    get_dataset,
)
from nsd_visuo_semantics.get_dnn_activities.task_helper_functions import (
    get_activities_model,
    load_and_override_hparams,
    load_model_from_path,
)

physical_devices = tf.config.list_physical_devices("GPU")
[tf.config.experimental.set_memory_growth(dev, True) for dev in physical_devices]
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"  # IMPORTANT: must be a dataset with NSD in nsd_id order as the test set
networks_basedir = "/share/klab/adoerig/adoerig/semantics_paper_nets/semantics_paper_ms_coco_nets"
resuts_dir = f"{networks_basedir}/extracted_activities"
os.makedirs(resuts_dir, exist_ok=True)
safety_check_plots_dir = "./safety_check_plots_dir"
os.makedirs(safety_check_plots_dir, exist_ok=True)

n_layers = 10
pre_post = "post"  # get activities pre or post norm
epoch = 200

n_nsd_imgs = 73000
print_N_samples = 20  # print shapes and plot images equally spaced throughout the dataset

shared_model_prefix = "blt_vNet_half_channels_semPaper_from_scratch_"
modelname2path = {
    "multihot_ff": f"{networks_basedir}/{shared_model_prefix}img_multi_hot_ff",
    "multihot_rec": f"{networks_basedir}/{shared_model_prefix}img_multi_hot_rec",
    "guse_ff": f"{networks_basedir}/{shared_model_prefix}guse_mean_embeddings_ff",
    "guse_rec": f"{networks_basedir}/{shared_model_prefix}guse_mean_embeddings_rec",
    "mpnet_ff": f"{networks_basedir}/{shared_model_prefix}all_mpnet_base_v2_mean_embeddings_ff",
    "mpnet_rec": f"{networks_basedir}/{shared_model_prefix}all_mpnet_base_v2_mean_embeddings_rec",
}

for model_name in ["multihot_rec", "mpnet_rec"]:
    # ['multihot_ff', 'multihot_rec', 'guse_ff', 'guse_rec', 'mpnet_ff', 'mpnet_rec']:

    print(modelname2path.keys())
    model_savedir = modelname2path[model_name]
    print(f"Creating {model_name} model and loading weights from {model_savedir}")
    hparams = load_and_override_hparams(model_savedir, dataset=dataset_path, batch_size=50)  # CAREFUL: batch_size must divide 73000 to an int, otherwise there will be images missing at the end of the dataset
    net, hparams = load_model_from_path(model_savedir, epoch, hparams=hparams, print_summary=True, test_mode=True)
    activities_model, readout_layer_names, readout_layer_shapes = get_activities_model(net, n_layers, hparams)
    print(f"Reading from layers: {readout_layer_names}")
    
    nsd_dataset = get_dataset(hparams, dataset_path=dataset_path, dataset="test")
    for x in nsd_dataset:
        btch_sz, imh, imw, imc = x[0].shape
        break

    print_every_N_batches = n_nsd_imgs // (btch_sz * print_N_samples)
    activations_file_name = (
        f"{resuts_dir}/{model_name}_nsd_activations_epoch{epoch}.h5"
    )
    with h5py.File(activations_file_name, "w") as activations_file:
        # prepare h5 file structure
        print(f"Preparing to save in {activations_file_name}")
        for lin in range(n_layers):
            for t in range(hparams["n_recurrent_steps"]):
                activations_file.create_dataset(
                    readout_layer_names[lin][t],
                    shape=(
                        n_nsd_imgs,
                        readout_layer_shapes[lin][t][-1],
                    ),  # we avg across space to keep size reasonable
                    dtype=np.float32,
                )

        # get activities
        for i, x in enumerate(nsd_dataset):
            batch_imgs, batch_labels, batch_class_weights = x
            layer_activities = activities_model(batch_imgs)

            if i % print_every_N_batches == 0:
                print(f'\nGetting NSD actifities for {model_name} dnn: {i/(hparams["batch_size"]*print_N_samples)*100}%')
                print(
                    "batch_imgs.shape, batch_labels['output_time_0'].shape: ",
                    batch_imgs.shape,
                    batch_labels["output_time_0"].shape,
                )
                img = batch_imgs[0].numpy()
                print(f"Img min/max - should be in [-1, 1]: {np.min(img)}, {np.max(img)}")  # PLEASE MAKE SURE THIS IS IN [-1,1]
                plt.imshow(img)
                plt.savefig(f"{safety_check_plots_dir}/check_batch_{i}.png")
                [print(f"l={readout_layer_names[lin]}, t={t} activities shape: {layer[t].numpy().shape}")
                    for (lin, layer) in enumerate(layer_activities)
                    for t in range(hparams["n_recurrent_steps"])]

            for lin in range(n_layers):
                for t in range(hparams["n_recurrent_steps"]):
                    activations_file[readout_layer_names[lin][t]][i * btch_sz : (i + 1) * btch_sz] = np.mean(layer_activities[lin][t], axis=(1, 2))
