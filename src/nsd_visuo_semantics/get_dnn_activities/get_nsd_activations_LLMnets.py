import os, h5py, pickle, torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nsd_visuo_semantics.get_dnn_activities.dataset_loader.make_tf_dataset import get_dataset
from nsd_visuo_semantics.get_dnn_activities.task_helper_functions import get_activities_model, get_activities_model_by_layername, load_and_override_hparams, load_model_from_path, \
                                                                         float2multihot, get_closest_caption, clip_preprocess_batch, brainscore_preprocess_batch, \
                                                                         get_brainscore_layer_activations, torchhub_preprocess_batch, google_simclr_preprocess_batch
from nsd_visuo_semantics.get_dnn_activities.ipcl_feature_extractor import FeatureExtractor
from nsd_visuo_semantics.get_dnn_activities.get_modelName2Path_dict import get_modelName2Path_dict
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import get_words_from_multihot
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91

"""Get activities for all layers of a network on the nsd dataset.
We average across space to keep size reasonable
"""


def get_nsd_activations_LLMnets(model_name, dataset_path,
                                networks_basedir, results_dir, safety_check_plots_dir,
                                nsd_captions_path=None, nsd_embeddings_path=None,
                                n_layers=10, epoch=400):


    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(safety_check_plots_dir, exist_ok=True)

    print_N_samples = 20  # print shapes and plot images equally spaced throughout the dataset

    # where to load stuff from
    modelname2path = get_modelName2Path_dict(networks_basedir)
    model_savedir = modelname2path[model_name]
    print(model_savedir)
    hparams = load_and_override_hparams(model_savedir, dataset=dataset_path, batch_size=50)  # CAREFUL: batch_size must divide n_images to an int, otherwise there will be images missing at the end of the dataset

    if 'mpnet' in model_name.lower():
        # if we're doing mpnet, we load the embeddings and captions
        # to plot nearest neighbours
        with open(nsd_captions_path, "rb") as fp:
            loaded_captions = pickle.load(fp)
        with open(nsd_embeddings_path, "rb") as fp:
            loaded_embeddings = pickle.load(fp)

    print(f"Creating {model_name} model and loading weights from {model_savedir}")
    net, hparams = load_model_from_path(model_savedir, epoch, hparams=hparams, print_summary=True)
    activities_model, readout_layer_names, readout_layer_shapes = get_activities_model(net, n_layers, hparams)
    print(f"Reading from layers: {readout_layer_names}")
    activations_file_name = f"{results_dir}/{model_name}_nsd_activations_epoch{epoch}.h5"
    
    # Get NSD dataset formatted as the networks expect
    input_dataset = get_dataset(hparams, dataset_path=dataset_path, dataset='test')
    with h5py.File(dataset_path, 'r') as f:
        n_imgs = f['test']['img_multi_hot'].shape[0]
    
    # Get the batch size, im_heigh, etc from the dataset
    for x in input_dataset:
        btch_sz, imh, imw, imc = x[0].shape
        break

    print_every_N_batches = n_imgs // (btch_sz * print_N_samples)
    # prepare h5 file structure to collect all layer activities
    print(f"Preparing to save in {activations_file_name}")
    with h5py.File(activations_file_name, 'w') as activations_file:
        for lin in range(n_layers):
            for t in range(hparams["n_recurrent_steps"]):
                activations_file.create_dataset(
                    readout_layer_names[lin][t],
                    shape=(n_imgs, readout_layer_shapes[lin][t][-1]),  # we avg across space to keep size reasonable
                    dtype=np.float32,
                )

    # get activities
    for i, x in enumerate(input_dataset):

        batch_imgs, batch_labels, batch_class_weights = x
        
        layer_activities = activities_model(batch_imgs)
        with h5py.File(activations_file_name, 'a') as activations_file:
            for lin in range(n_layers):
                for t in range(hparams["n_recurrent_steps"]):
                    activations_file[readout_layer_names[lin][t]][i * btch_sz : (i + 1) * btch_sz] = np.mean(layer_activities[lin][t], axis=(1, 2))

        if i % print_every_N_batches == 0:
            print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
            print("batch_imgs.shape: ", batch_imgs.shape)
            img = batch_imgs[0].numpy()
            print(f"Img min/max - should be in [-1, 1]: {np.min(img)}, {np.max(img)}")  # PLEASE MAKE SURE THIS IS IN [-1,1]
            img_to_plot = (img + 1) / 2  # rescale to [0, 1]
            plt.imshow(img_to_plot)
            
            model_out = net(batch_imgs)[-1].numpy()
            c = get_closest_caption(model_out[0], loaded_embeddings, loaded_captions)
            plt.title(f"closest nsd caption:\n{c}")

            plt.savefig(f"{safety_check_plots_dir}/{model_name}_ep{epoch}_check_batch_{i}.png")
            plt.close()

            # [print(f"l={readout_layer_names[lin]}, t={t} activities shape: {layer[t].numpy().shape}")
            #     for (lin, layer) in enumerate(layer_activities)
            #     for t in range(hparams["n_recurrent_steps"])]


if __name__ == "__main__":

    base_path = '/share/klab/adoerig/adoerig/'
    dataset_path = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5"  # IMPORTANT: must be a dataset with NSD in nsd_id order as the test set
    networks_basedir = f"{base_path}/blt_rdl_pipeline/save_dir"
    results_dir = f"./dnn_extracted_activities_carmenCheck"
    safety_check_plots_dir = "./dnn_activities_safety_check_plots_carmenCheck"
    nsd_captions_path = f"{base_path}/nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"
    nsd_embeddings_path = f"{base_path}/nsd_visuo_semantics/results_dir/saved_embeddings/nsd_mpnet_mean_embeddings.pkl"

    model_name = 'mpnet_rec_seed1'

    get_nsd_activations_LLMnets(model_name, dataset_path,
                                networks_basedir, results_dir, safety_check_plots_dir,
                                nsd_captions_path=nsd_captions_path, nsd_embeddings_path=nsd_embeddings_path,
                                n_layers=10, epoch=200)