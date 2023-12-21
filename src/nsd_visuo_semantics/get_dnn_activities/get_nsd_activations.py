import os, h5py, pickle, torch, clip
import matplotlib.pyplot as plt
import numpy as np
from nsd_visuo_semantics.get_dnn_activities.dataset_loader.make_tf_dataset import get_dataset
from nsd_visuo_semantics.get_dnn_activities.task_helper_functions import get_activities_model, load_and_override_hparams, load_model_from_path, \
                                                                         float2multihot, get_closest_caption, tf_to_torch_batch
from nsd_visuo_semantics.get_dnn_activities.get_modelName2Path_dict import get_modelName2Path_dict
from nsd_visuo_semantics.get_embeddings.nsd_embeddings_utils import get_words_from_multihot
from nsd_visuo_semantics.get_embeddings.word_lists import coco_categories_91

"""Get activities for all layers of a network on the nsd dataset.
We average across space to keep size reasonable
"""


def get_nsd_activations(MODEL_NAMES, dataset_path,
                        networks_basedir, results_dir, safety_check_plots_dir,
                        nsd_captions_path=None, nsd_embeddings_path=None,
                        n_layers=10, epoch=400, OVERWRITE=False):


    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(safety_check_plots_dir, exist_ok=True)

    n_nsd_imgs = 73000
    print_N_samples = 20  # print shapes and plot images equally spaced throughout the dataset

    modelname2path = get_modelName2Path_dict(networks_basedir)

    for model_name in MODEL_NAMES:

        model_savedir = modelname2path['default' if 'clip' in model_name.lower() else model_name]
        hparams = load_and_override_hparams(model_savedir, dataset=dataset_path, batch_size=50)  # CAREFUL: batch_size must divide 73000 to an int, otherwise there will be images missing at the end of the dataset


        if 'mpnet' in model_name:
            with open(nsd_captions_path, "rb") as fp:
                loaded_captions = pickle.load(fp)
            with open(nsd_embeddings_path, "rb") as fp:
                loaded_embeddings = pickle.load(fp)

        if 'clip' in model_name.lower():
            print('Using openAI clip')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if 'vit' in model_name.lower():
                model, preprocess = clip.load('ViT-B/32', device)
            elif 'rn50' in model_name.lower():
                model, preprocess = clip.load('RN50', device)
            else:
                raise ValueError(f"model_name {model_name} not recognized")
            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"
        else:
            print(f"Creating {model_name} model and loading weights from {model_savedir}")
            net, hparams = load_model_from_path(model_savedir, epoch, hparams=hparams, print_summary=True)
            activities_model, readout_layer_names, readout_layer_shapes = get_activities_model(net, n_layers, hparams)
            print(f"Reading from layers: {readout_layer_names}")
            activations_file_name = f"{results_dir}/{model_name}_nsd_activations_epoch{epoch}.h5"
        
        if os.path.exists(activations_file_name) and not OVERWRITE:
            print(f"Activations file {activations_file_name} already exists. Skipping.")
            continue
        else:
            print(f"Saving activations in {activations_file_name}")
        
        nsd_dataset = get_dataset(hparams, dataset_path=dataset_path, dataset="test")
        for x in nsd_dataset:
            if 'simclr' in model_name.lower():
                btch_sz, imh, imw, imc = x.shape
            else:
                btch_sz, imh, imw, imc = x[0].shape
            break

        print_every_N_batches = n_nsd_imgs // (btch_sz * print_N_samples)
        if 'clip' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = tf_to_torch_batch(x[0], preprocess)
                with torch.no_grad():
                    dummy_out = model.encode_image(dummy_batch)
                clip_features = np.zeros((n_nsd_imgs, dummy_out.shape[-1]))
                
        else:
            activations_file = h5py.File(activations_file_name, "w")
            # prepare h5 file structure
            print(f"Preparing to save in {activations_file_name}")
            for lin in range(n_layers):
                for t in range(hparams["n_recurrent_steps"]):
                    activations_file.create_dataset(
                        readout_layer_names[lin][t],
                        shape=(n_nsd_imgs, readout_layer_shapes[lin][t][-1]),  # we avg across space to keep size reasonable
                        dtype=np.float32,
                    )

        # get activities
        for i, x in enumerate(nsd_dataset):
            if 'simclr' in model_name.lower():
                batch_imgs = x
            else:
                batch_imgs, batch_labels, batch_class_weights = x

            if 'clip' in model_name.lower():
                torch_batch_imgs = tf_to_torch_batch(batch_imgs, preprocess)
                with torch.no_grad():
                    clip_features[i * btch_sz : (i + 1) * btch_sz] = model.encode_image(torch_batch_imgs).numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD actifities for {model_name} dnn: {i*hparams["batch_size"]/n_nsd_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", torch_batch_imgs.shape)
                    img = torch_batch_imgs[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be using openai's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    img_to_plot = np.moveaxis(img, 0, -1)
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            else:
                layer_activities = activities_model(batch_imgs)
                for lin in range(n_layers):
                    for t in range(hparams["n_recurrent_steps"]):
                        activations_file[readout_layer_names[lin][t]][i * btch_sz : (i + 1) * btch_sz] = np.mean(layer_activities[lin][t], axis=(1, 2))

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD actifities for {model_name} dnn: {i*hparams["batch_size"]/n_nsd_imgs*100}%')
                    print("batch_imgs.shape: ", batch_imgs.shape)
                    img = batch_imgs[0].numpy()
                    print(f"Img min/max - should be in [-1, 1]: {np.min(img)}, {np.max(img)}")  # PLEASE MAKE SURE THIS IS IN [-1,1]
                    img_to_plot = (img + 1) / 2  # rescale to [0, 1]
                    plt.imshow(img_to_plot)
                    if 'multihot' in model_name:
                        model_out = net(batch_imgs)[-1].numpy()  # numpy array of predictions for the last timestep
                        model_out_img = float2multihot(model_out[0], 3)  # multihot encoding of the 3 most highly activated categories
                        pred_categs = get_words_from_multihot(model_out_img, coco_categories_91)
                        plt.title(f"Predicted categories: {pred_categs}")
                    elif 'mpnet' in model_name:
                        if loaded_embeddings is None or loaded_captions is None:
                            pass
                        else:
                            model_out = net(batch_imgs)[-1].numpy()
                            c = get_closest_caption(model_out[0], loaded_embeddings, loaded_captions)
                            plt.title(f"closest nsd caption:\n{c}")

                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_ep{epoch}_check_batch_{i}.png")
                    plt.close()

                    # [print(f"l={readout_layer_names[lin]}, t={t} activities shape: {layer[t].numpy().shape}")
                    #     for (lin, layer) in enumerate(layer_activities)
                    #     for t in range(hparams["n_recurrent_steps"])]
                        

        if 'clip' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(clip_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")

    del activations_file, net, activities_model  # free memory space
