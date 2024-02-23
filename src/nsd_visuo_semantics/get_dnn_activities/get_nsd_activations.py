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


def get_nsd_activations(MODEL_NAMES, dataset_path,
                        networks_basedir, results_dir, safety_check_plots_dir,
                        nsd_captions_path=None, nsd_embeddings_path=None,
                        n_layers=10, epoch=400, OVERWRITE=False, train_val_nsd='nsd'):
    
    if train_val_nsd == 'nsd':
        # nsd is stored in the test part of our h5 file.
        # you can also request the train or val part of the dataset
        train_val_nsd = 'test'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(safety_check_plots_dir, exist_ok=True)

    print_N_samples = 20  # print shapes and plot images equally spaced throughout the dataset

    modelname2path = get_modelName2Path_dict(networks_basedir)


    for model_name in MODEL_NAMES:

        if ('brainscore' in model_name.lower() or 'clip' in model_name.lower() or 'konkle_' in model_name.lower() or 'resnext101_32x8d_wsl' in model_name.lower() or 'thingsvision' in model_name.lower() or 'google_' in model_name.lower() or 'timm_' in model_name.lower()):
            model_savedir = modelname2path['default']
        elif 'finalLayer' in model_name:
            model_savedir = modelname2path[model_name.replace('_finalLayer', '').replace('GAP', '')]
        else:
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

        if 'google_simclrv1_rn50' in model_name.lower():
            
            model = tf.saved_model.load('/share/klab/adoerig/adoerig/nsd_visuo_semantics/examples/google_simclr_models/ResNet50_1x/saved_model')
            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'timm' in model_name.lower():

            import timm

            timm_name = model_name.replace('timm_', '')
            model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            model.eval()

            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)

            if 'nf_resnet50' in model_name.lower():
                im_sz = 256
            else:
                im_sz = 224

            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'clip' in model_name.lower():

            import clip

            print('Using openAI clip')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if 'vit' in model_name.lower():
                model, preprocess = clip.load('ViT-B/32', device)
            elif 'rn50' in model_name.lower():
                model, preprocess = clip.load('RN50', device)
            else:
                raise ValueError(f"model_name {model_name} not recognized")
            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'brainscore' in model_name.lower():
            # expects model to be of the form brainscore_ModelNameFromBsModels or brainscore_ModelNameFromBsModels_IT

            from brainscore_vision import load_model

            model = load_model(model_name.replace('brainscore_', ''))  # remove brainscore_ prefix
            activations_model = model.activations_model
            model_img_size = activations_model.image_size
            readout_layer = model.layers[-1]

            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'konkle_' in model_name.lower():
            # Self-supervised models from Konkle & Alvarez (2022)
            # model_name should be one of those described in
            # https://github.com/harvard-visionlab/open_ipcl
            model, transform = torch.hub.load("harvard-visionlab/open_ipcl", model_name.replace('konkle_', '').replace('_01inputs', ''))
            model.eval()

            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'resnext101_32x8d_wsl' in model_name.lower():
            from torchvision import transforms
            model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
            model.eval()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

        elif 'thingsvision' in model_name.lower():
            from thingsvision import get_extractor
            from thingsvision.utils.data import HDF5Dataset, DataLoader

            activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"

            if 'cornet-s' in model_name.lower():
                source = 'custom'
                features_layer = 'decoder.avgpool'
            elif 'simclr-rn50' in model_name.lower():
                source = 'ssl'
                features_layer = 'avgpool'
            elif 'barlowtwins-rn50' in model_name.lower():
                source = 'ssl'
                features_layer = 'avgpool'
            else:
                raise ValueError(f"model_name {model_name} not recognized")

            if not os.path.exists(activations_file_name) or OVERWRITE:
                extractor = get_extractor(model_name=model_name.replace('thingsvision_', ''), 
                                          source=source, device='cpu', pretrained=True)
                # extractor.show_model()  # if you want to see the model architecture

                dataset = HDF5Dataset(hdf5_fp=dataset_path, img_ds_key='test/data', 
                                      backend=extractor.get_backend(), transforms=extractor.get_transformations())
                batches = DataLoader(dataset=dataset, batch_size=50, backend=extractor.get_backend())

                features = extractor.extract_features(batches=batches, module_name=features_layer, flatten_acts=True)
            
                with open(activations_file_name, "wb") as fp:
                    pickle.dump(features, fp)
                print(f"Saved {model_name} features in {activations_file_name}")
            else:
                print(f"Activations file {activations_file_name} already exists. Skipping.")

            continue

        else:
            print(f"Creating {model_name} model and loading weights from {model_savedir}")
            net, hparams = load_model_from_path(model_savedir, epoch, hparams=hparams, print_summary=True)
            if 'layer' in model_name.lower():
                if 'resnet' in model_name.lower():
                    readout_layer_name = 'avg_pool'
                elif 'GAP' in model_name:
                    readout_layer_name = 'GlobalAvgPool_Time_5'
                else:
                    readout_layer_name = 'LayerNorm_Layer_9_Time_5'
                activities_model = get_activities_model_by_layername(net, readout_layer_name)
                print('Reading from layer:', readout_layer_name)
                activations_file_name = f"{results_dir}/{model_name}_nsd_image_features.pkl"
            else:
                activities_model, readout_layer_names, readout_layer_shapes = get_activities_model(net, n_layers, hparams)
                print(f"Reading from layers: {readout_layer_names}")
                activations_file_name = f"{results_dir}/{model_name}_nsd_activations_epoch{epoch}.h5"


        if train_val_nsd != 'test':
            # insert '_train' or '_val' in the filename before extension
            activations_file_name = activations_file_name.replace('.h5', f'_{train_val_nsd}.h5')
            activations_file_name = activations_file_name.replace('.pkl', f'_{train_val_nsd}.pkl')

        if os.path.exists(activations_file_name) and not OVERWRITE:
            print(f"Activations file {activations_file_name} already exists. Skipping.")
            continue
        else:
            print(f"Saving activations in {activations_file_name}")
        
        try:
            dummy = nsd_dataset.__dict__
            print('Dataset exists, will not be recreated')
        except NameError:
            print('Creating dataset')
            nsd_dataset = get_dataset(hparams, dataset_path=dataset_path, dataset=train_val_nsd)
            with h5py.File(dataset_path, 'r') as f:
                n_imgs = f[train_val_nsd]['img_multi_hot'].shape[0]
        
        for x in nsd_dataset:
            if 'simclr' in model_name.lower() and 'google' not in model_name.lower():
                btch_sz, imh, imw, imc = x.shape
            else:
                btch_sz, imh, imw, imc = x[0].shape
            break

        print_every_N_batches = n_imgs // (btch_sz * print_N_samples)
        if 'clip' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = clip_preprocess_batch(x[0], preprocess, 224)
                with torch.no_grad():
                    dummy_out = model.encode_image(dummy_batch)
                clip_features = np.zeros((n_imgs, dummy_out.shape[-1]))

        elif 'timm' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = torchhub_preprocess_batch(x[0], transform, image_size=im_sz, scale='[0,255]')
                dummy_out = model(dummy_batch)
                timm_features = np.zeros((n_imgs, dummy_out.shape[-1]))

        elif 'google_simclrv1_rn50' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = google_simclr_preprocess_batch(x[0], 224)
                dummy_out = model(dummy_batch, trainable=False)['final_avg_pool']
                google_features = np.zeros((n_imgs, dummy_out.shape[-1]))

        elif 'brainscore' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = brainscore_preprocess_batch(x[0], activations_model, model_img_size)
                dummy_out = get_brainscore_layer_activations(activations_model, readout_layer=[readout_layer], batch=dummy_batch)
                if dummy_out.ndim == 4:
                    # if it is a conv layer, we will average across space (as we do in our blt models)
                    if 'pytorch' in str(type(activations_model)).lower():
                        # pytorch has channels first
                        brainscore_features = np.zeros((n_imgs, dummy_out.shape[1]))
                    else:
                        brainscore_features = np.zeros((n_imgs, dummy_out.shape[-1]))
                else:
                    brainscore_features = np.zeros((n_imgs, dummy_out.shape[-1]))

        elif 'konkle_' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = torchhub_preprocess_batch(x[0], transform, 224, scale='[0,1]' if '01inputs' in model_name else '[0,255]')
                with FeatureExtractor(model, 'fc7') as extractor:
                    dummy_out = extractor(dummy_batch)['fc7']
                ipcl_features = np.zeros((n_imgs, dummy_out.shape[-1]))

        elif 'resnext101_32x8d_wsl' in model_name.lower():
            if os.path.exists(activations_file_name) and not OVERWRITE:
                print(f"Activations file {activations_file_name} already exists. Skipping.")
                continue
            else:
                dummy_batch = torchhub_preprocess_batch(x[0], transform, 224, scale='[0,255]')
                with FeatureExtractor(model, 'avgpool') as extractor:
                    dummy_out = extractor(dummy_batch)['avgpool'].squeeze()
                resnext_features = np.zeros((n_imgs, dummy_out.shape[-1]))
                
        else:
            if 'finalLayer' in model_name:
                dummy_batch = x[0]
                dummy_out = activities_model(dummy_batch)
                if 'GAP' in model_name:
                    layer_features = np.zeros((n_imgs, dummy_out.shape[-1]), dtype=np.float32)
                else:
                    # flatten and save space with float16
                    dummy_out = dummy_out.numpy().reshape(dummy_out.shape[0], -1)
                    layer_features = np.zeros((n_imgs, dummy_out.shape[-1]), dtype=np.float16)
            else:
                # prepare h5 file structure to collect all layer activities
                print(f"Preparing to save in {activations_file_name}")
                for lin in range(n_layers):
                    for t in range(hparams["n_recurrent_steps"]):
                        activations_file.create_dataset(
                            readout_layer_names[lin][t],
                            shape=(n_imgs, readout_layer_shapes[lin][t][-1]),  # we avg across space to keep size reasonable
                            dtype=np.float32,
                        )

        # get activities
        for i, x in enumerate(nsd_dataset):
            if 'simclr' in model_name.lower() and 'google' not in model_name.lower():
                batch_imgs = x
            else:
                batch_imgs, batch_labels, batch_class_weights = x

            if 'clip' in model_name.lower():
                torch_batch_imgs = clip_preprocess_batch(batch_imgs, preprocess, 224)
                with torch.no_grad():
                    clip_features[i * btch_sz : (i + 1) * btch_sz] = model.encode_image(torch_batch_imgs).numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", torch_batch_imgs.shape)
                    img = torch_batch_imgs[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be using openai's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    img_to_plot = np.moveaxis(img, 0, -1)
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            elif 'timm' in model_name.lower():
                with torch.no_grad():
                    timm_batch = torchhub_preprocess_batch(batch_imgs, transform, image_size=im_sz, scale='[0,255]')
                    timm_out = model(timm_batch)
                    timm_features[i * btch_sz : (i + 1) * btch_sz] = timm_out.numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", timm_batch.shape)
                    img = timm_batch[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be using timm's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    img_to_plot = np.moveaxis(img, 0, -1)
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            elif 'google_simclrv1_rn50' in model_name.lower():
                google_batch = google_simclr_preprocess_batch(batch_imgs, 224)
                google_out = model(google_batch, trainable=False)['final_avg_pool']
                google_features[i * btch_sz : (i + 1) * btch_sz] = google_out.numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", google_batch.shape)
                    img_to_plot = google_batch[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be normalized to [0,1]: " \
                          f"{np.min(img_to_plot)}, {np.max(img_to_plot)}, {np.mean(img_to_plot)}, {np.std(img_to_plot)}")
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            elif 'brainscore' in model_name.lower():
                bs_batch = brainscore_preprocess_batch(batch_imgs, activations_model, model_img_size)
                bs_out = get_brainscore_layer_activations(activations_model, readout_layer=[readout_layer], batch=bs_batch)
                if bs_out.ndim == 4:
                    # if it is a conv layer, we average across space (as we do in our blt models)
                    if 'pytorch' in str(type(activations_model)).lower():
                        bs_out = bs_out.mean(axis=(2,3))
                    else:
                        bs_out = bs_out.mean(axis=(1,2))
                brainscore_features[i * btch_sz : (i + 1) * btch_sz] = bs_out

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", bs_batch.shape)
                    img = bs_batch[0]
                    print(f"Img min/max, and channel-wise-means/stds - should be using brainscore's's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    if 'pytorch' in str(type(activations_model)).lower():
                        img_to_plot = np.moveaxis(img, 0, -1)
                    else:
                        img_to_plot = img
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            elif 'konkle_' in model_name.lower():

                ipcl_batch = torchhub_preprocess_batch(batch_imgs, transform, 224, scale='[0,1]' if '01inputs' in model_name else '[0,255]')
                with FeatureExtractor(model, 'fc7') as extractor:
                    ipcl_out = extractor(ipcl_batch)['fc7']
                ipcl_features[i * btch_sz : (i + 1) * btch_sz] = ipcl_out.numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", ipcl_batch.shape)
                    img = ipcl_batch[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be using ipcl's's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    img_to_plot = np.moveaxis(img, 0, -1)
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            elif 'resnext101_32x8d_wsl' in model_name.lower():

                resnext_batch = torchhub_preprocess_batch(batch_imgs, transform, 224)
                with FeatureExtractor(model, 'avgpool') as extractor:
                    resnext_out = extractor(resnext_batch)['avgpool'].squeeze()
                resnext_features[i * btch_sz : (i + 1) * btch_sz] = resnext_out.numpy()

                if i % print_every_N_batches == 0:
                    print(f'\nGetting NSD activities for {model_name} dnn: {i*hparams["batch_size"]/n_imgs*100}%')
                    print("batch_imgs.shape, batch_labels['output_time_0'].shape: ", resnext_batch.shape)
                    img = resnext_batch[0].numpy()
                    print(f"Img min/max, and channel-wise-means/stds - should be using resnext's normalisation: " \
                          f"{np.min(img)}, {np.max(img)}, {np.mean(img, axis=(1,2))}, {np.std(img, axis=(1,2))}")
                    img_to_plot = np.moveaxis(img, 0, -1)
                    plt.imshow(img_to_plot)
                    plt.savefig(f"{safety_check_plots_dir}/{model_name}_check_batch_{i}.png")
                    plt.close()

            else:
                if 'finalLayer' in model_name:
                    layer_activities = activities_model(batch_imgs)
                    if 'GAP' in model_name:
                        layer_features[i * btch_sz : (i + 1) * btch_sz] = layer_activities.numpy()
                    else:
                        flat_layer_activities = layer_activities.numpy().reshape(dummy_out.shape[0], -1).astype(np.float16)
                        layer_features[i * btch_sz : (i + 1) * btch_sz] = flat_layer_activities
                else:
                    layer_activities = activities_model(batch_imgs)
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
                    if 'multihot' in model_name:
                        if isinstance(net(batch_imgs), list):
                            model_out = net(batch_imgs)[-1].numpy()  # numpy array of predictions for the last timestep
                            model_out_img = float2multihot(model_out[0], 3)  # multihot encoding of the 3 most highly activated categories
                        else:
                            model_out_img = float2multihot(net(batch_imgs).numpy()[0], 3)
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
            del clip_features, model, preprocess
        elif 'brainscore' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(brainscore_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del brainscore_features, model, activations_model
        elif 'konkle_' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(ipcl_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del ipcl_features, model, transform
        elif 'resnext101_32x8d_wsl' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(resnext_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del resnext_features, model, transform
        elif 'google_simclrv1_rn50' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(google_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del google_features, model
        elif 'timm' in model_name.lower():
            with open(activations_file_name, "wb") as fp:
                pickle.dump(timm_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del timm_features, model, transform
        elif 'finalLayer' in model_name:
            with open(activations_file_name, "wb") as fp:
                pickle.dump(layer_features, fp)
            print(f"Saved {model_name} features in {activations_file_name}")
            del layer_features, net
        else:
            try:
                del activations_file, net, activities_model  # free memory space
            except:
                pass
