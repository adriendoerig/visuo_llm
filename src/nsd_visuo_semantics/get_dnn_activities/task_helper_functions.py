import importlib, os, pickle, h5py, torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.transform import resize


def localdir_modulespec(module_name, dir_path):
    """This function allows us to import functions from remote folders (e.g., needed when loading the model
    from root/save_dir/.../_code_used_for_training) - used in root/task_helper_fuctions.load_model_from_path
    """
    file_path = dir_path + f"/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_model(hparams, n_classes, saved_model_path=None):
    """get model from model function"""

    # get input shape, using None as batch size so user can feed any batch size
    input_shape = [None, hparams["image_size"], hparams["image_size"], 3]

    file_path = saved_model_path + "/_code_used_for_training/models"
    module_name = "setup_model"
    setup_model = localdir_modulespec(module_name, file_path)

    # get model function and call it to get the keras model
    model_function = setup_model.get_model_function(hparams["model_name"])
    net = model_function(input_shape, n_classes, hparams)

    return net


def load_and_override_hparams(saved_model_path, **kwargs):
    
    with open(f"{saved_model_path}/hparams.pickle", "rb") as f:
            hparams = pickle.load(f)

    hparams["saved_model_path"] = saved_model_path
    
    for k, v in kwargs.items():
        hparams[k] = v

    return hparams


def load_model_from_path(saved_model_path, epoch_to_load, print_summary=True, hparams=None):

    if hparams is None:
        with open(f'{saved_model_path}/hparams.pickle', 'rb') as f:
            hparams = pickle.load(f)

    n_classes = get_n_classes(hparams)

    print('\ncreating model...')
    net = get_model(hparams, n_classes, saved_model_path=saved_model_path)

    print('\nloading weights...')
    if 'simclr' in hparams['model_name'] and 'finetune' not in hparams['model_name']:
        # Load encoder weights
        encoder_ckpt_path = os.path.join(saved_model_path, 'training_checkpoints', f'ckpt_ep{epoch_to_load:03d}.h5')
        net = net.encoder
        net.load_weights(encoder_ckpt_path)
    else:
        if epoch_to_load == 0:
            weights_filename = 'model_weights_init.h5'
        else:
            weights_filename = f'ckpt_ep{epoch_to_load:03d}.h5'
        net.load_weights(os.path.join(saved_model_path, 'training_checkpoints', weights_filename))

    hparams['test_mode'] = True
    print(f"test_mode={hparams['test_mode']}, setting trainable=False for all layers")
    for layer in net.layers:
        layer.trainable = False

    if print_summary:
        net.summary()
    
    net.compile()

    return net, hparams


def get_activities_model(net, n_layers, hparams, pre_post_norm="post", include_readout=False):
    timesteps = max(hparams["n_recurrent_steps"], 1)  # because hparams['n_recurrent_steps'] = 0 for ff nets

    if hparams["norm_type"] == "IN":
        norm_layer_name = "instancenorm"
    elif hparams["norm_type"] == "GN":
        norm_layer_name = "groupnorm"
    elif hparams["norm_type"] == "LN":
        norm_layer_name = "layernorm"
    elif hparams["norm_type"] == "DN":
        norm_layer_name = "divisivenorm"
    elif hparams["norm_type"] == "BN":
        norm_layer_name = "batchnorm"
    elif hparams["norm_type"] == "no_norm":
        norm_layer_name = ""
    else:
        raise Exception(f'hparams["norm_type"]={hparams["norm_type"]} is not understood in get_activities_model')

    # make keras model to collect layer activities
    if include_readout:
        n_layers += 1
    readout_layer_names = [[None] * timesteps for _ in range(n_layers)]  # list of lists [n_layers+output][hparams['timesteps']]
    readout_layer_shapes = [[None] * timesteps for _ in range(n_layers)]
    readout_layers = [[None] * timesteps for _ in range(n_layers)]
    for layer_id in range(n_layers):
        for this_layer in net.layers:
            for t in range(timesteps):
                n_digits = (2 if t >= 10 else 1)  # to deal with '11' vs. '1' in the layer name
                if pre_post_norm == "pre" or hparams["norm_type"] == "no_norm":
                    if (
                        f'{hparams["activation"]}_layer_{layer_id}_time_{t}' in this_layer.name.lower()
                        and (this_layer.name.lower()[-(n_digits + 1)]) == "_"
                    ):
                        readout_layers[layer_id][t] = this_layer.output
                        readout_layer_names[layer_id][t] = this_layer.name.lower()
                        readout_layer_shapes[layer_id][t] = this_layer.output.shape
                elif pre_post_norm == "post":
                    if (
                        f"{norm_layer_name}_layer_{layer_id}_time_{t}" in this_layer.name.lower()
                        and (this_layer.name.lower()[-(n_digits + 1)]) == "_"
                    ):
                        readout_layers[layer_id][t] = this_layer.output
                        readout_layer_names[layer_id][t] = this_layer.name.lower()
                        readout_layer_shapes[layer_id][t] = this_layer.output.shape
                else:
                    raise Exception('pre_post_norm not understood. use "pre" or "post"')

                if this_layer.name.lower() == f"output_time_{t}" and include_readout:
                    readout_layers[n_layers][t] = this_layer.output
                    readout_layer_names[n_layers][t] = this_layer.name.lower()
                    readout_layer_shapes[n_layers][t] = this_layer.output.shape

    activities_model = tf.keras.Model(inputs=net.input, outputs=readout_layers, name="activities_model")

    return activities_model, readout_layer_names, readout_layer_shapes


def get_n_classes(hparams=None, dataset_path=None, dataset_subset=None):

    if 'simclr' in hparams['model_name'].lower():
        return 0

    if [hparams, dataset_path] == [None, None]:
        raise Exception('hparams or dataset_path must be passed as arguments in get n_classes')

    dataset_path = hparams['dataset'] if dataset_path is None else dataset_path
    with h5py.File(dataset_path, "r") as f:
        print(f'getting n_classes from {dataset_path}')
        if hparams['embedding_target']:
            print(f'\tusing embeddings dimension from {hparams["target_dataset_name"]}')
            return f['train'][hparams['target_dataset_name']][0].shape[-1]
        else:
            print(f'\tusing len(categories)')
            return f['categories'][:].shape[0]
        

# this function receives a vector of floats, and returns a multihot vector with a 1 in the position of the N max values
def float2multihot(x, N):
    # x is a vector of floats
    # N is an integer
    # returns a multihot vector with a 1 in the position of the N max values
    x = np.array(x)
    x[x.argsort()[:-N]] = 0
    x[x.argsort()[-N:]] = 1
    return x


def get_closest_caption(predicted_embedding, embeddings, captions, n_closest=1):
    """get closest caption to predicted embedding"""
    # predicted_embedding is a vector of floats
    # embeddings is a matrix of shape (n_embeddings, embedding_dim)
    # captions is a list of strings
    # n_closest is an integer
    # returns a list of n_closest captions

    # get distances
    distances = np.linalg.norm(embeddings - predicted_embedding, axis=1)
    # get indices of closest captions
    closest_caption_indices = distances.argsort()[:n_closest]
    # get closest captions
    closest_captions = [captions[i] for i in closest_caption_indices]

    return [x[0] for x in closest_captions]  # return only the first of the 5 coco captions


def np_to_pillow_img(img, scale='[0,255]'):

    img = (img + 1) / 2  # rescale to [0, 1]
    if scale == '[0,255]':
        img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img, mode="RGB")
    return img


def torchhub_preprocess_batch(tf_batch, transform, image_size=224, scale='[0,255]'):
    """Formatting to translate between our data generation pipeline and the one
    used for ipcl (https://github.com/harvard-visionlab/open_ipcl) and other tochhub models"""

    np_batch = tf_batch.numpy()

    torch_batch = torch.zeros(np_batch.shape[0], np_batch.shape[3], image_size, image_size)
    # clip needs this kind of preprocessing
    for i in range(np_batch.shape[0]):
        img = np_to_pillow_img(np_batch[i], scale=scale)
        torch_batch[i] = transform(img).unsqueeze(0)

    return torch_batch


def clip_preprocess_batch(tf_batch, preprocess, image_size=224):
    """Our pipeline outputs a tf batch of im_size=128. This needs to be changed
    for clip"""

    np_batch = tf_batch.numpy()

    torch_batch = torch.zeros(np_batch.shape[0], np_batch.shape[3], image_size, image_size)
    # clip needs this kind of preprocessing
    for i in range(np_batch.shape[0]):
        img = np_to_pillow_img(np_batch[i])
        torch_batch[i] = preprocess(img).unsqueeze(0)
    
    return torch_batch


def brainscore_preprocess_batch(tf_batch, model, image_size):
    """Our pipeline outputs a tf batch of im_size=128. This needs to be changed
    for brainscore models"""

    np_batch = tf_batch.numpy()

    if 'pytorch' in str(type(model)).lower():
        # torch wants channels first
        out_batch = np.zeros((np_batch.shape[0], 3, image_size, image_size)) 
        np_batch = np.transpose(np_batch, [0, 3, 1, 2])
        for i in range(np_batch.shape[0]):
            img = resize(np_batch[i], (3, image_size, image_size), anti_aliasing=True)
            out_batch[i] = img
    
    else:
        # tf wants channels last
        out_batch = np.zeros((np_batch.shape[0], image_size, image_size, 3)) 
        for i in range(np_batch.shape[0]):
            img = resize(np_batch[i], (image_size, image_size, 3), anti_aliasing=True)
            out_batch[i] = img
    
    return np_batch


def get_brainscore_layer_activations(activations_model, readout_layer, batch):

    assert len(readout_layer) == 1, "readout_layer must be a list with one element"

    out_dict = activations_model.get_activations(batch, readout_layer)
    activations = out_dict[list(out_dict.keys())[0]]

    return activations


def google_simclr_preprocess_batch(tf_batch, image_size=224):
    """Our pipeline outputs a tf batch of im_size=128, normalized to [-1,1]. This needs to be changed
    for google_simclr models.
    Formatting requirements inferred from https://github.com/google-research/simclr/blob/master/tf2/colabs/load_and_inference.ipynb"""

    tf_batch = tf.image.convert_image_dtype(tf_batch, dtype=tf.float32)
    tf_batch = tf.image.resize(tf_batch, (image_size, image_size))
    
    # normalize from [-1,1] to [0,1]
    tf_batch = (tf_batch + 1) / 2

    return tf_batch