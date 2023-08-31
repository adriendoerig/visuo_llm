import importlib
import os
import pickle

import h5py
import tensorflow as tf


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
    if hparams["sequence_input"]:
        input_shape = [
            None,
            hparams["n_recurrent_steps"],
            hparams["image_size"],
            hparams["image_size"],
            3,
        ]
    else:
        input_shape = [None, hparams["image_size"], hparams["image_size"], 3]

    file_path = saved_model_path + "/_code_used_for_training/models"
    module_name = "setup_model"
    setup_model = localdir_modulespec(module_name, file_path)

    # get model function and call it to get the keras model
    model_function = setup_model.get_model_function(hparams["model_name"])
    net = model_function(input_shape, n_classes, hparams)

    return net


def load_and_override_hparams(saved_model_path, additional_units=0, **kwargs):
    if additional_units:
        with open(
            f"{saved_model_path}/hparams_addReadoutUnits.pickle", "rb"
        ) as f:
            hparams = pickle.load(f)
    else:
        with open(f"{saved_model_path}/hparams.pickle", "rb") as f:
            hparams = pickle.load(f)
    hparams["saved_model_path"] = saved_model_path
    if hparams["rsync_dataset_to"] != "no_rsync":
        len_to_remove = len(hparams["rsync_dataset_to"])
        hparams["dataset"] = (
            "/rds/project/rds-uqfOSPIEJ54" + hparams["dataset"][len_to_remove:]
        )
        print(
            f"Training used rsync_dataset_to and therefore overwrote the dataset path in loaded hparams. "
            f'Looking for dataset in {hparams["dataset"]}. If it is not there, change manually in analysis.py'
        )
    for k, v in kwargs.items():
        hparams[k] = v
    return hparams


def load_model_from_path(
    saved_model_path,
    epoch_to_load,
    n_classes=None,
    test_mode=False,
    print_summary=False,
    hparams=None,
    static_to_seq_input=False,
):
    if hparams is None:
        print(f"Loading model from {saved_model_path}.")

        with open(f"{saved_model_path}/hparams.pickle", "rb") as f:
            hparams = pickle.load(f)

    if static_to_seq_input:
        print(
            f"Loading a network trained on static inputs, transforming it into a network taking sequential input"
            f'as many timesteps as the network has recurrent timesteps ({hparams["n_recurrent_steps"]}).'
        )
        hparams["sequence_input"] = True
        hparams["n_identical_frames"] = 1

    print(f"Setting test_mode={test_mode}.")
    hparams["test_mode"] = test_mode

    # find n_output_neurons (depends on the kind of network, e.g. semantic_loss always has 300 outputs)
    if n_classes is None:
        # get n_classes from dataset if not passed as argument
        n_classes = get_n_classes(hparams=hparams)

    # original_nclasses = hparams['finetune_original_nclasses'] if hparams['finetune'] else n_classes

    print("\ncreating model...")
    net = get_model(hparams, n_classes, saved_model_path=saved_model_path)

    if hparams["test_mode"]:
        for layer in net.layers:
            layer.trainable = False

    print("\nloading weights...")
    if epoch_to_load == 0:
        weights_filename = "model_weights_init.h5"
    else:
        weights_filename = f"ckpt_ep{epoch_to_load:03d}.h5"
    net.load_weights(
        os.path.join(
            os.path.join(
                saved_model_path, "training_checkpoints", weights_filename
            )
        )
    )

    if print_summary:
        net.summary()

    metric_dict = {}
    for layer in net.output_names:
        metric_dict[layer] = [
            tf.keras.metrics.categorical_accuracy,
            tf.keras.metrics.top_k_categorical_accuracy,
        ]
    net.compile(metrics=metric_dict)

    return net, hparams


def convert_customsavedmodel_to_tfsavedmodel(saved_model_path, epoch_to_load):
    net, hparams = load_model_from_path(saved_model_path, epoch_to_load)
    net.save(
        os.path.join(
            saved_model_path, f"{hparams['model_name']}_ep{epoch_to_load:03d}"
        )
    )


def get_activities_model_single_layer(net, readout_layer_name):
    # make keras model to collect layer activities
    found = False
    for this_layer in net.layers:
        if readout_layer_name.lower() in this_layer.name.lower():
            if (
                "10" in this_layer.name.lower()
                and "10" not in readout_layer_name
            ):
                # to cope with the fact that 1 is in 10
                continue
            else:
                readout_layer = this_layer.output
                found = True
    if not found:
        raise Exception(
            f"Requested layer not found.\nNetwork layers: {[lin.name for lin in net.layers]}\nRequested layer: {readout_layer_name}"
        )

    activities_model = tf.keras.Model(
        inputs=net.input, outputs=readout_layer, name="activities_model"
    )

    return activities_model


def get_activities_model(net, n_layers, hparams, pre_post_norm="post"):
    timesteps = max(
        hparams["n_recurrent_steps"], 1
    )  # because hparams['n_recurrent_steps'] = 0 for ff nets

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
        raise Exception(
            f'hparams["norm_type"] = {hparams["norm_type"]} is not understood in get_activities_model'
        )

    # make keras model to collect layer activities
    readout_layer_names = [
        [None] * timesteps for _ in range(n_layers + 1)
    ]  # list of lists [n_layers+output][hparams['timesteps']]
    readout_layer_shapes = [[None] * timesteps for _ in range(n_layers + 1)]
    readout_layers = [[None] * timesteps for _ in range(n_layers + 1)]
    for layer_id in range(n_layers):
        for this_layer in net.layers:
            for t in range(timesteps):
                n_digits = (
                    2 if t >= 10 else 1
                )  # to deal with '11' vs. '1' in the layer name
                if pre_post_norm == "pre" or hparams["norm_type"] == "no_norm":
                    if (
                        f'{hparams["activation"]}_layer_{layer_id}_time_{t}'
                        in this_layer.name.lower()
                        and (this_layer.name.lower()[-(n_digits + 1)]) == "_"
                    ):
                        readout_layers[layer_id][t] = this_layer.output
                        readout_layer_names[layer_id][
                            t
                        ] = this_layer.name.lower()
                        readout_layer_shapes[layer_id][
                            t
                        ] = this_layer.output.shape
                elif pre_post_norm == "post":
                    if (
                        f"{norm_layer_name}_layer_{layer_id}_time_{t}"
                        in this_layer.name.lower()
                        and (this_layer.name.lower()[-(n_digits + 1)]) == "_"
                    ):
                        readout_layers[layer_id][t] = this_layer.output
                        readout_layer_names[layer_id][
                            t
                        ] = this_layer.name.lower()
                        readout_layer_shapes[layer_id][
                            t
                        ] = this_layer.output.shape
                else:
                    raise Exception(
                        'pre_post_norm not understood. use "pre" or "post"'
                    )

                if this_layer.name.lower() == f"output_time_{t}":
                    readout_layers[n_layers][t] = this_layer.output
                    readout_layer_names[n_layers][t] = this_layer.name.lower()
                    readout_layer_shapes[n_layers][t] = this_layer.output.shape

    activities_model = tf.keras.Model(
        inputs=net.input, outputs=readout_layers, name="activities_model"
    )

    return activities_model, readout_layer_names, readout_layer_shapes


def get_n_classes(
    hparams=None, dataset_path=None, dataset_subset=None, n_classes_manual=None
):
    if n_classes_manual is not None:
        print("returned manually specified n_classes")
        return n_classes_manual

    if dataset_path is None:
        print("n_classes using hparams dataset")
        dataset = hparams["dataset"]
    else:
        print(f"n_classes using specified dataset path: {dataset_path}")
        dataset = dataset_path

    with h5py.File(dataset, "r") as f:
        if dataset_subset is not None:
            print(f"n_classes using specified subset: {dataset_subset}")
            f = f[dataset_subset]
        if dataset_path is None:
            if hparams["embeddings_path"]:
                print("n_classes using hparams embeddings path")
                return 300  # you may need to change this depending on your embedding dimensionality
            elif hparams["embedding_target"]:
                print("n_classes using hparams embeddings target")
                return f["train"][hparams["target_dataset_name"]][0].shape[-1]
        return f["categories"][:].shape[0]
