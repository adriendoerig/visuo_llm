import os

import h5py
import numpy as np
import tensorflow as tf

from ..task_helper_functions import get_n_classes


def preprocess_batch(
    data,
    labels,
    sample_weights,
    hparams,
    dataset_path=None,
    dataset_subset=None,
    fixation_heatmap=None,
):
    # note: since we are using batches, we cannot use the flat_imgs datasets with different img sizes. All imgs are square and the same size.

    n_timesteps = max(
        hparams["n_recurrent_steps"], 1
    )  # to take care of feedforward case (where timesteps = 0)

    n_classes = get_n_classes(
        hparams=hparams,
        dataset_path=dataset_path,
        dataset_subset=dataset_subset,
    )

    # image preprocessing (i.e., reshape to rectangle, randomly crop to square, resize to hparams['image_size] and scale to [0,1])
    if not hparams["sequence_input"]:
        preprocessed_inputs = preprocess_batch_imgs(
            data, fixation_heatmap, hparams
        )
    else:
        raise NotImplementedError(
            "Sequences of inputs not implemented for preprocess_batch. Use img-wise version or implement batch-wise version."
        )

    # label formatting (depending on the network, we may have one or multiple outputs, and labels need to be formatted accordingly)
    if not hparams["sequence_input"]:
        if hparams["embedding_target"]:
            # "label" is returned by the dataset generator. this is an embedding if hparams['embedding_target'].
            formatted_labels = {
                f"output_time_{t}": labels["output"]
                for t in range(n_timesteps)
            }
        else:
            formatted_labels = {
                f"output_time_{t}": tf.one_hot(
                    labels["output"], depth=n_classes
                )
                for t in range(n_timesteps)
            }
    else:
        if hparams["embedding_target"]:
            print(
                "Please check in tf_dataset_helper_functions. I have not tested this case."
            )
            formatted_labels = {
                f"output_time_{t}": labels[f"output_time_{t}"]
                for t in range(n_timesteps)
            }
        else:
            if hparams["readout_times"] == []:
                formatted_labels = {
                    f"output_time_{t}": tf.one_hot(
                        labels[f"output_time_{t}"], depth=n_classes
                    )
                    for t in range(n_timesteps)
                }
            else:
                formatted_labels = {
                    f"output_time_{t}": tf.one_hot(
                        labels[f"output_time_{t}"], depth=n_classes
                    )
                    for t in range(len(hparams["readout_times"]))
                }

    return preprocessed_inputs, formatted_labels, sample_weights


def preprocess_batch_imgs(images, fixation_heatmap, hparams):
    """Pre-process image: reshape to rectangle if flat, crop, resize to desired shape and scale to 0,1"""

    img_size = images.shape[1:3]  # excluding color channels
    n_rows, n_cols = img_size[0], img_size[1]
    image_area = tf.cast(tf.math.reduce_prod(img_size), tf.float32)
    target_image_size = hparams["image_size"]

    if hparams["fixation_heatmap_path"] is None:
        max_crop_size = tf.cast(tf.reduce_min(img_size), tf.float32)
        min_crop_size = tf.math.minimum(
            max_crop_size, tf.math.ceil(tf.math.sqrt(image_area * 0.33))
        )  # crop must be at least 1/3 of the image area
        crop_sizes = tf.cast(
            tf.random.uniform(
                shape=[hparams["batch_size"]],
                minval=min_crop_size,
                maxval=max_crop_size,
            ),
            tf.int16,
        )

        images = tf.map_fn(
            fn=lambda inp: tf.image.resize(
                tf.image.random_crop(inp[0], [inp[1], inp[1], 3]),
                [target_image_size] * 2,
                antialias=True,
            ),
            elems=(images, crop_sizes),
            fn_output_signature=tf.float32,
            name="batch_rnd_crops",
        )

    else:
        # raise NotImplementedError('fixation heatmaps not implemented for preprocess_batch_imgs. Use img-wise version or implement batch-wise version.')
        max_crop_size = tf.math.ceil(
            tf.math.sqrt(image_area * 0.5)
        )  # crop must be at most 1/2 of the image area
        min_crop_size = tf.math.ceil(
            tf.math.sqrt(image_area * 0.33)
        )  # crop must be at least 1/3 of the image area
        crop_sizes = tf.cast(
            tf.random.uniform(
                shape=[hparams["batch_size"]],
                minval=min_crop_size,
                maxval=max_crop_size,
            ),
            tf.int32,
        )
        flat_heatmaps = tf.reshape(
            fixation_heatmap, [hparams["batch_size"], -1]
        )  # tf.random.categorical expects [b_s, dims] [HEATMAP SHOULD BE LOG_PROBS]
        sampled_locs_1d = tf.map_fn(
            fn=lambda inp: tf.random.categorical(
                tf.expand_dims(inp, 0), num_samples=1
            ),
            elems=flat_heatmaps,
            fn_output_signature=tf.int64,
            name="sampled_1d_fixlocs",
        )
        sampled_rows = tf.cast(
            tf.squeeze(sampled_locs_1d // n_cols), tf.int32
        )  # convert sampled location from 1d to 2d
        sampled_cols = tf.cast(
            tf.squeeze(tf.experimental.numpy.mod(sampled_locs_1d, n_cols)),
            tf.int32,
        )
        crop_topleft_rows = (
            sampled_rows - crop_sizes // 2
        )  # tf.image.crop_to_bounding_box wants the topleft corner of the bounding box
        crop_topleft_cols = sampled_cols - crop_sizes // 2
        crop_topleft_rows = tf.cast(
            tf.minimum(tf.maximum(crop_topleft_rows, 0), n_rows - crop_sizes),
            tf.int32,
        )  # make sure the crop does not go beyond the image
        crop_topleft_cols = tf.cast(
            tf.minimum(tf.maximum(crop_topleft_cols, 0), n_cols - crop_sizes),
            tf.int32,
        )

        images = tf.map_fn(
            fn=lambda inp: tf.image.resize(
                tf.image.crop_to_bounding_box(
                    inp[0], inp[1], inp[2], inp[3], inp[3]
                ),
                [target_image_size] * 2,
                antialias=True,
            ),
            elems=(images, crop_topleft_rows, crop_topleft_cols, crop_sizes),
            fn_output_signature=tf.float32,
            name="batch_cropped_imgs",
        )

    images = tf.keras.layers.Rescaling(scale=1 / 255.0, dtype=tf.float32)(
        images
    )  # REMOVE IF if image is already [0, 1]

    return images


def augment(images, labels, sample_weights, augment, hparams):
    random_seed = 170591  # only needed foro tf.image.random, can be removed once all operations are supported by keras layers
    # augmentation
    if augment:
        images = tf.keras.layers.RandomFlip(
            "horizontal", dtype=tf.float32, seed=random_seed
        )(images)
        # images = tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255), seed=random_seed)(images)  # only in tf 2.11
        images = tf.image.random_brightness(
            images, max_delta=32.0 / 255.0, seed=random_seed
        )
        print(
            "Warning: using tf.image.random_saturation since this does not yet exist as a keras layer. Recommended practice is to use keras layers when available"
        )
        images = tf.image.random_saturation(
            images, lower=0.5, upper=1.5, seed=random_seed
        )
        images = tf.keras.layers.RandomContrast(
            factor=0.5, dtype=tf.float32, seed=random_seed
        )(images)

    # normalization
    images = normalize(images, hparams["image_normalization"])

    return images, labels, sample_weights


def normalize(image, img_normalization):
    """normalize an image
    img_normalization: str or None, '[-1,1]' normalizes to [-1,1], 'z_scoring' normalizes to mean=0, std=1
    """

    image = tf.clip_by_value(image, 0.0, 1.0)
    if img_normalization == "z_scoring":
        print("z-scoring each image")
        image = tf.image.per_image_standardization(image)
    elif img_normalization == "[-1,1]":
        print("Normalizing to [-1,1]")
        image = tf.keras.layers.Rescaling(
            scale=2, offset=-1, dtype=tf.float32
        )(
            image
        )  # assumes we start from a [0,1] image
    elif img_normalization is None:
        print("No normalization")
    else:
        raise Exception(
            "Please use '[-1,1]' or 'z_scoring' for the normalization argument"
        )

    return image


def assess_data_generation_speed(tf_dataset):
    import time

    print("Assessing generator compute time")
    for epoch in range(2):
        start_time = time.perf_counter()
        counter = 0
        for i, x in enumerate(tf_dataset):
            # print(x[0])
            # print(i)
            counter += 1
        print(
            f"Epoch {epoch} -- batches produced: {counter} -- Time passed = {time.perf_counter()-start_time}"
        )
    raise  # stop the script here


def plot_generated_images(
    tf_dataset,
    hparams,
    dataset,
    dataset_path,
    fixation_heatmaps_path=None,
    dataset_subset=None,
    name="",
    n_epochs_to_show=1,
    max_n_imgs=10000,
    imgs_per_batch=1,
):
    import matplotlib.pyplot as plt

    # from dataset_creation.dataset_functions import format_rgb_img
    os.makedirs("./tf_generated_images", exist_ok=True)
    counter = 0
    img_counter = 0
    for epoch in range(n_epochs_to_show):
        epoch_counter = 0
        print(f"visualizing dataset - epoch {epoch}")
        for x in tf_dataset:
            # x[0] -> input, x[1] -> label, x[2] -> sample weight
            # print(x[0].shape, x[1]['output_time_0'].shape, x[2])
            for i in range(hparams["batch_size"]):
                if i % (hparams["batch_size"] // imgs_per_batch) == 0:
                    if fixation_heatmaps_path is None:
                        if x[0].ndim == 4:  # a batch of simgle images
                            plt.figure()
                            # plt.imshow(format_rgb_img(x[0][i]))  # this rescales the image prior to plotting it to match the imshow format
                            plt.imshow(
                                x[0][i]
                            )  # this rescales the image prior to plotting it to match the imshow format
                            try:
                                plt.title(
                                    f'l={np.argmax(x[1]["output_time_0"][i])}, sw={x[2][i]}, min: {np.min(x[0][i]):.2f}, max: {np.max(x[0][i]):.2f}'
                                )  # works with ff nets without semantic_loss, otherwise, you need to adapt or remove thiis line
                            except ValueError:
                                plt.title(
                                    f'l={np.argmax(x[1]["output_time_0"][i])}'
                                )  # in case of recurrence
                        elif x[0].ndim == 5:  # a batch of sequences of images
                            fig, ax = plt.subplots(
                                1, x[0].shape[1], squeeze=False
                            )
                            for t in range(x[0].shape[1]):
                                # ax[0][t].imshow(format_rgb_img(x[0][i,t]))  # this rescales the image prior to plotting it to match the imshow format
                                ax[0][t].imshow(x[0][i, t])
                                ax[0][t].set_title(
                                    np.argmax(x[1][f"output_time_{t}"][i])
                                )
                                ax[0][t].set_axis_off()
                        else:
                            raise Exception(
                                f"Generated input should have 4 or 5 dimensions (batches of images or of image sequences) -- found {x[0].ndim} dimensions."
                            )
                    else:
                        if x[0].ndim == 4:  # a batch of simgle images
                            fig, ax = plt.subplots(1, 3, squeeze=False)
                            ax[0][0].imshow(x[0][i])
                            ax[0][0].set_title(
                                f"min: {np.min(x[0][i]):.2f}, max: {np.max(x[0][i]):.2f}"
                            )
                            print(i, epoch_counter)
                            # NOTE: IF YOU USED SHUFFLING WHEN CREATING THE DATASET; THE PLOTTED RAW DATASET IMG WILL NOT MATCH THE X IMAGE
                            with h5py.File(dataset_path, "r") as raw_dataset:
                                if dataset_subset is None:
                                    ax[0][1].imshow(
                                        raw_dataset[dataset]["data"][
                                            epoch_counter
                                        ]
                                    )
                                else:
                                    ax[0][1].imshow(
                                        raw_dataset[dataset_subset][dataset][
                                            "data"
                                        ][epoch_counter]
                                    )
                            with h5py.File(
                                fixation_heatmaps_path, "r"
                            ) as fix_heatmap_dataset:
                                if dataset_subset is None:
                                    ax[0][2].imshow(
                                        fix_heatmap_dataset[dataset][
                                            epoch_counter
                                        ]
                                    )
                                else:
                                    ax[0][2].imshow(
                                        fix_heatmap_dataset[dataset_subset][
                                            dataset
                                        ][epoch_counter]
                                    )
                            try:
                                ax[0][1].set_title(
                                    f'l={np.argmax(x[1]["output"][i])}, sw={x[2][i]}'
                                )  # works with ff nets without semantic_loss, otherwise, you need to adapt or remove thiis line
                            except ValueError:
                                ax[0][1].set_title(
                                    f'l={np.argmax(x[1]["output_time_0"][i])}'
                                )  # in case of recurrence
                        elif x[0].ndim == 5:  # a batch of sequences of images
                            raise NotImplementedError(
                                "heatmaps not implemented for sequences of inputs"
                            )
                        else:
                            raise Exception(
                                f"Generated input should have 4 or 5 dimensions (batches of images or of image sequences) -- found {x[0].ndim} dimensions."
                            )
                    # plt.show()
                    plt.savefig(
                        f"./tf_generated_images/{name}_{dataset}_{counter}.png"
                    )
                    plt.close()
                    if img_counter >= max_n_imgs:
                        return
                    img_counter += 1
                epoch_counter += 1
                counter += 1
    # stop the script here in debugger
    # import pdb
    # pdb.set_trace()
