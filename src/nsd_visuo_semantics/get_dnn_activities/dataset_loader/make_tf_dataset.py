import math
import os

import h5py
import numpy as np
import tensorflow as tf

from .tf_dataset_helper_functions import (
    augment,
    plot_generated_images,
    preprocess_batch,
)


class HDF5Sequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        hparams,
        dataset_path,
        fixation_heatmaps_path,
        heatmap_subset,
        dataset,
        target_dataset_name,
        target_dataset_dtype,
        first_element,
        last_element,
        n_dataset_elements,
        dataset_subset,
        every_n_indices,
    ):
        self.hdf5_path = dataset_path
        self.fixation_heatmaps_path = fixation_heatmaps_path
        self.heatmap_subset = heatmap_subset
        self.dataset_subset = dataset_subset
        self.dataset = dataset  # str, train, test, val
        self.sequence_input = hparams["sequence_input"]
        self.target_dataset_name = target_dataset_name
        self.target_dataset_dtype = target_dataset_dtype
        self.batch_size = hparams["batch_size"]
        self.first_element = first_element
        self.last_element = last_element
        self.every_n_indices = every_n_indices
        self.n_dataset_elements = n_dataset_elements
        self.indices = np.arange(n_dataset_elements)
        self.use_class_weights = (
            hparams["calculate_class_weights"]
            if self.dataset == "train"
            else False
        )

        assert (
            not self.sequence_input
        ), "sequence_input not implemented for keras utils pipeline"

        self.load_dataset()
        self.maybe_load_class_weights()
        self.maybe_load_fix_heatmaps()

        self.on_epoch_end()
        self.counter = 0

    def __len__(self):
        return self.n_dataset_elements // self.batch_size

    def __getitem__(self, idx):
        # get data from numpy array loaded in memory
        batch_images = self.images[
            self.indices[self.batch_size * idx : self.batch_size * (idx + 1)]
        ]
        batch_labels = self.labels[
            self.indices[self.batch_size * idx : self.batch_size * (idx + 1)]
        ]
        if self.fixation_heatmaps_path is None:
            batch_fix_heatmaps = np.ones_like(
                batch_images[:, :, :, 0]
            )  # dummy
        else:
            batch_fix_heatmaps = self.fix_heatmaps[
                self.indices[
                    self.batch_size * idx : self.batch_size * (idx + 1)
                ]
            ]

        if self.use_class_weights:
            batch_sample_weights = np.array(
                [self.class_weights[lin] for lin in batch_labels]
            )
        else:
            batch_sample_weights = np.ones(self.batch_size)

        self.counter += 1
        if self.counter == self.__len__():
            self.on_epoch_end()

        return (
            batch_images,
            {"output": batch_labels},
            batch_sample_weights,
            batch_fix_heatmaps,
        )

    def on_epoch_end(self):
        self.counter = 0
        if self.dataset == "train":
            self.shuffle_indices()

    def shuffle_indices(self):
        print("Shuffling dataset indices")
        np.random.shuffle(self.indices)

    def load_dataset(self):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if self.dataset_subset is None:
                print("loading full dataset as np.array")
                self.images = np.empty(
                    (self.n_dataset_elements,)
                    + hdf5_file[self.dataset]["data"].shape[1:],
                    dtype=np.uint8,
                )
                self.labels = np.empty(
                    (self.n_dataset_elements,)
                    + hdf5_file[self.dataset][self.target_dataset_name].shape[
                        1:
                    ],
                    dtype=self.target_dataset_dtype,
                )
                if (
                    self.first_element > 0
                    or self.last_element
                    < hdf5_file[self.dataset]["data"].shape[0]
                    or self.every_n_indices > 1
                ):
                    # slower but more flexible
                    hdf5_file[self.dataset]["data"].read_direct(
                        self.images,
                        np.s_[
                            self.first_element : self.last_element : self.every_n_indices
                        ],
                    )
                    hdf5_file[self.dataset][
                        self.target_dataset_name
                    ].read_direct(
                        self.labels,
                        np.s_[
                            self.first_element : self.last_element : self.every_n_indices
                        ],
                    )
                else:
                    # fastest method
                    hdf5_file[self.dataset]["data"].read_direct(self.images)
                    hdf5_file[self.dataset][
                        self.target_dataset_name
                    ].read_direct(self.labels)
            else:
                print(f"loading {self.dataset_subset} dataset as np.array")
                self.images = np.empty(
                    (self.n_dataset_elements,)
                    + hdf5_file[self.dataset_subset][self.dataset][
                        "data"
                    ].shape[1:],
                    dtype=np.uint8,
                )
                self.labels = np.empty(
                    (self.n_dataset_elements,)
                    + hdf5_file[self.dataset_subset][self.dataset][
                        self.target_dataset_name
                    ].shape[1:],
                    dtype=self.target_dataset_dtype,
                )
                if (
                    self.first_element > 0
                    or self.last_element
                    < hdf5_file[self.dataset_subset][self.dataset][
                        "data"
                    ].shape[0]
                    or self.every_n_indices > 1
                ):
                    # slower but more flexible
                    hdf5_file[self.dataset_subset][self.dataset][
                        "data"
                    ].read_direct(
                        self.images,
                        np.s_[
                            self.first_element : self.last_element : self.every_n_indices
                        ],
                    )
                    hdf5_file[self.dataset_subset][self.dataset][
                        self.target_dataset_name
                    ].read_direct(
                        self.labels,
                        np.s_[
                            self.first_element : self.last_element : self.every_n_indices
                        ],
                    )
                else:
                    # fastest method
                    hdf5_file[self.dataset_subset][self.dataset][
                        "data"
                    ].read_direct(self.images)
                    hdf5_file[self.dataset_subset][self.dataset][
                        self.target_dataset_name
                    ].read_direct(self.labels)

    def maybe_load_class_weights(self):
        if self.use_class_weights:
            # dataset_path = self.hdf5_path.decode('utf-8')  # not sure exactly when this is needed. uncomment if you get weird error
            dataset_name = os.path.splitext(
                os.path.basename(os.path.normpath(self.hdf5_path))
            )[0]
            os.makedirs("./dataset_loader/class_weights", exist_ok=True)
            class_weights_path = (
                "./dataset_loader/class_weights/class_weights_"
                + dataset_name
                + ".npy"
            )
            if os.path.exists(class_weights_path):
                print("Class weights found: " + class_weights_path)
                self.class_weights = np.load(class_weights_path)
            else:
                print("Class weights not found: " + class_weights_path)
                print("Computing class weights.")
                self.class_weights = self.calculate_class_weights()
                np.save(class_weights_path, self.class_weights)

    def maybe_load_fix_heatmaps(self):
        if self.fixation_heatmaps_path is not None:
            print(
                f"Loading fixation heatmaps from {self.fixation_heatmaps_path}"
            )
            with h5py.File(self.fixation_heatmaps_path, "r") as fh_file:
                # self.fix_heatmaps = np.empty((self.n_dataset_elements,)+fh_file[self.dataset].shape[1:], dtype=np.float16)
                if self.dataset_subset is None:
                    self.fix_heatmaps = np.empty(
                        (self.n_dataset_elements,)
                        + fh_file[self.dataset].shape[1:],
                        dtype=np.float16,
                    )
                    if (
                        self.first_element > 0
                        or self.last_element < fh_file[self.dataset].shape[0]
                        or self.every_n_indices > 1
                    ):
                        # slower but more flexible
                        fh_file[self.dataset].read_direct(
                            self.fix_heatmaps,
                            np.s_[
                                self.first_element : self.last_element : self.every_n_indices
                            ],
                        )
                    else:
                        # fastest method
                        fh_file[self.dataset].read_direct(self.fix_heatmaps)
                else:
                    self.fix_heatmaps = np.empty(
                        (self.n_dataset_elements,)
                        + fh_file[self.heatmap_subset][self.dataset].shape[1:],
                        dtype=np.float16,
                    )
                    if (
                        self.first_element > 0
                        or self.last_element
                        < fh_file[self.dataset_subset][self.dataset].shape[0]
                        or self.every_n_indices > 1
                    ):
                        # slower but more flexible
                        fh_file[self.dataset_subset][self.dataset].read_direct(
                            self.fix_heatmaps,
                            np.s_[
                                self.first_element : self.last_element : self.every_n_indices
                            ],
                        )
                    else:
                        # fastest method
                        fh_file[self.dataset_subset][self.dataset].read_direct(
                            self.fix_heatmaps
                        )

    def calculate_class_weights(self):
        """Calculates weights for each class in inverse proportion to number of images"""
        print(
            "calculating class weights from HDF5, takes a few minutes for large datasets..."
        )
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if self.dataset_subset is None:
                _, num_per_class = np.unique(
                    hdf5_file[self.dataset]["labels"], return_counts=True
                )
            else:
                _, num_per_class = np.unique(
                    hdf5_file[self.dataset_subset][self.dataset]["labels"],
                    return_counts=True,
                )
            inv_num_per_class = 1 / num_per_class
            class_weights = (
                inv_num_per_class
                / np.sum(inv_num_per_class)
                * len(inv_num_per_class)
            )
        print("finished loading class weights")
        return np.array(class_weights)


def get_dataset(
    hparams,
    dataset,
    dataset_path=None,
    fixation_heatmaps_path=None,
    heatmap_subset=None,
    dataset_subset=None,
    begin_index=0,
    end_index=-1,
    every_n_indices=1,
    plot_generated_data=False,
):
    """Make a tf.data.Dataset based on a python generator gen (here, a keras sequence).
    hparams: dict containing pipeline options
    dataset: str, "train", "val" or "test"
    dataset_subset: str, if not None, get data from hdf5_file[dataset_subset] instead of from hdf5_file root
    split_index & split_direction: int & str, if split_index>0, use only elements with "higher" or "lower" indices,
             (set by split_direction). Useful for splitting the dataset in and only using one part
    every_n_indices: int, if > 1, skip every_n_items (e.g. if 2, use items 0,2,4,...
    """

    assert hparams["embeddings_path"] is None, (
        "embeddings_from_file not supported in preprocess_batch. "
        "Please include the embeddings in the .h5 and set embedding_target=True to "
        " use them instead of labels. Alternatively, you can implement a way to use "
        "embeddings_from_file below"
    )
    if dataset_path is None:
        # if no dataset_path is given, take the dataset path from hparams
        dataset_path = hparams["dataset"]
    if fixation_heatmaps_path is None:
        # if no dataset_path is given, take the dataset path from hparams
        fixation_heatmaps_path = hparams["fixation_heatmap_path"]

    print(
        f"Making {dataset} dataset from {dataset_path}"
        + f" subset {dataset_subset}"
        if dataset_subset is not None
        else ""
    )

    with h5py.File(dataset_path, "r") as hdf5_file:
        if dataset_subset is None:
            dataset_elements_shape = (hparams["batch_size"],) + hdf5_file[
                dataset
            ]["data"][0].shape
            n_dataset_elements = math.ceil(
                hdf5_file[dataset]["labels"].shape[0] / every_n_indices
            )
        else:
            dataset_elements_shape = (hparams["batch_size"],) + hdf5_file[
                dataset_subset
            ][dataset]["data"][0].shape
            n_dataset_elements = math.ceil(
                hdf5_file[dataset_subset][dataset]["labels"].shape[0]
                / every_n_indices
            )

        if hparams["embedding_target"]:
            target_dataset_name = hparams[
                "target_dataset_name"
            ]  # you could use "embeddings" or any name you gave to the embeddigns in your .h5 file
            target_dataset_dtype = "float32"
            tf_target_dataset_dtype = tf.float32
            label_shape = (hparams["batch_size"],) + hdf5_file[dataset][
                target_dataset_name
            ][0].shape
        else:
            target_dataset_name = hparams["target_dataset_name"]
            target_dataset_dtype = "int32"
            tf_target_dataset_dtype = tf.int32
            label_shape = (hparams["batch_size"],)

    # if requested, split the dataset at an element and only read one part of it (useful for example if the dataset
    # is ecoset + words/faces and you want to only access the ecoset or words/faces part)
    if begin_index > 0 or end_index != -1:  # is specified
        if end_index == -1:
            end_index = n_dataset_elements
        print(f"Only using elements from index {begin_index} to {end_index}")
        first_element = begin_index
        last_element = end_index
        n_dataset_elements = end_index - begin_index
    else:
        first_element = 0
        last_element = n_dataset_elements

    # create generator
    seq = HDF5Sequence(
        hparams,
        dataset_path,
        fixation_heatmaps_path,
        heatmap_subset,
        dataset,
        target_dataset_name,
        target_dataset_dtype,
        first_element,
        last_element,
        n_dataset_elements,
        dataset_subset,
        every_n_indices,
    )

    def data_iter():
        yield from seq

    # create tf.data.Dataset
    # if we are not dealing with sequences of inputs, a single image and label are needed per "item"
    if not hparams["sequence_input"]:
        tf_dataset = (
            tf.data.Dataset.from_generator(
                data_iter(),
                output_types=(
                    tf.uint8,
                    {"output": tf_target_dataset_dtype},
                    tf.float32,
                    tf.float16,
                ),
                output_shapes=(
                    dataset_elements_shape,
                    {"output": label_shape},
                    (hparams["batch_size"]),
                    dataset_elements_shape[:-1],
                ),
            )
            .map(
                lambda x, y, sw, fh: preprocess_batch(
                    x,
                    y,
                    sw,
                    hparams,
                    dataset_path,
                    dataset_subset,
                    fixation_heatmap=fh,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda x, y, sw: augment(
                    x,
                    y,
                    sw,
                    augment=True if dataset == "train" else False,
                    hparams=hparams,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
    # if we ARE dealing with sequences of inputs, a each input is a sequence of inputs and labels
    else:
        raise NotImplementedError(
            "Generator for sequences of inputs not implemented. Get inspired from older versions of the pipeline"
        )

    # Checks
    if plot_generated_data:
        # assess_data_generation_speed(tf_dataset)
        # below is good for looking sparsely at a large dataset
        # plot_generated_images(tf_dataset, hparams, dataset, dataset_path, max_n_imgs=50, name=hparams['model_name_suffix'],
        #                       fixation_heatmaps_path=fixation_heatmaps_path, dataset_subset=dataset_subset)
        # below is good for looking at a few images from a small dataset
        plot_generated_images(
            tf_dataset,
            hparams,
            dataset,
            dataset_path,
            max_n_imgs=50,
            imgs_per_batch=10,
            name=dataset_subset
            if dataset_subset is not None
            else "ecoset_faces",
            fixation_heatmaps_path=fixation_heatmaps_path,
            dataset_subset=dataset_subset,
        )

    return tf_dataset
