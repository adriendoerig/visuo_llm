import h5py, math
from .tf_dataset_helper_functions import *


class HDF5Sequence(tf.keras.utils.Sequence):

    def __init__(self, hparams, dataset_path, dataset, 
                 target_dataset_name, target_dataset_dtype, n_dataset_elements,
                 dataset_subset, no_labels_flag):
        self.hdf5_path = dataset_path
        self.dataset_subset = dataset_subset
        self.dataset = dataset  # str, train, test, val
        self.target_dataset_name = target_dataset_name
        self.target_dataset_dtype = target_dataset_dtype
        self.batch_size = hparams['batch_size']
        self.n_dataset_elements = n_dataset_elements
        self.indices = np.arange(n_dataset_elements)
        self.use_class_weights = hparams['calculate_class_weights'] if self.dataset == 'train' else False
        self.no_labels_flag = no_labels_flag

        self.load_dataset()
        if self.use_class_weights:
            self.maybe_load_class_weights()

        self.on_epoch_end()
        self.counter = 0

    def __len__(self):
        return self.n_dataset_elements//self.batch_size

    def __getitem__(self, idx):
        # get data from numpy array loaded in memory
        batch_images = self.images[self.indices[self.batch_size*idx:self.batch_size*(idx+1)]]

        if not self.no_labels_flag:
            batch_labels = self.labels[self.indices[self.batch_size*idx:self.batch_size*(idx+1)]]
            if self.use_class_weights:
                batch_sample_weights = np.array([self.class_weights[l] for l in batch_labels])
            else:
                batch_sample_weights = np.ones((self.batch_size))

        self.counter += 1
        if self.counter == self.__len__():
            self.on_epoch_end()

        if self.no_labels_flag:
            return batch_images
        else:
            return batch_images, {'output': batch_labels}, batch_sample_weights

    def on_epoch_end(self):
        self.counter = 0
        if self.dataset == 'train':
            self.shuffle_indices()

    def shuffle_indices(self):
        print('Shuffling dataset indices')
        np.random.shuffle(self.indices)

    def load_dataset(self):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if self.dataset_subset is None:
                print('loading full dataset as np.array')
                self.images = np.empty((self.n_dataset_elements,) + hdf5_file[self.dataset]['data'].shape[1:], dtype=np.uint8)
                hdf5_file[self.dataset]['data'].read_direct(self.images)
                if not self.no_labels_flag:
                    self.labels = np.empty((self.n_dataset_elements,) + hdf5_file[self.dataset][self.target_dataset_name].shape[1:], dtype=self.target_dataset_dtype)
                    hdf5_file[self.dataset][self.target_dataset_name].read_direct(self.labels)
            else:
                print(f'loading {self.dataset_subset} dataset as np.array')
                self.images = np.empty((self.n_dataset_elements,) + hdf5_file[self.dataset_subset][self.dataset]['data'].shape[1:], dtype=np.uint8)
                hdf5_file[self.dataset_subset][self.dataset]['data'].read_direct(self.images)
                if not self.no_labels_flag:
                    self.labels = np.empty((self.n_dataset_elements,) + hdf5_file[self.dataset_subset][self.dataset][self.target_dataset_name].shape[1:], dtype=self.target_dataset_dtype)
                    hdf5_file[self.dataset_subset][self.dataset][self.target_dataset_name].read_direct(self.labels)

    def maybe_load_class_weights(self):
        # dataset_path = self.hdf5_path.decode('utf-8')  # not sure exactly when this is needed. uncomment if you get weird error
        dataset_name = os.path.splitext(os.path.basename(os.path.normpath(self.hdf5_path)))[0]
        os.makedirs('./dataset_loader/class_weights', exist_ok=True)
        class_weights_path = './dataset_loader/class_weights/class_weights_' + dataset_name + '.npy'
        if os.path.exists(class_weights_path):
            print('Class weights found: ' + class_weights_path)
            self.class_weights = np.load(class_weights_path)
        else:
            print('Class weights not found: ' + class_weights_path)
            print('Computing class weights.')
            self.class_weights = self.calculate_class_weights()
            np.save(class_weights_path, self.class_weights)

    def calculate_class_weights(self):
        '''Calculates weights for each class in inverse proportion to number of images
        '''
        print('calculating class weights from HDF5, takes a few minutes for large datasets...')
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if self.dataset_subset is None:
                _, num_per_class = np.unique(hdf5_file[self.dataset]['labels'], return_counts=True)
            else:
                _, num_per_class = np.unique(hdf5_file[self.dataset_subset][self.dataset]['labels'], return_counts=True)
            inv_num_per_class = 1 / num_per_class
            class_weights = inv_num_per_class / np.sum(inv_num_per_class) * len(inv_num_per_class)
        print('finished loading class weights')
        return np.array(class_weights)


def get_dataset(hparams, dataset, dataset_path=None, dataset_subset=None, plot_generated_data=False):
    '''Make a tf.data.Dataset based on a python generator gen (here, a keras sequence).
       hparams: dict containing pipeline options
       dataset: str, "train", "val" or "test"
       dataset_subset: str, if not None, get data from hdf5_file[dataset_subset] instead of from hdf5_file root
       split_index & split_direction: int & str, if split_index>0, use only elements with "higher" or "lower" indices,
                (set by split_direction). Useful for splitting the dataset in and only using one part
       every_n_indices: int, if > 1, skip every_n_items (e.g. if 2, use items 0,2,4,...'''

    if dataset_path is None:
        # if no dataset_path is given, take the dataset path from hparams
        dataset_path = hparams['dataset']

    if 'simclr' in hparams['model_name']:
        # In some contexts, eg self-supervised learning, labels are not needed
        print('Dataset will not contain labels.')
        no_labels_flag = True
    else:
        no_labels_flag = False

    if dataset != 'train' or 'simclr' in hparams['model_name']:
        augment_data = False
    else:
        augment_data = True

    print(f'Making {dataset} dataset from {dataset_path}'+f' subset {dataset_subset}' if dataset_subset is not None else '')

    with h5py.File(dataset_path, "r") as hdf5_file:
        if dataset_subset is None:
            dataset_elements_shape = (hparams['batch_size'],) + hdf5_file[dataset]['data'][0].shape
            n_dataset_elements = math.ceil(hdf5_file[dataset]['labels'].shape[0])
        else:
            dataset_elements_shape = (hparams['batch_size'],) + hdf5_file[dataset_subset][dataset]['data'][0].shape
            n_dataset_elements = math.ceil(hdf5_file[dataset_subset][dataset]['labels'].shape[0])

        if no_labels_flag:
            target_dataset_name = None
            target_dataset_dtype = None
            tf_target_dataset_dtype = None
            label_shape = None
        else:
            if hparams['embedding_target']:
                target_dataset_name = hparams['target_dataset_name']  # you could use "embeddings" or any name you gave to the embeddigns in your .h5 file
                target_dataset_dtype = 'float32'
                tf_target_dataset_dtype = tf.float32
                label_shape = (hparams['batch_size'],) + hdf5_file[dataset][target_dataset_name][0].shape
            else:
                target_dataset_name = hparams['target_dataset_name']
                target_dataset_dtype = 'int32'
                tf_target_dataset_dtype = tf.int32
                label_shape = (hparams['batch_size'],)            

    # create generator
    seq = HDF5Sequence(hparams, dataset_path, dataset,
                       target_dataset_name, target_dataset_dtype, n_dataset_elements,
                       dataset_subset, no_labels_flag)
    data_iter = lambda: (s for s in seq)

    # create tf.data.Dataset
    if no_labels_flag:
        tf_dataset = (tf.data.Dataset.from_generator(data_iter, 
                                                    output_types=(tf.uint8), 
                                                    output_shapes=(dataset_elements_shape))
                                                    .map(lambda x: preprocess_batch(x, None, None, hparams, dataset_path, dataset_subset, no_labels=True), num_parallel_calls=tf.data.AUTOTUNE)
                                                    .map(lambda x: augment_and_normalize(x, None, None, augment_data, hparams, no_labels=True), num_parallel_calls=tf.data.AUTOTUNE)
                                                    .prefetch(tf.data.AUTOTUNE))
    else:
        tf_dataset = (tf.data.Dataset.from_generator(data_iter,
                                                    output_types=(tf.uint8, {'output': tf_target_dataset_dtype}, tf.float32),
                                                    output_shapes=(dataset_elements_shape, {'output': label_shape}, (hparams['batch_size'])))
                                                    .map(lambda x, y, sw: preprocess_batch(x, y, sw, hparams, dataset_path, dataset_subset), num_parallel_calls=tf.data.AUTOTUNE)
                                                    .map(lambda x, y, sw: augment_and_normalize(x, y, sw, augment_data, hparams), num_parallel_calls=tf.data.AUTOTUNE)
                                                    .prefetch(tf.data.AUTOTUNE))
    
    # Checks
    if plot_generated_data:
        # assess_data_generation_speed(tf_dataset)
        # below is good for looking sparsely at a large dataset
        # plot_generated_images(tf_dataset, hparams, dataset, dataset_path, max_n_imgs=50, name=hparams['model_name_suffix'],
        #                       fixation_heatmaps_path=fixation_heatmaps_path, dataset_subset=dataset_subset)
        # below is good for looking at a few images from a small dataset
        plot_generated_images(tf_dataset, hparams, dataset, dataset_path, max_n_imgs=50, imgs_per_batch=10, 
                              dataset_subset=dataset_subset)

    return tf_dataset
