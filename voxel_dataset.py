import tensorflow as tf
if __name__ == '__main__':
    tf.enable_eager_execution()
import numpy as np
import random

from os import listdir
from os.path import join, isfile

from visualization import plot_voxel, convert_to_sparse_voxel_grid

import pdb

def get_voxel_dataset(tffiles, batch_size=32):
    '''
    Given a list of TFRecord filenames, create a voxel dataset (assume data is in
    voxel partial/full format) and return it.
    '''

    dataset = tf.data.TFRecordDataset(tffiles)
    
    # Setup parsing of objects.
    voxel_feature_description = {
        'partial': tf.FixedLenFeature([], tf.string),
        'full': tf.FixedLenFeature([], tf.string),
    }

    def _parse_voxel_function(example_proto):
        voxel_example = tf.parse_single_example(example_proto, voxel_feature_description)

        partial = tf.reshape(tf.parse_tensor(voxel_example['partial'], out_type=tf.bool), (32,32,32,1))
        full = tf.reshape(tf.parse_tensor(voxel_example['full'], out_type=tf.bool), (32,32,32,1))

        partial_float = tf.cast(partial, dtype=tf.float32)
        full_float = tf.cast(full, dtype=tf.float32)        
        
        return partial_float, full_float

    dataset = dataset.map(_parse_voxel_function)
    #dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(500) # Shuffle buffer size is size of single TRFormat.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # Works w/ tf 1.9    
    dataset = dataset.prefetch(1)

    return dataset

if __name__ == '__main__':
    train_folder = '/dataspace/ICRA_Data/ReconstructionData/SDF_CF/Train'
    train_files = [join(train_folder, filename) for filename in listdir(train_folder) if ".tfrecord" in filename]
    
    dataset = get_voxel_dataset(train_files, batch_size=1)

    for x, y in dataset:
        plot_voxel(convert_to_sparse_voxel_grid(np.reshape(x, (32,32,32))), voxel_res=(32,32,32))
        plot_voxel(convert_to_sparse_voxel_grid(np.reshape(y, (32,32,32))), voxel_res=(32,32,32))        
    
