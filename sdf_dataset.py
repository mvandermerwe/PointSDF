# Handle SDF Dataset.

import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()

import numpy as np
import h5py
import os
import pdb
import sys

def get_sdf_dataset(tffiles, batch_size=32, sdf_count=128):
    '''
    Given a list of TFRecord filenames, create a SDF dataset (assume data is in
    point cloud, xyzs, SDFs format) and return it.
    '''

    dataset = tf.data.TFRecordDataset(tffiles)
    
    # Setup parsing of objects.
    sdf_feature_description = {
        'point_clouds': tf.FixedLenFeature([], tf.string),
        'xyzs': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.string),
    }

    def _parse_sdf_function(example_proto):
        sdf_example = tf.parse_single_example(example_proto, sdf_feature_description)

        point_clouds = tf.parse_tensor(sdf_example['point_clouds'], out_type=tf.float32)
        xyzs = tf.parse_tensor(sdf_example['xyzs'], out_type=tf.float32)
        labels = tf.parse_tensor(sdf_example['labels'], out_type=tf.float32)

        # Downsample SDF points randomly.
        idxs = tf.range(tf.shape(xyzs)[0])
        ridxs = tf.random_shuffle(idxs)[:sdf_count]

        # Important to use same indices on each in order to align labels properly.
        xyzs = tf.gather(xyzs, ridxs, axis=0)
        labels = tf.reshape(tf.gather(labels, ridxs, axis=0), (sdf_count,1))
        
        return point_clouds, xyzs, labels

    dataset = dataset.map(_parse_sdf_function, num_parallel_calls=4)
    dataset = dataset.shuffle(3000)
    #dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # Works w/ tf 1.9
    dataset = dataset.prefetch(64)

    return dataset

def get_point_clouds(views, pc_file):
    '''
    For an entire set of views (i.e., a batch of views). Get and return
    the point clouds.
    '''

    pc_clouds = list(map(lambda view: np.reshape(get_point_cloud(view, pc_file), (1,-1,3)), views))
    return np.concatenate(pc_clouds, axis=0)

def get_point_cloud(view, pc_file):
    '''
    Read in the point cloud for the requested view from the point cloud h5 file.
    '''

    with h5py.File(pc_file, 'r') as f:
        point_cloud = f[view][:]
        return point_cloud

def get_voxels(views, voxel_file):
    voxels = list(map(lambda view: np.reshape(get_voxel(view, voxel_file), (1,32,32,32,1)), views))
    return np.concatenate(voxels, axis=0)
    
def get_voxel(view, voxel_file):
    '''
    Read in the voxel for the request view from the voxel h5 file.
    '''

    with h5py.File(voxel_file, 'r') as f:
        voxel = f[view][:]
        return voxel
