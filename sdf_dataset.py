# Handle SDF Dataset.

import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()

import numpy as np
import h5py
import os
import pdb
import sys
import pypcd
from sklearn.decomposition import PCA

from visualization import plot_3d_points

from object_frame import find_object_frame

_POINT_CLOUD_SIZE = 1000

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

def get_pcd(view, pcd_database, object_frame=False, verbose=False):
    '''
    Read in the point cloud for the requested view from its pcd file.
    '''

    pcd_filename = os.path.join(pcd_database, view + '.pcd')
    
    try:
        point_cloud = pypcd.PointCloud.from_path(pcd_filename)
    except IOError:
        print("File, " + str(pcd_filename) + " doesn't exist. Ignoring.")
        return None

    # Point cloud size.
    print("PC Size: ", len(point_cloud.pc_data))
    
    # Some objects end up filling whole screen - this is not useful to us.
    if len(point_cloud.pc_data) == 307200:
        return None

    obj_cloud = np.ones((len(point_cloud.pc_data), 3), dtype=np.float32)
    obj_cloud[:,0] = point_cloud.pc_data['x']
    obj_cloud[:,1] = point_cloud.pc_data['y']
    obj_cloud[:,2] = point_cloud.pc_data['z']

    # Get object frame for this point cloud.
    if object_frame:
        object_transform, world_frame_center = find_object_frame(obj_cloud, verbose)

        # Transform our point cloud. I.e. center and rotate to new frame.
        for i in range(obj_cloud.shape[0]):
            obj_cloud[i] = np.dot(object_transform, [obj_cloud[i][0], obj_cloud[i][1], obj_cloud[i][2], 1])[:3]

        centroid_diff = np.array([0.0,0.0,0.0])
    else:
        pca_operator = PCA(n_components=3, svd_solver='full')
        pca_operator.fit(obj_cloud)
        pca_centroid = np.matrix(pca_operator.mean_).T
        centroid = np.array([
            (np.amax(obj_cloud[:,0]) + np.amin(obj_cloud[:,0])) / 2,
            (np.amax(obj_cloud[:,1]) + np.amin(obj_cloud[:,1])) / 2,
            (np.amax(obj_cloud[:,2]) + np.amin(obj_cloud[:,2])) / 2,
        ])

        centroid_diff = np.array([
            float(pca_centroid[0]) - centroid[0],
            float(pca_centroid[1]) - centroid[1],
            float(pca_centroid[2]) - centroid[2],        
        ])

        # Center.
        obj_cloud[:,0] -= float(centroid[0])
        obj_cloud[:,1] -= float(centroid[1])
        obj_cloud[:,2] -= float(centroid[2])
        
    # Determine scaling size.
    max_dim = max(
        np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0]),
        np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1]),
        np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2]),
    )

    # Scale so that max dimension is about 1.
    scale = (1.0/1.03) / max_dim
    print("Scale, ", scale)

    # Scale every point.
    obj_cloud = obj_cloud * scale

    # Down/Up Sample cloud so everything has the same # of points.
    idxs = np.random.choice(obj_cloud.shape[0], size=_POINT_CLOUD_SIZE, replace=True)
    obj_cloud = obj_cloud[idxs,:]

    if verbose:
        plot_3d_points(obj_cloud)

    return obj_cloud, (max_dim * (1.03/1.0)), scale, centroid_diff

if __name__ == '__main__':
    train_folder = '/dataspace/ReconstructionData/SDF_Full_Fix/Train'
    train_files = [os.path.join(train_folder, filename) for filename in os.listdir(train_folder) if ".tfrecord" in filename]
    
    dataset = get_sdf_dataset(train_files, batch_size=1, sdf_count=1024)

    for x, y, z in dataset:
        point_cloud_points = x.numpy()[0]
        plot_3d_points(point_cloud_points)

        points_inside = y.numpy()[0][np.where(np.reshape(z.numpy()[0], (-1,)) <= 0)]
        plot_3d_points(points_inside)

        all_points = np.concatenate([point_cloud_points, points_inside], axis=0)

        pt_cld_col = np.zeros(point_cloud_points.shape[0])
        true_cld_col = np.zeros(points_inside.shape[0]) + 1
        col = np.concatenate([pt_cld_col, true_cld_col])
        
        plot_3d_points(all_points, col)
