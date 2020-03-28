#!/usr/bin/env python

import h5py
import numpy as np
import trimesh
import pypcd
import h5py
import os
import pdb
import tensorflow as tf
tf.enable_eager_execution()

from show_voxel import *
from voxelize_point_cloud import point_cloud_to_voxel
from object_frame import find_object_frame
from generate_view_splits import get_view_splits

# Generate point cloud sample for partial view and voxel for full view, based on PCD files and world poses
# for each object and store into TFRecords to help effective training.

# Hyperparameters for TRFormat Generation.
_MESH_DATABASE = '/home/markvandermerwe/Meshes'
_DEPTH_DATABASE = '/dataspace/PyrenderData/Depth'
_OBJECT_POSE_FILE = '/dataspace/PyrenderData/object_poses.hdf5'
_OBJECT_FRAME = False # Create and align with an object frame via PCA.
_CENTERED = True # Center the reconstruction. If false, we align the voxelization of the full mesh with the point cloud.
_TRAIN_FILE_DATABASE = '/dataspace/ReconstructionData/PointVoxel/Train'
_VALIDATION_FILE_DATABASE = '/dataspace/ReconstructionData/PointVoxel/Validation'
_TEST_FILE_DATABASE = '/dataspace/ReconstructionData/PointVoxel/Test'
_NUM_POSES_PER_OBJ = 200
_OBJECTS_PER_TFR = 1500 # Number of examples per TFRecord file. Should be about 100 MB per file.
_FLIP_THRESHOLD = 0.0001 # Threshold to flip randomly in resulting voxel grid.
_VOXEL_RES_PARTIAL = 26 # Voxel res for partial view. This is smaller than full so that the full object will fit.
_VOXEL_RES_FULL = 32 # Voxel res for full voxel.
_VERBOSE = False

_POINT_CLOUD_SIZE = 1000 # Always sample point cloud to this size.

def plot_3d_points(points, signed_distances=None):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=signed_distances)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('SDF')

    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)

    pyplot.show()

def in_bounds(x):
    return x >= 0 and x < _VOXEL_RES_FULL

def voxelize_mesh(mesh_filename, transform_matrix, voxel_size, verbose=False):
    '''
    For the given mesh and transformation matrix, generate a voxel grid
    with the mesh transformed accordingly.
    '''

    # Load mesh.
    mesh = trimesh.load(mesh_filename)

    # Rotate mesh based on the object world pose.
    mesh = mesh.apply_transform(transform_matrix)

    # Determine offsets.
    if _CENTERED:
        x_offset = 0
        y_offset = 0
        z_offset = 0
    else:
        x_offset = int(transform_matrix[0,3] / voxel_size)
        y_offset = int(transform_matrix[1,3] / voxel_size)
        z_offset = int(transform_matrix[2,3] / voxel_size)

    # If centered, we ignore the voxel size passed to us.
    if _CENTERED:
        lo, hi = mesh.bounds
        max_dim = max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2])
        voxel_size = max_dim / _VOXEL_RES_PARTIAL # Voxelize to smaller than actual size.
    
    # Voxelize.
    voxelized_mesh = mesh.voxelized(voxel_size)

    # Trimesh doesn't necessarily return a 26x26x26 so we need to center it.
    voxel_buf = (_VOXEL_RES_FULL - _VOXEL_RES_PARTIAL) // 2
    x_buf = ((_VOXEL_RES_PARTIAL - voxelized_mesh.shape[0]) // 2) + voxel_buf
    y_buf = ((_VOXEL_RES_PARTIAL - voxelized_mesh.shape[1]) // 2) + voxel_buf
    z_buf = ((_VOXEL_RES_PARTIAL - voxelized_mesh.shape[2]) // 2) + voxel_buf
    
    voxel = np.zeros((_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL), dtype=bool)
    for pt in voxelized_mesh.sparse_surface:
        x_, y_, z_ = pt
        # Get true coordinates w/ offset

        x_true = x_ + x_offset + x_buf
        y_true = y_ + y_offset + y_buf
        z_true = z_ + z_offset + z_buf

        if in_bounds(x_true) and in_bounds(y_true) and in_bounds(z_true):
            voxel[x_true, y_true, z_true] = True # Center in full voxel.

    # Visualize.
    if verbose:
        plot_voxel(convert_to_sparse_voxel_grid(voxel), voxel_res=(_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL))

    # Return voxel and size of voxel.
    return voxel

def get_object_rotation_matrix(object_pose_file, object_pose_name, verbose=False):
    '''
    For given object pose name, determine the world pose used (as homogenous transformation matrix).
    '''
    with h5py.File(object_pose_file, 'r') as f:
        pose_object = f[object_pose_name]
        pose = pose_object['rotation'][:]

        # Reshape to the matrix.
        matrix = [
            [pose[0], pose[1], pose[2]],
            [pose[3], pose[4], pose[5]],
            [pose[6], pose[7], pose[8]],
        ]
        
        if verbose:
            print matrix
            
        return matrix

def get_point_cloud_points(pcd_filename):
    '''
    Read in point cloud from pcd file and scale down.
    '''
    
    # Load point cloud.
    try:
        point_cloud = pypcd.PointCloud.from_path(pcd_filename)
    except IOError:
        print("File, " + str(pcd_filename) + " doesn't exist. Ignoring.")
        return None, None, None

    # Some objects end up filling whole screen - this is not useful to us.
    if len(point_cloud.pc_data) == 307200:
        return None, None, None

    # Get object frame for this point cloud.
    obj_cloud = np.ones((len(point_cloud.pc_data), 3), dtype=np.float32)
    obj_cloud[:,0] = point_cloud.pc_data['x']
    obj_cloud[:,1] = point_cloud.pc_data['y']
    obj_cloud[:,2] = point_cloud.pc_data['z']    
    
    # Determine scaling size.
    max_dim = max(
        np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0]),
        np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1]),
        np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2]),
    )

    centroid = np.array([
        (np.amax(obj_cloud[:,0]) + np.amin(obj_cloud[:,0])) / 2,
        (np.amax(obj_cloud[:,1]) + np.amin(obj_cloud[:,1])) / 2,
        (np.amax(obj_cloud[:,2]) + np.amin(obj_cloud[:,2])) / 2,
        ])
    
    # Center.
    obj_cloud[:,0] -= centroid[0]
    obj_cloud[:,1] -= centroid[1]
    obj_cloud[:,2] -= centroid[2]

    # Scale so that max dimension is about 1.
    scale = (1.0/1.03) / max_dim

    # Scale every point.
    for i in range(obj_cloud.shape[0]):
        obj_cloud[i] = obj_cloud[i] * scale

    # Down/Up Sample cloud so everything has the same # of points.
    idxs = np.random.choice(obj_cloud.shape[0], size=_POINT_CLOUD_SIZE, replace=True)
    obj_cloud = obj_cloud[idxs,:]

    return obj_cloud, scale, centroid
    
def _bytes_feature(value):
    '''
    Returns a bytes_list from a string / byte.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(partial_point_cloud, full_voxel):
    '''
    Serialize given partial, full voxel pair.
    '''

    feature = {
        'partial': _bytes_feature(tf.io.serialize_tensor(partial_point_cloud).numpy()),
        'full': _bytes_feature(tf.io.serialize_tensor(full_voxel).numpy()),
    }

    # Serialize w/ tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(partial_point_cloud, full_voxel):
    tf_string = tf.py_func(
        serialize_example,
        (partial_point_cloud, full_voxel),
        tf.string)
    return tf.reshape(tf_string, ())

def generate_tfrecord(partial_point_clouds, full_voxels, write_file):
    '''
    Create a tfrecord file for the given voxels.
    '''

    # Make dataset.
    voxel_dataset = tf.data.Dataset.from_tensor_slices((partial_point_clouds, full_voxels))

    # Map to serialization.
    serialized_voxel_dataset = voxel_dataset.map(tf_serialize_example)

    # Write to file.
    writer = tf.data.experimental.TFRecordWriter(write_file)
    writer.write(serialized_voxel_dataset)

def write_views_to_tfrecord(views, file_prepend, mesh_database, depth_database, object_pose_file, objects_per_tfr, verbose=False):
    '''
    Write the given list of views to a TFRecord file of the
    size desired. File prepend should be folder to and prefix for
    each folder. E.g., /home/name/data_folder/train_ will write train_0.tfrecord,train_1.tfrecord
    ,etc. to /home/name/data_folder.
    '''
    
    partials = []
    fulls = []
    file_num = 0

    for view in views:
        # Remove pose number to get mesh name.
        mesh_name = "_".join(view.split("_")[:-1])

        # Get partial view point cloud.
        point_cloud, scale, pc_centroid = get_point_cloud_points(os.path.join(depth_database, view+'.pcd'))

        if point_cloud is None:
            continue

        if verbose:
            plot_3d_points(point_cloud)
        
        # Get rotation in world frame to align full mesh w/ partial view.        
        rotation = get_object_rotation_matrix(object_pose_file, view, verbose)

        # Combine transformation of full mesh in world frame to get transform to object frame.
        centroid = [0, 0, -1.5]
        d_o_f_w = centroid - np.array(np.transpose(pc_centroid))[0]
        d_o_f_o = np.dot(rotation, d_o_f_w)

        object_frame_transform = np.eye(4)
        object_frame_transform[:3,:3] = np.dot(object_frame_transform[:3,:3], rotation)
        object_frame_transform[0,3] = d_o_f_o[0]
        object_frame_transform[1,3] = d_o_f_o[1]
        object_frame_transform[2,3] = d_o_f_o[2]        
        
        voxels = voxelize_mesh(os.path.join(mesh_database, mesh_name + '.stl'), object_frame_transform, None, verbose)

        if voxels is not None and point_cloud is not None:
            partials.append(point_cloud)
            fulls.append(voxels)

        # If we have enough objects, create a new TFRecord file.
        if len(partials) == objects_per_tfr:
            filename = file_prepend + str(file_num) + '.tfrecord'
            generate_tfrecord(partials, fulls, filename)

            print "Generated: ", filename
            
            partials = []
            fulls = []
            file_num += 1

    # Write whatever is left.
    if len(partials) != 0:
        filename = file_prepend + str(file_num) + '.tfrecord'
        generate_tfrecord(partials, fulls, filename)

def generate_tfrecord_datasets(mesh_database, depth_database, object_pose_file, objects_per_tfr, num_poses_per_obj, train_file_database, validation_file_database, test_file_database, verbose=False):
    '''
    Read full dataset and generate partial full training pairs.
    '''
    
    training_views, validation_views, test_views = get_view_splits()

    # Write to files.
    write_views_to_tfrecord(training_views, os.path.join(train_file_database, 'train_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, verbose)
    write_views_to_tfrecord(validation_views, os.path.join(validation_file_database, 'validation_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, verbose)
    write_views_to_tfrecord(test_views, os.path.join(test_file_database, 'test_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, verbose)

def run():
    # Generate away.
    generate_tfrecord_datasets(_MESH_DATABASE,
                               _DEPTH_DATABASE,
                               _OBJECT_POSE_FILE,
                               _OBJECTS_PER_TFR,
                               _NUM_POSES_PER_OBJ,
                               _TRAIN_FILE_DATABASE,
                               _VALIDATION_FILE_DATABASE,
                               _TEST_FILE_DATABASE,
                               _VERBOSE)
    
if __name__ == '__main__':
    run()
    
