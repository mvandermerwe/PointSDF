#!/usr/bin/env python

import h5py
import numpy as np
import trimesh
from trimesh.voxel import creation
import pypcd
import h5py
import os
import pdb
from tqdm import tqdm
import tensorflow as tf
if __name__ == '__main__':
    tf.enable_eager_execution()

from show_voxel import *
from voxelize_point_cloud import point_cloud_to_voxel
from object_frame import find_object_frame
from generate_view_splits import get_view_splits
from generate_sdf import plot_3d_points
from object_cloud import process_object_cloud

# Generate voxels for partial and full view, based on PCD files and world poses
# for each object and store into TFRecords to help effective training.

# Hyperparameters for TRFormat Generation.
_MESH_DATABASE = '/home/markvandermerwe/Meshes'
_DEPTH_DATABASE = '/dataspace/ICRA_Data/PyrenderData/Depth'
_OBJECT_POSE_FILE = '/dataspace/ICRA_Data/PyrenderData/object_poses.hdf5'

_OBJECT_FRAME = False # Create and align with an object frame via PCA.
_CENTERED = False # Center the reconstruction. If false, we align the voxelization of the full mesh with the point cloud.

_TRAIN_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/Voxel_CF/Train'
_VALIDATION_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/Voxel_CF/Validation'
_TEST_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/Voxel_CF/Test'

_NUM_POSES_PER_OBJ = 200
_OBJECTS_PER_TFR = 1500 # Number of examples per TFRecord file. Should be about 100 MB per file.

_FLIP_THRESHOLD = 0.0001 # Threshold to flip randomly in resulting voxel grid.
_VOXEL_RES_PARTIAL = 26 # Voxel res for partial view. This is smaller than full so that the full object will fit.
_VOXEL_RES_FULL = 32 # Voxel res for full voxel.

_VERBOSE = False

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
        bounds = None
    else:
        x_min = -(voxel_size * (_VOXEL_RES_FULL))
        x_max = (voxel_size * (_VOXEL_RES_FULL))
        y_min = -(voxel_size * (_VOXEL_RES_FULL))
        y_max = (voxel_size * (_VOXEL_RES_FULL))
        z_min = -(voxel_size * (_VOXEL_RES_FULL))
        z_max = (voxel_size * (_VOXEL_RES_FULL))
        bounds = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
        #print bounds

    # If centered, we ignore the voxel size passed to us.
    if _CENTERED:
        lo, hi = mesh.bounds
        max_dim = max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2])
        voxel_size = max_dim / _VOXEL_RES_PARTIAL # Voxelize to smaller than actual size.
        
    # Voxelize.
    voxelized_mesh = creation.voxelize_binvox(mesh, pitch=voxel_size, bounds=bounds, exact=True)
    voxelized_mesh.fill()
    
    voxel = np.zeros((_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL), dtype=bool)
    for pt_ in voxelized_mesh.sparse_indices:
        x_, y_, z_ = pt_

        x_ = x_ - (_VOXEL_RES_FULL//2)
        y_ = y_ - (_VOXEL_RES_FULL//2)
        z_ = z_ - (_VOXEL_RES_FULL//2)

        if in_bounds(x_) and in_bounds(y_) and in_bounds(z_):
            voxel[x_, y_, z_] = True # Center in full voxel.

    # Visualize.
    if verbose:
        plot_voxel(convert_to_sparse_voxel_grid(voxel), voxel_res=(_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL))

    # Return voxel and size of voxel.
    return voxel

def voxelize_point_cloud(pcd_filename, noise=True, verbose=False):
    '''
    For given pcd file, determine an object frame and create a voxelization in 
    that frame.
    '''

    if verbose:
        print pcd_filename
    
    # Load point cloud.
    try:
        point_cloud = pypcd.PointCloud.from_path(pcd_filename)
    except IOError:
        print("File, " + str(pcd_filename) + " doesn't exist. Ignoring.")
        return None, None, None, None

    # Some objects end up filling whole screen - this is not useful to us.
    if len(point_cloud.pc_data) == 307200:
        return None, None, None, None
    
    # Get object frame for this point cloud.
    obj_cloud = np.ones((len(point_cloud.pc_data), 3), dtype=np.float32)
    obj_cloud[:,0] = point_cloud.pc_data['x']
    obj_cloud[:,1] = point_cloud.pc_data['y']
    obj_cloud[:,2] = point_cloud.pc_data['z']

    return_dict = process_object_cloud(obj_cloud, object_frame=_OBJECT_FRAME, voxelize=True, verbose=verbose)
    
    return return_dict['voxel'], return_dict['voxel_size'], return_dict['object_transform'], return_dict['centroid']

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

def _bytes_feature(value):
    '''
    Returns a bytes_list from a string / byte.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(partial_voxel, full_voxel):
    '''
    Serialize given partial, full voxel pair.
    '''

    feature = {
        'partial': _bytes_feature(tf.serialize_tensor(partial_voxel).numpy()),
        'full': _bytes_feature(tf.serialize_tensor(full_voxel).numpy()),
    }

    # Serialize w/ tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(partial_voxel, full_voxel):
    tf_string = tf.py_func(
        serialize_example,
        (partial_voxel, full_voxel),
        tf.string)
    return tf.reshape(tf_string, ())

def generate_tfrecord(partial_voxels, full_voxels, write_file):
    '''
    Create a tfrecord file for the given voxels.
    '''

    # Make dataset.
    voxel_dataset = tf.data.Dataset.from_tensor_slices((partial_voxels, full_voxels))

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

    for view in tqdm(views):
        # Remove pose number to get mesh name.
        mesh_name = "_".join(view.split("_")[:-1])

        # Get voxelization of partial view point cloud.
        partial_voxels, voxel_size, object_frame_transform, partial_view_center_world = voxelize_point_cloud(os.path.join(depth_database, view+'.pcd'), verbose)

        if partial_voxels is None:
            continue
        
        # Get rotation in world frame to align full mesh w/ partial view.        
        rotation = get_object_rotation_matrix(object_pose_file, view, verbose)

        # Combine transformation of full mesh in world frame to get transform to object frame.
        centroid = [0, 0, -1.5]
        d_o_f_w = centroid - partial_view_center_world
        d_o_f_o = np.dot(object_frame_transform[:3,:3], d_o_f_w)

        object_frame_transform[:3,:3] = np.dot(object_frame_transform[:3,:3], rotation)
        object_frame_transform[0,3] = d_o_f_o[0]
        object_frame_transform[1,3] = d_o_f_o[1]
        object_frame_transform[2,3] = d_o_f_o[2]
        
        voxels = voxelize_mesh(os.path.join(mesh_database, mesh_name + '.stl'), object_frame_transform, voxel_size, verbose)

        if voxels is not None and partial_voxels is not None:

            # Combined viz.
            if verbose:
                partial_points = convert_to_sparse_voxel_grid(partial_voxels)
                plot_voxel(partial_points, voxel_res=voxels.shape)
                full_points = convert_to_sparse_voxel_grid(voxels)
                all_points = np.concatenate((full_points, partial_points))

                partial_col = np.zeros(len(partial_points))
                full_col = np.zeros(len(full_points)) + 1
                all_col = np.concatenate((full_col, partial_col))

                plot_voxel(all_points, voxel_res=voxels.shape, colors=all_col)
            
            partials.append(partial_voxels)
            fulls.append(voxels)

        # If we have enough objects, create a new TFRecord file.
        if len(partials) == objects_per_tfr:
            filename = file_prepend + str(file_num) + '.tfrecord'
            generate_tfrecord(partials, fulls, filename)

            # print "Generated: ", filename
            
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
    
