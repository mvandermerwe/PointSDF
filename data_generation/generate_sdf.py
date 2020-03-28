import trimesh
import numpy as np
import math
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import pypcd
import h5py
import pdb
import os
from tqdm import tqdm
import tensorflow as tf
if __name__ == '__main__':
    tf.enable_eager_execution()

from generate_view_splits import get_view_splits

from show_voxel import plot_voxel, convert_to_sparse_voxel_grid
from object_frame import find_object_frame
from object_cloud import process_object_cloud

_MESH_DATABASE = '/home/markvandermerwe/Meshes'
_DEPTH_DATABASE = '/dataspace/ICRA_Data/PyrenderData/Depth'
_OBJECT_POSE_FILE = '/dataspace/ICRA_Data/PyrenderData/object_poses.hdf5'
_SDF_SAVE_FILE = '/dataspace/ICRA_Data/mesh_sdf.h5'

_OBJECT_FRAME = False # Create and align with an object frame via PCA.

_TRAIN_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/SDF_CF/Train'
_VALIDATION_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/SDF_CF/Validation'
_TEST_FILE_DATABASE = '/dataspace/ICRA_Data/ReconstructionData/SDF_CF/Test'

_NUM_POSES_PER_OBJ = 200
_OBJECTS_PER_TFR = 1500 # Number of examples per TFRecord file. Should be about 100 MB per file.

SURFACE_VARIANCE_A_ = 0.0025
SURFACE_VARIANCE_B_ = 0.00025

_POINT_CLOUD_SIZE = 1000 # Always sample point cloud to this size.

_VOXEL_RES_FULL = 32
_VOXEL_RES_PARTIAL = 26

_VERBOSE = True

def in_bounds(x):
    return x >= 0 and x < _VOXEL_RES_FULL

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

def get_sdf_from_file(mesh_name, sdf_save_file):
    '''
    For given mesh, read in all the sampled points and their corresponding
    signed distance function values.
    '''
    with h5py.File(sdf_save_file, 'r') as f:
        mesh_object = f[mesh_name]
        sample_points = mesh_object['sample_points'][:]
        signed_distances = mesh_object['signed_distances'][:]
            
        return sample_points, signed_distances

def get_signed_distance_points(mesh_database, mesh_name, pc_scale, transform_matrix, sdf_save_file):
    mesh_file = os.path.join(mesh_database, mesh_name + '.stl')
    mesh = trimesh.load(mesh_file)
    lo, hi = mesh.bounds
    lo = lo - mesh.centroid
    hi = hi - mesh.centroid
    max_dim = 2 * max(abs(hi[0]), abs(lo[0]), abs(hi[1]), abs(lo[1]), abs(hi[2]), abs(lo[2]))

    # Scale object to be in 1x1x1 box. To do this, scale to be inside slightly smaller 1/1.03 bounding box (same as DeepSDF).
    m_scale = (1.0/1.03) / max_dim

    # Read in the points/sdfs from h5 file.
    sample_points, signed_distances = get_sdf_from_file(mesh_name, sdf_save_file)
    for i in range(len(sample_points)):
        # Need to scale back into the normal frame.
        sample_points[i] = sample_points[i] / m_scale
        sample_points[i] = np.dot(transform_matrix, [sample_points[i][0], sample_points[i][1], sample_points[i][2], 1])[:3]
        sample_points[i] = sample_points[i] * pc_scale

    # Scale the signed distances.
    for i in range(len(signed_distances)):
        signed_distances[i] = signed_distances[i] * (pc_scale/m_scale)

    return sample_points, signed_distances

def get_point_cloud_points(pcd_filename, verbose):
    '''
    Read in point cloud from pcd file and scale down.
    '''
    
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

    return_dict = process_object_cloud(obj_cloud, object_frame=_OBJECT_FRAME, verbose=verbose)

    return return_dict['object_cloud'], return_dict['scale'], return_dict['object_transform'], return_dict['centroid']

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

def _float_feature(value):
    '''
    Returns a float_list from a float / double.
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(point_cloud_views, xyz, label):
    '''
    Serialize given SDF data.
    '''

    feature = {
        'point_clouds': _bytes_feature(tf.serialize_tensor(point_cloud_views).numpy()),
        'xyzs': _bytes_feature(tf.serialize_tensor(xyz).numpy()),
        'labels': _bytes_feature(tf.serialize_tensor(label).numpy()),
    }

    # Serialize w/ tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(point_cloud_views, xyz, label):
    tf_string = tf.py_func(
        serialize_example,
        (point_cloud_views, xyz, label),
        tf.string)
    return tf.reshape(tf_string, ())

def generate_tfrecord(views, xyzs, labels, write_file):
    '''
    Create a tfrecord file for the given SDF data.
    View is a string that will point to the pc cloud stored in the h5 file
    for this example.
    '''
    
    # Make dataset.
    sdf_dataset = tf.data.Dataset.from_tensor_slices((np.array(views).astype(np.float32), np.array(xyzs).astype(np.float32), np.array(labels).astype(np.float32)))

    # Map to serialization.
    serialized_sdf_dataset = sdf_dataset.map(tf_serialize_example)
    
    # Write to file.
    writer = tf.data.experimental.TFRecordWriter(write_file)
    writer.write(serialized_sdf_dataset)
    
def get_data_for_view(view, depth_database, mesh_database, sdf_save_file, verbose=False):
    '''
    For the given view, read all SDF examples in.
    '''

    # Remove pose number to get mesh name.
    mesh_name = "_".join(view.split("_")[:-1])

    # Get point cloud into set of points.
    point_cloud_points, pc_scale, object_frame_transform, partial_view_center_world = get_point_cloud_points(os.path.join(depth_database, view+'.pcd'), verbose)

    if point_cloud_points is None:
        return None, None, None 

    if verbose:
        plot_3d_points(point_cloud_points)

    # Get world frame transformation for the object.
    rotation_matrix = get_object_rotation_matrix(_OBJECT_POSE_FILE, view, verbose)

    # Combine transformation of full mesh in world frame to get transform to object frame.
    mesh_centroid = np.array([0, 0, -1.5])
    d_o_f_w = mesh_centroid - partial_view_center_world
    d_o_f_o = np.dot(object_frame_transform[:3,:3], d_o_f_w)

    object_frame_transform[:3,:3] = np.dot(object_frame_transform[:3,:3], rotation_matrix)
    object_frame_transform[0,3] = d_o_f_o[0]
    object_frame_transform[1,3] = d_o_f_o[1]
    object_frame_transform[2,3] = d_o_f_o[2]

    sample_points, signed_distances = get_signed_distance_points(mesh_database, mesh_name, pc_scale, object_frame_transform, sdf_save_file)

    if verbose:
        points_inside = sample_points[np.where(signed_distances <= 0.0)]
        plot_3d_points(points_inside)
        plot_3d_points(sample_points, signed_distances)

    if verbose:
        points_inside = sample_points[np.where(signed_distances <= 0.0)]
        all_points = np.concatenate([points_inside, point_cloud_points], axis=0)

        pt_cld_col = np.zeros(point_cloud_points.shape[0])
        true_cld_col = np.zeros(points_inside.shape[0]) + 1
        col = np.concatenate([true_cld_col, pt_cld_col])
        
        plot_3d_points(all_points, col)
    
    return point_cloud_points, sample_points, signed_distances

def shuffle_in_unison(a, b, c, seed=None):

    np.random.seed(seed)
    
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    return a,b,c

def write_views_to_tfrecord(views, file_prepend, mesh_database, depth_database, object_pose_file, objects_per_tfr, sdf_save_file, verbose=False):
    '''
    Write the given list of views to a TFRecord file of the
    size desired. File prepend should be folder to and prefix for
    each folder. E.g., /home/name/data_folder/train_ will write train_0.tfrecord,train_1.tfrecord
    ,etc. to /home/name/data_folder.
    '''
    
    point_cloud_views = []
    xyzs = []
    labels = []
    file_num = 0

    for view in tqdm(views):

        # Get all points for this view.
        point_cloud_points, sample_points, signed_distances = get_data_for_view(view,  depth_database, mesh_database, sdf_save_file, verbose)

        # Figure out how to write to file well.
        if sample_points is not None:
            xyzs.append(sample_points)
            labels.append(signed_distances)
            point_cloud_views.append(point_cloud_points)

        # If we have enough objects, create a new TFRecord file.
        if len(xyzs) == objects_per_tfr:
            filename = file_prepend + str(file_num) + '.tfrecord'

            # Concatentate the examples together.
            views_ = np.array(point_cloud_views)
            xyzs_ = np.array(xyzs)
            labels_ = np.array(labels)

            # Shuffle.
            views_, xyzs_, labels_ = shuffle_in_unison(views_, xyzs_, labels_)

            # Generate a record for these examples.
            generate_tfrecord(views_, xyzs_, labels_, filename)
            # print "Generated: ", filename
            
            point_cloud_views = []
            xyzs = []
            labels = []
            file_num += 1

    # Write whatever is left.
    if len(xyzs) != 0:
        filename = file_prepend + str(file_num) + '.tfrecord'

        # Concatentate the examples together.
        views_ = np.array(point_cloud_views)
        xyzs_ = np.array(xyzs)
        labels_ = np.array(labels)

        # Shuffle.
        views_, xyzs_, labels_ = shuffle_in_unison(views_, xyzs_, labels_)
        
        generate_tfrecord(views_, xyzs_, labels_, filename)

# def get_donut_view_splits():
#     object_name = 'donut_poisson_000'

#     training_views = list(map(lambda x: object_name + '_' + str(x), list(range(10))))
#     validation_views = list(map(lambda x: object_name + '_' + str(x), list(range(10,15))))
#     test_views = list(map(lambda x: object_name + '_' + str(x), list(range(15,20))))

#     return training_views, validation_views, test_views

def generate_tfrecord_datasets(mesh_database, depth_database, object_pose_file, objects_per_tfr, num_poses_per_obj, train_file_database, validation_file_database, test_file_database, sdf_save_file, verbose=False):
    '''
    Read full dataset and generate partial full training pairs.
    '''
    
    training_views, validation_views, test_views = get_view_splits()
    #training_views, validation_views, test_views = get_donut_view_splits()
    
    # Write to files.
    write_views_to_tfrecord(training_views, os.path.join(train_file_database, 'train_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, sdf_save_file, verbose)
    write_views_to_tfrecord(validation_views, os.path.join(validation_file_database, 'validation_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, sdf_save_file, verbose)
    write_views_to_tfrecord(test_views, os.path.join(test_file_database, 'test_'), mesh_database, depth_database, object_pose_file, objects_per_tfr, sdf_save_file, verbose)
    
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
                               _SDF_SAVE_FILE,
                               _VERBOSE)

    # Generate TFRecords with a single object in each - this makes testing easier.
    # write_single_objects_to_tf(_MESH_DATABASE,
    #                            _DEPTH_DATABASE,
    #                            _OBJECT_POSE_FILE,
    #                            _OBJECTS_PER_TFR,
    #                            _NUM_POSES_PER_OBJ,
    #                            _TRAIN_FILE_DATABASE,
    #                            _VALIDATION_FILE_DATABASE,
    #                            _TEST_FILE_DATABASE,
    #                            _VERBOSE)
    
if __name__ == '__main__':
    run()
