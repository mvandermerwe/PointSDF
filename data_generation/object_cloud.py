import numpy as np
# from geometry_msgs.msg import Pose, Quaternion, PoseStamped, PointStamped
# import tf
from matplotlib import pyplot

# from voxelize_point_cloud import point_cloud_to_voxel

_POINT_CLOUD_SIZE = 1000
_VOXEL_RES_PARTIAL = 26
_VOXEL_RES_FULL = 32
_FLIP_THRESHOLD = 0.0001 # Threshold to flip randomly in resulting voxel grid.
_NOISE = False

def plot_3d_points(points, signed_distances=None):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=signed_distances)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)

    pyplot.show()

def process_object_cloud(obj_cloud, object_frame=False, voxelize=False, verbose=False):
    '''
    Process the given point cloud the way we want it to be processed.
    '''
    # Store everything in dictionary for ease.
    return_dict = {}
    
    # Down/Up Sample cloud so everything has the same # of points.
    idxs = np.random.choice(obj_cloud.shape[0], size=_POINT_CLOUD_SIZE, replace=True)
    obj_cloud = obj_cloud[idxs,:]
    
    if object_frame:
        object_transform, centroid = find_object_frame(obj_cloud, verbose)
        centroid = np.transpose(np.array(world_frame_center))[0]
    if not object_frame:
        # If we don't want things in the object frame, we should change the transform to be only
        # translation and in the world frame.
        centroid = np.array([
            (np.amax(obj_cloud[:,0]) + np.amin(obj_cloud[:,0])) / 2,
            (np.amax(obj_cloud[:,1]) + np.amin(obj_cloud[:,1])) / 2,
            (np.amax(obj_cloud[:,2]) + np.amin(obj_cloud[:,2])) / 2,
            ])
        object_transform = np.eye(4)
        object_transform[0,3] = -centroid[0]
        object_transform[1,3] = -centroid[1]
        object_transform[2,3] = -centroid[2]        

    # Transform our point cloud. I.e. center and rotate to new frame.
    for i in range(obj_cloud.shape[0]):
        obj_cloud[i] = np.dot(object_transform, [obj_cloud[i][0], obj_cloud[i][1], obj_cloud[i][2], 1])[:3]

    # Create pose.
    # object_pose = Pose()
    # object_transform_inv = np.linalg.inv(object_transform)
    # pose_quat = tf.transformations.quaternion_from_matrix(object_transform_inv)
    # object_pose.position.x = object_transform_inv[0,3]
    # object_pose.position.y = object_transform_inv[1,3]
    # object_pose.position.z = object_transform_inv[2,3]
    # object_pose.orientation.x = pose_quat[0]
    # object_pose.orientation.y = pose_quat[1]
    # object_pose.orientation.z = pose_quat[2]
    # object_pose.orientation.w = pose_quat[3]

    # obj_pose_stamp = PoseStamped()
    # obj_pose_stamp.header.frame_id = 'kinect_pyrender' # Everything should at this point be in camera frame.
    # obj_pose_stamp.pose = object_pose

    # return_dict['object_pose'] = obj_pose_stamp
    # return_dict['object_transform'] = object_transform
    # return_dict['centroid'] = centroid

    # Find width (x), height (y), and depth (z) of the point cloud after alignment. Used by grasp success model.
    width = np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0])
    height = np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1])
    depth = np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2])
    whd = [width, height, depth]

    # Find w/h/d but from the center out - this is used to form a bounding box to speed up PointSDF checks.
    w_from_center = max(abs(np.amax(obj_cloud[:,0])), abs(np.amin(obj_cloud[:,0])))
    h_from_center = max(abs(np.amax(obj_cloud[:,1])), abs(np.amin(obj_cloud[:,1])))
    d_from_center = max(abs(np.amax(obj_cloud[:,2])), abs(np.amin(obj_cloud[:,2])))
    sdf_whd = [w_from_center, h_from_center, d_from_center]

    return_dict['whd'] = whd
    return_dict['sdf_whd'] = sdf_whd
    
    # Determine scaling size.
    max_dim = max(
        np.amax(obj_cloud[:,0]) - np.amin(obj_cloud[:,0]),
        np.amax(obj_cloud[:,1]) - np.amin(obj_cloud[:,1]),
        np.amax(obj_cloud[:,2]) - np.amin(obj_cloud[:,2]),
    )
    # Scale so that max dimension is about 1.
    scale = (1.0/1.03) / max_dim

    return_dict['max_dim'] = max_dim
    return_dict['scale'] = scale

    # Scale every point.
    scaled_obj_cloud = obj_cloud * scale

    return_dict['object_cloud'] = obj_cloud
    return_dict['scaled_object_cloud'] = scaled_obj_cloud

    if verbose:
        plot_3d_points(obj_cloud)

    # Voxelize if desired.
    if voxelize:
        voxel_size = max_dim / _VOXEL_RES_PARTIAL
        return_dict['voxel_size'] = voxel_size
        
        # Convert to voxel.
        voxel = point_cloud_to_voxel(obj_cloud, voxel_size, _VOXEL_RES_PARTIAL, _VOXEL_RES_FULL, verbose)

        if _NOISE:
            # Simple noise adding by flipping a bit w/ very low probability.
            to_flip = np.where(np.random.rand(_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL) < _FLIP_THRESHOLD)

            for x,y,z in zip(*to_flip):
                voxel[x,y,z] = not voxel[x,y,z]

        return_dict['voxel'] = voxel
        
        if verbose:
            plot_voxel(convert_to_sparse_voxel_grid(voxel), voxel_res=(_VOXEL_RES_FULL, _VOXEL_RES_FULL, _VOXEL_RES_FULL))
        
    return return_dict

