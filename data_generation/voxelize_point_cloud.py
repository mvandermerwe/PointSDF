
import numpy as np
from show_voxel import *

def point_to_voxel(i, object_cloud, voxel_min_loc, voxel_size):
    '''
    Convert single point to a voxel.
    '''
    pt = object_cloud[i]
    x_idx = int((pt[0] - voxel_min_loc[0]) / voxel_size)
    y_idx = int((pt[1] - voxel_min_loc[1]) / voxel_size)
    z_idx = int((pt[2] - voxel_min_loc[2]) / voxel_size)

    return (x_idx, y_idx, z_idx)

def rotation(theta):
    '''
    Generate 3D rotation about the z axis for the specified theta.
    '''

    rot = np.matrix([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    return rot

def point_cloud_to_voxel(obj_cloud, voxel_size, voxel_res_partial, voxel_res_full, verbose=False):
    '''
    Given a point cloud (i.e., 3D points) generate a centered voxelization, using the given voxel_size. Voxelizes to smaller (voxel_res_partial) then centers in full voxel (voxel_res_full) so that the full view still "likely" fits. This is a fundamental assumption of this approach.
    Assume the points are centered.
    '''

    # Start by voxelizing to smaller size.
    voxel_res = (voxel_res_partial, voxel_res_partial, voxel_res_partial)

    # Set voxel limits
    voxel_min_loc = [0.0, 0.0, 0.0]
    voxel_min_loc[0] = -voxel_res[0] / 2 * voxel_size # x_min
    voxel_min_loc[1] = -voxel_res[1] / 2 * voxel_size # y_min
    voxel_min_loc[2] = -voxel_res[2] / 2 * voxel_size # z_min    

    # Determine number of points.
    num_pts = len(obj_cloud)
    if verbose:
        print "Object cloud len: ", num_pts

    # Determine a buffer to add to each object to center it.
    voxel_buf = (voxel_res_full - voxel_res_partial) // 2
        
    # Build voxel.
    voxel = np.zeros((voxel_res_full, voxel_res_full, voxel_res_full), dtype=bool)
    for i in xrange(num_pts):
        (x_idx, y_idx, z_idx) = point_to_voxel(i, obj_cloud, voxel_min_loc, voxel_size)

        # Check voxel bounds
        if (x_idx >= 0 and x_idx < voxel_res[0] and
            y_idx >= 0 and y_idx < voxel_res[1] and
            z_idx >= 0 and z_idx < voxel_res[2]):
            voxel[x_idx+voxel_buf, y_idx+voxel_buf, z_idx+voxel_buf] = True

    # Show if asked.
    if verbose:
        plot_voxel(convert_to_sparse_voxel_grid(voxel), voxel_res=(voxel_res_full, voxel_res_full, voxel_res_full))
        
    return voxel
