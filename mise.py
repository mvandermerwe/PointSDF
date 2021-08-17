# Reimplementation of MISE from Occupancy Networks paper.

import trimesh
import h5py
import numpy as np
import tensorflow as tf
import mcubes
import sys
import os
import cv2
import pickle
import pdb
import pymesh
from tqdm import tqdm

from sdf_pointconv_model import get_pointconv_model, get_sdf_model, get_embedding_model, get_sdf_prediction
from visualization import plot_3d_points, plot_voxel, convert_to_sparse_voxel_grid, visualize_points_overlay
from sdf_dataset import get_sdf_dataset, get_pcd

_MODEL_FUNC = get_pointconv_model
_MODEL_PATH = '/home/markvandermerwe/models/ICRA_Models/reconstruction/pointconv_mse_cf'
_SPLITS_PATH = '/home/markvandermerwe/catkin_ws/src/ll4ma_3d_reconstruction/src/data_generation/data_split/test_fold.txt'
_SAVE_PATH = 'out/test_dir/'
_PCD_DATABASE = '/dataspace/ICRA_Data/PyrenderData/Depth/'
_GRASP_DATABASE = False
_OBJECT_FRAME = False

def is_occupied(x, y, z, pre_voxelized):
    '''
    x,y,z is bottom left.
    '''

    # Check if all are on.
    return (pre_voxelized[x,y,z] == 1.0 and
            pre_voxelized[x,y,z+1] == 1.0 and
            pre_voxelized[x,y+1,z] == 1.0 and
            pre_voxelized[x,y+1,z+1] == 1.0 and 
            pre_voxelized[x+1,y,z] == 1.0 and
            pre_voxelized[x+1,y,z+1] == 1.0 and
            pre_voxelized[x+1,y+1,z] == 1.0 and
            pre_voxelized[x+1,y+1,z+1] == 1.0)

def is_active(x, y, z, pre_voxelized):
    '''
    Check if given x,y,z in given voxel grid is active.
    '''
    voxel_occupancy = pre_voxelized[x,y,z]

    base = pre_voxelized[x,y,z]
    return (pre_voxelized[x,y,z+1] != base or
            pre_voxelized[x,y+1,z] != base or
            pre_voxelized[x,y+1,z+1] != base or 
            pre_voxelized[x+1,y,z] != base or
            pre_voxelized[x+1,y,z+1] != base or
            pre_voxelized[x+1,y+1,z] != base or
            pre_voxelized[x+1,y+1,z+1] != base)

def get_grid_points(active_voxels, current_voxel_resolution, bound):
    grid_pts = set()
    voxel_size = (2*bound) / float(current_voxel_resolution)
    for x,y,z in active_voxels:
        x_ = -bound + (((2*bound) / float(current_voxel_resolution)) * x)
        y_ = -bound + (((2*bound) / float(current_voxel_resolution)) * y)
        z_ = -bound + (((2*bound) / float(current_voxel_resolution)) * z)
        grid_pts.add((x_,y_,z_))
        grid_pts.add((x_,y_,z_+voxel_size))
        grid_pts.add((x_,y_+voxel_size,z_))
        grid_pts.add((x_,y_+voxel_size,z_+voxel_size))
        grid_pts.add((x_+voxel_size,y_,z_))
        grid_pts.add((x_+voxel_size,y_,z_+voxel_size))
        grid_pts.add((x_+voxel_size,y_+voxel_size,z_))
        grid_pts.add((x_+voxel_size,y_+voxel_size,z_+voxel_size))
    return np.array(list(grid_pts))

# get_sdf: take in ONLY query points and go to SDF.
def mise_voxel(get_sdf, bound, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff, save_path, verbose=False):
    '''
    get_sdf: map from query points to SDF (assume everything else already embedded in func (i.e., point cloud/embedding)).
    bound: sample within [-bound, bound] in x,y,z.
    initial/final_voxel_resolution: powers of two representing voxel resolution to evaluate at.
    voxel_size: size of each voxel (in final res) determined by view.
    centroid_diff: offset if needed.
    '''

    # Number to evaluate in single pass.
    sdf_count_ = 8192

    # Active voxels: voxels we want to evaluate grid points of.
    active_voxels = []
    # Full voxelization.
    voxelized = np.zeros((final_voxel_resolution,final_voxel_resolution,final_voxel_resolution), dtype=np.float32)
    # Intermediate voxelization. This represents the grid points for the voxels (so is resolution + 1 in each dim).
    partial_voxelized = None

    # Init active voxels to all voxels in the initial resolution.
    for x in range(initial_voxel_resolution):
        for y in range(initial_voxel_resolution):
            for z in range(initial_voxel_resolution):
                active_voxels.append([x, y, z])
    active_voxels = np.array(active_voxels)

    # Start main loop that ups resolution.
    current_voxel_resolution = initial_voxel_resolution
    while current_voxel_resolution <= final_voxel_resolution:
        # print(current_voxel_resolution)

        # Setup voxelizations at this dimension.
        partial_voxelized = np.zeros((current_voxel_resolution + 1,current_voxel_resolution + 1,current_voxel_resolution + 1), dtype=np.float32)

        # Get the grid points for this resolution.
        grid_pts = get_grid_points(active_voxels, current_voxel_resolution, bound)
        try:
            pt_splits = np.array_split(grid_pts, grid_pts.shape[0] // sdf_count_)
        except ValueError:
            pt_splits = [grid_pts]
        # print(len(pt_splits))
        
        # For all points sample SDF given the point cloud.
        for pts_ in pt_splits:
            sdf_ = get_sdf(pts_)

            for pt_, sdf in zip(np.reshape(pts_, (-1,3)), np.reshape(sdf_, (-1,))):
                if sdf <= 0.0:
                    # Convert points into grid voxels and set.
                    x_ = int(round(((pt_[0] + bound)/(2 * bound)) * float(current_voxel_resolution)))
                    y_ = int(round(((pt_[1] + bound)/(2 * bound)) * float(current_voxel_resolution)))
                    z_ = int(round(((pt_[2] + bound)/(2 * bound)) * float(current_voxel_resolution)))
                    partial_voxelized[x_,y_,z_] = 1.0


        # Determine filled and active voxels.
        new_active_voxels = []
        for x,y,z in active_voxels:
            if is_occupied(x, y, z, partial_voxelized):
                # Set all associated voxels on in full voxelization.
                voxels_per_voxel = final_voxel_resolution // current_voxel_resolution

                # Set all corresponding voxels in the full resolution to on.
                for x_ in range(voxels_per_voxel*x, voxels_per_voxel*x + voxels_per_voxel):
                    for y_ in range(voxels_per_voxel*y, voxels_per_voxel*y + voxels_per_voxel):
                        for z_ in range(voxels_per_voxel*z, voxels_per_voxel*z + voxels_per_voxel):
                            voxelized[x_, y_, z_] = 1.0
            elif is_active(x, y, z, partial_voxelized):
                # If final resolution, just set it as active.
                if current_voxel_resolution == final_voxel_resolution:
                    voxelized[x,y,z] = 1.0
                    continue
                
                # Up voxel position to match doubling of voxel resolution.
                x_base = 2*x
                y_base = 2*y
                z_base = 2*z

                # Add new voxels for higher resolution. Each voxel gets split into 8 new.
                new_active_voxels.append([x_base, y_base, z_base])
                new_active_voxels.append([x_base, y_base, z_base+1])
                new_active_voxels.append([x_base, y_base+1, z_base])
                new_active_voxels.append([x_base, y_base+1, z_base+1])
                new_active_voxels.append([x_base+1, y_base, z_base])
                new_active_voxels.append([x_base+1, y_base, z_base+1])
                new_active_voxels.append([x_base+1, y_base+1, z_base])
                new_active_voxels.append([x_base+1, y_base+1, z_base+1])                
        active_voxels = np.array(new_active_voxels)
        current_voxel_resolution = current_voxel_resolution * 2

    # print("Done with extraction.")

    # Padding to prevent holes if go up to edge.
    voxels = voxelized
    voxelized = np.pad(voxelized, ((1,1),(1,1),(1,1)), mode='constant')
    
    # Mesh w/ mcubes.
    vertices, triangles = mcubes.marching_cubes(voxelized, 0)
    vertices = vertices * voxel_size

    # Center mesh.
    vertices[:,0] -= voxel_size * (((final_voxel_resolution) / 2) + 1)
    vertices[:,1] -= voxel_size * (((final_voxel_resolution) / 2) + 1)
    vertices[:,2] -= voxel_size * (((final_voxel_resolution) / 2) + 1)

    vertices[:,0] -= centroid_diff[0]
    vertices[:,1] -= centroid_diff[1]
    vertices[:,2] -= centroid_diff[2]
    
    #save_file = os.path.join(save_path, view + '.off')
    mcubes.export_obj(vertices, triangles, save_path)

    # Display mesh.
    if verbose:
        gen_mesh = trimesh.load(save_path)
        gen_mesh.show()

    return None # convert_to_sparse_voxel_grid(voxels, threshold=0.5)

def get_test_meshes(grasp_database=True, ycb_database=False):
    meshes = set()
    with open(_SPLITS_PATH) as f:
        for view in f:
            if grasp_database and 'poisson' in view:
                meshes.add('_'.join(view.split('_')[:-1]))
            elif ycb_database and 'poisson' not in view:
                meshes.add('_'.join(view.split('_')[:-1]))
    
    fin_meshes = []
    for mesh in meshes:
        fin_meshes.append(mesh + '_10') # Only use the 10th rendered view.
        
    return fin_meshes
        
def mesh_objects(model_func, model_path, save_path, pcd_folder, grasp_database=True):
    # Setup model.
    get_sdf, get_embedding, _ = get_sdf_prediction(model_func, model_path)

    # Get names of partial views.
    meshes = get_test_meshes(grasp_database=grasp_database, ycb_database=(not grasp_database))
    
    # Bounds of 3D space to evaluate in: [-bound, bound] in each dim.
    bound = 0.8
    # Starting voxel resolution.
    initial_voxel_resolution = 32
    # Final voxel resolution.
    final_voxel_resolution = 512
    
    # Mesh the views.
    for mesh in tqdm(meshes):
        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_pcd(mesh, pcd_folder, object_frame=_OBJECT_FRAME, verbose=False);
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)    
        if pc_ is None:
            print(view, " has no point cloud.")
            continue
        point_clouds_ = np.reshape(pc_, (1,1000,3))

        # Make view specific sdf func.
        def get_sdf_query(query_points):
            return get_sdf(point_clouds_, query_points)

        recon_voxel_pts = mise_voxel(get_sdf_query, bound, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff, os.path.join(save_path, mesh + '.obj'), verbose=False)
    
if __name__ == '__main__':
    mesh_objects(_MODEL_FUNC, _MODEL_PATH, _SAVE_PATH, _PCD_DATABASE, _GRASP_DATABASE)
