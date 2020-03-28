# Offscreen render with GPU.

import os
import trimesh
import pyrender
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import open3d
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

# Hyperparameters for data collection.
_NUM_POSES_PER_OBJ = 200
_MESH_DATABASE = '/dataspace/ICRA_Data/Meshes/' # Meshes defined in meters.
_RGB_SAVE_DIR = '/dataspace/ICRA_Data/PyrenderData/RGB/'
_DEPTH_SAVE_DIR = '/dataspace/ICRA_Data/PyrenderData/Depth/'
_POSE_SAVE_FILENAME = '/dataspace/ICRA_Data/PyrenderData/object_poses.hdf5'
_VISUALIZE_TRIMESH = False # True
_VISUALIZE_PYRENDER = False # True
_NOISE_SIGMA = 0.001 # Source: http://wiki.ros.org/openni_kinect/kinect_accuracy

def object_pose_completed(object_pose_name, pose_save_filename):
    '''
    Determine if the given object_pose has been completed.
    '''

    with h5py.File(pose_save_filename, 'a') as save_cloud_file:
        return object_pose_name in save_cloud_file

def save_object_pose(object_pose_name, random_pose, pose_save_filename):
    '''
    Write the object pose in the world frame to h5 file.
    '''

    # Save the rotation w/ h5.
    with h5py.File(pose_save_filename, 'a') as save_pose_file:
        # Create new group for this object/pose.
        obj_group = save_pose_file.create_group(object_pose_name)

        # Get Rotation
        rotation = [random_pose[0,0], random_pose[0,1], random_pose[0,2], random_pose[1,0], random_pose[1,1], random_pose[1,2], random_pose[2,0], random_pose[2,1], random_pose[2,2]]

        # Save rotation.
        obj_group.create_dataset('rotation', data=rotation)

def generate_pcd_from_depth(depth, kuf, kvf):
    '''
    Given a depth image and camera intrinsics, convert to a point cloud
    and save as PCD file.
    '''
    # Convert depth to a point cloud and store as PCD.
    height = depth.shape[0]
    width = depth.shape[1]

    # Go through each nonzero and convert to a 3D point using camera matrix.
    mask = np.where(depth > 0)
    x = mask[1]
    y = mask[0]
    norm_x = (x.astype(np.float32)-(width*0.5)) / (width*0.5)
    norm_y = (y.astype(np.float32)-(height*0.5)) / (height*0.5)
    world_x = norm_x * depth[y,x] / kuf
    world_y = norm_y * depth[y,x] / kvf
    world_z = -depth[y,x]
    points = np.vstack((world_x, -world_y, world_z)).T # Negative on y because of image matrix layout.
    return points

def generate_occupancy_sample_pcd_from_depth(depth, kuf, kvf):
    '''
    Given a depth image and camera intrinsics, convert to a point cloud
    and save as PCD file.
    '''

    # Convert depth to a point cloud and store as PCD.
    height = depth.shape[0]
    width = depth.shape[1]

    # Go through each nonzero and convert to a 3D point using camera matrix.
    mask = np.where(depth > 0)
    x = mask[1]
    y = mask[0]
    norm_x = (x.astype(np.float32)-(width*0.5)) / (width*0.5)
    norm_y = (y.astype(np.float32)-(height*0.5)) / (height*0.5)
    world_x = norm_x * depth[y,x] / kuf
    world_y = norm_y * depth[y,x] / kvf
    world_z = -depth[y,x]
    points_occ = np.vstack((world_x, -world_y, world_z)).T # Negative on y because of image matrix layout.

    # TODO: Sample smaller depths along the camer beam for unoccupied points
    # for each point in depth sample n points randomly along the length 0 to d and compute their 3d pointsx
    return points_occ, points_empty

def write_pcd_from_points(points, save_directory, object_pose_name):
    # Save to PCD.
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    open3d.write_point_cloud(os.path.join(save_directory, object_pose_name + ".pcd"), pcd)

def read_object_meshes(mesh_database):
    # Determine meshes.
    object_meshes = [filename for filename in os.listdir(mesh_database) if ".stl" in filename]
    objects = list(map(lambda x: x.replace('.stl', ''), object_meshes))
    return (objects, object_meshes)

def visualize_images(color,depth):
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()

def generate_point_clouds(mesh_database, rgb_save_directory, depth_save_directory,
                          pose_save_filename, num_poses_per_obj):
    '''
    Generate point clouds for all objects in the mesh database.
    '''
    objects, object_meshes = read_object_meshes(mesh_database)

    # Setup the scene.
    scene = pyrender.Scene()

    # Camera/Light
    cam = pyrender.PerspectiveCamera(yfov=0.820305, aspectRatio=(4.0/3.0))
    # Get some camera instrinsics to do depth to point cloud conversion.
    kuf = cam.get_projection_matrix()[0][0]
    kvf = cam.get_projection_matrix()[1][1]
    camera_pose = np.eye(4)
    scene.add(cam, pose=camera_pose)
    light_pose = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 10.],
        [0., 0., 0., 1.]
        ]
    scene.add(pyrender.DirectionalLight(), pose=light_pose)

    # Setup renderer.
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)

    # Go through each object and do a bunch of settings.
    for object_name, object_mesh in tqdm(zip(objects, object_meshes)):
        # print object_name

        # Add the mesh to the scene.
        tm = trimesh.load(os.path.join(mesh_database, object_mesh))
        if _VISUALIZE_TRIMESH:
            tm.show()
            
        m = pyrender.Mesh.from_trimesh(tm)
        mesh_pose = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., -1.5],
            [0., 0., 0., 1.]
            ]
        mesh_node = pyrender.Node(mesh=m, matrix=mesh_pose)
        scene.add_node(mesh_node)

        # Do a bunch of orientations for each object.
        for rot_num in range(num_poses_per_obj):
            object_pose_name = object_name + "_" + str(rot_num)

            # Check if pose has already been done (this is to allow easy restarts).
            if object_pose_completed(object_pose_name, pose_save_filename):
                continue

            # Generate a random transformation.
            random_pose = trimesh.transformations.random_rotation_matrix()
            random_pose[2][3] = -1.5 # Move to spot we want.

            # Write object pose to h5.
            save_object_pose(object_pose_name, random_pose, pose_save_filename)

            # Update the pose.
            scene.set_pose(mesh_node, pose=random_pose)

            # Render.
            color, depth = r.render(scene)

            # Add noise to depth image.
            noise = np.random.normal(0.0, _NOISE_SIGMA, depth.shape)
            nonzeros = np.nonzero(depth)
            depth[nonzeros] = depth[nonzeros] + noise[nonzeros]

            if _VISUALIZE_PYRENDER:
                visualize_images(color,depth)

            # Save RGB.
            cv2.imwrite(os.path.join(rgb_save_directory, object_pose_name + ".png"), color)

            # Save PCD.
            points = generate_pcd_from_depth(depth, kuf, kvf)
            write_pcd_from_points(points, depth_save_directory, object_pose_name)

        # Remove object.
        scene.remove_node(mesh_node)
    # Cleanup
    r.delete()

def run():
    # Generate away.
    generate_point_clouds(_MESH_DATABASE,
                          _RGB_SAVE_DIR,
                          _DEPTH_SAVE_DIR,
                          _POSE_SAVE_FILENAME,
                          _NUM_POSES_PER_OBJ)

if __name__ == '__main__':
    run()
