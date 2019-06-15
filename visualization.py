# Code for visualizing voxels and point clouds.

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def plot_voxel(voxel, img_path=None, voxel_res=(32,32,32)):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(voxel) != 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('voxel')

    if voxel_res is not None:
        ax.set_xlim3d(0, voxel_res[0])
        ax.set_ylim3d(0, voxel_res[1])
        ax.set_zlim3d(0, voxel_res[2])
    
    pyplot.show()
    if img_path is not None:
        pyplot.savefig(img_path)

def convert_to_sparse_voxel_grid(voxel_grid, threshold=0.5):
    sparse_voxel_grid = []
    voxel_dim = voxel_grid.shape
    for i in xrange(voxel_dim[0]):
        for j in xrange(voxel_dim[1]):
            for k in xrange(voxel_dim[2]):
                if voxel_grid[i, j, k] > threshold:
                    sparse_voxel_grid.append([i, j, k])
    return np.asarray(sparse_voxel_grid)
