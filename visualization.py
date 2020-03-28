# Code for visualizing voxels and point clouds.

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pdb

def plot_3d_points(points, bound=0.5, signed_distances=None):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0 and signed_distances is not None:
        ax.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2], c=signed_distances)
    else:
        ax.scatter(xs=points[:, 0], ys=points[:, 1], zs=points[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SDF')

    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)

    pyplot.show()

def visualize_points_overlay(point_sets, bound=0.5, show=False):
    num_sets = len(point_sets)
    colors = ['red', 'blue', 'yellow', 'orange', 'green']
    assert(num_sets <= len(colors))

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(range(num_sets)):
        ax.scatter(point_sets[i][:, 0], point_sets[i][:, 1], point_sets[i][:, 2], c=colors[i])

    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)

    pyplot.show()

def plot_voxel(voxel, img_path=None, voxel_res=(32,32,32)):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(voxel) != 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
