from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_voxel(voxel, img_path=None, voxel_res=None, centroid=None, pca_axes=None, colors=None):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(voxel) != 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], c=colors)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('voxel')

    if voxel_res is not None:
        ax.set_xlim3d(0, voxel_res[0])
        ax.set_ylim3d(0, voxel_res[1])
        ax.set_zlim3d(0, voxel_res[2])

    # if centroid is not None and pca_axes is not None:
    #     for pca_ax in pca_axes:
    #         ax.plot([centroid[0], centroid[0] + pca_ax[0]], [centroid[1], centroid[1] + pca_ax[1]], [centroid[2], centroid[2] + pca_ax[2]], ax)
    
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


if __name__ == '__main__':
    # voxel_grid = np.random.rand(20, 20, 20)
    # voxel_grid[voxel_grid >= 0.5] = 1.
    # voxel_grid[voxel_grid <= 0.5] = 0.
    voxel_file = '/home/markvandermerwe/ws/datasets/ycb_train/042_adjustable_wrench_0.npy'
    voxel_grid = np.load(voxel_file)
    sparse_voxel_grid = convert_to_sparse_voxel_grid(voxel_grid)
    # print sparse_voxel_grid
    plot_voxel(sparse_voxel_grid, voxel_res=(32,32,32))
