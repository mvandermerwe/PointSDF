import trimesh
import numpy as np
import os
import math
import h5py
import pdb
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

_MESH_DATABASE = '/dataspace/meshes/BigBird/'
SURFACE_SAMPLES_ = 1024
RANDOM_SAMPLES_ = 1024
SURFACE_VARIANCE_A_ = 0.0025
SURFACE_VARIANCE_B_ = 0.00025
_SDF_SAVE_FILE = '/dataspace/ICRA_Data/GraspData/sdf_big_bird.h5'
_VERBOSE = False

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

def scale_mesh(mesh):
    '''
    Scale the given mesh to fit in a 1x1x1 box. This normalization is to simplify
    the learning problem so everything is always in the same scale.
    '''

    # Find the maximum dimension size to determine how we need to scale the object.
    lo, hi = mesh.bounds
    lo = lo - mesh.centroid
    hi = hi - mesh.centroid
    max_dim = 2 * max(abs(hi[0]), abs(lo[0]), abs(hi[1]), abs(lo[1]), abs(hi[2]), abs(lo[2]))

    # Scale object to be in 1x1x1 box. To do this, scale to be inside slightly smaller 1/1.03 bounding box (same as DeepSDF).
    scale_ = (1.0/1.03) / max_dim
    mesh.apply_scale(scale_)
    
    return mesh, scale_

def get_signed_distance_points(mesh_name, verbose=False):
    '''
    For given mesh file, load and shrink to 1x1x1 bounding box. Then
    generate a bunch of samples w/ SDF labels.
    '''
    mesh_file = os.path.join(_MESH_DATABASE, mesh_name + '.stl')
    
    mesh = trimesh.load(mesh_file)

    # Scale mesh to fit in a 1x1x1 bounding box.
    mesh, m_scale = scale_mesh(mesh)
    #print m_scale
    #mesh.show()

    # Sample points on the surface.
    surface_points = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES_)

    # For each point on the surface, we generate two nearby points by adding slight offsets sampled from
    # two zero-mean Gaussian distributions.
    offset_a = np.random.normal(0.0, math.sqrt(SURFACE_VARIANCE_A_), (SURFACE_SAMPLES_,3))
    offset_b = np.random.normal(0.0, math.sqrt(SURFACE_VARIANCE_B_), (SURFACE_SAMPLES_,3))

    surface_a = surface_points[0] + offset_a
    surface_b = surface_points[0] + offset_b

    # Additionally sample points randomly in the 1x1x1 bounding box.
    random_points = np.zeros((RANDOM_SAMPLES_, 3))
    random_points[:,0] = mesh.centroid[0] + np.random.uniform(-0.6, 0.6, (RANDOM_SAMPLES_))
    random_points[:,1] = mesh.centroid[1] + np.random.uniform(-0.6, 0.6, (RANDOM_SAMPLES_))
    random_points[:,2] = mesh.centroid[2] + np.random.uniform(-0.6, 0.6, (RANDOM_SAMPLES_))

    # Concatenate together all these points.
    sample_points = np.concatenate((surface_a, surface_b, random_points))
    #sample_points = surface_points[0]
    # sample_points = np.concatenate((surface_points[0], random_points))

    # Get the signed distance for each point to our mesh.
    signed_distances = []
    start_idx = 0
    for i in range(math.ceil(len(sample_points) / 10000.)):
        signed_distances.extend(-trimesh.proximity.signed_distance(mesh, sample_points[start_idx:min(start_idx+10000, len(sample_points))]))
        start_idx += 10000

    if verbose:
        plot_3d_points(sample_points, np.array(signed_distances))

    write_h5_signed_distances(mesh_name, sample_points, signed_distances)

def write_h5_signed_distances(mesh_name, sample_points, signed_distances):
    with h5py.File(_SDF_SAVE_FILE, 'a') as f:
        # Create new group for mesh.
        mesh_group = f.create_group(mesh_name)

        # Save the sample points and signed distances.
        mesh_group.create_dataset("sample_points", (len(sample_points), 3), data=sample_points)
        mesh_group.create_dataset("signed_distances", (len(sample_points),), data=signed_distances)

def generate_sdf_samples(mesh_database, verbose=False):
    # Full list of objects.
    object_meshes = [filename for filename in os.listdir(mesh_database) if ".stl" in filename]
    objects = list(map(lambda x: x.replace('.stl', ''), object_meshes))
    print objects

    # Generate and save
    for object_ in tqdm(objects):
        # print object_
        get_signed_distance_points(object_, verbose)

def run():
    generate_sdf_samples(_MESH_DATABASE,
                         _VERBOSE)
    
if __name__ == '__main__':
    run()
