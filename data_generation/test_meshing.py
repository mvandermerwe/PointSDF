#!/usr/bin/env python

import numpy as np
import mcubes

from generate_voxels import voxelize_mesh

# Shows simple marching cubes using PyMCubes

mesh_file = '/home/markvandermerwe/YCB/ycb_meshes/002_master_chef_can.stl'

voxels = voxelize_mesh(mesh_file, np.eye(4), 0.1, True)

vertices, triangles = mcubes.marching_cubes(voxels, 0)
mcubes.export_mesh(vertices, triangles, 'test.dae', 'test')
