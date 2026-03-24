#!/usr/bin/env python3
"""Create a simple test STL file (a cube) for testing the unfolder."""

import numpy as np
import trimesh

# Create a simple cube mesh
cube = trimesh.creation.box(extents=[1, 1, 1])

# Save as STL
cube.export('/workspace/test_cube.stl')
print("Created test_cube.stl")

# Also create a pyramid
vertices = np.array([
    [0, 0, 1],      # apex
    [-0.5, -0.5, 0], # base corner 1
    [0.5, -0.5, 0],  # base corner 2
    [0.5, 0.5, 0],   # base corner 3
    [-0.5, 0.5, 0],  # base corner 4
])

faces = np.array([
    [0, 1, 2],  # side 1
    [0, 2, 3],  # side 2
    [0, 3, 4],  # side 3
    [0, 4, 1],  # side 4
    [1, 4, 3],  # base part 1
    [1, 3, 2],  # base part 2
])

pyramid = trimesh.Trimesh(vertices=vertices, faces=faces)
pyramid.export('/workspace/test_pyramid.stl')
print("Created test_pyramid.stl")
