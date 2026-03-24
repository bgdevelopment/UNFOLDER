#!/usr/bin/env python3
"""
STL Unfolder - Loads an STL file and unfolds its mesh to a 2D plane.

This program reads a 3D mesh from an STL file, unwraps it by laying out
all triangles flat on a 2D plane while preserving edge connectivity where possible.
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import json
import sys


def load_stl(filepath):
    """Load an STL file and return the mesh."""
    mesh = trimesh.load(filepath)
    
    # If it's a scene, extract the geometry
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Scene contains no geometry")
        # Combine all geometries into a single mesh
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    
    return mesh


def compute_triangle_2d_coords(vertices):
    """
    Compute 2D coordinates for a triangle given its 3D vertices.
    Places the first vertex at origin, second on x-axis, third computed from edge lengths.
    """
    v0, v1, v2 = vertices
    
    # Edge lengths
    a = np.linalg.norm(v2 - v1)  # opposite to v0
    b = np.linalg.norm(v2 - v0)  # opposite to v1
    c = np.linalg.norm(v1 - v0)  # opposite to v2
    
    # Place v0 at origin, v1 on x-axis
    p0 = np.array([0.0, 0.0])
    p1 = np.array([c, 0.0])
    
    # Compute position of p2 using law of cosines
    # p2 = (x, y) where x^2 + y^2 = b^2 and (x-c)^2 + y^2 = a^2
    x = (b**2 + c**2 - a**2) / (2 * c) if c > 1e-10 else 0.0
    y_sq = b**2 - x**2
    y = np.sqrt(max(0, y_sq))
    
    p2 = np.array([x, y])
    
    return np.array([p0, p1, p2])


def unfold_mesh_to_2d(mesh):
    """
    Unfold a 3D mesh to 2D by laying out triangles.
    
    This uses a simple approach: place each triangle in 2D space,
    trying to maintain connectivity with neighboring triangles.
    
    Returns:
        faces_2d: List of 2D triangle coordinates (each is 3x2 array)
        face_indices: Original face indices
    """
    vertices = mesh.vertices
    faces = mesh.faces
    
    n_faces = len(faces)
    
    # Build adjacency graph (which faces share edges)
    # Key: sorted edge tuple, Value: list of face indices
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1) % 3]]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Compute 2D coordinates for each triangle independently first
    faces_2d = []
    for fi, face in enumerate(faces):
        v_indices = face
        v_coords = vertices[v_indices]
        coords_2d = compute_triangle_2d_coords(v_coords)
        faces_2d.append(coords_2d)
    
    # Now try to align connected triangles
    # Use BFS to propagate transformations
    visited = [False] * n_faces
    transforms = [None] * n_faces  # (rotation, translation) for each face
    
    # Start with first face at identity transform
    start_face = 0
    visited[start_face] = True
    transforms[start_face] = (np.eye(2), np.array([0.0, 0.0]))
    
    queue = [start_face]
    
    while queue:
        current_face = queue.pop(0)
        current_verts_2d = faces_2d[current_face]
        curr_rot, curr_trans = transforms[current_face]
        
        # Apply current transform to get world-space 2D coords
        current_world = (curr_rot @ current_verts_2d.T).T + curr_trans
        
        # Find neighboring faces
        face_vertices = faces[current_face]
        for i in range(3):
            v1, v2 = face_vertices[i], face_vertices[(i+1) % 3]
            edge = tuple(sorted([v1, v2]))
            
            if edge in edge_to_faces:
                for neighbor_fi in edge_to_faces[edge]:
                    if neighbor_fi != current_face and not visited[neighbor_fi]:
                        # Compute transform to align this neighbor
                        neighbor_verts_2d = faces_2d[neighbor_fi]
                        neighbor_vertices = faces[neighbor_fi]
                        
                        # Find which edge in neighbor corresponds to this edge
                        n_v_indices = list(neighbor_vertices)
                        local_i1 = n_v_indices.index(v1)
                        local_i2 = n_v_indices.index(v2)
                        
                        # The shared edge should match
                        # Current world coords for v1, v2
                        curr_local_i = i
                        curr_local_j = (i + 1) % 3
                        world_v1 = current_world[curr_local_i]
                        world_v2 = current_world[curr_local_j]
                        
                        # Neighbor's local coords for these vertices
                        neighbor_v1_local = neighbor_verts_2d[local_i1]
                        neighbor_v2_local = neighbor_verts_2d[local_i2]
                        
                        # Compute rotation and translation to map neighbor edge to world edge
                        # Vector in neighbor local space
                        neighbor_edge_vec = neighbor_v2_local - neighbor_v1_local
                        # Vector in world space
                        world_edge_vec = world_v2 - world_v1
                        
                        # Compute rotation angle
                        norm_neighbor = np.linalg.norm(neighbor_edge_vec)
                        norm_world = np.linalg.norm(world_edge_vec)
                        
                        if norm_neighbor > 1e-10 and norm_world > 1e-10:
                            neighbor_edge_unit = neighbor_edge_vec / norm_neighbor
                            world_edge_unit = world_edge_vec / norm_world
                            
                            # 2D rotation angle
                            angle = np.arctan2(world_edge_unit[1], world_edge_unit[0]) - \
                                    np.arctan2(neighbor_edge_unit[1], neighbor_edge_unit[0])
                            
                            rot_matrix = np.array([
                                [np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]
                            ])
                            
                            # Rotate neighbor's v1 to match world v1
                            rotated_v1 = rot_matrix @ neighbor_v1_local
                            translation = world_v1 - rotated_v1
                            
                            # Check if we need to flip (for non-manifold or inconsistent orientation)
                            # Compute where v2 would end up
                            rotated_v2 = rot_matrix @ neighbor_v2_local + translation
                            if np.linalg.norm(rotated_v2 - world_v2) > 1e-6:
                                # Try flipping
                                angle += np.pi
                                rot_matrix = np.array([
                                    [np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]
                                ])
                                rotated_v1 = rot_matrix @ neighbor_v1_local
                                translation = world_v1 - rotated_v1
                            
                            transforms[neighbor_fi] = (rot_matrix, translation)
                            visited[neighbor_fi] = True
                            queue.append(neighbor_fi)
    
    # Apply transforms to all faces
    final_faces_2d = []
    for fi in range(n_faces):
        verts_2d = faces_2d[fi]
        if transforms[fi] is not None:
            rot, trans = transforms[fi]
            transformed = (rot @ verts_2d.T).T + trans
        else:
            # Unconnected face, keep original
            transformed = verts_2d
        final_faces_2d.append(transformed)
    
    return final_faces_2d, list(range(n_faces))


def compute_unfolded_bounds(faces_2d):
    """Compute bounding box of all 2D faces."""
    all_points = np.vstack([f for f in faces_2d])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    return min_coords, max_coords


def center_and_pack(faces_2d):
    """Center all faces around origin."""
    all_points = np.vstack([f for f in faces_2d])
    centroid = all_points.mean(axis=0)
    
    centered_faces = []
    for face in faces_2d:
        centered_faces.append(face - centroid)
    
    return centered_faces


def save_unfolded_as_svg(faces_2d, output_path, scale=1.0):
    """Save unfolded mesh as SVG file."""
    min_coords, max_coords = compute_unfolded_bounds(faces_2d)
    
    # Add some padding
    padding = 20
    width = (max_coords[0] - min_coords[0]) * scale + 2 * padding
    height = (max_coords[1] - min_coords[1]) * scale + 2 * padding
    
    # SVG header
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <g transform="translate({padding}, {padding})">',
    ]
    
    # Convert coordinates: flip Y for SVG coordinate system
    def to_svg_coords(point):
        x = (point[0] - min_coords[0]) * scale
        y = height - (point[1] - min_coords[1]) * scale - padding * 2 - (max_coords[1] - min_coords[1]) * scale
        # Actually, let's do it simpler
        y = (max_coords[1] - point[1]) * scale + padding
        return x, y
    
    # Draw each triangle
    for i, face in enumerate(faces_2d):
        points = [to_svg_coords(p) for p in face]
        path_d = f"M {points[0][0]:.2f},{points[0][1]:.2f} L {points[1][0]:.2f},{points[1][1]:.2f} L {points[2][0]:.2f},{points[2][1]:.2f} Z"
        
        # Random color for each face
        color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
        
        svg_lines.append(
            f'    <path d="{path_d}" fill="none" stroke="{color}" stroke-width="0.5" opacity="0.7"/>'
        )
    
    svg_lines.append('  </g>')
    svg_lines.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Saved SVG to: {output_path}")


def save_unfolded_as_json(faces_2d, face_indices, output_path):
    """Save unfolded mesh data as JSON."""
    data = {
        'faces': [
            {
                'index': idx,
                'vertices': face.tolist()
            }
            for idx, face in zip(face_indices, faces_2d)
        ],
        'num_faces': len(faces_2d)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved JSON to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python stl_unfolder.py <input.stl> [output_prefix]")
        print("  input.stl     - Path to input STL file")
        print("  output_prefix - Optional prefix for output files (default: 'unfolded')")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'unfolded'
    
    print(f"Loading STL from: {input_path}")
    
    try:
        mesh = load_stl(input_path)
    except Exception as e:
        print(f"Error loading STL: {e}")
        sys.exit(1)
    
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    print("Unfolding mesh to 2D...")
    faces_2d, face_indices = unfold_mesh_to_2d(mesh)
    
    print("Centering and packing...")
    faces_2d = center_and_pack(faces_2d)
    
    # Compute bounds
    min_coords, max_coords = compute_unfolded_bounds(faces_2d)
    print(f"Unfolded bounds: ({min_coords[0]:.2f}, {min_coords[1]:.2f}) to ({max_coords[0]:.2f}, {max_coords[1]:.2f})")
    print(f"Total area in 2D: {(max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1]):.2f} square units")
    
    # Save outputs
    svg_path = f"{output_prefix}.svg"
    json_path = f"{output_prefix}.json"
    
    save_unfolded_as_svg(faces_2d, svg_path, scale=10.0)
    save_unfolded_as_json(faces_2d, face_indices, json_path)
    
    print("\nUnfolding complete!")
    print(f"  - SVG visualization: {svg_path}")
    print(f"  - JSON data: {json_path}")


if __name__ == '__main__':
    main()
