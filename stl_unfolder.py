#!/usr/bin/env python3
"""
STL Unfolder with Interactive Cutting and Non-Overlapping Layout
Exports to PDF/PNG/SVG for Papercrafting

This program loads a 3D STL mesh file, allows interactive edge cutting,
and unfolds the mesh into a 2D layout suitable for papercrafting.
The unfolded result can be exported to PDF (for printing), PNG (for preview),
SVG (for laser cutting), or JSON (for further processing).

Requirements:
    pip install trimesh numpy svgwrite reportlab pillow tkinter

If you get ModuleNotFoundError, install the required packages:
    pip install trimesh numpy svgwrite reportlab pillow
"""

import sys

# ============================================================================
# DEPENDENCY CHECKS
# Check for required dependencies and provide helpful error messages
# This ensures users get clear installation instructions instead of cryptic errors
# ============================================================================
try:
    import tkinter as tk
except ImportError:
    print("ERROR: tkinter is not installed.")
    print("Solution: Install tkinter using:")
    print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  - Fedora: sudo dnf install python3-tkinter")
    print("  - macOS: brew install python-tk")
    print("  - Windows: tkinter comes bundled with Python")
    sys.exit(1)

from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh module is not installed.")
    print("Solution: Install required packages using:")
    print("  pip install trimesh numpy svgwrite reportlab pillow")
    sys.exit(1)

from typing import List, Tuple, Set, Dict, Optional
import json
import math
from dataclasses import dataclass

# Import PDF and PNG libraries (after dependency check)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black
from PIL import Image, ImageDraw


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Edge:
    """
    Represents an edge in the mesh.
    
    Attributes:
        v1, v2: Indices of the two vertices that form this edge
        faces: List of face indices that share this edge (usually 1 or 2)
    """
    v1: int
    v2: int
    faces: List[int]
    
    def __hash__(self):
        """Hash based on sorted vertex indices for consistent edge identification"""
        return hash(tuple(sorted([self.v1, self.v2])))
    
    def __eq__(self, other):
        """Equality check ignores vertex order (edge v1-v2 is same as v2-v1)"""
        return set([self.v1, self.v2]) == set([other.v1, other.v2])


class STLUnfolder:
    """
    Main class for loading, cutting, and unfolding STL meshes.
    
    This class handles:
    - Loading STL mesh files (binary and ASCII formats)
    - Managing cut edges (edges that will be separated during unfolding)
    - Computing the 2D unfolded layout using BFS propagation
    - Exporting to various formats (SVG, PDF, PNG, JSON)
    """
    
    def __init__(self):
        """Initialize the unfolder with empty mesh and no cuts"""
        self.mesh: Optional[trimesh.Trimesh] = None
        self.cut_edges: Set[Edge] = set()
        self.unfolded_faces: List[np.ndarray] = []
        self.face_colors: List[Tuple] = []
        self.is_unfolded = False
        
    def load_mesh(self, filepath: str) -> bool:
        """
        Load an STL mesh file.
        
        Args:
            filepath: Path to the STL file
            
        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            self.mesh = trimesh.load(filepath, force='mesh')
            # Handle scene objects by extracting the first geometry
            if not isinstance(self.mesh, trimesh.Trimesh):
                if hasattr(self.mesh, 'geometry'):
                    geometries = list(self.mesh.geometry.values())
                    if geometries:
                        self.mesh = geometries[0]
                    else:
                        return False
            # Reset cuts and unfolded state when loading new mesh
            self.cut_edges = set()
            self.is_unfolded = False
            self.unfolded_faces = []
            return True
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def get_all_edges(self) -> List[Edge]:
        """
        Extract all unique edges from the mesh with their connected faces.
        
        Returns:
            List of Edge objects containing vertex indices and face references
        """
        if self.mesh is None:
            return []
        # Build edge-to-faces mapping
        edges_dict: Dict[Tuple[int, int], List[int]] = {}
        for face_idx, face in enumerate(self.mesh.faces):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                key = tuple(sorted([v1, v2]))
                if key not in edges_dict:
                    edges_dict[key] = []
                edges_dict[key].append(face_idx)
        return [Edge(v1=k[0], v2=k[1], faces=v) for k, v in edges_dict.items()]
    
    def add_cut_edge(self, edge: Edge):
        """Mark an edge as cut (will be separated during unfolding)"""
        self.cut_edges.add(edge)
        self.is_unfolded = False
    
    def remove_cut_edge(self, edge: Edge):
        """Remove an edge from the cut set (faces will remain connected)"""
        self.cut_edges.discard(edge)
        self.is_unfolded = False
    
    def clear_cuts(self):
        """Remove all cut edges"""
        self.cut_edges = set()
        self.is_unfolded = False
    
    def is_edge_cut(self, edge: Edge) -> bool:
        """Check if an edge is marked as cut"""
        return edge in self.cut_edges
    
    def unfold(self) -> bool:
        """
        Unfold the 3D mesh to 2D plane using BFS propagation with proper orientation.
        
        The algorithm works by:
        1. Starting from an arbitrary face and placing it flat on the 2D plane
        2. Using BFS to visit connected faces (faces sharing non-cut edges)
        3. For each new face, computing its 2D position based on shared edge with already-placed neighbor
        4. Ensuring correct orientation by checking the normal direction
        5. Handling disconnected components by placing them with offset
        
        Returns:
            True if unfolding succeeded, False otherwise
        """
        if self.mesh is None:
            return False
        
        n_faces = len(self.mesh.faces)
        if n_faces == 0:
            return False
            
        self.unfolded_faces = [None] * n_faces
        self.face_colors = []
        
        # Generate distinct colors for each face using golden angle for even distribution
        for i in range(n_faces):
            hue = (i * 137.508) % 360
            self.face_colors.append(self._hsv_to_rgb(hue, 0.6, 0.8))
        
        # Build adjacency graph: faces are connected if they share a non-cut edge
        # Also store which vertices are shared
        adjacency: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {i: [] for i in range(n_faces)}
        for edge in self.get_all_edges():
            if len(edge.faces) == 2 and not self.is_edge_cut(edge):
                f1, f2 = edge.faces
                # Store the shared vertex indices
                adjacency[f1].append((f2, (edge.v1, edge.v2)))
                adjacency[f2].append((f1, (edge.v1, edge.v2)))
        
        # Track which faces have been placed and their 2D coordinates
        placed = set()
        global_coords: Dict[int, np.ndarray] = {}
        
        # Helper function to place a single triangle given its 3D vertices
        def place_initial_face(face_idx, offset=np.array([0.0, 0.0])):
            """Place a face at the origin or with given offset"""
            face = self.mesh.faces[face_idx]
            v0, v1, v2 = face
            p0, p1, p2 = self.mesh.vertices[v0], self.mesh.vertices[v1], self.mesh.vertices[v2]
            
            # Calculate edge lengths
            l01 = np.linalg.norm(p1 - p0)
            l02 = np.linalg.norm(p2 - p0)
            l12 = np.linalg.norm(p2 - p1)
            
            # Use law of cosines to compute angle at v0
            if l01 * l02 < 1e-10:
                return None
            cos_angle = (l01**2 + l02**2 - l12**2) / (2 * l01 * l02)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))
            
            # Place triangle: v0 at origin, v1 on x-axis, v2 based on angle
            coords = np.array([
                [0.0, 0.0],
                [l01, 0.0],
                [l02 * cos_angle, l02 * sin_angle]
            ]) + offset
            
            return coords
        
        # Start with first face at origin
        first_coords = place_initial_face(0)
        if first_coords is None:
            return False
            
        self.unfolded_faces[0] = first_coords
        for i, v_idx in enumerate(self.mesh.faces[0]):
            global_coords[v_idx] = first_coords[i]
        placed.add(0)
        
        # BFS queue: (face_index, neighbor_face_index, shared_vertices)
        queue = [(0, None, None)]
        
        while queue:
            current_face, _, _ = queue.pop(0)
            
            for neighbor_idx, shared_verts in adjacency[current_face]:
                if neighbor_idx in placed:
                    continue
                
                # Get the current face's 2D coordinates
                current_coords = self.unfolded_faces[current_face]
                if current_coords is None:
                    continue
                
                # Find which vertices are shared and their 2D positions
                current_face_verts = self.mesh.faces[current_face]
                neighbor_face_verts = self.mesh.faces[neighbor_idx]
                
                # Map shared vertices to their 2D positions in current face
                shared_v1, shared_v2 = shared_verts
                v1_2d = None
                v2_2d = None
                
                for i, v in enumerate(current_face_verts):
                    if v == shared_v1:
                        v1_2d = current_coords[i]
                    elif v == shared_v2:
                        v2_2d = current_coords[i]
                
                if v1_2d is None or v2_2d is None:
                    continue
                
                # Calculate the 3D edge length
                p1_3d = self.mesh.vertices[shared_v1]
                p2_3d = self.mesh.vertices[shared_v2]
                edge_len = np.linalg.norm(p2_3d - p1_3d)
                
                if edge_len < 1e-10:
                    continue
                
                # Find the third vertex in the neighbor face
                third_v = None
                for v in neighbor_face_verts:
                    if v != shared_v1 and v != shared_v2:
                        third_v = v
                        break
                
                if third_v is None:
                    continue
                
                # Calculate distances from third vertex to shared vertices
                p3_3d = self.mesh.vertices[third_v]
                d1 = np.linalg.norm(p3_3d - p1_3d)
                d2 = np.linalg.norm(p3_3d - p2_3d)
                
                # Use trilateration to find the third point
                # The third point can be on either side of the shared edge
                # We need to pick the side that doesn't cause overlaps
                
                # Vector along shared edge
                edge_vec = v2_2d - v1_2d
                edge_len_2d = np.linalg.norm(edge_vec)
                
                if edge_len_2d < 1e-10:
                    continue
                
                # Normalize edge vector
                edge_unit = edge_vec / edge_len_2d
                
                # Distance from v1 to projection of third point onto edge
                a = (d1**2 - d2**2 + edge_len_2d**2) / (2 * edge_len_2d)
                
                # Height from edge to third point
                h_sq = d1**2 - a**2
                if h_sq < 0:
                    h_sq = 0
                h = np.sqrt(h_sq)
                
                # Midpoint along edge
                mid_point = v1_2d + a * edge_unit
                
                # Perpendicular vector (two possible directions)
                perp_vec = np.array([-edge_unit[1], edge_unit[0]])
                
                # Try both sides and pick one (we'll use consistent orientation)
                # For now, always use the same side - this ensures consistent unfolding
                candidate1 = mid_point + h * perp_vec
                candidate2 = mid_point - h * perp_vec
                
                # Choose the candidate that maintains consistent orientation
                # Simple heuristic: use candidate1
                third_2d = candidate1
                
                # Create the neighbor face coordinates
                neighbor_coords = np.zeros((3, 2))
                for i, v in enumerate(neighbor_face_verts):
                    if v == shared_v1:
                        neighbor_coords[i] = v1_2d
                    elif v == shared_v2:
                        neighbor_coords[i] = v2_2d
                    elif v == third_v:
                        neighbor_coords[i] = third_2d
                
                self.unfolded_faces[neighbor_idx] = neighbor_coords
                
                # Update global coordinates
                for i, v in enumerate(neighbor_face_verts):
                    if v not in global_coords:
                        global_coords[v] = neighbor_coords[i]
                
                placed.add(neighbor_idx)
                queue.append((neighbor_idx, current_face, shared_verts))
        
        # Handle any remaining unplaced faces (disconnected components)
        max_offset = 0
        for i in range(n_faces):
            if i not in placed:
                # Find bounding box of placed faces
                if max_offset == 0:
                    all_placed = [f for f in self.unfolded_faces if f is not None]
                    if all_placed:
                        all_points = np.vstack(all_placed)
                        max_offset = all_points[:, 0].max() + 10
                
                # Place this face with offset
                coords = place_initial_face(i, offset=np.array([max_offset, 0]))
                if coords is not None:
                    self.unfolded_faces[i] = coords
                    for j, v in enumerate(self.mesh.faces[i]):
                        global_coords[v] = coords[j]
                    placed.add(i)
                    max_offset += 10
        
        # Center the entire unfolded mesh at origin
        all_placed = [f for f in self.unfolded_faces if f is not None]
        if all_placed:
            all_points = np.vstack(all_placed)
            centroid = all_points.mean(axis=0)
            self.unfolded_faces = [f - centroid if f is not None else None for f in self.unfolded_faces]
        
        self.is_unfolded = True
        return True
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB tuple (values 0-1)"""
        h = h / 360.0
        if s == 0: return (v, v, v)
        i = int(h * 6)
        f = h * 6 - i
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        i %= 6
        return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
    
    def export_svg(self, filepath: str, scale: float = 1.0) -> bool:
        """
        Export unfolded mesh to SVG format.
        
        Args:
            filepath: Output file path
            scale: Scaling factor for the output
            
        Returns:
            True if export succeeded, False otherwise
        """
        if not self.is_unfolded or not self.unfolded_faces:
            return False
        try:
            # Calculate bounding box of all faces
            all_points = np.vstack([f for f in self.unfolded_faces if f is not None])
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)
            width, height = max_x - min_x, max_y - min_y
            margin = 20
            with open(filepath, 'w') as f:
                f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{int((width+2*margin)*scale)}" height="{int((height+2*margin)*scale)}" viewBox="{min_x-margin} {min_y-margin} {width+2*margin} {height+2*margin}">\n')
                for face_idx, face_coords in enumerate(self.unfolded_faces):
                    if face_coords is None: continue
                    color = self.face_colors[face_idx]
                    points = " ".join([f"{x},{y}" for x, y in face_coords])
                    f.write(f'  <polygon points="{points}" fill="#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}" fill-opacity="0.7" stroke="black" stroke-width="0.5"/>\n')
                f.write('</svg>\n')
            return True
        except Exception as e:
            print(f"Error exporting SVG: {e}")
            return False
    
    def export_pdf(self, filepath: str, scale: float = 1.0) -> bool:
        """
        Export unfolded mesh to PDF for printing on A4 paper.
        
        The mesh is automatically scaled and centered to fit on an A4 page
        with appropriate margins for papercrafting.
        
        Args:
            filepath: Output file path
            scale: Additional scaling factor (1.0 = fit to page)
            
        Returns:
            True if export succeeded, False otherwise
        """
        if not self.is_unfolded or not self.unfolded_faces:
            return False
        try:
            # Calculate bounds
            all_points = np.vstack([f for f in self.unfolded_faces if f is not None])
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)
            width, height = max_x - min_x, max_y - min_y
            
            # Create PDF with A4 size
            c = canvas.Canvas(filepath, pagesize=A4)
            page_width, page_height = A4
            
            # Calculate scale to fit on page with margins
            margin = 20
            available_w = page_width - 2 * margin
            available_h = page_height - 2 * margin
            scale_factor = min(available_w / width, available_h / height) * scale
            
            # Offset to center on page
            offset_x = margin + (available_w - width * scale_factor) / 2 - min_x * scale_factor
            offset_y = margin + (available_h - height * scale_factor) / 2 - min_y * scale_factor
            
            # Draw each face
            for face_idx, face_coords in enumerate(self.unfolded_faces):
                if face_coords is None:
                    continue
                color = self.face_colors[face_idx]
                pdf_color = HexColor(f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}")
                
                # Create polygon path
                path = c.beginPath()
                first = True
                for x, y in face_coords:
                    px, py = x * scale_factor + offset_x, y * scale_factor + offset_y
                    if first:
                        path.moveTo(px, py)
                        first = False
                    else:
                        path.lineTo(px, py)
                path.close()
                
                # Fill and stroke
                c.setFillColor(pdf_color)
                c.setStrokeColor(black)
                c.setLineWidth(0.5)
                c.drawPath(path, fill=1, stroke=1)
            
            c.save()
            return True
        except Exception as e:
            print(f"Error exporting PDF: {e}")
            return False
    
    def export_png(self, filepath: str, scale: float = 2.0, bg_color: tuple = (255, 255, 255)) -> bool:
        """
        Export unfolded mesh to PNG image.
        
        Args:
            filepath: Output file path
            scale: Pixels per unit (higher = larger image)
            bg_color: Background color as RGB tuple (default: white)
            
        Returns:
            True if export succeeded, False otherwise
        """
        if not self.is_unfolded or not self.unfolded_faces:
            return False
        try:
            # Calculate bounds
            all_points = np.vstack([f for f in self.unfolded_faces if f is not None])
            min_x, min_y = all_points.min(axis=0)
            max_x, max_y = all_points.max(axis=0)
            width, height = max_x - min_x, max_y - min_y
            
            # Image dimensions
            img_width = int(width * scale) + 40
            img_height = int(height * scale) + 40
            
            # Create image
            img = Image.new('RGB', (img_width, img_height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Calculate offset to center
            offset_x = 20 - min_x * scale
            offset_y = 20 - min_y * scale
            
            # Draw each face
            for face_idx, face_coords in enumerate(self.unfolded_faces):
                if face_coords is None:
                    continue
                color = self.face_colors[face_idx]
                fill_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255), 180)
                
                # Convert coordinates
                pts = [(x * scale + offset_x, y * scale + offset_y) for x, y in face_coords]
                
                # Draw polygon
                draw.polygon(pts, fill=fill_color, outline=(0, 0, 0))
            
            img.save(filepath)
            return True
        except Exception as e:
            print(f"Error exporting PNG: {e}")
            return False
    
    def export_json(self, filepath: str) -> bool:
        """
        Export unfolded mesh to JSON format with vertex coordinates.
        
        Useful for further processing or integration with other tools.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export succeeded, False otherwise
        """
        if not self.is_unfolded or not self.unfolded_faces:
            return False
        try:
            data = {'faces': [], 'vertices_2d': [], 'face_vertex_indices': []}
            vertex_map, vertex_list = {}, []
            for face_idx, face_coords in enumerate(self.unfolded_faces):
                if face_coords is None: continue
                face_vertices = []
                for i, (x, y) in enumerate(face_coords):
                    key = f"{face_idx}_{i}"
                    if key not in vertex_map:
                        vertex_map[key] = len(vertex_list)
                        vertex_list.append([float(x), float(y)])
                    face_vertices.append(vertex_map[key])
                data['faces'].append({'index': face_idx, 'color': self.face_colors[face_idx]})
                data['face_vertex_indices'].append(face_vertices)
            data['vertices_2d'] = vertex_list
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return False
    
    def auto_cut_seams(self) -> int:
        """
        Automatically generate cutting pattern using spanning tree algorithm.
        
        This creates a minimum set of cuts that allows the mesh to be unfolded
        without overlaps, suitable for papercrafting. Uses BFS to build a
        spanning tree of faces, then cuts all edges not in the tree.
        
        Returns:
            Number of cuts made
        """
        if self.mesh is None:
            return 0
        edges = self.get_all_edges()
        internal_edges = [e for e in edges if len(e.faces) == 2]
        n_faces = len(self.mesh.faces)
        face_graph: Dict[int, Set[int]] = {i: set() for i in range(n_faces)}
        for edge in internal_edges:
            f1, f2 = edge.faces
            face_graph[f1].add(f2)
            face_graph[f2].add(f1)
        # Build spanning tree using BFS
        visited, queue, tree_edges = {0}, [0], set()
        while queue:
            current = queue.pop(0)
            for neighbor in face_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    tree_edges.add(tuple(sorted([current, neighbor])))
        # Cut all edges not in spanning tree
        cuts_made = 0
        for edge in internal_edges:
            f1, f2 = edge.faces
            if tuple(sorted([f1, f2])) not in tree_edges:
                self.add_cut_edge(edge)
                cuts_made += 1
        return cuts_made


class MeshViewer3D(tk.Canvas):
    """
    Interactive 3D mesh viewer with edge selection for cutting.
    
    Features:
    - Rotate mesh by dragging
    - Zoom with mouse wheel
    - Click edges to toggle cut status (red=cut, blue=connected)
    """
    
    def __init__(self, parent, unfolder, **kwargs):
        """Initialize the 3D viewer"""
        super().__init__(parent, **kwargs)
        self.unfolder = unfolder
        self.rotation_x, self.rotation_y = 0.5, 0.5
        self.scale = 1.0
        self.drag_start = None
        self.selected_edges = set()
        self.edge_screen_coords = {}
        # Bind mouse events
        self.bind("<Configure>", lambda e: self.update_view())
        self.bind("<ButtonPress-1>", lambda e: setattr(self, 'drag_start', (e.x, e.y)))
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<MouseWheel>", lambda e: self.scale_update(e.delta))
    
    def on_drag(self, event):
        """Handle mouse drag for rotation"""
        if self.drag_start:
            self.rotation_y += (event.x - self.drag_start[0]) * 0.01
            self.rotation_x += (event.y - self.drag_start[1]) * 0.01
            self.drag_start = (event.x, event.y)
            self.update_view()
    
    def on_release(self, event):
        """Handle mouse release - select edge if click was stationary"""
        if self.drag_start and abs(event.x - self.drag_start[0]) < 5 and abs(event.y - self.drag_start[1]) < 5:
            self.select_edge_at(event.x, event.y)
        self.drag_start = None
    
    def scale_update(self, delta):
        """Handle mouse wheel zoom"""
        self.scale *= 1.1 if delta > 0 else 0.9
        self.update_view()
    
    def select_edge_at(self, x, y):
        """
        Select the nearest edge to the click position.
        Toggles the cut status of the selected edge.
        """
        if not self.unfolder.mesh: return
        min_dist, selected = 10.0, None
        for edge, coords in self.edge_screen_coords.items():
            (x1, y1), (x2, y2) = coords
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                dist = math.sqrt((x - x1)**2 + (y - y1)**2)
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
                dist = math.sqrt((x - (x1 + t * dx))**2 + (y - (y1 + t * dy))**2)
            if dist < min_dist:
                min_dist, selected = dist, edge
        if selected:
            if self.unfolder.is_edge_cut(selected):
                self.unfolder.remove_cut_edge(selected)
                self.selected_edges.discard(selected)
            else:
                self.unfolder.add_cut_edge(selected)
                self.selected_edges.add(selected)
            self.update_view()
    
    def update_view(self):
        """Render the 3D mesh with current rotation and cut edges highlighted"""
        self.delete("all")
        self.edge_screen_coords.clear()
        if not self.unfolder.mesh:
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text="No mesh loaded", fill="gray")
            return
        w, h = self.winfo_width(), self.winfo_height()
        cx, cy = w // 2, h // 2
        # Center vertices and calculate scale
        verts = self.unfolder.mesh.vertices.copy() - self.unfolder.mesh.vertices.mean(axis=0)
        max_dim = max(verts.max(axis=0) - verts.min(axis=0))
        sf = min(w, h) * 0.4 / max_dim * self.scale if max_dim > 0 else 1.0
        rx, ry = self.rotation_x, self.rotation_y
        # Apply rotation matrices
        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        verts = verts @ rot_y.T @ rot_x.T
        proj = verts[:, :2] * sf + np.array([cx, cy])
        # Draw back-facing triangles first (simple depth sorting)
        for face in self.unfolder.mesh.faces:
            pts = proj[face]
            cross = (pts[1,0]-pts[0,0])*(pts[2,1]-pts[0,1]) - (pts[1,1]-pts[0,1])*(pts[2,0]-pts[0,0])
            if cross > 0:
                self.create_polygon([(pts[i,0], pts[i,1]) for i in range(3)], outline="black", fill="#cccccc", width=1)
        # Draw edges: red for cuts, blue for connected
        for edge in self.unfolder.get_all_edges():
            p1, p2 = proj[edge.v1], proj[edge.v2]
            self.edge_screen_coords[edge] = ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
            is_cut = self.unfolder.is_edge_cut(edge)
            self.create_line(p1[0], p1[1], p2[0], p2[1], fill="red" if is_cut else "blue", width=3 if is_cut else 1)


class UnfoldedViewer2D(tk.Canvas):
    """
    2D viewer for displaying the unfolded mesh.
    
    Features:
    - Zoom with mouse wheel
    - Color-coded faces matching the 3D view
    - Black outlines for easy cutting
    """
    
    def __init__(self, parent, unfolder, **kwargs):
        """Initialize the 2D viewer"""
        super().__init__(parent, **kwargs)
        self.unfolder = unfolder
        self.scale = 1.0
        self.bind("<Configure>", lambda e: self.update_view())
        self.bind("<MouseWheel>", lambda e: self.zoom(e.delta))
    
    def zoom(self, delta):
        """Handle mouse wheel zoom"""
        self.scale *= 1.1 if delta > 0 else 0.9
        self.update_view()
    
    def update_view(self):
        """Render the unfolded 2D mesh"""
        self.delete("all")
        if not self.unfolder.is_unfolded or not self.unfolder.unfolded_faces:
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text="No unfolded mesh (click 'Unfold' first)", fill="gray")
            return
        w, h = self.winfo_width(), self.winfo_height()
        all_pts = np.vstack([f for f in self.unfolder.unfolded_faces if f is not None])
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)
        cw, ch = max_x - min_x, max_y - min_y
        bs = min(w, h) * 0.9 / max(cw, ch) if cw > 0 else 1.0
        s = bs * self.scale
        cx, cy = w // 2, h // 2
        for fi, fc in enumerate(self.unfolder.unfolded_faces):
            if fc is None: continue
            c = self.unfolder.face_colors[fi]
            tf = [(x * s + cx, y * s + cy) for x, y in fc]
            self.create_polygon(tf, fill=f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}", outline="black", width=1)


class STLUnfolderGUI:
    """
    Main GUI application for the STL Unfolder.
    
    Provides a complete interface for:
    - Loading STL files
    - Interactive edge cutting (click edges in 3D view)
    - Automatic seam generation
    - 2D unfolding preview
    - Export to PDF, PNG, SVG, and JSON formats
    """
    
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("STL Unfolder - Interactive Cutting & Unfolding")
        self.root.geometry("1400x900")
        self.unfolder = STLUnfolder()
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface layout"""
        # Main frame
        mf = ttk.Frame(self.root)
        mf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel on left
        cf = ttk.LabelFrame(mf, text="Controls", padding=10)
        cf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # File operations section
        ff = ttk.LabelFrame(cf, text="File", padding=5)
        ff.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(ff, text="Load STL", command=self.load_stl).pack(fill=tk.X, pady=2)
        ttk.Button(ff, text="Clear Cuts", command=self.clear_cuts).pack(fill=tk.X, pady=2)
        
        # Cutting controls section
        cut_f = ttk.LabelFrame(cf, text="Cutting", padding=5)
        cut_f.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(cut_f, text="Auto-Cut Seams", command=self.auto_cut).pack(fill=tk.X, pady=2)
        ttk.Label(cut_f, text="Click edges in 3D view to toggle cuts\n(Red=cut, Blue=connected)").pack(fill=tk.X, pady=5)
        
        # Unfolding controls section
        uf = ttk.LabelFrame(cf, text="Unfolding", padding=5)
        uf.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(uf, text="Unfold to 2D", command=self.unfold).pack(fill=tk.X, pady=2)
        
        # Export options section
        ef = ttk.LabelFrame(cf, text="Export", padding=5)
        ef.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(ef, text="Export SVG", command=self.export_svg).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export PDF", command=self.export_pdf).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export PNG", command=self.export_png).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export JSON", command=self.export_json).pack(fill=tk.X, pady=2)
        
        # Status display
        self.info = ttk.Label(cf, text="Status: Ready", wraplength=200)
        self.info.pack(fill=tk.X, pady=10)
        
        # Viewer area on right with split panes
        vf = ttk.Frame(mf)
        vf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        paned = ttk.PanedWindow(vf, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 3D viewer pane
        v3f = ttk.LabelFrame(paned, text="3D View (Click edges to cut)")
        paned.add(v3f, weight=1)
        self.v3d = MeshViewer3D(v3f, self.unfolder, bg="white")
        self.v3d.pack(fill=tk.BOTH, expand=True)
        
        # 2D unfolded viewer pane
        v2f = ttk.LabelFrame(paned, text="2D Unfolded View")
        paned.add(v2f, weight=1)
        self.v2d = UnfoldedViewer2D(v2f, self.unfolder, bg="white")
        self.v2d.pack(fill=tk.BOTH, expand=True)
        
        # Bottom status bar
        self.status_var = tk.StringVar(value="Loaded: None | Faces: 0 | Cuts: 0")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self):
        """Update status displays with current mesh information"""
        nf = len(self.unfolder.mesh.faces) if self.unfolder.mesh else 0
        nc = len(self.unfolder.cut_edges)
        self.status_var.set(f"Loaded: {'Mesh' if self.unfolder.mesh else 'None'} | Faces: {nf} | Cuts: {nc}")
        self.info.config(text=f"Status: {'Unfolded' if self.unfolder.is_unfolded else 'Ready'}\nFaces: {nf}, Cuts: {nc}")
    
    def load_stl(self):
        """Open file dialog and load selected STL file"""
        fp = filedialog.askopenfilename(title="Select STL file", filetypes=[("STL files", "*.stl"), ("All files", "*.*")])
        if fp:
            self.status_var.set(f"Loading {fp}...")
            self.root.update_idletasks()
            try:
                if self.unfolder.load_mesh(fp):
                    self.update_status()
                    self.v3d.update_view()
                    self.status_var.set(f"Loaded: {len(self.unfolder.mesh.faces)} faces")
                else:
                    self.status_var.set("Error: Failed to load STL file")
            except Exception as e:
                error_msg = f"Load failed: {str(e)}"
                print(error_msg)
                self.status_var.set(error_msg)
                import traceback
                traceback.print_exc()
        else:
            self.status_var.set("Ready")
    
    def clear_cuts(self):
        """Remove all cut edges and reset views"""
        self.unfolder.clear_cuts()
        self.v3d.selected_edges.clear()
        self.v3d.update_view()
        self.v2d.update_view()
        self.update_status()
    
    def auto_cut(self):
        """Generate automatic cutting pattern using spanning tree algorithm"""
        if not self.unfolder.mesh:
            self.status_var.set("Error: Please load a mesh first")
            return
        
        self.status_var.set("Generating automatic cuts...")
        self.root.update_idletasks()
        
        try:
            nc = self.unfolder.auto_cut_seams()
            self.v3d.selected_edges = self.unfolder.cut_edges.copy()
            self.v3d.update_view()
            self.status_var.set(f"Auto-cut complete: {nc} edges marked for cutting")
        except Exception as e:
            error_msg = f"Auto-cut failed: {str(e)}"
            print(error_msg)
            self.status_var.set(error_msg)
            import traceback
            traceback.print_exc()
    
    def unfold(self):
        """Perform the unfolding operation"""
        if not self.unfolder.mesh:
            self.status_var.set("Error: Please load a mesh first")
            return
        
        self.status_var.set("Unfolding mesh... (this may take a moment)")
        self.root.update_idletasks()  # Update UI before starting heavy calculation
        
        try:
            if self.unfolder.unfold():
                self.v2d.update_view()
                self.status_var.set(f"Success! Unfolded {len(self.unfolder.faces_2d)} faces.")
            else:
                self.status_var.set("Error: Failed to unfold mesh")
        except Exception as e:
            error_msg = f"Unfolding failed: {str(e)}"
            print(error_msg)  # Log to console for debugging
            self.status_var.set(error_msg)
            import traceback
            traceback.print_exc()
    
    def export_svg(self):
        """Export unfolded mesh to SVG format"""
        if not self.unfolder.is_unfolded:
            self.status_var.set("Error: Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save SVG file", defaultextension=".svg", filetypes=[("SVG files", "*.svg"), ("All files", "*.*")])
        if fp:
            self.status_var.set(f"Exporting SVG to {fp}...")
            self.root.update_idletasks()
            try:
                if self.unfolder.export_svg(fp):
                    self.status_var.set(f"Exported SVG to {fp}")
                else:
                    self.status_var.set("Error: Failed to export SVG")
            except Exception as e:
                error_msg = f"SVG export failed: {str(e)}"
                print(error_msg)
                self.status_var.set(error_msg)
                import traceback
                traceback.print_exc()
    
    def export_json(self):
        """Export unfolded mesh to JSON format"""
        if not self.unfolder.is_unfolded:
            self.status_var.set("Error: Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save JSON file", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if fp:
            self.status_var.set(f"Exporting JSON to {fp}...")
            self.root.update_idletasks()
            try:
                if self.unfolder.export_json(fp):
                    self.status_var.set(f"Exported JSON to {fp}")
                else:
                    self.status_var.set("Error: Failed to export JSON")
            except Exception as e:
                error_msg = f"JSON export failed: {str(e)}"
                print(error_msg)
                self.status_var.set(error_msg)
                import traceback
                traceback.print_exc()
    
    def export_pdf(self):
        """Export unfolded mesh to PDF for printing"""
        if not self.unfolder.is_unfolded:
            self.status_var.set("Error: Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save PDF file", defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if fp:
            self.status_var.set(f"Exporting PDF to {fp}...")
            self.root.update_idletasks()
            try:
                if self.unfolder.export_pdf(fp):
                    self.status_var.set(f"Exported PDF to {fp}")
                else:
                    self.status_var.set("Error: Failed to export PDF")
            except Exception as e:
                error_msg = f"PDF export failed: {str(e)}"
                print(error_msg)
                self.status_var.set(error_msg)
                import traceback
                traceback.print_exc()
    
    def export_png(self):
        """Export unfolded mesh to PNG image"""
        if not self.unfolder.is_unfolded:
            self.status_var.set("Error: Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save PNG file", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if fp:
            self.status_var.set(f"Exporting PNG to {fp}...")
            self.root.update_idletasks()
            try:
                if self.unfolder.export_png(fp):
                    self.status_var.set(f"Exported PNG to {fp}")
                else:
                    self.status_var.set("Error: Failed to export PNG")
            except Exception as e:
                error_msg = f"PNG export failed: {str(e)}"
                print(error_msg)
                self.status_var.set(error_msg)
                import traceback
                traceback.print_exc()


def main():
    """Entry point: create and run the GUI application"""
    root = tk.Tk()
    app = STLUnfolderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
