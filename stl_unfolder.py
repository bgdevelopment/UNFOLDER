#!/usr/bin/env python3
"""
STL Unfolder with Interactive Cutting and GUI Interface

Requirements:
    pip install trimesh numpy svgwrite tkinter

If you get ModuleNotFoundError, install the required packages:
    pip install trimesh numpy svgwrite
"""

import sys

# Check for required dependencies and provide helpful error message
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
    print("  pip install trimesh numpy svgwrite")
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


@dataclass
class Edge:
    v1: int
    v2: int
    faces: List[int]
    
    def __hash__(self):
        return hash(tuple(sorted([self.v1, self.v2])))
    
    def __eq__(self, other):
        return set([self.v1, self.v2]) == set([other.v1, other.v2])


class STLUnfolder:
    def __init__(self):
        self.mesh: Optional[trimesh.Trimesh] = None
        self.cut_edges: Set[Edge] = set()
        self.unfolded_faces: List[np.ndarray] = []
        self.face_colors: List[Tuple] = []
        self.is_unfolded = False
        
    def load_mesh(self, filepath: str) -> bool:
        try:
            self.mesh = trimesh.load(filepath, force='mesh')
            if not isinstance(self.mesh, trimesh.Trimesh):
                if hasattr(self.mesh, 'geometry'):
                    geometries = list(self.mesh.geometry.values())
                    if geometries:
                        self.mesh = geometries[0]
                    else:
                        return False
            self.cut_edges = set()
            self.is_unfolded = False
            self.unfolded_faces = []
            return True
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def get_all_edges(self) -> List[Edge]:
        if self.mesh is None:
            return []
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
        self.cut_edges.add(edge)
        self.is_unfolded = False
    
    def remove_cut_edge(self, edge: Edge):
        self.cut_edges.discard(edge)
        self.is_unfolded = False
    
    def clear_cuts(self):
        self.cut_edges = set()
        self.is_unfolded = False
    
    def is_edge_cut(self, edge: Edge) -> bool:
        return edge in self.cut_edges
    
    def unfold(self) -> bool:
        if self.mesh is None:
            return False
        n_faces = len(self.mesh.faces)
        self.unfolded_faces = [None] * n_faces
        self.face_colors = []
        for i in range(n_faces):
            hue = (i * 137.508) % 360
            self.face_colors.append(self._hsv_to_rgb(hue, 0.6, 0.8))
        
        adjacency: Dict[int, List[int]] = {i: [] for i in range(n_faces)}
        for edge in self.get_all_edges():
            if len(edge.faces) == 2 and not self.is_edge_cut(edge):
                f1, f2 = edge.faces
                adjacency[f1].append(f2)
                adjacency[f2].append(f1)
        
        visited = set()
        queue = [0]
        visited.add(0)
        
        face0 = self.mesh.faces[0]
        v0, v1, v2 = face0
        p0, p1, p2 = self.mesh.vertices[v0], self.mesh.vertices[v1], self.mesh.vertices[v2]
        l01 = np.linalg.norm(p1 - p0)
        l02 = np.linalg.norm(p2 - p0)
        l12 = np.linalg.norm(p2 - p1)
        cos_angle = (l01**2 + l02**2 - l12**2) / (2 * l01 * l02 + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        self.unfolded_faces[0] = np.array([[0.0, 0.0], [l01, 0.0], [l02 * cos_angle, l02 * np.sqrt(max(0.0, 1.0 - cos_angle**2))]])
        
        global_coords: Dict[int, np.ndarray] = {}
        for i, coord in enumerate(self.unfolded_faces[0]):
            global_coords[self.mesh.faces[0][i]] = coord
        
        while queue:
            current_face = queue.pop(0)
            for neighbor in adjacency[current_face]:
                if neighbor not in visited:
                    shared_verts = set(self.mesh.faces[current_face]) & set(self.mesh.faces[neighbor])
                    neighbor_coords = {v: global_coords[v] for v in shared_verts if v in global_coords}
                    if neighbor_coords:
                        new_coords = self._compute_face_transform(neighbor, neighbor_coords, global_coords)
                        if new_coords is not None:
                            self.unfolded_faces[neighbor] = new_coords
                            for i, v_idx in enumerate(self.mesh.faces[neighbor]):
                                if v_idx not in global_coords:
                                    global_coords[v_idx] = new_coords[i]
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        for i in range(n_faces):
            if i not in visited:
                queue = [i]
                visited.add(i)
                face = self.mesh.faces[i]
                v0, v1, v2 = face
                p0, p1, p2 = self.mesh.vertices[v0], self.mesh.vertices[v1], self.mesh.vertices[v2]
                l01, l02, l12 = np.linalg.norm(p1-p0), np.linalg.norm(p2-p0), np.linalg.norm(p2-p1)
                cos_angle = (l01**2 + l02**2 - l12**2) / (2 * l01 * l02 + 1e-10)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                offset_x = max([fc[:, 0].max() for fc in self.unfolded_faces if fc is not None], default=0) + 10
                self.unfolded_faces[i] = np.array([[offset_x, 0.0], [offset_x + l01, 0.0], [offset_x + l02 * cos_angle, l02 * np.sqrt(max(0.0, 1.0 - cos_angle**2))]])
                for j, v_idx in enumerate(self.mesh.faces[i]):
                    if v_idx not in global_coords:
                        global_coords[v_idx] = self.unfolded_faces[i][j]
                while queue:
                    current_face = queue.pop(0)
                    for neighbor in adjacency[current_face]:
                        if neighbor not in visited:
                            shared_verts = set(self.mesh.faces[current_face]) & set(self.mesh.faces[neighbor])
                            neighbor_coords = {v: global_coords[v] for v in shared_verts if v in global_coords}
                            if neighbor_coords:
                                new_coords = self._compute_face_transform(neighbor, neighbor_coords, global_coords)
                                if new_coords is not None:
                                    self.unfolded_faces[neighbor] = new_coords
                                    for k, v_idx in enumerate(self.mesh.faces[neighbor]):
                                        if v_idx not in global_coords:
                                            global_coords[v_idx] = new_coords[k]
                                    visited.add(neighbor)
                                    queue.append(neighbor)
        
        all_points = np.vstack([f for f in self.unfolded_faces if f is not None])
        centroid = all_points.mean(axis=0)
        self.unfolded_faces = [f - centroid if f is not None else None for f in self.unfolded_faces]
        self.is_unfolded = True
        return True
    
    def _compute_face_transform(self, face_idx, neighbor_coords, global_coords):
        if self.mesh is None:
            return None
        face = self.mesh.faces[face_idx]
        v0, v1, v2 = face
        p0, p1, p2 = self.mesh.vertices[v0], self.mesh.vertices[v1], self.mesh.vertices[v2]
        l01, l02, l12 = np.linalg.norm(p1-p0), np.linalg.norm(p2-p0), np.linalg.norm(p2-p1)
        
        placed = {}
        for vi in [v0, v1, v2]:
            if vi in neighbor_coords:
                placed[vi] = neighbor_coords[vi]
        
        if len(placed) >= 2:
            placed_vs = list(placed.keys())[:2]
            va, vb = placed_vs[0], placed_vs[1]
            verts = [v0, v1, v2]
            vc = [v for v in [v0, v1, v2] if v not in placed][0]
            if vc == v0: d_av, d_bv = l01, l02
            elif vc == v1: d_av, d_bv = l01, l12
            else: d_av, d_bv = l02, l12
            pa, pb = placed[va], placed[vb]
            d_between = np.linalg.norm(pb - pa)
            if d_between < 1e-10:
                return None
            a = (d_av**2 - d_bv**2 + d_between**2) / (2 * d_between)
            h = np.sqrt(max(0.0, d_av**2 - a**2))
            p_mid = pa + a * (pb - pa) / d_between
            offset = h * np.array([-(pb[1] - pa[1]), (pb[0] - pa[0])]) / d_between
            placed[vc] = p_mid + offset
        elif len(placed) == 1:
            anchor_v = list(placed.keys())[0]
            other_vs = [v for v in [v0, v1, v2] if v != anchor_v]
            second_v = other_vs[0]
            dist = l01 if (anchor_v == v0 and second_v == v1) or (anchor_v == v1 and second_v == v0) else l02 if (anchor_v == v0 and second_v == v2) or (anchor_v == v2 and second_v == v0) else l12
            placed[second_v] = placed[anchor_v] + np.array([dist, 0.0])
            third_v = other_vs[1]
            if third_v == v0: d_a, d_b = (l01, l02) if second_v == v1 else (l02, l01)
            elif third_v == v1: d_a, d_b = (l01, l12) if second_v == v0 else (l12, l01)
            else: d_a, d_b = (l02, l12) if second_v == v0 else (l12, l02)
            pa, pb = placed[anchor_v], placed[second_v]
            d_between = np.linalg.norm(pb - pa)
            if d_between < 1e-10:
                placed[third_v] = pa + np.array([d_a, 0.0])
            else:
                a_val = (d_a**2 - d_b**2 + d_between**2) / (2 * d_between)
                h = np.sqrt(max(0.0, d_a**2 - a_val**2))
                p_mid = pa + a_val * (pb - pa) / d_between
                offset = h * np.array([-(pb[1] - pa[1]), (pb[0] - pa[0])]) / d_between
                placed[third_v] = p_mid + offset
        else:
            placed[v0] = np.array([0.0, 0.0])
            placed[v1] = np.array([l01, 0.0])
            cos_angle = (l01**2 + l02**2 - l12**2) / (2 * l01 * l02 + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            placed[v2] = np.array([l02 * cos_angle, l02 * np.sqrt(max(0.0, 1.0 - cos_angle**2))])
        
        result = np.zeros((3, 2))
        result[0], result[1], result[2] = placed[v0], placed[v1], placed[v2]
        return result
    
    def _hsv_to_rgb(self, h, s, v):
        h = h / 360.0
        if s == 0: return (v, v, v)
        i = int(h * 6)
        f = h * 6 - i
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        i %= 6
        return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
    
    def export_svg(self, filepath: str, scale: float = 1.0) -> bool:
        if not self.is_unfolded or not self.unfolded_faces:
            return False
        try:
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
        """Export unfolded mesh to PDF for printing"""
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
        """Export unfolded mesh to PNG image"""
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
        visited, queue, tree_edges = {0}, [0], set()
        while queue:
            current = queue.pop(0)
            for neighbor in face_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    tree_edges.add(tuple(sorted([current, neighbor])))
        cuts_made = 0
        for edge in internal_edges:
            f1, f2 = edge.faces
            if tuple(sorted([f1, f2])) not in tree_edges:
                self.add_cut_edge(edge)
                cuts_made += 1
        return cuts_made


class MeshViewer3D(tk.Canvas):
    def __init__(self, parent, unfolder, **kwargs):
        super().__init__(parent, **kwargs)
        self.unfolder = unfolder
        self.rotation_x, self.rotation_y = 0.5, 0.5
        self.scale = 1.0
        self.drag_start = None
        self.selected_edges = set()
        self.edge_screen_coords = {}
        self.bind("<Configure>", lambda e: self.update_view())
        self.bind("<ButtonPress-1>", lambda e: setattr(self, 'drag_start', (e.x, e.y)))
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<MouseWheel>", lambda e: self.scale_update(e.delta))
    
    def on_drag(self, event):
        if self.drag_start:
            self.rotation_y += (event.x - self.drag_start[0]) * 0.01
            self.rotation_x += (event.y - self.drag_start[1]) * 0.01
            self.drag_start = (event.x, event.y)
            self.update_view()
    
    def on_release(self, event):
        if self.drag_start and abs(event.x - self.drag_start[0]) < 5 and abs(event.y - self.drag_start[1]) < 5:
            self.select_edge_at(event.x, event.y)
        self.drag_start = None
    
    def scale_update(self, delta):
        self.scale *= 1.1 if delta > 0 else 0.9
        self.update_view()
    
    def select_edge_at(self, x, y):
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
        self.delete("all")
        self.edge_screen_coords.clear()
        if not self.unfolder.mesh:
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text="No mesh loaded", fill="gray")
            return
        w, h = self.winfo_width(), self.winfo_height()
        cx, cy = w // 2, h // 2
        verts = self.unfolder.mesh.vertices.copy() - self.unfolder.mesh.vertices.mean(axis=0)
        max_dim = max(verts.max(axis=0) - verts.min(axis=0))
        sf = min(w, h) * 0.4 / max_dim * self.scale if max_dim > 0 else 1.0
        rx, ry = self.rotation_x, self.rotation_y
        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        verts = verts @ rot_y.T @ rot_x.T
        proj = verts[:, :2] * sf + np.array([cx, cy])
        for face in self.unfolder.mesh.faces:
            pts = proj[face]
            cross = (pts[1,0]-pts[0,0])*(pts[2,1]-pts[0,1]) - (pts[1,1]-pts[0,1])*(pts[2,0]-pts[0,0])
            if cross > 0:
                self.create_polygon([(pts[i,0], pts[i,1]) for i in range(3)], outline="black", fill="#cccccc", width=1)
        for edge in self.unfolder.get_all_edges():
            p1, p2 = proj[edge.v1], proj[edge.v2]
            self.edge_screen_coords[edge] = ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
            is_cut = self.unfolder.is_edge_cut(edge)
            self.create_line(p1[0], p1[1], p2[0], p2[1], fill="red" if is_cut else "blue", width=3 if is_cut else 1)


class UnfoldedViewer2D(tk.Canvas):
    def __init__(self, parent, unfolder, **kwargs):
        super().__init__(parent, **kwargs)
        self.unfolder = unfolder
        self.scale = 1.0
        self.bind("<Configure>", lambda e: self.update_view())
        self.bind("<MouseWheel>", lambda e: self.zoom(e.delta))
    
    def zoom(self, delta):
        self.scale *= 1.1 if delta > 0 else 0.9
        self.update_view()
    
    def update_view(self):
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
    def __init__(self, root):
        self.root = root
        self.root.title("STL Unfolder - Interactive Cutting & Unfolding")
        self.root.geometry("1400x900")
        self.unfolder = STLUnfolder()
        self.setup_ui()
    
    def setup_ui(self):
        mf = ttk.Frame(self.root)
        mf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        cf = ttk.LabelFrame(mf, text="Controls", padding=10)
        cf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        ff = ttk.LabelFrame(cf, text="File", padding=5)
        ff.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(ff, text="Load STL", command=self.load_stl).pack(fill=tk.X, pady=2)
        ttk.Button(ff, text="Clear Cuts", command=self.clear_cuts).pack(fill=tk.X, pady=2)
        cut_f = ttk.LabelFrame(cf, text="Cutting", padding=5)
        cut_f.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(cut_f, text="Auto-Cut Seams", command=self.auto_cut).pack(fill=tk.X, pady=2)
        ttk.Label(cut_f, text="Click edges in 3D view to toggle cuts\n(Red=cut, Blue=connected)").pack(fill=tk.X, pady=5)
        uf = ttk.LabelFrame(cf, text="Unfolding", padding=5)
        uf.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(uf, text="Unfold to 2D", command=self.unfold).pack(fill=tk.X, pady=2)
        ef = ttk.LabelFrame(cf, text="Export", padding=5)
        ef.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(ef, text="Export SVG", command=self.export_svg).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export PDF", command=self.export_pdf).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export PNG", command=self.export_png).pack(fill=tk.X, pady=2)
        ttk.Button(ef, text="Export JSON", command=self.export_json).pack(fill=tk.X, pady=2)
        self.info = ttk.Label(cf, text="Status: Ready", wraplength=200)
        self.info.pack(fill=tk.X, pady=10)
        vf = ttk.Frame(mf)
        vf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        paned = ttk.PanedWindow(vf, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        v3f = ttk.LabelFrame(paned, text="3D View (Click edges to cut)")
        paned.add(v3f, weight=1)
        self.v3d = MeshViewer3D(v3f, self.unfolder, bg="white")
        self.v3d.pack(fill=tk.BOTH, expand=True)
        v2f = ttk.LabelFrame(paned, text="2D Unfolded View")
        paned.add(v2f, weight=1)
        self.v2d = UnfoldedViewer2D(v2f, self.unfolder, bg="white")
        self.v2d.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="Loaded: None | Faces: 0 | Cuts: 0")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self):
        nf = len(self.unfolder.mesh.faces) if self.unfolder.mesh else 0
        nc = len(self.unfolder.cut_edges)
        self.status_var.set(f"Loaded: {'Mesh' if self.unfolder.mesh else 'None'} | Faces: {nf} | Cuts: {nc}")
        self.info.config(text=f"Status: {'Unfolded' if self.unfolder.is_unfolded else 'Ready'}\nFaces: {nf}, Cuts: {nc}")
    
    def load_stl(self):
        fp = filedialog.askopenfilename(title="Select STL file", filetypes=[("STL files", "*.stl"), ("All files", "*.*")])
        if fp and self.unfolder.load_mesh(fp):
            self.update_status()
            self.v3d.update_view()
            messagebox.showinfo("Success", f"Loaded mesh with {len(self.unfolder.mesh.faces)} faces")
        elif fp:
            messagebox.showerror("Error", "Failed to load STL file")
    
    def clear_cuts(self):
        self.unfolder.clear_cuts()
        self.v3d.selected_edges.clear()
        self.v3d.update_view()
        self.v2d.update_view()
        self.update_status()
    
    def auto_cut(self):
        if not self.unfolder.mesh:
            messagebox.showwarning("Warning", "Please load a mesh first")
            return
        nc = self.unfolder.auto_cut_seams()
        self.v3d.selected_edges = self.unfolder.cut_edges.copy()
        self.v3d.update_view()
        self.update_status()
        messagebox.showinfo("Auto-Cut", f"Made {nc} automatic cuts")
    
    def unfold(self):
        if not self.unfolder.mesh:
            messagebox.showwarning("Warning", "Please load a mesh first")
            return
        if self.unfolder.unfold():
            self.v2d.update_view()
            self.update_status()
            messagebox.showinfo("Success", "Mesh unfolded successfully!")
        else:
            messagebox.showerror("Error", "Failed to unfold mesh")
    
    def export_svg(self):
        if not self.unfolder.is_unfolded:
            messagebox.showwarning("Warning", "Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save SVG file", defaultextension=".svg", filetypes=[("SVG files", "*.svg"), ("All files", "*.*")])
        if fp and self.unfolder.export_svg(fp):
            messagebox.showinfo("Success", f"Exported to {fp}")
        elif fp:
            messagebox.showerror("Error", "Failed to export SVG")
    
    def export_json(self):
        if not self.unfolder.is_unfolded:
            messagebox.showwarning("Warning", "Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save JSON file", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if fp and self.unfolder.export_json(fp):
            messagebox.showinfo("Success", f"Exported to {fp}")
        elif fp:
            messagebox.showerror("Error", "Failed to export JSON")
    
    def export_pdf(self):
        if not self.unfolder.is_unfolded:
            messagebox.showwarning("Warning", "Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save PDF file", defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if fp and self.unfolder.export_pdf(fp):
            messagebox.showinfo("Success", f"Exported to {fp}")
        elif fp:
            messagebox.showerror("Error", "Failed to export PDF")
    
    def export_png(self):
        if not self.unfolder.is_unfolded:
            messagebox.showwarning("Warning", "Please unfold the mesh first")
            return
        fp = filedialog.asksaveasfilename(title="Save PNG file", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if fp and self.unfolder.export_png(fp):
            messagebox.showinfo("Success", f"Exported to {fp}")
        elif fp:
            messagebox.showerror("Error", "Failed to export PNG")


def main():
    root = tk.Tk()
    app = STLUnfolderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
