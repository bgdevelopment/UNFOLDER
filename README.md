# STL Unfolder - Interactive Cutting & Unfolding Tool

A comprehensive Python application for loading STL files, interactively cutting edges, and unfolding 3D meshes to 2D plane.

## Features

- **Interactive 3D Viewer**: Rotate, zoom, and explore your 3D mesh
- **Edge Cutting**: Click on edges in the 3D view to mark them as cuts (red = cut, blue = connected)
- **Auto-Cut Seams**: Automatically generate cuts for closed meshes using spanning tree algorithm
- **Real-time 2D Preview**: See the unfolded result instantly
- **Export Options**: Save unfolded mesh as SVG or JSON
- **Support**: Both binary and ASCII STL formats

## Requirements

```bash
pip install numpy trimesh
```

Tkinter is usually included with Python installations. On Linux, you may need to install it separately:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

## Usage

### Running the Application

```bash
python stl_unfolder.py
```

### Interface Guide

#### Left Panel - Controls

**File Operations:**
- **Load STL**: Open an STL file dialog to load your 3D mesh
- **Clear Cuts**: Remove all edge cuts and reset the mesh

**Cutting Operations:**
- **Auto-Cut Seams**: Automatically select edges to cut for unfolding closed meshes
- **Manual Cutting**: Click on edges in the 3D viewer to toggle cuts
  - Red edges = cut (faces will separate here)
  - Blue edges = connected (faces stay joined)

**Unfolding:**
- **Unfold to 2D**: Generate the 2D unfolded pattern based on current cuts

**Export:**
- **Export SVG**: Save the unfolded pattern as an SVG file (great for printing/cutting)
- **Export JSON**: Save vertex coordinates and face data as JSON

#### 3D Viewer (Left Side)

- **Rotate**: Click and drag to rotate the view
- **Zoom**: Use mouse wheel to zoom in/out
- **Select Edges**: Click on an edge to toggle it as a cut
  - First click: marks edge as cut (turns red)
  - Second click: removes cut (turns blue)

#### 2D Viewer (Right Side)

- **Zoom**: Use mouse wheel to zoom in/out
- Shows the unfolded mesh with each face in a different color
- Faces are arranged based on the cut pattern

## Workflow Example

1. **Load an STL file** using the "Load STL" button
2. **Review the mesh** in the 3D viewer (rotate to see all angles)
3. **Choose cutting strategy**:
   - For simple shapes: Use "Auto-Cut Seams" for automatic cutting
   - For complex/custom patterns: Manually click edges to define cuts
4. **Click "Unfold to 2D"** to generate the flat pattern
5. **Review the result** in the 2D viewer
6. **Export** as SVG for printing or JSON for further processing

## Tips for Good Unfoldings

- **Minimize cuts**: Fewer cuts generally mean less assembly work
- **Strategic placement**: Cut along natural seams or hidden edges when possible
- **Closed meshes**: Must be cut to form a spanning tree (auto-cut does this automatically)
- **Developable surfaces**: Some shapes (cylinders, cones) unfold perfectly; others will have distortion

## Algorithm Details

The unfolding algorithm uses:
1. **BFS propagation**: Starting from one face, propagates 2D coordinates to neighbors
2. **Edge length preservation**: Maintains exact edge lengths during unfolding
3. **Spanning tree**: For closed meshes, creates a tree structure of connected faces
4. **Component handling**: Manages disconnected components with offset positioning

## File Formats

### SVG Export
- Vector format suitable for:
  - Paper crafting
  - Laser cutting
  - CNC routing
  - Printing templates

### JSON Export
Contains:
- `vertices_2d`: Array of 2D vertex coordinates
- `faces`: Face metadata including colors
- `face_vertex_indices`: Mapping of faces to vertices

## Troubleshooting

**Mesh doesn't unfold completely:**
- Ensure the mesh is manifold (watertight)
- Check that cuts create a proper spanning tree
- Try auto-cut first, then adjust manually

**Overlapping faces in 2D view:**
- This is normal for complex shapes
- Add more cuts to reduce overlap
- Some overlap is unavoidable for non-developable surfaces

**Application won't start:**
- Verify tkinter is installed
- Check that numpy and trimesh are installed
- Ensure you're using Python 3.7+

## License

This software is provided as-is for educational and personal use.

## Contributing

Feel free to enhance this tool with features like:
- DXF export
- Nesting optimization
- Tab generation for assembly
- UV mapping export
