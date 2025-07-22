# simpeg-mesh-viewer

Interactive viewer with a movable slicer for meshes and models created using `discretize` or `simpeg`.

> **⚠️ CAUTION:** A lot of this was created using AI (Claude). Plotting code is tedious…

Built for `discretize` **TensorMesh** and **TreeMesh** models using **Plotly + ipywidgets**.  
Explore 3D models, toggle log scales/outlines/3D points, and step through slices with sliders, prev/next buttons, or custom jump sizes.


## Usage

1. Download `mesh_viewer.py`.
2. Import and run as :
   ```python
   from mesh_viewer import create_mesh_viewer
   viewer_auto = create_mesh_viewer(mesh, model, cmap="viridis", log_scale=True)
   viewer_auto.show()

## Files
- `large_octree_mesh.txt` is an Octree mesh file with 769288 cells.
- `mesh-plotter_v0.ipynb` contains examples of viewer being used to plot Tensor and Octree meshes.

## Screenshot
<img width="1579" height="809" alt="TreemeshViewer" src="https://github.com/user-attachments/assets/d43a9616-cee8-455f-b76d-5f76460a235b" />
