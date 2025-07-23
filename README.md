# simpeg-mesh-viewer

Interactive viewer with a movable slicer for meshes and models created using `discretize` or `simpeg`.

> **⚠️ CAUTION:** A lot of this was created using AI (Claude). Plotting code is tedious…


Built for `discretize` **TensorMesh** and **TreeMesh** models using **Plotly + ipywidgets**.  
Step through model slices with sliders, prev/next buttons, or custom jump sizes. Toggle log scales/outlines/3D points


## Usage

1. Download `mesh_viewer.py`.
2. Import and run as :
   ```python
   from mesh_viewer import create_mesh_viewer
   create_mesh_viewer(mesh, model, cmap="viridis", log_scale=True)

## Files
- `large_octree_mesh.txt` is an Octree mesh file with 769288 cells.
- `mesh-plotter_v0.ipynb` contains examples of viewer being used to plot Tensor and Octree meshes.

## Screenshot
<img width="885" height="841" alt="TreemeshViewer" src="https://github.com/user-attachments/assets/33f491a3-28be-4aae-af0f-9bef63ebd491" />
