"""
Lightweight Mesh Viewer for discretize TensorMesh objects
A clean, interactive viewer with slice controls and toggleable features.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False



class MeshViewer:
    """
    Lightweight mesh viewer with single slice display and interactive controls.
    
    Parameters
    ----------
    mesh : discretize.TensorMesh
        The mesh object to visualize
    model_values : np.ndarray
        Model values for each cell (length = mesh.n_cells)
    cmap : str, optional
        Plotly colormap name (default: "plasma")
    show_cell_outlines : bool, optional
        Show cell boundaries in 2D slices (default: True)
    outline_step : int, optional
        Step size for cell outline spacing (default: 3)
    show_3d_points : bool, optional
        Show 3D point cloud representation (default: False)
    log_scale : bool, optional
        Use logarithmic color scale (default: False)
    
    Examples
    --------
    >>> from discretize import TensorMesh
    >>> import numpy as np
    >>> from mesh_viewer import MeshViewer
    >>> 
    >>> # Create a simple mesh
    >>> hx = hy = hz = np.ones(10)
    >>> mesh = TensorMesh([hx, hy, hz], x0="CCC")
    >>> 
    >>> # Create model values
    >>> model = np.ones(mesh.n_cells)
    >>> model[mesh.gridCC[:, 2] < 0] = 0.01
    >>> 
    >>> # Create and show viewer
    >>> viewer = MeshViewer(mesh, model, cmap="viridis")
    >>> viewer.show()
    """
    
    def __init__(self, mesh, model_values, cmap="plasma", show_cell_outlines=True, 
                 outline_step=1, show_3d_points=False, log_scale=False):
        self.mesh = mesh
        self.model_values = model_values
        self.model_3d = model_values.reshape(mesh.shape_cells, order='F')
        self.cmap = cmap
        self.show_cell_outlines = show_cell_outlines
        self.outline_step = outline_step
        self.show_3d_points = show_3d_points
        self.log_scale = log_scale

        # Mesh properties
        self.nx, self.ny, self.nz = mesh.shape_cells
        self.x_coords = mesh.cell_centers_x
        self.y_coords = mesh.cell_centers_y
        self.z_coords = mesh.cell_centers_z
        self.x_nodes = mesh.nodes_x
        self.y_nodes = mesh.nodes_y
        self.z_nodes = mesh.nodes_z
        
        # Calculate display values and color scale limits
        self._update_display_values()
        
        # Current slice indices
        self.ix_current = self.nx // 2
        self.iy_current = self.ny // 2
        self.iz_current = self.nz // 2
        
        # Current slice direction
        self.slice_direction = 'X'
        
        # Create widgets
        self.create_widgets()
        
        # Create figure widget
        self.fig_widget = widgets.Output()
        self.create_figure()
    
    def _update_display_values(self):
        """Update display values and color limits based on log scale setting"""
        if self.log_scale and self.model_values.min() > 0:
            self.vmin = np.log10(self.model_values.min())
            self.vmax = np.log10(self.model_values.max())
            self.display_values = np.log10(self.model_values)
            self.display_3d = np.log10(self.model_3d)
        else:
            self.vmin = self.model_values.min()
            self.vmax = self.model_values.max()
            self.display_values = self.model_values
            self.display_3d = self.model_3d
    
    def create_figure(self):
        """Create the interactive figure with 3D and 2D views"""
        
        with self.fig_widget:
            clear_output(wait=True)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    '3D Mesh with Active Slice', 
                    f'{self.slice_direction} Slice at {self.get_current_position():.2f}'
                ),
                specs=[[{"type": "scene"}, {"type": "xy"}]],
                horizontal_spacing=0.15,
                column_widths=[0.6, 0.4]
            )
            
            # Add 3D point cloud if requested
            if self.show_3d_points:
                self._add_3d_points(fig)
            
            # Add mesh boundary wireframe
            self._add_mesh_wireframe(fig)
            
            # Add active slice plane in 3D
            self._add_active_slice_plane_3d(fig)
            
            # Add 2D slice
            self._add_2d_slice(fig)
            
            # Update layout
            scale_text = " (Log Scale)" if self.log_scale else ""
            fig.update_layout(
                title=f"Mesh Viewer - {self.slice_direction} Slice Mode{scale_text}",
                height=600,
                width=1200,
                showlegend=False
            )
            
            fig.update_scenes(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
            
            fig.show()
    
    def _add_3d_points(self, fig):
        """Add 3D point cloud representation"""
        skip = max(1, min(self.nx, self.ny, self.nz) // 8)
        X, Y, Z = np.meshgrid(
            self.x_coords[::skip], 
            self.y_coords[::skip], 
            self.z_coords[::skip], 
            indexing='ij'
        )
        
        colorbar_title = "Model Value"  # Keep title simple when using scientific notation
        
        fig.add_trace(
            go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                mode='markers',
                marker=dict(
                    size=2,
                    color=self.display_3d[::skip, ::skip, ::skip].flatten(),
                    colorscale=self.cmap,
                    opacity=0.5,
                    cmin=self.vmin,
                    cmax=self.vmax,
                    colorbar=dict(
                        title=colorbar_title, 
                        x=1.02, 
                        len=0.8,
                        tickmode="array" if self.log_scale else None,
                        tickvals=np.log10([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]) if self.log_scale else None,
                        ticktext=["10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹"] if self.log_scale else None
                    )
                ),
                name="3D_Points",
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    def _add_mesh_wireframe(self, fig):
        """Add mesh boundary wireframe"""
        x_min, x_max = self.x_nodes[0], self.x_nodes[-1]
        y_min, y_max = self.y_nodes[0], self.y_nodes[-1]
        z_min, z_max = self.z_nodes[0], self.z_nodes[-1]
        
        edges = [
            # Bottom face
            ([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], [z_min]*5),
            # Top face  
            ([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], [z_max]*5),
            # Vertical edges
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
            ([x_min, x_min], [y_max, y_max], [z_min, z_max])
        ]
        
        for x_edge, y_edge, z_edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge, y=y_edge, z=z_edge,
                    mode='lines',
                    line=dict(color='darkgray', width=2),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    def _add_active_slice_plane_3d(self, fig):
        """Add the active slice plane in 3D view"""
        
        if self.slice_direction == 'X':
            y_plane, z_plane = np.meshgrid(self.y_coords, self.z_coords)
            x_plane = np.full_like(y_plane, self.x_coords[self.ix_current])
            surfacecolor = self.display_3d[self.ix_current, :, :].T
            
        elif self.slice_direction == 'Y':
            x_plane, z_plane = np.meshgrid(self.x_coords, self.z_coords)
            y_plane = np.full_like(x_plane, self.y_coords[self.iy_current])
            surfacecolor = self.display_3d[:, self.iy_current, :].T
            
        else:  # Z
            x_plane, y_plane = np.meshgrid(self.x_coords, self.y_coords)
            z_plane = np.full_like(x_plane, self.z_coords[self.iz_current])
            surfacecolor = self.display_3d[:, :, self.iz_current].T
        
        fig.add_trace(
            go.Surface(
                x=x_plane, y=y_plane, z=z_plane,
                surfacecolor=surfacecolor,
                colorscale=self.cmap,
                cmin=self.vmin,
                cmax=self.vmax,
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title="Model Value", 
                    x=1.02, 
                    len=0.8,
                    tickmode="array" if self.log_scale else None,
                    tickvals=np.log10([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]) if self.log_scale else None,
                    ticktext=["10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹"] if self.log_scale else None
                ),
                name=f"{self.slice_direction}_Slice_Plane"
            ),
            row=1, col=1
        )
    
    def _add_2d_slice(self, fig):
        """Add 2D slice view using data from TensorMesh"""
        
        if self.slice_direction == 'X':
            x_coords = self.y_coords
            y_coords = self.z_coords
            x_nodes = self.y_nodes
            y_nodes = self.z_nodes
            slice_data = self.display_3d[self.ix_current, :, :].T
            x_label, y_label = "Y", "Z"
            
        elif self.slice_direction == 'Y':
            x_coords = self.x_coords
            y_coords = self.z_coords
            x_nodes = self.x_nodes
            y_nodes = self.z_nodes
            slice_data = self.display_3d[:, self.iy_current, :].T
            x_label, y_label = "X", "Z"
            
        else:  # Z
            x_coords = self.x_coords
            y_coords = self.y_coords
            x_nodes = self.x_nodes
            y_nodes = self.y_nodes
            slice_data = self.display_3d[:, :, self.iz_current].T
            x_label, y_label = "X", "Y"
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                x=x_coords,
                y=y_coords,
                z=slice_data,
                colorscale=self.cmap,
                zmin=self.vmin,
                zmax=self.vmax,
                showscale=True,
                colorbar=dict(
                    title="Model Value",
                    x=1.02,
                    len=0.8,
                    tickmode="array" if self.log_scale else None,
                    tickvals=np.log10([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]) if self.log_scale else None,
                    ticktext=["10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹"] if self.log_scale else None
                ),
                name=f"{self.slice_direction}_Slice_2D",
                hovertemplate=f'{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<br>Value: %{{z:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add cell outlines if requested
        if self.show_cell_outlines:
            self._add_2d_slice_outlines(fig, x_nodes, y_nodes)
        
        # Update axis labels
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)

    def _get_color_rgba(self, value):
        """Convert value to RGBA color string using colormap"""
        # Normalize value to 0-1
        norm_val = (value - self.vmin) / (self.vmax - self.vmin)
        norm_val = np.clip(norm_val, 0, 1)

        # Simple plasma colormap approximation
        if self.cmap == 'plasma':
            if norm_val < 0.25:
                r, g, b = 13 + norm_val * 4 * 100, 8 + norm_val * 4 * 60, 135 + norm_val * 4 * 80
            elif norm_val < 0.5:
                r, g, b = 113 + (norm_val - 0.25) * 4 * 100, 68 + (norm_val - 0.25) * 4 * 100, 215 - (norm_val - 0.25) * 4 * 100
            elif norm_val < 0.75:
                r, g, b = 213 + (norm_val - 0.5) * 4 * 42, 168 + (norm_val - 0.5) * 4 * 80, 115 - (norm_val - 0.5) * 4 * 115
            else:
                r, g, b = 255, 248 - (norm_val - 0.75) * 4 * 80, 0 + (norm_val - 0.75) * 4 * 33
        else:
            # Fallback: simple blue to red
            r = int(norm_val * 255)
            g = 0
            b = int((1 - norm_val) * 255)

        return f"{int(r)},{int(g)},{int(b)}"


    def _add_2d_slice_outlines(self, fig, x_nodes, y_nodes):
        """Add properly aligned cell outlines to 2D slice"""

        # Vertical lines at node positions (cell boundaries)
        for i in range(0, len(x_nodes), self.outline_step):
            if i < len(x_nodes):
                fig.add_trace(
                    go.Scatter(
                        x=[x_nodes[i], x_nodes[i]],
                        y=[y_nodes[0], y_nodes[-1]],
                        mode='lines',
                        line=dict(color='white', width=1.5),  # Slightly thinner for outline_step=1
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )

        # Horizontal lines at node positions (cell boundaries)
        for j in range(0, len(y_nodes), self.outline_step):
            if j < len(y_nodes):
                fig.add_trace(
                    go.Scatter(
                        x=[x_nodes[0], x_nodes[-1]],
                        y=[y_nodes[j], y_nodes[j]],
                        mode='lines',
                        line=dict(color='white', width=1.5),  # Slightly thinner for outline_step=1
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )

    def get_current_position(self):
        """Get current slice position value"""
        if self.slice_direction == 'X':
            return self.x_coords[self.ix_current]
        elif self.slice_direction == 'Y':
            return self.y_coords[self.iy_current]
        else:
            return self.z_coords[self.iz_current]
    
    def create_widgets(self):
        """Create interactive control widgets"""
        
        # Direction selector
        self.direction_dropdown = widgets.Dropdown(
            options=['X', 'Y', 'Z'],
            value='X',
            description='Slice Direction:',
            style={'description_width': 'initial'}
        )
        
        # Slice index slider
        self.slice_slider = widgets.IntSlider(
            value=self.ix_current,
            min=0,
            max=self.nx-1,
            step=1,
            description='Slice Index:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # 3D points toggle
        self.points_toggle = widgets.Checkbox(
            value=self.show_3d_points,
            description='Show 3D Points',
            style={'description_width': 'initial'}
        )
        
        # Cell outline toggle
        self.outline_toggle = widgets.Checkbox(
            value=self.show_cell_outlines,
            description='Show Cell Outlines (2D)',
            style={'description_width': 'initial'}
        )
        
        # Log scale toggle
        self.log_toggle = widgets.Checkbox(
            value=self.log_scale,
            description='Log Scale',
            style={'description_width': 'initial'}
        )
        
        # Position display
        self.position_label = widgets.HTML(
            value=f"<b>Position: {self.get_current_position():.2f}</b>"
        )
        
        # Connect event handlers
        self.direction_dropdown.observe(self._on_direction_change, names='value')
        self.slice_slider.observe(self._on_slice_change, names='value')
        self.points_toggle.observe(self._on_points_toggle, names='value')
        self.outline_toggle.observe(self._on_outline_toggle, names='value')
        self.log_toggle.observe(self._on_log_toggle, names='value')
    
    def _on_direction_change(self, change):
        """Handle slice direction change"""
        self.slice_direction = change['new']
        
        # Update slider range and value
        if self.slice_direction == 'X':
            self.slice_slider.max = self.nx - 1
            self.slice_slider.value = self.ix_current
        elif self.slice_direction == 'Y':
            self.slice_slider.max = self.ny - 1
            self.slice_slider.value = self.iy_current
        else:  # Z
            self.slice_slider.max = self.nz - 1
            self.slice_slider.value = self.iz_current
        
        self._update_position_label()
        self.create_figure()
    
    def _on_slice_change(self, change):
        """Handle slice index change"""
        if self.slice_direction == 'X':
            self.ix_current = change['new']
        elif self.slice_direction == 'Y':
            self.iy_current = change['new']
        else:  # Z
            self.iz_current = change['new']
        
        self._update_position_label()
        self.create_figure()
    
    def _on_points_toggle(self, change):
        """Handle 3D points toggle"""
        self.show_3d_points = change['new']
        self.create_figure()
    
    def _on_outline_toggle(self, change):
        """Handle outline toggle"""
        self.show_cell_outlines = change['new']
        self.create_figure()
    
    def _on_log_toggle(self, change):
        """Handle log scale toggle"""
        self.log_scale = change['new']
        
        # Recalculate display values and limits
        self._update_display_values()
        self.create_figure()
    
    def _update_position_label(self):
        """Update position label text"""
        self.position_label.value = f"<b>Position: {self.get_current_position():.2f}</b>"
    
    def show(self):
        """Display the interactive mesh viewer"""
        
        # Create control panel
        controls = widgets.VBox([
            widgets.HTML("<h3>📊 Mesh Slice Viewer</h3>"),
            self.direction_dropdown,
            widgets.HBox([self.slice_slider, self.position_label]),
            widgets.HBox([self.points_toggle, self.outline_toggle, self.log_toggle]),
            widgets.HTML("<hr>")
        ])
        
        # Display everything
        display(controls)
        display(self.fig_widget)
    
    def update_model(self, new_model_values):
        """
        Update the model values and refresh the display
        
        Parameters
        ----------
        new_model_values : np.ndarray
            New model values (length = mesh.n_cells)
        """
        self.model_values = new_model_values
        self.model_3d = new_model_values.reshape(self.mesh.shape_cells, order='F')
        
        # Recalculate display values and limits
        self._update_display_values()
        self.create_figure()
    
    def save_figure(self, filename="mesh_viewer.html"):
        """
        Save the current figure as an HTML file
        
        Parameters
        ----------
        filename : str
            Output filename (default: "mesh_viewer.html")
        """
        # Create a static version of the current figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '3D Mesh with Active Slice', 
                f'{self.slice_direction} Slice at {self.get_current_position():.2f}'
            ),
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            horizontal_spacing=0.15,
            column_widths=[0.6, 0.4]
        )
        
        if self.show_3d_points:
            self._add_3d_points(fig)
        self._add_mesh_wireframe(fig)
        self._add_active_slice_plane_3d(fig)
        self._add_2d_slice(fig)
        
        scale_text = " (Log Scale)" if self.log_scale else ""
        fig.update_layout(
            title=f"Mesh Viewer - {self.slice_direction} Slice Mode{scale_text}",
            height=600,
            width=1200,
            showlegend=False
        )
        
        fig.write_html(filename)
        print(f"Figure saved as {filename}")

class TreeMeshViewer:
    """
    Interactive viewer for TreeMesh/OcTree meshes with the same interface as MeshViewer.
    Uses Plotly for consistent look and feel.
    
    Parameters
    ----------
    mesh : discretize.TreeMesh
        The tree mesh object to visualize
    model_values : np.ndarray
        Model values for each cell (length = mesh.n_cells)
    cmap : str, optional
        Plotly colormap name (default: "plasma")
    show_3d_points : bool, optional
        Show 3D point cloud representation (default: False for large meshes)
    log_scale : bool, optional
        Use logarithmic color scale (default: False)
    """
    
    def __init__(self, mesh, model_values, cmap="plasma", show_3d_points=False, show_cell_outlines=True, log_scale=False):
        
        # Validate inputs
        if not hasattr(mesh, '_meshType') or mesh._meshType.lower() != 'tree':
            raise ValueError("TreeMeshViewer only works with TreeMesh objects")
            
        if len(model_values) != mesh.n_cells:
            raise ValueError(f"Model has {len(model_values)} values but mesh has {mesh.n_cells} cells")
        
        self.mesh = mesh
        self.model_values = model_values
        self.cmap = cmap
        self.show_3d_points = show_3d_points
        self.show_cell_outlines = show_cell_outlines
        self.log_scale = log_scale
        
        # Get cell centers and bounds
        self.cell_centers = mesh.gridCC
        self.x_bounds = [self.cell_centers[:, 0].min(), self.cell_centers[:, 0].max()]
        self.y_bounds = [self.cell_centers[:, 1].min(), self.cell_centers[:, 1].max()]
        self.z_bounds = [self.cell_centers[:, 2].min(), self.cell_centers[:, 2].max()]

        # Get unique cell center positions for each axis (for discrete slider)
        self.unique_x_positions = np.unique(self.cell_centers[:, 0])
        self.unique_y_positions = np.unique(self.cell_centers[:, 1])
        self.unique_z_positions = np.unique(self.cell_centers[:, 2])

        # Set default slice parameters
        self.slice_direction = 'Z'
        self.slice_position = self.unique_z_positions[len(self.unique_z_positions)//2]  # Middle position
        
        # Calculate display values
        self._update_display_values()
        
        # Create figure widget first (before create_widgets which may trigger create_figure)
        self.fig_widget = widgets.Output()
        
        self._creating_widgets = True
        # Create widgets
        self.create_widgets()
        self._creating_widgets = False

        # Create initial figure only once at the end
        self.create_figure()

    def _update_display_values(self):
        """Update display values based on log scale setting"""
        if self.log_scale and self.model_values.min() > 0:
            self.display_values = np.log10(self.model_values)
        else:
            self.display_values = self.model_values.copy()
            
        self.vmin = self.display_values.min()
        self.vmax = self.display_values.max()
    
    def create_figure(self):
        """Create the interactive figure with 3D and 2D views"""

        with self.fig_widget:
            clear_output(wait=True)

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    '', 
                    f'{self.slice_direction} Slice at {self.slice_position:.2f}'
                ),
                specs=[[{"type": "scene"}, {"type": "xy"}]],
                horizontal_spacing=0.15,
                column_widths=[0.6, 0.4]
            )

            # Add 3D point cloud if requested
            if self.show_3d_points:
                self._add_3d_points(fig)

            # Add mesh boundary wireframe
            self._add_mesh_wireframe(fig)

            # Add filled slice plane in 3D (change this line)
            self._add_slice_plane_3d(fig)

            # Add 2D slice view (main data visualization)
            self._add_2d_slice(fig)

            # Update layout
            scale_text = " (Log Scale)" if self.log_scale else ""
            fig.update_layout(
                title=f"TreeMesh Viewer - {self.slice_direction} Slice Mode{scale_text}",
                height=600,
                width=1200,
                showlegend=False
            )

            fig.update_scenes(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )

            fig.show()
    
    def _add_3d_points(self, fig):
        """Add 3D point cloud representation for TreeMesh"""
        # Only add points if explicitly requested
        if not self.show_3d_points:
            return
            
        # Set up colorbar title
        colorbar_title = "Model Value"  # Keep title simple when using scientific notation

        fig.add_trace(
            go.Scatter3d(
                x=self.cell_centers[:, 0],
                y=self.cell_centers[:, 1],
                z=self.cell_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=self.display_values,
                    colorscale=self.cmap,
                    opacity=0.4,
                    cmin=self.vmin,
                    cmax=self.vmax,
                    colorbar=dict(
                        title=colorbar_title, 
                        x=1.02, 
                        len=0.8,
                        tickmode="array" if self.log_scale else None,
                        tickvals=np.log10([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]) if self.log_scale else None,
                        ticktext=["10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹"] if self.log_scale else None
                    )
                ),
                name="TreeMesh_Points",
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Value: %{marker.color:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    def _add_mesh_wireframe(self, fig):
        """Add mesh boundary wireframe"""
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        z_min, z_max = self.z_bounds
        
        edges = [
            # Bottom face
            ([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], [z_min]*5),
            # Top face  
            ([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], [z_max]*5),
            # Vertical edges
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
            ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
            ([x_min, x_min], [y_max, y_max], [z_min, z_max])
        ]
        
        for x_edge, y_edge, z_edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge, y=y_edge, z=z_edge,
                    mode='lines',
                    line=dict(color='darkgray', width=2),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
    
    def _add_slice_plane_3d(self, fig):
        """Add filled slice plane to 3D view (no data visualization, just a solid plane)"""

        # Get mesh bounds
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        z_min, z_max = self.z_bounds

        # Create slice plane based on direction
        if self.slice_direction == 'X':
            # YZ plane at x = slice_position
            y_plane, z_plane = np.meshgrid(
                np.linspace(y_min, y_max, 10), 
                np.linspace(z_min, z_max, 10)
            )
            x_plane = np.full_like(y_plane, self.slice_position)

        elif self.slice_direction == 'Y':
            # XZ plane at y = slice_position
            x_plane, z_plane = np.meshgrid(
                np.linspace(x_min, x_max, 10), 
                np.linspace(z_min, z_max, 10)
            )
            y_plane = np.full_like(x_plane, self.slice_position)

        else:  # Z direction
            # XY plane at z = slice_position
            x_plane, y_plane = np.meshgrid(
                np.linspace(x_min, x_max, 10), 
                np.linspace(y_min, y_max, 10)
            )
            z_plane = np.full_like(x_plane, self.slice_position)

        # Add filled slice plane (solid color, no data)
        fig.add_trace(
            go.Surface(
                x=x_plane, 
                y=y_plane, 
                z=z_plane,
                surfacecolor=np.ones_like(x_plane),  # Uniform color
                colorscale=[[0, 'rgba(255,0,0,0.6)'], [1, 'rgba(255,0,0,0.6)']],  # Semi-transparent red
                showscale=False,
                name=f"{self.slice_direction} Slice Plane",
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    def _add_2d_slice(self, fig):
        """Add 2D slice exactly like discretize plot_slice - no interpolation gaps"""

        # Step 1: Set up slice plane (same as discretize)
        normalInd = {"X": 0, "Y": 1, "Z": 2}[self.slice_direction]
        antiNormalInd = {"X": [1, 2], "Y": [0, 2], "Z": [0, 1]}[self.slice_direction]

        # Get slice location (exact cell center position)
        slice_loc = self.slice_position

        slice_origin = np.array([0.0, 0.0, 0.0])
        slice_origin[normalInd] = slice_loc
        normal = [0, 0, 0]
        normal[normalInd] = 1

        # Step 2: Get intersecting cells
        try:
            inds = self.mesh.get_cells_on_plane(slice_origin, normal)
        except:
            # Fallback method
            mask = np.abs(self.cell_centers[:, normalInd] - slice_loc) < 1e-10
            inds = np.where(mask)[0]

        if len(inds) == 0:
            fig.add_annotation(
                text=f"No cells found at {self.slice_direction}={slice_loc:.2f}",
                x=0.5, y=0.5, xref="x domain", yref="y domain",
                showarrow=False, font=dict(size=14, color="red"),
                row=1, col=2
            )
            return

        # Step 3: Create temporary 2D TreeMesh (same as discretize)
        h2d = (self.mesh.h[antiNormalInd[0]], self.mesh.h[antiNormalInd[1]])
        x2d = (self.mesh.origin[antiNormalInd[0]], self.mesh.origin[antiNormalInd[1]])

        import discretize
        temp_mesh = discretize.TreeMesh(h2d, x2d, diagonal_balance=False)
        level_diff = self.mesh.max_level - temp_mesh.max_level

        # Get 2D grid and levels for intersecting cells
        levels = self.mesh._cell_levels_by_indexes(inds) - level_diff
        grid2d = self.cell_centers[inds][:, antiNormalInd]

        # Insert cells into 2D mesh
        temp_mesh.insert_cells(grid2d, levels)

        # Step 4: Get cell polygons and values
        node_grid = np.r_[temp_mesh.nodes, temp_mesh.hanging_nodes]
        cell_nodes = temp_mesh.cell_nodes[:, (0, 1, 3, 2)]  # Reorder for proper polygon
        cell_verts = node_grid[cell_nodes]

        # Get interpolated values for 2D mesh
        tm_gridboost = np.empty((temp_mesh.n_cells, 3))
        tm_gridboost[:, antiNormalInd] = temp_mesh.cell_centers
        tm_gridboost[:, normalInd] = slice_loc

        ind_3d_to_2d = self.mesh.get_containing_cells(tm_gridboost)
        v2d = self.display_values[ind_3d_to_2d]

        # Step 5: Use the actual cell boundaries to create a proper filled area
        # Instead of heatmap interpolation, use the actual mesh structure

        # Get the complete mesh boundary for this slice
        x_coords = temp_mesh.cell_centers[:, 0]
        y_coords = temp_mesh.cell_centers[:, 1]

        # Create a denser regular grid that covers the FULL mesh extent
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Use a finer grid to ensure we don't miss any cells
        nx_grid = min(200, len(np.unique(x_coords)) * 2)
        ny_grid = min(200, len(np.unique(y_coords)) * 2)

        x_grid = np.linspace(x_min, x_max, nx_grid)
        y_grid = np.linspace(y_min, y_max, ny_grid)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # For each grid point, find the containing cell and use its value
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

        # Create 3D points for the grid to query the temp mesh
        grid_3d = np.zeros((len(grid_points), 3))
        grid_3d[:, antiNormalInd] = grid_points
        grid_3d[:, normalInd] = slice_loc

        # Find containing cells for each grid point
        try:
            # Use the 3D mesh to find containing cells
            containing_cells = self.mesh.get_containing_cells(grid_3d)

            # Get values for these cells
            Z_values = np.full(len(grid_points), np.nan)
            valid_mask = containing_cells != -1
            Z_values[valid_mask] = self.display_values[containing_cells[valid_mask]]

        except:
            # Fallback: use nearest neighbor
            if HAS_SCIPY:
                from scipy.spatial import cKDTree
                tree = cKDTree(temp_mesh.cell_centers)
                distances, indices = tree.query(grid_points)
                Z_values = v2d[indices]
                # Mask points that are too far from any cell
                cell_sizes = np.array([temp_mesh.h[0].min(), temp_mesh.h[1].min()])
                max_distance = np.sqrt(cell_sizes[0]**2 + cell_sizes[1]**2) / 2
                Z_values[distances > max_distance] = np.nan
            else:
                Z_values = np.full(len(grid_points), v2d.mean())

        # Reshape back to grid
        Z_grid = Z_values.reshape(X_grid.shape)

        # Add the filled heatmap
        fig.add_trace(
            go.Heatmap(
                x=x_grid,
                y=y_grid,
                z=Z_grid,
                colorscale=self.cmap,
                zmin=self.vmin,
                zmax=self.vmax,
                showscale=True,
                colorbar=dict(
                    title="Model Value",
                    x=1.02,
                    len=0.8,
                    tickmode="array" if self.log_scale else None,
                    tickvals=np.log10([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]) if self.log_scale else None,
                    ticktext=["10⁻⁸", "10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹"] if self.log_scale else None
                ),
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Value: %{z:.3f}<extra></extra>',
                name="TreeMesh_Data"
            ),
            row=1, col=2
        )

        # Step 6: Add cell outlines if requested
        if self.show_cell_outlines:
            # Create batched outlines
            x_all = []
            y_all = []

            for verts in cell_verts:
                # Add polygon vertices + None separator
                x_all.extend([verts[0, 0], verts[1, 0], verts[2, 0], verts[3, 0], verts[0, 0], None])
                y_all.extend([verts[0, 1], verts[1, 1], verts[2, 1], verts[3, 1], verts[0, 1], None])

            # Single trace for all outlines
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=y_all,
                    mode='lines',
                    line=dict(color='white', width=0.5),
                    showlegend=False,
                    hoverinfo='skip',
                    name="TreeMesh_Outlines"
                ),
                row=1, col=2
            )

        # Set axis labels
        axis_labels = {"X": ["Y", "Z"], "Y": ["X", "Z"], "Z": ["X", "Y"]}
        x_label, y_label = axis_labels[self.slice_direction]
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)

    def _get_slice_tolerance(self):
        """Calculate tolerance for slice selection - small since using discrete positions"""
        return 1e-10
    
    def _get_color_rgba(self, value):
        """Convert value to RGBA color string using colormap"""
        # Normalize value to 0-1
        norm_val = (value - self.vmin) / (self.vmax - self.vmin)
        norm_val = np.clip(norm_val, 0, 1)

        # Simple plasma colormap approximation
        if self.cmap == 'plasma':
            if norm_val < 0.25:
                r, g, b = 13 + norm_val * 4 * 100, 8 + norm_val * 4 * 60, 135 + norm_val * 4 * 80
            elif norm_val < 0.5:
                r, g, b = 113 + (norm_val - 0.25) * 4 * 100, 68 + (norm_val - 0.25) * 4 * 100, 215 - (norm_val - 0.25) * 4 * 100
            elif norm_val < 0.75:
                r, g, b = 213 + (norm_val - 0.5) * 4 * 42, 168 + (norm_val - 0.5) * 4 * 80, 115 - (norm_val - 0.5) * 4 * 115
            else:
                r, g, b = 255, 248 - (norm_val - 0.75) * 4 * 80, 0 + (norm_val - 0.75) * 4 * 33
        else:
            # Fallback: simple blue to red
            r = int(norm_val * 255)
            g = 0
            b = int((1 - norm_val) * 255)

        return f"{int(r)},{int(g)},{int(b)}"

    def _update_slice_positions(self):
        """Update available slice positions based on current direction"""
        if self.slice_direction == 'X':
            self.current_positions = self.unique_x_positions
        elif self.slice_direction == 'Y':
            self.current_positions = self.unique_y_positions
        else:  # Z
            self.current_positions = self.unique_z_positions

    def _update_button_states(self):
        """Update button enabled/disabled states based on current position"""
        current_idx = np.where(self.current_positions == self.slice_position)[0]
        if len(current_idx) > 0:
            idx = current_idx[0]
            # Convert numpy booleans to Python booleans
            self.slice_prev_button.disabled = bool(idx == 0)
            self.slice_next_button.disabled = bool(idx == len(self.current_positions) - 1)
    
    def create_widgets(self):
        """Create interactive control widgets with discrete slice positions"""
        
        # Direction selector
        self.direction_dropdown = widgets.Dropdown(
            options=['X', 'Y', 'Z'],
            value='Z',
            description='Slice Direction:',
            style={'description_width': 'initial'}
        )

        # Get initial position options
        self._update_slice_positions()
        
        # Slice position selector (discrete options)
        self.slice_selector = widgets.Dropdown(
            options=[(f"{pos:.2f}", pos) for pos in self.current_positions],
            value=self.slice_position,
            description='Slice Position:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Up/Down arrow buttons for slice navigation
        self.slice_prev_button = widgets.Button(
            description='◀ Prev',
            button_style='info',
            layout=widgets.Layout(width='80px')
        )
        
        self.slice_next_button = widgets.Button(
            description='Next ▶',
            button_style='info', 
            layout=widgets.Layout(width='80px')
        )
        
        # Update initial button states
        self._update_button_states()
        
        # Log scale toggle
        self.log_toggle = widgets.Checkbox(
            value=self.log_scale,
            description='Log Scale',
            style={'description_width': 'initial'}
        )
        
        # Cell outlines toggle
        self.outlines_toggle = widgets.Checkbox(
            value=self.show_cell_outlines,
            description='Show Cell Outlines',
            style={'description_width': 'initial'}
        )

        # Mesh info
        self.info_label = widgets.HTML(
            value=f"<b>TreeMesh:</b> {self.mesh.n_cells} cells"
        )
        
        # Connect event handlers
        self.direction_dropdown.observe(self._on_direction_change, names='value')
        self.slice_selector.observe(self._on_slice_change, names='value')
        self.slice_prev_button.on_click(self._on_slice_prev)
        self.slice_next_button.on_click(self._on_slice_next)
        self.outlines_toggle.observe(self._on_outlines_toggle, names='value')
        self.log_toggle.observe(self._on_log_toggle, names='value')

    def _on_direction_change(self, change):
        """Handle slice direction change"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        self.slice_direction = change['new']
        self._update_slice_positions()

        # Update dropdown options without triggering events
        self.slice_selector.unobserve(self._on_slice_change, names='value')
        self.slice_selector.options = [(f"{pos:.2f}", pos) for pos in self.current_positions]
        # Set to middle position
        self.slice_position = self.current_positions[len(self.current_positions)//2]
        self.slice_selector.value = self.slice_position
        self.slice_selector.observe(self._on_slice_change, names='value')
        
        # Update button states
        self._update_button_states()

        self.create_figure()
    
    def _on_slice_change(self, change):
        """Handle slice position change"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        self.slice_position = change['new']
        self._update_button_states()
        self.create_figure()
    
    def _on_slice_prev(self, button):
        """Handle previous slice button"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        current_idx = np.where(self.current_positions == self.slice_position)[0]
        if len(current_idx) > 0 and current_idx[0] > 0:
            new_idx = current_idx[0] - 1
            self.slice_position = self.current_positions[new_idx]
            self.slice_selector.value = self.slice_position
    
    def _on_slice_next(self, button):
        """Handle next slice button"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        current_idx = np.where(self.current_positions == self.slice_position)[0]
        if len(current_idx) > 0 and current_idx[0] < len(self.current_positions) - 1:
            new_idx = current_idx[0] + 1
            self.slice_position = self.current_positions[new_idx]
            self.slice_selector.value = self.slice_position
    
    def _on_log_toggle(self, change):
        """Handle log scale toggle"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        self.log_scale = change['new']
        self._update_display_values()
        self.create_figure()

    def _on_outlines_toggle(self, change):
        """Handle cell outlines toggle"""
        if hasattr(self, '_creating_widgets') and self._creating_widgets:
            return
        self.show_cell_outlines = change['new']
        self.create_figure()

    def show(self):
        """Display the interactive mesh viewer"""
        
        # Create control panel
        controls = widgets.VBox([
            widgets.HTML("<h3>🌳 TreeMesh Viewer</h3>"),
            self.info_label,
            self.direction_dropdown,
            widgets.HBox([self.slice_selector, self.slice_prev_button, self.slice_next_button]),
            widgets.HBox([self.outlines_toggle, self.log_toggle]),
            widgets.HTML("<hr>")
        ])
        
        # Display everything
        display(controls)
        display(self.fig_widget)
    
    def update_model(self, new_model_values):
        """Update the model values and refresh the display"""
        if len(new_model_values) != self.mesh.n_cells:
            raise ValueError(f"Model has {len(new_model_values)} values but mesh has {self.mesh.n_cells} cells")
            
        self.model_values = new_model_values
        self._update_display_values()
        self.create_figure()

# Convenience function to automatically choose the right viewer
def create_mesh_viewer(mesh, model_values, **kwargs):
    """
    Automatically create the appropriate mesh viewer based on mesh type.
    
    Parameters
    ----------
    mesh : discretize mesh object
        The mesh to visualize (TensorMesh or TreeMesh)
    model_values : np.ndarray
        Model values for each cell
    **kwargs
        Additional arguments passed to the viewer
    
    Returns
    -------
    MeshViewer or TreeMeshViewer
        The appropriate viewer for the mesh type
    """
    if hasattr(mesh, '_meshType'):
        if mesh._meshType.lower() == 'tree':
            return TreeMeshViewer(mesh, model_values, **kwargs)
        elif mesh._meshType.lower() == 'tensor':
            return MeshViewer(mesh, model_values, **kwargs)
    
    # Fallback: try to detect by attributes
    if hasattr(mesh, 'shape_cells'):
        return MeshViewer(mesh, model_values, **kwargs)
    else:
        return TreeMeshViewer(mesh, model_values, **kwargs)