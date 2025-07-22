"""
Mesh viewers for discretize TensorMesh and TreeMesh
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

# SciPy optional (TreeMesh NN fallback)
try:
    from scipy.spatial import cKDTree  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------

def _log_ticks_from_range(vmin, vmax):
    """Return (tickvals_log10, ticktext) covering [vmin, vmax] decades."""
    # guard: if non-positive, bail to linear ticks
    if vmin <= 0 or vmax <= 0:
        return None, None
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil(np.log10(vmax)))
    exps = np.arange(lo, hi + 1)
    tickvals = exps.astype(float)        # in log10 space
    ticktext = [f"10^{e}" if e != 0 else "10⁰" for e in exps]
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    ticktext = [f"10{str(e).translate(sup)}" for e in exps]
    return tickvals, ticktext




def _add_wireframe(fig, bounds, row: int = 1, col: int = 1) -> None:
    """Add a gray bounding-box wireframe to a 3D subplot.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    bounds : ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    row, col : subplot location
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds

    edges = [
        # Bottom face
        ([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         [z_min] * 5),
        # Top face
        ([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         [z_max] * 5),
        # Vertical edges
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
    ]

    for x_edge, y_edge, z_edge in edges:
        fig.add_trace(
            go.Scatter3d(
                x=x_edge, y=y_edge, z=z_edge,
                mode="lines",
                line=dict(color="darkgray", width=2),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )


# -----------------------------------------------------------------------------
# TensorMesh Viewer
# -----------------------------------------------------------------------------


class MeshViewer:
    """Interactive viewer for discretize TensorMesh.

    Parameters are identical to the original implementation; behavior preserved.
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

        # Mesh info
        self.nx, self.ny, self.nz = mesh.shape_cells
        self.x_coords = mesh.cell_centers_x
        self.y_coords = mesh.cell_centers_y
        self.z_coords = mesh.cell_centers_z
        self.x_nodes = mesh.nodes_x
        self.y_nodes = mesh.nodes_y
        self.z_nodes = mesh.nodes_z

        self._update_display_values()

        # Current slice
        self.ix_current = self.nx // 2
        self.iy_current = self.ny // 2
        self.iz_current = self.nz // 2
        self.slice_direction = 'X'

        self.create_widgets()
        self.fig_widget = widgets.Output()
        self.create_figure()

    # ----------------- helpers -----------------

    def _update_display_values(self):
        if self.log_scale and self.model_values.min() > 0:
            self.vmin = np.log10(self.model_values.min())
            self.vmax = np.log10(self.model_values.max())
            self.display_values = np.log10(self.model_values)
            self.display_3d = np.log10(self.model_3d)
            self._tickvals, self._ticktext = _log_ticks_from_range(10**self.vmin, 10**self.vmax)
        else:
            self.vmin = self.model_values.min()
            self.vmax = self.model_values.max()
            self.display_values = self.model_values
            self.display_3d = self.model_3d
            self._tickvals, self._ticktext = (None, None)

    def _colorbar(self):
        cb = dict(title="Model Value", x=1.02, len=0.8)
        if self.log_scale and self._tickvals is not None:
            cb.update(dict(tickmode="array", tickvals=self._tickvals, ticktext=self._ticktext))
        return cb

    def get_current_position(self):
        if self.slice_direction == 'X':
            return self.x_coords[self.ix_current]
        if self.slice_direction == 'Y':
            return self.y_coords[self.iy_current]
        return self.z_coords[self.iz_current]

    # ----------------- figure build -----------------

    def create_figure(self):
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

            if self.show_3d_points:
                self._add_3d_points(fig)

            bounds = (
                (self.x_nodes[0], self.x_nodes[-1]),
                (self.y_nodes[0], self.y_nodes[-1]),
                (self.z_nodes[0], self.z_nodes[-1])
            )
            _add_wireframe(fig, bounds)

            self._add_active_slice_plane_3d(fig)
            self._add_2d_slice(fig)

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
        skip = max(1, min(self.nx, self.ny, self.nz) // 8)
        X, Y, Z = np.meshgrid(
            self.x_coords[::skip],
            self.y_coords[::skip],
            self.z_coords[::skip],
            indexing='ij'
        )
        fig.add_trace(
            go.Scatter3d(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                mode='markers',
                marker=dict(
                    size=2,
                    color=self.display_3d[::skip, ::skip, ::skip].flatten(),
                    colorscale=self.cmap,
                    opacity=0.5,
                    cmin=self.vmin, cmax=self.vmax,
                    colorbar=self._colorbar()
                ),
                name="3D_Points",
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    def _add_active_slice_plane_3d(self, fig):
        if self.slice_direction == 'X':
            y_plane, z_plane = np.meshgrid(self.y_coords, self.z_coords)
            x_plane = np.full_like(y_plane, self.x_coords[self.ix_current])
            surfacecolor = self.display_3d[self.ix_current, :, :].T
        elif self.slice_direction == 'Y':
            x_plane, z_plane = np.meshgrid(self.x_coords, self.z_coords)
            y_plane = np.full_like(x_plane, self.y_coords[self.iy_current])
            surfacecolor = self.display_3d[:, self.iy_current, :].T
        else:
            x_plane, y_plane = np.meshgrid(self.x_coords, self.y_coords)
            z_plane = np.full_like(x_plane, self.z_coords[self.iz_current])
            surfacecolor = self.display_3d[:, :, self.iz_current].T

        fig.add_trace(
            go.Surface(
                x=x_plane, y=y_plane, z=z_plane,
                surfacecolor=surfacecolor,
                colorscale=self.cmap,
                cmin=self.vmin, cmax=self.vmax,
                opacity=0.8,
                showscale=True,
                colorbar=self._colorbar(),
                name=f"{self.slice_direction}_Slice_Plane"
            ),
            row=1, col=1
        )

    def _add_2d_slice(self, fig):
        if self.slice_direction == 'X':
            x_coords = self.y_coords; y_coords = self.z_coords
            x_nodes = self.y_nodes;   y_nodes = self.z_nodes
            slice_data = self.display_3d[self.ix_current, :, :].T
            x_label, y_label = "Y", "Z"
        elif self.slice_direction == 'Y':
            x_coords = self.x_coords; y_coords = self.z_coords
            x_nodes = self.x_nodes;   y_nodes = self.z_nodes
            slice_data = self.display_3d[:, self.iy_current, :].T
            x_label, y_label = "X", "Z"
        else:
            x_coords = self.x_coords; y_coords = self.y_coords
            x_nodes = self.x_nodes;   y_nodes = self.y_nodes
            slice_data = self.display_3d[:, :, self.iz_current].T
            x_label, y_label = "X", "Y"

        fig.add_trace(
            go.Heatmap(
                x=x_coords, y=y_coords, z=slice_data,
                colorscale=self.cmap,
                zmin=self.vmin, zmax=self.vmax,
                showscale=True,
                colorbar=self._colorbar(),
                name=f"{self.slice_direction}_Slice_2D",
                hovertemplate=f'{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<br>Value: %{{z:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )

        if self.show_cell_outlines:
            self._add_2d_slice_outlines(fig, x_nodes, y_nodes)

        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)

    def _add_2d_slice_outlines(self, fig, x_nodes, y_nodes):
        """Batch outlines into one trace for speed."""
        xs, ys = [], []
        for i in range(0, len(x_nodes), self.outline_step):
            xs.extend([x_nodes[i], x_nodes[i], None])
            ys.extend([y_nodes[0], y_nodes[-1], None])
        for j in range(0, len(y_nodes), self.outline_step):
            xs.extend([x_nodes[0], x_nodes[-1], None])
            ys.extend([y_nodes[j], y_nodes[j], None])

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color="white", width=1.0),
                opacity=0.8,
                showlegend=False,
                hoverinfo="skip",
                name="Outlines"
            ),
            row=1, col=2
        )

    # ----------------- widgets -----------------

    def create_widgets(self):
        self.direction_dropdown = widgets.Dropdown(
            options=['X', 'Y', 'Z'], value='X', description='Slice Direction:',
            style={'description_width': 'initial'}
        )

        self.slice_slider = widgets.IntSlider(
            value=self.ix_current, min=0, max=self.nx - 1, step=1,
            description='Slice Index:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            continuous_update=False
        )

        # Prev/Next buttons for parity with TreeMesh
        self.prev_btn = widgets.Button(description='◀ Prev', layout=widgets.Layout(width='70px'))
        self.next_btn = widgets.Button(description='Next ▶', layout=widgets.Layout(width='70px'))

        self.points_toggle = widgets.Checkbox(
            value=self.show_3d_points, description='Show 3D Points',
            style={'description_width': 'initial'}
        )
        self.outline_toggle = widgets.Checkbox(
            value=self.show_cell_outlines, description='Show Cell Outlines (2D)',
            style={'description_width': 'initial'}
        )
        self.log_toggle = widgets.Checkbox(
            value=self.log_scale, description='Log Scale',
            style={'description_width': 'initial'}
        )

        self.position_label = widgets.HTML(value=f"<b>Position: {self.get_current_position():.2f}</b>")
        self.jump_input = widgets.BoundedIntText(
            value=1, min=1, max=10_000, step=1,
            description='Jump:', layout=widgets.Layout(width='120px'),
            style={'description_width': '50px'}
        )
        # Observers
        self.direction_dropdown.observe(self._on_direction_change, names='value')
        self.slice_slider.observe(self._on_slice_change, names='value')
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.points_toggle.observe(self._on_points_toggle, names='value')
        self.outline_toggle.observe(self._on_outline_toggle, names='value')
        self.log_toggle.observe(self._on_log_toggle, names='value')

    # callbacks
    def _on_direction_change(self, change):
        self.slice_direction = change['new']
        if self.slice_direction == 'X':
            self.slice_slider.max = self.nx - 1
            self.slice_slider.value = self.ix_current
        elif self.slice_direction == 'Y':
            self.slice_slider.max = self.ny - 1
            self.slice_slider.value = self.iy_current
        else:
            self.slice_slider.max = self.nz - 1
            self.slice_slider.value = self.iz_current
        self._update_position_label()
        self.create_figure()

    def _on_slice_change(self, change):
        if self.slice_direction == 'X':
            self.ix_current = change['new']
        elif self.slice_direction == 'Y':
            self.iy_current = change['new']
        else:
            self.iz_current = change['new']
        self._update_position_label()
        self.create_figure()

    def _on_prev(self, _):
        step = self.jump_input.value
        if self.slice_direction == 'X':
            self.slice_slider.value = max(0, self.ix_current - step)
        elif self.slice_direction == 'Y':
            self.slice_slider.value = max(0, self.iy_current - step)
        else:
            self.slice_slider.value = max(0, self.iz_current - step)

    def _on_next(self, _):
        step = self.jump_input.value
        if self.slice_direction == 'X':
            self.slice_slider.value = min(self.nx - 1, self.ix_current + step)
        elif self.slice_direction == 'Y':
            self.slice_slider.value = min(self.ny - 1, self.iy_current + step)
        else:
            self.slice_slider.value = min(self.nz - 1, self.iz_current + step)

    def _on_points_toggle(self, change):
        self.show_3d_points = change['new']
        self.create_figure()

    def _on_outline_toggle(self, change):
        self.show_cell_outlines = change['new']
        self.create_figure()

    def _on_log_toggle(self, change):
        self.log_scale = change['new']
        self._update_display_values()
        self.create_figure()

    def _update_position_label(self):
        self.position_label.value = f"<b>Position: {self.get_current_position():.2f}</b>"

    # ----------------- public -----------------

    def show(self):
        controls = widgets.VBox([
            widgets.HTML("<h3> TensorMesh Viewer</h3>"),
            self.direction_dropdown,
            widgets.HBox([self.slice_slider, self.prev_btn, self.next_btn, self.jump_input, self.position_label]),
            widgets.HBox([self.points_toggle, self.outline_toggle, self.log_toggle]),
            widgets.HTML("<hr>")
        ])
        display(controls)
        display(self.fig_widget)

    def update_model(self, new_model_values):
        self.model_values = new_model_values
        self.model_3d = new_model_values.reshape(self.mesh.shape_cells, order='F')
        self._update_display_values()
        self.create_figure()

    def save_figure(self, filename="mesh_viewer.html"):
        # Re-build fig then save
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
        bounds = (
            (self.x_nodes[0], self.x_nodes[-1]),
            (self.y_nodes[0], self.y_nodes[-1]),
            (self.z_nodes[0], self.z_nodes[-1])
        )
        _add_wireframe(fig, bounds)
        self._add_active_slice_plane_3d(fig)
        self._add_2d_slice(fig)
        fig.update_layout(title=f"Mesh Viewer - {self.slice_direction} Slice Mode",
                          height=600, width=1200, showlegend=False)
        fig.write_html(filename)
        print(f"Figure saved as {filename}")


# -----------------------------------------------------------------------------
# TreeMesh Viewer
# -----------------------------------------------------------------------------


class TreeMeshViewer:
    """Interactive viewer for discretize TreeMesh/OcTree with colored slice plane & sliders."""

    def __init__(self, mesh, model_values, cmap="plasma",
                 show_3d_points=False, show_cell_outlines=True, log_scale=False):
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

        self.cell_centers = mesh.gridCC
        self.x_bounds = [self.cell_centers[:, 0].min(), self.cell_centers[:, 0].max()]
        self.y_bounds = [self.cell_centers[:, 1].min(), self.cell_centers[:, 1].max()]
        self.z_bounds = [self.cell_centers[:, 2].min(), self.cell_centers[:, 2].max()]

        self.unique_x_positions = np.unique(self.cell_centers[:, 0])
        self.unique_y_positions = np.unique(self.cell_centers[:, 1])
        self.unique_z_positions = np.unique(self.cell_centers[:, 2])

        self.slice_direction = 'Z'
        self.slice_position = self.unique_z_positions[len(self.unique_z_positions) // 2]
        self._current_positions = None  # filled in update

        self._last_slice_grid = None  # (X_grid, Y_grid, Z_grid) for colored 3D plane

        self._update_display_values()

        self.fig_widget = widgets.Output()
        self._creating_widgets = True
        self.create_widgets()
        self._creating_widgets = False
        self.create_figure()

    # ----------------- helpers -----------------

    def _update_display_values(self):
        if self.log_scale and self.model_values.min() > 0:
            self.display_values = np.log10(self.model_values)
            # store linear mins/max for tick computation
            vmin_lin = self.model_values.min()
            vmax_lin = self.model_values.max()
            self._tickvals, self._ticktext = _log_ticks_from_range(vmin_lin, vmax_lin)
        else:
            self.display_values = self.model_values.copy()
            self._tickvals, self._ticktext = (None, None)

        self.vmin = self.display_values.min()
        self.vmax = self.display_values.max()

    def _colorbar(self):
        cb = dict(title="Model Value", x=1.02, len=0.8)
        if self.log_scale and self._tickvals is not None:
            cb.update(dict(tickmode="array",
                           tickvals=self._tickvals,
                           ticktext=self._ticktext))
        return cb

    def _update_slice_positions(self):
        if self.slice_direction == 'X':
            self._current_positions = self.unique_x_positions
        elif self.slice_direction == 'Y':
            self._current_positions = self.unique_y_positions
        else:
            self._current_positions = self.unique_z_positions

    def _idx_of_current(self):
        return int(np.where(self._current_positions == self.slice_position)[0][0])

    def _update_prev_next_disabled(self):
        idx = self._idx_of_current()
        self.slice_prev_button.disabled = idx == 0
        self.slice_next_button.disabled = idx == len(self._current_positions) - 1

    # ----------------- figure build -----------------

    def create_figure(self):
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

            if self.show_3d_points:
                self._add_3d_points(fig)

            _add_wireframe(fig, (tuple(self.x_bounds), tuple(self.y_bounds), tuple(self.z_bounds)))

            # Important: _add_2d_slice computes self._last_slice_grid used by 3D plane color
            self._add_2d_slice(fig)
            self._add_slice_plane_3d(fig)

            scale_text = " (Log Scale)" if self.log_scale else ""
            fig.update_layout(
                title=f"TreeMesh Viewer - {self.slice_direction} Slice Mode{scale_text}",
                height=600, width=1200, showlegend=False
            )
            fig.update_scenes(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
            fig.show()

    def _add_3d_points(self, fig):
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
                    cmin=self.vmin, cmax=self.vmax,
                    colorbar=self._colorbar()
                ),
                name="TreeMesh_Points",
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Value: %{marker.color:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

    def _add_slice_plane_3d(self, fig):
        # If we already computed a regular grid for the 2D heatmap, reuse it to color the plane
        if self._last_slice_grid is not None:
            Xg, Yg, Zg = self._last_slice_grid
            if self.slice_direction == 'X':
                x_plane = np.full_like(Xg, self.slice_position)
                y_plane, z_plane = Xg, Yg
                surfacecolor = Zg
            elif self.slice_direction == 'Y':
                y_plane = np.full_like(Xg, self.slice_position)
                x_plane, z_plane = Xg, Yg
                surfacecolor = Zg
            else:  # 'Z'
                z_plane = np.full_like(Xg, self.slice_position)
                x_plane, y_plane = Xg, Yg
                surfacecolor = Zg

            fig.add_trace(
                go.Surface(
                    x=x_plane, y=y_plane, z=z_plane,
                    surfacecolor=surfacecolor,
                    colorscale=self.cmap,
                    cmin=self.vmin, cmax=self.vmax,
                    opacity=0.8,
                    showscale=False,
                    name=f"{self.slice_direction}_Slice_Plane"
                ),
                row=1, col=1
            )
        else:
            # fallback: simple red plane
            x_min, x_max = self.x_bounds
            y_min, y_max = self.y_bounds
            z_min, z_max = self.z_bounds
            if self.slice_direction == 'X':
                y_plane, z_plane = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
                x_plane = np.full_like(y_plane, self.slice_position)
            elif self.slice_direction == 'Y':
                x_plane, z_plane = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
                y_plane = np.full_like(x_plane, self.slice_position)
            else:
                x_plane, y_plane = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
                z_plane = np.full_like(x_plane, self.slice_position)
            fig.add_trace(
                go.Surface(
                    x=x_plane, y=y_plane, z=z_plane,
                    surfacecolor=np.ones_like(x_plane),
                    colorscale=[[0, 'rgba(255,0,0,0.6)'], [1, 'rgba(255,0,0,0.6)']],
                    showscale=False,
                    name=f"{self.slice_direction} Slice Plane",
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

    def _add_2d_slice(self, fig):
        normalInd = {"X": 0, "Y": 1, "Z": 2}[self.slice_direction]
        antiNormalInd = {"X": [1, 2], "Y": [0, 2], "Z": [0, 1]}[self.slice_direction]
        slice_loc = self.slice_position

        slice_origin = np.zeros(3)
        slice_origin[normalInd] = slice_loc
        normal = [0, 0, 0]; normal[normalInd] = 1

        try:
            inds = self.mesh.get_cells_on_plane(slice_origin, normal)
        except Exception:
            mask = np.abs(self.cell_centers[:, normalInd] - slice_loc) < 1e-10
            inds = np.where(mask)[0]

        if len(inds) == 0:
            fig.add_annotation(text=f"No cells found at {self.slice_direction}={slice_loc:.2f}",
                               x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, font=dict(size=14, color="red"),
                               row=1, col=2)
            return

        import discretize
        h2d = (self.mesh.h[antiNormalInd[0]], self.mesh.h[antiNormalInd[1]])
        x2d = (self.mesh.origin[antiNormalInd[0]], self.mesh.origin[antiNormalInd[1]])
        temp_mesh = discretize.TreeMesh(h2d, x2d, diagonal_balance=False)
        level_diff = self.mesh.max_level - temp_mesh.max_level

        levels = self.mesh._cell_levels_by_indexes(inds) - level_diff
        grid2d = self.cell_centers[inds][:, antiNormalInd]
        temp_mesh.insert_cells(grid2d, levels)

        node_grid = np.r_[temp_mesh.nodes, temp_mesh.hanging_nodes]
        cell_nodes = temp_mesh.cell_nodes[:, (0, 1, 3, 2)]
        cell_verts = node_grid[cell_nodes]

        tm_gridboost = np.empty((temp_mesh.n_cells, 3))
        tm_gridboost[:, antiNormalInd] = temp_mesh.cell_centers
        tm_gridboost[:, normalInd] = slice_loc
        ind_3d_to_2d = self.mesh.get_containing_cells(tm_gridboost)
        v2d = self.display_values[ind_3d_to_2d]

        # Regular grid sampling for heatmap and 3D surface coloring
        x_coords = temp_mesh.cell_centers[:, 0]
        y_coords = temp_mesh.cell_centers[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        nx_grid = min(200, len(np.unique(x_coords)) * 2)
        ny_grid = min(200, len(np.unique(y_coords)) * 2)
        x_grid = np.linspace(x_min, x_max, nx_grid)
        y_grid = np.linspace(y_min, y_max, ny_grid)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        grid_3d = np.zeros((len(grid_points), 3))
        grid_3d[:, antiNormalInd] = grid_points
        grid_3d[:, normalInd] = slice_loc

        try:
            containing_cells = self.mesh.get_containing_cells(grid_3d)
            Z_values = np.full(len(grid_points), np.nan)
            valid_mask = containing_cells != -1
            Z_values[valid_mask] = self.display_values[containing_cells[valid_mask]]
        except Exception:
            if HAS_SCIPY:
                tree = cKDTree(temp_mesh.cell_centers)
                distances, indices = tree.query(grid_points)
                Z_values = v2d[indices]
                cell_sizes = np.array([temp_mesh.h[0].min(), temp_mesh.h[1].min()])
                max_distance = np.sqrt(cell_sizes[0] ** 2 + cell_sizes[1] ** 2) / 2
                Z_values[distances > max_distance] = np.nan
            else:
                Z_values = np.full(len(grid_points), v2d.mean())

        Z_grid = Z_values.reshape(X_grid.shape)

        # store for 3D surface coloring
        self._last_slice_grid = (X_grid, Y_grid, Z_grid)

        fig.add_trace(
            go.Heatmap(
                x=x_grid, y=y_grid, z=Z_grid,
                colorscale=self.cmap,
                zmin=self.vmin, zmax=self.vmax,
                showscale=True,
                colorbar=self._colorbar(),
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Value: %{z:.3f}<extra></extra>',
                name="TreeMesh_Data"
            ),
            row=1, col=2
        )

        if self.show_cell_outlines:
            x_all, y_all = [], []
            for verts in cell_verts:
                x_all.extend([verts[0, 0], verts[1, 0], verts[2, 0], verts[3, 0], verts[0, 0], None])
                y_all.extend([verts[0, 1], verts[1, 1], verts[2, 1], verts[3, 1], verts[0, 1], None])
            fig.add_trace(
                go.Scatter(
                    x=x_all, y=y_all,
                    mode='lines', line=dict(color='white', width=0.5),
                    showlegend=False, hoverinfo='skip', name="TreeMesh_Outlines"
                ),
                row=1, col=2
            )

        axis_labels = {"X": ["Y", "Z"], "Y": ["X", "Z"], "Z": ["X", "Y"]}
        x_label, y_label = axis_labels[self.slice_direction]
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)

    # ----------------- widgets -----------------

    def create_widgets(self):
        self.direction_dropdown = widgets.Dropdown(
            options=['X', 'Y', 'Z'], value='Z', description='Slice Direction:',
            style={'description_width': 'initial'}
        )

        self._update_slice_positions()
        self.slice_selector = widgets.Dropdown(
            options=[(f"{pos:.2f}", pos) for pos in self._current_positions],
            value=self.slice_position,
            description='Slice Position:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px')
        )

        # IntSlider for index-based navigation
        idx0 = np.where(self._current_positions == self.slice_position)[0][0]
        self.slice_index_slider = widgets.IntSlider(
            min=0, max=len(self._current_positions) - 1, value=int(idx0),
            description='Slice Idx:',
            continuous_update=False,
            layout=widgets.Layout(width='180px')
        )
        
        self.jump_input = widgets.BoundedIntText(
            value=1, min=1, max=10_000, step=1,
            description='Jump:', layout=widgets.Layout(width='120px'),
            style={'description_width': '50px'}
        )

        self.slice_prev_button = widgets.Button(description='◀ Prev', button_style='info', layout=widgets.Layout(width='70px'))
        self.slice_next_button = widgets.Button(description='Next ▶', button_style='info', layout=widgets.Layout(width='70px'))
        self._update_prev_next_disabled()

        self.log_toggle = widgets.Checkbox(value=self.log_scale, description='Log Scale', style={'description_width': 'initial'})
        self.outlines_toggle = widgets.Checkbox(value=self.show_cell_outlines, description='Show Cell Outlines', style={'description_width': 'initial'})
        self.points_toggle = widgets.Checkbox(value=self.show_3d_points, description='Show 3D Points', style={'description_width': 'initial'})

        self.info_label = widgets.HTML(value=f"<b>TreeMesh:</b> {self.mesh.n_cells} cells")

        # observers
        self.direction_dropdown.observe(self._on_direction_change, names='value')
        self.slice_selector.observe(self._on_slice_dropdown, names='value')
        self.slice_index_slider.observe(self._on_slice_index_slider, names='value')
        self.slice_prev_button.on_click(self._on_slice_prev)
        self.slice_next_button.on_click(self._on_slice_next)
        self.outlines_toggle.observe(self._on_outlines_toggle, names='value')
        self.log_toggle.observe(self._on_log_toggle, names='value')
        self.points_toggle.observe(self._on_points_toggle, names='value')

    # callbacks
    def _on_direction_change(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        self.slice_direction = change['new']
        self._update_slice_positions()

        # update dropdown & slider without triggering loops
        self.slice_selector.unobserve(self._on_slice_dropdown, names='value')
        self.slice_selector.options = [(f"{pos:.2f}", pos) for pos in self._current_positions]
        self.slice_position = self._current_positions[len(self._current_positions)//2]
        self.slice_selector.value = self.slice_position
        self.slice_selector.observe(self._on_slice_dropdown, names='value')

        self.slice_index_slider.unobserve(self._on_slice_index_slider, names='value')
        self.slice_index_slider.max = len(self._current_positions) - 1
        self.slice_index_slider.value = len(self._current_positions)//2
        self.slice_index_slider.observe(self._on_slice_index_slider, names='value')

        self._update_prev_next_disabled()
        self.create_figure()

    def _on_slice_dropdown(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        self.slice_position = change['new']
        self.slice_index_slider.unobserve(self._on_slice_index_slider, names='value')
        self.slice_index_slider.value = self._idx_of_current()
        self.slice_index_slider.observe(self._on_slice_index_slider, names='value')
        self._update_prev_next_disabled()
        self.create_figure()

    def _on_slice_index_slider(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        idx = change['new']
        self.slice_position = self._current_positions[idx]
        self.slice_selector.unobserve(self._on_slice_dropdown, names='value')
        self.slice_selector.value = self.slice_position
        self.slice_selector.observe(self._on_slice_dropdown, names='value')
        self._update_prev_next_disabled()
        self.create_figure()

    def _on_slice_prev(self, _):
        step = self.jump_input.value
        idx = self._idx_of_current()
        self.slice_index_slider.value = max(0, idx - step)   # prev 
        if getattr(self, '_creating_widgets', False):
            return
        idx = self._idx_of_current()
        if idx > 0:
            self.slice_index_slider.value = idx - 1

    def _on_slice_next(self, _):
        step = self.jump_input.value
        idx = self._idx_of_current()
        self.slice_index_slider.value = min(len(self._current_positions)-1, idx + step)  # next

        if getattr(self, '_creating_widgets', False):
            return
        idx = self._idx_of_current()
        if idx < len(self._current_positions) - 1:
            self.slice_index_slider.value = idx + 1

    def _on_log_toggle(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        self.log_scale = change['new']
        self._update_display_values()
        self.create_figure()

    def _on_outlines_toggle(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        self.show_cell_outlines = change['new']
        self.create_figure()

    def _on_points_toggle(self, change):
        if getattr(self, '_creating_widgets', False):
            return
        self.show_3d_points = change['new']
        self.create_figure()

    # ----------------- public -----------------

    def show(self):
        controls = widgets.VBox([
            widgets.HTML("<h3>TreeMesh Viewer</h3>"),
            self.info_label,
            self.direction_dropdown,
            widgets.HBox([self.slice_selector, self.slice_index_slider,
              self.slice_prev_button, self.slice_next_button, self.jump_input]),
            widgets.HBox([self.points_toggle, self.outlines_toggle, self.log_toggle]),
            widgets.HTML("<hr>")
        ])
        display(controls)
        display(self.fig_widget)

    def update_model(self, new_model_values):
        if len(new_model_values) != self.mesh.n_cells:
            raise ValueError(f"Model has {len(new_model_values)} values but mesh has {self.mesh.n_cells} cells")
        self.model_values = new_model_values
        self._update_display_values()
        self.create_figure()


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def create_mesh_viewer(mesh, model_values, **kwargs):
    """Return MeshViewer or TreeMeshViewer depending on mesh type."""
    if hasattr(mesh, '_meshType'):
        mtype = mesh._meshType.lower()
        if mtype == 'tree':
            return TreeMeshViewer(mesh, model_values, **kwargs)
        if mtype == 'tensor':
            return MeshViewer(mesh, model_values, **kwargs)

    # fallback heuristics
    if hasattr(mesh, 'shape_cells'):
        return MeshViewer(mesh, model_values, **kwargs)
    return TreeMeshViewer(mesh, model_values, **kwargs)
