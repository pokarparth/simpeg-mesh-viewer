"""
Mesh viewers for discretize TensorMesh and TreeMesh

USAGE
-----
from mesh_viewer import create_mesh_viewer

create_mesh_viewer(mesh, model, cmap="viridis")
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

try:
    from scipy.spatial import cKDTree  # optional fallback for TreeMesh interpolation
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Shared utils
# -----------------------------------------------------------------------------
def _log_ticks_from_range(vmin_lin, vmax_lin):
    """Return (tickvals_in_log10, ticktext) spanning [vmin_lin, vmax_lin] decades."""
    if vmin_lin <= 0 or vmax_lin <= 0:
        return None, None
    lo = int(np.floor(np.log10(vmin_lin)))
    hi = int(np.ceil(np.log10(vmax_lin)))
    exps = np.arange(lo, hi + 1)
    tickvals = exps.astype(float)
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    ticktext = [f"10{str(e).translate(sup)}" for e in exps]
    return tickvals, ticktext


def _add_wireframe(fig, bounds, row=1, col=1):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    edges = [
        ([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         [z_min] * 5),
        ([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min],
         [z_max] * 5),
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
    ]
    for xe, ye, ze in edges:
        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode="lines",
            line=dict(color="darkgray", width=2),
            opacity=0.85, showlegend=False, hoverinfo="skip"),
            row=row, col=col
        )


def _set_box_visible(box: widgets.Box, visible: bool):
    box.layout.display = "" if visible else "none"


def _subplot_layout(num_extras, for_tensor=True):
    """
    Return (rows, cols, specs, titles, column_widths, height)
    Layout: primary 3D + primary 2D on top row; extras on second row.
    The 3D scene spans both rows to keep it large.
    """
    base_title_3d = "3D Mesh with Slice(s)" if for_tensor else ""
    if num_extras == 0:
        rows, cols = 1, 2
        specs = [[{"type": "scene"}, {"type": "xy"}]]
        titles = [base_title_3d, None]
        widths = [0.65, 0.35]
        height = 600
    elif num_extras == 1:
        rows, cols = 2, 2
        specs = [[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                 [None, {"type": "xy"}]]
        titles = [base_title_3d, None, None]
        widths = [0.6, 0.4]
        height = 800
    else:  # 2 extras
        rows, cols = 2, 3
        specs = [[{"type": "scene", "rowspan": 2}, {"type": "xy"}, {"type": "xy"}],
                 [None, {"type": "xy"}, {"type": "xy"}]]
        titles = [base_title_3d, None, None, None, None]
        widths = [0.5, 0.25, 0.25]
        height = 850
    return rows, cols, specs, titles, widths, height


# -----------------------------------------------------------------------------
# TensorMesh Viewer
# -----------------------------------------------------------------------------
class MeshViewer:
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

        self.nx, self.ny, self.nz = mesh.shape_cells
        self.xc, self.yc, self.zc = mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z
        self.xn, self.yn, self.zn = mesh.nodes_x, mesh.nodes_y, mesh.nodes_z

        self._update_display_values()

        # primary slice
        self.dir = 'X'
        self.ix = self.nx // 2
        self.iy = self.ny // 2
        self.iz = self.nz // 2

        # extra slices
        self.second_on = False
        self.second_dir = 'Y'
        self.second_idx = self.ny // 2

        self.third_on = False
        self.third_dir = 'Z'
        self.third_idx = self.nz // 2

        self._creating = True
        self._build_widgets()
        self._creating = False
        self.fig_out = widgets.Output()
        self._draw()

    # ----- internals -----
    def _update_display_values(self):
        if self.log_scale and self.model_values.min() > 0:
            self.disp_vals = np.log10(self.model_values)
            self.disp_3d = np.log10(self.model_3d)
            self.vmin, self.vmax = self.disp_vals.min(), self.disp_vals.max()
            self._tickvals, self._ticktext = _log_ticks_from_range(10**self.vmin, 10**self.vmax)
        else:
            # disable log if invalid
            if self.log_scale and self.model_values.min() <= 0:
                self.log_scale = False
            self.disp_vals = self.model_values
            self.disp_3d = self.model_3d
            self.vmin, self.vmax = self.disp_vals.min(), self.disp_vals.max()
            self._tickvals, self._ticktext = (None, None)

    def _colorbar(self, primary=True):
        if not primary:
            return None
        cb = dict(title="Model Value", x=1.02, len=0.8)
        if self.log_scale and self._tickvals is not None:
            cb.update(dict(tickmode="array", tickvals=self._tickvals, ticktext=self._ticktext))
        return cb

    def _dir_arrays(self, d):
        return {'X': (self.xc, self.xn, self.nx),
                'Y': (self.yc, self.yn, self.ny),
                'Z': (self.zc, self.zn, self.nz)}[d]

    def _get_idx(self, d):
        return {'X': self.ix, 'Y': self.iy, 'Z': self.iz}[d]

    def _set_idx(self, d, v):
        if d == 'X': self.ix = v
        elif d == 'Y': self.iy = v
        else: self.iz = v

    def _slice_arrays(self, d, idx):
        if d == 'X':
            y, z = np.meshgrid(self.yc, self.zc)
            x = np.full_like(y, self.xc[idx])
            col = self.disp_3d[idx, :, :].T
        elif d == 'Y':
            x, z = np.meshgrid(self.xc, self.zc)
            y = np.full_like(x, self.yc[idx])
            col = self.disp_3d[:, idx, :].T
        else:
            x, y = np.meshgrid(self.xc, self.yc)
            z = np.full_like(x, self.zc[idx])
            col = self.disp_3d[:, :, idx].T
        return x, y, z, col

    # ----- figure -----
    def _draw(self):
        with self.fig_out:
            clear_output(wait=True)

            num_extras = int(self.second_on) + int(self.third_on)
            rows, cols, specs, titles, widths, height = _subplot_layout(num_extras, for_tensor=True)

            # primary
            arr_primary, _, _ = self._dir_arrays(self.dir)
            idx_primary = self._get_idx(self.dir)
            titles[1] = f"{self.dir} Slice @ {arr_primary[idx_primary]:.2f}"
            # one extra
            if num_extras == 1:
                if self.second_on:
                    arr2, _, _ = self._dir_arrays(self.second_dir)
                    titles[2] = f"{self.second_dir} Slice @ {arr2[self.second_idx]:.2f}"
                else:
                    arr3, _, _ = self._dir_arrays(self.third_dir)
                    titles[2] = f"{self.third_dir} Slice @ {arr3[self.third_idx]:.2f}"
            # two extras
            elif num_extras == 2:
                arr2, _, _ = self._dir_arrays(self.second_dir)
                arr3, _, _ = self._dir_arrays(self.third_dir)
                titles[3] = f"{self.second_dir} Slice @ {arr2[self.second_idx]:.2f}"
                titles[4] = f"{self.third_dir} Slice @ {arr3[self.third_idx]:.2f}"

            fig = make_subplots(
                rows=rows, cols=cols, specs=specs,
                subplot_titles=tuple(titles),
                horizontal_spacing=0.05, vertical_spacing=0.12,
                column_widths=widths
            )

            # 3D scene
            if self.show_3d_points:
                skip = max(1, min(self.nx, self.ny, self.nz)//8)
                X, Y, Z = np.meshgrid(self.xc[::skip], self.yc[::skip], self.zc[::skip], indexing='ij')
                fig.add_trace(go.Scatter3d(
                    x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                    marker=dict(size=2,
                                color=self.disp_3d[::skip, ::skip, ::skip].ravel(),
                                colorscale=self.cmap, opacity=0.65,
                                cmin=self.vmin, cmax=self.vmax,
                                colorbar=self._colorbar(False)),
                    hoverinfo='skip'),
                    row=1, col=1)

            _add_wireframe(fig, ((self.xn[0], self.xn[-1]),
                                 (self.yn[0], self.yn[-1]),
                                 (self.zn[0], self.zn[-1])), 1, 1)

            # primary slice (always row1-col2)
            self._add_slice(fig, self.dir, self._get_idx(self.dir), row=1, col=2, primary=True)

            # extras (row2)
            if num_extras >= 1:
                col_bottom = 2 if rows == 2 and cols == 2 else 2  # starting col on second row
                if self.second_on:
                    # position depends on cols
                    target_col = 2 if num_extras == 1 else 2
                    self._add_slice(fig, self.second_dir, self.second_idx, row=2, col=target_col, primary=False)
                    if num_extras == 2:
                        col_bottom = 3
                if self.third_on:
                    target_col = 3 if cols == 3 else 2  # if 2 extras -> col3, else same col2
                    self._add_slice(fig, self.third_dir, self.third_idx, row=2, col=target_col, primary=False)

            fig.update_layout(
                title=f"TensorMesh Viewer{' (Log)' if self.log_scale else ''}",
                height=height, autosize=True, showlegend=False
            )
            fig.update_scenes(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                              aspectmode='cube',
                              camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))
            fig.show()

    def _add_slice(self, fig, d, idx, row, col, primary):
        # 3D plane
        x, y, z, col_vals = self._slice_arrays(d, idx)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=col_vals,
            colorscale=self.cmap, cmin=self.vmin, cmax=self.vmax,
            opacity=1.0 if primary else 0.8,
            showscale=primary, colorbar=self._colorbar(primary),
            name=f"{d}_plane"), row=1, col=1)

        # 2D heatmap
        if d == 'X':
            xs, ys, xn, yn, xl, yl = self.yc, self.zc, self.yn, self.zn, "Y", "Z"
        elif d == 'Y':
            xs, ys, xn, yn, xl, yl = self.xc, self.zc, self.xn, self.zn, "X", "Z"
        else:
            xs, ys, xn, yn, xl, yl = self.xc, self.yc, self.xn, self.yn, "X", "Y"

        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=col_vals,
            colorscale=self.cmap, zmin=self.vmin, zmax=self.vmax,
            showscale=False,
            hovertemplate=f'{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<br>Val: %{{z:.3f}}<extra></extra>',
            name=f"{d}_slice2d"), row=row, col=col)

        if self.show_cell_outlines:
            xs_l, ys_l = [], []
            for i in range(0, len(xn), self.outline_step):
                xs_l.extend([xn[i], xn[i], None])
                ys_l.extend([yn[0], yn[-1], None])
            for j in range(0, len(yn), self.outline_step):
                xs_l.extend([xn[0], xn[-1], None])
                ys_l.extend([yn[j], yn[j], None])
            fig.add_trace(go.Scatter(x=xs_l, y=ys_l, mode='lines',
                                     line=dict(color='black', width=1),
                                     hoverinfo='skip', showlegend=False),
                          row=row, col=col)

        fig.update_xaxes(title_text=xl, row=row, col=col)
        fig.update_yaxes(title_text=yl, row=row, col=col)

    # ----- widgets -----
    def _build_widgets(self):
        # primary
        self.dir_dd = widgets.Dropdown(options=['X', 'Y', 'Z'], value='X', description='Dir:')
        self.idx_slider = widgets.IntSlider(value=self.ix, min=0, max=self.nx-1,
                                            description='Idx:', continuous_update=False,
                                            layout=widgets.Layout(width='280px'))
        self.prev_btn = widgets.Button(description='◀')
        self.next_btn = widgets.Button(description='▶')
        self.jump_txt = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')
        self.pos_lbl = widgets.HTML()

        self.points_cb = widgets.Checkbox(value=self.show_3d_points, description='3D Points')
        self.outline_cb = widgets.Checkbox(value=self.show_cell_outlines, description='Show Cell Outlines')
        self.log_cb = widgets.Checkbox(value=self.log_scale, description='Log Scaling')

        # second
        self.second_cb = widgets.Checkbox(value=False, description='Second slice')
        self.second_dir_dd = widgets.Dropdown(options=['X','Y','Z'], value='Y', description='Dir:')
        self.second_idx_slider = widgets.IntSlider(value=self.second_idx, min=0, max=self.ny-1,
                                                   description='Idx:', continuous_update=False,
                                                   layout=widgets.Layout(width='280px'))
        self.second_prev = widgets.Button(description='◀')
        self.second_next = widgets.Button(description='▶')
        self.second_jump = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')

        # third
        self.third_cb = widgets.Checkbox(value=False, description='Third slice')
        self.third_dir_dd = widgets.Dropdown(options=['X','Y','Z'], value='Z', description='Dir:')
        self.third_idx_slider = widgets.IntSlider(value=self.third_idx, min=0, max=self.nz-1,
                                                  description='Idx:', continuous_update=False,
                                                  layout=widgets.Layout(width='280px'))
        self.third_prev = widgets.Button(description='◀')
        self.third_next = widgets.Button(description='▶')
        self.third_jump = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')

        # containers (dynamic visibility)
        self.second_controls = widgets.HBox([self.second_dir_dd, self.second_idx_slider,
                                             self.second_prev, self.second_next, self.second_jump])
        self.third_controls = widgets.HBox([self.third_dir_dd, self.third_idx_slider,
                                            self.third_prev, self.third_next, self.third_jump])
        _set_box_visible(self.second_controls, self.second_on)
        _set_box_visible(self.third_controls, self.third_on)

        # observers
        self.dir_dd.observe(self._on_dir, names='value')
        self.idx_slider.observe(self._on_idx, names='value')
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.points_cb.observe(lambda ch: self._toggle('show_3d_points', ch), names='value')
        self.outline_cb.observe(lambda ch: self._toggle('show_cell_outlines', ch), names='value')
        self.log_cb.observe(self._on_log, names='value')

        self.second_cb.observe(self._on_second_toggle, names='value')
        self.second_dir_dd.observe(self._on_second_dir, names='value')
        self.second_idx_slider.observe(self._on_second_idx, names='value')
        self.second_prev.on_click(self._on_second_prev)
        self.second_next.on_click(self._on_second_next)

        self.third_cb.observe(self._on_third_toggle, names='value')
        self.third_dir_dd.observe(self._on_third_dir, names='value')
        self.third_idx_slider.observe(self._on_third_idx, names='value')
        self.third_prev.on_click(self._on_third_prev)
        self.third_next.on_click(self._on_third_next)

        self._update_pos_label()

    # primary callbacks
    def _on_dir(self, ch):
        if self._creating: return
        d = ch['new']
        self.dir = d
        _, _, n = self._dir_arrays(d)
        self.idx_slider.max = n - 1
        self.idx_slider.value = self._get_idx(d)
        self._update_pos_label()
        self._draw()

    def _on_idx(self, ch):
        if self._creating: return
        idx = ch['new']
        self._set_idx(self.dir, idx)
        self._update_pos_label()
        self._draw()

    def _on_prev(self, _):
        step = self.jump_txt.value
        cur = self._get_idx(self.dir)
        self.idx_slider.value = max(0, cur - step)

    def _on_next(self, _):
        step = self.jump_txt.value
        _, _, n = self._dir_arrays(self.dir)
        cur = self._get_idx(self.dir)
        self.idx_slider.value = min(n - 1, cur + step)

    def _on_log(self, ch):
        if self._creating: return
        self.log_scale = ch['new']
        self._update_display_values()
        self._draw()

    def _toggle(self, attr, ch):
        if self._creating: return
        setattr(self, attr, ch['new'])
        self._draw()

    def _update_pos_label(self):
        arr, _, _ = self._dir_arrays(self.dir)
        self.pos_lbl.value = f"<b>Pos: {arr[self._get_idx(self.dir)]:.2f}</b>"

    # second callbacks
    def _on_second_toggle(self, ch):
        if self._creating: return
        self.second_on = ch['new']
        _set_box_visible(self.second_controls, self.second_on)
        self._draw()

    def _on_second_dir(self, ch):
        if self._creating: return
        d = ch['new']
        self.second_dir = d
        _, _, n = self._dir_arrays(d)
        self.second_idx_slider.max = n - 1
        self.second_idx_slider.value = n // 2
        self.second_idx = self.second_idx_slider.value
        self._draw()

    def _on_second_idx(self, ch):
        if self._creating: return
        self.second_idx = ch['new']
        self._draw()

    def _on_second_prev(self, _):
        step = self.second_jump.value
        self.second_idx_slider.value = max(0, self.second_idx - step)

    def _on_second_next(self, _):
        step = self.second_jump.value
        _, _, n = self._dir_arrays(self.second_dir)
        self.second_idx_slider.value = min(n - 1, self.second_idx + step)

    # third callbacks
    def _on_third_toggle(self, ch):
        if self._creating: return
        self.third_on = ch['new']
        _set_box_visible(self.third_controls, self.third_on)
        self._draw()

    def _on_third_dir(self, ch):
        if self._creating: return
        d = ch['new']
        self.third_dir = d
        _, _, n = self._dir_arrays(d)
        self.third_idx_slider.max = n - 1
        self.third_idx_slider.value = n // 2
        self.third_idx = self.third_idx_slider.value
        self._draw()

    def _on_third_idx(self, ch):
        if self._creating: return
        self.third_idx = ch['new']
        self._draw()

    def _on_third_prev(self, _):
        step = self.third_jump.value
        self.third_idx_slider.value = max(0, self.third_idx - step)

    def _on_third_next(self, _):
        step = self.third_jump.value
        _, _, n = self._dir_arrays(self.third_dir)
        self.third_idx_slider.value = min(n - 1, self.third_idx + step)

    # public
    def show(self):
        row1 = widgets.HBox([self.dir_dd, self.idx_slider, self.prev_btn, self.next_btn,
                             self.jump_txt, self.pos_lbl])
        row2 = widgets.HBox([self.points_cb, self.outline_cb, self.log_cb])

        second_box = widgets.VBox([self.second_cb, self.second_controls])
        third_box = widgets.VBox([self.third_cb, self.third_controls])

        controls = widgets.VBox([
            widgets.HTML("<h3>TensorMesh Viewer</h3>"),
            row1, row2,
            widgets.HTML("<b>Additional slices</b>"),
            second_box, third_box,
            widgets.HTML("<hr>")
        ])
        display(controls)
        display(self.fig_out)


# -----------------------------------------------------------------------------
# TreeMesh Viewer
# -----------------------------------------------------------------------------
class TreeMeshViewer:
    def __init__(self, mesh, model_values, cmap="plasma",
                 show_3d_points=False, show_cell_outlines=True, log_scale=False):
        if not hasattr(mesh, '_meshType') or mesh._meshType.lower() != 'tree':
            raise ValueError("TreeMeshViewer only works with TreeMesh objects")
        if len(model_values) != mesh.n_cells:
            raise ValueError("Model length mismatch")

        self.mesh = mesh
        self.model_values = model_values
        self.cmap = cmap
        self.show_3d_points = show_3d_points
        self.show_cell_outlines = show_cell_outlines
        self.log_scale = log_scale

        self.cc = mesh.gridCC
        self.xb = [self.cc[:, 0].min(), self.cc[:, 0].max()]
        self.yb = [self.cc[:, 1].min(), self.cc[:, 1].max()]
        self.zb = [self.cc[:, 2].min(), self.cc[:, 2].max()]

        self.ux = np.unique(self.cc[:, 0])
        self.uy = np.unique(self.cc[:, 1])
        self.uz = np.unique(self.cc[:, 2])

        # primary slice
        self.dir = 'X'
        self.pos = self.ux[len(self.ux)//2]

        # extras
        self.second_on = False
        self.second_dir = 'Y'
        self.second_pos = self.uy[len(self.uy)//2]

        self.third_on = False
        self.third_dir = 'Z'
        self.third_pos = self.uz[len(self.uz)//2]

        # caches
        self._grid1 = self._grid2 = self._grid3 = None

        self._update_display_values()

        self.fig_out = widgets.Output()
        self.fig_out.layout.width = "100%"
        self._creating = True
        self._build_widgets()
        self._creating = False
        self._draw()

    # ----- internals -----
    def _update_display_values(self):
        if self.log_scale and self.model_values.min() > 0:
            self.disp_vals = np.log10(self.model_values)
            vmin_lin, vmax_lin = self.model_values.min(), self.model_values.max()
            self._tickvals, self._ticktext = _log_ticks_from_range(vmin_lin, vmax_lin)
        else:
            if self.log_scale and self.model_values.min() <= 0:
                print("Warning: Log scale disabled due to non-positive model values.")
                self.log_scale = False
            self.disp_vals = self.model_values.copy()
            self._tickvals, self._ticktext = (None, None)
        self.vmin, self.vmax = self.disp_vals.min(), self.disp_vals.max()

    def _colorbar(self, primary=True):
        if not primary:
            return None
        base = dict(title="Model Value", x=1.02, len=0.8)
        if self.log_scale and self._tickvals is not None:
            base.update(dict(tickmode="array", tickvals=self._tickvals, ticktext=self._ticktext))
        return base

    def _arr_for_dir(self, d):
        return {'X': self.ux, 'Y': self.uy, 'Z': self.uz}[d]

    # ----- figure -----
    def _draw(self):
        with self.fig_out:
            clear_output(wait=True)

            num_extras = int(self.second_on) + int(self.third_on)
            rows, cols, specs, titles, widths, height = _subplot_layout(num_extras, for_tensor=False)

            titles[1] = f"{self.dir} Slice @ {self.pos:.2f}"
            if num_extras == 1:
                if self.second_on:
                    titles[2] = f"{self.second_dir} Slice @ {self.second_pos:.2f}"
                else:
                    titles[2] = f"{self.third_dir} Slice @ {self.third_pos:.2f}"
            elif num_extras == 2:
                titles[3] = f"{self.second_dir} Slice @ {self.second_pos:.2f}"
                titles[4] = f"{self.third_dir} Slice @ {self.third_pos:.2f}"

            fig = make_subplots(
                rows=rows, cols=cols, specs=specs,
                subplot_titles=tuple(titles),
                horizontal_spacing=0.05, vertical_spacing=0.12,
                column_widths=widths
            )

            if self.show_3d_points:
                fig.add_trace(go.Scatter3d(
                    x=self.cc[:, 0], y=self.cc[:, 1], z=self.cc[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=self.disp_vals,
                                colorscale=self.cmap, opacity=0.65,
                                cmin=self.vmin, cmax=self.vmax,
                                colorbar=self._colorbar(False))),
                    row=1, col=1)

            _add_wireframe(fig, (tuple(self.xb), tuple(self.yb), tuple(self.zb)), 1, 1)

            # primary slice -> row1 col2
            self._grid1 = self._tree_slice(fig, self.dir, self.pos, row=1, col=2, primary=True)

            # extras
            if num_extras >= 1:
                if self.second_on:
                    # one extra -> row2 col2; two extras -> row2 col2 (first)
                    self._grid2 = self._tree_slice(fig, self.second_dir, self.second_pos,
                                                   row=2, col=2 if cols == 2 else 2, primary=False)
                if self.third_on:
                    self._grid3 = self._tree_slice(fig, self.third_dir, self.third_pos,
                                                   row=2, col=3 if cols == 3 else 2, primary=False)

            fig.update_layout(title=f"TreeMesh Viewer{' (Log)' if self.log_scale else ''}",
                              height=height, autosize=True, showlegend=False)
            fig.update_scenes(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                              aspectmode='cube',
                              camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))

            display(fig)

    def _tree_slice(self, fig, d, pos, row, col, primary):
        normal = {"X": 0, "Y": 1, "Z": 2}[d]
        others = {"X": [1, 2], "Y": [0, 2], "Z": [0, 1]}[d]

        origin = np.zeros(3); origin[normal] = pos
        nvec = [0, 0, 0]; nvec[normal] = 1

        try:
            inds = self.mesh.get_cells_on_plane(origin, nvec)
        except Exception:
            mask = np.abs(self.cc[:, normal] - pos) < 1e-10
            inds = np.where(mask)[0]

        if len(inds) == 0:
            if primary:
                fig.add_annotation(text=f"No cells @ {d}={pos:.2f}",
                                   x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"),
                                   row=row, col=col)
            return None, None, None

        import discretize
        h2d = (self.mesh.h[others[0]], self.mesh.h[others[1]])
        o2d = (self.mesh.origin[others[0]], self.mesh.origin[others[1]])
        tm2 = discretize.TreeMesh(h2d, o2d, diagonal_balance=False)
        level_diff = self.mesh.max_level - tm2.max_level

        levels = self.mesh._cell_levels_by_indexes(inds) - level_diff
        grid2d = self.cc[inds][:, others]
        tm2.insert_cells(grid2d, levels)

        node_grid = np.r_[tm2.nodes, tm2.hanging_nodes]
        cell_nodes = tm2.cell_nodes[:, (0, 1, 3, 2)]
        cell_verts = node_grid[cell_nodes]

        boost = np.empty((tm2.n_cells, 3))
        boost[:, others] = tm2.cell_centers
        boost[:, normal] = pos
        idx3 = self.mesh.get_containing_cells(boost)
        v2d = self.disp_vals[idx3]

        x_coords = tm2.cell_centers[:, 0]
        y_coords = tm2.cell_centers[:, 1]
        nx = min(200, len(np.unique(x_coords))*2)
        ny = min(200, len(np.unique(y_coords))*2)
        xg = np.linspace(x_coords.min(), x_coords.max(), nx)
        yg = np.linspace(y_coords.min(), y_coords.max(), ny)
        Xg, Yg = np.meshgrid(xg, yg)

        pts2d = np.column_stack([Xg.ravel(), Yg.ravel()])
        pts3d = np.zeros((len(pts2d), 3))
        pts3d[:, others] = pts2d
        pts3d[:, normal] = pos

        try:
            cont = self.mesh.get_containing_cells(pts3d)
            Z = np.full(len(pts2d), np.nan)
            mask = cont != -1
            Z[mask] = self.disp_vals[cont[mask]]
        except Exception:
            if HAS_SCIPY:
                tree = cKDTree(tm2.cell_centers)
                dist, idd = tree.query(pts2d)
                Z = v2d[idd]
                cell_sizes = np.array([tm2.h[0].min(), tm2.h[1].min()])
                maxd = np.sqrt(cell_sizes[0]**2 + cell_sizes[1]**2)/2
                Z[dist > maxd] = np.nan
            else:
                Z = np.full(len(pts2d), v2d.mean())
        Zg = Z.reshape(Xg.shape)


        # 2D heatmap
        labels = {"X": ["Y", "Z"], "Y": ["X", "Z"], "Z": ["X", "Y"]}
        fig.add_trace(go.Heatmap(
            x=xg, y=yg, z=Zg, colorscale=self.cmap,
            zmin=self.vmin, zmax=self.vmax,
            showscale=False,
            hovertemplate=f'{labels[d][0]}:%{{x:.2f}}<br>{labels[d][1]}:%{{y:.2f}}<br>Val:%{{z:.3f}}<extra></extra>',
        ), row=row, col=col)

        # Add cell outlines for all slices
        xs, ys = [], []
        for v in cell_verts:
            xs.extend([v[0, 0], v[1, 0], v[2, 0], v[3, 0], v[0, 0], None])
            ys.extend([v[0, 1], v[1, 1], v[2, 1], v[3, 1], v[0, 1], None])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                 line=dict(color='black', width=0.5),
                                 hoverinfo='skip', showlegend=False),
                      row=row, col=col)

        fig.update_xaxes(title_text=labels[d][0], row=row, col=col)
        fig.update_yaxes(title_text=labels[d][1], row=row, col=col)

        # 3D plane
        if d == 'X':
            x_plane = np.full_like(Xg, pos); y_plane, z_plane = Xg, Yg
        elif d == 'Y':
            y_plane = np.full_like(Xg, pos); x_plane, z_plane = Xg, Yg
        else:
            z_plane = np.full_like(Xg, pos); x_plane, y_plane = Xg, Yg

        fig.add_trace(go.Surface(
            x=x_plane, y=y_plane, z=z_plane,
            surfacecolor=Zg, colorscale=self.cmap,
            cmin=self.vmin, cmax=self.vmax,
            opacity=1.0 if primary else 0.8,
            showscale=primary, colorbar=self._colorbar(primary)
        ), row=1, col=1)

        return (Xg, Yg, Zg)

    # ----- widgets -----
    def _build_widgets(self):
        self.dir_dd = widgets.Dropdown(options=['X', 'Y', 'Z'], value='X', description='Dir:')
        arr = self._arr_for_dir(self.dir)
        self.idx_slider = widgets.IntSlider(
            min=0, max=len(arr) - 1,
            value=np.where(arr == self.pos)[0][0],
            description='Idx:', continuous_update=False,
            layout=widgets.Layout(width='280px')
        )
        self.jump_txt = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')
        self.prev_btn = widgets.Button(description='◀')
        self.next_btn = widgets.Button(description='▶')

        self.points_cb = widgets.Checkbox(value=self.show_3d_points, description='3D Points')
        self.outline_cb = widgets.Checkbox(value=self.show_cell_outlines, description='Show Cell Outlines')
        self.log_cb = widgets.Checkbox(value=self.log_scale, description='Log Scaling')

        self.info_lbl = widgets.HTML(f"<b>TreeMesh:</b> {self.mesh.n_cells} cells")

        # second slice
        self.second_cb = widgets.Checkbox(value=False, description='Second slice')
        self.second_dir_dd = widgets.Dropdown(options=['X', 'Y', 'Z'], value='Y', description='Dir:')

        arr2 = self._arr_for_dir('Y')
        if len(arr2) == 0:
            raise ValueError("No valid Y-coordinates found in the mesh.")
        if self.second_pos not in arr2:
            self.second_pos = arr2[len(arr2) // 2]
        self.second_idx_slider = widgets.IntSlider(
            min=0, max=len(arr2) - 1,
            value=np.where(arr2 == self.second_pos)[0][0],
            description='Idx:', continuous_update=False,
            layout=widgets.Layout(width='280px')
        )

        self.second_jump = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')
        self.second_prev = widgets.Button(description='◀')
        self.second_next = widgets.Button(description='▶')
        self.second_controls = widgets.HBox([self.second_dir_dd, self.second_idx_slider,
                                             self.second_prev, self.second_next, self.second_jump])
        _set_box_visible(self.second_controls, self.second_on)

        # third slice
        self.third_cb = widgets.Checkbox(value=False, description='Third slice')
        self.third_dir_dd = widgets.Dropdown(options=['X', 'Y', 'Z'], value='Z', description='Dir:')
        arr3 = self._arr_for_dir('Z')
        self.third_idx_slider = widgets.IntSlider(
            min=0, max=len(arr3) - 1,
            value=np.where(arr3 == self.third_pos)[0][0],
            description='Idx:', continuous_update=False,
            layout=widgets.Layout(width='280px')
        )
        self.third_jump = widgets.BoundedIntText(value=1, min=1, max=10000, description='Move by:')
        self.third_prev = widgets.Button(description='◀')
        self.third_next = widgets.Button(description='▶')
        self.third_controls = widgets.HBox([self.third_dir_dd, self.third_idx_slider,
                                            self.third_prev, self.third_next, self.third_jump])
        _set_box_visible(self.third_controls, self.third_on)

        # observers
        self.dir_dd.observe(self._on_dir, names='value')
        self.idx_slider.observe(self._on_idx_slider, names='value')
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)

        self.points_cb.observe(lambda ch: self._toggle('show_3d_points', ch), names='value')
        self.outline_cb.observe(lambda ch: self._toggle('show_cell_outlines', ch), names='value')
        self.log_cb.observe(self._on_log, names='value')

        self.second_cb.observe(self._on_second_toggle, names='value')
        self.second_dir_dd.observe(self._on_second_dir, names='value')
        self.second_idx_slider.observe(self._on_second_idx, names='value')
        self.second_prev.on_click(self._on_second_prev)
        self.second_next.on_click(self._on_second_next)

        self.third_cb.observe(self._on_third_toggle, names='value')
        self.third_dir_dd.observe(self._on_third_dir, names='value')
        self.third_idx_slider.observe(self._on_third_idx, names='value')
        self.third_prev.on_click(self._on_third_prev)
        self.third_next.on_click(self._on_third_next)

    # primary callbacks
    def _on_dir(self, ch):
        if self._creating: return
        self.dir = ch['new']
        arr = self._arr_for_dir(self.dir)
        self.pos = arr[len(arr)//2]

        self.idx_slider.unobserve(self._on_idx_slider, names='value')
        self.idx_slider.max = len(arr)-1
        self.idx_slider.value = len(arr)//2
        self.idx_slider.observe(self._on_idx_slider, names='value')

        self._draw()

    def _on_idx_slider(self, ch):
        if self._creating: return
        idx = ch['new']
        arr = self._arr_for_dir(self.dir)
        self.pos = arr[idx]
        self._draw()

    def _on_prev(self, _):
        step = self.jump_txt.value
        arr = self._arr_for_dir(self.dir)
        idx = np.where(arr == self.pos)[0][0]
        new_idx = max(0, idx - step)
        self.idx_slider.value = new_idx

    def _on_next(self, _):
        step = self.jump_txt.value
        arr = self._arr_for_dir(self.dir)
        idx = np.where(arr == self.pos)[0][0]
        new_idx = min(len(arr) - 1, idx + step)
        self.idx_slider.value = new_idx

    def _on_log(self, ch):
        if self._creating: return
        self.log_scale = ch['new']
        self._update_display_values()
        self._draw()

    def _toggle(self, attr, ch):
        setattr(self, attr, ch['new'])
        self._draw()

    # second callbacks
    def _on_second_toggle(self, ch):
        if self._creating: return
        self.second_on = ch['new']
        _set_box_visible(self.second_controls, self.second_on)
        self._draw()

    def _on_second_dir(self, ch):
        if self._creating: return
        self.second_dir = ch['new']
        arr = self._arr_for_dir(self.second_dir)
        self.second_idx_slider.unobserve(self._on_second_idx, names='value')
        self.second_idx_slider.max = len(arr)-1
        self.second_idx_slider.value = len(arr)//2
        self.second_idx_slider.observe(self._on_second_idx, names='value')
        self.second_pos = arr[self.second_idx_slider.value]
        self._draw()

    def _on_second_idx(self, ch):
        if self._creating: return
        arr = self._arr_for_dir(self.second_dir)
        self.second_pos = arr[ch['new']]
        self._draw()

    def _on_second_prev(self, _):
        step = self.second_jump.value
        arr = self._arr_for_dir(self.second_dir)
        idx = np.where(arr == self.second_pos)[0][0]
        self.second_idx_slider.value = max(0, idx - step)

    def _on_second_next(self, _):
        step = self.second_jump.value
        arr = self._arr_for_dir(self.second_dir)
        idx = np.where(arr == self.second_pos)[0][0]
        self.second_idx_slider.value = min(len(arr)-1, idx + step)

    # third callbacks
    def _on_third_toggle(self, ch):
        if self._creating: return
        self.third_on = ch['new']
        _set_box_visible(self.third_controls, self.third_on)
        self._draw()

    def _on_third_dir(self, ch):
        if self._creating: return
        self.third_dir = ch['new']
        arr = self._arr_for_dir(self.third_dir)
        self.third_idx_slider.unobserve(self._on_third_idx, names='value')
        self.third_idx_slider.max = len(arr)-1
        self.third_idx_slider.value = len(arr)//2
        self.third_idx_slider.observe(self._on_third_idx, names='value')
        self.third_pos = arr[self.third_idx_slider.value]
        self._draw()

    def _on_third_idx(self, ch):
        if self._creating: return
        arr = self._arr_for_dir(self.third_dir)
        self.third_pos = arr[ch['new']]
        self._draw()

    def _on_third_prev(self, _):
        step = self.third_jump.value
        arr = self._arr_for_dir(self.third_dir)
        idx = np.where(arr == self.third_pos)[0][0]
        self.third_idx_slider.value = max(0, idx - step)

    def _on_third_next(self, _):
        step = self.third_jump.value
        arr = self._arr_for_dir(self.third_dir)
        idx = np.where(arr == self.third_pos)[0][0]
        self.third_idx_slider.value = min(len(arr)-1, idx + step)

    # public
    def show(self):
        r1 = widgets.HBox([self.dir_dd, self.idx_slider, self.prev_btn, self.next_btn, self.jump_txt])
        r2 = widgets.HBox([self.points_cb, self.outline_cb, self.log_cb])
    
        sec_box = widgets.VBox([self.second_cb, self.second_controls])
        third_box = widgets.VBox([self.third_cb, self.third_controls])
    
        controls = widgets.VBox([
            widgets.HTML("<h3>TreeMesh Viewer</h3>"),
            self.info_lbl,
            r1, r2,
            widgets.HTML("<b>Additional slices</b>"),
            sec_box, third_box,
            widgets.HTML("<hr>")
        ])
        display(controls)
        display(self.fig_out)
# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_mesh_viewer(mesh, model_values, **kwargs):
    """Return MeshViewer or TreeMeshViewer depending on mesh._meshType."""
    if hasattr(mesh, '_meshType'):
        t = mesh._meshType.lower()
        if t == 'tree':
            return TreeMeshViewer(mesh, model_values, **kwargs).show()
        if t == 'tensor':
            return MeshViewer(mesh, model_values, **kwargs).show()
    # fallback
    return MeshViewer(mesh, model_values, **kwargs).show()
