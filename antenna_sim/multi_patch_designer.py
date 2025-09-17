from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .models import PatchAntennaParams, Metal, metal_defaults
from .solver_fdtd_openems_microstrip import FeedDirection, calculate_microstrip_width
from .physics import design_patch_for_frequency


@dataclass
class PatchInstance:
    name: str
    params: PatchAntennaParams
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    center_z_m: float = 0.0
    feed_direction: FeedDirection = FeedDirection.NEG_X
    rot_x_deg: float = 0.0  # rotation about global X (degrees)
    rot_y_deg: float = 0.0  # rotation about global Y (degrees)
    rot_z_deg: float = 0.0  # rotation about global Z (degrees)


class MultiPatchPanel(ttk.Frame):
    """Embeddable panel for arranging multiple microstrip patch antennas.

    This class contains the full UI (figure + right-side controls) and can be
    placed either in a Toplevel window or embedded inside an existing frame
    (e.g., inside the GUI's Geometry tab).
    """

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.patches: List[PatchInstance] = []
        self._current_index: Optional[int] = None
        self._build_ui()
        self._init_axes()
        self._draw_scene()

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.left = ttk.Frame(self)
        self.left.grid(row=0, column=0, sticky='nsew')
        self.right = ttk.Frame(self, width=320)
        self.right.grid(row=0, column=1, sticky='ns')

        # Figure
        self.fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        try:
            NavigationToolbar2Tk(self.canvas, self.left)
        except Exception:
            pass
        self.canvas.draw()
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Right controls
        self._build_right_controls(self.right)

    def _build_right_controls(self, parent: ttk.Frame):
        ttk.Label(parent, text="Multi Patch Controls", font=("Segoe UI", 12, "bold")).pack(fill='x', padx=10, pady=(12, 6))

        ttk.Button(parent, text="Add Patch Antenna", command=self._on_add_patch).pack(fill='x', padx=10, pady=(0, 10))

        sel = ttk.Frame(parent)
        sel.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Label(sel, text="Select Patch:").grid(row=0, column=0, sticky='w')
        self.sel_var = tk.StringVar()
        self.sel_combo = ttk.Combobox(sel, textvariable=self.sel_var, state='readonly')
        self.sel_combo.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        sel.columnconfigure(1, weight=1)
        self.sel_combo.bind('<<ComboboxSelected>>', self._on_select_patch)

        props = ttk.LabelFrame(parent, text="Selected Patch Properties")
        props.pack(fill='x', padx=10, pady=(0, 10))
        try:
            props.columnconfigure(1, weight=1)
        except Exception:
            pass

        self.var_cx = tk.DoubleVar(value=0.0)
        self.var_cy = tk.DoubleVar(value=0.0)
        self.var_freq = tk.DoubleVar(value=2.45)
        self.var_eps = tk.DoubleVar(value=4.3)
        self.var_h = tk.DoubleVar(value=1.6)
        self.var_loss = tk.DoubleVar(value=0.02)
        self.var_metal = tk.StringVar(value=Metal.COPPER.value)
        # Rotation (deg) about global axes
        self.var_rx = tk.DoubleVar(value=0.0)
        self.var_ry = tk.DoubleVar(value=0.0)
        self.var_rz = tk.DoubleVar(value=0.0)
        # Z translation
        self.var_cz = tk.DoubleVar(value=0.0)

        r = 0
        def mk_row(lbl, entry_var, field_key, is_combo=False):
            nonlocal r
            ttk.Label(props, text=lbl).grid(row=r, column=0, sticky='w')
            if not is_combo:
                e = ttk.Entry(props, textvariable=entry_var, width=12)
                e.grid(row=r, column=1, sticky='ew')
                e.bind('<Return>', lambda ev, f=field_key: self._apply_single_field(f))
            else:
                e = ttk.Combobox(props, textvariable=entry_var, state='readonly', values=[m.value for m in Metal])
                e.grid(row=r, column=1, sticky='ew')
                e.bind('<<ComboboxSelected>>', lambda ev, f=field_key: self._apply_single_field(f))
            ttk.Button(props, text='Set', width=6, command=lambda f=field_key: self._apply_single_field(f)).grid(row=r, column=2, padx=(6,0))
            r += 1
            return e

        mk_row("Center X (m)", self.var_cx, 'cx')
        mk_row("Center Y (m)", self.var_cy, 'cy')
        mk_row("Frequency (GHz)", self.var_freq, 'freq')
        mk_row("Dielectric εr", self.var_eps, 'eps')
        mk_row("Substrate h (mm)", self.var_h, 'h')
        mk_row("Loss tangent", self.var_loss, 'loss')
        self.metal_combo = mk_row("Metal", self.var_metal, 'metal', is_combo=True)
        mk_row("Center Z (m)", self.var_cz, 'cz')
        mk_row("Rotate X (°)", self.var_rx, 'rx')
        mk_row("Rotate Y (°)", self.var_ry, 'ry')
        mk_row("Rotate Z (°)", self.var_rz, 'rz')

        ttk.Button(parent, text="Apply Changes", command=self._on_apply_changes).pack(fill='x', padx=10, pady=(0, 6))
        ttk.Button(parent, text="Remove Selected", command=self._on_remove_selected).pack(fill='x', padx=10, pady=(0, 6))
        ttk.Label(parent, text="Tip: Rotate with mouse, scroll to zoom.").pack(fill='x', padx=10, pady=(8, 10))

    # ---------- Axes and rendering ----------
    def _init_axes(self):
        ax = self.ax
        ax.clear()
        self._lim = 0.15  # meters -> ~0.3 m span (smaller default view)
        ax.set_xlim([-self._lim, self._lim])
        ax.set_ylim([-self._lim, self._lim])
        ax.set_zlim([-self._lim, self._lim])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_facecolor('#2b2b2b')
        self.fig.patch.set_facecolor('#2b2b2b')
        ax.grid(False)
        ax.set_axis_off()

        # Axis lines and labels
        L = self._lim
        ax.plot([-L, L], [0, 0], [0, 0], color='red', linewidth=2.0)
        ax.plot([0, 0], [-L, L], [0, 0], color='green', linewidth=2.0)
        ax.plot([0, 0], [0, 0], [-L, L], color='blue', linewidth=2.0)
        ax.text(L + 0.05, 0, 0, '+X', color='red', weight='bold')
        ax.text(0, L + 0.05, 0, '+Y', color='green', weight='bold')
        ax.text(0, 0, L + 0.05, '+Z', color='blue', weight='bold')

    def _draw_scene(self):
        # Preserve view
        elev = getattr(self.ax, 'elev', None)
        azim = getattr(self.ax, 'azim', None)
        self._init_axes()
        for p in self.patches:
            self._draw_patch(p)
        # Restore view if available
        try:
            if elev is not None and azim is not None:
                self.ax.view_init(elev=elev, azim=azim)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _rotation_matrix(self, rx_deg: float, ry_deg: float, rz_deg: float):
        """Row-vector transform: world = local @ R.
        Extrinsic rotations about global X, then Y, then Z."""
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def _box_faces(self, center: np.ndarray, W: float, L: float, H: float, R: np.ndarray, omit_top: bool = False, bottom_only: bool = False, sides_only: bool = False) -> List[List[List[float]]]:
        # Local half-dimensions
        hx, hy, hz = W/2.0, L/2.0, H/2.0
        # 8 corners in local coordinates (x, y, z)
        corners = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [ -hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ -hx,  hy,  hz],
        ])
        # Rotate and translate
        rotated = (corners @ R) + center
        # Faces: bottom, top, sides (each as list of 4 points)
        idx = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4], [3,2,6,7], [0,3,7,4], [1,2,6,5]
        ]
        if sides_only:
            idx = [idx[2], idx[3], idx[4], idx[5]]  # four side faces only
        elif bottom_only:
            idx = [idx[0]]
        elif omit_top:
            idx = [idx[0]] + idx[2:]  # drop the top face [4,5,6,7]
        faces = [[rotated[i].tolist() for i in face] for face in idx]
        return faces

    def _box_edges(self, center: np.ndarray, W: float, L: float, H: float, R: np.ndarray, which: str = 'top') -> List[List[List[float]]]:
        """Return list of 3D line segments for the rectangle edges of 'top' or 'bottom' face."""
        hx, hy, hz = W/2.0, L/2.0, H/2.0
        corners = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [ -hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ -hx,  hy,  hz],
        ])
        rotated = (corners @ R) + center
        if which == 'top':
            ids = [4,5,6,7]
        else:
            ids = [0,1,2,3]
        ring = [rotated[i] for i in ids]
        # edges as consecutive pairs and last to first
        edges = []
        for i in range(4):
            a = ring[i]
            b = ring[(i+1)%4]
            edges.append([a.tolist(), b.tolist()])
        return edges

    def _substrate_top_ring(self, center: np.ndarray, sub_W: float, sub_L: float, h: float, R: np.ndarray,
                             inner_W: float, inner_L: float, ring_gap: float) -> List[List[List[float]]]:
        """Return a set of quads that cover the substrate top face but leave a rectangular hole
        around the patch footprint (inner_W x inner_L) with a small margin ring_gap.
        This makes the substrate visible from the top while never covering the patch area."""
        hx_o, hy_o = sub_W/2.0, sub_L/2.0
        hx_i, hy_i = inner_W/2.0 + ring_gap, inner_L/2.0 + ring_gap
        z = h/2.0

        # Define four rectangles (left, right, front, back) in substrate-local coordinates
        rects = [
            [-hx_o, -hy_o, -hx_i,  hy_o],  # left   (x0, y0, x1, y1)
            [ hx_i, -hy_o,  hx_o,  hy_o],  # right
            [-hx_i, -hy_o,  hx_i, -hy_i],  # front (negative y)
            [-hx_i,  hy_i,  hx_i,  hy_o],  # back  (positive y)
        ]
        faces = []
        for x0, y0, x1, y1 in rects:
            local = np.array([
                [x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]
            ])
            world = (local @ R) + center
            faces.append(world.tolist())
        return faces

    def _draw_patch(self, inst: PatchInstance):
        # Determine patch dimensions from params
        if inst.params.patch_length_m and inst.params.patch_width_m:
            L_m = inst.params.patch_length_m
            W_m = inst.params.patch_width_m
        else:
            L_m, W_m, _ = design_patch_for_frequency(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
        # Visual thicknesses
        t_patch = max(35e-6, 0.5e-4)  # ~35um copper default
        t_ground = 35e-6
        h = inst.params.h_m
        cx, cy, cz = inst.center_x_m, inst.center_y_m, inst.center_z_m
        C = np.array([cx, cy, cz])
        R = self._rotation_matrix(inst.rot_x_deg, inst.rot_y_deg, inst.rot_z_deg)

        # Substrate size (patch + margin)
        margin = 0.35 * max(L_m, W_m)
        sub_L = L_m + 2*margin
        sub_W = W_m + 2*margin

        # Substrate (semi-transparent green); leave a visual gap below patch so patch is never hidden
        # Use a gap proportional to patch size to avoid painter's algorithm artifacts in mplot3d
        visual_gap = max(8e-4, 0.01 * max(L_m, W_m))  # at least 0.8 mm or 1% of patch size
        sub_center = C + (np.array([0,0,-(t_patch/2 + visual_gap + h/2)]) @ R)  # shift along local -Z then rotate
        sub_faces = self._box_faces(sub_center, sub_W, sub_L, h, R, sides_only=True)
        # Add a top "ring" around the patch so the substrate is visible without covering the patch area
        ring_faces = self._substrate_top_ring(sub_center, sub_W, sub_L, h, R, inner_W=W_m, inner_L=L_m, ring_gap=visual_gap)
        sub_faces.extend(ring_faces)
        substrate = Poly3DCollection(sub_faces, alpha=0.25, facecolor='#3ba56d', edgecolor='#2d7d52', linewidth=0.8)
        try:
            substrate.set_zsort('min'); substrate.set_zorder(10)
        except Exception:
            pass
        self.ax.add_collection3d(substrate)

        # Ground plane (gray), bottom of substrate
        gnd_center = C + (np.array([0,0,-(t_patch/2 + visual_gap + h + t_ground/2)]) @ R)
        # Draw only the top edges of the ground plane as wireframe to avoid occluding the patch
        gnd_edges = self._box_edges(gnd_center, sub_W, sub_L, t_ground, R, which='top')
        ground = Line3DCollection(gnd_edges, colors='#6b6f73', linewidths=1.0, alpha=0.9)
        try:
            ground.set_zorder(5)
        except Exception:
            pass
        self.ax.add_collection3d(ground)

        # Patch box (gold), centered at C, thickness symmetric about z=0 (draw last for visibility)
        patch_faces = self._box_faces(C, W_m, L_m, t_patch, R)
        patch = Poly3DCollection(patch_faces, alpha=1.0, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=1.2)
        try:
            patch.set_zsort('max'); patch.set_zorder(100)
        except Exception:
            pass
        self.ax.add_collection3d(patch)

        # Optional: short feed stub for orientation (rotated with patch)
        try:
            feed_w = calculate_microstrip_width(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
            fw = feed_w
            length = 3*fw
            # Local center of the stub relative to patch center C before rotation
            if inst.feed_direction == FeedDirection.NEG_X:
                local_center = np.array([-(W_m/2 + length/2), 0.0, 0.0])
                dims = (length, fw, t_patch)
            elif inst.feed_direction == FeedDirection.POS_X:
                local_center = np.array([(W_m/2 + length/2), 0.0, 0.0])
                dims = (length, fw, t_patch)
            elif inst.feed_direction == FeedDirection.NEG_Y:
                local_center = np.array([0.0, -(L_m/2 + length/2), 0.0])
                dims = (fw, length, t_patch)
            else:  # POS_Y
                local_center = np.array([0.0, (L_m/2 + length/2), 0.0])
                dims = (fw, length, t_patch)
            feed_center = C + (local_center @ R)
            faces = self._box_faces(feed_center, dims[0], dims[1], dims[2], R)
            feed = Poly3DCollection(faces, alpha=0.98, facecolor='#ff6f3d', edgecolor='#a74323', linewidth=0.7)
            try:
                feed.set_zorder(110)
            except Exception:
                pass
            self.ax.add_collection3d(feed)
        except Exception:
            pass

        # Center marker
        self.ax.plot([cx], [cy], [cz], marker='o', color='#ff4081', markersize=4)

    # ---------- Events ----------
    def _on_scroll(self, event):
        try:
            factor = 0.9 if event.button == 'up' else 1.1
            def scale(lims):
                c = 0.5*(lims[0]+lims[1]); r = 0.5*(lims[1]-lims[0])*factor; return (c-r, c+r)
            self.ax.set_xlim(scale(self.ax.get_xlim()))
            self.ax.set_ylim(scale(self.ax.get_ylim()))
            self.ax.set_zlim(scale(self.ax.get_zlim()))
            self.canvas.draw_idle()
        except Exception:
            pass

    # ---------- Actions ----------
    def _on_add_patch(self):
        try:
            # Create params from current fields
            metal_enum = Metal(self.var_metal.get())
            p = PatchAntennaParams(
                frequency_hz=float(self.var_freq.get())*1e9,
                eps_r=float(self.var_eps.get()),
                h_m=float(self.var_h.get())*1e-3,
                loss_tangent=float(self.var_loss.get()),
                metal=metal_defaults[metal_enum],
                patch_length_m=None,
                patch_width_m=None,
            )
            name = f"Patch {len(self.patches)+1}"
            inst = PatchInstance(name=name, params=p, center_x_m=0.0, center_y_m=0.0, center_z_m=0.0)
            self.patches.append(inst)
            self._refresh_selector(select_index=len(self.patches)-1)
            self._draw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add patch: {e}")

    def _refresh_selector(self, select_index: Optional[int] = None):
        names = [p.name for p in self.patches]
        self.sel_combo['values'] = names
        if names:
            idx = select_index if select_index is not None else 0
            idx = max(0, min(idx, len(names)-1))
            self.sel_combo.current(idx)
            self._current_index = idx
            self._load_current_into_fields()
        else:
            self.sel_combo.set('')
            self._current_index = None

    def _on_select_patch(self, event=None):
        name = self.sel_var.get()
        for i, p in enumerate(self.patches):
            if p.name == name:
                self._current_index = i
                self._load_current_into_fields()
                break

    def _load_current_into_fields(self):
        if self._current_index is None:
            return
        p = self.patches[self._current_index]
        self.var_cx.set(p.center_x_m)
        self.var_cy.set(p.center_y_m)
        self.var_freq.set(p.params.frequency_hz/1e9)
        self.var_eps.set(p.params.eps_r)
        self.var_h.set(p.params.h_m*1e3)
        self.var_loss.set(p.params.loss_tangent)
        self.var_cz.set(p.center_z_m)
        self.var_rx.set(p.rot_x_deg)
        self.var_ry.set(p.rot_y_deg)
        self.var_rz.set(p.rot_z_deg)
        # best-effort mapping back to enum name
        try:
            for m in Metal:
                if metal_defaults[m] == p.params.metal:
                    self.var_metal.set(m.value)
                    break
        except Exception:
            pass

    def _on_apply_changes(self):
        if self._current_index is None:
            return
        try:
            metal_enum = Metal(self.var_metal.get())
            p = self.patches[self._current_index]
            p.center_x_m = float(self.var_cx.get())
            p.center_y_m = float(self.var_cy.get())
            p.center_z_m = float(self.var_cz.get())
            p.rot_x_deg = float(self.var_rx.get())
            p.rot_y_deg = float(self.var_ry.get())
            p.rot_z_deg = float(self.var_rz.get())
            p.params = PatchAntennaParams(
                frequency_hz=float(self.var_freq.get())*1e9,
                eps_r=float(self.var_eps.get()),
                h_m=float(self.var_h.get())*1e-3,
                loss_tangent=float(self.var_loss.get()),
                metal=metal_defaults[metal_enum],
                patch_length_m=None,
                patch_width_m=None,
            )
            self._draw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

    # ---- Single-field apply helpers ----
    def _rebuild_params(self, p: PatchInstance, **overrides):
        base = p.params
        return PatchAntennaParams(
            frequency_hz=overrides.get('frequency_hz', base.frequency_hz),
            eps_r=overrides.get('eps_r', base.eps_r),
            h_m=overrides.get('h_m', base.h_m),
            loss_tangent=overrides.get('loss_tangent', base.loss_tangent),
            metal=overrides.get('metal', base.metal),
            patch_length_m=None,
            patch_width_m=None,
        )

    def _apply_single_field(self, field: str):
        if self._current_index is None:
            return
        try:
            p = self.patches[self._current_index]
            if field == 'cx':
                p.center_x_m = float(self.var_cx.get())
            elif field == 'cy':
                p.center_y_m = float(self.var_cy.get())
            elif field == 'cz':
                p.center_z_m = float(self.var_cz.get())
            elif field == 'rx':
                p.rot_x_deg = float(self.var_rx.get())
            elif field == 'ry':
                p.rot_y_deg = float(self.var_ry.get())
            elif field == 'rz':
                p.rot_z_deg = float(self.var_rz.get())
            elif field == 'freq':
                p.params = self._rebuild_params(p, frequency_hz=float(self.var_freq.get())*1e9)
            elif field == 'eps':
                p.params = self._rebuild_params(p, eps_r=float(self.var_eps.get()))
            elif field == 'h':
                p.params = self._rebuild_params(p, h_m=float(self.var_h.get())*1e-3)
            elif field == 'loss':
                p.params = self._rebuild_params(p, loss_tangent=float(self.var_loss.get()))
            elif field == 'metal':
                metal_enum = Metal(self.var_metal.get())
                p.params = self._rebuild_params(p, metal=metal_defaults[metal_enum])
            self._draw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set {field}: {e}")

    def _on_remove_selected(self):
        if self._current_index is None:
            return
        try:
            self.patches.pop(self._current_index)
            self._refresh_selector(select_index=min(self._current_index, max(0, len(self.patches)-1)))
            self._draw_scene()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove: {e}")


class MultiPatchDesigner(tk.Toplevel):
    """Designer for arranging multiple microstrip patch antennas.

    - Clean 3D workspace (≈3 m span) showing only XYZ axes.
    - Add patches, select by name, and edit per-instance properties.
    - Uses same design rules as the existing microstrip 3D solver (size from freq/er/h).
    """

    def __init__(self, master: Optional[tk.Misc] = None):
        super().__init__(master)
        self.title("Multi Patch Designer")
        try:
            self.state('zoomed')
        except Exception:
            self.geometry("1200x800")
        # Host the embeddable panel inside this Toplevel
        self.panel = MultiPatchPanel(self)
        self.panel.pack(fill='both', expand=True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
