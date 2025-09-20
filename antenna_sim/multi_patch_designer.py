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
        self._change_cb = None  # external listener for scene updates
        # Track UI widgets for enable/disable and overlay
        self._control_widgets: List[tk.Widget] = []
        self._original_states: dict = {}
        self._controls_container: Optional[ttk.Frame] = None
        self._disabled_overlay: Optional[tk.Frame] = None
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
        # Reset control tracking and remember container
        self._control_widgets = []
        self._original_states = {}
        self._controls_container = parent

        # Notebook with two tabs: Geometry Controls and Simulation Controls
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        tab_controls = ttk.Frame(notebook)
        tab_sim = ttk.Frame(notebook)
        notebook.add(tab_controls, text='Geometry Controls')
        notebook.add(tab_sim, text='Simulation Controls')

        # ---- Geometry tab ----
        ttk.Label(tab_controls, text="Geometry Controls", font=("Segoe UI", 12, "bold")).pack(fill='x', padx=10, pady=(12, 6))
        btn_add = ttk.Button(tab_controls, text="Add Patch Antenna", command=self._on_add_patch)
        btn_add.pack(fill='x', padx=10, pady=(0, 10))
        self._control_widgets.append(btn_add)

        sel = ttk.Frame(tab_controls)
        sel.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Label(sel, text="Select Patch:").grid(row=0, column=0, sticky='w')
        self.sel_var = tk.StringVar()
        self.sel_combo = ttk.Combobox(sel, textvariable=self.sel_var, state='readonly')
        self.sel_combo.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        sel.columnconfigure(1, weight=1)
        self.sel_combo.bind('<<ComboboxSelected>>', self._on_select_patch)
        self._control_widgets.append(self.sel_combo)

        props = ttk.LabelFrame(tab_controls, text="Selected Patch Properties")
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
                self._control_widgets.append(e)
                self._original_states[e] = 'normal'
            else:
                e = ttk.Combobox(props, textvariable=entry_var, state='readonly', values=[m.value for m in Metal])
                e.grid(row=r, column=1, sticky='ew')
                e.bind('<<ComboboxSelected>>', lambda ev, f=field_key: self._apply_single_field(f))
                self._control_widgets.append(e)
                self._original_states[e] = 'readonly'
            btn = ttk.Button(props, text='Set', width=6, command=lambda f=field_key: self._apply_single_field(f))
            btn.grid(row=r, column=2, padx=(6,0))
            self._control_widgets.append(btn)
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

        # Display controls
        disp = ttk.LabelFrame(tab_controls, text="Display")
        disp.pack(fill='x', padx=10, pady=(0, 10))
        try:
            disp.columnconfigure(1, weight=1)
        except Exception:
            pass
        self.var_show_substrate = tk.BooleanVar(value=True)
        self.var_show_ground = tk.BooleanVar(value=True)
        self.var_ground_style = tk.StringVar(value='edges')  # 'edges' | 'ring'
        cb1 = ttk.Checkbutton(disp, text="Show substrate", variable=self.var_show_substrate, command=self._draw_scene)
        cb1.grid(row=0, column=0, sticky='w')
        cb2 = ttk.Checkbutton(disp, text="Show ground", variable=self.var_show_ground, command=self._draw_scene)
        cb2.grid(row=0, column=1, sticky='w', padx=(10,0))
        self._control_widgets.extend([cb1, cb2])
        ttk.Label(disp, text="Ground style").grid(row=1, column=0, sticky='w')
        grd = ttk.Combobox(disp, textvariable=self.var_ground_style, state='readonly', values=['edges','ring'])
        grd.grid(row=1, column=1, sticky='ew')
        grd.bind('<<ComboboxSelected>>', lambda ev: self._draw_scene())
        self._control_widgets.append(grd)
        self._original_states[grd] = 'readonly'

        # ---- Simulation tab ----
        ttk.Label(tab_sim, text="Simulation Controls", font=("Segoe UI", 12, "bold")).pack(fill='x', padx=10, pady=(10, 6))
        ff = ttk.LabelFrame(tab_sim, text="Far-Field Sampling")
        ff.pack(fill='x', padx=10, pady=(0, 10))
        try:
            ff.columnconfigure(1, weight=1)
        except Exception:
            pass
        self.var_theta_step = tk.DoubleVar(value=2.0)
        self.var_phi_step = tk.DoubleVar(value=5.0)
        ttk.Label(ff, text="Theta step (°)").grid(row=0, column=0, sticky='w')
        e_th = ttk.Entry(ff, textvariable=self.var_theta_step, width=10)
        e_th.grid(row=0, column=1, sticky='ew')
        ttk.Label(ff, text="Phi step (°)").grid(row=1, column=0, sticky='w')
        e_ph = ttk.Entry(ff, textvariable=self.var_phi_step, width=10)
        e_ph.grid(row=1, column=1, sticky='ew')
        self._control_widgets.extend([e_th, e_ph])
        self._original_states[e_th] = 'normal'
        self._original_states[e_ph] = 'normal'
        ttk.Label(ff, text="Tip: smaller steps increase simulation time").grid(row=2, column=0, columnspan=2, sticky='w', pady=(4,0))
        # Enter-to-apply for sampling
        try:
            e_th.bind('<Return>', lambda ev: self._apply_sim_params())
            e_ph.bind('<Return>', lambda ev: self._apply_sim_params())
        except Exception:
            pass

        # Mesh quality control (1..5)
        mesh_frame = ttk.LabelFrame(tab_sim, text="Mesh Quality")
        mesh_frame.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Label(mesh_frame, text="Resolution").grid(row=0, column=0, sticky='w')
        self.var_mesh_quality = tk.IntVar(value=3)
        self.mesh_combo = ttk.Combobox(mesh_frame, state='readonly', width=18,
                                       values=["1 - Coarse","2 - Medium-","3 - Medium","4 - Medium+","5 - Fine"])
        self.mesh_combo.grid(row=0, column=1, sticky='ew')
        # Set to default index 2 (value 3)
        try:
            self.mesh_combo.current(2)
        except Exception:
            pass
        try:
            self.mesh_combo.bind('<<ComboboxSelected>>', lambda ev: self._apply_sim_params())
        except Exception:
            pass
        self._control_widgets.append(self.mesh_combo)
        self._original_states[self.mesh_combo] = 'readonly'
        # Remember last-applied sampling so we can report changes
        try:
            self._prev_theta_step = float(self.var_theta_step.get())
            self._prev_phi_step = float(self.var_phi_step.get())
            self._prev_mesh_quality = 3
        except Exception:
            self._prev_theta_step = None
            self._prev_phi_step = None
            self._prev_mesh_quality = None
        # Previous NF2FF center and sim box state
        self._prev_nf_center = 'Origin'
        self._prev_simbox_mode = 'Auto'
        self._prev_margin_x = 80.0
        self._prev_margin_y = 80.0
        self._prev_margin_z = 160.0
        self._prev_box_x = 400.0
        self._prev_box_y = 400.0
        self._prev_box_z = 160.0

        # NF2FF center control
        nf_frame = ttk.LabelFrame(tab_sim, text="NF2FF Center")
        nf_frame.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Label(nf_frame, text="Center").grid(row=0, column=0, sticky='w')
        self.var_nf2ff_center = tk.StringVar(value='Origin')
        self.nf_center_combo = ttk.Combobox(nf_frame, textvariable=self.var_nf2ff_center, state='readonly',
                                            values=['Origin','Centroid'])
        self.nf_center_combo.grid(row=0, column=1, sticky='ew')
        try:
            self.nf_center_combo.bind('<<ComboboxSelected>>', lambda ev: self._apply_sim_params())
        except Exception:
            pass
        self._control_widgets.append(self.nf_center_combo)
        self._original_states[self.nf_center_combo] = 'readonly'

        # Simulation Box controls
        box_frame = ttk.LabelFrame(tab_sim, text="Simulation Box")
        box_frame.pack(fill='x', padx=10, pady=(0, 10))
        ttk.Label(box_frame, text="Mode").grid(row=0, column=0, sticky='w')
        self.var_simbox_mode = tk.StringVar(value='Auto')
        self.simbox_mode_combo = ttk.Combobox(box_frame, textvariable=self.var_simbox_mode, state='readonly',
                                              values=['Auto','Manual'])
        self.simbox_mode_combo.grid(row=0, column=1, sticky='ew')
        try:
            self.simbox_mode_combo.bind('<<ComboboxSelected>>', lambda ev: self._apply_sim_params())
        except Exception:
            pass
        self._control_widgets.append(self.simbox_mode_combo)
        self._original_states[self.simbox_mode_combo] = 'readonly'

        # Auto margins (mm)
        ttk.Label(box_frame, text="Auto margins (mm): X").grid(row=1, column=0, sticky='w')
        self.var_margin_x = tk.DoubleVar(value=80.0)
        e_mx = ttk.Entry(box_frame, textvariable=self.var_margin_x, width=8)
        e_mx.grid(row=1, column=1, sticky='w')
        ttk.Label(box_frame, text="Y").grid(row=1, column=2, sticky='w')
        self.var_margin_y = tk.DoubleVar(value=80.0)
        e_my = ttk.Entry(box_frame, textvariable=self.var_margin_y, width=8)
        e_my.grid(row=1, column=3, sticky='w')
        ttk.Label(box_frame, text="Z").grid(row=1, column=4, sticky='w')
        self.var_margin_z = tk.DoubleVar(value=160.0)
        e_mz = ttk.Entry(box_frame, textvariable=self.var_margin_z, width=8)
        e_mz.grid(row=1, column=5, sticky='w')
        for w in (e_mx, e_my, e_mz):
            try:
                w.bind('<Return>', lambda ev: self._apply_sim_params())
            except Exception:
                pass
            self._control_widgets.append(w)
            self._original_states[w] = 'normal'

        # Manual box size (mm)
        ttk.Label(box_frame, text="Manual size (mm): X").grid(row=2, column=0, sticky='w', pady=(4,0))
        self.var_box_x = tk.DoubleVar(value=400.0)
        e_bx = ttk.Entry(box_frame, textvariable=self.var_box_x, width=8)
        e_bx.grid(row=2, column=1, sticky='w', pady=(4,0))
        ttk.Label(box_frame, text="Y").grid(row=2, column=2, sticky='w', pady=(4,0))
        self.var_box_y = tk.DoubleVar(value=400.0)
        e_by = ttk.Entry(box_frame, textvariable=self.var_box_y, width=8)
        e_by.grid(row=2, column=3, sticky='w', pady=(4,0))
        ttk.Label(box_frame, text="Z").grid(row=2, column=4, sticky='w', pady=(4,0))
        self.var_box_z = tk.DoubleVar(value=160.0)
        e_bz = ttk.Entry(box_frame, textvariable=self.var_box_z, width=8)
        e_bz.grid(row=2, column=5, sticky='w', pady=(4,0))
        for w in (e_bx, e_by, e_bz):
            try:
                w.bind('<Return>', lambda ev: self._apply_sim_params())
            except Exception:
                pass
            self._control_widgets.append(w)
            self._original_states[w] = 'normal'

        # Apply Simulation Parameters button and local status message
        btn_apply_sim = ttk.Button(tab_sim, text="Apply Simulation Parameters", command=self._apply_sim_params)
        btn_apply_sim.pack(fill='x', padx=10, pady=(0, 6))
        self._control_widgets.append(btn_apply_sim)
        try:
            self.sim_status_msg = tk.StringVar(value="")
            self.sim_status_label = ttk.Label(tab_sim, textvariable=self.sim_status_msg)
            self.sim_status_label.pack(fill='x', padx=10, pady=(0, 6))
        except Exception:
            pass

        # Port Diagnostics UI for MultiPatchPanel removed to avoid duplication.

        # View controls (Multi Patch tab)
        btn_fit = ttk.Button(tab_controls, text="Fit View", command=self._fit_view)
        btn_fit.pack(fill='x', padx=10, pady=(0, 8))
        self._control_widgets.append(btn_fit)

        # Removed "Apply Changes" button (Enter and Set buttons already apply changes)
        btn_remove = ttk.Button(tab_controls, text="Remove Selected", command=self._on_remove_selected)
        btn_remove.pack(fill='x', padx=10, pady=(0, 6))
        self._control_widgets.extend([btn_remove])
        # Status line for user feedback after applying changes
        try:
            self.status_msg = tk.StringVar(value="")
            self.status_label = ttk.Label(tab_controls, textvariable=self.status_msg)
            self.status_label.pack(fill='x', padx=10, pady=(0, 6))
        except Exception:
            pass
        ttk.Label(tab_controls, text="Tip: Rotate with mouse, scroll to zoom.").pack(fill='x', padx=10, pady=(8, 10))

    # ---- Port log helpers (used by GUI thread-safe via root.after) ----
    def clear_port_log(self):
        try:
            if hasattr(self, 'port_log') and self.port_log is not None:
                self.port_log.delete('1.0', 'end')
        except Exception:
            pass

    def append_port_log(self, text: str):
        try:
            if hasattr(self, 'port_log') and self.port_log is not None:
                self.port_log.insert('end', str(text) + '\n')
                self.port_log.see('end')
        except Exception:
            pass

    # ---- Enable/disable controls with overlay ----
    def lock_controls(self):
        try:
            # Disable all interactive widgets
            for w in self._control_widgets:
                try:
                    if isinstance(w, ttk.Combobox):
                        w.configure(state='disabled')
                    else:
                        w.configure(state='disabled')
                except Exception:
                    pass
            # Overlay
            if self._controls_container is not None and self._disabled_overlay is None:
                ov = tk.Frame(self._controls_container, bg='#2b2b2b')
                ov.place(relx=0, rely=0, relwidth=1, relheight=1)
                try:
                    ov.lift()  # ensure overlay is above all children
                except Exception:
                    pass
                # Swallow mouse/keyboard events to block interaction
                try:
                    for ev in ('<Button-1>','<Button-2>','<Button-3>','<MouseWheel>','<Key>'):
                        ov.bind(ev, lambda e: 'break')
                except Exception:
                    pass
                msg = tk.Label(ov, text="Locked while simulation is running...", fg='white', bg='#2b2b2b')
                msg.place(relx=0.5, rely=0.5, anchor='center')
                self._disabled_overlay = ov
        except Exception:
            pass

    def unlock_controls(self):
        try:
            # Restore original states
            for w in self._control_widgets:
                try:
                    orig = self._original_states.get(w, 'normal')
                    # Combobox must be 'readonly' to prevent typing when enabled
                    if isinstance(w, ttk.Combobox):
                        w.configure(state=orig if orig in ('readonly', 'normal') else 'readonly')
                    else:
                        w.configure(state=orig if orig in ('normal',) else 'normal')
                except Exception:
                    pass
            # Remove overlay
            if self._disabled_overlay is not None:
                try:
                    self._disabled_overlay.destroy()
                except Exception:
                    pass
                self._disabled_overlay = None
        except Exception:
            pass

    # ---------- Axes and rendering ----------
    def _init_axes(self, limits: Optional[tuple] = None):
        ax = self.ax
        ax.clear()
        default_lim = 0.15  # meters -> ~0.3 m span (smaller default view)
        if limits is None:
            self._lim = default_lim
            xlim = (-self._lim, self._lim)
            ylim = (-self._lim, self._lim)
            zlim = (-self._lim, self._lim)
        else:
            xlim, ylim, zlim = limits
            # choose an axis line length that covers the current view nicely
            span = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
            self._lim = max(default_lim, 0.5*span)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
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
        # Preserve view (angles and zoom/pan)
        elev = getattr(self.ax, 'elev', None)
        azim = getattr(self.ax, 'azim', None)
        try:
            xlim = self.ax.get_xlim3d(); ylim = self.ax.get_ylim3d(); zlim = self.ax.get_zlim3d()
            limits = (xlim, ylim, zlim)
        except Exception:
            limits = None
        self._init_axes(limits)
        for p in self.patches:
            self._draw_patch(p)
        # Restore view if available
        try:
            if elev is not None and azim is not None:
                self.ax.view_init(elev=elev, azim=azim)
        except Exception:
            pass
        self.canvas.draw_idle()
        # Notify external listeners that the scene changed
        try:
            if self._change_cb is not None:
                self._change_cb(self.patches)
        except Exception:
            pass

    def set_change_callback(self, cb):
        """Register a callback called after the scene redraws.
        Signature: cb(patches: List[PatchInstance]) -> None"""
        self._change_cb = cb

    def _rotation_matrix(self, rx_deg: float, ry_deg: float, rz_deg: float):
        """Row-vector transform: world = local @ R.
        Extrinsic rotations about global X, then Y, then Z.
        For CSXCAD's extrinsic order and our row-vector convention, use R = (Rz @ Ry @ Rx).T
        so the UI matches the solver.
        """
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return (Rz @ Ry @ Rx).T

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

    def _box_corners(self, center: np.ndarray, W: float, L: float, H: float, R: np.ndarray) -> np.ndarray:
        """Return (8,3) array of oriented box corners in world coordinates."""
        hx, hy, hz = W/2.0, L/2.0, H/2.0
        corners = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [ -hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ -hx,  hy,  hz],
        ])
        return (corners @ R) + center

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

        # Substrate (semi-transparent green)
        if getattr(self, 'var_show_substrate', None) is None or self.var_show_substrate.get():
            # Leave a visual gap below patch so patch is never hidden
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
        if getattr(self, 'var_show_ground', None) is None or self.var_show_ground.get():
            gnd_center = C + (np.array([0,0,-(t_patch/2 + max(8e-4, 0.01 * max(L_m, W_m)) + h + t_ground/2)]) @ R)
            ground_style = getattr(self, 'var_ground_style', None).get() if getattr(self, 'var_ground_style', None) else 'edges'
            if ground_style == 'edges':
                # Draw only the top edges to avoid occluding the patch
                gnd_edges = self._box_edges(gnd_center, sub_W, sub_L, t_ground, R, which='top')
                ground = Line3DCollection(gnd_edges, colors='#b87333', linewidths=1.6, alpha=0.95)
                try:
                    ground.set_zorder(5)
                except Exception:
                    pass
                self.ax.add_collection3d(ground)
            else:
                # Render a top ring on the ground as a face so it's visible but never occludes patch center
                g_ring = self._substrate_top_ring(gnd_center, sub_W, sub_L, t_ground, R, inner_W=W_m, inner_L=L_m, ring_gap=max(8e-4, 0.01*max(L_m, W_m)))
                ground = Poly3DCollection(g_ring, alpha=0.7, facecolor='#b87333', edgecolor='#a35f2d', linewidth=0.8)
                try:
                    ground.set_zorder(6)
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
            t_feed = t_patch
            # Local center of the stub relative to patch center C before rotation
            if inst.feed_direction == FeedDirection.NEG_X:
                local_center = np.array([-(W_m/2 + length/2), 0.0, 0.0])
                dims = (length, fw, t_feed)
            elif inst.feed_direction == FeedDirection.POS_X:
                local_center = np.array([(W_m/2 + length/2), 0.0, 0.0])
                dims = (length, fw, t_feed)
            elif inst.feed_direction == FeedDirection.NEG_Y:
                local_center = np.array([0.0, -(L_m/2 + length/2), 0.0])
                dims = (fw, length, t_feed)
            else:  # POS_Y
                local_center = np.array([0.0, (L_m/2 + length/2), 0.0])
                dims = (fw, length, t_feed)
            feed_center = C + (local_center @ R)
            faces = self._box_faces(feed_center, dims[0], dims[1], dims[2], R)
            feed = Poly3DCollection(faces, alpha=0.98, facecolor='#ff6f3d', edgecolor='#a74323', linewidth=0.7)
            try:
                feed.set_zorder(110)
            except Exception:
                pass
            self.ax.add_collection3d(feed)

            # No overlay; rely on visibility plus motion
        except Exception:
            pass

        # Center marker
        self.ax.plot([cx], [cy], [cz], marker='o', color='#ff4081', markersize=4)

    def _fit_view(self):
        """Adjust axes limits to enclose all patches with a small margin, preserving camera orientation."""
        try:
            if not self.patches:
                self._init_axes()
                self.canvas.draw_idle()
                return
            xs, ys, zs = [], [], []
            for inst in self.patches:
                # Geometry parameters
                if inst.params.patch_length_m and inst.params.patch_width_m:
                    L_m = inst.params.patch_length_m; W_m = inst.params.patch_width_m
                else:
                    L_m, W_m, _ = design_patch_for_frequency(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
                t_patch = max(35e-6, 0.5e-4)
                t_ground = 35e-6
                h = inst.params.h_m
                C = np.array([inst.center_x_m, inst.center_y_m, inst.center_z_m])
                R = self._rotation_matrix(inst.rot_x_deg, inst.rot_y_deg, inst.rot_z_deg)
                margin = 0.35 * max(L_m, W_m)
                sub_L = L_m + 2*margin
                sub_W = W_m + 2*margin
                visual_gap = max(8e-4, 0.01 * max(L_m, W_m))
                # Include patch, substrate, and ground boxes
                sub_center = C + (np.array([0,0,-(t_patch/2 + visual_gap + h/2)]) @ R)
                gnd_center = C + (np.array([0,0,-(t_patch/2 + visual_gap + h + t_ground/2)]) @ R)
                for (ctr, W, L, H) in [
                    (C, W_m, L_m, t_patch),
                    (sub_center, sub_W, sub_L, h),
                    (gnd_center, sub_W, sub_L, t_ground),
                ]:
                    corners = self._box_corners(ctr, W, L, H, R)
                    xs.extend(corners[:,0]); ys.extend(corners[:,1]); zs.extend(corners[:,2])
            # Compute limits with padding
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)
            pad = 0.08 * max(x_max-x_min, y_max-y_min, z_max-z_min)
            xlim = (x_min - pad, x_max + pad)
            ylim = (y_min - pad, y_max + pad)
            zlim = (z_min - pad, z_max + pad)
            # Preserve camera orientation
            elev = getattr(self.ax, 'elev', None)
            azim = getattr(self.ax, 'azim', None)
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim); self.ax.set_zlim(zlim)
            try:
                if elev is not None and azim is not None:
                    self.ax.view_init(elev=elev, azim=azim)
            except Exception:
                pass
            self.canvas.draw_idle()
            # Notify listeners so external views (e.g., PyVista) can also reset their camera
            try:
                # Mark last action for host to detect a Fit View request
                self._last_action = 'fit'
                if self._change_cb is not None:
                    self._change_cb(self.patches)
            except Exception:
                pass
        except Exception:
            pass

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
            # Track what changes are being applied
            changes = []
            metal_enum = Metal(self.var_metal.get())
            p = self.patches[self._current_index]
            # Old values (for diff)
            old = dict(cx=p.center_x_m, cy=p.center_y_m, cz=p.center_z_m,
                       rx=p.rot_x_deg, ry=p.rot_y_deg, rz=p.rot_z_deg,
                       freq=p.params.frequency_hz/1e9, eps=p.params.eps_r,
                       h=p.params.h_m*1e3, loss=p.params.loss_tangent,
                       metal=str(self.var_metal.get()))
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
            # Compare and collect patch diffs (only if user cares)
            new = dict(cx=p.center_x_m, cy=p.center_y_m, cz=p.center_z_m,
                       rx=p.rot_x_deg, ry=p.rot_y_deg, rz=p.rot_z_deg,
                       freq=p.params.frequency_hz/1e9, eps=p.params.eps_r,
                       h=p.params.h_m*1e3, loss=p.params.loss_tangent,
                       metal=str(self.var_metal.get()))
            key_labels = {
                'cx': 'Cx (m)', 'cy': 'Cy (m)', 'cz': 'Cz (m)',
                'rx': 'Rx (°)', 'ry': 'Ry (°)', 'rz': 'Rz (°)',
                'freq': 'Frequency (GHz)', 'eps': 'εr', 'h': 'h (mm)', 'loss': 'loss tan', 'metal': 'metal'
            }
            for k in new:
                try:
                    if abs(float(new[k]) - float(old[k])) > 1e-12:
                        if isinstance(new[k], float):
                            changes.append(f"{key_labels[k]} = {new[k]:.4g}")
                        else:
                            changes.append(f"{key_labels[k]} = {new[k]}")
                except Exception:
                    if new[k] != old[k]:
                        changes.append(f"{key_labels[k]} = {new[k]}")
            # Theta/Phi sampling diffs
            try:
                th_now = float(self.var_theta_step.get())
                ph_now = float(self.var_phi_step.get())
                if self._prev_theta_step is None or abs(th_now - self._prev_theta_step) > 1e-12:
                    changes.append(f"theta step = {th_now:g}°")
                if self._prev_phi_step is None or abs(ph_now - self._prev_phi_step) > 1e-12:
                    changes.append(f"phi step = {ph_now:g}°")
                self._prev_theta_step = th_now
                self._prev_phi_step = ph_now
            except Exception:
                pass
            # Mesh quality diff
            try:
                # Combobox text -> int (first char)
                sel = self.mesh_combo.get().strip()
                mq = int(sel.split('-',1)[0].strip()) if sel else 3
                if self._prev_mesh_quality is None or mq != self._prev_mesh_quality:
                    label_map = {1:"coarse",2:"medium-",3:"medium",4:"medium+",5:"fine"}
                    changes.append(f"mesh = {label_map.get(mq, 'medium')} ({mq}/5)")
                self._prev_mesh_quality = mq
            except Exception:
                pass
            # NF2FF center diff
            try:
                nf_now = self.nf_center_combo.get().strip() or 'Origin'
                if nf_now != self._prev_nf_center:
                    changes.append(f"NF2FF center = {nf_now}")
                self._prev_nf_center = nf_now
            except Exception:
                pass
            # Simulation box diffs
            try:
                mode_now = self.simbox_mode_combo.get().strip() or 'Auto'
                if mode_now != self._prev_simbox_mode:
                    changes.append(f"SimBox mode = {mode_now}")
                self._prev_simbox_mode = mode_now
                # Margins
                mx, my, mz = float(self.var_margin_x.get()), float(self.var_margin_y.get()), float(self.var_margin_z.get())
                if abs(mx - self._prev_margin_x) > 1e-9 or abs(my - self._prev_margin_y) > 1e-9 or abs(mz - self._prev_margin_z) > 1e-9:
                    changes.append(f"Auto margins = ({mx:g},{my:g},{mz:g}) mm")
                self._prev_margin_x, self._prev_margin_y, self._prev_margin_z = mx, my, mz
                # Manual sizes
                bx, by, bz = float(self.var_box_x.get()), float(self.var_box_y.get()), float(self.var_box_z.get())
                if abs(bx - self._prev_box_x) > 1e-9 or abs(by - self._prev_box_y) > 1e-9 or abs(bz - self._prev_box_z) > 1e-9:
                    changes.append(f"Manual box = ({bx:g},{by:g},{bz:g}) mm")
                self._prev_box_x, self._prev_box_y, self._prev_box_z = bx, by, bz
            except Exception:
                pass
            self._draw_scene()
            # Show status line
            try:
                if changes:
                    msg = "Applied: " + ", ".join(changes)
                    self.status_msg.set(msg)
                    try:
                        # Mirror into sim status as well so it's visible near that section
                        self.sim_status_msg.set(msg)
                    except Exception:
                        pass
                else:
                    self.status_msg.set("No changes detected.")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

    def _apply_sim_params(self):
        """Apply only simulation parameter changes (sampling, mesh) and show a status message.
        This is used for Enter-to-apply and dropdown change events."""
        try:
            changes = []
            # theta/phi
            try:
                th_now = float(self.var_theta_step.get())
                ph_now = float(self.var_phi_step.get())
                if self._prev_theta_step is None or abs(th_now - self._prev_theta_step) > 1e-12:
                    changes.append(f"theta step = {th_now:g}°")
                if self._prev_phi_step is None or abs(ph_now - self._prev_phi_step) > 1e-12:
                    changes.append(f"phi step = {ph_now:g}°")
                self._prev_theta_step = th_now
                self._prev_phi_step = ph_now
            except Exception:
                pass
            # mesh
            try:
                sel = self.mesh_combo.get().strip()
                mq = int(sel.split('-',1)[0].strip()) if sel else 3
                if self._prev_mesh_quality is None or mq != self._prev_mesh_quality:
                    label_map = {1:"coarse",2:"medium-",3:"medium",4:"medium+",5:"fine"}
                    changes.append(f"mesh = {label_map.get(mq, 'medium')} ({mq}/5)")
                self._prev_mesh_quality = mq
            except Exception:
                pass
            # NF2FF center
            try:
                nf_now = self.nf_center_combo.get().strip() or 'Origin'
                if nf_now != self._prev_nf_center:
                    changes.append(f"NF2FF center = {nf_now}")
                self._prev_nf_center = nf_now
            except Exception:
                pass
            # Sim box
            try:
                mode_now = self.simbox_mode_combo.get().strip() or 'Auto'
                if mode_now != self._prev_simbox_mode:
                    changes.append(f"SimBox mode = {mode_now}")
                self._prev_simbox_mode = mode_now
                mx, my, mz = float(self.var_margin_x.get()), float(self.var_margin_y.get()), float(self.var_margin_z.get())
                if abs(mx - self._prev_margin_x) > 1e-9 or abs(my - self._prev_margin_y) > 1e-9 or abs(mz - self._prev_margin_z) > 1e-9:
                    changes.append(f"Auto margins = ({mx:g},{my:g},{mz:g}) mm")
                self._prev_margin_x, self._prev_margin_y, self._prev_margin_z = mx, my, mz
                bx, by, bz = float(self.var_box_x.get()), float(self.var_box_y.get()), float(self.var_box_z.get())
                if abs(bx - self._prev_box_x) > 1e-9 or abs(by - self._prev_box_y) > 1e-9 or abs(bz - self._prev_box_z) > 1e-9:
                    changes.append(f"Manual box = ({bx:g},{by:g},{bz:g}) mm")
                self._prev_box_x, self._prev_box_y, self._prev_box_z = bx, by, bz
            except Exception:
                pass
            if changes:
                msg = "Applied: " + ", ".join(changes)
                try:
                    self.status_msg.set(msg)
                except Exception:
                    pass
                try:
                    self.sim_status_msg.set(msg)
                except Exception:
                    pass
        except Exception:
            pass

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
        
