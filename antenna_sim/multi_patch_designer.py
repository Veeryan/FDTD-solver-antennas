from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .models import PatchAntennaParams, Metal, metal_defaults
from .solver_fdtd_openems_microstrip import FeedDirection, calculate_microstrip_width
from .physics import design_patch_for_frequency


@dataclass
class PatchInstance:
    name: str
    params: PatchAntennaParams
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    feed_direction: FeedDirection = FeedDirection.NEG_X


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

        self.var_cx = tk.DoubleVar(value=0.0)
        self.var_cy = tk.DoubleVar(value=0.0)
        self.var_freq = tk.DoubleVar(value=2.45)
        self.var_eps = tk.DoubleVar(value=4.3)
        self.var_h = tk.DoubleVar(value=1.6)
        self.var_loss = tk.DoubleVar(value=0.02)
        self.var_metal = tk.StringVar(value=Metal.COPPER.value)

        r = 0
        ttk.Label(props, text="Center X (m)").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_cx, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Center Y (m)").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_cy, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Frequency (GHz)").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_freq, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Dielectric εr").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_eps, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Substrate h (mm)").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_h, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Loss tangent").grid(row=r, column=0, sticky='w'); ttk.Entry(props, textvariable=self.var_loss, width=10).grid(row=r, column=1, sticky='ew'); r+=1
        ttk.Label(props, text="Metal").grid(row=r, column=0, sticky='w')
        self.metal_combo = ttk.Combobox(props, textvariable=self.var_metal, state='readonly', values=[m.value for m in Metal])
        self.metal_combo.grid(row=r, column=1, sticky='ew'); r+=1

        ttk.Button(parent, text="Apply Changes", command=self._on_apply_changes).pack(fill='x', padx=10, pady=(0, 6))
        ttk.Button(parent, text="Remove Selected", command=self._on_remove_selected).pack(fill='x', padx=10, pady=(0, 6))
        ttk.Label(parent, text="Tip: Rotate with mouse, scroll to zoom.").pack(fill='x', padx=10, pady=(8, 10))

    # ---------- Axes and rendering ----------
    def _init_axes(self):
        ax = self.ax
        ax.clear()
        self._lim = 1.5  # meters -> ~3 m span
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
        self._init_axes()
        for p in self.patches:
            self._draw_patch(p)
        self.canvas.draw_idle()

    def _draw_patch(self, inst: PatchInstance):
        # Determine patch dimensions from params
        if inst.params.patch_length_m and inst.params.patch_width_m:
            L_m = inst.params.patch_length_m
            W_m = inst.params.patch_width_m
        else:
            L_m, W_m, _ = design_patch_for_frequency(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
        # Visual thickness
        t = max(0.00008, 0.00006 * inst.params.h_m)  # meters
        cx, cy = inst.center_x_m, inst.center_y_m

        # Patch box (gold)
        x0, x1 = cx - W_m/2, cx + W_m/2
        y0, y1 = cy - L_m/2, cy + L_m/2
        z0, z1 = 0.0, t
        verts = [
            # bottom, top, and four sides
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
        ]
        patch = Poly3DCollection(verts, alpha=0.98, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=0.8)
        try:
            patch.set_zsort('max')
        except Exception:
            pass
        self.ax.add_collection3d(patch)

        # Optional: short feed stub for orientation
        try:
            feed_w = calculate_microstrip_width(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
            fw = feed_w
            if inst.feed_direction == FeedDirection.NEG_X:
                xs = [x0 - 3*fw, x0]; ys = [cy - fw/2, cy + fw/2]
            elif inst.feed_direction == FeedDirection.POS_X:
                xs = [x1, x1 + 3*fw]; ys = [cy - fw/2, cy + fw/2]
            elif inst.feed_direction == FeedDirection.NEG_Y:
                xs = [cx - fw/2, cx + fw/2]; ys = [y0 - 3*fw, y0]
            else:
                xs = [cx - fw/2, cx + fw/2]; ys = [y1, y1 + 3*fw]
            feed = Poly3DCollection([
                [[xs[0], ys[0], z0], [xs[1], ys[0], z0], [xs[1], ys[1], z0], [xs[0], ys[1], z0]],
                [[xs[0], ys[0], z1], [xs[1], ys[0], z1], [xs[1], ys[1], z1], [xs[0], ys[1], z1]],
            ], alpha=0.98, facecolor='#ff6f3d', edgecolor='#a74323', linewidth=0.7)
            self.ax.add_collection3d(feed)
        except Exception:
            pass

        # Center marker
        self.ax.plot([cx], [cy], [0], marker='o', color='#ff4081', markersize=4)

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
            inst = PatchInstance(name=name, params=p, center_x_m=0.0, center_y_m=0.0)
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
        
