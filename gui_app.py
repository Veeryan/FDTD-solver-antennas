#!/usr/bin/env python3
"""
Patch Antenna Simulator - Desktop GUI Application
Modern Windows GUI using tkinter with matplotlib integration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import os
import sys
from pathlib import Path
import traceback

# Import our existing working code
from antenna_sim.models import PatchAntennaParams, Metal, metal_defaults
from antenna_sim.solver_approx import AnalyticalPatchSolver
# Using new simplified plotting functions
from antenna_sim.plotting import draw_patch_3d_geometry
from antenna_sim.solver_fdtd_openems_microstrip import FeedDirection, calculate_microstrip_width
from antenna_sim import (
    probe_openems_fixed, 
    prepare_openems_patch_fixed, 
    run_prepared_openems_fixed
)


class ModernStyle:
    """Modern color scheme and styling constants"""
    
    # Color palette (high-contrast, friendly "bubbly" accents)
    BG_DARK = "#161a1f"       # app background
    BG_MEDIUM = "#1f2530"     # cards / panels
    BG_LIGHT = "#273043"      # inputs
    ACCENT_BLUE = "#0ea5e9"   # primary
    ACCENT_GREEN = "#22c55e"  # success
    ACCENT_ORANGE = "#f59e0b" # warning
    ACCENT_PURPLE = "#7c3aed" # highlight
    TEXT_WHITE = "#f5f7fb"
    TEXT_GRAY = "#c7cbe0"
    BORDER = "#3a4152"
    
    # Fonts
    FONT_MAIN = ("Segoe UI", 10)
    FONT_HEADER = ("Segoe UI", 12, "bold")
    FONT_SMALL = ("Segoe UI", 8)
    
    @classmethod
    def configure_ttk_style(cls, root):
        """Configure modern ttk styles"""
        style = ttk.Style(root)
        
        # Configure styles for modern appearance
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Modern.TFrame', background=cls.BG_DARK)
        style.configure('Card.TFrame', background=cls.BG_MEDIUM, relief='solid', borderwidth=1)
        style.configure('Modern.TLabel', background=cls.BG_DARK, foreground=cls.TEXT_WHITE, font=cls.FONT_MAIN)
        style.configure('Header.TLabel', background=cls.BG_DARK, foreground=cls.TEXT_WHITE, font=cls.FONT_HEADER)

        # Inputs (high-contrast, readable in all states)
        style.configure('Modern.TEntry',
                        fieldbackground='#374151',  # slate-700
                        foreground=cls.TEXT_WHITE,
                        borderwidth=1,
                        relief='solid',
                        insertcolor=cls.TEXT_WHITE,
                        padding=4)
        style.map('Modern.TEntry',
                  fieldbackground=[('disabled', '#3f4c63'), ('readonly', '#3a475d')],
                  foreground=[('disabled', '#cbd5e1'), ('readonly', cls.TEXT_WHITE)])

        style.configure('Modern.TCombobox',
                        fieldbackground='#374151',
                        background='#111827',
                        foreground=cls.TEXT_WHITE,
                        borderwidth=1,
                        relief='solid',
                        padding=4)
        style.map('Modern.TCombobox',
                  fieldbackground=[('readonly', '#334155'), ('disabled', '#3f4c63')],
                  foreground=[('readonly', '#f8fafc'), ('disabled', '#cbd5e1')],
                  selectbackground=[('!disabled', '#1e40af')],  # indigo-800
                  selectforeground=[('!disabled', '#ffffff')])

        # Notebook tabs
        style.configure('TNotebook', background=cls.BG_DARK, borderwidth=0)
        style.configure('TNotebook.Tab', background=cls.BG_MEDIUM, foreground=cls.TEXT_GRAY, padding=(10, 6))
        style.map('TNotebook.Tab',
                  background=[('selected', '#334155')],
                  foreground=[('selected', '#f8fafc')])

        # Buttons (bubbly, brighter)
        style.configure('Modern.TButton', background=cls.ACCENT_BLUE, foreground=cls.TEXT_WHITE,
                        borderwidth=0, focuscolor='none', padding=(10, 6))
        style.configure('Success.TButton', background=cls.ACCENT_GREEN, foreground=cls.TEXT_WHITE,
                        borderwidth=0, focuscolor='none', padding=(10, 6))
        style.configure('Warning.TButton', background=cls.ACCENT_ORANGE, foreground=cls.TEXT_WHITE,
                        borderwidth=0, focuscolor='none', padding=(10, 6))
        
        # Map styles for different states
        style.map('Modern.TButton',
                 background=[('active', '#38bdf8'), ('pressed', '#0ea5e9')])
        style.map('Success.TButton',
                 background=[('active', '#0e6e0e'), ('pressed', '#0c5d0c')])


class ParameterFrame(ttk.Frame):
    """Frame for antenna parameters input"""
    
    def __init__(self, parent, style_class):
        super().__init__(parent, style='Card.TFrame')
        self.style_class = style_class
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Label(self, text="📐 Antenna Parameters", style='Header.TLabel')
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        # Parameters frame
        params_frame = ttk.Frame(self, style='Modern.TFrame')
        params_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create parameter inputs
        self.vars = {}
        
        # Row 0: Frequency
        ttk.Label(params_frame, text="Frequency (GHz):", style='Modern.TLabel').grid(row=0, column=0, sticky='w', pady=2)
        self.vars['frequency'] = tk.DoubleVar(value=2.45)
        freq_entry = ttk.Entry(params_frame, textvariable=self.vars['frequency'], style='Modern.TEntry', width=15)
        freq_entry.grid(row=0, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 1: Dielectric constant
        ttk.Label(params_frame, text="Dielectric εr:", style='Modern.TLabel').grid(row=1, column=0, sticky='w', pady=2)
        self.vars['eps_r'] = tk.DoubleVar(value=4.3)
        eps_entry = ttk.Entry(params_frame, textvariable=self.vars['eps_r'], style='Modern.TEntry', width=15)
        eps_entry.grid(row=1, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 2: Substrate thickness
        ttk.Label(params_frame, text="Thickness h (mm):", style='Modern.TLabel').grid(row=2, column=0, sticky='w', pady=2)
        self.vars['thickness'] = tk.DoubleVar(value=1.6)
        thick_entry = ttk.Entry(params_frame, textvariable=self.vars['thickness'], style='Modern.TEntry', width=15)
        thick_entry.grid(row=2, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 3: Loss tangent
        ttk.Label(params_frame, text="Loss tangent:", style='Modern.TLabel').grid(row=3, column=0, sticky='w', pady=2)
        self.vars['loss_tan'] = tk.DoubleVar(value=0.02)
        loss_entry = ttk.Entry(params_frame, textvariable=self.vars['loss_tan'], style='Modern.TEntry', width=15)
        loss_entry.grid(row=3, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 4: Metal selection
        ttk.Label(params_frame, text="Metal:", style='Modern.TLabel').grid(row=4, column=0, sticky='w', pady=2)
        self.vars['metal'] = tk.StringVar(value=Metal.COPPER.value)
        metal_combo = ttk.Combobox(params_frame, textvariable=self.vars['metal'], style='Modern.TCombobox', width=12)
        metal_combo['values'] = [m.value for m in Metal]
        metal_combo.grid(row=4, column=1, sticky='ew', padx=(5, 0), pady=2)
        metal_combo.state(['readonly'])
        
        # Row 5: Solver type selection
        ttk.Label(params_frame, text="Solver Type:", style='Modern.TLabel').grid(row=5, column=0, sticky='w', pady=2)
        self.vars['solver_type'] = tk.StringVar(value="Simple (Lumped Port)")
        solver_combo = ttk.Combobox(params_frame, textvariable=self.vars['solver_type'], style='Modern.TCombobox', width=20)
        solver_combo['values'] = ["Simple (Lumped Port)", "Microstrip Fed (MSL Port)", "Microstrip Fed (MSL Port, 3D)"]
        solver_combo.grid(row=5, column=1, sticky='ew', padx=(5, 0), pady=2)
        solver_combo.state(['readonly'])
        solver_combo.bind('<<ComboboxSelected>>', self.on_solver_type_change)
        
        # Row 6: Feed direction (only for microstrip)
        self.feed_dir_label = ttk.Label(params_frame, text="Feed Direction:", style='Modern.TLabel')
        self.feed_dir_label.grid(row=6, column=0, sticky='w', pady=2)
        self.vars['feed_direction'] = tk.StringVar(value="-X")
        self.feed_dir_combo = ttk.Combobox(params_frame, textvariable=self.vars['feed_direction'], style='Modern.TCombobox', width=12)
        self.feed_dir_combo['values'] = ["-X", "+X", "-Y", "+Y"]
        self.feed_dir_combo.grid(row=6, column=1, sticky='ew', padx=(5, 0), pady=2)
        self.feed_dir_combo.state(['readonly'])
        
        # Row 7: Boundary type
        ttk.Label(params_frame, text="Boundary:", style='Modern.TLabel').grid(row=7, column=0, sticky='w', pady=2)
        self.vars['boundary'] = tk.StringVar(value="MUR")
        boundary_combo = ttk.Combobox(params_frame, textvariable=self.vars['boundary'], style='Modern.TCombobox', width=12)
        boundary_combo['values'] = ["MUR", "PML_8"]
        boundary_combo.grid(row=7, column=1, sticky='ew', padx=(5,0), pady=2)
        boundary_combo.state(['readonly'])

        # Row 8: Theta/Phi sampling (for 3D)
        ttk.Label(params_frame, text="θ step (deg):", style='Modern.TLabel').grid(row=8, column=0, sticky='w', pady=2)
        self.vars['theta_step'] = tk.DoubleVar(value=2.0)
        ttk.Entry(params_frame, textvariable=self.vars['theta_step'], style='Modern.TEntry', width=12).grid(row=8, column=1, sticky='w', padx=(5,0), pady=2)

        ttk.Label(params_frame, text="φ step (deg):", style='Modern.TLabel').grid(row=9, column=0, sticky='w', pady=2)
        self.vars['phi_step'] = tk.DoubleVar(value=5.0)
        ttk.Entry(params_frame, textvariable=self.vars['phi_step'], style='Modern.TEntry', width=12).grid(row=9, column=1, sticky='w', padx=(5,0), pady=2)

        # Row 10: Normalization toggle
        ttk.Label(params_frame, text="3D scale:", style='Modern.TLabel').grid(row=10, column=0, sticky='w', pady=2)
        self.vars['norm_mode'] = tk.StringVar(value="dBi")
        norm_combo = ttk.Combobox(params_frame, textvariable=self.vars['norm_mode'], style='Modern.TCombobox', width=12)
        norm_combo['values'] = ["dBi", "Normalized"]
        norm_combo.grid(row=10, column=1, sticky='ew', padx=(5,0), pady=2)
        norm_combo.state(['readonly'])

        # Row 11: openEMS DLL path
        ttk.Label(params_frame, text="openEMS DLL:", style='Modern.TLabel').grid(row=11, column=0, sticky='w', pady=2)
        self.vars['dll_path'] = tk.StringVar(value=os.path.abspath("openEMS"))
        dll_frame = ttk.Frame(params_frame, style='Modern.TFrame')
        dll_frame.grid(row=11, column=1, sticky='ew', padx=(5, 0), pady=2)
        dll_frame.columnconfigure(0, weight=1)
        
        dll_entry = ttk.Entry(dll_frame, textvariable=self.vars['dll_path'], style='Modern.TEntry')
        dll_entry.grid(row=0, column=0, sticky='ew')
        
        browse_btn = ttk.Button(dll_frame, text="...", width=3, command=self.browse_dll_path)
        browse_btn.grid(row=0, column=1, padx=(2, 0))
        
        # Configure column weights
        params_frame.columnconfigure(1, weight=1)
        
        # Initially hide feed direction controls (show only for microstrip)
        self.on_solver_type_change()
        
    def browse_dll_path(self):
        """Browse for openEMS DLL directory"""
        path = filedialog.askdirectory(title="Select openEMS Installation Directory")
        if path:
            self.vars['dll_path'].set(path)
    
    def on_solver_type_change(self, event=None):
        """Handle solver type selection change"""
        solver_type = self.vars['solver_type'].get()
        is_microstrip = "Microstrip" in solver_type
        
        # Show/hide feed direction controls based on solver type
        if is_microstrip:
            self.feed_dir_label.grid()
            self.feed_dir_combo.grid()
        else:
            self.feed_dir_label.grid_remove()
            self.feed_dir_combo.grid_remove()
    
    def get_parameters(self):
        """Get current parameter values as PatchAntennaParams object"""
        try:
            metal_name = self.vars['metal'].get()
            metal_enum = Metal(metal_name)
            metal_props = metal_defaults[metal_enum]
            
            return PatchAntennaParams(
                frequency_hz=self.vars['frequency'].get() * 1e9,
                eps_r=self.vars['eps_r'].get(),
                h_m=self.vars['thickness'].get() * 1e-3,
                loss_tangent=self.vars['loss_tan'].get(),
                metal=metal_props,  # Pass MetalProperties object, not enum
                patch_length_m=None,  # Auto-calculate
                patch_width_m=None    # Auto-calculate
            )
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")

    def set_params_state(self, state: str):
        try:
            for child in self.winfo_children():
                pass
            # brute-force: iterate entries/combos under this frame
            for widget in self.winfo_children():
                for sub in widget.winfo_children():
                    try:
                        sub.configure(state=state)
                    except Exception:
                        pass
        except Exception:
            pass


class ControlFrame(ttk.Frame):
    """Frame for control buttons"""
    
    def __init__(self, parent, on_geometry_update, on_run_simulation):
        super().__init__(parent, style='Card.TFrame')
        self.on_geometry_update = on_geometry_update
        self.on_run_simulation = on_run_simulation
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Label(self, text="🔧 Controls", style='Header.TLabel')
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        # Buttons frame
        btn_frame = ttk.Frame(self, style='Modern.TFrame')
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        # Update Geometry button
        self.geometry_btn = ttk.Button(
            btn_frame, 
            text="📐 Update Geometry", 
            style='Modern.TButton',
            command=self.on_geometry_update
        )
        self.geometry_btn.pack(fill='x', pady=2)
        
        # Run Simulation button
        self.sim_btn = ttk.Button(
            btn_frame, 
            text="⚡ Run FDTD Simulation", 
            style='Success.TButton',
            command=self.on_run_simulation
        )
        self.sim_btn.pack(fill='x', pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(btn_frame, textvariable=self.status_var, style='Modern.TLabel')
        self.status_label.pack(fill='x', pady=(5, 0))
    
    def set_status(self, status, color='white'):
        """Update status text"""
        self.status_var.set(status)
    
    def set_simulation_running(self, running):
        """Enable/disable simulation button"""
        if running:
            self.sim_btn.config(state='disabled', text="⏳ Running...")
        else:
            self.sim_btn.config(state='normal', text="⚡ Run FDTD Simulation")


class LogFrame(ttk.Frame):
    """Frame for simulation log display"""
    
    def __init__(self, parent):
        super().__init__(parent, style='Card.TFrame')
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Label(self, text="📋 Simulation Log", style='Header.TLabel')
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        # Log text widget with scrollbar
        log_container = ttk.Frame(self, style='Modern.TFrame')
        log_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create text widget with dark theme
        self.log_text = tk.Text(
            log_container,
            bg='#1e1e1e',
            fg='#00ff00',  # Green text like terminal
            font=('Consolas', 9),
            insertbackground='white',
            selectbackground='#404040',
            wrap='word',
            state='disabled'
        )
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(log_container, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack text and scrollbar
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Clear button
        clear_btn = ttk.Button(
            self,
            text="🗑️ Clear Log",
            style='Modern.TButton',
            command=self.clear_log
        )
        clear_btn.pack(pady=(0, 10))
        
    def append_log(self, text):
        """Append text to the log"""
        self.log_text.config(state='normal')
        
        # Handle timestep updates by replacing the last line if it's a timestep
        if 'Timestep:' in text and 'Speed:' in text:
            # This looks like a timestep update, check if we should replace the last line
            current_content = self.log_text.get('end-2l', 'end-1l').strip()
            if 'Timestep:' in current_content and 'Speed:' in current_content:
                # Replace the last timestep line
                self.log_text.delete('end-2l', 'end-1l')
                self.log_text.insert('end-1l', text + '\n')
            else:
                # Add new timestep line
                self.log_text.insert('end', text + '\n')
        else:
            # Regular log message
            self.log_text.insert('end', text + '\n')
        
        # Limit log size to prevent memory issues (keep last 1000 lines)
        lines = self.log_text.get(1.0, 'end').count('\n')
        if lines > 1000:
            self.log_text.delete(1.0, f'{lines-1000}.0')
        
        self.log_text.see('end')  # Auto-scroll to bottom
        self.log_text.config(state='disabled')
        
        # Force GUI update
        self.log_text.update_idletasks()
        
    def clear_log(self):
        """Clear the log"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, 'end')
        self.log_text.config(state='disabled')
    
    


class PlotFrame(ttk.Frame):
    """Frame for matplotlib plots"""
    
    def __init__(self, parent):
        super().__init__(parent, style='Card.TFrame')
        # mode management
        self.mode = 'single'  # 'single' | 'multi'
        self.multi_panel = None
        self._mode_changed_cb = None
        self._multi_placeholder_2d = None
        self._multi_placeholder_3d = None
        self._banner_geom = None
        self._banner_2d = None
        self._banner_3d = None
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ttk.Label(self, text="📊 Visualization", style='Header.TLabel')
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        # Small toolbar above tabs (mode selector)
        toolbar = ttk.Frame(self, style='Modern.TFrame')
        toolbar.pack(fill='x', padx=10, pady=(0, 4))
        self.btn_single = ttk.Button(toolbar, text="Single Antenna", style='Modern.TButton', command=self.set_mode_single)
        self.btn_single.pack(side='left', padx=(0, 6))
        self.btn_multi = ttk.Button(toolbar, text="Multi Antenna", style='Modern.TButton', command=self.set_mode_multi)
        self.btn_multi.pack(side='left')
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Geometry tab
        self.geometry_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.geometry_frame, text="Geometry")
        
        # 2D Patterns tab
        self.pattern_2d_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.pattern_2d_frame, text="2D Patterns")
        
        # 3D Pattern tab
        self.pattern_3d_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.pattern_3d_frame, text="3D Pattern")

        # Add small mode banners at the top of each tab
        self._create_banners()

        # Zoom controls for Geometry and 3D Pattern
        def _add_zoom_controls(container, get_canvas, get_axes):
            btns = ttk.Frame(container, style='Modern.TFrame')
            btns.pack(side='top', anchor='ne', padx=6, pady=4)
            ttk.Button(btns, text='+', width=3, command=lambda: _zoom(get_axes(), 0.8, get_canvas())).pack(side='left')
            ttk.Button(btns, text='-', width=3, command=lambda: _zoom(get_axes(), 1.25, get_canvas())).pack(side='left', padx=(4,0))

        def _zoom(ax, factor, canvas):
            if ax is None:
                return
            try:
                xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
                def _scale(lims):
                    c = 0.5*(lims[0]+lims[1]); r = 0.5*(lims[1]-lims[0]); r *= factor; return (c-r, c+r)
                ax.set_xlim(_scale(xlim)); ax.set_ylim(_scale(ylim)); ax.set_zlim(_scale(zlim))
                if canvas: canvas.draw_idle()
            except Exception:
                pass

        # lazy getters filled when plots are created
        self._geom_ax = None
        self._geom_canvas = None
        self._p3d_ax = None
        self._p3d_canvas = None

        _add_zoom_controls(self.geometry_frame, lambda: self._geom_canvas, lambda: self._geom_ax)
        _add_zoom_controls(self.pattern_3d_frame, lambda: self._p3d_canvas, lambda: self._p3d_ax)
        
        # Initialize empty plots
        self.geometry_canvas = None
        self.pattern_2d_canvas = None
        self.pattern_3d_canvas = None
        self._update_mode_buttons()
        self._update_mode_banners()
    
    # ---- Mode management ----
    def set_mode_changed_callback(self, cb):
        self._mode_changed_cb = cb
    
    def _update_mode_buttons(self):
        try:
            if self.mode == 'single':
                self.btn_single.state(['disabled'])
                self.btn_multi.state(['!disabled'])
            else:
                self.btn_multi.state(['disabled'])
                self.btn_single.state(['!disabled'])
        except Exception:
            pass

    def _create_banners(self):
        def make_banner(parent):
            frame = ttk.Frame(parent, style='Modern.TFrame')
            frame.pack(fill='x', padx=6, pady=(2, 2))
            lbl = ttk.Label(frame, text='', style='Modern.TLabel')
            lbl.pack(side='left')
            return lbl
        try:
            self._banner_geom = make_banner(self.geometry_frame)
            self._banner_2d = make_banner(self.pattern_2d_frame)
            self._banner_3d = make_banner(self.pattern_3d_frame)
        except Exception:
            pass

    def _update_mode_banners(self):
        try:
            mode_text = '🧍 Single Antenna view' if self.mode == 'single' else '🧩 Multi Antenna view'
            for lbl in (self._banner_geom, self._banner_2d, self._banner_3d):
                if lbl is not None:
                    lbl.config(text=mode_text)
        except Exception:
            pass

    def set_mode_single(self):
        if self.mode == 'single':
            return
        # remove multi panel if present
        if self.multi_panel is not None:
            try:
                self.multi_panel.destroy()
            except Exception:
                pass
            self.multi_panel = None
        self.mode = 'single'
        self._update_mode_buttons()
        self._update_mode_banners()
        # remove multi placeholders on pattern tabs
        self._hide_multi_placeholder('2d')
        self._hide_multi_placeholder('3d')
        if self._mode_changed_cb:
            try:
                self._mode_changed_cb('single')
            except Exception:
                pass

    def set_mode_multi(self):
        if self.mode == 'multi':
            return
        # destroy single geometry canvas if present
        try:
            if self.geometry_canvas:
                self.geometry_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        self.geometry_canvas = None
        self._geom_canvas = None
        self._geom_ax = None
        # create embedded panel
        try:
            from antenna_sim.multi_patch_designer import MultiPatchPanel
            self.multi_panel = MultiPatchPanel(self.geometry_frame)
            self.multi_panel.pack(fill='both', expand=True)
        except Exception as e:
            try:
                messagebox.showerror("Error", f"Failed to open Multi Antenna view: {e}")
            except Exception:
                pass
            return
        self.mode = 'multi'
        self._update_mode_buttons()
        self._update_mode_banners()
        # show placeholders for patterns while multi-solver isn't wired
        self._show_multi_placeholder('2d')
        self._show_multi_placeholder('3d')
        if self._mode_changed_cb:
            try:
                self._mode_changed_cb('multi')
            except Exception:
                pass

    # --- Placeholders for Multi mode ---
    def _show_multi_placeholder(self, which: str):
        try:
            if which == '2d':
                # clear any existing 2D canvas
                if self.pattern_2d_canvas:
                    self.pattern_2d_canvas.get_tk_widget().destroy()
                    self.pattern_2d_canvas = None
                if self._multi_placeholder_2d is None:
                    frame = ttk.Frame(self.pattern_2d_frame, style='Modern.TFrame')
                    lbl = ttk.Label(frame, text="Multi Antenna 2D Patterns\n(coming soon)", style='Header.TLabel')
                    lbl.pack(pady=20)
                    info = ttk.Label(frame, text="Edit antennas in the Geometry tab.\nRun multi-antenna FDTD will be added here.", style='Modern.TLabel')
                    info.pack()
                    frame.pack(fill='both', expand=True)
                    self._multi_placeholder_2d = frame
            elif which == '3d':
                # clear any existing 3D canvas
                if self.pattern_3d_canvas:
                    self.pattern_3d_canvas.get_tk_widget().destroy()
                    self.pattern_3d_canvas = None
                    self._p3d_ax = None
                    self._p3d_canvas = None
                if self._multi_placeholder_3d is None:
                    frame = ttk.Frame(self.pattern_3d_frame, style='Modern.TFrame')
                    lbl = ttk.Label(frame, text="Multi Antenna 3D Patterns\n(coming soon)", style='Header.TLabel')
                    lbl.pack(pady=20)
                    info = ttk.Label(frame, text="Edit antennas in the Geometry tab.\nRun multi-antenna FDTD will be added here.", style='Modern.TLabel')
                    info.pack()
                    frame.pack(fill='both', expand=True)
                    self._multi_placeholder_3d = frame
        except Exception:
            pass

    def _hide_multi_placeholder(self, which: str):
        try:
            if which == '2d' and self._multi_placeholder_2d is not None:
                self._multi_placeholder_2d.destroy()
                self._multi_placeholder_2d = None
            if which == '3d' and self._multi_placeholder_3d is not None:
                self._multi_placeholder_3d.destroy()
                self._multi_placeholder_3d = None
        except Exception:
            pass
        
    def open_multi_patch(self):
        """Open the Multi Patch Designer window."""
        try:
            from antenna_sim.multi_patch_designer import MultiPatchDesigner
            MultiPatchDesigner(self.winfo_toplevel())
        except Exception as e:
            try:
                messagebox.showerror("Error", f"Failed to open Multi Patch Designer: {e}")
            except Exception:
                pass
        
    def update_geometry_plot(self, params, solver_type: str = "Simple (Lumped Port)", feed_direction_str: str = "-X"):
        """Update the geometry visualization"""
        try:
            # If in multi mode, skip single-antenna drawing
            if getattr(self, 'mode', 'single') == 'multi':
                return
            print("DEBUG: Starting geometry plot update...")
            print(f"DEBUG: Params - freq={params.frequency_hz/1e9:.2f}GHz, εr={params.eps_r}, h={params.h_m*1e3:.1f}mm")
            
            # Clear existing plot
            if self.geometry_canvas:
                print("DEBUG: Destroying existing canvas...")
                self.geometry_canvas.get_tk_widget().destroy()
            
            # Calculate patch dimensions if not provided (same as Streamlit)
            print("DEBUG: Calculating patch dimensions...")
            from antenna_sim.physics import design_patch_for_frequency
            if params.patch_length_m and params.patch_width_m:
                L_m = params.patch_length_m
                W_m = params.patch_width_m
                print(f"DEBUG: Using provided dimensions: L={L_m*1e3:.1f}mm, W={W_m*1e3:.1f}mm")
            else:
                L_m, W_m, _ = design_patch_for_frequency(params.frequency_hz, params.eps_r, params.h_m)
                print(f"DEBUG: Calculated dimensions: L={L_m*1e3:.1f}mm, W={W_m*1e3:.1f}mm")
            
            # Create enhanced geometry plot
            print("DEBUG: Creating enhanced geometry plot...")
            if solver_type == "Microstrip Fed (MSL Port)" or solver_type == "Microstrip Fed (MSL Port, 3D)":
                # Base geometry
                geometry_fig = draw_patch_3d_geometry(L_m, W_m, params.h_m, fig_size=(8, 6), show_labels=False)
                ax_back = geometry_fig.gca()
                ax_overlay = ax_back
                # Overlay a simple 50Ω microstrip trace matching FDTD coordinates
                feed_direction = FeedDirection(feed_direction_str)
                feed_width_m = calculate_microstrip_width(params.frequency_hz, params.eps_r, params.h_m)
                feed_width_mm = feed_width_m * 1e3
                # Substrate outline (same as plotting.py)
                mm = 1e3
                L = L_m * mm
                W = W_m * mm
                h = params.h_m * mm
                margin = max(5.0, 0.2 * max(L, W))
                sub_L = L + 2 * margin
                sub_W = W + 2 * margin
                z_plane = 0.02  # slightly above patch plane for visibility
                if feed_direction == FeedDirection.NEG_X:
                    feed_start = [-sub_L/2, -feed_width_mm/2, z_plane]
                    feed_stop  = [-L/2,     +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.POS_X:
                    feed_start = [ L/2,     -feed_width_mm/2, z_plane]
                    feed_stop  = [ sub_L/2, +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.NEG_Y:
                    feed_start = [-feed_width_mm/2, -sub_W/2, z_plane]
                    feed_stop  = [ +feed_width_mm/2, -W/2,   z_plane]
                else:  # POS_Y
                    feed_start = [-feed_width_mm/2, W/2,   z_plane]
                    feed_stop  = [ +feed_width_mm/2, sub_W/2, z_plane]
                # Draw the microstrip as a small 3D box (prism) for better realism
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                # Use the same copper thickness as the patch rendering for consistency
                t = max(0.08, 0.06 * h)  # mm
                x0, y0 = feed_start[0], feed_start[1]
                x1, y1 = feed_stop[0], feed_stop[1]
                z0, z1 = z_plane, z_plane + t
                # Build prism faces depending on orientation
                if abs(x1 - x0) > abs(y1 - y0):
                    # strip along X
                    verts = [
                        # bottom
                        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
                        # top
                        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                        # sides
                        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
                        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
                        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
                    ]
                else:
                    # strip along Y
                    verts = [
                        # bottom
                        [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
                        # top
                        [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]],
                        # sides
                        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
                        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
                        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
                    ]
                strip = Poly3DCollection(verts, alpha=0.99, facecolor='#ff6f3d', edgecolor='#a74323', linewidth=0.9)
                try:
                    strip.set_zsort('max')
                except Exception:
                    pass
                strip.set_zorder(10)
                ax_overlay.add_collection3d(strip)
                # Draw a top cap to guarantee the top face always renders above the substrate
                top_cap = Poly3DCollection([[
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
                ]], alpha=1.0, facecolor='#ff6f3d', edgecolor='#a74323', linewidth=0.7)
                try:
                    top_cap.set_zsort('max')
                except Exception:
                    pass
                top_cap.set_zorder(11)
                ax_overlay.add_collection3d(top_cap)
                # Also draw a patch top cap on overlay axis to avoid blending artifacts
                patch_thickness = max(0.08, 0.06 * h)
                patch_cap = Poly3DCollection([[[-L/2, -W/2, patch_thickness], [L/2, -W/2, patch_thickness], [L/2, W/2, patch_thickness], [-L/2, W/2, patch_thickness]]],
                                             alpha=1.0, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=0.9)
                try:
                    patch_cap.set_zsort('max')
                except Exception:
                    pass
                patch_cap.set_zorder(12)
                ax_overlay.add_collection3d(patch_cap)
                print(f"DEBUG: Microstrip overlay added dir={feed_direction} width={feed_width_mm:.2f} mm")
                ax_main = ax_back
            else:
                # Simple patch visualization (original approach)
                geometry_fig = draw_patch_3d_geometry(L_m, W_m, params.h_m, fig_size=(8, 6), show_labels=True)
                print("DEBUG: Simple patch geometry created")
                ax_overlay = None
                ax_main = geometry_fig.gca()
            print("DEBUG: Enhanced geometry plot created successfully")
            
            # Convert matplotlib figure to tkinter-compatible figure
            print("DEBUG: Converting to tkinter figure...")
            # Get the axes from the created figure
            ax = geometry_fig.gca()
            overlay_ax = None
            
            # Style the plot to match dark theme
            print("DEBUG: Applying dark theme styling...")
            geometry_fig.patch.set_facecolor('#2b2b2b')
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)
            
            # Update title with frequency info
            ax.set_title(f'Patch Antenna Geometry\n{params.frequency_hz/1e9:.2f} GHz, εr={params.eps_r}', 
                        color='white', fontsize=12, pad=20)

            # Add dimension labels LAST so they are always on top
            try:
                mm = 1e3
                L = L_m * mm
                W = W_m * mm
                h = params.h_m * mm
                margin = max(5.0, 0.2 * max(L, W))
                label_box = dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.65)
                import matplotlib.patheffects as pe
                # draw labels on main axes for the simple view; overlay if available (microstrip view)
                label_ax = ax_overlay if ax_overlay is not None else ax
                txt1 = label_ax.text(0, -W/2-0.9*margin, 0.08*h, f'L = {L:.1f} mm', ha='center', fontsize=13, color='white', bbox=label_box)
                txt2 = label_ax.text(L/2+0.9*margin, 0, 0.08*h, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=13, color='white', bbox=label_box)
                for t in (txt1, txt2):
                    t.set_zorder(100)
                    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
            except Exception:
                pass
            
            geometry_fig.tight_layout()
            print("DEBUG: Dark theme styling complete")
            # Corner triad overlay (top-left) to replace the in-scene axes
            ax_triad = None
            try:
                inset_rect = [0.06, 0.78, 0.12, 0.16]  # top-left corner
                ax_triad = geometry_fig.add_axes(inset_rect, projection='3d')
                ax_triad.set_axis_off(); ax_triad.patch.set_alpha(0)
                ax_triad.set_xlim([-0.05, 1.05]); ax_triad.set_ylim([-0.05, 1.05]); ax_triad.set_zlim([-0.05, 1.05])
                try: ax_triad.set_box_aspect([1,1,1])
                except Exception: pass
                arrow_kw = dict(length=1.0, normalize=True, arrow_length_ratio=0.2, linewidth=2.2)
                ax_triad.quiver(0,0,0, 1,0,0, color='red', **arrow_kw)
                ax_triad.quiver(0,0,0, 0,1,0, color='green', **arrow_kw)
                ax_triad.quiver(0,0,0, 0,0,1, color='blue', **arrow_kw)
                ax_triad.text(1.02,0,0,'x', color='red', weight='bold')
                ax_triad.text(0,1.02,0,'y', color='green', weight='bold')
                ax_triad.text(0,0,1.02,'z', color='blue', weight='bold')
            except Exception:
                ax_triad = None
            
            # Synchronize triad with main axes during user rotations
            try:
                if True:
                    def _sync_views(event=None):
                        try:
                            elev = getattr(ax, 'elev', 22)
                            azim = getattr(ax, 'azim', -45)
                            # keep labels on overlay updated if axes changed limits
                            if ax_triad is not None:
                                ax_triad.view_init(elev=elev, azim=azim)
                            self.geometry_canvas.draw_idle()
                        except Exception:
                            pass
                    geometry_fig.canvas.mpl_connect('motion_notify_event', _sync_views)
                    geometry_fig.canvas.mpl_connect('button_release_event', _sync_views)
                    geometry_fig.canvas.mpl_connect('scroll_event', _sync_views)
            except Exception:
                pass

            # Add to GUI
            print("DEBUG: Adding canvas to GUI...")
            self.geometry_canvas = FigureCanvasTkAgg(geometry_fig, self.geometry_frame)
            self.geometry_canvas.get_tk_widget().pack(fill='both', expand=True)
            self.geometry_canvas.draw()
            # Expose for zoom controls and mouse scroll
            self._geom_ax = ax
            self._geom_canvas = self.geometry_canvas
            def _on_scroll(event):
                try:
                    factor = 0.9 if event.button == 'up' else 1.1
                    xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
                    def _scale(lims):
                        c = 0.5*(lims[0]+lims[1]); r = 0.5*(lims[1]-lims[0]); r *= factor; return (c-r, c+r)
                    ax.set_xlim(_scale(xlim)); ax.set_ylim(_scale(ylim)); ax.set_zlim(_scale(zlim))
                    self.geometry_canvas.draw_idle()
                except Exception:
                    pass
            geometry_fig.canvas.mpl_connect('scroll_event', _on_scroll)
            
            # Force GUI update
            self.geometry_canvas.get_tk_widget().update_idletasks()
            print("DEBUG: Geometry plot update complete!")
            
        except Exception as e:
            print(f"❌ ERROR updating geometry plot: {e}")
            import traceback
            traceback.print_exc()
    
    def update_2d_patterns(self, theta, intensity):
        """Update 2D radiation pattern plots - EXACT copy of Streamlit version"""
        try:
            # Multi mode: show placeholder and exit
            if getattr(self, 'mode', 'single') == 'multi':
                self._show_multi_placeholder('2d')
                return
            # In single mode ensure placeholder is hidden
            self._hide_multi_placeholder('2d')
            # Clear existing plot
            if self.pattern_2d_canvas:
                self.pattern_2d_canvas.get_tk_widget().destroy()
            
            if intensity.ndim == 2 and intensity.shape[1] >= 2:
                # Extract E-plane and H-plane - EXACT same as Streamlit
                th_deg = np.rad2deg(theta)
                E_plane = intensity[:, 0]  # phi=0 (E-plane)
                H_plane = intensity[:, 1] if intensity.shape[1] > 1 else intensity[:, 0]  # phi=90 (H-plane)
                
                # Need to extend pattern to full 360° for proper polar display - EXACT same as Streamlit
                # The current theta goes 0° to 180°, but we need full circle
                # Mirror the pattern: 0°-180° data becomes 0°-180° and 180°-360°
                th_full = np.concatenate([th_deg, th_deg[1:] + 180])  # 0°..360° (open)
                E_full = np.concatenate([E_plane, E_plane[1:][::-1]])  # Mirror E-plane
                H_full = np.concatenate([H_plane, H_plane[1:][::-1]])  # Mirror H-plane
                # Close the curves to avoid a gap at 0°/+Z
                th_closed = np.concatenate([th_full, [360.0]])
                E_closed = np.concatenate([E_full, [E_full[0]]])
                H_closed = np.concatenate([H_full, [H_full[0]]])
                
                # Create polar plots - same size as 3D plot (20x10) - EXACT same as Streamlit
                fig = Figure(figsize=(20*0.8, 10*0.8), facecolor='#2b2b2b')  # Using ui_scale=0.8 equivalent
                ax1 = fig.add_subplot(121, projection='polar', facecolor='#2b2b2b')
                ax2 = fig.add_subplot(122, projection='polar', facecolor='#2b2b2b')
                
                # E-plane (phi=0°) - full 360° - EXACT same as Streamlit
                ax1.plot(np.deg2rad(th_closed), E_closed, 'r-', linewidth=3, label='E-plane (phi=0°)')
                ax1.set_title('E-plane (ZX, phi=0°)', fontsize=14, pad=25, color='white')
                ax1.set_theta_zero_location('N')
                ax1.set_theta_direction(-1)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([max(-20, np.min(E_full)-2), np.max(E_full)+2])
                # Add coordinate labels next to degrees - E-plane cuts through X direction
                ax1.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                                 ["0°\n(+Z)", "45°", "90°\n(+X)", "135°", "180°\n(-Z)", "225°", "270°\n(-X)", "315°"])
                ax1.tick_params(colors='white')
                
                # H-plane (phi=90°) - full 360° - EXACT same as Streamlit
                ax2.plot(np.deg2rad(th_closed), H_closed, 'b-', linewidth=3, label='H-plane (phi=90°)')
                ax2.set_title('H-plane (YZ, phi=90°)', fontsize=14, pad=25, color='white')
                ax2.set_theta_zero_location('N')
                ax2.set_theta_direction(-1)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([max(-20, np.min(H_full)-2), np.max(H_full)+2])
                # Add coordinate labels next to degrees - H-plane cuts through Y direction
                ax2.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                                 ["0°\n(+Z)", "45°", "90°\n(+Y)", "135°", "180°\n(-Z)", "225°", "270°\n(-Y)", "315°"])
                ax2.tick_params(colors='white')
                
                fig.tight_layout()
                
                # Add to GUI
                self.pattern_2d_canvas = FigureCanvasTkAgg(fig, self.pattern_2d_frame)
                self.pattern_2d_canvas.get_tk_widget().pack(fill='both', expand=True)
                self.pattern_2d_canvas.draw()
            else:
                print(f"Data shape: {intensity.shape} - Cannot extract plane cuts")
            
        except Exception as e:
            print(f"Error updating 2D patterns: {e}")
            import traceback
            traceback.print_exc()
    
    def update_3d_pattern(self, theta, phi, intensity, params):
        """Update 3D radiation pattern plot - EXACT copy of Streamlit version"""
        try:
            # Multi mode: show placeholder and exit
            if getattr(self, 'mode', 'single') == 'multi':
                self._show_multi_placeholder('3d')
                return
            # In single mode ensure placeholder is hidden
            self._hide_multi_placeholder('3d')
            # Clear existing plot
            if self.pattern_3d_canvas:
                self.pattern_3d_canvas.get_tk_widget().destroy()
            
            if intensity.ndim == 2 and intensity.shape[1] >= 2:
                # We have E-plane (phi=0°) and H-plane (phi=90°) cuts - EXACT same as Streamlit
                # Create a full 3D pattern by interpolating/extending these cuts
                
                th_deg = np.rad2deg(theta)  # theta in degrees
                E_plane_data = intensity[:, 0]  # phi=0° cut
                H_plane_data = intensity[:, 1]  # phi=90° cut
                
                # Create full phi range for 3D visualization (0° to 360°) - EXACT same as Streamlit
                phi_full = np.linspace(0, 2*np.pi, 73)  # 73 points = 5° resolution
                phi_full_deg = np.rad2deg(phi_full)
                
                # Create full 3D pattern by interpolating between E and H planes - EXACT same as Streamlit
                pattern_3d = np.zeros((len(theta), len(phi_full)))
                
                for i, phi_val in enumerate(phi_full):
                    # Interpolate between E-plane (0°/180°) and H-plane (90°/270°)
                    phi_norm = (phi_val % (2*np.pi))  # Normalize to [0, 2π]
                    
                    if phi_norm <= np.pi/2:  # 0° to 90°
                        weight = phi_norm / (np.pi/2)
                        pattern_3d[:, i] = (1-weight) * E_plane_data + weight * H_plane_data
                    elif phi_norm <= np.pi:  # 90° to 180°
                        weight = (phi_norm - np.pi/2) / (np.pi/2)
                        pattern_3d[:, i] = (1-weight) * H_plane_data + weight * E_plane_data
                    elif phi_norm <= 3*np.pi/2:  # 180° to 270°
                        weight = (phi_norm - np.pi) / (np.pi/2)
                        pattern_3d[:, i] = (1-weight) * E_plane_data + weight * H_plane_data
                    else:  # 270° to 360°
                        weight = (phi_norm - 3*np.pi/2) / (np.pi/2)
                        pattern_3d[:, i] = (1-weight) * H_plane_data + weight * E_plane_data
                
                # Convert to spherical coordinates for 3D plotting - EXACT same as Streamlit
                TH, PH = np.meshgrid(theta, phi_full, indexing='ij')
                
                # Normalize pattern for radius (linear scale, 0 to 1) - EXACT same as Streamlit
                pattern_norm = pattern_3d - np.max(pattern_3d)  # Normalize to peak = 0 dB
                pattern_linear = np.maximum(0.01, 10**(pattern_norm/20))  # Convert to linear, min 0.01
                
                # Convert to Cartesian coordinates - EXACT same as Streamlit
                X = pattern_linear * np.sin(TH) * np.cos(PH)
                Y = pattern_linear * np.sin(TH) * np.sin(PH) 
                Z = pattern_linear * np.cos(TH)
                
                # Create the 3D plot - EXACT same as Streamlit
                fig_3d = Figure(figsize=(10*0.8, 8*0.8), facecolor='#2b2b2b')  # Using ui_scale=0.8 equivalent
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                ax_3d.set_facecolor('#2b2b2b')
                
                # Plot the 3D surface with color mapping - EXACT same as Streamlit
                surf = ax_3d.plot_surface(
                    X, Y, Z,
                    facecolors=plt.cm.jet((pattern_3d - np.min(pattern_3d)) / (np.max(pattern_3d) - np.min(pattern_3d))),
                    linewidth=0,
                    antialiased=True,
                    alpha=0.8,
                )
                
                # Add patch antenna at the bottom - EXACT same as Streamlit
                patch_size = 0.15
                patch_x = [-patch_size/2, patch_size/2, patch_size/2, -patch_size/2, -patch_size/2]
                patch_y = [-patch_size/3, -patch_size/3, patch_size/3, patch_size/3, -patch_size/3]
                patch_z = [-0.9, -0.9, -0.9, -0.9, -0.9]
                ax_3d.plot(patch_x, patch_y, patch_z, color='orange', linewidth=3, label='Patch Antenna')
                
                # Ground plane - EXACT same as Streamlit
                ground_size = 0.3
                ax_3d.plot([-ground_size, ground_size], [-ground_size, -ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                ax_3d.plot([-ground_size, ground_size], [ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                ax_3d.plot([-ground_size, -ground_size], [-ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                ax_3d.plot([ground_size, ground_size], [-ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                
                # Coordinate system arrows - EXACT same as Streamlit
                ax_3d.quiver(0, 0, -1.0, 0.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='+X')
                ax_3d.quiver(0, 0, -1.0, 0, 0.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='+Y')
                ax_3d.quiver(0, 0, -1.0, 0, 0, 0.3, color='blue', arrow_length_ratio=0.1, linewidth=2, label='+Z')
                
                # Labels and formatting - EXACT same as Streamlit
                ax_3d.set_xlabel('X', fontsize=12, color='white')
                ax_3d.set_ylabel('Y', fontsize=12, color='white')
                ax_3d.set_zlabel('Z', fontsize=12, color='white')
                ax_3d.set_title(f'3D Radiation Pattern\nMax Gain: {np.max(pattern_3d):.1f} dBi @ {params.frequency_hz/1e9:.2f} GHz', 
                               fontsize=14, pad=20, color='white')
                
                # Set equal aspect ratio and good viewing angle - EXACT same as Streamlit
                max_range = 1.2
                ax_3d.set_xlim([-max_range, max_range])
                ax_3d.set_ylim([-max_range, max_range])
                ax_3d.set_zlim([-1.1, max_range])
                ax_3d.view_init(elev=20, azim=-60)
                
                # Style the plot - match Streamlit
                ax_3d.tick_params(colors='white')
                ax_3d.xaxis.pane.fill = False
                ax_3d.yaxis.pane.fill = False
                ax_3d.zaxis.pane.fill = False
                
                # Add colorbar - EXACT same as Streamlit
                m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
                m.set_array(pattern_3d)
                cbar = fig_3d.colorbar(m, ax=ax_3d, shrink=0.8, aspect=20)
                cbar.set_label('Gain (dBi)', fontsize=12, color='white')
                cbar.ax.tick_params(colors='white')
                
                # Add text info - EXACT same as Streamlit
                info_text = f"Frequency: {params.frequency_hz/1e9:.2f} GHz\nMax Gain: {np.max(pattern_3d):.1f} dBi\nMin Gain: {np.min(pattern_3d):.1f} dBi"
                ax_3d.text2D(
                    0.02, 0.98, info_text, transform=ax_3d.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                
                fig_3d.tight_layout()
                
                # Add to GUI
                self.pattern_3d_canvas = FigureCanvasTkAgg(fig_3d, self.pattern_3d_frame)
                self.pattern_3d_canvas.get_tk_widget().pack(fill='both', expand=True)
                self.pattern_3d_canvas.draw()
                # Expose for zoom and scroll
                self._p3d_ax = ax_3d
                self._p3d_canvas = self.pattern_3d_canvas
                def _on_scroll_3d(event):
                    try:
                        factor = 0.9 if event.button == 'up' else 1.1
                        xlim = ax_3d.get_xlim(); ylim = ax_3d.get_ylim(); zlim = ax_3d.get_zlim()
                        def _scale(lims):
                            c = 0.5*(lims[0]+lims[1]); r = 0.5*(lims[1]-lims[0]); r *= factor; return (c-r, c+r)
                        ax_3d.set_xlim(_scale(xlim)); ax_3d.set_ylim(_scale(ylim)); ax_3d.set_zlim(_scale(zlim))
                        self.pattern_3d_canvas.draw_idle()
                    except Exception:
                        pass
                fig_3d.canvas.mpl_connect('scroll_event', _on_scroll_3d)
            else:
                print(f"Cannot create 3D plot: insufficient data dimensions")
            
        except Exception as e:
            print(f"Error updating 3D pattern: {e}")
            import traceback
            traceback.print_exc()


class AntennaSimulatorGUI:
    """Main GUI application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_ui()
        self.current_params = None
        self.simulation_thread = None
        
        # Set up proper cleanup when window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_window(self):
        """Configure main window"""
        self.root.title("🛰️ Patch Antenna Simulator")
        # Start full screen (zoomed) and set a reasonable minimum
        try:
            self.root.state('zoomed')  # Windows full-screen style
        except Exception:
            self.root.attributes('-zoomed', True)
        self.root.minsize(1200, 800)
        
        # Configure modern styling
        self.root.configure(bg=ModernStyle.BG_DARK)
        ModernStyle.configure_ttk_style(self.root)
        
        # Icon (if available)
        try:
            # You can add an icon file here
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel (parameters and controls)
        left_panel = ttk.Frame(main_frame, style='Modern.TFrame')
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        
        # Parameters frame
        self.param_frame = ParameterFrame(left_panel, ModernStyle)
        self.param_frame.pack(fill='x', pady=(0, 5))
        
        # Controls frame
        self.control_frame = ControlFrame(left_panel, self.update_geometry, self.run_simulation)
        self.control_frame.pack(fill='x')
        
        # Right panel (plots and log)
        right_panel = ttk.Frame(main_frame, style='Modern.TFrame')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Create plot frame (no log panel)
        self.plot_frame = PlotFrame(right_panel)
        self.plot_frame.pack(fill='both', expand=True)
        # When switching between Single/Multi, refresh as needed
        try:
            self.plot_frame.set_mode_changed_callback(self.on_plot_mode_changed)
        except Exception:
            pass
        
        # Initialize with default geometry
        self.update_geometry()
    
    def update_geometry(self):
        """Update geometry visualization"""
        try:
            self.current_params = self.param_frame.get_parameters()
            self.control_frame.set_status("Updating geometry...")
            
            # Update geometry plot
            solver_type = self.param_frame.vars['solver_type'].get()
            feed_dir = self.param_frame.vars['feed_direction'].get()
            self.plot_frame.update_geometry_plot(self.current_params, solver_type, feed_dir)
            
            self.control_frame.set_status("Geometry updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update geometry: {e}")
            self.control_frame.set_status("Error updating geometry")

    def on_plot_mode_changed(self, mode: str):
        """React to Single/Multi mode changes in PlotFrame."""
        try:
            if mode == 'single':
                # Rebuild the single-antenna geometry view with current params
                if self.current_params is None:
                    self.current_params = self.param_frame.get_parameters()
                solver_type = self.param_frame.vars['solver_type'].get()
                feed_dir = self.param_frame.vars['feed_direction'].get()
                self.plot_frame.update_geometry_plot(self.current_params, solver_type, feed_dir)
            else:
                # Entered multi mode: nothing to do yet (future: disable Run Simulation)
                pass
        except Exception as e:
            print(f"Mode change refresh error: {e}")
    
    def run_simulation(self):
        """Run FDTD simulation in background thread"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Warning", "Simulation is already running!")
            return
        
        if not self.current_params:
            self.update_geometry()
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(target=self._run_simulation_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _run_simulation_thread(self):
        """Background thread for running simulation"""
        import sys
        import io
        import contextlib
        
        try:
            # Update UI
            self.root.after(0, lambda: self.control_frame.set_simulation_running(True))
            self.root.after(0, lambda: self.control_frame.set_status("Probing openEMS..."))
            print("Starting FDTD simulation...")
            
            # Get DLL path and solver type
            dll_path = self.param_frame.vars['dll_path'].get()
            solver_type = self.param_frame.vars['solver_type'].get()
            is_microstrip = "Microstrip" in solver_type
            
            # Import appropriate solver functions
            if is_microstrip:
                from antenna_sim.solver_fdtd_openems_microstrip import probe_openems_microstrip
                probe_func = probe_openems_microstrip
            else:
                from antenna_sim.solver_fdtd_openems_fixed import probe_openems_fixed
                probe_func = probe_openems_fixed
            
            # Probe openEMS
            print(f"Probing openEMS at: {dll_path}")
            probe_result = probe_func(dll_path)
            if not probe_result.ok:
                raise Exception(f"openEMS probe failed: {probe_result.message}")
            
            print("openEMS probe successful")
            # Disable parameter editing while simulation is running
            try:
                self.param_frame.set_params_state('disabled')
            except Exception:
                pass
            self.root.after(0, lambda: self.control_frame.set_status("Preparing simulation..."))
            
            # Prepare simulation
            solver_name = "microstrip" if is_microstrip else "simple"
            print(f"Preparing {solver_name} simulation...")

            # disable parameter edits while running
            try:
                for w in (self.param_frame,):
                    pass
                # Disable key inputs
                for var_name in ['frequency','eps_r','thickness','loss_tan','metal','solver_type','feed_direction','boundary','theta_step','phi_step','norm_mode','dll_path']:
                    try:
                        # find corresponding widget by tracing grid; fallback to global state
                        pass
                    except Exception:
                        pass
            except Exception:
                pass

            if is_microstrip:
                from antenna_sim.solver_fdtd_openems_microstrip import FeedDirection
                if solver_type == "Microstrip Fed (MSL Port, 3D)":
                    from antenna_sim.solver_fdtd_openems_microstrip_3d import prepare_openems_microstrip_patch_3d as prepare_openems_microstrip_patch
                else:
                    from antenna_sim.solver_fdtd_openems_microstrip import prepare_openems_microstrip_patch
                # Get feed direction for microstrip solver
                feed_dir_str = self.param_frame.vars['feed_direction'].get()
                feed_direction = FeedDirection(feed_dir_str)

                prepared = prepare_openems_microstrip_patch(
                    self.current_params,
                    dll_dir=dll_path,
                    feed_direction=feed_direction,
                    boundary=self.param_frame.vars['boundary'].get(),
                    theta_step_deg=self.param_frame.vars['theta_step'].get(),
                    phi_step_deg=self.param_frame.vars['phi_step'].get() if solver_type == "Microstrip Fed (MSL Port, 3D)" else 5.0,
                    verbose=1
                )
            else:
                from antenna_sim.solver_fdtd_openems_fixed import prepare_openems_patch_fixed
                prepared = prepare_openems_patch_fixed(
                    self.current_params,
                    dll_dir=dll_path,
                    verbose=1
                )
            
            if not prepared.ok:
                raise Exception(f"Simulation preparation failed: {prepared.message}")
            
            print("Simulation prepared")
            self.root.after(0, lambda: self.control_frame.set_status("Running FDTD simulation..."))
            
            # Create a comprehensive output capture for both stdout and stderr
            class LogCapture:
                def __init__(self, gui_log_func, original_stream):
                    self.gui_log_func = gui_log_func
                    self.original_stream = original_stream
                    self.buffer = ""
                    
                def write(self, text):
                    # Always write to original stream first
                    self.original_stream.write(text)
                    self.original_stream.flush()
                    
                    # Process text for GUI log
                    if text.strip():
                        # Handle timestep lines specially (they often come with \r)
                        lines = text.replace('\r', '\n').split('\n')
                        for line in lines:
                            if line.strip():
                                self.gui_log_func(line.strip())
                        
                def flush(self):
                    self.original_stream.flush()
            
            # Function to send log messages safely to the real console (avoid recursion)
            def log_to_gui(message):
                try:
                    if len(message) > 3:
                        sys.__stdout__.write(message + "\n")
                        sys.__stdout__.flush()
                except Exception:
                    pass
                
            # Also try to capture console output using Windows-specific method
            def try_capture_console_output():
                """Attempt to capture console output on Windows"""
                try:
                    if os.name == 'nt':  # Windows
                        import ctypes
                        from ctypes import wintypes
                        
                        # This is a simplified attempt - may not work for all cases
                        kernel32 = ctypes.windll.kernel32
                        # Try to redirect console output (experimental)
                        pass
                except:
                    pass  # Fallback to progress messages only
            
            try_capture_console_output()

            print("Running FDTD simulation...")
            print("This may take 30-60 seconds...")
            
            # Run simulation without re-routing stdout to avoid recursion issues
            if is_microstrip:
                    if solver_type == "Microstrip Fed (MSL Port, 3D)":
                        from antenna_sim.solver_fdtd_openems_microstrip_3d import run_prepared_openems_microstrip_3d
                        result = run_prepared_openems_microstrip_3d(
                            prepared,
                            frequency_hz=self.current_params.frequency_hz,
                            verbose=2
                        )
                    else:
                        from antenna_sim.solver_fdtd_openems_microstrip import run_prepared_openems_microstrip
                        result = run_prepared_openems_microstrip(
                            prepared,
                            frequency_hz=self.current_params.frequency_hz,
                            verbose=2
                        )
            else:
                from antenna_sim.solver_fdtd_openems_fixed import run_prepared_openems_fixed
                result = run_prepared_openems_fixed(
                    prepared,
                    frequency_hz=self.current_params.frequency_hz,
                    verbose=2
                )
            
            if not result.ok:
                raise Exception(f"Simulation failed: {result.message}")
            
            print("Simulation completed successfully!")
            
            if not result.ok:
                raise Exception(f"Simulation failed: {result.message}")
            
            print("FDTD simulation completed successfully!")
            print("Processing results...")
            
            # Update plots with results
            self.root.after(0, lambda: self._update_simulation_results(result))
            self.root.after(0, lambda: self.control_frame.set_status("Simulation completed successfully!"))
            print("All processing complete!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", error_msg))
            self.root.after(0, lambda: self.control_frame.set_status(f"Simulation failed: {error_msg[:50]}..."))
        
        finally:
            self.root.after(0, lambda: self.control_frame.set_simulation_running(False))
            # Re-enable parameters
            try:
                self.root.after(0, lambda: self.param_frame.set_params_state('normal'))
            except Exception:
                pass
            # re-enable parameter edits after simulation
            try:
                for w in (self.param_frame,):
                    pass
                # Enable key inputs
                for var_name in ['frequency','eps_r','thickness','loss_tan','metal','solver_type','feed_direction','boundary','theta_step','phi_step','norm_mode','dll_path']:
                    try:
                        # find corresponding widget by tracing grid; fallback to global state
                        pass
                    except Exception:
                        pass
            except Exception:
                pass
    
    def _update_simulation_results(self, result):
        """Update plots with simulation results"""
        try:
            theta = result.theta
            phi = result.phi
            intensity = np.asarray(result.intensity)
            
            # If we have a full 3D grid (theta x phi), show 3D pattern directly
            if intensity.ndim == 2 and phi is not None and theta is not None and intensity.shape == (len(theta), len(phi)):
                self.plot_frame.update_3d_pattern(theta, phi, intensity, self.current_params)
                # Also synthesize E/H plane cuts for 2D tab: phi=0 and phi=90
                try:
                    phi_vals = np.asarray(phi)
                    idx0 = int(np.argmin(np.abs(phi_vals - 0.0)))
                    idx90 = int(np.argmin(np.abs(phi_vals - (np.pi/2))))
                    cuts = np.stack([intensity[:, idx0], intensity[:, idx90]], axis=1)
                    self.plot_frame.update_2d_patterns(theta, cuts)
                except Exception:
                    self.plot_frame.update_2d_patterns(theta, intensity)
            else:
                # Fall back to existing behavior: update 2D cuts first, then 3D interpolation
                self.plot_frame.update_2d_patterns(theta, intensity)
                self.plot_frame.update_3d_pattern(theta, phi, intensity, self.current_params)
            
            # Switch to results tab
            self.plot_frame.notebook.select(1)  # Switch to 2D patterns tab
            
        except Exception as e:
            print(f"Error updating simulation results: {e}")
    
    def on_closing(self):
        """Handle application closure - ensures complete termination"""
        try:
            # Stop any running simulation threads
            if self.simulation_thread and self.simulation_thread.is_alive():
                print("Stopping simulation thread...")
                # Note: We can't forcefully stop threads in Python, but we can clean up
                
            # Destroy all matplotlib figures to prevent memory leaks
            import matplotlib.pyplot as plt
            plt.close('all')
            
            # Destroy the root window
            self.root.quit()
            self.root.destroy()
            
            # Force exit the entire Python process
            import sys
            import os
            print("Application terminated.")
            os._exit(0)  # Force exit - ensures complete termination
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Force exit anyway
            import os
            os._exit(0)
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()


def main():
    """Main entry point"""
    try:
        app = AntennaSimulatorGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        try:
            messagebox.showerror("Application Error", f"Failed to start application: {e}")
        except:
            pass  # GUI might not be available
    finally:
        # Ensure complete termination
        import sys
        import os
        print("Exiting application...")
        os._exit(0)


if __name__ == "__main__":
    main()
