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
        # Use default ttk styling to match Multi Patch Controls look
        super().__init__(parent)
        self.style_class = style_class
        self.setup_ui()
        
    def setup_ui(self):
        # Header (match Multi Patch Controls: bold plain ttk)
        header = ttk.Label(self, text="📐 Single Patch Controls", font=("Segoe UI", 12, "bold"))
        header.pack(fill='x', padx=10, pady=(10, 5))
        # Small view-only badge (only shown while simulation runs)
        self.view_only_label = ttk.Label(self, text="🔒 View-only during run", style='Modern.TLabel')
        
        # Parameters frame
        params_frame = ttk.Frame(self)
        params_frame.pack(fill='both', expand=True, padx=10, pady=5)
        # Keep a reference so we can place the badge just above these controls
        self._params_container = params_frame
        
        # Create parameter inputs
        self.vars = {}
        
        # Row 0: Frequency
        ttk.Label(params_frame, text="Frequency (GHz):").grid(row=0, column=0, sticky='w', pady=2)
        self.vars['frequency'] = tk.DoubleVar(value=2.45)
        freq_entry = ttk.Entry(params_frame, textvariable=self.vars['frequency'], width=15)
        freq_entry.grid(row=0, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 1: Dielectric constant
        ttk.Label(params_frame, text="Dielectric εr:").grid(row=1, column=0, sticky='w', pady=2)
        self.vars['eps_r'] = tk.DoubleVar(value=4.3)
        eps_entry = ttk.Entry(params_frame, textvariable=self.vars['eps_r'], width=15)
        eps_entry.grid(row=1, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 2: Substrate thickness
        ttk.Label(params_frame, text="Thickness h (mm):").grid(row=2, column=0, sticky='w', pady=2)
        self.vars['thickness'] = tk.DoubleVar(value=1.6)
        thick_entry = ttk.Entry(params_frame, textvariable=self.vars['thickness'], width=15)
        thick_entry.grid(row=2, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 3: Loss tangent
        ttk.Label(params_frame, text="Loss tangent:").grid(row=3, column=0, sticky='w', pady=2)
        self.vars['loss_tan'] = tk.DoubleVar(value=0.02)
        loss_entry = ttk.Entry(params_frame, textvariable=self.vars['loss_tan'], width=15)
        loss_entry.grid(row=3, column=1, sticky='ew', padx=(5, 0), pady=2)
        
        # Row 4: Metal selection
        ttk.Label(params_frame, text="Metal:").grid(row=4, column=0, sticky='w', pady=2)
        self.vars['metal'] = tk.StringVar(value=Metal.COPPER.value)
        metal_combo = ttk.Combobox(params_frame, textvariable=self.vars['metal'], width=12)
        metal_combo['values'] = [m.value for m in Metal]
        metal_combo.grid(row=4, column=1, sticky='ew', padx=(5, 0), pady=2)
        metal_combo.state(['readonly'])
        
        # Row 5: Solver type selection
        ttk.Label(params_frame, text="Solver Type:").grid(row=5, column=0, sticky='w', pady=2)
        self.vars['solver_type'] = tk.StringVar(value="Simple (Lumped Port)")
        solver_combo = ttk.Combobox(params_frame, textvariable=self.vars['solver_type'], width=20)
        solver_combo['values'] = ["Simple (Lumped Port)", "Microstrip Fed (MSL Port)", "Microstrip Fed (MSL Port, 3D)"]
        solver_combo.grid(row=5, column=1, sticky='ew', padx=(5, 0), pady=2)
        solver_combo.state(['readonly'])
        solver_combo.bind('<<ComboboxSelected>>', self.on_solver_type_change)
        
        # Row 6: Feed direction (only for microstrip)
        self.feed_dir_label = ttk.Label(params_frame, text="Feed Direction:")
        self.feed_dir_label.grid(row=6, column=0, sticky='w', pady=2)
        self.vars['feed_direction'] = tk.StringVar(value="-X")
        self.feed_dir_combo = ttk.Combobox(params_frame, textvariable=self.vars['feed_direction'], width=12)
        self.feed_dir_combo['values'] = ["-X", "+X", "-Y", "+Y"]
        self.feed_dir_combo.grid(row=6, column=1, sticky='ew', padx=(5, 0), pady=2)
        self.feed_dir_combo.state(['readonly'])
        
        # Row 7: Boundary type
        ttk.Label(params_frame, text="Boundary:").grid(row=7, column=0, sticky='w', pady=2)
        self.vars['boundary'] = tk.StringVar(value="MUR")
        boundary_combo = ttk.Combobox(params_frame, textvariable=self.vars['boundary'], width=12)
        boundary_combo['values'] = ["MUR", "PML_8"]
        boundary_combo.grid(row=7, column=1, sticky='ew', padx=(5,0), pady=2)
        boundary_combo.state(['readonly'])

        # Row 8: Theta/Phi sampling (for 3D)
        ttk.Label(params_frame, text="θ step (deg):").grid(row=8, column=0, sticky='w', pady=2)
        self.vars['theta_step'] = tk.DoubleVar(value=2.0)
        ttk.Entry(params_frame, textvariable=self.vars['theta_step'], width=12).grid(row=8, column=1, sticky='w', padx=(5,0), pady=2)

        ttk.Label(params_frame, text="φ step (deg):").grid(row=9, column=0, sticky='w', pady=2)
        self.vars['phi_step'] = tk.DoubleVar(value=5.0)
        ttk.Entry(params_frame, textvariable=self.vars['phi_step'], width=12).grid(row=9, column=1, sticky='w', padx=(5,0), pady=2)

        # Row 10: Normalization toggle
        ttk.Label(params_frame, text="3D scale:").grid(row=10, column=0, sticky='w', pady=2)
        self.vars['norm_mode'] = tk.StringVar(value="dBi")
        norm_combo = ttk.Combobox(params_frame, textvariable=self.vars['norm_mode'], width=12)
        norm_combo['values'] = ["dBi", "Normalized"]
        norm_combo.grid(row=10, column=1, sticky='ew', padx=(5,0), pady=2)
        norm_combo.state(['readonly'])

        # Row 11: openEMS DLL path
        ttk.Label(params_frame, text="openEMS DLL:").grid(row=11, column=0, sticky='w', pady=2)
        self.vars['dll_path'] = tk.StringVar(value=os.path.abspath("openEMS"))
        dll_frame = ttk.Frame(params_frame)
        dll_frame.grid(row=11, column=1, sticky='ew', padx=(5, 0), pady=2)
        dll_frame.columnconfigure(0, weight=1)
        
        dll_entry = ttk.Entry(dll_frame, textvariable=self.vars['dll_path'])
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
            # Toggle the small view-only badge immediately
            try:
                if state == 'normal':
                    self.view_only_label.pack_forget()
                else:
                    # Place badge above the parameter grid, under the header
                    self.view_only_label.pack_forget()
                    self.view_only_label.pack(before=self._params_container, fill='x', padx=10, pady=(0, 4))
            except Exception:
                pass

            # Iterate all descendants and set sensible states
            for widget in self.winfo_children():
                for sub in widget.winfo_children():
                    try:
                        wclass = sub.winfo_class()
                        # Inputs should be readable but not editable during run -> use 'readonly'
                        if wclass in ('TEntry', 'Entry', 'TCombobox', 'Combobox'):
                            target = state if state in ('normal', 'readonly') else 'readonly' if state == 'disabled' else state
                            sub.configure(state=target)
                        elif wclass in ('TButton', 'Button'):
                            # Buttons inside parameter panel (e.g., DLL browse) disabled while running
                            sub.configure(state=('normal' if state == 'normal' else 'disabled'))
                        else:
                            # Best effort
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
        # Lock hint (shown during simulation)
        self.lock_hint = ttk.Label(self, text="Locked while simulation is running...", style='Modern.TLabel', anchor='center')
        # Will pack/unpack in set_simulation_running

        # Separate Port Diagnostics panel (stays visible while controls are locked)
        try:
            self.port_frame = ttk.LabelFrame(self, text="Port Diagnostics")
            self.port_frame.pack(fill='x', padx=10, pady=(8, 10))
            try:
                # Keep a modest height so the sidebar size stays similar
                self.port_frame.configure(height=120)
                self.port_frame.pack_propagate(False)
            except Exception:
                pass
            inner = ttk.Frame(self.port_frame)
            inner.pack(fill='x', expand=True)
            self.port_text = tk.Text(inner, height=6, width=1, bg="#14181f", fg="#d8dee9",
                                     font=('Consolas', 9), wrap='word')
            try:
                self.port_text.configure(insertbackground='white')
            except Exception:
                pass
            vsb = ttk.Scrollbar(inner, orient='vertical', command=self.port_text.yview)
            self.port_text.configure(yscrollcommand=vsb.set)
            self.port_text.pack(side='left', fill='x', expand=True)
            vsb.pack(side='right', fill='y')
        except Exception:
            pass
    
    def set_status(self, status, color='white'):
        """Update status text"""
        self.status_var.set(status)
    
    def set_simulation_running(self, running):
        """Enable/disable simulation button"""
        if running:
            self.sim_btn.config(state='disabled', text="⏳ Running...")
            try:
                self.geometry_btn.config(state='disabled')
            except Exception:
                pass
            try:
                # Show lock hint near bottom of controls card
                self.lock_hint.pack(fill='x', padx=10, pady=(12, 8))
            except Exception:
                pass
        else:
            self.sim_btn.config(state='normal', text="⚡ Run FDTD Simulation")
            try:
                self.geometry_btn.config(state='normal')
            except Exception:
                pass
            try:
                self.lock_hint.pack_forget()
            except Exception:
                pass

    # ---- Port diagnostics helpers ----
    def clear_port_log(self):
        try:
            if hasattr(self, 'port_text') and self.port_text is not None:
                self.port_text.delete('1.0', 'end')
        except Exception:
            pass

    def append_port_log(self, text: str):
        try:
            if hasattr(self, 'port_text') and self.port_text is not None:
                self.port_text.insert('end', str(text) + '\n')
                self.port_text.see('end')
        except Exception:
            pass


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
        self.vtk_view = None
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

        # Primary Geometry tab (PyVista)
        self.multi_pv_tab = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.multi_pv_tab, text="Geometry")
        self._pv_placeholder = ttk.Label(
            self.multi_pv_tab,
            text="Switch to Multi Antenna mode to enable 3D Geometry view\n(install 'pyvista' if missing)",
            style='Modern.TLabel'
        )
        self._pv_placeholder.pack(expand=True)
        self._multi_pv_added = True

        # Geometry Legacy tab (Matplotlib)
        self.geometry_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.geometry_frame, text="Geometry Legacy")

        # 2D Patterns tab
        self.pattern_2d_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.pattern_2d_frame, text="2D Patterns")
        
        # 3D Pattern tab
        self.pattern_3d_frame = ttk.Frame(self.notebook, style='Modern.TFrame')
        self.notebook.add(self.pattern_3d_frame, text="3D Pattern")
        # PyVista tab is now always present; placeholder shows in Single mode
        # Track visibility state of tabs so we can show/hide per mode
        self._legacy_tab_visible = True
        self._pv_tab_visible = True

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
        # Default mode is 'single' -> hide PyVista 'Geometry' tab
        try:
            if self.mode == 'single':
                # ensure the legacy tab is visible and PyVista hidden
                self._show_legacy_geometry_tab()
                self._hide_pv_geometry_tab()
        except Exception:
            pass
        
    
    # ---- Mode management ----
    def set_mode_switch_enabled(self, enabled: bool):
        """Enable/disable switching between Single and Multi modes.
        Prevents accidental switching during a running simulation."""
        try:
            state = 'normal' if enabled else 'disabled'
            self.btn_single.config(state=state)
            self.btn_multi.config(state=state)
        except Exception:
            pass

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

    # ---- Tab show/hide helpers ----
    def _hide_legacy_geometry_tab(self):
        """Hide the Matplotlib 'Geometry Legacy' tab from the notebook if present."""
        try:
            if self._legacy_tab_visible:
                self.notebook.forget(self.geometry_frame)
                self._legacy_tab_visible = False
        except Exception:
            self._legacy_tab_visible = False

    def _show_legacy_geometry_tab(self):
        """Show the Matplotlib 'Geometry Legacy' tab in the notebook if hidden."""
        try:
            if not self._legacy_tab_visible:
                # Insert after the PyVista 'Geometry' tab (index 0)
                try:
                    self.notebook.insert(1, self.geometry_frame, text="Geometry Legacy")
                except Exception:
                    self.notebook.add(self.geometry_frame, text="Geometry Legacy")
                self._legacy_tab_visible = True
        except Exception:
            pass

    def _hide_pv_geometry_tab(self):
        """Hide the PyVista 'Geometry' tab from the notebook if present (Single mode)."""
        try:
            if self._pv_tab_visible:
                self.notebook.forget(self.multi_pv_tab)
                self._pv_tab_visible = False
        except Exception:
            self._pv_tab_visible = False

    def _show_pv_geometry_tab(self):
        """Show the PyVista 'Geometry' tab in the notebook if hidden (Multi mode)."""
        try:
            if not self._pv_tab_visible:
                # Insert at index 0 so it is the primary Geometry tab
                try:
                    self.notebook.insert(0, self.multi_pv_tab, text="Geometry")
                except Exception:
                    self.notebook.add(self.multi_pv_tab, text="Geometry")
                self._pv_tab_visible = True
        except Exception:
            pass

    def set_mode_single(self):
        if self.mode == 'single':
            return
        # remove multi panel if present (we keep the object only as needed later)
        if self.multi_panel is not None:
            try:
                self.multi_panel.destroy()
            except Exception:
                pass
            self.multi_panel = None
        # teardown PyVista tab content and hide PV tab
        try:
            self._clear_pv_tab()
        except Exception:
            pass
        self._hide_pv_geometry_tab()
        # show legacy geometry tab for Single mode and select it
        self._show_legacy_geometry_tab()
        try:
            self.notebook.select(self.geometry_frame)
        except Exception:
            pass
        # update state and UI
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
        # create MultiPatchPanel for controls/state (do not pack into Geometry tab)
        try:
            from antenna_sim.multi_patch_designer import MultiPatchPanel
            self.multi_panel = MultiPatchPanel(self)
            # Do not pack the panel; the main app will embed its right-side controls in the sidebar
            try:
                self.multi_panel.right.grid_remove()
            except Exception:
                pass
            # Hook PyVista tab to mirror multi-panel changes
            self._init_pv_tab()
            # When main controls change, update PyVista and PV controls list
            def _on_multi_changed(patches):
                # Update PyVista view
                try:
                    self._update_pv_view(patches)
                except Exception:
                    pass
                # Sync PV controls list/selection if present
                try:
                    if getattr(self, 'pv_controls', None) is not None:
                        self.pv_controls.patches = self.multi_panel.patches  # shared list (same object)
                        idx = getattr(self.multi_panel, '_current_index', None)
                        if idx is None:
                            self.pv_controls._refresh_selector()
                        else:
                            self.pv_controls._refresh_selector(select_index=idx)
                except Exception:
                    pass
            try:
                self.multi_panel.set_change_callback(_on_multi_changed)
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Error", f"Failed to open Multi Antenna view: {e}")
            except Exception:
                pass
            return
        # update state and UI
        self.mode = 'multi'
        self._update_mode_buttons()
        self._update_mode_banners()
        # Hide Legacy, show PyVista Geometry tab and select it
        self._hide_legacy_geometry_tab()
        self._show_pv_geometry_tab()
        try:
            self.notebook.select(self.multi_pv_tab)
        except Exception:
            pass
        # show placeholders for patterns while multi-solver isn't wired
        self._show_multi_placeholder('2d')
        self._show_multi_placeholder('3d')
        # Initial PyVista + PV controls sync
        try:
            if self.multi_panel:
                self._update_pv_view(self.multi_panel.patches)
        except Exception:
            pass
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

    # ---- High fidelity PyVista tab helpers ----
    def _init_pv_tab(self):
        # If already initialized, keep it
        if getattr(self, 'pv_view', None) is not None:
            return
        # Clear placeholder
        try:
            if getattr(self, '_pv_placeholder', None) is not None:
                self._pv_placeholder.destroy()
                self._pv_placeholder = None
        except Exception:
            pass
        # Try to create a PyVista view only (controls will be on the left sidebar in multi mode)
        try:
            self.pv_view = PyVistaMultiAntennaView(self.multi_pv_tab)
            # Do not require immediate availability; embedding finishes asynchronously
            try:
                self.pv_view.pack(fill='both', expand=True)
            except Exception:
                pass
        except Exception as e:
            # Show placeholder with instructions
            msg = (
                "3D viewer unavailable.\n"
                "Install with: pip install pyvista pyvistaqt PyQt5\n\n"
                f"Details: {e}"
            )
            self._pv_placeholder = ttk.Label(self.multi_pv_tab, text=msg, style='Modern.TLabel', justify='left')
            self._pv_placeholder.pack(expand=True)
            self.pv_view = None

    def _clear_pv_tab(self):
        # Destroy PyVista child widgets and show placeholder again
        try:
            if self.pv_view is not None:
                self.pv_view.destroy()
        except Exception:
            pass
        self.pv_view = None
        for child in self.multi_pv_tab.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        self._pv_placeholder = ttk.Label(self.multi_pv_tab,
                                          text="Switch to Multi Antenna mode to enable PyVista viewer\n(install 'pyvista' and 'pyvistaqt')",
                                          style='Modern.TLabel')
        self._pv_placeholder.pack(expand=True)

    def _update_pv_view(self, patches):
        try:
            if self.pv_view is not None and self.pv_view.available:
                self.pv_view.rebuild(patches)
        except Exception:
            pass

    def update_geometry_plot(self, params, solver_type: str = "Simple (Lumped Port)", feed_direction_str: str = "-X"):
        """Update the single-antenna geometry visualization (Matplotlib).
        No-op while in multi mode."""
        try:
            # If in multi mode, skip single-antenna drawing
            if getattr(self, 'mode', 'single') == 'multi':
                return
            # Clear existing plot
            if self.geometry_canvas:
                try:
                    self.geometry_canvas.get_tk_widget().destroy()
                except Exception:
                    pass
                self.geometry_canvas = None

            # Calculate patch dimensions if not provided
            from antenna_sim.physics import design_patch_for_frequency
            if params.patch_length_m and params.patch_width_m:
                L_m = params.patch_length_m
                W_m = params.patch_width_m
            else:
                L_m, W_m, _ = design_patch_for_frequency(params.frequency_hz, params.eps_r, params.h_m)

            # Create enhanced geometry plot
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
                # Right-hand mapping: X spans width W, Y spans length L
                if feed_direction == FeedDirection.NEG_X:
                    feed_start = [-sub_W/2, -feed_width_mm/2, z_plane]
                    feed_stop  = [ -W/2,     +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.POS_X:
                    feed_start = [  W/2,     -feed_width_mm/2, z_plane]
                    feed_stop  = [  sub_W/2, +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.NEG_Y:
                    feed_start = [ -feed_width_mm/2, -sub_L/2, z_plane]
                    feed_stop  = [ +feed_width_mm/2,   -L/2,   z_plane]
                else:  # POS_Y
                    feed_start = [ -feed_width_mm/2,   L/2,   z_plane]
                    feed_stop  = [ +feed_width_mm/2,  sub_L/2, z_plane]
                # Draw the microstrip as a small 3D box (prism) for better realism
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                t = max(0.08, 0.06 * h)  # mm
                x0, y0 = feed_start[0], feed_start[1]
                x1, y1 = feed_stop[0], feed_stop[1]
                z0, z1 = z_plane, z_plane + t
                if abs(x1 - x0) > abs(y1 - y0):
                    verts = [
                        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
                        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
                        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
                        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
                    ]
                else:
                    verts = [
                        [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
                        [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]],
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
                # Patch top cap overlay
                patch_thickness = max(0.08, 0.06 * h)
                patch_cap = Poly3DCollection([[[-L/2, -W/2, patch_thickness], [L/2, -W/2, patch_thickness], [L/2, W/2, patch_thickness], [-L/2, W/2, patch_thickness]]],
                                             alpha=1.0, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=0.9)
                try:
                    patch_cap.set_zsort('max')
                except Exception:
                    pass
                patch_cap.set_zorder(12)
                ax_overlay.add_collection3d(patch_cap)
                ax_main = ax_back
            else:
                geometry_fig = draw_patch_3d_geometry(L_m, W_m, params.h_m, fig_size=(8, 6), show_labels=True)
                ax_overlay = None
                ax_main = geometry_fig.gca()

            # Style
            ax = geometry_fig.gca()
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

            # Ensure title is readable on dark theme
            try:
                current_title = ax.get_title()
                if current_title:
                    ax.set_title(current_title, color='white', fontsize=12, pad=20)
            except Exception:
                pass

            geometry_fig.tight_layout()

            # Add to GUI
            self.geometry_canvas = FigureCanvasTkAgg(geometry_fig, self.geometry_frame)
            self.geometry_canvas.get_tk_widget().pack(fill='both', expand=True)
            self.geometry_canvas.draw()
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

            self.geometry_canvas.get_tk_widget().update_idletasks()

        except Exception as e:
            print(f"❌ ERROR updating geometry plot: {e}")
            import traceback
            traceback.print_exc()

    def update_2d_patterns(self, theta, intensity):
        """Update 2D plots."""
        try:
            self._hide_multi_placeholder('2d')
            if self.pattern_2d_canvas:
                self.pattern_2d_canvas.get_tk_widget().destroy()
            if intensity.ndim == 2 and intensity.shape[1] >= 2:
                th_deg = np.rad2deg(theta)
                E_plane = intensity[:, 0]
                H_plane = intensity[:, 1] if intensity.shape[1] > 1 else intensity[:, 0]
                th_full = np.concatenate([th_deg, th_deg[1:] + 180])
                E_full = np.concatenate([E_plane, E_plane[1:][::-1]])
                H_full = np.concatenate([H_plane, H_plane[1:][::-1]])
                th_closed = np.concatenate([th_full, [360.0]])
                E_closed = np.concatenate([E_full, [E_full[0]]])
                H_closed = np.concatenate([H_full, [H_full[0]]])
                fig = Figure(figsize=(16, 8), facecolor='#2b2b2b')
                ax1 = fig.add_subplot(121, projection='polar', facecolor='#2b2b2b')
                ax2 = fig.add_subplot(122, projection='polar', facecolor='#2b2b2b')
                ax1.plot(np.deg2rad(th_closed), E_closed, 'r-', linewidth=3, label='E-plane (phi=0°)')
                ax1.set_title('E-plane (ZX, phi=0°)', fontsize=14, pad=25, color='white')
                ax1.set_theta_zero_location('N'); ax1.set_theta_direction(-1); ax1.grid(True, alpha=0.3)
                ax1.set_ylim([max(-20, np.min(E_full)-2), np.max(E_full)+2]); ax1.tick_params(colors='white')
                ax1.set_thetagrids([0,45,90,135,180,225,270,315], ["0°\n(+Z)","45°","90°\n(+X)","135°","180°\n(-Z)","225°","270°\n(-X)","315°"])
                ax2.plot(np.deg2rad(th_closed), H_closed, 'b-', linewidth=3, label='H-plane (phi=90°)')
                ax2.set_title('H-plane (YZ, phi=90°)', fontsize=14, pad=25, color='white')
                ax2.set_theta_zero_location('N'); ax2.set_theta_direction(-1); ax2.grid(True, alpha=0.3)
                ax2.set_ylim([max(-20, np.min(H_full)-2), np.max(H_full)+2]); ax2.tick_params(colors='white')
                ax2.set_thetagrids([0,45,90,135,180,225,270,315], ["0°\n(+Z)","45°","90°\n(+Y)","135°","180°\n(-Z)","225°","270°\n(-Y)","315°"])
                fig.tight_layout()
                self.pattern_2d_canvas = FigureCanvasTkAgg(fig, self.pattern_2d_frame)
                self.pattern_2d_canvas.get_tk_widget().pack(fill='both', expand=True); self.pattern_2d_canvas.draw()
            else:
                print(f"Data shape: {intensity.shape} - Cannot extract plane cuts")
        except Exception as e:
            print(f"Error updating 2D patterns: {e}")
            import traceback; traceback.print_exc()

    def update_3d_pattern(self, theta, phi, intensity, params, norm_mode: str = 'dBi'):
        """Update 3D radiation pattern plot - EXACT copy of Streamlit version"""
        try:
            self._hide_multi_placeholder('3d')
            if self.pattern_3d_canvas:
                self.pattern_3d_canvas.get_tk_widget().destroy()
            
            # If we have a full theta x phi grid, render it directly
            if (phi is not None) and (intensity.ndim == 2) and (intensity.shape == (len(theta), len(phi))):
                TH, PH = np.meshgrid(theta, phi, indexing='ij')
                patt_abs = np.asarray(intensity, dtype=float)
                # Geometry radius uses normalized relative dB (0 dB at max) for a nice shape
                patt_rel = patt_abs - np.nanmax(patt_abs)
                R = np.maximum(0.01, 10**(patt_rel/20.0))
                X = R * np.sin(TH) * np.cos(PH)
                Y = R * np.sin(TH) * np.sin(PH)
                Z = R * np.cos(TH)
                fig_3d = Figure(figsize=(8, 6), facecolor='#2b2b2b')
                ax_3d = fig_3d.add_subplot(111, projection='3d'); ax_3d.set_facecolor('#2b2b2b')
                # Color by absolute dBi or normalized (dB rel. max) depending on UI
                if isinstance(norm_mode, str) and norm_mode.lower().startswith('norm'):
                    color_vals = patt_rel  # dB rel. max, max at 0 dB
                    vmax = 0.0
                    vmin = float(np.nanpercentile(color_vals, 10))
                    C = np.clip(color_vals, vmin, vmax)
                    cbar_label = 'Gain (dB rel. max)'
                else:
                    color_vals = patt_abs
                    vmin = float(np.nanpercentile(color_vals, 10))
                    vmax = float(np.nanmax(color_vals))
                    C = np.clip(color_vals, vmin, vmax)
                    cbar_label = 'Gain (dBi)'
                norm = (C - vmin) / max(1e-9, (vmax - vmin))
                surf = ax_3d.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(norm), linewidth=0, antialiased=True, alpha=0.95)
                ax_3d.tick_params(colors='white'); ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False
                max_range = 1.2
                ax_3d.set_xlim([-max_range, max_range]); ax_3d.set_ylim([-max_range, max_range]); ax_3d.set_zlim([-1.1, max_range])
                # Default isometric-like orientation (≈45° yaw, slight tilt)
                ax_3d.view_init(elev=30, azim=45)
                m = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
                m.set_array(color_vals)
                try:
                    m.set_clim(vmin, vmax)
                except Exception:
                    pass
                cbar = fig_3d.colorbar(m, ax=ax_3d, shrink=0.85, aspect=24)
                cbar.set_label(cbar_label, fontsize=12, color='white'); cbar.ax.tick_params(colors='white')

                # Origin triad axes (+X,+Y,+Z)
                L = 0.9 * max_range
                ax_3d.plot([0, L], [0, 0], [0, 0], color='red', linewidth=2.2)
                ax_3d.plot([0, 0], [0, L], [0, 0], color='green', linewidth=2.2)
                ax_3d.plot([0, 0], [0, 0], [0, L], color='blue', linewidth=2.2)
                ax_3d.text(L+0.05, 0, 0, '+X', color='red', weight='bold')
                ax_3d.text(0, L+0.05, 0, '+Y', color='green', weight='bold')
                ax_3d.text(0, 0, L+0.05, '+Z', color='blue', weight='bold')

                # Remove axis box/grid but keep the plotted data
                try:
                    ax_3d.grid(False)
                    ax_3d.set_axis_off()
                except Exception:
                    pass

                # Info box (use absolute dBi if available)
                try:
                    gmax = float(np.nanmax(patt_abs)); gmin = float(np.nanmin(patt_abs))
                    info = f"Frequency: {params.frequency_hz/1e9:.2f} GHz\nMax Gain: {gmax:.1f} dBi\nMin Gain: {gmin:.1f} dBi"
                    ax_3d.text2D(0.02, 0.98, info, transform=ax_3d.transAxes, va='top',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
                fig_3d.tight_layout()
                self.pattern_3d_canvas = FigureCanvasTkAgg(fig_3d, self.pattern_3d_frame)
                self.pattern_3d_canvas.get_tk_widget().pack(fill='both', expand=True); self.pattern_3d_canvas.draw()
                self._p3d_ax = ax_3d; self._p3d_canvas = self.pattern_3d_canvas
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
            elif intensity.ndim == 2 and intensity.shape[1] >= 2:
                # We have E-plane (phi=0°) and H-plane (phi=90°) cuts - create a synthetic 3D pattern
                th_deg = np.rad2deg(theta)  # theta in degrees
                E_plane_data = intensity[:, 0]  # phi=0° cut
                H_plane_data = intensity[:, 1]  # phi=90° cut

                # Create full phi range for 3D visualization (0° to 360°)
                phi_full = np.linspace(0, 2*np.pi, 73)  # 73 points = 5° resolution
                pattern_3d = np.zeros((len(theta), len(phi_full)))

                for i, phi_val in enumerate(phi_full):
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
                TH, PH = np.meshgrid(theta, phi_full, indexing='ij')
                pattern_norm = pattern_3d - np.max(pattern_3d)
                pattern_linear = np.maximum(0.01, 10**(pattern_norm/20))
                X = pattern_linear * np.sin(TH) * np.cos(PH)
                Y = pattern_linear * np.sin(TH) * np.sin(PH)
                Z = pattern_linear * np.cos(TH)
                fig_3d = Figure(figsize=(8, 6), facecolor='#2b2b2b')
                ax_3d = fig_3d.add_subplot(111, projection='3d'); ax_3d.set_facecolor('#2b2b2b')
                # Color by absolute dBi or normalized depending on UI
                pattern_rel = pattern_3d - np.max(pattern_3d)
                if isinstance(norm_mode, str) and norm_mode.lower().startswith('norm'):
                    color_vals = pattern_rel
                    vmax = 0.0
                    vmin = float(np.nanpercentile(color_vals, 10))
                    C = np.clip(color_vals, vmin, vmax)
                    cbar_label = 'Gain (dB rel. max)'
                else:
                    color_vals = pattern_3d
                    vmin = float(np.nanpercentile(color_vals, 10))
                    vmax = float(np.nanmax(color_vals))
                    C = np.clip(color_vals, vmin, vmax)
                    cbar_label = 'Gain (dBi)'
                norm = (C - vmin) / max(1e-9, (vmax - vmin))
                surf = ax_3d.plot_surface(
                    X, Y, Z,
                    facecolors=plt.cm.plasma(norm),
                    linewidth=0, antialiased=True, alpha=0.8,
                )
                ax_3d.tick_params(colors='white'); ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False
                max_range = 1.2
                ax_3d.set_xlim([-max_range, max_range]); ax_3d.set_ylim([-max_range, max_range]); ax_3d.set_zlim([-1.1, max_range])
                # Default isometric-like orientation (≈45° yaw, slight tilt)
                ax_3d.view_init(elev=30, azim=45)
                m = plt.cm.ScalarMappable(cmap=plt.cm.plasma); m.set_array(color_vals)
                try:
                    m.set_clim(vmin, vmax)
                except Exception:
                    pass
                cbar = fig_3d.colorbar(m, ax=ax_3d, shrink=0.85, aspect=24)
                cbar.set_label(cbar_label, fontsize=12, color='white'); cbar.ax.tick_params(colors='white')

                # Origin triad axes (+X,+Y,+Z)
                L = 0.9 * max_range
                ax_3d.plot([0, L], [0, 0], [0, 0], color='red', linewidth=2.2)
                ax_3d.plot([0, 0], [0, L], [0, 0], color='green', linewidth=2.2)
                ax_3d.plot([0, 0], [0, 0], [0, L], color='blue', linewidth=2.2)
                ax_3d.text(L+0.05, 0, 0, '+X', color='red', weight='bold')
                ax_3d.text(0, L+0.05, 0, '+Y', color='green', weight='bold')
                ax_3d.text(0, 0, L+0.05, '+Z', color='blue', weight='bold')

                # Remove axis box/grid but keep the plotted data
                try:
                    ax_3d.grid(False)
                    ax_3d.set_axis_off()
                except Exception:
                    pass

                # Info box
                try:
                    gmax = float(np.nanmax(pattern_3d)); gmin = float(np.nanmin(pattern_3d))
                    info = f"Frequency: {params.frequency_hz/1e9:.2f} GHz\nMax Gain: {gmax:.1f} dBi\nMin Gain: {gmin:.1f} dBi"
                    ax_3d.text2D(0.02, 0.98, info, transform=ax_3d.transAxes, va='top',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception:
                    pass
                fig_3d.tight_layout()
                self.pattern_3d_canvas = FigureCanvasTkAgg(fig_3d, self.pattern_3d_frame)
                self.pattern_3d_canvas.get_tk_widget().pack(fill='both', expand=True); self.pattern_3d_canvas.draw()
                self._p3d_ax = ax_3d; self._p3d_canvas = self.pattern_3d_canvas
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
                print("Cannot create 3D plot: insufficient data dimensions")
        except Exception as e:
            print(f"Error updating 3D pattern: {e}")
            import traceback; traceback.print_exc()


# ===== High-fidelity PyVista view (embedded via VTK) =====
class PyVistaMultiAntennaView(ttk.Frame):
    """Embedded high-fidelity 3D view using a PyVista Plotter hosted in a Tk widget.

    Renders inside the tab so you can compare side-by-side with Matplotlib Geometry.
    """

    def __init__(self, parent):
        super().__init__(parent, style='Modern.TFrame')
        self.available = False
        self.error = None
        self._iren_widget = None
        self._camera_snapshot = None
        self._latest_patches = []
        self._pv_plotter = None
        self._plotter = None  # embedded BackgroundPlotter (Qt window reparented into Tk)
        # Dynamic origin axes state (actors)
        self._axis = {
            'x': {'mesh': None, 'actor': None},
            'y': {'mesh': None, 'actor': None},
            'z': {'mesh': None, 'actor': None},
        }
        self._scene_char_len = 0.5  # characteristic length of the scene (updated per rebuild)
        # Optional camera toolbar removed to maximize embedded view area
        self._toolbar = None
        try:
            import os
            # Prefer Desktop OpenGL for stability on Windows
            os.environ.setdefault('QT_OPENGL', 'desktop')
            import pyvista as pv
            from pyvistaqt import QtInteractor, BackgroundPlotter
            import ctypes
            try:
                from PyQt5.QtWidgets import QApplication
                from PyQt5.QtGui import QSurfaceFormat
                from PyQt5.QtCore import Qt
            except Exception:
                try:
                    from PySide2.QtWidgets import QApplication
                    from PySide2.QtGui import QSurfaceFormat
                    from PySide2.QtCore import Qt
                except Exception:
                    QApplication = None
                    QSurfaceFormat = None
                    Qt = None

            # Theme
            try:
                pv.global_theme.background = '#2b2b2b'
                pv.global_theme.foreground = 'white'
                pv.global_theme.edge_color = 'black'
                # Avoid MSAA FBO attachments which can fail on embedded contexts
                pv.global_theme.multi_samples = 0
                # Prefer FXAA
                try:
                    pv.global_theme.anti_aliasing = 'fxaa'
                except Exception:
                    pass
            except Exception:
                pass

            # Host frame inside this tab for embedding the Qt window
            self.host = ttk.Frame(self, style='Modern.TFrame')
            self.host.pack(fill='both', expand=True)
            self.update_idletasks()

            # Default to embedded mode inside the Tk tab (previous behavior).
            # You can force external window by setting environment variable PV_EMBED=0 before launch.
            use_embedded = True
            try:
                use_embedded = os.environ.get('PV_EMBED', '1') == '1'
            except Exception:
                pass

            if not use_embedded:
                # External robust path
                try:
                    self._ext_plotter = BackgroundPlotter(title='Multi Antenna (PyVista)', auto_update=True)
                    # Make sure VTK MSAA is disabled on this window too
                    try:
                        ren_win = getattr(self._ext_plotter, 'ren_win', None) or getattr(self._ext_plotter, 'render_window', None)
                        if ren_win is not None:
                            try: ren_win.SetMultiSamples(0)
                            except Exception: pass
                    except Exception:
                        pass
                    # UI in tab
                    info = ttk.Label(self.host, text=(
                        "The 3D viewer runs in a separate window for maximum stability on Windows.\n"
                        "You can force embedded mode by setting environment variable PV_EMBED=1 (experimental)."),
                        style='Modern.TLabel', justify='left')
                    info.pack(pady=(12,8))
                    btn = ttk.Button(self.host, text='Focus 3D Window', style='Modern.TButton', command=lambda: self._ext_plotter.show())
                    btn.pack()
                    # Wire external plotter as active so rebuild() updates it
                    self._plotter = self._ext_plotter
                    self.available = True
                    # If patches already queued, render them
                    if self._latest_patches:
                        try:
                            self.rebuild(self._latest_patches)
                        except Exception:
                            pass
                    return
                except Exception:
                    # If external creation fails, fall back to embedded path below
                    pass

            # Ensure a Qt application exists BEFORE creating any QWidget
            self._qt_app = None
            try:
                if QApplication is not None:
                    # Set a sane default OpenGL surface format (no MSAA, depth buffer)
                    if QSurfaceFormat is not None:
                        fmt = QSurfaceFormat()
                        try:
                            fmt.setSamples(0)
                        except Exception:
                            pass
                        try:
                            fmt.setDepthBufferSize(24)
                        except Exception:
                            pass
                        try:
                            QSurfaceFormat.setDefaultFormat(fmt)
                        except Exception:
                            pass
                    self._qt_app = QApplication.instance() or QApplication([])
            except Exception:
                pass

            # Create a PyVistaQt QtInteractor (QWidget) but defer showing until sized
            self._plotter = QtInteractor(None)
            try:
                self._plotter.setMinimumSize(100, 100)
                self._plotter.setUpdatesEnabled(False)
                # Ensure frameless look in embedded mode
                if Qt is not None:
                    try:
                        self._plotter.setWindowFlag(Qt.FramelessWindowHint, True)
                    except Exception:
                        pass
            except Exception:
                pass

            # Defer reparent and first show until host has a valid size to avoid 0-height FBO
            def _finalize_embed():
                try:
                    w = int(self.host.winfo_width())
                    h = int(self.host.winfo_height())
                except Exception:
                    w, h = 0, 0
                if w < 50 or h < 50:
                    try:
                        self.after(30, _finalize_embed)
                    except Exception:
                        pass
                    return

                # Reparent Qt widget into this Tk frame (Windows only)
                try:
                    self._hwnd_tk = int(self.host.winfo_id())
                    self._hwnd_qt = int(self._plotter.winId())
                    user32 = ctypes.windll.user32
                    self._user32 = user32
                    # Ensure the Qt window becomes a proper child (minimal style changes)
                    try:
                        GWL_STYLE = -16
                        GWL_EXSTYLE = -20
                        WS_CHILD = 0x40000000
                        WS_VISIBLE = 0x10000000
                        WS_CLIPSIBLINGS = 0x04000000
                        WS_CLIPCHILDREN = 0x02000000
                        WS_POPUP = 0x80000000
                        WS_CAPTION = 0x00C00000
                        WS_THICKFRAME = 0x00040000
                        WS_MINIMIZEBOX = 0x00020000
                        WS_MAXIMIZEBOX = 0x00010000
                        WS_SYSMENU = 0x00080000
                        WS_BORDER = 0x00800000
                        WS_EX_APPWINDOW = 0x00040000
                        WS_EX_WINDOWEDGE = 0x00000100
                        WS_EX_DLGMODALFRAME = 0x00000001
                        try:
                            style = user32.GetWindowLongPtrW(self._hwnd_qt, GWL_STYLE)
                        except Exception:
                            style = user32.GetWindowLongW(self._hwnd_qt, GWL_STYLE)
                        style &= ~(WS_POPUP | WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU | WS_BORDER)
                        style |= (WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN)
                        try:
                            user32.SetWindowLongPtrW(self._hwnd_qt, GWL_STYLE, style)
                        except Exception:
                            user32.SetWindowLongW(self._hwnd_qt, GWL_STYLE, style)
                        # Also clear extended styles that can force a title bar
                        try:
                            exstyle = user32.GetWindowLongPtrW(self._hwnd_qt, GWL_EXSTYLE)
                        except Exception:
                            exstyle = user32.GetWindowLongW(self._hwnd_qt, GWL_EXSTYLE)
                        exstyle &= ~(WS_EX_APPWINDOW | WS_EX_WINDOWEDGE | WS_EX_DLGMODALFRAME)
                        try:
                            user32.SetWindowLongPtrW(self._hwnd_qt, GWL_EXSTYLE, exstyle)
                        except Exception:
                            user32.SetWindowLongW(self._hwnd_qt, GWL_EXSTYLE, exstyle)
                    except Exception:
                        pass
                    user32.SetParent(self._hwnd_qt, self._hwnd_tk)
                    # Apply style changes
                    try:
                        SWP_NOSIZE = 0x0001; SWP_NOMOVE = 0x0002; SWP_NOZORDER = 0x0004; SWP_FRAMECHANGED = 0x0020
                        user32.SetWindowPos(self._hwnd_qt, 0, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED)
                    except Exception:
                        pass
                    # Fit the child to host size (use host size in logical px; let Qt handle DPI)
                    def _resize(ev=None):
                        try:
                            w2 = max(200, int(self.host.winfo_width()))
                            h2 = max(150, int(self.host.winfo_height()))
                            # Use logical TK pixels; embedding uses same coordinate space as host
                            user32.MoveWindow(self._hwnd_qt, 0, 0, int(w2), int(h2), True)
                            try:
                                # Also tell Qt about the new size
                                self._plotter.resize(int(w2), int(h2))
                                self._plotter.update()
                                try:
                                    self._plotter.render()
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        except Exception:
                            pass
                    self.host.bind('<Configure>', _resize)
                    try:
                        self.bind('<Configure>', _resize)
                    except Exception:
                        pass
                    _resize()
                    try:
                        self.after(50, _resize)
                        self.after(250, _resize)
                        self.after(600, _resize)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Show and enable rendering now that we have a real size
                try:
                    self._plotter.show()
                    self._plotter.setUpdatesEnabled(True)
                except Exception:
                    pass

                # Initial axes corner widget
                try:
                    # Make sure VTK itself has MSAA disabled and depth-peeling off
                    try:
                        ren_win = getattr(self._plotter, 'ren_win', None)
                        if ren_win is None:
                            ren_win = self._plotter.render_window  # older attr name
                        if ren_win is not None:
                            try:
                                ren_win.SetMultiSamples(0)
                            except Exception:
                                pass
                            try:
                                ren = self._plotter.renderer
                                ren.SetUseDepthPeeling(False)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Clickable camera orientation widget (top-right)
                    try:
                        if hasattr(self._plotter, 'add_camera_orientation_widget'):
                            self._plotter.add_camera_orientation_widget()
                    except Exception:
                        pass
                    # Disable post-process AA entirely to avoid FBO usage
                    try:
                        if hasattr(self._plotter, 'disable_anti_aliasing'):
                            self._plotter.disable_anti_aliasing()
                    except Exception:
                        pass
                except Exception:
                    pass

                # Periodically pump Qt events so the embedded window stays responsive
                def _pump_qt():
                    try:
                        if QApplication is not None and self._qt_app is not None:
                            self._qt_app.processEvents()
                    except Exception:
                        pass
                    try:
                        self.after(16, _pump_qt)
                    except Exception:
                        pass
                _pump_qt()

                self.available = True
                # First render (even if empty) to draw axes and set camera
                try:
                    self.rebuild(self._latest_patches)
                except Exception:
                    pass
                # Install camera callbacks to keep origin axes scaled with zoom
                try:
                    self._install_camera_callbacks()
                except Exception:
                    pass

            _finalize_embed()
        except Exception as e:
            # Robust fallback: open external window with BackgroundPlotter
            self.error = str(e)
            self.available = False
            try:
                self._ext_plotter = BackgroundPlotter(title='Multi Antenna (PyVista)', auto_update=True)
            except Exception:
                self._ext_plotter = None
            # Provide a focus button in the tab
            fallback = ttk.Frame(self, style='Modern.TFrame')
            msg = ttk.Label(fallback, text=(
                "Embedded 3D viewer failed to initialize (falling back to a separate window).\n"
                f"Details: {e}\n\nClick the button below to focus the external 3D window."),
                style='Modern.TLabel', justify='left')
            msg.pack(pady=(12,8))
            def _focus_ext():
                try:
                    if getattr(self, '_ext_plotter', None) is not None:
                        self._ext_plotter.show()
                except Exception:
                    pass
            ttk.Button(fallback, text='Focus 3D Window', command=_focus_ext, style='Modern.TButton').pack()
            fallback.pack(fill='both', expand=True)

    # ---- Utilities ----
    @staticmethod
    def _rot_matrix(rx_deg: float, ry_deg: float, rz_deg: float):
        rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    # ---- External PyVista fallback ----
    def _open_or_focus_external(self):
        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
            if self._plotter is None:
                self._plotter = BackgroundPlotter(title='Multi Antenna (PyVista)', auto_update=True)
            else:
                try:
                    self._plotter.show()
                except Exception:
                    pass
            # Rebuild with latest patches
            self.rebuild(self._latest_patches)
        except Exception:
            pass

    def _pv_box(self, plotter, dims_xyz, center_world, angles_xyz_deg, color_rgb=(1,1,1), opacity=1.0, show_edges=False):
        try:
            import pyvista as pv
            # Build oriented box corners exactly like MultiPatchPanel
            W, L, H = float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])
            hx, hy, hz = W/2.0, L/2.0, H/2.0
            local = np.array([
                [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [ -hx,  hy, -hz],
                [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [ -hx,  hy,  hz],
            ])
            rx, ry, rz = angles_xyz_deg
            R = self._rot_matrix(rx, ry, rz)
            corners = (local @ R) + np.array(center_world)
            # Faces connectivity (6 quads)
            faces = np.hstack([
                [4, 0,1,2,3],
                [4, 4,5,6,7],
                [4, 0,1,5,4],
                [4, 3,2,6,7],
                [4, 0,3,7,4],
                [4, 1,2,6,5],
            ]).astype(np.int64)
            mesh = pv.PolyData(corners, faces)
            plotter.add_mesh(mesh, color=color_rgb, opacity=float(opacity), show_edges=show_edges)
        except Exception:
            pass

    def _build_scene_pyvista(self, plotter, patches):
        try:
            plotter.clear()
        except Exception:
            pass
        # Ensure the small bottom-left axes actor is removed/disabled
        try:
            if hasattr(plotter, 'show_axes'):
                try:
                    plotter.show_axes(False)
                except Exception:
                    pass
            a = getattr(plotter, 'axes_actor', None)
            if a is not None:
                try:
                    plotter.remove_actor(a)
                except Exception:
                    pass
                try:
                    # Fallback remove through renderer
                    plotter.renderer.RemoveActor(a)
                except Exception:
                    pass
                try:
                    plotter.axes_actor = None
                except Exception:
                    pass
        except Exception:
            pass
        # Reset dynamic axes state on a fresh scene
        self._axis = {
            'x': {'mesh': None, 'actor': None},
            'y': {'mesh': None, 'actor': None},
            'z': {'mesh': None, 'actor': None},
        }
        # Determine characteristic scene length from patches (used for axis scaling)
        try:
            dims = []
            for inst in patches or []:
                if inst.params.patch_length_m and inst.params.patch_width_m:
                    dims.append(max(inst.params.patch_length_m, inst.params.patch_width_m))
                else:
                    from antenna_sim.physics import design_patch_for_frequency
                    L_m, W_m, _ = design_patch_for_frequency(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
                    dims.append(max(L_m, W_m))
            if dims:
                self._scene_char_len = max(0.5, max(dims))
            else:
                self._scene_char_len = 0.5
        except Exception:
            self._scene_char_len = 0.5
        color_sub = (0.23, 0.65, 0.43)
        color_gnd = (0.72, 0.45, 0.20)
        color_patch = (1.0, 0.83, 0.30)
        color_feed = (1.0, 0.43, 0.24)
        for inst in patches or []:
            try:
                if inst.params.patch_length_m and inst.params.patch_width_m:
                    L_m = inst.params.patch_length_m; W_m = inst.params.patch_width_m
                else:
                    from antenna_sim.physics import design_patch_for_frequency
                    L_m, W_m, _ = design_patch_for_frequency(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
                t_patch = max(35e-6, 0.5e-4)
                t_ground = 35e-6
                h = inst.params.h_m
                C = np.array([inst.center_x_m, inst.center_y_m, inst.center_z_m])
                angles = (inst.rot_x_deg, inst.rot_y_deg, inst.rot_z_deg)
                R = self._rot_matrix(*angles)
                margin = 0.35 * max(L_m, W_m)
                sub_L = L_m + 2*margin
                sub_W = W_m + 2*margin
                sub_center = C + (np.array([0,0,-(t_patch/2 + h/2)]) @ R)
                self._pv_box(plotter, (sub_W, sub_L, h), sub_center, angles, color_rgb=color_sub, opacity=1.0, show_edges=True)
                gnd_center = C + (np.array([0,0,-(t_patch/2 + h + t_ground/2)]) @ R)
                self._pv_box(plotter, (sub_W, sub_L, t_ground), gnd_center, angles, color_rgb=color_gnd, opacity=0.98, show_edges=False)
                self._pv_box(plotter, (W_m, L_m, t_patch), C, angles, color_rgb=color_patch, opacity=1.0, show_edges=False)
                from antenna_sim.solver_fdtd_openems_microstrip import calculate_microstrip_width, FeedDirection
                fw = calculate_microstrip_width(inst.params.frequency_hz, inst.params.eps_r, inst.params.h_m)
                length = 3*fw
                if inst.feed_direction == FeedDirection.NEG_X:
                    local_center = np.array([-(W_m/2 + length/2), 0.0, 0.0]); dims = (length, fw, t_patch)
                elif inst.feed_direction == FeedDirection.POS_X:
                    local_center = np.array([(W_m/2 + length/2), 0.0, 0.0]); dims = (length, fw, t_patch)
                elif inst.feed_direction == FeedDirection.NEG_Y:
                    local_center = np.array([0.0, -(L_m/2 + length/2), 0.0]); dims = (fw, length, t_patch)
                else:
                    local_center = np.array([0.0, (L_m/2 + length/2), 0.0]); dims = (fw, length, t_patch)
                feed_center = C + (local_center @ R)
                self._pv_box(plotter, dims, feed_center, angles, color_rgb=color_feed, opacity=0.98, show_edges=False)
            except Exception:
                pass
        try:
            plotter.reset_camera()
        except Exception:
            pass
        # Build/update dynamic origin axes now that the camera is valid
        try:
            self._rescale_origin_axes(initial_build=True)
        except Exception:
            pass

    def rebuild(self, patches):
        self._latest_patches = patches
        if not self.available or self._plotter is None:
            return
        # Save camera
        try:
            self._camera_snapshot = tuple(self._plotter.camera_position)
        except Exception:
            self._camera_snapshot = None

        # Build scene with PyVista
        try:
            self._build_scene_pyvista(self._plotter, patches)
        except Exception:
            pass

        # Restore camera
        try:
            if self._camera_snapshot is None:
                self._plotter.reset_camera()
            else:
                self._plotter.camera_position = self._camera_snapshot
        except Exception:
            pass

        # Render
        try:
            self._plotter.render()
        except Exception:
            pass

    def destroy(self):
        try:
            if self._iren_widget is not None:
                self._iren_widget.Destroy()
        except Exception:
            pass
        super().destroy()

    # ---- Camera/axes helpers ----
    def _install_camera_callbacks(self):
        iren = getattr(self._plotter, 'iren', None)
        if iren is None:
            return
        def _evt(obj=None, evt:str=None):
            try:
                # Slight debounce by scheduling on Tk loop
                self.after(0, self._rescale_origin_axes)
            except Exception:
                pass
        try:
            for ev in ['EndInteractionEvent', 'MouseWheelForwardEvent', 'MouseWheelBackwardEvent', 'InteractionEvent']:
                try:
                    iren.AddObserver(ev, _evt)
                except Exception:
                    pass
        except Exception:
            pass

    def _compute_axis_len(self):
        try:
            cam = self._plotter.camera
            import math
            fp = np.array(cam.focal_point)
            pos = np.array(cam.position)
            dist = float(np.linalg.norm(pos - fp))
            va = float(getattr(cam, 'view_angle', 30.0))
            height = 2.0 * dist * math.tan(math.radians(va * 0.5))
            # Use 55% of current view height, with a generous minimum based on scene size
            return max(1.1 * self._scene_char_len, 0.55 * height)
        except Exception:
            return max(1.1 * self._scene_char_len, 0.8)

    def _rescale_origin_axes(self, initial_build=False):
        if not self.available or self._plotter is None:
            return
        try:
            import pyvista as pv
            L = float(self._compute_axis_len())
            # Create or update line meshes
            def _ensure(axis_key, p0, p1, color):
                rec = self._axis[axis_key]
                if rec['mesh'] is None:
                    rec['mesh'] = pv.Line(p0, p1)
                    rec['actor'] = self._plotter.add_mesh(rec['mesh'], color=color, line_width=6)
                else:
                    rec['mesh'].points[:] = np.array([p0, p1])
            _ensure('x', (-L,0,0), (L,0,0), (1.0,0.3,0.2))
            _ensure('y', (0,-L,0), (0,L,0), (0.2,1.0,0.3))
            _ensure('z', (0,0,-L), (0,0,L), (0.3,0.5,1.0))
            # No labels; keep the view clean since the orientation widget labels axes
            # Render
            if not initial_build:
                try:
                    self._plotter.render()
                except Exception:
                    pass
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
                    feed_start = [-sub_W/2, -feed_width_mm/2, z_plane]
                    feed_stop  = [ -W/2,     +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.POS_X:
                    feed_start = [  W/2,     -feed_width_mm/2, z_plane]
                    feed_stop  = [  sub_W/2, +feed_width_mm/2, z_plane]
                elif feed_direction == FeedDirection.NEG_Y:
                    feed_start = [-feed_width_mm/2, -sub_L/2, z_plane]
                    feed_stop  = [ +feed_width_mm/2,   -L/2,   z_plane]
                else:  # POS_Y
                    feed_start = [-feed_width_mm/2,   L/2,   z_plane]
                    feed_stop  = [ +feed_width_mm/2,  sub_L/2, z_plane]
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
            # In either mode, hide placeholder and render results
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
                ax1.set_thetagrids([0,45,90,135,180,225,270,315], 
                                 ["0°\n(+Z)","45°","90°\n(+X)","135°","180°\n(-Z)","225°","270°\n(-X)","315°"])
                ax1.tick_params(colors='white')
                
                # H-plane (phi=90°) - full 360° - EXACT same as Streamlit
                ax2.plot(np.deg2rad(th_closed), H_closed, 'b-', linewidth=3, label='H-plane (phi=90°)')
                ax2.set_title('H-plane (YZ, phi=90°)', fontsize=14, pad=25, color='white')
                ax2.set_theta_zero_location('N')
                ax2.set_theta_direction(-1)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([max(-20, np.min(H_full)-2), np.max(H_full)+2])
                # Add coordinate labels next to degrees - H-plane cuts through Y direction
                ax2.set_thetagrids([0,45,90,135,180,225,270,315], 
                                 ["0°\n(+Z)","45°","90°\n(+Y)","135°","180°\n(-Z)","225°","270°\n(-Y)","315°"])
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
            # In either mode, hide placeholder and render results
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
        self._sidebar_overlay = None
        # cache last simulation results for quick re-render of 3D coloring
        self._last_theta = None
        self._last_phi = None
        self._last_intensity = None
        self._last_params = None
        
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
        self.left_panel = left_panel
        # Host for Multi Patch Controls when in multi mode (created lazily)
        self.multi_controls_host = None
        
        # Controls frame (place on top)
        self.control_frame = ControlFrame(left_panel, self.update_geometry, self.run_simulation)
        self.control_frame.pack(fill='x')
        
        # Single Patch Controls under the FDTD controls
        self.param_frame = ParameterFrame(left_panel, ModernStyle)
        self.param_frame.pack(fill='x', pady=(0, 5))
        # When 3D scale changes, re-render the 3D plot using cached data
        try:
            self.param_frame.vars['norm_mode'].trace_add('write', lambda *args: self._on_norm_mode_change())
        except Exception:
            pass
        
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

    # --- Sidebar overlay helpers ---
    def _show_sidebar_overlay(self, text: str = "Locked while simulation is running..."):
        try:
            if self._sidebar_overlay is not None:
                return
            # Create a canvas overlay that intercepts clicks and draws a stippled dimmer
            overlay = tk.Canvas(self.left_panel, highlightthickness=0, bg='')
            overlay.place(x=0, y=0, relwidth=1, relheight=1)
            w = self.left_panel.winfo_width() or 200
            h = self.left_panel.winfo_height() or 400
            # Draw dimmer rectangle using stipple to simulate transparency
            try:
                overlay.create_rectangle(0, 0, w, h, fill='#000000', stipple='gray25', outline='')
            except Exception:
                overlay.create_rectangle(0, 0, w, h, fill='#222222', outline='')
            # Centered text
            overlay.create_text(w//2, h//2, text=text, fill='white', font=("Segoe UI", 10, "bold"))
            # Keep it updated on resize
            def _on_resize(event):
                overlay.delete('all')
                overlay.create_rectangle(0, 0, event.width, event.height, fill='#000000', stipple='gray25', outline='')
                overlay.create_text(event.width//2, event.height//2, text=text, fill='white', font=("Segoe UI", 10, "bold"))
            self.left_panel.bind('<Configure>', _on_resize)
            self._sidebar_overlay = overlay
        except Exception:
            self._sidebar_overlay = None

    def _hide_sidebar_overlay(self):
        try:
            if self._sidebar_overlay is not None:
                self._sidebar_overlay.destroy()
                self._sidebar_overlay = None
        except Exception:
            self._sidebar_overlay = None
    
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
                # Restore Single Patch Controls on the left
                try:
                    if self.multi_controls_host is not None:
                        for w in list(self.multi_controls_host.winfo_children()):
                            try:
                                w.destroy()
                            except Exception:
                                pass
                        try:
                            self.multi_controls_host.pack_forget()
                        except Exception:
                            pass
                        self.multi_controls_host.destroy()
                        self.multi_controls_host = None
                except Exception:
                    pass
                try:
                    self.param_frame.pack_forget()
                    self.param_frame.pack(fill='x', pady=(0, 5))
                except Exception:
                    pass
            else:
                # Entered multi mode: nothing to do yet (future: disable Run Simulation)
                # Hide Single Patch Controls; create Multi Patch Controls host on left
                try:
                    self.param_frame.pack_forget()
                except Exception:
                    pass
                if self.multi_controls_host is None:
                    try:
                        self.multi_controls_host = ttk.Frame(self.left_panel, style='Card.TFrame')
                        self.multi_controls_host.pack(fill='x', pady=(0,5))
                    except Exception:
                        self.multi_controls_host = None
                # Build Multi Patch Controls into the left sidebar host using the active MultiPatchPanel
                try:
                    if getattr(self.plot_frame, 'multi_panel', None) is not None:
                        host = self.multi_controls_host or self.left_panel
                        for w in list(host.winfo_children()):
                            try:
                                w.destroy()
                            except Exception:
                                pass
                        self.plot_frame.multi_panel._build_right_controls(host)
                except Exception:
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
            # Show sidebar overlay to lock the whole left panel
            self.root.after(0, lambda: self._show_sidebar_overlay())
            # Prevent switching between Single/Multi while running
            self.root.after(0, lambda: self.plot_frame.set_mode_switch_enabled(False))
            print("Starting FDTD simulation...")
            
            # Get DLL path and solver type
            dll_path = self.param_frame.vars['dll_path'].get()
            solver_type = self.param_frame.vars['solver_type'].get()
            is_microstrip = "Microstrip" in solver_type
            is_multi = getattr(self.plot_frame, 'mode', 'single') == 'multi'
            
            # Import appropriate probe function
            if is_multi or is_microstrip:
                from antenna_sim.solver_fdtd_openems_microstrip import probe_openems_microstrip as probe_func
            else:
                from antenna_sim.solver_fdtd_openems_fixed import probe_openems_fixed as probe_func
            
            # Probe openEMS
            print(f"Probing openEMS at: {dll_path}")
            probe_result = probe_func(dll_path)
            if not probe_result.ok:
                raise Exception(f"openEMS probe failed: {probe_result.message}")
            
            print("openEMS probe successful")
            # Disable parameter editing while simulation is running (schedule on UI thread)
            try:
                self.root.after(0, lambda: self.param_frame.set_params_state('disabled'))
            except Exception:
                pass
            # In Multi mode, lock the Multi Patch Controls panel as well
            try:
                if is_multi and getattr(self.plot_frame, 'multi_panel', None) is not None:
                    self.root.after(0, lambda: self.plot_frame.multi_panel.lock_controls())
            except Exception:
                pass
            self.root.after(0, lambda: self.control_frame.set_status("Preparing simulation..."))
            
            # Prepare simulation
            if is_multi:
                print("Preparing multi-antenna microstrip 3D simulation...")
            else:
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

            if is_multi:
                # Multi-antenna always uses microstrip 3D variant
                try:
                    patches = list(getattr(self.plot_frame.multi_panel, 'patches', []) or [])
                except Exception:
                    patches = []
                if not patches:
                    raise Exception("No antennas in Multi view. Add at least one patch before running FDTD.")
                from antenna_sim.solver_fdtd_openems_microstrip_multi_3d import prepare_openems_microstrip_multi_3d
                # Prefer theta/phi step from the multi panel if present
                try:
                    theta_step = float(getattr(self.plot_frame.multi_panel, 'var_theta_step', None).get())
                    phi_step = float(getattr(self.plot_frame.multi_panel, 'var_phi_step', None).get())
                    # Mesh quality: parse combobox selection like "3 - Medium"
                    mq = 3
                    try:
                        mq_sel = getattr(self.plot_frame.multi_panel, 'mesh_combo', None)
                        if mq_sel is not None:
                            sel_txt = mq_sel.get().strip()
                            if sel_txt:
                                mq = int(sel_txt.split('-',1)[0].strip())
                    except Exception:
                        mq = 3
                except Exception:
                    theta_step = self.param_frame.vars['theta_step'].get()
                    phi_step = self.param_frame.vars['phi_step'].get()
                    mq = 3
                # Clear the bottom Port Diagnostics panel before preparing a new model
                try:
                    if self.control_frame is not None:
                        self.root.after(0, self.control_frame.clear_port_log)
                except Exception:
                    pass
                # Define a log callback that appends to the bottom Port Diagnostics panel
                def _port_log_cb(msg: str):
                    try:
                        if self.control_frame is not None:
                            self.root.after(0, lambda m=str(msg): self.control_frame.append_port_log(m))
                    except Exception:
                        try:
                            print(str(msg))
                        except Exception:
                            pass

                prepared = prepare_openems_microstrip_multi_3d(
                    patches,
                    dll_dir=dll_path,
                    boundary=self.param_frame.vars['boundary'].get(),
                    theta_step_deg=theta_step,
                    phi_step_deg=phi_step,
                    mesh_quality=mq,
                    nf_center_mode=(getattr(self.plot_frame.multi_panel, 'nf_center_combo', None).get().lower() if getattr(self.plot_frame.multi_panel, 'nf_center_combo', None) else 'origin'),
                    simbox_mode=(getattr(self.plot_frame.multi_panel, 'simbox_mode_combo', None).get().lower() if getattr(self.plot_frame.multi_panel, 'simbox_mode_combo', None) else 'auto'),
                    auto_margin_mm=(
                        float(getattr(self.plot_frame.multi_panel, 'var_margin_x', None).get()) if getattr(self.plot_frame.multi_panel, 'var_margin_x', None) else 80.0,
                        float(getattr(self.plot_frame.multi_panel, 'var_margin_y', None).get()) if getattr(self.plot_frame.multi_panel, 'var_margin_y', None) else 80.0,
                        float(getattr(self.plot_frame.multi_panel, 'var_margin_z', None).get()) if getattr(self.plot_frame.multi_panel, 'var_margin_z', None) else 160.0,
                    ),
                    manual_size_mm=(
                        (
                            float(getattr(self.plot_frame.multi_panel, 'var_box_x', None).get()),
                            float(getattr(self.plot_frame.multi_panel, 'var_box_y', None).get()),
                            float(getattr(self.plot_frame.multi_panel, 'var_box_z', None).get()),
                        ) if (getattr(self.plot_frame.multi_panel, 'simbox_mode_combo', None) and getattr(self.plot_frame.multi_panel, 'simbox_mode_combo', None).get().lower().startswith('man')) else None
                    ),
                    port_mode='lumped',  # Restore stable behavior: Lumped ports for all elements
                    verbose=1,
                    log_cb=_port_log_cb,
                )
            elif is_microstrip:
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
            if is_multi:
                from antenna_sim.solver_fdtd_openems_microstrip_multi_3d import run_prepared_openems_microstrip_multi_3d
                try:
                    patches = list(getattr(self.plot_frame.multi_panel, 'patches', []) or [])
                    f_multi = patches[0].params.frequency_hz if patches else self.current_params.frequency_hz
                except Exception:
                    f_multi = self.current_params.frequency_hz
                result = run_prepared_openems_microstrip_multi_3d(
                    prepared,
                    frequency_hz=f_multi,
                    verbose=2
                )
            elif is_microstrip:
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
            # Re-enable Single/Multi switching
            self.root.after(0, lambda: self.plot_frame.set_mode_switch_enabled(True))
            # Remove sidebar overlay
            self.root.after(0, self._hide_sidebar_overlay)
            # Re-enable parameters
            try:
                self.root.after(0, lambda: self.param_frame.set_params_state('normal'))
            except Exception:
                pass
            # Unlock Multi Patch Controls if present
            try:
                if getattr(self.plot_frame, 'mode', 'single') == 'multi' and getattr(self.plot_frame, 'multi_panel', None) is not None:
                    self.root.after(0, lambda: self.plot_frame.multi_panel.unlock_controls())
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
                self.plot_frame.update_3d_pattern(theta, phi, intensity, self.current_params, norm_mode=self.param_frame.vars['norm_mode'].get())
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
                self.plot_frame.update_3d_pattern(theta, phi, intensity, self.current_params, norm_mode=self.param_frame.vars['norm_mode'].get())
            
            # Switch to results tab
            # Switch to 2D Patterns tab explicitly by widget
            try:
                self.plot_frame.notebook.select(self.plot_frame.pattern_2d_frame)
            except Exception:
                pass
            
            # cache last
            self._last_theta = theta
            self._last_phi = phi
            self._last_intensity = intensity
            self._last_params = self.current_params
            
        except Exception as e:
            print(f"Error updating simulation results: {e}")

    def _on_norm_mode_change(self):
        try:
            if self._last_theta is None or self._last_intensity is None:
                return
            mode = self.param_frame.vars['norm_mode'].get()
            self.plot_frame.update_3d_pattern(self._last_theta, self._last_phi, self._last_intensity, self._last_params or self.current_params, norm_mode=mode)
        except Exception as e:
            print(f"norm_mode change error: {e}")
    
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
