from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

from antenna_sim.models import PatchAntennaParams, Metal, metal_defaults
from antenna_sim.solver_approx import AnalyticalPatchSolver
from antenna_sim.plotting import plot_cross_sections, plot_3d_pattern, draw_patch_top_view, draw_patch_3d_geometry
from antenna_sim.solver_fdtd_openems import probe_openems, prepare_openems_patch, run_prepared_openems
from antenna_sim.solver_fdtd_openems_2d import prepare_openems_patch_2d
# Import fixed solvers
from antenna_sim import (
    probe_openems_fixed, 
    prepare_openems_patch_fixed, 
    run_prepared_openems_fixed
)

st.set_page_config(page_title="Patch Antenna Simulator", layout="wide")

st.title("🛰️ Patch Antenna Simulator")
st.markdown("**Design and analyze rectangular microstrip patch antennas with real-time visualization**")

if "openems_prepared" not in st.session_state:
    st.session_state.openems_prepared = None
if "abort_openems" not in st.session_state:
    st.session_state.abort_openems = False

with st.sidebar:
    st.header("Inputs")
    f_ghz = st.number_input("Frequency (GHz)", value=2.45, min_value=0.1, step=0.01)
    er = st.number_input("Dielectric constant εr", value=4.3, min_value=1.0, step=0.01)
    h_mm = st.number_input("Substrate thickness h (mm)", value=1.6, min_value=0.1, step=0.1)
    L_mm = st.number_input("Patch length L (mm) [optional]", value=0.0, min_value=0.0, step=0.1)
    W_mm = st.number_input("Patch width W (mm) [optional]", value=0.0, min_value=0.0, step=0.1)
    tan_delta = st.number_input("Loss tangent tanδ", value=0.02, min_value=0.0, step=0.001)

    metal_options = {m.value: metal_defaults[m].display() for m in Metal}
    chosen_display = st.selectbox("Metal (conductivity, thickness)", list(metal_options.values()))
    inv = {v: k for k, v in metal_options.items()}
    metal_name = inv[chosen_display]
    thickness_um = st.number_input("Metal thickness (µm)", value=35.0, min_value=0.1, step=1.0)

    st.header("Display Options")
    ui_scale = st.slider("Figure size", 0.5, 2.0, 0.8, 0.1, help="Adjust figure sizes (smaller = more responsive)")
    geometry_view = st.selectbox("Geometry view", ["3D detailed", "Top view"], index=0, help="Choose how to visualize the antenna structure")

    st.header("Simulation")
    run_btn = st.button("🔄 Run Radiation Pattern Analysis (Analytical)", type="primary")

    st.header("openEMS FDTD")
    st.info("Tip: Start with 2D (fast) to sanity-check ports and fields, then run 3D (full).")
    
    with st.expander("🔧 openEMS Troubleshooting", expanded=False):
        st.write("**Current Status:** openEMS C++ engine fails during initialization")
        st.write("**Recommendation:** Use the analytical solver for reliable patch antenna analysis")
        st.write("**Alternative:** Try conda-forge installation: `conda install -c conda-forge openems`")
        
        dll_dir = st.text_input("openEMS DLL folder", value=os.path.abspath("openEMS"), help="Folder containing CSXCAD.dll/openEMS.dll")
        mode = st.selectbox("Mode", ["2D (fast)", "3D (full)"] , index=0)
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            probe_btn = st.button("🔎 Probe API")
        with colp2:
            prep_btn = st.button("⚙️ Prepare openEMS")
        with colp3:
            run_openems_btn = st.button("⚡ Run prepared")
        abort_btn = st.button("🛑 Abort running simulation")
    
    st.header("🔧 Fixed openEMS Solvers")
    st.success("**NEW:** Tutorial-based fixed solvers that should work reliably!")
    st.info("These are rewritten from scratch based on the working Simple_Patch_Antenna.py tutorial.")
    
    use_fixed_solvers = st.checkbox("Use Fixed Solvers (Recommended)", value=True, help="Use the new fixed solvers based on tutorial approach")
    
    if use_fixed_solvers:
        fixed_dll_dir = st.text_input("openEMS DLL folder (Fixed)", value=os.path.abspath("openEMS"), help="Folder containing CSXCAD.dll/openEMS.dll")
        st.info("**Mode**: 3D Tutorial-based (provides both E-plane and H-plane cuts automatically)")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            probe_fixed_btn = st.button("🔎 Probe Fixed API")
        with col_f2:
            prep_fixed_btn = st.button("⚙️ Prepare Fixed")
        with col_f3:
            run_fixed_btn = st.button("⚡ Run Fixed", type="primary")

if abort_btn:
    st.session_state.abort_openems = True

params = PatchAntennaParams.from_user_units(
    frequency_ghz=f_ghz,
    er=er,
    h_mm=h_mm,
    L_mm=None if L_mm == 0 else L_mm,
    W_mm=None if W_mm == 0 else W_mm,
    metal=metal_name,
    loss_tangent=tan_delta,
    metal_thickness_um=thickness_um,
)
solver = AnalyticalPatchSolver(params)

geo_col, perf_col = st.columns([1.2, 0.8])

with geo_col:
    st.subheader("📐 Antenna Geometry")
    if geometry_view == "3D detailed":
        fig_geo = draw_patch_3d_geometry(solver.L_m, solver.W_m, params.h_m, fig_size=(8 * ui_scale, 6 * ui_scale))
        st.pyplot(fig_geo, width='content')
    else:
        fig_geo, ax_geo = plt.subplots(figsize=(6 * ui_scale, 5 * ui_scale))
        draw_patch_top_view(ax_geo, solver.L_m, solver.W_m, params.h_m)
        st.pyplot(fig_geo, width='content')

with perf_col:
    st.subheader("📊 Performance")
    summary = solver.summary()
    st.metric("Length L", f"{summary['L_mm']:.1f} mm")
    st.metric("Width W", f"{summary['W_mm']:.1f} mm")
    st.metric("Peak Directivity", f"{summary['D0_dBi']:.1f} dBi")
    st.metric("Peak Gain", f"{summary['G0_dBi']:.1f} dBi")
    st.metric("Efficiency", f"{summary['efficiency']*100:.1f}%")

if probe_btn:
    with st.spinner("Probing openEMS…"):
        info = probe_openems(dll_dir)
        if info.ok:
            st.success(info.message)
            with st.expander("Detected API (truncated)", expanded=False):
                for k, v in info.api.items():
                    st.write(k)
                    st.code(", ".join(v[:50]))
        else:
            st.error(info.message)

if run_btn:
    st.subheader("📡 Radiation Pattern Analysis (Analytical)")
    st.info(f"📡 **Analysis Frequency:** {f_ghz:.2f} GHz")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_cross_sections(solver, fig_size=(14 * ui_scale, 7 * ui_scale)), width='content')
    with col2:
        st.pyplot(plot_3d_pattern(solver, fig_size=(8 * ui_scale, 8 * ui_scale)), width='content')

if prep_btn:
    with st.spinner("Preparing openEMS scene…"):
        if mode.startswith("2D"):
            prepared = prepare_openems_patch_2d(params, dll_dir=dll_dir, work_dir="openems_out_2d", cleanup=True, verbose=1)
        else:
            prepared = prepare_openems_patch(params, dll_dir=dll_dir, work_dir="openems_out", cleanup=True, verbose=1)
        if not prepared.ok:
            st.error(prepared.message)
        else:
            st.session_state.openems_prepared = prepared
            st.session_state.abort_openems = False
            st.session_state.openems_mode = mode
            st.success(f"Prepared ({mode}). You can now Run prepared or Abort.")

if run_openems_btn:
    if not st.session_state.openems_prepared:
        st.warning("Prepare the simulation first.")
    else:
        st.subheader(f"⚡ FDTD (openEMS – Python) – {st.session_state.get('openems_mode','')} ")
        with st.spinner("Running openEMS… (click Abort to stop after current step)"):
            if st.session_state.abort_openems:
                st.warning("Aborted before start.")
            else:
                r = run_prepared_openems(st.session_state.openems_prepared, frequency_hz=params.frequency_hz, verbose=1)
                if not r.ok:
                    st.error(r.message)
                else:
                    st.success(r.message)
                    th = r.theta
                    ph = r.phi

                    # Normalize intensity shape to (len(th), len(ph))
                    arr = np.asarray(r.intensity)
                    nth, nph = len(th), len(ph)
                    if arr.ndim == 1:
                        if arr.size == nth:
                            arr = np.tile(arr.reshape(nth, 1), (1, nph))
                        elif arr.size == nph:
                            arr = np.tile(arr.reshape(1, nph), (nth, 1))
                        else:
                            arr = arr.reshape(nth, nph)
                    if arr.shape == (nph, nth):
                        arr = arr.T
                    if arr.shape[0] == 1 and nth > 1:
                        arr = np.tile(arr, (nth, 1))
                    if arr.shape[1] == 1 and nph > 1:
                        arr = np.tile(arr, (1, nph))

                    # Cross-sectional cuts: ZX plane (phi=0°) and ZY plane (phi=90°)
                    # Use dBi directly if backend provided it; otherwise normalized dB
                    if getattr(r, 'is_dBi', False):
                        dBi_grid = arr.astype(float)
                    else:
                        arr_norm = arr / max(1e-16, np.max(arr))
                        dBi_grid = 10*np.log10(np.maximum(1e-16, arr_norm))

                    # Find phi indices for ZX plane (phi=0°) and ZY plane (phi=90°)
                    ph_wrapped = (ph + 2*np.pi) % (2*np.pi)
                    def idx_near(val):
                        return int(np.argmin(np.abs(ph_wrapped - val)))
                    zx_idx = idx_near(0.0)  # ZX plane (phi=0°)
                    zy_idx = idx_near(np.pi/2)  # ZY plane (phi=90°)

                    # Extract cuts along theta for these planes
                    G_zx_dBi = dBi_grid[:, zx_idx]  # ZX plane cut
                    G_zy_dBi = dBi_grid[:, zy_idx]  # ZY plane cut

                    # Fixed, readable polar scaling: normalized to peak (0 dB at max),
                    # rmin snapped to 5 dB, clamped to [-40, 0]. Always 5 dB ticks.
                    def normalize_and_bounds(curve):
                        cur = np.asarray(curve, dtype=float)
                        cur = cur - float(np.max(cur))
                        rmin = max(-40.0, 5.0 * np.floor(float(np.min(cur)) / 5.0))
                        rmax = 0.0
                        return cur, rmin, rmax

# === Fixed Solver Button Handlers ===
if 'use_fixed_solvers' in locals() and use_fixed_solvers:
    if probe_fixed_btn:
        with st.spinner("Probing fixed openEMS…"):
            info = probe_openems_fixed(fixed_dll_dir)
            if info.ok:
                st.success(f"✅ Fixed Solver: {info.message}")
                with st.expander("Fixed API (truncated)", expanded=False):
                    for k, v in info.api.items():
                        st.write(k)
                        st.code(", ".join(v[:50]))
            else:
                st.error(f"❌ Fixed Solver: {info.message}")

    if prep_fixed_btn:
        with st.spinner("Preparing fixed openEMS scene…"):
            prepared = prepare_openems_patch_fixed(params, dll_dir=fixed_dll_dir, work_dir="openems_out_fixed", cleanup=True, verbose=1)
            
            if not prepared.ok:
                st.error(f"❌ Fixed Prep: {prepared.message}")
            else:
                st.session_state.openems_prepared_fixed = prepared
                st.success(f"✅ Fixed Prepared (3D Tutorial-based). Ready to run!")

    if run_fixed_btn:
        if not st.session_state.get('openems_prepared_fixed'):
            st.warning("Prepare the fixed simulation first.")
        else:
            st.subheader(f"⚡ Fixed FDTD (3D Tutorial-based)")
            with st.spinner("Running fixed openEMS… (Tutorial approach)"):
                prepared_fixed = st.session_state.openems_prepared_fixed
                r = run_prepared_openems_fixed(prepared_fixed, frequency_hz=params.frequency_hz, verbose=1)
                
                if not r.ok:
                    st.error(f"❌ Fixed Run: {r.message}")
                else:
                    st.success(f"✅ Fixed Success: {r.message}")
                    
                    # Display the results
                    th = r.theta
                    ph = r.phi
                    arr = np.asarray(r.intensity)
                    
                    st.info(f"📡 **Fixed Analysis Frequency:** {params.frequency_hz/1e9:.2f} GHz")
                    
                    # Pattern Analysis - horizontal layout above 2D plots
                    st.write("**Pattern Analysis**")
                    if arr.size > 0:
                        max_gain = float(np.max(arr))
                        min_gain = float(np.min(arr))
                        dynamic_range = max_gain - min_gain
                        
                        # Horizontal metrics layout
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Max Gain", f"{max_gain:.1f} dBi")
                        with col2:
                            st.metric("Min Gain", f"{min_gain:.1f} dBi") 
                        with col3:
                            st.metric("Dynamic Range", f"{dynamic_range:.1f} dB")
                        with col4:
                            # Pattern type analysis
                            if dynamic_range < 3:
                                pattern_type = "⚠️ Too isotropic (check physics)"
                            elif dynamic_range > 20:
                                pattern_type = "✅ Highly directional"
                            else:
                                pattern_type = "✅ Moderately directional"
                            st.metric("Pattern Type", pattern_type)
                    
                    # 2D Cross-sections - full width to match 3D plot size
                    st.write("**Fixed 2D Cross-sections (dBi)**")
                    # Extract E-plane and H-plane cuts for polar plots
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        # Tutorial gives us theta vs [phi=0, phi=90] typically
                        th_deg = np.rad2deg(th)
                        E_plane = arr[:, 0]  # phi=0 (E-plane)
                        H_plane = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]  # phi=90 (H-plane)
                        
                        # Need to extend pattern to full 360° for proper polar display
                        # The current theta goes 0° to 180°, but we need full circle
                        # Mirror the pattern: 0°-180° data becomes 0°-180° and 180°-360°
                        th_full = np.concatenate([th_deg, th_deg[1:] + 180])  # 0° to 360°
                        E_full = np.concatenate([E_plane, E_plane[1:][::-1]])  # Mirror E-plane
                        H_full = np.concatenate([H_plane, H_plane[1:][::-1]])  # Mirror H-plane
                        
                        # Create polar plots - same size as 3D plot (10x8)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20*ui_scale, 10*ui_scale), subplot_kw={'projection': 'polar'})
                        
                        # E-plane (phi=0°) - full 360°
                        ax1.plot(np.deg2rad(th_full), E_full, 'r-', linewidth=3, label='E-plane (phi=0°)')
                        ax1.set_title('E-plane (ZX, phi=0°)', fontsize=14, pad=25)
                        ax1.set_theta_zero_location('N')
                        ax1.set_theta_direction(-1)
                        ax1.grid(True, alpha=0.3)
                        ax1.set_ylim([max(-20, np.min(E_full)-2), np.max(E_full)+2])
                        # Add coordinate labels next to degrees - E-plane cuts through X direction
                        ax1.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                                         ["0°\n(+Z)", "45°", "90°\n(+X)", "135°", "180°\n(-Z)", "225°", "270°\n(-X)", "315°"])
                        
                        # H-plane (phi=90°) - full 360°
                        ax2.plot(np.deg2rad(th_full), H_full, 'b-', linewidth=3, label='H-plane (phi=90°)')
                        ax2.set_title('H-plane (YZ, phi=90°)', fontsize=14, pad=25)
                        ax2.set_theta_zero_location('N')
                        ax2.set_theta_direction(-1)
                        ax2.grid(True, alpha=0.3)
                        ax2.set_ylim([max(-20, np.min(H_full)-2), np.max(H_full)+2])
                        # Add coordinate labels next to degrees - H-plane cuts through Y direction
                        ax2.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], 
                                         ["0°\n(+Z)", "45°", "90°\n(+Y)", "135°", "180°\n(-Z)", "225°", "270°\n(-Y)", "315°"])
                        
                        plt.tight_layout()
                        st.pyplot(fig, width='content')
                    else:
                        st.write(f"Data shape: {arr.shape} - Cannot extract plane cuts")

                    
                    # Show analytical comparison for reference
                    with st.expander("📈 Compare with Analytical Solution", expanded=False):
                        st.write("**Expected patch antenna pattern from theory:**")
                        analytical_fig = plot_cross_sections(solver, fig_size=(12 * ui_scale, 6 * ui_scale))
                        st.pyplot(analytical_fig, width='content')
                        st.write("**Note**: Analytical solution provides theoretical baseline for comparison with FDTD results.")


                    # =========================
                    # 3D Antenna Pattern (Clean, Human-Readable)
                    # =========================
                    st.subheader("🌐 3D Antenna Radiation Pattern")
                    
                    # Get the data from fixed solver results
                    # th, ph are already in radians from the solver
                    # arr is the intensity data (should be 2D: theta x phi)
                    
                    # Create full theta-phi grid for 3D visualization
                    # We have limited phi data (only [0°, 90°]), so let's create a full pattern
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        # We have E-plane (phi=0°) and H-plane (phi=90°) cuts
                        # Create a full 3D pattern by interpolating/extending these cuts
                        
                        th_deg = np.rad2deg(th)  # theta in degrees
                        E_plane_data = arr[:, 0]  # phi=0° cut
                        H_plane_data = arr[:, 1]  # phi=90° cut
                        
                        # Create full phi range for 3D visualization (0° to 360°)
                        phi_full = np.linspace(0, 2*np.pi, 73)  # 73 points = 5° resolution
                        phi_full_deg = np.rad2deg(phi_full)
                        
                        # Create full 3D pattern by interpolating between E and H planes
                        pattern_3d = np.zeros((len(th), len(phi_full)))
                        
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
                        
                        # Convert to spherical coordinates for 3D plotting
                        TH, PH = np.meshgrid(th, phi_full, indexing='ij')
                        
                        # Normalize pattern for radius (linear scale, 0 to 1)
                        pattern_norm = pattern_3d - np.max(pattern_3d)  # Normalize to peak = 0 dB
                        pattern_linear = np.maximum(0.01, 10**(pattern_norm/20))  # Convert to linear, min 0.01
                        
                        # Convert to Cartesian coordinates
                        X = pattern_linear * np.sin(TH) * np.cos(PH)
                        Y = pattern_linear * np.sin(TH) * np.sin(PH) 
                        Z = pattern_linear * np.cos(TH)
                        
                        # Create the 3D plot
                        fig_3d = plt.figure(figsize=(10*ui_scale, 8*ui_scale))
                        ax_3d = fig_3d.add_subplot(111, projection='3d')
                        
                        # Plot the 3D surface with color mapping
                        surf = ax_3d.plot_surface(X, Y, Z, 
                                                facecolors=plt.cm.jet((pattern_3d - np.min(pattern_3d)) / (np.max(pattern_3d) - np.min(pattern_3d))),
                                                linewidth=0, antialiased=True, alpha=0.8)
                        
                        # Add patch antenna at the bottom
                        patch_size = 0.15
                        patch_x = [-patch_size/2, patch_size/2, patch_size/2, -patch_size/2, -patch_size/2]
                        patch_y = [-patch_size/3, -patch_size/3, patch_size/3, patch_size/3, -patch_size/3]
                        patch_z = [-0.9, -0.9, -0.9, -0.9, -0.9]
                        ax_3d.plot(patch_x, patch_y, patch_z, color='orange', linewidth=3, label='Patch Antenna')
                        
                        # Ground plane
                        ground_size = 0.3
                        ax_3d.plot([-ground_size, ground_size], [-ground_size, -ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                        ax_3d.plot([-ground_size, ground_size], [ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                        ax_3d.plot([-ground_size, -ground_size], [-ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                        ax_3d.plot([ground_size, ground_size], [-ground_size, ground_size], [-1.0, -1.0], color='gray', linewidth=2)
                        
                        # Coordinate system arrows
                        ax_3d.quiver(0, 0, -1.0, 0.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='+X')
                        ax_3d.quiver(0, 0, -1.0, 0, 0.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='+Y')
                        ax_3d.quiver(0, 0, -1.0, 0, 0, 0.3, color='blue', arrow_length_ratio=0.1, linewidth=2, label='+Z')
                        
                        # Labels and formatting
                        ax_3d.set_xlabel('X', fontsize=12)
                        ax_3d.set_ylabel('Y', fontsize=12)
                        ax_3d.set_zlabel('Z', fontsize=12)
                        ax_3d.set_title(f'3D Radiation Pattern\nMax Gain: {np.max(pattern_3d):.1f} dBi @ {params.frequency_hz/1e9:.2f} GHz', 
                                       fontsize=14, pad=20)
                        
                        # Set equal aspect ratio and good viewing angle
                        max_range = 1.2
                        ax_3d.set_xlim([-max_range, max_range])
                        ax_3d.set_ylim([-max_range, max_range])
                        ax_3d.set_zlim([-1.1, max_range])
                        ax_3d.view_init(elev=20, azim=-60)
                        
                        # Add colorbar
                        m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
                        m.set_array(pattern_3d)
                        cbar = fig_3d.colorbar(m, ax=ax_3d, shrink=0.8, aspect=20)
                        cbar.set_label('Gain (dBi)', fontsize=12)
                        
                        # Add text info
                        info_text = f"Frequency: {params.frequency_hz/1e9:.2f} GHz\nMax Gain: {np.max(pattern_3d):.1f} dBi\nMin Gain: {np.min(pattern_3d):.1f} dBi"
                        ax_3d.text2D(0.02, 0.98, info_text, transform=ax_3d.transAxes, 
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        plt.tight_layout()
                        st.pyplot(fig_3d, width='content')
                        
                        # Pattern analysis
                        st.write("**3D Pattern Analysis:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Gain", f"{np.max(pattern_3d):.1f} dBi")
                        with col2:
                            st.metric("Min Gain", f"{np.min(pattern_3d):.1f} dBi")
                        with col3:
                            st.metric("Dynamic Range", f"{np.max(pattern_3d) - np.min(pattern_3d):.1f} dB")
                        
                    else:
                        st.error("Cannot create 3D plot: insufficient data dimensions")
