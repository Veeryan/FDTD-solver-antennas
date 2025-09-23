"""
Microstrip-Fed Patch Antenna FDTD Solver using openEMS

This solver creates a realistic patch antenna with microstrip feed line,
similar to what you'd find on a PCB. Features:

- Full 3D simulation (no symmetry assumptions)
- Proper 50Ω microstrip transmission line feed
- Four feed direction options (+X, -X, +Y, -Y)
- Realistic PCB substrate and ground plane geometry
- MSL (Microstrip Line) port for accurate impedance matching

Based on openEMS tutorials, particularly Simple_Patch_Antenna.py
"""

import os
import shutil
import time
import random
import glob
import numpy as np
from typing import Optional, Literal
from enum import Enum

from .models import PatchAntennaParams
from .solver_fdtd_openems_fixed import OpenEMSPrepared, OpenEMSResult


class FeedDirection(str, Enum):
    """Feed direction options for microstrip line"""
    POS_X = "+X"  # Feed from positive X direction
    NEG_X = "-X"  # Feed from negative X direction  
    POS_Y = "+Y"  # Feed from positive Y direction
    NEG_Y = "-Y"  # Feed from negative Y direction


def _coerce_win_path(p: str) -> str:
    """Coerce a path that may come from Git Bash (/c/Users/...) or /openEMS to a Windows absolute path."""
    if not p:
        return p
    # Already absolute Windows path
    drive, _ = os.path.splitdrive(p)
    if drive:
        return os.path.normpath(p)
    # MSYS style /c/... → C:\...
    if p.startswith("/") and len(p) > 3 and p[2] == "/" and p[1].isalpha():
        drive = p[1].upper()
        return os.path.normpath(f"{drive}:{p[2:].replace('/', os.sep)}")
    # Fallback: prefix current drive
    if p.startswith("/"):
        cur_drive, _ = os.path.splitdrive(os.getcwd())
        return os.path.normpath(f"{cur_drive}{p.replace('/', os.sep)}")
    return os.path.normpath(os.path.abspath(p))


def _find_openems_dir(base_dir: str) -> Optional[str]:
    """Return an absolute directory that actually contains the openEMS DLLs."""
    candidates = []
    if base_dir:
        candidates.append(base_dir)
        candidates.append(os.path.join(base_dir, "openEMS"))
    for cand in candidates:
        cand_abs = _coerce_win_path(cand)
        if os.path.isdir(cand_abs):
            has_dll = any(
                os.path.isfile(os.path.join(cand_abs, dll))
                for dll in ("CSXCAD.dll", "openEMS.dll")
            )
            if has_dll:
                return cand_abs
    return None


def _add_dll_dirs(root: str) -> None:
    root = _coerce_win_path(root)
    # Set env for modules that look for this
    os.environ["OPENEMS_INSTALL_PATH"] = root
    os.add_dll_directory(root)
    qt = os.path.join(root, "qt5")
    if os.path.isdir(qt):
        os.add_dll_directory(qt)


def calculate_microstrip_width(freq_hz: float, eps_r: float, h_m: float, z0: float = 50.0) -> float:
    """
    Calculate microstrip line width for given characteristic impedance.
    
    Uses Wheeler's equations for microstrip line design.
    
    Args:
        freq_hz: Frequency in Hz
        eps_r: Relative permittivity of substrate
        h_m: Substrate thickness in meters
        z0: Target characteristic impedance in Ohms (default 50Ω)
        
    Returns:
        Microstrip width in meters
    """
    # Wheeler's equations for microstrip width calculation
    # Reference: "Transmission Line Design Handbook" by Brian C. Wadell
    
    if z0 < 44:
        # Wide microstrip (W/h > 1)
        A = (z0/60) * np.sqrt((eps_r + 1)/2) + (eps_r - 1)/(eps_r + 1) * (0.23 + 0.11/eps_r)
        W_h = 8*np.exp(A) / (np.exp(2*A) - 2)
    else:
        # Narrow microstrip (W/h < 1)
        B = 377*np.pi / (2*z0*np.sqrt(eps_r))
        W_h = (2/np.pi) * (B - 1 - np.log(2*B - 1) + 
                          (eps_r - 1)/(2*eps_r) * (np.log(B - 1) + 0.39 - 0.61/eps_r))
    
    return W_h * h_m


def probe_openems_microstrip(dll_dir: str) -> OpenEMSResult:
    """Probe openEMS installation for microstrip solver"""
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSResult(False, f"Could not find openEMS DLLs in '{dll_dir}'")
        
        _add_dll_dirs(resolved)
        
        # Test import
        from openEMS import openEMS as oem  # type: ignore
        from openEMS import CSXCAD  # type: ignore
        
        return OpenEMSResult(True, f"openEMS found at: {resolved}")
        
    except Exception as e:
        return OpenEMSResult(False, f"openEMS import failed: {e}")


def prepare_openems_microstrip_patch(
    params: PatchAntennaParams,
    *,
    dll_dir: str,
    feed_direction: FeedDirection = FeedDirection.NEG_X,
    feed_line_length_mm: float = 20.0,
    boundary: str = "MUR",
    theta_step_deg: float = 2.0,
    work_dir: str = "openems_out_microstrip",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMSPrepared:
    """
    Prepare microstrip-fed patch antenna simulation.
    
    Args:
        params: Antenna parameters
        dll_dir: openEMS installation directory
        feed_direction: Direction of microstrip feed line
        feed_line_length_mm: Length of feed line in mm
        work_dir: Working directory for simulation files
        cleanup: Whether to clean existing files
        verbose: Verbosity level
        
    Returns:
        OpenEMSPrepared object with simulation setup
    """
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        
        _add_dll_dirs(resolved)
        from CSXCAD import ContinuousStructure  # type: ignore
        from openEMS import openEMS  # type: ignore
        from openEMS.physical_constants import C0, EPS0  # type: ignore
        
        # Simulation parameters (following tutorial patterns)
        unit = 1e-3  # All dimensions in mm
        f0 = params.frequency_hz
        fc = f0 / 2.0  # 20 dB corner frequency
        
        # Calculate patch dimensions if not provided
        if params.patch_length_m and params.patch_width_m:
            patch_L = params.patch_length_m * 1e3  # Convert to mm
            patch_W = params.patch_width_m * 1e3
        else:
            from .physics import design_patch_for_frequency
            L_m, W_m, _ = design_patch_for_frequency(f0, params.eps_r, params.h_m)
            patch_L = L_m * 1e3  # along Y (length)
            patch_W = W_m * 1e3  # along X (width)
        
        h = params.h_m * 1e3  # Substrate thickness in mm
        
        # Calculate microstrip feed line width for 50Ω
        feed_width = calculate_microstrip_width(f0, params.eps_r, params.h_m) * 1e3  # Convert to mm
        
        # Substrate and ground plane dimensions (larger than patch + feed)
        substrate_margin = 30.0  # mm margin around patch
        feed_line_length = feed_line_length_mm
        
        # Calculate substrate dimensions based on feed direction
        if feed_direction in [FeedDirection.POS_X, FeedDirection.NEG_X]:
            # Feed along X direction
            substrate_W = patch_W + 2 * substrate_margin + feed_line_length
            substrate_L = patch_L + 2 * substrate_margin
        else:
            # Feed along Y direction  
            substrate_W = patch_W + 2 * substrate_margin
            substrate_L = patch_L + 2 * substrate_margin + feed_line_length
        
        # Simulation box (air box around substrate)
        air_margin = 50.0  # mm
        SimBox_X = substrate_W + 2 * air_margin
        SimBox_Y = substrate_L + 2 * air_margin
        SimBox_Z = 160.0  # mm above and below (more robust against boundary reflections)
        
        if verbose:
            print(f"Patch dimensions: {patch_W:.1f} x {patch_L:.1f} mm")
            print(f"Substrate: {substrate_W:.1f} x {substrate_L:.1f} x {h:.3f} mm")
            print(f"Feed line: width={feed_width:.2f} mm, length={feed_line_length:.1f} mm")
            print(f"Feed direction: {feed_direction}")
        
        # FDTD setup (following Simple_Patch_Antenna.py tutorial exactly)
        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)  # Same as tutorial
        FDTD.SetGaussExcite(f0, fc)
        bc = ['MUR'] * 6 if boundary.upper().startswith('MUR') else ['PML_8'] * 6
        FDTD.SetBoundaryCond(bc)
        
        # Geometry and mesh setup (tutorial order)
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)
        
        # Mesh resolution
        mesh_res = C0 / (f0 + fc) / unit / 20  # lambda/20
        
        # Initialize mesh with simulation box (following tutorial pattern)
        mesh.AddLine('x', [-SimBox_X/2, SimBox_X/2])
        mesh.AddLine('y', [-SimBox_Y/2, SimBox_Y/2])
        mesh.AddLine('z', [-SimBox_Z/3, SimBox_Z*2/3])  # Tutorial pattern: -1/3, +2/3
        
        # Create substrate
        substrate_kappa = 2*np.pi*f0 * EPS0*params.eps_r * params.loss_tangent
        substrate = CSX.AddMaterial('substrate', epsilon=params.eps_r, kappa=substrate_kappa)
        
        sub_start = [-substrate_W/2, -substrate_L/2, 0]
        sub_stop = [substrate_W/2, substrate_L/2, h]
        substrate.AddBox(priority=0, start=sub_start, stop=sub_stop)
        
        # Add mesh lines for substrate
        mesh.AddLine('z', np.linspace(0, h, 5))  # Discretize substrate thickness
        
        # Create ground plane (bottom of substrate)
        gnd = CSX.AddMetal('ground')
        gnd_start = sub_start.copy()
        gnd_stop = sub_stop.copy()
        gnd_start[2] = 0
        gnd_stop[2] = 0
        gnd.AddBox(priority=10, start=gnd_start, stop=gnd_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
        
        # Create patch antenna (on top of substrate)
        patch = CSX.AddMetal('patch')
        patch_start = [-patch_W/2, -patch_L/2, h]
        patch_stop = [patch_W/2, patch_L/2, h]
        patch.AddBox(priority=10, start=patch_start, stop=patch_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)
        
        # Create microstrip feed line and determine port location
        feed_metal = CSX.AddMetal('feed_line')
        
        if feed_direction == FeedDirection.NEG_X:
            # Feed from negative X direction
            feed_start = [-substrate_W/2, -feed_width/2, h]
            feed_stop = [-patch_W/2, feed_width/2, h]
            # Port spans all 3 dimensions (following MSL_NotchFilter.py pattern)
            port_start = [-substrate_W/2, -feed_width/2, h]  # One corner (top of substrate)
            port_stop = [-substrate_W/2 + min(feed_line_length, (substrate_W - patch_W)/2), feed_width/2, 0]  # extend into dielectric
            port_dir = 'x'
            
        elif feed_direction == FeedDirection.POS_X:
            # Feed from positive X direction
            feed_start = [patch_W/2, -feed_width/2, h]
            feed_stop = [substrate_W/2, feed_width/2, h]
            # Port spans all 3 dimensions (following MSL_NotchFilter.py pattern)
            port_start = [substrate_W/2, -feed_width/2, h]
            port_stop = [substrate_W/2 - min(feed_line_length, (substrate_W - patch_W)/2), feed_width/2, 0]
            port_dir = 'x'
            
        elif feed_direction == FeedDirection.NEG_Y:
            # Feed from negative Y direction
            feed_start = [-feed_width/2, -substrate_L/2, h]
            feed_stop = [feed_width/2, -patch_L/2, h]
            # Port spans all 3 dimensions (following MSL_NotchFilter.py pattern)
            port_start = [-feed_width/2, -substrate_L/2, h]
            port_stop = [feed_width/2, -substrate_L/2 + min(feed_line_length, (substrate_L - patch_L)/2), 0]
            port_dir = 'y'
            
        else:  # FeedDirection.POS_Y
            # Feed from positive Y direction
            feed_start = [-feed_width/2, patch_L/2, h]
            feed_stop = [feed_width/2, substrate_L/2, h]
            # Port spans all 3 dimensions (following MSL_NotchFilter.py pattern)
            port_start = [-feed_width/2, substrate_L/2, h]
            port_stop = [feed_width/2, substrate_L/2 - min(feed_line_length, (substrate_L - patch_L)/2), 0]
            port_dir = 'y'
        
        # Add feed line
        feed_metal.AddBox(priority=10, start=feed_start, stop=feed_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=feed_metal, metal_edge_res=mesh_res/2)
        
        # Replace MSL port with a LumpedPort bridging patch to ground at the feed location.
        # Determine feed point at the patch edge center along the selected axis.
        if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
            feed_px = -patch_W/2 if feed_direction == FeedDirection.NEG_X else patch_W/2
            feed_py = 0.0
        else:
            feed_px = 0.0
            feed_py = -patch_L/2 if feed_direction == FeedDirection.NEG_Y else patch_L/2
        # Snap mesh lines at feed location and across substrate thickness
        try:
            mesh.AddLine('x', [float(feed_px)])
            mesh.AddLine('y', [float(feed_py)])
            mesh.AddLine('z', [0.0, float(h)])
        except Exception:
            pass
        port_start = [float(feed_px), float(feed_py), 0.0]
        port_stop  = [float(feed_px), float(feed_py), float(h)]
        port = FDTD.AddLumpedPort(1, 50.0, port_start, port_stop, 'z', 1.0, priority=5, edges2grid='xy')
        
        # Add mesh lines for feed line (following MSL tutorial patterns)
        if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
            # X-direction feed: add mesh lines for feed width
            mesh.AddLine('y', [-feed_width/2, 0, feed_width/2])
        else:
            # Y-direction feed: add mesh lines for feed width  
            mesh.AddLine('x', [-feed_width/2, 0, feed_width/2])
        
        # Smooth mesh (tutorial pattern)
        mesh.SmoothMeshLines('all', mesh_res, 1.4)
        
        # Create NF2FF recording box
        nf2ff = FDTD.CreateNF2FFBox()
        
        # Setup simulation directory (unique per run to avoid collisions across GUI instances)
        suffix = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{random.randint(1000,9999)}"
        sim_path = os.path.abspath(f"{work_dir}_{suffix}")
        if cleanup and os.path.isdir(sim_path):
            shutil.rmtree(sim_path, ignore_errors=True)
        # Do not pre-create; FDTD.Run will create it
        
        # Define angles for far-field calculation (include zenith explicitly to avoid gaps)
        step = max(0.5, float(theta_step_deg))
        theta = np.arange(0.0, 181.0, step)  # 0° to 180° inclusive
        # Force fixed phi cuts for classic E/H planes
        phi = np.array([0.0, 90.0])
        nf_center = np.array([0.0, 0.0, h/2000.0])  # Center at substrate middle (convert to meters)
        
        return OpenEMSPrepared(
            True, 
            f"Microstrip patch prepared (feed: {feed_direction}, DLLs: {resolved})",
            FDTD=FDTD,
            nf=nf2ff,
            sim_path=sim_path,
            theta=theta,
            phi=phi,
            nf_center=nf_center
        )
        
    except Exception as e:
        return OpenEMSPrepared(False, f"Microstrip solver prepare failed: {e}")


def run_prepared_openems_microstrip(
    prepared: OpenEMSPrepared,
    *,
    frequency_hz: float,
    verbose: int = 1,
) -> OpenEMSResult:
    """
    Run the prepared microstrip patch antenna simulation.
    
    Args:
        prepared: Prepared simulation object
        frequency_hz: Simulation frequency in Hz
        verbose: Verbosity level
        
    Returns:
        OpenEMSResult with simulation results
    """
    try:
        if not prepared.ok or prepared.FDTD is None or prepared.nf is None:
            return OpenEMSResult(False, prepared.message)
        
        FDTD = prepared.FDTD
        nf2ff = prepared.nf
        sim_path = prepared.sim_path or "openems_out_microstrip"
        port = getattr(prepared, 'port', None)
        
        print(f"Starting microstrip FDTD simulation in: {sim_path}")
        print("This may take 45-90 seconds...")
        import sys
        sys.stdout.flush()
        
        # Run FDTD simulation
        FDTD.Run(sim_path, verbose=verbose, cleanup=True)
        print("Microstrip FDTD simulation completed!")
        sys.stdout.flush()
        
        # Post-processing
        # Calculate port parameters if available
        if port is not None:
            f = np.linspace(max(1e9, frequency_hz*0.7), frequency_hz*1.3, 201)
            port.CalcPort(sim_path, f)
            
            # Find resonance frequency for far-field calculation
            s11 = port.uf_ref / port.uf_inc
            s11_dB = 20.0 * np.log10(np.abs(s11))
            
            # Find minimum S11 for resonance
            idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
            if len(idx) == 1:
                f_res = f[idx[0]]
                if verbose:
                    print(f"Found resonance at {f_res/1e9:.3f} GHz (S11 = {s11_dB[idx[0]]:.1f} dB)")
            else:
                f_res = frequency_hz
                if verbose:
                    print(f"No clear resonance found, using target frequency {f_res/1e9:.3f} GHz")
        else:
            f_res = frequency_hz
        
        # Calculate far-field pattern
        theta = np.asarray(prepared.theta)  # degrees
        phi = np.asarray(prepared.phi)      # degrees
        
        print("Calculate NF2FF (microstrip method)")
        nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=prepared.nf_center)
        
        if nf2ff_res is None:
            return OpenEMSResult(False, "NF2FF returned None - no field data recorded")
        
        # Extract far-field results
        try:
            E_norm = getattr(nf2ff_res, 'E_norm', None)
            Dmax = getattr(nf2ff_res, 'Dmax', None)
            
            if E_norm is None or Dmax is None:
                return OpenEMSResult(False, "NF2FF results missing E_norm or Dmax")
            
            E_norm_array = np.asarray(E_norm[0])  # First frequency
            Dmax_val = float(np.asarray(Dmax[0]))
            
            if verbose > 0:
                print(f"E_norm shape: {E_norm_array.shape}")
                print(f"Dmax: {Dmax_val:.2f} ({10*np.log10(Dmax_val):.1f} dBi)")
            
            # Calculate directivity in dBi (tutorial method)
            E_max = np.max(E_norm_array)
            intensity_dB = 20.0*np.log10(E_norm_array/E_max) + 10*np.log10(Dmax_val)
            
            # Convert angles to radians for return
            theta_rad = theta * np.pi / 180.0
            phi_rad = phi * np.pi / 180.0
            
            # Prepare results
            # Return in the harmonized result structure used by the fixed solver
            return OpenEMSResult(
                True,
                "Microstrip simulation completed successfully",
                theta=theta_rad,
                phi=phi_rad,
                intensity=intensity_dB,
                sim_path=sim_path,
                is_dBi=True,
            )
            
        except Exception as e:
            return OpenEMSResult(False, f"Far-field processing failed: {e}")
        
    except Exception as e:
        return OpenEMSResult(False, f"Microstrip simulation failed: {e}")
