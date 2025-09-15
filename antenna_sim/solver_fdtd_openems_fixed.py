from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .models import PatchAntennaParams


@dataclass
class OpenEMSProbe:
    ok: bool
    message: str
    api: Dict[str, List[str]]


@dataclass
class OpenEMSResult:
    ok: bool
    message: str
    theta: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    sim_path: Optional[str] = None
    is_dBi: bool = False


@dataclass
class OpenEMSPrepared:
    ok: bool
    message: str
    FDTD: Optional[object] = None
    nf: Optional[object] = None
    sim_path: Optional[str] = None
    theta: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    nf_center: Optional[np.ndarray] = None


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


def probe_openems_fixed(dll_dir: str) -> OpenEMSProbe:
    api: Dict[str, List[str]] = {}
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSProbe(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'. Select the folder that contains the DLLs (usually the 'openEMS' folder).", api)
        _add_dll_dirs(resolved)
        from openEMS import openEMS as oem  # type: ignore
        from openEMS import CSXCAD  # type: ignore
        from openEMS.physical_constants import C0, EPS0  # type: ignore

        api["openEMS.openEMS"] = [n for n in dir(oem) if not n.startswith("_")]
        api["CSXCAD.CSProperties"] = [n for n in dir(CSXCAD.CSProperties) if not n.startswith("_")]
        api["CSXCAD.CSPrimitives"] = [n for n in dir(CSXCAD.CSPrimitives) if not n.startswith("_")]
        csx = CSXCAD.ContinuousStructure()
        api["CSXCAD.ContinuousStructure"] = [n for n in dir(csx) if not n.startswith("_")]
        return OpenEMSProbe(True, f"openEMS Python API detected (DLLs from: {resolved})", api)
    except Exception as e:  # pragma: no cover
        return OpenEMSProbe(False, f"openEMS import failed: {e}", api)


def prepare_openems_patch_fixed(
    params: PatchAntennaParams,
    *,
    dll_dir: str,
    work_dir: str = "openems_out_fixed",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMSPrepared:
    """
    Create a fixed 3D openEMS patch antenna simulation based exactly on Simple_Patch_Antenna.py tutorial.
    This version follows the tutorial step-by-step to avoid the issues we've been having.
    """
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        _add_dll_dirs(resolved)
        
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS
        from openEMS.physical_constants import C0, EPS0
        
        # === EXACT TUTORIAL SETUP ===
        
        # General parameter setup (from tutorial)
        unit = 1e-3  # all length in mm
        
        # Calculate patch dimensions if needed
        if params.patch_length_m and params.patch_width_m:
            # Convert to mm and follow tutorial naming: patch_width = resonant length in x-direction
            patch_width = params.patch_width_m * 1e3   # tutorial calls this "resonant length" (x-direction)
            patch_length = params.patch_length_m * 1e3  # y-direction
        else:
            from .physics import design_patch_for_frequency
            L_m, W_m, _ = design_patch_for_frequency(params.frequency_hz, params.eps_r, params.h_m)
            patch_width = W_m * 1e3   # x-direction (resonant dimension)
            patch_length = L_m * 1e3  # y-direction
        
        # Substrate setup (tutorial values adapted)
        substrate_epsR = params.eps_r
        substrate_kappa = 1e-3 * 2*np.pi*params.frequency_hz * EPS0*substrate_epsR * params.loss_tangent
        substrate_width = 60   # mm (tutorial default)
        substrate_length = 60  # mm (tutorial default)
        substrate_thickness = params.h_m * 1e3  # convert to mm
        substrate_cells = 4
        
        # Setup feeding (tutorial defaults)
        feed_pos = -6  # feeding position in x-direction (mm)
        feed_R = 50    # feed resistance
        
        # Size of the simulation box (tutorial)
        SimBox = np.array([200, 200, 150])
        
        # Setup FDTD parameter & excitation function (tutorial)
        f0 = params.frequency_hz  # center frequency
        fc = f0 / 2.0             # 20 dB corner frequency (tutorial uses f0/2)
        
        # FDTD setup - EXACTLY like tutorial
        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
        FDTD.SetGaussExcite(f0, fc)
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])  # tutorial uses MUR, not PML
        
        # Setup Geometry & Mesh - EXACTLY like tutorial
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)
        mesh_res = C0/(f0+fc)/unit/20
        
        # Generate properties, primitives and mesh-grid
        # Initialize the mesh with the "air-box" dimensions - EXACTLY like tutorial
        mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2])
        mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2])
        mesh.AddLine('z', [-SimBox[2]/3, SimBox[2]*2/3])
        
        # Create patch - EXACTLY like tutorial
        patch = CSX.AddMetal('patch')  # create a perfect electric conductor (PEC)
        start = [-patch_width/2, -patch_length/2, substrate_thickness]
        stop  = [patch_width/2,  patch_length/2,  substrate_thickness]
        patch.AddBox(priority=10, start=start, stop=stop)  # add a box-primitive to the metal property 'patch'
        FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)
        
        # Create substrate - EXACTLY like tutorial
        substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
        start = [-substrate_width/2, -substrate_length/2, 0]
        stop  = [substrate_width/2,  substrate_length/2,  substrate_thickness]
        substrate.AddBox(priority=0, start=start, stop=stop)
        
        # Add extra cells to discretize the substrate thickness - EXACTLY like tutorial
        mesh.AddLine('z', np.linspace(0, substrate_thickness, substrate_cells+1))
        
        # Create ground (same size as substrate) - EXACTLY like tutorial
        gnd = CSX.AddMetal('gnd')  # create a perfect electric conductor (PEC)
        start[2] = 0
        stop[2] = 0
        gnd.AddBox(start, stop, priority=10)
        
        FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
        
        # Apply the excitation & resist as a current source - EXACTLY like tutorial
        start = [feed_pos, 0, 0]
        stop  = [feed_pos, 0, substrate_thickness]
        port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5, edges2grid='xy')
        
        mesh.SmoothMeshLines('all', mesh_res, 1.4)
        
        # Add the nf2ff recording box - EXACTLY like tutorial
        nf2ff = FDTD.CreateNF2FFBox()
        
        # Setup simulation path
        sim_path = os.path.abspath(work_dir)
        if cleanup and os.path.isdir(sim_path):
            shutil.rmtree(sim_path, ignore_errors=True)
        os.makedirs(sim_path, exist_ok=True)
        
        # Define theta/phi arrays for NF2FF calculation
        theta = np.arange(0.0, 180.0, 2.0)  # degrees, 0° (zenith) to 180° (nadir) for proper spherical coordinates
        phi = [0., 90.]  # degrees, like tutorial (E-plane and H-plane cuts)
        
        # NF2FF center (tutorial uses [0,0,1e-3] for patch antennas)
        nf_center = np.array([0, 0, 1e-3])  # 1mm above ground
        
        if verbose:
            print(f"DEBUG: Fixed solver setup complete")
            print(f"DEBUG: Patch dimensions: {patch_width:.1f} x {patch_length:.1f} mm")
            print(f"DEBUG: Substrate: εr={substrate_epsR:.2f}, h={substrate_thickness:.3f}mm")
            print(f"DEBUG: Feed position: x={feed_pos}mm")
        
        return OpenEMSPrepared(
            True, 
            f"Fixed solver prepared (DLLs from: {resolved})", 
            FDTD=FDTD, 
            nf=nf2ff, 
            sim_path=sim_path, 
            theta=theta, 
            phi=phi, 
            nf_center=nf_center
        )
        
    except Exception as e:
        return OpenEMSPrepared(False, f"Fixed solver prepare failed: {e}")


def run_prepared_openems_fixed(
    prepared: OpenEMSPrepared,
    *,
    frequency_hz: float,
    verbose: int = 1,
) -> OpenEMSResult:
    """
    Run the fixed openEMS simulation and extract results exactly like the tutorial.
    """
    try:
        if not prepared.ok or prepared.FDTD is None or prepared.nf is None:
            return OpenEMSResult(False, prepared.message)
            
        FDTD = prepared.FDTD
        nf2ff = prepared.nf
        sim_path = prepared.sim_path or "openems_out_fixed"
        
        print(f"Starting fixed FDTD simulation in: {sim_path}")
        print("This may take 30-60 seconds...")
        import sys
        sys.stdout.flush()
        
        # Run the simulation - EXACTLY like tutorial
        FDTD.Run(sim_path, verbose=verbose, cleanup=True)
        print("Fixed FDTD simulation completed!")
        sys.stdout.flush()
        
        # Post-processing and plotting - EXACTLY like tutorial
        f = np.linspace(max(1e9, frequency_hz*0.5), frequency_hz*1.5, 401)
        
        # Find resonance frequency (tutorial approach)
        # For now, just use the requested frequency
        f_res = frequency_hz
        
        theta = np.asarray(prepared.theta)  # degrees
        phi = np.asarray(prepared.phi)      # degrees
        
        print("Calculate NF2FF (tutorial method)")
        # Calculate NF2FF exactly like tutorial
        nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=prepared.nf_center)
        
        if nf2ff_res is None:
            return OpenEMSResult(False, "NF2FF returned None - no field data recorded")
        
        # Extract results exactly like tutorial
        try:
            # Tutorial uses E_norm and Dmax
            E_norm = nf2ff_res.E_norm[0] if hasattr(nf2ff_res, 'E_norm') else None
            Dmax = nf2ff_res.Dmax[0] if hasattr(nf2ff_res, 'Dmax') else 1.0
            
            if E_norm is not None:
                # Tutorial approach: normalize E_norm and add Dmax in dB
                E_norm_array = np.asarray(E_norm)
                E_max = np.max(E_norm_array)
                if E_max > 0:
                    # Convert to dBi like tutorial: 20*log10(E_norm/E_max) + 10*log10(Dmax)
                    intensity_dB = 20.0*np.log10(E_norm_array/E_max) + 10*np.log10(Dmax)
                else:
                    intensity_dB = np.full_like(E_norm_array, -50.0)  # Very low value
                
                # Convert theta, phi back to radians for our plotting system
                theta_rad = np.deg2rad(theta)
                phi_rad = np.deg2rad(phi)
                
                if verbose:
                    print(f"DEBUG: E_norm shape: {E_norm_array.shape}")
                    print(f"DEBUG: Dmax: {Dmax:.2f} ({10*np.log10(Dmax):.1f} dBi)")
                    print(f"DEBUG: Pattern range: {intensity_dB.min():.1f} to {intensity_dB.max():.1f} dBi")
                
                return OpenEMSResult(
                    True, 
                    "Fixed openEMS FDTD completed", 
                    theta=theta_rad, 
                    phi=phi_rad, 
                    intensity=intensity_dB, 
                    sim_path=sim_path, 
                    is_dBi=True
                )
            else:
                return OpenEMSResult(False, "No E_norm data found in NF2FF result")
                
        except Exception as e:
            return OpenEMSResult(False, f"NF2FF result processing failed: {e}")
            
    except Exception as e:
        return OpenEMSResult(False, f"Fixed openEMS run failed: {e}")
