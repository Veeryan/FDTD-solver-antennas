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


def _normalize_intensity(intensity: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    arr = np.asarray(intensity)
    nth, nph = len(theta), len(phi)
    if arr.ndim == 1:
        if arr.size == nth:
            arr = arr.reshape(nth, 1)
        elif arr.size == nph:
            arr = arr.reshape(1, nph)
        else:
            arr = arr.reshape(nth, -1) if arr.size % nth == 0 else arr.reshape(-1, nph)
    if arr.shape == (nph, nth):  # transpose if swapped
        arr = arr.T
    # If either dimension is 1, tile to form full grid (for plotting cuts this is fine)
    if arr.shape[0] == 1 and nth > 1:
        arr = np.tile(arr, (nth, 1))
    if arr.shape[1] == 1 and nph > 1:
        arr = np.tile(arr, (1, nph))
    return arr


def probe_openems(dll_dir: str) -> OpenEMSProbe:
    api: Dict[str, List[str]] = {}
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSProbe(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'. Select the folder that contains the DLLs (usually the 'openEMS' folder).", api)
        _add_dll_dirs(resolved)
        from openEMS import openEMS as oem  # type: ignore
        from openEMS import CSXCAD  # type: ignore
        from openEMS.physical_constants import EPS0  # type: ignore
        from openEMS.physical_constants import EPS0  # type: ignore
        from openEMS import nf2ff as nf  # type: ignore
        from openEMS import ports  # type: ignore
        from openEMS import utilities as util  # type: ignore
        from openEMS import automesh as am  # type: ignore

        api["openEMS.openEMS"] = [n for n in dir(oem) if not n.startswith("_")]
        api["CSXCAD.CSProperties"] = [n for n in dir(CSXCAD.CSProperties) if not n.startswith("_")]
        api["CSXCAD.CSPrimitives"] = [n for n in dir(CSXCAD.CSPrimitives) if not n.startswith("_")]
        csx = CSXCAD.ContinuousStructure()
        api["CSXCAD.ContinuousStructure"] = [n for n in dir(csx) if not n.startswith("_")]
        api["nf2ff"] = [n for n in dir(nf) if not n.startswith("_")]
        api["ports"] = [n for n in dir(ports) if not n.startswith("_")]
        api["utilities"] = [n for n in dir(util) if not n.startswith("_")]
        api["automesh"] = [n for n in dir(am) if not n.startswith("_")]
        return OpenEMSProbe(True, f"openEMS Python API detected (DLLs from: {resolved})", api)
    except Exception as e:  # pragma: no cover
        return OpenEMSProbe(False, f"openEMS import failed: {e}", api)


def prepare_openems_patch(
    params: PatchAntennaParams,
    *,
    dll_dir: str,
    work_dir: str = "openems_out",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMSPrepared:
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        _add_dll_dirs(resolved)
        from openEMS import openEMS as oem  # type: ignore
        from openEMS import CSXCAD  # type: ignore

        # Use openEMS tutorial approach - proven to work
        f0 = params.frequency_hz
        c0 = 299792458.0
        unit = 1e-3  # mm
        
        # Calculate patch dimensions (mm). IMPORTANT: map W to x, L to y (tutorial convention)
        if params.patch_length_m and params.patch_width_m:
            L = params.patch_length_m * 1e3  # along y
            W = params.patch_width_m * 1e3   # along x
        else:
            from .physics import design_patch_for_frequency
            L_m, W_m, _ = design_patch_for_frequency(f0, params.eps_r, params.h_m)
            L = L_m * 1e3
            W = W_m * 1e3

        h = params.h_m * 1e3  # substrate thickness in mm

        # Feed position (x-direction), use fraction of width similar to tutorial examples
        feed_x = -0.2 * W
        feed_y = 0.0

        # Simulation domain (mm)
        SimBox = np.array([200.0, 200.0, 150.0])

        # Excitation and FDTD settings
        fc = f0 / 2.0
        res = c0 / (f0 + fc) / 1e-3 / 20.0

        # Use conservative timestep count and stronger end criteria for stable NF2FF
        FDTD = oem(NrTS=60000, EndCriteria=1e-5)
        FDTD.SetGaussExcite(f0, fc)
        # Prefer PML-8 boundaries for correct NF2FF (numeric code 3)
        FDTD.SetBoundaryCond([3, 3, 3, 3, 3, 3])

        # Geometry and mesh
        csx = CSXCAD.ContinuousStructure()
        FDTD.SetCSX(csx)
        mesh = csx.GetGrid()
        mesh.SetDeltaUnit(unit)

        # Initialize mesh with coarse lines (tutorial style)
        mesh.AddLine('x', [-SimBox[0]/2.0, SimBox[0]/2.0])
        mesh.AddLine('y', [-SimBox[1]/2.0, SimBox[1]/2.0])
        mesh.AddLine('z', [-SimBox[2]/3.0, SimBox[2]*2.0/3.0])

        # Substrate material including dielectric loss from tanδ
        # Use explicit EPS0 to avoid import ambiguity in some environments
        EPS0_VAL = 8.854187817e-12
        kappa = 2.0 * np.pi * f0 * EPS0_VAL * params.eps_r * max(0.0, params.loss_tangent)
        try:
            substrate = csx.AddMaterial('substrate', epsilon=params.eps_r, kappa=kappa)
        except TypeError:
            substrate = csx.AddMaterial('substrate')
            # Older bindings use string property keys
            substrate.SetMaterialProperty('Eps', params.eps_r)
            substrate.SetMaterialProperty('Kappa', kappa)
        substrate.AddBox([-W/2.0 - (SimBox[0]-W)/2.0, -L/2.0 - (SimBox[1]-L)/2.0, 0.0],
                         [ W/2.0 + (SimBox[0]-W)/2.0,  L/2.0 + (SimBox[1]-L)/2.0,  h])

        # Add extra cells across substrate thickness
        mesh.AddLine('z', np.linspace(0.0, h, 4 + 1).tolist())

        # Ground plane (z=0) and patch (z=h) as zero-thickness PEC surfaces (tutorial style)
        gnd = csx.AddMetal('gnd')
        gnd.AddBox([-W/2.0 - (SimBox[0]-W)/2.0, -L/2.0 - (SimBox[1]-L)/2.0, 0.0],
                   [ W/2.0 + (SimBox[0]-W)/2.0,  L/2.0 + (SimBox[1]-L)/2.0, 0.0], priority=10)

        patch = csx.AddMetal('patch')
        patch.AddBox([-W/2.0, -L/2.0, h], [W/2.0, L/2.0, h], priority=10)

        # Snap metal edges to mesh and refine around them
        FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=res/2.0)
        FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

        # Lumped port (z-directed) from ground to patch at feed_x
        # Snap port to mesh by explicitly adding lines at feed and z planes
        mesh.AddLine('x', [float(feed_x)])
        mesh.AddLine('z', [0.0, float(h)])
        port_start = [float(feed_x), float(feed_y), 0.0]
        port_stop  = [float(feed_x), float(feed_y), float(h)]

        if verbose:
            print(f"DEBUG: Dimensions (mm): W(x)={W:.2f}, L(y)={L:.2f}, h={h:.3f}")
            print(f"DEBUG: Feed x-position: {feed_x:.2f} mm")
            print(f"DEBUG: tanδ={params.loss_tangent:.4f} -> kappa={kappa:.3e}")

        # Try tutorial signature first, fallback to legacy
        try:
            port = FDTD.AddLumpedPort(1, 50, port_start, port_stop, 'z', 1.0, priority=5, edges2grid='xy')
        except Exception as e1:
            try:
                port = FDTD.AddLumpedPort(1, 50, port_start, port_stop, 2, excite=1.0)
            except Exception as e2:
                raise e2

        # Smooth mesh after edges2grid to ensure well-behaved steps
        mesh.SmoothMeshLines('all', res, 1.4)

        # Create NF2FF recording box (default)
        nf = FDTD.CreateNF2FFBox()
        
        sim_path = os.path.abspath(work_dir)
        if cleanup and os.path.isdir(sim_path):
            shutil.rmtree(sim_path, ignore_errors=True)
        os.makedirs(sim_path, exist_ok=True)

        theta = np.linspace(0, np.pi, 91)
        phi = np.linspace(0, 2*np.pi, 181)
        # NF2FF phase center: 1 mm above ground (tutorial-style)
        nf_center = np.array([0.0, 0.0, 1e-3], dtype=float)
        return OpenEMSPrepared(True, f"Prepared (DLLs from: {resolved})", FDTD=FDTD, nf=nf, sim_path=sim_path, theta=theta, phi=phi, nf_center=nf_center)
    except Exception as e:
        return OpenEMSPrepared(False, f"prepare failed: {e}")


def run_prepared_openems(
    prepared: OpenEMSPrepared,
    *,
    frequency_hz: float,
    verbose: int = 1,
) -> OpenEMSResult:
    try:
        if not prepared.ok or prepared.FDTD is None or prepared.nf is None:
            return OpenEMSResult(False, prepared.message)
        FDTD = prepared.FDTD
        nf = prepared.nf
        sim_path = prepared.sim_path or "openems_out"
        
        print(f"Starting FDTD simulation in: {sim_path}")
        print("This may take 30-60 seconds...")
        import sys
        sys.stdout.flush()  # Force output to appear immediately
        
        FDTD.Run(sim_path, cleanup=False, verbose=verbose)
        print("FDTD simulation completed!")
        sys.stdout.flush()
        
        # Try NF2FF calculation with error handling
        try:
            center = getattr(prepared, 'nf_center', None)
            if center is None:
                center = [0.0, 0.0, 1e-3]
            # openEMS Python expects angles in degrees; prepared arrays are radians
            theta_deg = np.asarray(prepared.theta, dtype=float) * (180.0/np.pi)
            phi_deg   = np.asarray(prepared.phi, dtype=float) * (180.0/np.pi)
            res = nf.CalcNF2FF(sim_path, float(frequency_hz), theta_deg, phi_deg, center=center)
            if res is None:
                return OpenEMSResult(False, "NF2FF returned None - no field data recorded")
        except Exception as e:
            return OpenEMSResult(False, f"NF2FF calculation failed: {e}")

        def get_arr(obj, names):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return None

        # Debug: show what attributes the result object has
        if verbose > 0:
            attrs = [attr for attr in dir(res) if not attr.startswith('_')]
            print(f"DEBUG: NF2FF result attributes: {attrs}")

        Eth = get_arr(res, ["E_theta", "Eth", "E_th", "Etheta"])
        Eph = get_arr(res, ["E_phi", "Eph", "E_ph", "Ephi"])
        P_rad = get_arr(res, ["P_rad", "P", "U"])  # try power density first for dBi
        Prad  = get_arr(res, ["Prad", "Ptot", "P_rad_tot"])  # total radiated power
        
        if verbose > 0:
            print(f"DEBUG: Found E_theta: {Eth is not None}, E_phi: {Eph is not None}")
            if Eth is not None:
                print(f"DEBUG: E_theta shape: {np.array(Eth).shape}, max: {np.max(np.abs(Eth))}")
            if Eph is not None:
                print(f"DEBUG: E_phi shape: {np.array(Eph).shape}, max: {np.max(np.abs(Eph))}")
        
        # Prefer directivity via P_rad/Prad (absolute, gives dBi); else fall back to |E|^2 normalized
        use_dbi = False
        arr = None
        if P_rad is not None and Prad is not None:
            try:
                prad_val = float(np.asarray(Prad).flat[0])
                pr_grid = np.asarray(P_rad)
                # squeeze freq dimension if present
                if pr_grid.ndim == 3:
                    pr_grid = pr_grid[0]
                pr_grid = np.asarray(pr_grid, dtype=float)
                arr = (pr_grid / max(1e-16, prad_val)) * (4.0 * np.pi)  # directivity (linear)
                arr = np.maximum(1e-16, arr)
                arr = 10.0 * np.log10(arr)  # dBi
                use_dbi = True
                if verbose > 0:
                    print(f"DEBUG: Computed directivity (dBi) from P_rad/Prad. Range: {arr.min():.2f}..{arr.max():.2f} dBi")
            except Exception as _:
                arr = None
        # If dBi exists but looks obviously wrong (e.g., -100 dBi everywhere), try E_norm + Dmax like tutorials
        if use_dbi and (np.nanmax(arr) < -10.0 or np.isnan(np.nanmax(arr))):
            try:
                E_norm = get_arr(res, ["E_norm"])  # normalized field amplitude grid
                Dmax = get_arr(res, ["Dmax"])     # scalar or array
                if E_norm is not None and Dmax is not None:
                    Eg = np.asarray(E_norm)
                    if Eg.ndim == 3:
                        Eg = Eg[0]
                    Eg = np.asarray(Eg, dtype=float)
                    Eg /= max(1e-16, float(Eg.max()))
                    Dmax_val = float(np.asarray(Dmax).flat[0])
                    dmax_db = 10.0 * np.log10(max(1e-16, Dmax_val))
                    arr = 20.0 * np.log10(np.maximum(1e-16, Eg)) + dmax_db
                    use_dbi = True
                    if verbose > 0:
                        print(f"DEBUG: Recomputed dBi from E_norm + Dmax (tutorial method). Range: {arr.min():.2f}..{arr.max():.2f} dBi")
            except Exception as _:
                pass
        if arr is None:
            if Eth is None or Eph is None:
                U = get_arr(res, ["U", "Gain", "E_norm"])  # try normalized fields
                if verbose > 0:
                    print(f"DEBUG: Falling back to field-based magnitude...")
                if U is None:
                    # Try any numeric array attribute
                    for attr in dir(res):
                        if not attr.startswith('_'):
                            val = getattr(res, attr)
                            if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                                U = val
                                break
                    if U is None:
                        return OpenEMSResult(False, f"NF2FF result has no usable field data. Available: {attrs}")
                arr_lin = np.abs(U).astype(float)
            else:
                arr_lin = (np.abs(Eth) ** 2 + np.abs(Eph) ** 2).astype(float)
                if verbose > 0:
                    print(f"DEBUG: Combined field magnitude max: {np.max(arr_lin)}")
            # normalize to peak and keep as linear for downstream
            arr = arr_lin / max(1e-16, float(np.max(arr_lin)))
            use_dbi = False

        # Ensure proper 2D shape for plotting
        n_th, n_ph = len(prepared.theta), len(prepared.phi)
        if arr.size == n_th * n_ph:
            arr = arr.reshape(n_th, n_ph)
        elif arr.size > n_th * n_ph:
            # Take first n_th*n_ph elements and reshape
            arr = arr.flat[:n_th * n_ph].reshape(n_th, n_ph)
        else:
            # Pad with zeros if too small
            arr_new = np.zeros((n_th, n_ph))
            arr_new.flat[:arr.size] = arr.flat
            arr = arr_new
        
        arr = np.asarray(arr, dtype=float)
        # If arr is dBi, do not normalize. Else ensure [0,1] linear
        if not use_dbi:
            arr /= max(1e-16, float(arr.max()))
        return OpenEMSResult(True, "openEMS FDTD completed", theta=prepared.theta, phi=prepared.phi, intensity=arr, sim_path=sim_path, is_dBi=use_dbi)
    except Exception as e:
        return OpenEMSResult(False, f"openEMS run failed: {e}")
