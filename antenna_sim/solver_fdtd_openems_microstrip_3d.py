from __future__ import annotations

from typing import Optional
import os
import numpy as np

from .models import PatchAntennaParams
from .solver_fdtd_openems_fixed import OpenEMSPrepared, OpenEMSResult
from .solver_fdtd_openems_microstrip import (
    FeedDirection,
    _find_openems_dir,
    _add_dll_dirs,
    calculate_microstrip_width,
)


def prepare_openems_microstrip_patch_3d(
    params: PatchAntennaParams,
    *,
    dll_dir: str,
    feed_direction: FeedDirection = FeedDirection.NEG_X,
    feed_line_length_mm: float = 20.0,
    boundary: str = "MUR",
    theta_step_deg: float = 2.0,
    phi_step_deg: float = 5.0,
    work_dir: str = "openems_out_microstrip",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMSPrepared:
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        _add_dll_dirs(resolved)

        from CSXCAD import ContinuousStructure  # type: ignore
        from openEMS import openEMS  # type: ignore
        from openEMS.physical_constants import C0, EPS0  # type: ignore

        unit = 1e-3
        f0 = params.frequency_hz
        fc = f0 / 2.0

        # Patch dims
        if params.patch_length_m and params.patch_width_m:
            patch_L = params.patch_length_m * 1e3
            patch_W = params.patch_width_m * 1e3
        else:
            from .physics import design_patch_for_frequency
            L_m, W_m, _ = design_patch_for_frequency(f0, params.eps_r, params.h_m)
            patch_L = L_m * 1e3
            patch_W = W_m * 1e3

        h = params.h_m * 1e3
        feed_width = calculate_microstrip_width(f0, params.eps_r, params.h_m) * 1e3

        substrate_margin = 30.0
        feed_len = feed_line_length_mm

        if feed_direction in [FeedDirection.POS_X, FeedDirection.NEG_X]:
            substrate_W = patch_W + 2 * substrate_margin + feed_len
            substrate_L = patch_L + 2 * substrate_margin
        else:
            substrate_W = patch_W + 2 * substrate_margin
            substrate_L = patch_L + 2 * substrate_margin + feed_len

        air_margin = 80.0
        SimBox_X = substrate_W + 2 * air_margin
        SimBox_Y = substrate_L + 2 * air_margin
        SimBox_Z = 160.0

        if verbose:
            print(f"Patch dimensions: {patch_W:.1f} x {patch_L:.1f} mm")
            print(f"Substrate: {substrate_W:.1f} x {substrate_L:.1f} x {h:.3f} mm")
            print(f"Feed line: width={feed_width:.2f} mm, length={feed_len:.1f} mm")
            print(f"Feed direction: {feed_direction}")

        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
        FDTD.SetGaussExcite(f0, fc)
        bc = ['MUR'] * 6 if boundary.upper().startswith('MUR') else ['PML_8'] * 6
        FDTD.SetBoundaryCond(bc)

        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)

        mesh_res = C0 / (f0 + fc) / unit / 20

        mesh.AddLine('x', [-SimBox_X/2, SimBox_X/2])
        mesh.AddLine('y', [-SimBox_Y/2, SimBox_Y/2])
        mesh.AddLine('z', [-SimBox_Z/3, SimBox_Z*2/3])

        substrate_kappa = 2*np.pi*f0 * EPS0*params.eps_r * params.loss_tangent
        substrate = CSX.AddMaterial('substrate', epsilon=params.eps_r, kappa=substrate_kappa)
        sub_start = [-substrate_W/2, -substrate_L/2, 0]
        sub_stop = [substrate_W/2, substrate_L/2, h]
        substrate.AddBox(priority=0, start=sub_start, stop=sub_stop)
        mesh.AddLine('z', np.linspace(0, h, 5))

        gnd = CSX.AddMetal('ground')
        gnd_start = sub_start.copy(); gnd_stop = sub_stop.copy()
        gnd_start[2] = 0; gnd_stop[2] = 0
        gnd.AddBox(priority=10, start=gnd_start, stop=gnd_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

        patch = CSX.AddMetal('patch')
        patch_start = [-patch_W/2, -patch_L/2, h]
        patch_stop = [patch_W/2, patch_L/2, h]
        patch.AddBox(priority=10, start=patch_start, stop=patch_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)

        feed_metal = CSX.AddMetal('feed_line')
        if feed_direction == FeedDirection.NEG_X:
            feed_start = [-substrate_W/2, -feed_width/2, h]
            feed_stop = [-patch_W/2, feed_width/2, h]
            port_start = [-substrate_W/2, -feed_width/2, h]
            port_stop = [-substrate_W/2 + min(feed_len, (substrate_W - patch_W)/2), feed_width/2, 0]
            port_dir = 'x'
        elif feed_direction == FeedDirection.POS_X:
            feed_start = [patch_W/2, -feed_width/2, h]
            feed_stop = [substrate_W/2, feed_width/2, h]
            port_start = [substrate_W/2, -feed_width/2, h]
            port_stop = [substrate_W/2 - min(feed_len, (substrate_W - patch_W)/2), feed_width/2, 0]
            port_dir = 'x'
        elif feed_direction == FeedDirection.NEG_Y:
            feed_start = [-feed_width/2, -substrate_L/2, h]
            feed_stop = [feed_width/2, -patch_L/2, h]
            port_start = [-feed_width/2, -substrate_L/2, h]
            port_stop = [feed_width/2, -substrate_L/2 + min(feed_len, (substrate_L - patch_L)/2), 0]
            port_dir = 'y'
        else:
            feed_start = [-feed_width/2, patch_L/2, h]
            feed_stop = [feed_width/2, substrate_L/2, h]
            port_start = [-feed_width/2, substrate_L/2, h]
            port_stop = [feed_width/2, substrate_L/2 - min(feed_len, (substrate_L - patch_L)/2), 0]
            port_dir = 'y'

        feed_metal.AddBox(priority=10, start=feed_start, stop=feed_stop)
        FDTD.AddEdges2Grid(dirs='xy', properties=feed_metal, metal_edge_res=mesh_res/2)

        port = FDTD.AddMSLPort(1, feed_metal, port_start, port_stop, port_dir, 'z', excite=-1,
                               FeedShift=10*mesh_res, MeasPlaneShift=feed_len/4, priority=5)

        mesh.SmoothMeshLines('all', mesh_res, 1.4)
        nf2ff = FDTD.CreateNF2FFBox()

        sim_path = os.path.abspath(work_dir)
        if cleanup and os.path.isdir(sim_path):
            import shutil
            shutil.rmtree(sim_path, ignore_errors=True)
        os.makedirs(sim_path, exist_ok=True)

        # Dense 3D sampling (user configurable)
        theta = np.arange(0.0, 181.0, max(0.5, float(theta_step_deg)))
        phi = np.arange(0.0, 361.0, max(1.0, float(phi_step_deg)))

        return OpenEMSPrepared(True, "Microstrip 3D prepared", FDTD=FDTD, nf=nf2ff, sim_path=sim_path,
                               theta=theta, phi=phi, nf_center=np.array([0.0, 0.0, h/2000.0]))
    except Exception as e:
        return OpenEMSPrepared(False, f"Microstrip 3D prepare failed: {e}")


def run_prepared_openems_microstrip_3d(
    prepared: OpenEMSPrepared,
    *,
    frequency_hz: float,
    verbose: int = 1,
) -> OpenEMSResult:
    try:
        if not prepared.ok or prepared.FDTD is None or prepared.nf is None:
            return OpenEMSResult(False, prepared.message)

        FDTD = prepared.FDTD
        nf2ff = prepared.nf
        sim_path = prepared.sim_path or "openems_out_microstrip"

        print(f"Starting microstrip FDTD (3D) in: {sim_path}")
        FDTD.Run(sim_path, verbose=verbose, cleanup=True)
        print("Microstrip FDTD (3D) completed!")

        theta = np.asarray(prepared.theta)
        phi = np.asarray(prepared.phi)
        f_res = frequency_hz

        # Sweep phi values and stack E_norm
        E_stack = []
        Dmax_val: Optional[float] = None
        for ph in phi:
            res = nf2ff.CalcNF2FF(sim_path, f_res, theta, np.array([ph]), center=prepared.nf_center)
            if res is None or not hasattr(res, 'E_norm'):
                return OpenEMSResult(False, "NF2FF returned None or missing E_norm")
            e = np.asarray(res.E_norm[0])
            e = np.squeeze(e)  # ensure (len(theta),)
            if e.ndim != 1:
                e = e.reshape(-1)
            E_stack.append(e)
            # Capture Dmax (directivity) once; expected to be constant for given frequency
            if Dmax_val is None and hasattr(res, 'Dmax'):
                try:
                    Dmax_val = float(np.asarray(res.Dmax)[0])
                except Exception:
                    Dmax_val = None

        E_arr = np.stack(E_stack, axis=1)  # (theta, phi)
        E_max = np.max(E_arr)
        if E_max <= 0:
            E_max = 1.0
        # Convert to absolute dBi using Dmax if available; else normalized to 0 dB max
        if Dmax_val is not None and Dmax_val > 0:
            intensity_dB = 20.0*np.log10(E_arr / E_max + 1e-16) + 10.0*np.log10(Dmax_val)
        else:
            intensity_dB = 20.0*np.log10(E_arr / E_max + 1e-16)

        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        return OpenEMSResult(True, "Microstrip 3D pattern computed", theta=theta_rad, phi=phi_rad,
                              intensity=intensity_dB, sim_path=sim_path, is_dBi=True)
    except Exception as e:
        return OpenEMSResult(False, f"Microstrip 3D run failed: {e}")


