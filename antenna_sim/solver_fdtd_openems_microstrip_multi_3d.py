from __future__ import annotations

from typing import List, Optional, Sequence
import os
import time
import random
import math
import numpy as np

from .models import PatchAntennaParams
from .solver_fdtd_openems_fixed import OpenEMSPrepared, OpenEMSResult
from .solver_fdtd_openems_microstrip import (
    FeedDirection,
    _find_openems_dir,
    _add_dll_dirs,
    calculate_microstrip_width,
)


class PatchLike:
    """Lightweight Protocol-like class for the attributes we need from MultiPatchPanel.PatchInstance.
    We avoid importing GUI classes here. Any object with these attributes will work.
    """
    name: str
    params: PatchAntennaParams
    center_x_m: float
    center_y_m: float
    center_z_m: float
    feed_direction: FeedDirection
    rot_x_deg: float
    rot_y_deg: float
    rot_z_deg: float


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _transform_point_local_to_global(p_local_mm: Sequence[float], Rz: np.ndarray, T_mm: np.ndarray) -> list[float]:
    p = np.asarray(p_local_mm, dtype=float)
    return (Rz @ p + T_mm).tolist()


def _compute_patch_dims_mm(params: PatchAntennaParams) -> tuple[float, float, float]:
    """Return (patch_W_mm, patch_L_mm, h_mm)."""
    if params.patch_length_m and params.patch_width_m:
        patch_L = float(params.patch_length_m) * 1e3
        patch_W = float(params.patch_width_m) * 1e3
    else:
        from .physics import design_patch_for_frequency
        L_m, W_m, _ = design_patch_for_frequency(params.frequency_hz, params.eps_r, params.h_m)
        patch_L = L_m * 1e3
        patch_W = W_m * 1e3
    h = float(params.h_m) * 1e3
    return patch_W, patch_L, h


def _rect_bounds_world(x0: float, x1: float, y0: float, y1: float, z: float, Rz: np.ndarray, T_mm: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute axis-aligned world start/stop for a local axis-aligned rectangle at fixed z.

    Only valid for Z-rotations of 0/90/180/270 degrees (Rz orthonormal with axes swapped or flipped).
    Returns two 3D points: [xmin, ymin, z] and [xmax, ymax, z].
    """
    corners = np.array([
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
    ], dtype=float)
    world = (corners @ Rz.T) + T_mm
    x_min, y_min = float(np.min(world[:, 0])), float(np.min(world[:, 1]))
    x_max, y_max = float(np.max(world[:, 0])), float(np.max(world[:, 1]))
    return [x_min, y_min, z], [x_max, y_max, z]


def prepare_openems_microstrip_multi_3d(
    patches: List[PatchLike],
    *,
    dll_dir: str,
    boundary: str = "MUR",
    theta_step_deg: float = 2.0,
    phi_step_deg: float = 5.0,
    mesh_quality: int = 3,
    feed_line_length_mm: float = 20.0,
    work_dir: str = "openems_out_multi",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMSPrepared:
    """Prepare a multi-patch microstrip-fed openEMS 3D simulation.

    - Places each patch instance (substrate, ground, patch, feed) using affine transforms.
    - Adds a dedicated MSL port per element; all ports are excited simultaneously.
    - NF2FF is sampled on a dense theta/phi grid like the single-patch 3D solver.

    Initial constraints for stability with MSL ports:
    - Rotations around X/Y are ignored (treated as 0). Only Z-rotation is supported.
    - Z-rotation is snapped to the nearest multiple of 90° for correct MSL port orientation.
    """
    try:
        if not patches:
            return OpenEMSPrepared(False, "No patch instances provided.")

        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMSPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        _add_dll_dirs(resolved)

        # Imports from openEMS/CSXCAD
        from CSXCAD import ContinuousStructure  # type: ignore
        from openEMS import openEMS  # type: ignore
        from openEMS.physical_constants import C0, EPS0  # type: ignore

        # Global settings
        unit = 1e-3  # mm
        f0 = float(patches[0].params.frequency_hz)
        fc = f0 / 2.0

        # Build global bounding box (XY) across all rotated substrates
        all_x: list[float] = []
        all_y: list[float] = []
        max_h = 0.0
        for inst in patches:
            W_mm, L_mm, h_mm = _compute_patch_dims_mm(inst.params)
            max_h = max(max_h, h_mm)
            # Substrate size includes margin + feed length depending on feed direction
            substrate_margin = 30.0
            if inst.feed_direction in (FeedDirection.POS_X, FeedDirection.NEG_X):
                sub_W = W_mm + 2 * substrate_margin + feed_line_length_mm
                sub_L = L_mm + 2 * substrate_margin
            else:
                sub_W = W_mm + 2 * substrate_margin
                sub_L = L_mm + 2 * substrate_margin + feed_line_length_mm
            # Local rectangle corners at z=0 in mm
            half_W, half_L = sub_W / 2.0, sub_L / 2.0
            corners = np.array([
                [-half_W, -half_L, 0.0],
                [ half_W, -half_L, 0.0],
                [ half_W,  half_L, 0.0],
                [-half_W,  half_L, 0.0],
            ])
            T = np.array([inst.center_x_m * 1e3, inst.center_y_m * 1e3, inst.center_z_m * 1e3], dtype=float)
            rz_snap = round(float(inst.rot_z_deg) / 90.0) * 90.0
            R = _rot_z(rz_snap)
            world = (corners @ R.T) + T
            all_x.extend(world[:, 0].tolist())
            all_y.extend(world[:, 1].tolist())

        if not all_x or not all_y:
            return OpenEMSPrepared(False, "Failed to derive simulation bounds from patches.")

        x_min, x_max = float(min(all_x)), float(max(all_x))
        y_min, y_max = float(min(all_y)), float(max(all_y))
        # Air margin around the envelope
        air_margin = 80.0
        SimBox_X = (x_max - x_min) + 2 * air_margin
        SimBox_Y = (y_max - y_min) + 2 * air_margin
        SimBox_Z = max(160.0, 6 * max_h)

        if verbose:
            print(f"SimBox (mm): X={SimBox_X:.1f} Y={SimBox_Y:.1f} Z={SimBox_Z:.1f}")

        # openEMS setup
        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
        FDTD.SetGaussExcite(f0, fc)
        bc = ['MUR'] * 6 if boundary.upper().startswith('MUR') else ['PML_8'] * 6
        FDTD.SetBoundaryCond(bc)

        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)

        # Base mesh lines
        mesh.AddLine('x', [-SimBox_X/2, SimBox_X/2])
        mesh.AddLine('y', [-SimBox_Y/2, SimBox_Y/2])
        mesh.AddLine('z', [-SimBox_Z/3, SimBox_Z*2/3])

        # Mesh resolution driven by quality level (points-per-wavelength)
        try:
            q = int(mesh_quality)
        except Exception:
            q = 3
        q = max(1, min(5, q))
        ppw_map = {1: 12.0, 2: 16.0, 3: 20.0, 4: 25.0, 5: 32.0}
        ppw = ppw_map.get(q, 20.0)
        mesh_res = C0 / (f0 + fc) / unit / ppw

        # Shared theta/phi sampling
        theta = np.arange(0.0, 181.0, max(0.5, float(theta_step_deg)))
        phi = np.arange(0.0, 361.0, max(1.0, float(phi_step_deg)))

        # Single shared materials for simplicity
        # (Different eps_r per element would need one material per instance.)

        # We will create separate metal properties per element to simplify edge meshing.
        for idx, inst in enumerate(patches, start=1):
            p = inst.params
            W_mm, L_mm, h_mm = _compute_patch_dims_mm(p)
            feed_w = calculate_microstrip_width(p.frequency_hz, p.eps_r, p.h_m) * 1e3
            substrate_margin = 30.0
            if inst.feed_direction in (FeedDirection.POS_X, FeedDirection.NEG_X):
                sub_W = W_mm + 2 * substrate_margin + feed_line_length_mm
                sub_L = L_mm + 2 * substrate_margin
            else:
                sub_W = W_mm + 2 * substrate_margin
                sub_L = L_mm + 2 * substrate_margin + feed_line_length_mm

            # Materials & metals for this element
            substrate_kappa = 2*np.pi*p.frequency_hz * EPS0*p.eps_r * p.loss_tangent
            mat_sub = CSX.AddMaterial(f'substrate_{idx}', epsilon=p.eps_r, kappa=substrate_kappa)
            m_gnd = CSX.AddMetal(f'ground_{idx}')
            m_patch = CSX.AddMetal(f'patch_{idx}')
            m_feed = CSX.AddMetal(f'feed_{idx}')

            # Local-to-world placement (no CSX transforms). Z-rotations snapped to 90° multiples only.
            T = np.array([inst.center_x_m * 1e3, inst.center_y_m * 1e3, inst.center_z_m * 1e3], dtype=float)
            rz_snap = round(float(inst.rot_z_deg) / 90.0) * 90.0
            R = _rot_z(rz_snap)
            if verbose and abs(rz_snap - float(inst.rot_z_deg)) > 1e-6:
                print(f"[warn] '{inst.name}': Z-rotation snapped from {inst.rot_z_deg}° to {rz_snap}° for MSL port alignment.")
            if (abs(getattr(inst, 'rot_x_deg', 0.0)) > 1e-6) or (abs(getattr(inst, 'rot_y_deg', 0.0)) > 1e-6):
                print(f"[warn] '{inst.name}': rotations about X/Y are ignored in this version.")

            # Substrate box from z0..z1 in WORLD coordinates
            z0 = float(T[2])
            z1 = z0 + h_mm
            sub_local_start = [-sub_W/2, -sub_L/2]
            sub_local_stop  = [ sub_W/2,  sub_L/2]
            sub0, _ = _rect_bounds_world(sub_local_start[0], sub_local_stop[0], sub_local_start[1], sub_local_stop[1], z0, R, T)
            _,   sub1 = _rect_bounds_world(sub_local_start[0], sub_local_stop[0], sub_local_start[1], sub_local_stop[1], z1, R, T)
            sub_start_world = [sub0[0], sub0[1], z0]
            sub_stop_world  = [sub1[0], sub1[1], z1]
            mat_sub.AddBox(priority=0, start=sub_start_world, stop=sub_stop_world)
            # Add vertical mesh refinement across substrate thickness
            mesh.AddLine('z', np.linspace(z0, z1, 5))

            # Ground plane (z=z0) in WORLD coordinates
            gnd_start_world = [sub_start_world[0], sub_start_world[1], z0]
            gnd_stop_world  = [sub_stop_world[0],  sub_stop_world[1],  z0]
            m_gnd.AddBox(priority=10, start=gnd_start_world, stop=gnd_stop_world)
            FDTD.AddEdges2Grid(dirs='xy', properties=m_gnd)

            # Patch (top metal at z=z1) in WORLD coordinates
            patch_local_start = [-W_mm/2, -L_mm/2]
            patch_local_stop  = [ W_mm/2,  L_mm/2]
            p0, p1 = _rect_bounds_world(patch_local_start[0], patch_local_stop[0], patch_local_start[1], patch_local_stop[1], z1, R, T)
            patch_start_world = [p0[0], p0[1], z1]
            patch_stop_world  = [p1[0], p1[1], z1]
            m_patch.AddBox(priority=10, start=patch_start_world, stop=patch_stop_world)
            FDTD.AddEdges2Grid(dirs='xy', properties=m_patch, metal_edge_res=mesh_res/2)

            # Feed line (on top of substrate, z=z1) in WORLD coordinates, plus MSL port
            if inst.feed_direction == FeedDirection.NEG_X:
                feed_local_start = [-sub_W/2, -feed_w/2]
                feed_local_stop  = [-W_mm/2,   feed_w/2]
                port_local_start = [-sub_W/2, -feed_w/2, h_mm]  # local top to bottom
                port_local_stop  = [-sub_W/2 + min(feed_line_length_mm, (sub_W - W_mm)/2), feed_w/2, 0]
                feed_axis  = np.array([1.0, 0.0, 0.0])
            elif inst.feed_direction == FeedDirection.POS_X:
                feed_local_start = [ W_mm/2,  -feed_w/2]
                feed_local_stop  = [ sub_W/2,  feed_w/2]
                port_local_start = [ sub_W/2, -feed_w/2, h_mm]
                port_local_stop  = [ sub_W/2 - min(feed_line_length_mm, (sub_W - W_mm)/2), feed_w/2, 0]
                feed_axis  = np.array([1.0, 0.0, 0.0])
            elif inst.feed_direction == FeedDirection.NEG_Y:
                feed_local_start = [-feed_w/2, -sub_L/2]
                feed_local_stop  = [ feed_w/2, -L_mm/2]
                port_local_start = [-feed_w/2, -sub_L/2, h_mm]
                port_local_stop  = [ feed_w/2, -sub_L/2 + min(feed_line_length_mm, (sub_L - L_mm)/2), 0]
                feed_axis  = np.array([0.0, 1.0, 0.0])
            else:  # POS_Y
                feed_local_start = [-feed_w/2,  L_mm/2]
                feed_local_stop  = [ feed_w/2,  sub_L/2]
                port_local_start = [-feed_w/2,  sub_L/2, h_mm]
                port_local_stop  = [ feed_w/2,  sub_L/2 - min(feed_line_length_mm, (sub_L - L_mm)/2), 0]
                feed_axis  = np.array([0.0, 1.0, 0.0])

            # Feed metal bounds in WORLD
            x0 = min(feed_local_start[0], feed_local_stop[0])
            x1 = max(feed_local_start[0], feed_local_stop[0])
            y0 = min(feed_local_start[1], feed_local_stop[1])
            y1 = max(feed_local_start[1], feed_local_stop[1])
            fs, fe = _rect_bounds_world(x0, x1, y0, y1, z1, R, T)
            m_feed.AddBox(priority=10, start=fs, stop=fe)
            FDTD.AddEdges2Grid(dirs='xy', properties=m_feed, metal_edge_res=mesh_res/2)

            # Port placement in WORLD coordinates
            p_start = _transform_point_local_to_global(port_local_start, R, T)
            p_stop  = _transform_point_local_to_global(port_local_stop,  R, T)
            axis_dir_world = R @ feed_axis
            port_dir = 'x' if abs(axis_dir_world[0]) >= abs(axis_dir_world[1]) else 'y'

            # Ensure sufficient mesh lines along propagation direction near the port (>=5 lines)
            if port_dir == 'x':
                x_min = min(p_start[0], p_stop[0])
                x_max = max(p_start[0], p_stop[0])
                width = abs(fe[1] - fs[1])
                # Dense lines across propagation segment
                mesh.AddLine('x', np.linspace(x_min - max(1.0, width), x_max + max(1.0, width), 7))
                # Feed width refinement (perpendicular direction)
                y_c = 0.5 * (fs[1] + fe[1])
                mesh.AddLine('y', [fs[1], y_c, fe[1]])
            else:  # port_dir == 'y'
                y_min = min(p_start[1], p_stop[1])
                y_max = max(p_start[1], p_stop[1])
                width = abs(fe[0] - fs[0])
                mesh.AddLine('y', np.linspace(y_min - max(1.0, width), y_max + max(1.0, width), 7))
                x_c = 0.5 * (fs[0] + fe[0])
                mesh.AddLine('x', [fs[0], x_c, fe[0]])

            # Add the MSL port
            FDTD.AddMSLPort(
                idx, m_feed, p_start, p_stop, port_dir, 'z',
                excite=-1,
                FeedShift=10*mesh_res,
                MeasPlaneShift=feed_line_length_mm/4,
                priority=5,
            )

        mesh.SmoothMeshLines('all', mesh_res, 1.4)
        nf2ff = FDTD.CreateNF2FFBox()

        # Use a unique path per invocation to avoid collisions across GUI instances
        suffix = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{random.randint(1000,9999)}"
        sim_path = os.path.abspath(f"{work_dir}_{suffix}")
        if cleanup and os.path.isdir(sim_path):
            import shutil
            shutil.rmtree(sim_path, ignore_errors=True)
        # Do not pre-create the folder here; FDTD.Run will create it.

        # NF2FF phase center: average of instance centers
        cx = float(np.mean([p.center_x_m for p in patches])) * 1e3
        cy = float(np.mean([p.center_y_m for p in patches])) * 1e3
        cz = float(np.mean([p.center_z_m for p in patches])) * 1e3 + max_h/2000.0

        return OpenEMSPrepared(
            True,
            "Microstrip multi-antenna 3D prepared",
            FDTD=FDTD,
            nf=nf2ff,
            sim_path=sim_path,
            theta=theta,
            phi=phi,
            nf_center=np.array([cx, cy, cz], dtype=float),
        )
    except Exception as e:
        return OpenEMSPrepared(False, f"Microstrip multi-3D prepare failed: {e}")


def run_prepared_openems_microstrip_multi_3d(
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
        sim_path = prepared.sim_path or "openems_out_multi"

        print(f"Starting multi-antenna microstrip FDTD (3D) in: {sim_path}")
        FDTD.Run(sim_path, verbose=verbose, cleanup=True)
        print("Multi-antenna microstrip FDTD (3D) completed!")

        theta = np.asarray(prepared.theta)
        phi = np.asarray(prepared.phi)
        f_res = frequency_hz

        # Sweep phi; stack E_norm like the single-antenna 3D solver
        E_stack = []
        Dmax_val: Optional[float] = None
        for ph in phi:
            res = nf2ff.CalcNF2FF(sim_path, f_res, theta, np.array([ph]), center=prepared.nf_center)
            if res is None or not hasattr(res, 'E_norm'):
                return OpenEMSResult(False, "NF2FF returned None or missing E_norm")
            e = np.asarray(res.E_norm[0])
            e = np.squeeze(e)
            if e.ndim != 1:
                e = e.reshape(-1)
            E_stack.append(e)
            if Dmax_val is None and hasattr(res, 'Dmax'):
                try:
                    Dmax_val = float(np.asarray(res.Dmax)[0])
                except Exception:
                    Dmax_val = None

        E_arr = np.stack(E_stack, axis=1)
        E_max = float(np.max(E_arr)) if E_arr.size else 1.0
        if E_max <= 0:
            E_max = 1.0
        if Dmax_val is not None and Dmax_val > 0:
            intensity_dB = 20.0*np.log10(E_arr / E_max + 1e-16) + 10.0*np.log10(Dmax_val)
        else:
            intensity_dB = 20.0*np.log10(E_arr / E_max + 1e-16)

        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)

        return OpenEMSResult(True, "Microstrip multi-3D pattern computed", theta=theta_rad, phi=phi_rad,
                              intensity=intensity_dB, sim_path=sim_path, is_dBi=True)
    except Exception as e:
        return OpenEMSResult(False, f"Microstrip multi-3D run failed: {e}")
