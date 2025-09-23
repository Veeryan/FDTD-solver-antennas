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


def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Extrinsic rotations about global X, then Y, then Z.
    Row-vector convention used throughout: world = local @ R + T.
    Important: CSXCAD applies transforms in the given order using column vectors.
    For extrinsic X->Y->Z, the column-form rotation is: R_col = Rz @ Ry @ Rx.
    Therefore for our row-vector form, use the transpose: R = R_col.T.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return (Rz @ Ry @ Rx).T


def _transform_point_local_to_global(p_local_mm: Sequence[float], R: np.ndarray, T_mm: np.ndarray) -> list[float]:
    p = np.asarray(p_local_mm, dtype=float)
    # Row-vector: world = p(row) @ R + T
    return (p @ R + T_mm).tolist()


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
    nf_center_mode: str = "origin",  # 'origin' | 'centroid'
    simbox_mode: str = "auto",       # 'auto' | 'manual'
    auto_margin_mm: tuple[float, float, float] = (80.0, 80.0, 160.0),
    manual_size_mm: Optional[tuple[float, float, float]] = None,
    feed_line_length_mm: float = 20.0,
    port_mode: str = "lumped",  # 'auto' | 'lumped'
    end_criteria_db: float = -25.0,
    work_dir: str = "openems_out_multi",
    cleanup: bool = True,
    verbose: int = 0,
    log_cb: Optional[callable] = None,
) -> OpenEMSPrepared:
    """Prepare a multi-patch microstrip-fed openEMS 3D simulation.

    - Full 3D rotations supported via CSXCAD primitive transforms (Rx/Ry/Rz + translation).
    - Hybrid port strategy per element:
      - If the substrate normal is aligned with global Z and feed axis with X/Y, use an MSL port.
      - Otherwise, use a LumpedPort (coax-like) between feed strip and ground at the feed location.
    - All ports are excited simultaneously with equal amplitude/phase (current design choice).
    - NF2FF sampled on theta/phi grid; phase center configurable (origin/centroid).
    - Simulation box can be Auto (from oriented bounds + margins) or Manual.
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

        # Build global bounding box across all rotated substrates (full 3D)
        all_x: list[float] = []
        all_y: list[float] = []
        all_z: list[float] = []
        max_h = 0.0
        for inst in patches:
            W_mm, L_mm, h_mm = _compute_patch_dims_mm(inst.params)
            max_h = max(max_h, h_mm)
            substrate_margin = 30.0
            if inst.feed_direction in (FeedDirection.POS_X, FeedDirection.NEG_X):
                sub_W = W_mm + 2 * substrate_margin + feed_line_length_mm
                sub_L = L_mm + 2 * substrate_margin
            else:
                sub_W = W_mm + 2 * substrate_margin
                sub_L = L_mm + 2 * substrate_margin + feed_line_length_mm
            # Build rotation and translation
            rx, ry, rz = float(getattr(inst, 'rot_x_deg', 0.0)), float(getattr(inst, 'rot_y_deg', 0.0)), float(getattr(inst, 'rot_z_deg', 0.0))
            R = _rotation_matrix(rx, ry, rz)
            T = np.array([inst.center_x_m * 1e3, inst.center_y_m * 1e3, inst.center_z_m * 1e3], dtype=float)
            # Substrate oriented box corners (local, centered at origin)
            hx, hy, hz = sub_W/2.0, sub_L/2.0, h_mm/2.0
            corners = [
                [-hx,-hy,-hz],[ hx,-hy,-hz],[ hx, hy,-hz],[-hx, hy,-hz],
                [-hx,-hy, hz],[ hx,-hy, hz],[ hx, hy, hz],[-hx, hy, hz],
            ]
            local = np.array(corners, dtype=float)
            world = (local @ R) + T
            all_x.extend(world[:, 0].tolist())
            all_y.extend(world[:, 1].tolist())
            all_z.extend(world[:, 2].tolist())

        if not all_x or not all_y:
            return OpenEMSPrepared(False, "Failed to derive simulation bounds from patches.")

        x_min, x_max = float(min(all_x)), float(max(all_x))
        y_min, y_max = float(min(all_y)), float(max(all_y))
        z_min, z_max = float(min(all_z)), float(max(all_z))
        # Simulation box sizing
        if (simbox_mode or "auto").lower().startswith('man') and manual_size_mm is not None:
            SimBox_X, SimBox_Y, SimBox_Z = manual_size_mm
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            cz = 0.5 * (z_min + z_max)
        else:
            mx, my, mz = auto_margin_mm
            SimBox_X = (x_max - x_min) + 2 * float(mx)
            SimBox_Y = (y_max - y_min) + 2 * float(my)
            SimBox_Z = (z_max - z_min) + 2 * float(mz)
            # Center the sim box around the scene bounds center
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            cz = 0.5 * (z_min + z_max)

        def _log(msg: str):
            try:
                if log_cb is not None:
                    log_cb(msg)
                else:
                    print(msg)
            except Exception:
                try:
                    print(msg)
                except Exception:
                    pass

        if verbose:
            _log(f"SimBox (mm): X={SimBox_X:.1f} Y={SimBox_Y:.1f} Z={SimBox_Z:.1f}")

        # Determine mesh resolution (points-per-wavelength) and pick NrTS based on quality
        try:
            q = int(mesh_quality)
        except Exception:
            q = 3
        q = max(1, min(10, q))
        ppw_map = {
            1: 12.0, 2: 16.0, 3: 20.0, 4: 25.0, 5: 32.0,
            6: 40.0, 7: 50.0, 8: 65.0, 9: 80.0, 10: 100.0,
        }
        ppw = ppw_map.get(q, 20.0)
        mesh_res = C0 / (f0 + fc) / unit / ppw

        # Choose a larger max number of timesteps for finer meshes so the Gaussian excitation isn't truncated.
        # This prevents the "Requested excitation pulse would be ... timesteps. Cutting to max number of timesteps!" warning
        # and reduces the chance of hitting the max-timestep limit before the -40 dB end-criteria.
        if q <= 5:
            nr_ts = 30000
        elif q == 6:
            nr_ts = 50000
        elif q == 7:
            nr_ts = 70000
        elif q == 8:
            nr_ts = 100000
        elif q == 9:
            nr_ts = 130000
        else:
            nr_ts = 160000

        # Additional bump for very thin copper: tiny thickness can enforce a very small dt
        # Estimate excitation duration and dt to set a safer NrTS threshold even at modest mesh quality.
        try:
            # Excitation duration is approximately ~3.35/fc (empirical from openEMS logs)
            exc_dur = 3.35 / fc  # seconds
            # Estimate min spatial step from copper thickness or mesh_res (mm -> m)
            min_dim_mm = None
            try:
                # Gather min copper thickness across all patches (as used)
                t_list = []
                for inst in patches:
                    t_list.append(max(0.02, float(inst.params.metal.thickness_m) * 1e3))
                if t_list:
                    min_dim_mm = min(min(t_list), float(mesh_res))
                else:
                    min_dim_mm = float(mesh_res)
            except Exception:
                min_dim_mm = float(mesh_res)
            min_dim_m = float(min_dim_mm) * 1e-3
            # Courant-like dt estimate on conservative factor ~1.8
            dt_est = min_dim_m / (C0 * 1.8)
            exc_ts_est = max(10000, int(exc_dur / max(1e-15, dt_est)))
            nr_ts = max(nr_ts, min(220000, int(2.2 * exc_ts_est)))
            if verbose:
                _log(f"NrTS adjusted for thin metal: min_dim≈{min_dim_mm:.4f} mm, exc_ts_est≈{exc_ts_est}, NrTS→{nr_ts}")
        except Exception:
            pass

        if verbose:
            _log(f"Mesh: q={q} -> ppw={ppw:g}, mesh_res={mesh_res:.3f} mm, NrTS={nr_ts}")

        # Map end_criteria_db (dB) to linear EndCriteria for openEMS
        try:
            ec_db = float(end_criteria_db)
        except Exception:
            ec_db = -25.0
        # Clamp to sane range
        ec_db = max(-80.0, min(-10.0, ec_db))
        ec_lin = 10.0**(ec_db/20.0)
        if verbose:
            _log(f"Termination: EndCriteria={ec_db:g} dB -> {ec_lin:.6g}")

        # openEMS setup
        FDTD = openEMS(NrTS=nr_ts, EndCriteria=ec_lin)
        FDTD.SetGaussExcite(f0, fc)
        bc = ['MUR'] * 6 if boundary.upper().startswith('MUR') else ['PML_8'] * 6
        if verbose:
            _log(f"Boundary: {bc[0]} on all sides")
        FDTD.SetBoundaryCond(bc)

        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)

        # Base mesh lines centered on the scene bounds to ensure translated arrays are inside
        mesh.AddLine('x', [cx - SimBox_X/2, cx + SimBox_X/2])
        mesh.AddLine('y', [cy - SimBox_Y/2, cy + SimBox_Y/2])
        mesh.AddLine('z', [cz - SimBox_Z/2, cz + SimBox_Z/2])

        # Mesh resolution already computed above into mesh_res

        # Helper: add manual mesh planes spanning the bounding box of a rotated rectangle.
        # This compensates for CSXCAD's limitation: edges2grid cannot operate when transforms are present
        # and thin rotated metals may become "unused" if no grid plane intersects them.
        def _add_mesh_bbox_for_plane(corners_world: np.ndarray, density: int = 9):
            try:
                arr = np.asarray(corners_world, dtype=float)
                if arr.shape != (4, 3):
                    return
                nx = max(3, int(density))
                ny = nx
                nz = nx
                xs = np.linspace(float(arr[:, 0].min()), float(arr[:, 0].max()), nx).tolist()
                ys = np.linspace(float(arr[:, 1].min()), float(arr[:, 1].max()), ny).tolist()
                zs = np.linspace(float(arr[:, 2].min()), float(arr[:, 2].max()), nz).tolist()
                mesh.AddLine('x', xs)
                mesh.AddLine('y', ys)
                mesh.AddLine('z', zs)
            except Exception:
                pass

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

            # Local-to-world placement using rotation matrix + translation (row-vector convention)
            T = np.array([inst.center_x_m * 1e3, inst.center_y_m * 1e3, inst.center_z_m * 1e3], dtype=float)
            rx, ry, rz = float(getattr(inst, 'rot_x_deg', 0.0)), float(getattr(inst, 'rot_y_deg', 0.0)), float(getattr(inst, 'rot_z_deg', 0.0))
            R = _rotation_matrix(rx, ry, rz)
            t_cu_mm = max(0.02, float(p.metal.thickness_m) * 1e3)

            # Substrate (centered at local origin)
            sub_box = mat_sub.AddBox(priority=0, start=[-sub_W/2, -sub_L/2, -h_mm/2], stop=[sub_W/2, sub_L/2, h_mm/2])
            if abs(rx) > 1e-9: sub_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: sub_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: sub_box.AddTransform('RotateAxis', 'z', rz)
            sub_box.AddTransform('Translate', T.tolist())
            # Mesh through substrate thickness around its local z span, but along the world axis
            # that aligns with the rotated substrate normal. This keeps symmetry for rotated boards.
            try:
                n_world = np.array([0.0, 0.0, 1.0]) @ R
                th_axis = int(np.argmax(np.abs(n_world)))  # 0:x, 1:y, 2:z
                axis_char = 'xyz'[th_axis]
                # The metal planes are tangential to the thickness axis; choose those two axes for edge meshing
                plane_dirs = ''.join(ch for i, ch in enumerate('xyz') if i != th_axis)  # e.g., 'yz' if th_axis='x'
                c = float(T[th_axis])
                lines = np.linspace(c - h_mm/2, c + h_mm/2, 5)
                mesh.AddLine(axis_char, lines.tolist())
                if verbose:
                    _log(f"           Mesh thickness axis='{axis_char}' lines around {c:.2f}mm; plane_dirs='{plane_dirs}' for edges")
            except Exception:
                pass

            # Ground (thin metal) on local bottom face
            gnd_box = m_gnd.AddBox(priority=10, start=[-sub_W/2, -sub_L/2, -h_mm/2 - t_cu_mm/2], stop=[sub_W/2, sub_L/2, -h_mm/2 + t_cu_mm/2])
            if abs(rx) > 1e-9: gnd_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: gnd_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: gnd_box.AddTransform('RotateAxis', 'z', rz)
            gnd_box.AddTransform('Translate', T.tolist())
            # Manual mesh refinement across the rotated ground plane bounding box
            try:
                gnd_plane_local = [
                    [-sub_W/2, -sub_L/2, -h_mm/2],
                    [ sub_W/2, -sub_L/2, -h_mm/2],
                    [ sub_W/2,  sub_L/2, -h_mm/2],
                    [-sub_W/2,  sub_L/2, -h_mm/2],
                ]
                gnd_world = (np.asarray(gnd_plane_local, dtype=float) @ R) + T
                _add_mesh_bbox_for_plane(gnd_world, density=6 + 2*int(q))
            except Exception:
                pass

            # Patch (thin metal) on local top face
            patch_box = m_patch.AddBox(priority=10, start=[-W_mm/2, -L_mm/2, h_mm/2 - t_cu_mm/2], stop=[W_mm/2, L_mm/2, h_mm/2 + t_cu_mm/2])
            if abs(rx) > 1e-9: patch_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: patch_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: patch_box.AddTransform('RotateAxis', 'z', rz)
            patch_box.AddTransform('Translate', T.tolist())
            # Manual mesh refinement across the rotated patch plane bounding box
            try:
                patch_plane_local = [
                    [-W_mm/2, -L_mm/2,  h_mm/2],
                    [ W_mm/2, -L_mm/2,  h_mm/2],
                    [ W_mm/2,  L_mm/2,  h_mm/2],
                    [-W_mm/2,  L_mm/2,  h_mm/2],
                ]
                patch_world = (np.asarray(patch_plane_local, dtype=float) @ R) + T
                _add_mesh_bbox_for_plane(patch_world, density=6 + 2*int(q))
            except Exception:
                pass

            # Determine feed axis and feed point on the patch edge (local coords)
            if inst.feed_direction == FeedDirection.NEG_X:
                feed_axis_local  = np.array([1.0, 0.0, 0.0])
                feed_point_local = [-W_mm/2, 0.0, h_mm/2]
            elif inst.feed_direction == FeedDirection.POS_X:
                feed_axis_local  = np.array([1.0, 0.0, 0.0])
                feed_point_local = [ W_mm/2, 0.0, h_mm/2]
            elif inst.feed_direction == FeedDirection.NEG_Y:
                feed_axis_local  = np.array([0.0, 1.0, 0.0])
                feed_point_local = [0.0, -L_mm/2, h_mm/2]
            else:  # POS_Y
                feed_axis_local  = np.array([0.0, 1.0, 0.0])
                feed_point_local = [0.0,  L_mm/2, h_mm/2]

            # For LumpedPort operation (current default), do NOT draw a long feed trace.
            # Instead, place a small square feed pad on top of the patch near the feed point to localize currents.
            try:
                pad_w = max(1.0, float(feed_w))  # mm
            except Exception:
                pad_w = 1.0
            fx, fy = float(feed_point_local[0]), float(feed_point_local[1])
            pad_start = [fx - pad_w/2, fy - pad_w/2, h_mm/2 - t_cu_mm/2]
            pad_stop  = [fx + pad_w/2, fy + pad_w/2, h_mm/2 + t_cu_mm/2]
            pad_box = m_feed.AddBox(priority=11, start=pad_start, stop=pad_stop)
            if abs(rx) > 1e-9: pad_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: pad_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: pad_box.AddTransform('RotateAxis', 'z', rz)
            pad_box.AddTransform('Translate', T.tolist())
            try:
                pad_plane_local = [
                    [pad_start[0], pad_start[1], h_mm/2],
                    [pad_stop[0],  pad_start[1], h_mm/2],
                    [pad_stop[0],  pad_stop[1],  h_mm/2],
                    [pad_start[0], pad_stop[1],  h_mm/2],
                ]
                pad_world = (np.asarray(pad_plane_local, dtype=float) @ R) + T
                _add_mesh_bbox_for_plane(pad_world, density=8 + 2*int(q))
            except Exception:
                pass

            # Determine port strategy: MSL if axis-aligned; else Lumped
            # Compute world directions with rotation matrix
            normal_world = np.array([0.0, 0.0, 1.0]) @ R
            normal_world = normal_world / max(1e-12, np.linalg.norm(normal_world))
            feed_axis_world = feed_axis_local @ R
            feed_axis_world = feed_axis_world / max(1e-12, np.linalg.norm(feed_axis_world))
            aligned_normal = abs(normal_world[2]) >= 0.9999  # substrate normal ~ z
            aligned_feed = (abs(feed_axis_world[0]) >= 0.9999) or (abs(feed_axis_world[1]) >= 0.9999)
            # Temporarily disable MSL ports due to regression; always use LumpedPort
            use_msl = False
            if verbose:
                _log(f"Patch {idx}: center(mm)={np.round(T,3).tolist()} rot(deg)=(x={rx:g},y={ry:g},z={rz:g})")
                _log(f"           normal_world={np.round(normal_world,6)} feed_axis_world={np.round(feed_axis_world,6)} port_mode={port_mode} -> port={'MSL' if use_msl else 'Lumped'}")

            if not use_msl:
                # Lumped port at feed location bridging patch to ground along nearest world axis
                c_world = _transform_point_local_to_global(feed_point_local, R, T)
                # Choose world axis most aligned with substrate normal
                # Prefer Z in near-tie cases to span thickness more robustly for yaw rotations
                absn = np.abs(normal_world)
                axis = int(np.argmax(absn))
                try:
                    if abs(absn[2] - absn[axis]) < 1e-6:
                        axis = 2
                except Exception:
                    pass
                # p_dir for AddLumpedPort must be axis index (0:x,1:y,2:z)
                p_dir = axis
                # Projected thickness along chosen axis; ensure span traverses full dielectric thickness
                comp = float(abs(normal_world[axis])) if float(abs(normal_world[axis])) > 1e-6 else 1.0
                # Use exact world coordinates of ground/patch planes at the FEED x,y location
                # This is crucial for rotated boards where plane z varies with x/y
                try:
                    fx, fy = float(feed_point_local[0]), float(feed_point_local[1])
                except Exception:
                    fx, fy = 0.0, 0.0
                ground_c = _transform_point_local_to_global([fx, fy, -h_mm/2], R, T)
                patch_c  = _transform_point_local_to_global([fx, fy, +h_mm/2], R, T)
                eps = max(0.1, 0.25*mesh_res)  # small extension to ensure overlap with metal thickness
                # Previous stable behavior: derive min/max span and order start->stop so vector aligns with +normal_world
                axis_vals = sorted([ground_c[axis], patch_c[axis]])
                axis_min = float(axis_vals[0] - eps)
                axis_max = float(axis_vals[1] + eps)
                # Use a compact port cross-section for robustness on rotated boards
                # Limit lateral span to a fraction of the mesh resolution to avoid spanning far along the rotated normal
                half_w = max(0.4, min(0.6 * float(feed_w), 0.35 * float(mesh_res)))
                half_other = half_w
                sgn = 1.0 if float(normal_world[axis]) >= 0.0 else -1.0
                if axis == 0:
                    s0, s1 = (axis_min, axis_max) if sgn >= 0 else (axis_max, axis_min)
                    start = [s0, c_world[1] - half_w, c_world[2] - half_other]
                    stop  = [s1, c_world[1] + half_w, c_world[2] + half_other]
                elif axis == 1:
                    s0, s1 = (axis_min, axis_max) if sgn >= 0 else (axis_max, axis_min)
                    start = [c_world[0] - half_w, s0, c_world[2] - half_other]
                    stop  = [c_world[0] + half_w, s1, c_world[2] + half_other]
                else:  # axis == 2
                    s0, s1 = (axis_min, axis_max) if sgn >= 0 else (axis_max, axis_min)
                    start = [c_world[0] - half_w, c_world[1] - half_other, s0]
                    stop  = [c_world[0] + half_w, c_world[1] + half_other, s1]
                # Local mesh refinement around port
                try:
                    if axis == 2:
                        mesh.AddLine('z', [start[2], c_world[2], stop[2]])
                        mesh.AddLine('x', [start[0], c_world[0], stop[0]])
                        mesh.AddLine('y', [start[1], c_world[1], stop[1]])
                    elif axis == 0:
                        mesh.AddLine('x', [start[0], c_world[0], stop[0]])
                        mesh.AddLine('y', [start[1], c_world[1], stop[1]])
                        mesh.AddLine('z', [start[2], c_world[2], stop[2]])
                    else:
                        mesh.AddLine('y', [start[1], c_world[1], stop[1]])
                        mesh.AddLine('x', [start[0], c_world[0], stop[0]])
                        mesh.AddLine('z', [start[2], c_world[2], stop[2]])
                except Exception:
                    pass
                if verbose:
                    _log(f"           LumpedPort[{idx}] axis={p_dir} start={np.round(start,2)} stop={np.round(stop,2)} excite=+1")
                # Choose edges2grid plane perpendicular to the port axis for better local meshing
                try:
                    port_edges_dirs = ''.join(ch for i, ch in enumerate('xyz') if i != p_dir)
                except Exception:
                    port_edges_dirs = 'xy'
                FDTD.AddLumpedPort(idx, 50.0, start, stop, p_dir, excite=+1, priority=5, edges2grid=port_edges_dirs)
                # Diagnostics for lumped port alignment
                if verbose:
                    try:
                        v = np.array(stop) - np.array(start)
                        axis_span = float(abs(v[p_dir]))
                        lateral_span = float(np.linalg.norm(np.delete(v, p_dir)))
                        diag_span = float(np.linalg.norm(v))
                        axis_ratio = axis_span / max(1e-12, diag_span)
                        # Distances of start/stop to ground/patch planes along the rotated normal
                        n = np.array(normal_world); n = n / max(1e-12, np.linalg.norm(n))
                        # Recompute centers used above
                        fx, fy = float(feed_point_local[0]), float(feed_point_local[1])
                        ground_c = _transform_point_local_to_global([fx, fy, -h_mm/2], R, T)
                        patch_c  = _transform_point_local_to_global([fx, fy, +h_mm/2], R, T)
                        d_start_ground = float(np.dot(np.array(start) - ground_c, n))
                        d_stop_patch   = float(np.dot(np.array(stop)  - patch_c,  n))
                        _log(f"           diag: axis_span={axis_span:.3f}mm lateral_span={lateral_span:.3f}mm axis_ratio={axis_ratio:.3f}")
                        _log(f"           diag: d(start,ground-plane)={d_start_ground:.3f}mm  d(stop,patch-plane)={d_stop_patch:.3f}mm")
                    except Exception:
                        pass

        mesh.SmoothMeshLines('all', mesh_res, 1.4)
        nf2ff = FDTD.CreateNF2FFBox()

        # Use a unique path per invocation to avoid collisions across GUI instances
        suffix = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}-{random.randint(1000,9999)}"
        sim_path = os.path.abspath(f"{work_dir}_{suffix}")
        if cleanup and os.path.isdir(sim_path):
            import shutil
            shutil.rmtree(sim_path, ignore_errors=True)
        # Do not pre-create the folder here; FDTD.Run will create it.

        # NF2FF phase center
        if (nf_center_mode or 'origin').lower().startswith('cent'):
            cx = float(np.mean([p.center_x_m for p in patches])) * 1e3
            cy = float(np.mean([p.center_y_m for p in patches])) * 1e3
            cz = float(np.mean([p.center_z_m for p in patches])) * 1e3 + max_h/2000.0
        else:
            cx, cy, cz = 0.0, 0.0, max_h/2000.0

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

        # Diagnostics: report direction of maximum field (world coordinates)
        try:
            if E_arr.size:
                t_idx, p_idx = np.unravel_index(int(np.argmax(E_arr)), E_arr.shape)
                th = float(theta_rad[t_idx])
                ph = float(phi_rad[p_idx])
                x = math.sin(th) * math.cos(ph)
                y = math.sin(th) * math.sin(ph)
                z = math.cos(th)
                print(f"NF2FF max at theta={math.degrees(th):.1f}°, phi={math.degrees(ph):.1f}°  dir≈[{x:.3f},{y:.3f},{z:.3f}]")
        except Exception:
            pass

        return OpenEMSResult(True, "Microstrip multi-3D pattern computed", theta=theta_rad, phi=phi_rad,
                              intensity=intensity_dB, sim_path=sim_path, is_dBi=True)
    except Exception as e:
        return OpenEMSResult(False, f"Microstrip multi-3D run failed: {e}")
