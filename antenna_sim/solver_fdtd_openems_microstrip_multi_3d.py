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
        else:
            mx, my, mz = auto_margin_mm
            SimBox_X = (x_max - x_min) + 2 * float(mx)
            SimBox_Y = (y_max - y_min) + 2 * float(my)
            SimBox_Z = (z_max - z_min) + 2 * float(mz)

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

        # openEMS setup
        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
        FDTD.SetGaussExcite(f0, fc)
        bc = ['MUR'] * 6 if boundary.upper().startswith('MUR') else ['PML_8'] * 6
        FDTD.SetBoundaryCond(bc)

        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(unit)

        # Base mesh lines (symmetric about origin to avoid bias)
        mesh.AddLine('x', [-SimBox_X/2, SimBox_X/2])
        mesh.AddLine('y', [-SimBox_Y/2, SimBox_Y/2])
        mesh.AddLine('z', [-SimBox_Z/2, SimBox_Z/2])

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
            try:
                FDTD.AddEdges2Grid(dirs=plane_dirs, properties=m_gnd)
            except Exception:
                FDTD.AddEdges2Grid(dirs='all', properties=m_gnd)

            # Patch (thin metal) on local top face
            patch_box = m_patch.AddBox(priority=10, start=[-W_mm/2, -L_mm/2, h_mm/2 - t_cu_mm/2], stop=[W_mm/2, L_mm/2, h_mm/2 + t_cu_mm/2])
            if abs(rx) > 1e-9: patch_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: patch_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: patch_box.AddTransform('RotateAxis', 'z', rz)
            patch_box.AddTransform('Translate', T.tolist())
            try:
                FDTD.AddEdges2Grid(dirs=plane_dirs, properties=m_patch, metal_edge_res=mesh_res/2)
            except Exception:
                FDTD.AddEdges2Grid(dirs='all', properties=m_patch, metal_edge_res=mesh_res/2)

            # Feed line on local top face
            if inst.feed_direction == FeedDirection.NEG_X:
                feed_local_start = [-sub_W/2, -feed_w/2, h_mm/2 - t_cu_mm/2]
                feed_local_stop  = [-W_mm/2,   feed_w/2, h_mm/2 + t_cu_mm/2]
                feed_axis_local  = np.array([1.0, 0.0, 0.0])
                feed_point_local = [-W_mm/2, 0.0, h_mm/2]
            elif inst.feed_direction == FeedDirection.POS_X:
                feed_local_start = [ W_mm/2,  -feed_w/2, h_mm/2 - t_cu_mm/2]
                feed_local_stop  = [ sub_W/2,  feed_w/2, h_mm/2 + t_cu_mm/2]
                feed_axis_local  = np.array([1.0, 0.0, 0.0])
                feed_point_local = [ W_mm/2, 0.0, h_mm/2]
            elif inst.feed_direction == FeedDirection.NEG_Y:
                feed_local_start = [-feed_w/2, -sub_L/2, h_mm/2 - t_cu_mm/2]
                feed_local_stop  = [ feed_w/2, -L_mm/2,  h_mm/2 + t_cu_mm/2]
                feed_axis_local  = np.array([0.0, 1.0, 0.0])
                feed_point_local = [0.0, -L_mm/2, h_mm/2]
            else:  # POS_Y
                feed_local_start = [-feed_w/2,  L_mm/2,  h_mm/2 - t_cu_mm/2]
                feed_local_stop  = [ feed_w/2,  sub_L/2, h_mm/2 + t_cu_mm/2]
                feed_axis_local  = np.array([0.0, 1.0, 0.0])
                feed_point_local = [0.0,  L_mm/2, h_mm/2]

            feed_box = m_feed.AddBox(priority=10, start=feed_local_start, stop=feed_local_stop)
            if abs(rx) > 1e-9: feed_box.AddTransform('RotateAxis', 'x', rx)
            if abs(ry) > 1e-9: feed_box.AddTransform('RotateAxis', 'y', ry)
            if abs(rz) > 1e-9: feed_box.AddTransform('RotateAxis', 'z', rz)
            feed_box.AddTransform('Translate', T.tolist())
            try:
                FDTD.AddEdges2Grid(dirs=plane_dirs, properties=m_feed, metal_edge_res=mesh_res/2)
            except Exception:
                FDTD.AddEdges2Grid(dirs='all', properties=m_feed, metal_edge_res=mesh_res/2)

            # Determine port strategy: MSL if axis-aligned; else Lumped
            # Compute world directions with rotation matrix
            normal_world = np.array([0.0, 0.0, 1.0]) @ R
            normal_world = normal_world / max(1e-12, np.linalg.norm(normal_world))
            feed_axis_world = feed_axis_local @ R
            feed_axis_world = feed_axis_world / max(1e-12, np.linalg.norm(feed_axis_world))
            aligned_normal = abs(normal_world[2]) >= 0.9999  # substrate normal ~ z
            aligned_feed = (abs(feed_axis_world[0]) >= 0.9999) or (abs(feed_axis_world[1]) >= 0.9999)
            use_msl = (port_mode.lower() == 'auto') and (aligned_normal and aligned_feed)
            if verbose:
                _log(f"Patch {idx}: center(mm)={np.round(T,3).tolist()} rot(deg)=(x={rx:g},y={ry:g},z={rz:g})")
                _log(f"           normal_world={np.round(normal_world,6)} feed_axis_world={np.round(feed_axis_world,6)} port_mode={port_mode} -> port={'MSL' if use_msl else 'Lumped'}")

            if use_msl:
                # Reuse prior MSL local port definition: start at top into dielectric
                if inst.feed_direction == FeedDirection.NEG_X:
                    # local top (+h/2) to bottom (-h/2)
                    port_local_start = [-sub_W/2, -feed_w/2, +h_mm/2]
                    port_local_stop  = [-sub_W/2 + min(feed_line_length_mm, (sub_W - W_mm)/2), +feed_w/2, -h_mm/2]
                elif inst.feed_direction == FeedDirection.POS_X:
                    port_local_start = [ +sub_W/2, -feed_w/2, +h_mm/2]
                    port_local_stop  = [ +sub_W/2 - min(feed_line_length_mm, (sub_W - W_mm)/2), +feed_w/2, -h_mm/2]
                elif inst.feed_direction == FeedDirection.NEG_Y:
                    port_local_start = [ -feed_w/2, -sub_L/2, +h_mm/2]
                    port_local_stop  = [ +feed_w/2, -sub_L/2 + min(feed_line_length_mm, (sub_L - L_mm)/2), -h_mm/2]
                else:
                    port_local_start = [ -feed_w/2,  +sub_L/2, +h_mm/2]
                    port_local_stop  = [ +feed_w/2,  +sub_L/2 - min(feed_line_length_mm, (sub_L - L_mm)/2), -h_mm/2]
                # Choose global port_dir by world orientation of the feed axis
                if abs(feed_axis_world[0]) >= 0.999:
                    port_dir = 'x'
                elif abs(feed_axis_world[1]) >= 0.999:
                    port_dir = 'y'
                else:
                    # Should not happen due to use_msl gating; fallback to lumped behavior
                    use_msl = False
                    port_dir = 'x'
                p_start = _transform_point_local_to_global(port_local_start, R, T)
                p_stop  = _transform_point_local_to_global(port_local_stop,  R, T)
                # Mesh refinement along propagation
                try:
                    if port_dir == 'x':
                        x_min = min(p_start[0], p_stop[0]); x_max = max(p_start[0], p_stop[0])
                        mesh.AddLine('x', np.linspace(x_min - max(1.0, feed_w), x_max + max(1.0, feed_w), 7))
                    else:
                        y_min = min(p_start[1], p_stop[1]); y_max = max(p_start[1], p_stop[1])
                        mesh.AddLine('y', np.linspace(y_min - max(1.0, feed_w), y_max + max(1.0, feed_w), 7))
                except Exception:
                    pass
                # Stable defaults: positive shifts independent of direction
                feed_shift = float(10*mesh_res)
                meas_shift = float(feed_line_length_mm/4)
                if verbose:
                    _log(f"           MSLPort[{idx}] dir={port_dir} start={np.round(p_start,2)} stop={np.round(p_stop,2)} excite=+1")
                FDTD.AddMSLPort(idx, m_feed, p_start, p_stop, port_dir, 'z', excite=-1,
                                FeedShift=feed_shift, MeasPlaneShift=meas_shift, priority=5)
                # Diagnostics: check that the port lies across the substrate thickness and aligns with normal/feed axes
                if verbose:
                    try:
                        # Top/bottom plane centers in world
                        top_c  = _transform_point_local_to_global([0.0, 0.0, +h_mm/2], R, T)
                        bot_c  = _transform_point_local_to_global([0.0, 0.0, -h_mm/2], R, T)
                        v = np.array(p_stop) - np.array(p_start)
                        v_n = v / max(1e-12, np.linalg.norm(v))
                        nw = np.array(normal_world)
                        nw_n = nw / max(1e-12, np.linalg.norm(nw))
                        align = float(np.dot(v_n, -nw_n))  # expect ~+1 (top->bottom)
                        # Signed distances of endpoints from the planes
                        def plane_signed_dist(P, Pc, n):
                            return float(np.dot(np.array(P) - np.array(Pc), n / max(1e-12, np.linalg.norm(n))))
                        ds_top  = plane_signed_dist(p_start, top_c, nw)
                        ds_bot  = plane_signed_dist(p_stop,  bot_c, nw)
                        _log(f"           diag: v_norm·(-n)={align:.5f}  d(start,top)={ds_top:.5f}mm  d(stop,bot)={ds_bot:.5f}mm")
                        # Feed axis vs port_dir
                        if port_dir == 'x':
                            pax = abs(feed_axis_world[0])
                            _log(f"           diag: feed_axis_world.x={pax:.6f} (expect ~1)")
                        else:
                            pay = abs(feed_axis_world[1])
                            _log(f"           diag: feed_axis_world.y={pay:.6f} (expect ~1)")
                    except Exception:
                        pass
            else:
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
                # Use exact world coordinates of ground/patch planes along this axis for robust contact
                ground_c = _transform_point_local_to_global([0.0, 0.0, -h_mm/2], R, T)
                patch_c  = _transform_point_local_to_global([0.0, 0.0, +h_mm/2], R, T)
                eps = max(0.1, 0.25*mesh_res)  # small extension to ensure overlap with metal thickness
                # Previous stable behavior: derive min/max span and order start->stop so vector aligns with +normal_world
                axis_vals = sorted([ground_c[axis], patch_c[axis]])
                axis_min = float(axis_vals[0] - eps)
                axis_max = float(axis_vals[1] + eps)
                # Slightly larger cross-section improves contact on coarse meshes when rotated
                half_w = max(1.0, 0.75 * float(feed_w))
                half_other = max(1.0, 0.75 * float(feed_w))
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
                        ground_c = _transform_point_local_to_global([0.0, 0.0, -h_mm/2], R, T)
                        patch_c  = _transform_point_local_to_global([0.0, 0.0, +h_mm/2], R, T)
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

        return OpenEMSResult(True, "Microstrip multi-3D pattern computed", theta=theta_rad, phi=phi_rad,
                              intensity=intensity_dB, sim_path=sim_path, is_dBi=True)
    except Exception as e:
        return OpenEMSResult(False, f"Microstrip multi-3D run failed: {e}")
