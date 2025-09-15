from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import PatchAntennaParams


@dataclass
class OpenEMS2DPrepared:
    ok: bool
    message: str
    FDTD: Optional[object] = None
    nf: Optional[object] = None
    sim_path: Optional[str] = None
    theta: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    nf_center: Optional[np.ndarray] = None


def _coerce_win_path(p: str) -> str:
    drive, _ = os.path.splitdrive(p)
    if drive:
        return os.path.normpath(p)
    if p.startswith("/") and len(p) > 3 and p[2] == "/" and p[1].isalpha():
        drive = p[1].upper()
        return os.path.normpath(f"{drive}:{p[2:].replace('/', os.sep)}")
    return os.path.normpath(os.path.abspath(p))


def _find_openems_dir(base_dir: str) -> Optional[str]:
    if not base_dir:
        return None
    for cand in (base_dir, os.path.join(base_dir, "openEMS")):
        cand_abs = _coerce_win_path(cand)
        if os.path.isdir(cand_abs):
            if any(os.path.isfile(os.path.join(cand_abs, d)) for d in ("CSXCAD.dll", "openEMS.dll")):
                return cand_abs
    return None


def _add_dll_dirs(root: str) -> None:
    root = _coerce_win_path(root)
    os.environ["OPENEMS_INSTALL_PATH"] = root
    os.add_dll_directory(root)
    qt = os.path.join(root, "qt5")
    if os.path.isdir(qt):
        os.add_dll_directory(qt)


def prepare_openems_patch_2d(
    params: PatchAntennaParams,
    *,
    dll_dir: str,
    work_dir: str = "openems_out_2d",
    cleanup: bool = True,
    verbose: int = 0,
) -> OpenEMS2DPrepared:
    try:
        resolved = _find_openems_dir(dll_dir)
        if not resolved:
            return OpenEMS2DPrepared(False, f"Could not find CSXCAD/openEMS DLLs in '{dll_dir}'.")
        _add_dll_dirs(resolved)

        from openEMS import openEMS as oem  # type: ignore
        from openEMS import CSXCAD  # type: ignore
        from openEMS.physical_constants import EPS0  # type: ignore

        f0 = params.frequency_hz
        c0 = 299792458.0
        unit = 1e-3  # mm

        # Map W to x, L to y
        if params.patch_length_m and params.patch_width_m:
            L = params.patch_length_m * 1e3
            W = params.patch_width_m * 1e3
        else:
            from .physics import design_patch_for_frequency
            L_m, W_m, _ = design_patch_for_frequency(f0, params.eps_r, params.h_m)
            L = L_m * 1e3
            W = W_m * 1e3

        h = params.h_m * 1e3

        # 2D-like: use a thin slice in y but keep a full simulation box like the tutorial
        slice_len = max(6.0, L/40.0)  # mm small thickness in y
        feed_x = -6.0  # mirror tutorial default feed position (mm)

        fc = f0 / 2.0
        res = c0 / (f0 + fc) / 1e-3 / 25.0  # slightly finer in 2D

        # Use similar stability settings as 3D
        FDTD = oem(NrTS=60000, EndCriteria=1e-5)
        FDTD.SetGaussExcite(f0, fc)
        FDTD.SetBoundaryCond([3,3,3,3,3,3])

        csx = CSXCAD.ContinuousStructure()
        FDTD.SetCSX(csx)
        mesh = csx.GetGrid()
        mesh.SetDeltaUnit(unit)

        # Simulation box identical style to test_openems.py
        SimBox = np.array([200.0, 200.0, 150.0])
        mesh.AddLine('x', [-SimBox[0]/2.0, -W/2.0, 0.0, W/2.0, SimBox[0]/2.0])
        mesh.AddLine('y', [-slice_len/2.0, 0.0, slice_len/2.0])  # thin slice but with center line
        mesh.AddLine('z', [-SimBox[2]/3.0, 0.0, h, SimBox[2]*2.0/3.0])

        # Substrate with loss
        kappa = 2.0 * np.pi * f0 * EPS0 * params.eps_r * max(0.0, params.loss_tangent)
        try:
            substrate = csx.AddMaterial('substrate', epsilon=params.eps_r, kappa=kappa)
        except TypeError:
            substrate = csx.AddMaterial('substrate')
            substrate.SetMaterialProperty('Eps', params.eps_r)
            substrate.SetMaterialProperty('Kappa', kappa)
        sub_w = 60.0
        sub_l = 60.0 if slice_len < 60.0 else slice_len
        substrate.AddBox([-sub_w/2.0, -sub_l/2.0, 0.0], [sub_w/2.0, sub_l/2.0, h])

        # Zero-thickness ground and patch
        gnd = csx.AddMetal('gnd')
        gnd.AddBox([-sub_w/2.0, -sub_l/2.0, 0.0], [sub_w/2.0, sub_l/2.0, 0.0], priority=10)

        patch = csx.AddMetal('patch')
        patch.AddBox([-W/2.0, -slice_len/2.0, h], [W/2.0, slice_len/2.0, h], priority=10)

        FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=res/2.0)
        FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

        # Port in z between ground and patch at (feed_x, 0)
        mesh.AddLine('x', [float(feed_x)])
        mesh.AddLine('z', [0.0, float(h)])
        port_start = [float(feed_x), 0.0, 0.0]
        port_stop  = [float(feed_x), 0.0, float(h)]
        try:
            FDTD.AddLumpedPort(1, 50, port_start, port_stop, 'z', 1.0, priority=5, edges2grid='xy')
        except Exception:
            FDTD.AddLumpedPort(1, 50, port_start, port_stop, 2, excite=1.0)

        mesh.SmoothMeshLines('all', res, 1.4)

        # Smooth mesh (tutorial) and then NF2FF default box
        # Add extra cells across substrate thickness (as in tutorial)
        mesh.AddLine('z', np.linspace(0.0, h, 4 + 1).tolist())
        mesh.SmoothMeshLines('all', res, 1.4)
        nf = FDTD.CreateNF2FFBox()

        sim_path = os.path.abspath(work_dir)
        if cleanup and os.path.isdir(sim_path):
            import shutil
            shutil.rmtree(sim_path, ignore_errors=True)
        os.makedirs(sim_path, exist_ok=True)

        theta = np.linspace(0, np.pi, 121)
        phi = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # cuts only for quasi-2D
        nf_center = np.array([0.0, 0.0, 1e-3], dtype=float)
        return OpenEMS2DPrepared(True, f"Prepared 2D-like (DLLs from: {resolved})", FDTD=FDTD, nf=nf, sim_path=sim_path, theta=theta, phi=phi, nf_center=nf_center)
    except Exception as e:
        return OpenEMS2DPrepared(False, f"prepare_2d failed: {e}")


