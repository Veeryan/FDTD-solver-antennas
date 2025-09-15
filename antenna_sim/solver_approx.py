from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .models import PatchAntennaParams
from .physics import (
    c0,
    wavelength,
    effective_eps,
    delta_L,
    design_patch_for_frequency,
    rect_patch_power_pattern,
    estimate_efficiency,
)


@dataclass
class SolverResult:
    theta: np.ndarray
    phi: np.ndarray
    directivity: np.ndarray  # linear
    gain: np.ndarray  # linear
    peak_directivity_lin: float
    peak_gain_lin: float


class AnalyticalPatchSolver:
    def __init__(self, params: PatchAntennaParams):
        self.params = params
        self._resolved_dimensions()

    def _resolved_dimensions(self) -> None:
        p = self.params
        if p.patch_width_m is None or p.patch_length_m is None:
            L, W, eps_eff = design_patch_for_frequency(p.frequency_hz, p.eps_r, p.h_m)
            self.L_m = L
            self.W_m = W
            self.eps_eff = eps_eff
        else:
            self.L_m = p.patch_length_m
            self.W_m = p.patch_width_m
            self.eps_eff = effective_eps(p.eps_r, p.h_m, p.patch_width_m)
        self.dL_m = delta_L(self.eps_eff, p.h_m, self.W_m)
        self.L_eff_m = self.L_m + 2.0 * self.dL_m

    # ---- Core computations ----
    def compute_full_pattern(self, num_theta: int = 181, num_phi: int = 361) -> SolverResult:
        f = self.params.frequency_hz
        k0 = 2.0 * math.pi / wavelength(f)
        theta = np.linspace(0.0, math.pi, num_theta)
        phi = np.linspace(0.0, 2.0 * math.pi, num_phi)
        th, ph = np.meshgrid(theta, phi, indexing="ij")

        U = rect_patch_power_pattern(self.L_eff_m, self.W_m, k0, th, ph)
        # Normalize and integrate for directivity
        U_max = float(np.max(U))
        if U_max <= 0:
            U_max = 1.0
        # Power integral: ∫∫ U sinθ dθ dφ
        sin_th = np.sin(th)
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        Prad = float(np.sum(U * sin_th) * dtheta * dphi)
        D = 4.0 * math.pi * U / Prad

        # Efficiency -> gain
        eta = estimate_efficiency(
            self.params.eps_r,
            self.params.loss_tangent,
            self.params.metal.conductivity_s_per_m,
            self.params.metal.thickness_m,
            self.params.frequency_hz,
        )
        G = eta * D

        return SolverResult(
            theta=theta,
            phi=phi,
            directivity=D,
            gain=G,
            peak_directivity_lin=float(np.max(D)),
            peak_gain_lin=float(np.max(G)),
        )

    # ---- Convenience helpers ----
    def cross_section_gain_lin(self, plane: str = "E", num_theta: int = 721) -> Tuple[np.ndarray, np.ndarray]:
        """Return (theta, gain_linear) for phi=0 (E-plane) or phi=90deg (H-plane)."""
        theta = np.linspace(0.0, math.pi, num_theta)
        if plane.upper() == "E":
            phi_value = 0.0
        else:
            phi_value = math.pi / 2.0
        phi = np.array([phi_value])
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        f = self.params.frequency_hz
        k0 = 2.0 * math.pi * f / c0
        U = rect_patch_power_pattern(self.L_eff_m, self.W_m, k0, th, ph)[:, 0]

        # Compute overall directivity/gain scaling using full integral
        full = self.compute_full_pattern(num_theta=361, num_phi=361)
        # Map this cut to directivity using the same peak
        # Normalize U to full peak and multiply by peak directivity
        U_norm = U / np.max(U)
        D_cut = U_norm * full.peak_directivity_lin
        eta = full.peak_gain_lin / full.peak_directivity_lin
        G_cut = eta * D_cut
        return theta, G_cut

    @staticmethod
    def lin_to_dbi(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(1e-16, x))

    def summary(self) -> Dict[str, float]:
        res = self.compute_full_pattern()
        return {
            "L_mm": self.L_m * 1e3,
            "W_mm": self.W_m * 1e3,
            "L_eff_mm": self.L_eff_m * 1e3,
            "efficiency": float(res.peak_gain_lin / res.peak_directivity_lin),
            "D0_dBi": 10.0 * math.log10(res.peak_directivity_lin),
            "G0_dBi": 10.0 * math.log10(res.peak_gain_lin),
        }
