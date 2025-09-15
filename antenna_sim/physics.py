from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Physical constants
c0 = 299_792_458.0
mu0 = 4 * math.pi * 1e-7
eps0 = 1.0 / (mu0 * c0 * c0)
eta0 = math.sqrt(mu0 / eps0)


def wavelength(f_hz: float) -> float:
    return c0 / f_hz


def effective_eps(eps_r: float, h_m: float, W_m: float) -> float:
    """Effective permittivity (Hammerstad-Jensen)."""
    if W_m <= 0 or h_m <= 0:
        return eps_r
    w_h = W_m / h_m
    term = 1.0 + 12.0 / w_h
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 / math.sqrt(term))
    if w_h > 1:
        eps_eff = ((eps_r + 1.0) / 2.0) + ((eps_r - 1.0) / 2.0) * (1.0 / math.sqrt(1.0 + 12.0 / w_h))
    return eps_eff


def delta_L(eps_eff: float, h_m: float, W_m: float) -> float:
    """Edge extension (Hammerstad-Jensen)."""
    if W_m <= 0 or h_m <= 0:
        return 0.0
    w_h = W_m / h_m
    a = (eps_eff + 0.3) * (w_h + 0.264)
    b = (eps_eff - 0.258) * (w_h + 0.8)
    return 0.412 * h_m * (a / b)


def design_patch_for_frequency(f_hz: float, eps_r: float, h_m: float) -> Tuple[float, float, float]:
    """Return (L_m, W_m, eps_eff) designed for TM10 resonance at f_hz."""
    W = c0 / (2 * f_hz) * math.sqrt(2.0 / (eps_r + 1.0))
    eps_eff = effective_eps(eps_r, h_m, W)
    L_eff = c0 / (2 * f_hz * math.sqrt(eps_eff))
    dL = delta_L(eps_eff, h_m, W)
    L = L_eff - 2.0 * dL
    return L, W, eps_eff


def jinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def rect_patch_power_pattern(L_eff: float, W: float, k0: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Unnormalized power pattern U(theta, phi) for a rectangular patch TM10.

    Two-slot model with element and array factors:
    - Array factor: separation L_eff along x → depends on sinθ·cosφ
    - Element factor: slot aperture along y → jinc(k0 W/2 · sinθ · sinφ)
    - Polarization factor for orthogonal components
    """
    th = theta
    ph = phi

    # Array factor (broadside maximum)
    a = 0.5 * k0 * L_eff * np.sin(th) * np.cos(ph)
    F_len = np.cos(a)

    # Element factor (slot of width W along y)
    x = 0.5 * k0 * W * np.sin(th) * np.sin(ph)
    F_wid = jinc(x)

    # Polarization mixture (dominant components)
    pol = (np.cos(ph) ** 2) + ((np.cos(th) ** 2) * (np.sin(ph) ** 2))

    U = (F_len ** 2) * (F_wid ** 2) * pol
    return U


def estimate_efficiency(eps_r: float, loss_tangent: float, conductivity_s_per_m: float, thickness_m: float, frequency_hz: float) -> float:
    """Crude overall efficiency estimate in [0.5, 0.98]."""
    eta_d = max(0.55, 1.0 - 1.6 * loss_tangent)
    sigma_ratio = min(1.2, conductivity_s_per_m / 5.8e7)
    thickness_ratio = min(1.5, max(0.2, thickness_m / 35e-6))
    freq_ghz = frequency_hz / 1e9
    eta_c = 0.93 * (sigma_ratio ** 0.2) * (thickness_ratio ** 0.05) / (1.0 + 0.02 * math.sqrt(max(0.0, freq_ghz - 1e-9)))
    eta_c = min(0.98, max(0.6, eta_c))
    eta = max(0.5, min(0.98, eta_d * eta_c))
    return eta
