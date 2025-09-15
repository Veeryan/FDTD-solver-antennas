from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Metal(str, Enum):
    COPPER = "copper"
    ALUMINUM = "aluminum"
    GOLD = "gold"
    SILVER = "silver"
    TIN = "tin"


class MetalProperties(BaseModel):
    name: str
    conductivity_s_per_m: float = Field(gt=0)
    thickness_m: float = Field(default=35e-6, gt=0, description="Metal thickness (default ~1 oz copper)")

    def display(self) -> str:
        ms = self.conductivity_s_per_m / 1e7
        return f"{self.name} (σ≈{ms:.1f}×10^7 S/m, t={self.thickness_m*1e6:.0f} µm)"


metal_defaults: dict[Metal, MetalProperties] = {
    Metal.COPPER: MetalProperties(name="Copper", conductivity_s_per_m=5.8e7, thickness_m=35e-6),
    Metal.ALUMINUM: MetalProperties(name="Aluminum", conductivity_s_per_m=3.5e7, thickness_m=35e-6),
    Metal.GOLD: MetalProperties(name="Gold", conductivity_s_per_m=4.1e7, thickness_m=2e-6),
    Metal.SILVER: MetalProperties(name="Silver", conductivity_s_per_m=6.3e7, thickness_m=10e-6),
    Metal.TIN: MetalProperties(name="Tin", conductivity_s_per_m=9.1e6, thickness_m=5e-6),
}


class PatchAntennaParams(BaseModel):
    """Primary inputs for a rectangular microstrip (patch) antenna.

    Internally we use SI units. Convenience classmethod `from_user_units` accepts mm/GHz inputs.
    - frequency_hz: Operating frequency
    - eps_r: Relative permittivity of the substrate
    - h_m: Substrate thickness
    - patch_length_m/patch_width_m: Optional; if omitted they will be designed for resonance
    - loss_tangent: Dielectric loss tangent (tanδ)
    - metal: Conductor properties
    """

    frequency_hz: float = Field(gt=0)
    eps_r: float = Field(gt=1)
    h_m: float = Field(gt=0)
    loss_tangent: float = Field(default=0.0, ge=0)
    metal: MetalProperties = Field(default_factory=lambda: metal_defaults[Metal.COPPER])

    patch_length_m: Optional[float] = Field(default=None, gt=0)
    patch_width_m: Optional[float] = Field(default=None, gt=0)

    @classmethod
    def from_user_units(
        cls,
        *,
        frequency_ghz: float,
        er: float,
        h_mm: float,
        L_mm: Optional[float] = None,
        W_mm: Optional[float] = None,
        metal: str = "copper",
        loss_tangent: float = 0.0,
        metal_thickness_um: Optional[float] = None,
    ) -> "PatchAntennaParams":
        frequency_hz = frequency_ghz * 1e9
        h_m = h_mm * 1e-3
        L_m = None if L_mm is None else L_mm * 1e-3
        W_m = None if W_mm is None else W_mm * 1e-3

        try:
            metal_enum = Metal(metal.lower())
        except Exception:
            metal_enum = Metal.COPPER
        metal_props = metal_defaults[metal_enum].model_copy(deep=True)
        if metal_thickness_um is not None:
            metal_props.thickness_m = max(1e-7, metal_thickness_um * 1e-6)

        return cls(
            frequency_hz=frequency_hz,
            eps_r=er,
            h_m=h_m,
            patch_length_m=L_m,
            patch_width_m=W_m,
            metal=metal_props,
            loss_tangent=loss_tangent,
        )

    @property
    def frequency_ghz(self) -> float:
        return self.frequency_hz / 1e9

    @property
    def h_mm(self) -> float:
        return self.h_m * 1e3

    @property
    def L_mm(self) -> Optional[float]:
        return None if self.patch_length_m is None else self.patch_length_m * 1e3

    @property
    def W_mm(self) -> Optional[float]:
        return None if self.patch_width_m is None else self.patch_width_m * 1e3
