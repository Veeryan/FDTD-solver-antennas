from .models import PatchAntennaParams, Metal, MetalProperties, metal_defaults
from .solver_approx import AnalyticalPatchSolver
from .plotting import plot_cross_sections, plot_3d_pattern, draw_patch_3d_geometry

# Import fixed solvers
from .solver_fdtd_openems_fixed import (
    probe_openems_fixed, 
    prepare_openems_patch_fixed, 
    run_prepared_openems_fixed,
    OpenEMSProbe,
    OpenEMSResult,
    OpenEMSPrepared,
)
# 2D solver removed - 3D solver provides E-plane/H-plane cuts automatically

__all__ = [
    "PatchAntennaParams",
    "Metal",
    "MetalProperties",
    "metal_defaults",
    "AnalyticalPatchSolver",
    "plot_cross_sections",
    "plot_3d_pattern",
    "draw_patch_3d_geometry",
    # Fixed solvers (3D only - provides E-plane/H-plane cuts)
    "probe_openems_fixed",
    "prepare_openems_patch_fixed", 
    "run_prepared_openems_fixed",
    "OpenEMSProbe",
    "OpenEMSResult",
    "OpenEMSPrepared",
]
