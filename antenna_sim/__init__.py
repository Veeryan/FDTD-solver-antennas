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

# Import microstrip solver
from .solver_fdtd_openems_microstrip import (
    probe_openems_microstrip,
    prepare_openems_microstrip_patch,
    run_prepared_openems_microstrip,
    FeedDirection,
    calculate_microstrip_width,
)
# Multi-antenna microstrip 3D solver
from .solver_fdtd_openems_microstrip_multi_3d import (
    prepare_openems_microstrip_multi_3d,
    run_prepared_openems_microstrip_multi_3d,
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
    # New plotting functions
    "draw_microstrip_patch_3d_geometry",
    # Fixed solvers (3D only - provides E-plane/H-plane cuts)
    "probe_openems_fixed",
    "prepare_openems_patch_fixed", 
    "run_prepared_openems_fixed",
    "OpenEMSProbe",
    "OpenEMSResult",
    "OpenEMSPrepared",
    # Microstrip solvers
    "probe_openems_microstrip",
    "prepare_openems_microstrip_patch",
    "run_prepared_openems_microstrip",
    "FeedDirection",
    "calculate_microstrip_width",
    # Multi-antenna microstrip 3D solver
    "prepare_openems_microstrip_multi_3d",
    "run_prepared_openems_microstrip_multi_3d",
]
