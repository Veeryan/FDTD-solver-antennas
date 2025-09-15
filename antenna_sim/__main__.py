from __future__ import annotations

import argparse
from pathlib import Path

from .models import PatchAntennaParams, Metal
from .solver_approx import AnalyticalPatchSolver
from .plotting import plot_cross_sections, plot_3d_pattern


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch antenna simulator (analytical)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("simulate", help="Run an analytical simulation and save plots")
    sim.add_argument("--frequency-ghz", type=float, required=True)
    sim.add_argument("--er", type=float, required=True)
    sim.add_argument("--h-mm", type=float, required=True)
    sim.add_argument("--L-mm", type=float, default=None)
    sim.add_argument("--W-mm", type=float, default=None)
    sim.add_argument("--metal", type=str, default="copper")
    sim.add_argument("--loss-tangent", type=float, default=0.0)
    sim.add_argument("--outdir", type=str, default="outputs")

    args = parser.parse_args()

    params = PatchAntennaParams.from_user_units(
        frequency_ghz=args.frequency_ghz,
        er=args.er,
        h_mm=args.h_mm,
        L_mm=args.L_mm,
        W_mm=args.W_mm,
        metal=args.metal,
        loss_tangent=args.loss_tangent,
    )

    solver = AnalyticalPatchSolver(params)
    summary = solver.summary()
    print("Design:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Make plots
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_cuts = plot_cross_sections(solver)
    fig_3d = plot_3d_pattern(solver)

    cuts_path = outdir / "cuts.png"
    fig_cuts.savefig(cuts_path, dpi=160, bbox_inches="tight")
    fig_3d_path = outdir / "pattern_3d.png"
    fig_3d.savefig(fig_3d_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {cuts_path}")
    print(f"Saved: {fig_3d_path}")


if __name__ == "__main__":
    main()
