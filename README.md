# Patch Antenna Simulator (Python)

A lightweight, extensible tool to design and simulate rectangular microstrip (patch) antennas. It includes:

- A fast analytical/cavity-model solver for patterns and key metrics
- Optional FDTD backend (Meep) for higher-fidelity simulations (when available)
- 2D E/H-plane plots and a 3D radiation pattern with an isotropic reference sphere
- A Streamlit web UI and a CLI entrypoint

## Quick start

1) Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

2) Launch the GUI:

```bash
streamlit run streamlit_app.py
```

3) Or run from CLI (example at 2.45 GHz on FR-4, h=1.6 mm):

```bash
python -m antenna_sim simulate --frequency-ghz 2.45 --er 4.3 --h-mm 1.6 --metal copper
```

The GUI offers the analytical/cavity-model solver (default) and, if the `meep` Python package is available, an FDTD simulation option.

## Optional: Meep FDTD backend

Meep provides a powerful FDTD engine with near-to-far field transforms. On Windows, the simplest way is via Conda/WSL. If Meep is not installed, the app will gracefully fall back to the analytical solver.

- Conda (recommended):

```bash
conda create -n meep_env -c conda-forge python=3.11 meep
conda activate meep_env
pip install -r requirements.txt
```

If `meep` is installed, the Streamlit UI will show the FDTD option.

## Project layout

```
antenna_sim/
  __init__.py
  __main__.py
  models.py
  physics.py
  solver_approx.py
  solver_fdtd_meep.py   # optional backend, imports guarded
  plotting.py
streamlit_app.py
requirements.txt
README.md
```

## Notes

- Dimensions are SI units internally (meters, Hertz). The UI accepts user-friendly units (mm/GHz) and converts.
- Gain is computed as `G = η · D`, where `D` is directivity from the analytical model and `η` is an efficiency estimate based on dielectric loss tangent and metal conductivity. For rigorous efficiency modeling, prefer the FDTD backend.
- The analytical model targets the dominant TM10 mode of a rectangular patch.
