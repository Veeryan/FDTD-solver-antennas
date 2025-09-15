#!/usr/bin/env python3
import os, tempfile
import numpy as np

# Ensure DLLs are found (Windows)
os.add_dll_directory(r'C:\Users\veery\Documents\Coding Projects\Antenna EM Sims\openEMS')
os.environ['OPENEMS_INSTALL_PATH'] = r'C:\Users\veery\Documents\Coding Projects\Antenna EM Sims\openEMS'

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

print('=== Simple Patch Antenna (updated tutorial-aligned) ===')

# Simulation path (temp dir)
Sim_Path = os.path.join(tempfile.gettempdir(), 'Simp_Patch_Python')

# Geometry and simulation parameters (tutorial-style)
patch_width  = 32.0   # mm (x)
patch_length = 40.0   # mm (y)

substrate_epsR   = 3.38
substrate_thickness = 1.524  # mm
substrate_width  = 60.0  # mm
substrate_length = 60.0  # mm
substrate_cells = 4

feed_pos = -6.0   # mm in x-direction
feed_R   = 50.0   # Ohm

SimBox = np.array([200.0, 200.0, 150.0])  # mm

f0 = 2e9  # center frequency
fc = 1e9  # 20 dB corner frequency

try:
    # FDTD setup (stricter convergence, PML-8)
    FDTD = openEMS(NrTS=60000, EndCriteria=1e-5)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond([3,3,3,3,3,3])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # mm as drawing unit

    # Mesh resolution (λ/20)
    mesh_res = C0/(f0+fc)/1e-3/20.0

    # Initialize mesh with air-box dims
    mesh.AddLine('x', [-SimBox[0]/2.0, SimBox[0]/2.0])
    mesh.AddLine('y', [-SimBox[1]/2.0, SimBox[1]/2.0])
    mesh.AddLine('z', [-SimBox[2]/3.0, SimBox[2]*2.0/3.0])

    # Patch (PEC)
    patch = CSX.AddMetal('patch')
    start = [-patch_width/2.0, -patch_length/2.0, substrate_thickness]
    stop  = [ patch_width/2.0,  patch_length/2.0, substrate_thickness]
    patch.AddBox(priority=10, start=start, stop=stop)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2.0)

    # Substrate with loss (tanδ example ~0.001 scaled to EPS0)
    substrate_kappa  = 2.0*np.pi*f0*EPS0*substrate_epsR*1e-3
    try:
        substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
    except TypeError:
        substrate = CSX.AddMaterial('substrate')
        substrate.SetMaterialProperty('Eps', substrate_epsR)
        substrate.SetMaterialProperty('Kappa', substrate_kappa)

    start = [-substrate_width/2.0, -substrate_length/2.0, 0.0]
    stop  = [ substrate_width/2.0,  substrate_length/2.0, substrate_thickness]
    substrate.AddBox(priority=0, start=start, stop=stop)

    # Extra z-cells across the substrate thickness
    mesh.AddLine('z', np.linspace(0.0, substrate_thickness, substrate_cells+1).tolist())

    # Ground (PEC)
    gnd = CSX.AddMetal('gnd')
    start[2] = 0.0
    stop[2]  = 0.0
    gnd.AddBox(start, stop, priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

    # Ensure port lies on mesh lines
    mesh.AddLine('x', [feed_pos])
    mesh.AddLine('z', [0.0, substrate_thickness])

    # Lumped port (z-directed coax-like)
    start = [feed_pos, 0.0, 0.0]
    stop  = [feed_pos, 0.0, substrate_thickness]
    FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5, edges2grid='xy')

    # Smooth mesh lines (important)
    mesh.SmoothMeshLines('all', mesh_res, 1.4)

    # NF2FF box (default around origin)
    nf2ff = FDTD.CreateNF2FFBox()

    print(f'Running simulation in: {Sim_Path}')
    FDTD.Run(Sim_Path, verbose=1, cleanup=True)
    print('SUCCESS: Tutorial-aligned patch simulation ran!')

except Exception as e:
    import traceback
    print('FAILED:')
    traceback.print_exc()
    print(f'Error: {type(e).__name__}: {e}')
