import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .solver_approx import AnalyticalPatchSolver
from .solver_fdtd_openems_microstrip import FeedDirection, calculate_microstrip_width


def draw_patch_3d_geometry(L_m: float, W_m: float, h_m: float, fig_size=(8, 6)):
    """Create a detailed 3D visualization of the patch antenna geometry."""
    mm = 1e3
    L = L_m * mm
    W = W_m * mm
    h = h_m * mm
    
    # Substrate dimensions (larger than patch)
    margin = max(5.0, 0.2 * max(L, W))
    sub_L = L + 2 * margin
    sub_W = W + 2 * margin
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (sides + bottom only to avoid occluding patch)
    substrate_side_bottom = [
        # bottom
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]],
        # left side
        [[-sub_L/2, -sub_W/2, -h], [-sub_L/2, -sub_W/2, 0], [-sub_L/2, sub_W/2, 0], [-sub_L/2, sub_W/2, -h]],
        # right side
        [[sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, 0], [sub_L/2, sub_W/2, 0], [sub_L/2, sub_W/2, -h]],
        # front side
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, 0], [-sub_L/2, -sub_W/2, 0]],
        # back side
        [[-sub_L/2, sub_W/2, -h], [sub_L/2, sub_W/2, -h], [sub_L/2, sub_W/2, 0], [-sub_L/2, sub_W/2, 0]],
    ]
    
    # FR-4 style green substrate (transparent)
    substrate = Poly3DCollection(substrate_side_bottom, alpha=0.65, facecolor='#2e7d32', edgecolor='#1b5e20', linewidth=1.2)
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate)
    ground_verts = [[[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.9, facecolor='#9ea7ad', edgecolor='#6b7074')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    # Visually exaggerate copper thickness to make height obvious
    patch_thickness = max(0.2, 0.12 * h)  # visual thickness (mm)
    
    patch_verts = [
        # bottom face
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, L/2, 0], [-W/2, L/2, 0]],
        # top face
        [[-W/2, -L/2, patch_thickness], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],
        # side faces for 3D effect
        [[-W/2, -L/2, 0], [-W/2, -L/2, patch_thickness], [-W/2, L/2, patch_thickness], [-W/2, L/2, 0]],
        [[W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [W/2, L/2, 0]],
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [-W/2, -L/2, patch_thickness]],
        [[-W/2, L/2, 0], [W/2, L/2, 0], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],
    ]
    
    patch = Poly3DCollection(patch_verts, alpha=0.95, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=1.5)
    ax.add_collection3d(patch)
    
    # Set equal aspect ratio and limits
    max_range = max(sub_L, sub_W, h*10) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*2, h*8])
    
    # Labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Patch Antenna Geometry\n{L:.1f} × {W:.1f} mm, h={h:.2f} mm')
    
    # Add dimension labels with better positioning and visibility
    label_box = dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8)
    
    # Move length and width labels further from edges and make them larger
    ax.text(0, -L/2-0.9*margin, patch_thickness*1.2, f'L = {L:.1f} mm', ha='center', fontsize=13, color='white', bbox=label_box)
    ax.text(W/2+0.9*margin, 0, patch_thickness*1.2, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=13, color='white', bbox=label_box)
    
    # Add vertical ruler for thickness h
    xh, yh = -sub_L*0.4, sub_W*0.4
    ax.plot([xh, xh], [yh, yh], [-h, 0], color='orange', linewidth=3)
    ax.text(xh, yh, -h*0.5, f'h = {h:.2f} mm', color='white', fontsize=13, bbox=label_box, ha='center')
    
    return fig


def draw_microstrip_patch_3d_geometry(L_m: float, W_m: float, h_m: float, 
                                     feed_direction: FeedDirection,
                                     frequency_hz: float, eps_r: float,
                                     feed_line_length_mm: float = 20.0,
                                     fig_size=(10, 8)):
    """
    Create a 3D visualization of microstrip-fed patch antenna.
    Uses the EXACT same geometry as the FDTD solver.
    """
    mm = 1e3
    L = L_m * mm  # patch length (Y direction)
    W = W_m * mm  # patch width (X direction) 
    h = h_m * mm  # substrate thickness
    
    # Calculate 50Ω microstrip width
    feed_width_m = calculate_microstrip_width(frequency_hz, eps_r, h_m)
    feed_width = feed_width_m * mm
    
    # Substrate dimensions - EXACTLY match FDTD solver
    substrate_margin = 30.0  # mm
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        substrate_W = W + 2 * substrate_margin + feed_line_length_mm
        substrate_L = L + 2 * substrate_margin
    else:
        substrate_W = W + 2 * substrate_margin
        substrate_L = L + 2 * substrate_margin + feed_line_length_mm
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (sides + bottom only)
    substrate_faces = [
        # bottom
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], 
         [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]],
        # sides
        [[-substrate_W/2, -substrate_L/2, -h], [-substrate_W/2, -substrate_L/2, 0], 
         [-substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, -h]],
        [[substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], 
         [substrate_W/2, substrate_L/2, 0], [substrate_W/2, substrate_L/2, -h]],
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], 
         [substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, -substrate_L/2, 0]],
        [[-substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], 
         [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]],
    ]
    
    substrate = Poly3DCollection(substrate_faces, alpha=0.6, facecolor='#2e7d32', edgecolor='#1b5e20', linewidth=1)
    ax.add_collection3d(substrate)
    
    # Ground plane
    ground_verts = [[[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], 
                    [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.9, facecolor='#9ea7ad', edgecolor='#6b7074')
    ax.add_collection3d(ground)
    
    # Patch antenna - same as simple version
    patch_thickness = max(0.2, 0.12 * h)
    patch_verts = [
        [[-W/2, -L/2, 0.02], [W/2, -L/2, 0.02], [W/2, L/2, 0.02], [-W/2, L/2, 0.02]],
        [[-W/2, -L/2, 0.02 + patch_thickness], [W/2, -L/2, 0.02 + patch_thickness], 
         [W/2, L/2, 0.02 + patch_thickness], [-W/2, L/2, 0.02 + patch_thickness]],
        # side faces
        [[-W/2, -L/2, 0.02], [-W/2, -L/2, 0.02 + patch_thickness], 
         [-W/2, L/2, 0.02 + patch_thickness], [-W/2, L/2, 0.02]],
        [[W/2, -L/2, 0.02], [W/2, -L/2, 0.02 + patch_thickness], 
         [W/2, L/2, 0.02 + patch_thickness], [W/2, L/2, 0.02]],
        [[-W/2, -L/2, 0.02], [W/2, -L/2, 0.02], 
         [W/2, -L/2, 0.02 + patch_thickness], [-W/2, -L/2, 0.02 + patch_thickness]],
        [[-W/2, L/2, 0.02], [W/2, L/2, 0.02], 
         [W/2, L/2, 0.02 + patch_thickness], [-W/2, L/2, 0.02 + patch_thickness]],
    ]
    patch = Poly3DCollection(patch_verts, alpha=0.95, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=1.5)
    ax.add_collection3d(patch)
    
    # Microstrip feed line - EXACTLY match FDTD coordinates
    if feed_direction == FeedDirection.NEG_X:
        feed_start_x = -substrate_W/2
        feed_stop_x = -W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.POS_X:
        feed_start_x = W/2
        feed_stop_x = substrate_W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.NEG_Y:
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = -substrate_L/2
        feed_stop_y = -L/2
    else:  # POS_Y
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = L/2
        feed_stop_y = substrate_L/2
    
    # Draw microstrip trace
    trace_z = 0.04  # slightly above patch
    trace_verts = [
        [[feed_start_x, feed_start_y, trace_z], [feed_stop_x, feed_start_y, trace_z], 
         [feed_stop_x, feed_stop_y, trace_z], [feed_start_x, feed_stop_y, trace_z]],
        [[feed_start_x, feed_start_y, trace_z + patch_thickness], [feed_stop_x, feed_start_y, trace_z + patch_thickness], 
         [feed_stop_x, feed_stop_y, trace_z + patch_thickness], [feed_start_x, feed_stop_y, trace_z + patch_thickness]],
    ]
    trace = Poly3DCollection(trace_verts, alpha=0.95, facecolor='#ff6347', edgecolor='#cc3300', linewidth=2)
    ax.add_collection3d(trace)
    
    # Styling
    max_range = max(substrate_W, substrate_L) * 0.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*1.5, patch_thickness*6])
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Microstrip Patch Antenna\n{frequency_hz/1e9:.2f} GHz, Feed: {feed_direction.value}')
    
    # Labels
    label_box = dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8)
    ax.text(0, -L/2-max_range*0.15, patch_thickness*2, f'L = {L:.1f} mm', 
            ha='center', fontsize=12, color='white', bbox=label_box)
    ax.text(W/2+max_range*0.15, 0, patch_thickness*2, f'W = {W:.1f} mm', 
            ha='center', rotation=90, fontsize=12, color='white', bbox=label_box)
    ax.text(-max_range*0.3, max_range*0.3, -h*0.5, f'h = {h:.2f} mm', 
            ha='center', fontsize=12, color='white', bbox=label_box)
    
    # Feed width label
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        feed_center_x = (feed_start_x + feed_stop_x) / 2
        ax.text(feed_center_x, max_range*0.25, patch_thickness*3, f'Feed: {feed_width:.2f} mm\n(50Ω)', 
                ha='center', fontsize=10, color='red', bbox=label_box)
    else:
        feed_center_y = (feed_start_y + feed_stop_y) / 2
        ax.text(max_range*0.25, feed_center_y, patch_thickness*3, f'Feed: {feed_width:.2f} mm\n(50Ω)', 
                ha='center', fontsize=10, color='red', bbox=label_box)
    
    return fig


# Keep all the other plotting functions from the original file unchanged
def draw_patch_top_view(ax: plt.Axes, L_m: float, W_m: float, h_m: float):
    """Render a simple top-view geometry: patch rectangle on substrate outline."""
    mm = 1e3
    L = L_m * mm
    W = W_m * mm

    # Substrate outline slightly larger around the patch
    margin = max(2.0, 0.1 * max(L, W))
    sub_L = L + 2 * margin
    sub_W = W + 2 * margin

    # Substrate (light)
    sub = plt.Rectangle((-sub_L / 2, -sub_W / 2), sub_L, sub_W, color="#d0e6f6", alpha=0.4)
    ax.add_patch(sub)
    # Patch (metal)
    patch = plt.Rectangle((-L / 2, -W / 2), L, W, color="#d69c2f", alpha=0.9)
    ax.add_patch(patch)

    # Feed point
    ax.plot(-L/2, 0, 'ro', markersize=8, label='Feed point')
    
    # Dimension lines
    ax.annotate('', xy=(L/2, -W/2-margin/4), xytext=(-L/2, -W/2-margin/4), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(0, -W/2-margin/3, f'L = {L:.1f} mm', ha='center', fontsize=10)
    
    ax.annotate('', xy=(W/2+margin/4, W/2), xytext=(W/2+margin/4, -W/2), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(W/2+margin/3, 0, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=10)

    ax.set_xlim(-sub_L / 2 - margin/2, sub_L / 2 + margin/2)
    ax.set_ylim(-sub_W / 2 - margin/2, sub_W / 2 + margin/2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Patch Geometry (Top View)")
    ax.grid(True, alpha=0.3)
    ax.legend()


# Import the rest of the functions from the original plotting.py
def plot_cross_sections(solver: AnalyticalPatchSolver, *, fig_size=(12, 6)):
    th_e, G_e = solver.cross_section_gain_lin("E")
    th_h, G_h = solver.cross_section_gain_lin("H")

    G_e_dBi = solver.lin_to_dbi(G_e)
    G_h_dBi = solver.lin_to_dbi(G_h)

    peak = max(np.max(G_e_dBi), np.max(G_h_dBi))
    
    # Round peak to nearest 5 dB increment
    peak_rounded = np.ceil(peak / 5) * 5
    rmin = peak_rounded - 40.0
    
    # Create nice 5 dB increments
    rticks = np.arange(rmin, peak_rounded + 5, 5)

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=fig_size)
    
    titles = [
        "E-plane (Length direction)\nφ=0°, varies θ", 
        "H-plane (Width direction)\nφ=90°, varies θ"
    ]
    
    for ax, th, G_dBi, title in zip(axes, [th_e, th_h], [G_e_dBi, G_h_dBi], titles):
        ax.plot(th, G_dBi, color="#1b9e77", lw=2.5)
        ax.set_thetalim(0, np.pi)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_rmax(peak_rounded)
        ax.set_rmin(rmin)
        ax.set_rticks(rticks)
        ax.set_rlabel_position(180)
        ax.grid(True)

    plt.tight_layout()
    return fig


def plot_3d_pattern(solver: AnalyticalPatchSolver, *, fig_size=(8, 9)):
    """Render a 3D far-field surface from analytical solution."""
    th_deg = np.linspace(0, 180, 91)
    ph_deg = np.linspace(0, 360, 181)
    
    th_rad = np.deg2rad(th_deg)
    ph_rad = np.deg2rad(ph_deg)
    
    THG, PHG = np.meshgrid(th_rad, ph_rad, indexing="ij")
    
    # Analytical gain (linear)
    G_lin = solver.gain_3d_pattern(THG, PHG)
    G_dBi = solver.lin_to_dbi(G_lin)
    
    return plot_3d_pattern_from_grid(th_rad, ph_rad, G_dBi, fig_size=fig_size)


def plot_3d_pattern_from_grid(theta: np.ndarray, phi: np.ndarray, intensity_dBi: np.ndarray, 
                             *, L_m: float | None = None, W_m: float | None = None, h_m: float | None = None,
                             dB_min: float | None = None, dB_max: float = 0.0, fig_size=(8, 9),
                             colors_db: np.ndarray | None = None, clip_db: float | None = None):
    """Render a professional 3D far-field surface from (theta, phi, intensity) grids."""
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    th = np.asarray(theta).reshape(-1)
    ph = np.asarray(phi).reshape(-1)
    G_dBi = np.asarray(intensity_dBi)
    
    if G_dBi.ndim == 1:
        if G_dBi.size == th.size:
            G_dBi = np.tile(G_dBi.reshape(th.size, 1), (1, ph.size))
        elif G_dBi.size == ph.size:
            G_dBi = np.tile(G_dBi.reshape(1, ph.size), (th.size, 1))
    
    thg, phg = np.meshgrid(th, ph, indexing="ij")

    # Clip very low values if requested
    if clip_db is not None:
        G_dBi = np.maximum(G_dBi, clip_db)

    # Auto-set color range if not provided
    if dB_min is None:
        dB_min = max(np.percentile(G_dBi, 10), dB_max - 25)

    # Convert to Cartesian for 3D plotting (normalized radius)
    G_lin = 10**(G_dBi / 10)
    G_norm = G_lin / np.nanmax(G_lin)
    
    x = G_norm * np.sin(thg) * np.cos(phg)
    y = G_norm * np.sin(thg) * np.sin(phg)
    z = G_norm * np.cos(thg)

    # Create figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with turbo colormap
    surf = ax.plot_surface(x, y, z, facecolors=cm.turbo((G_dBi - dB_min) / (dB_max - dB_min)),
                          alpha=0.9, linewidth=0, antialiased=True, shade=True)

    # Add colorbar
    mappable = cm.ScalarMappable(cmap='turbo', norm=colors.Normalize(vmin=dB_min, vmax=dB_max))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Gain (dBi)', fontsize=12, fontweight='bold')

    # Styling
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xlabel('X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z', fontsize=10, fontweight='bold')

    # Enhanced title
    if L_m and W_m and h_m:
        title = f'3D Radiation Pattern (dBi)\nPatch: {L_m*1000:.1f}×{W_m*1000:.1f} mm, h={h_m*1000:.2f} mm'
    else:
        title = '3D Radiation Pattern (dBi)'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    # Professional viewing angle
    ax.view_init(elev=15, azim=45)

    plt.tight_layout()
    return fig
