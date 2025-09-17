import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .solver_approx import AnalyticalPatchSolver
from .solver_fdtd_openems_microstrip import FeedDirection, calculate_microstrip_width


def draw_patch_3d_geometry(L_m: float, W_m: float, h_m: float, fig_size=(8, 6), show_labels: bool = True):
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
    # Keep a tiny visual gap under the patch plane to reduce alpha overlays at oblique angles
    side_top_z = -0.02
    substrate_side_bottom = [
        # bottom
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]],
        # left side
        [[-sub_L/2, -sub_W/2, -h], [-sub_L/2, -sub_W/2, side_top_z], [-sub_L/2, sub_W/2, side_top_z], [-sub_L/2, sub_W/2, -h]],
        # right side
        [[sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, side_top_z], [sub_L/2, sub_W/2, side_top_z], [sub_L/2, sub_W/2, -h]],
        # front side
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, side_top_z], [-sub_L/2, -sub_W/2, side_top_z]],
        # back side
        [[-sub_L/2, sub_W/2, -h], [sub_L/2, sub_W/2, -h], [sub_L/2, sub_W/2, side_top_z], [-sub_L/2, sub_W/2, side_top_z]],
    ]
    
    # FR-4 style green substrate (transparent) - thinner visual block, original color
    # Slightly lower alpha to prevent dominance in top views
    substrate = Poly3DCollection(substrate_side_bottom, alpha=0.45, facecolor='#2e7d32', edgecolor='#1b5e20', linewidth=1.0)
    try:
        substrate.set_zsort('min')
    except Exception:
        pass
    try:
        substrate.set_zsort('min')  # draw behind metals
    except Exception:
        pass
    substrate.set_zorder(1)
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate) - original solid style
    ground_verts = [[[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.9, facecolor='#9ea7ad', edgecolor='#6b7074')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    # Copper thickness (visual) taken as a small fraction of substrate height
    # If you want to drive this from a user metal thickness, thread it through here
    patch_thickness = max(0.08, 0.06 * h)  # mm
    patch_verts = [
        [[-L/2, -W/2, 0], [L/2, -W/2, 0], [L/2, W/2, 0], [-L/2, W/2, 0]],  # bottom
        [[-L/2, -W/2, patch_thickness], [L/2, -W/2, patch_thickness], [L/2, W/2, patch_thickness], [-L/2, W/2, patch_thickness]],  # top
        [[-L/2, -W/2, 0], [-L/2, -W/2, patch_thickness], [-L/2, W/2, patch_thickness], [-L/2, W/2, 0]], # left
        [[L/2, -W/2, 0], [L/2, -W/2, patch_thickness], [L/2, W/2, patch_thickness], [L/2, W/2, 0]],     # right
        [[-L/2, -W/2, 0], [L/2, -W/2, 0], [L/2, -W/2, patch_thickness], [-L/2, -W/2, patch_thickness]], # front
        [[-L/2, W/2, 0], [L/2, W/2, 0], [L/2, W/2, patch_thickness], [-L/2, W/2, patch_thickness]]      # back
    ]
    
    # Slightly higher alpha so patch dominates top-down
    patch = Poly3DCollection(patch_verts, alpha=0.98, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=1.2)
    try:
        patch.set_zsort('max')
    except Exception:
        pass
    patch.set_zorder(5)
    ax.add_collection3d(patch)

    # Remove separate cap (revert to simpler original rendering)
    # Ensure patch appears visually above substrate sides
    try:
        for coll in ax.collections:
            if coll is substrate:
                coll.set_zorder(1)
            if coll is ground:
                coll.set_zorder(2)
            if coll is patch:
                coll.set_zorder(5)
    except Exception:
        pass
    
    # Feed point (small cylinder at edge)
    feed_x = -L/2
    feed_y = 0
    feed_z = np.linspace(-h, patch_thickness, 20)
    feed_r = 0.5  # mm radius
    theta = np.linspace(0, 2*np.pi, 20)
    
    for i in range(len(feed_z)-1):
        x_circle = feed_x + feed_r * np.cos(theta)
        y_circle = feed_y + feed_r * np.sin(theta)
        z_circle = np.full_like(theta, feed_z[i])
        ax.plot(x_circle, y_circle, z_circle, color='red', alpha=0.7, linewidth=1)
    
    # Dimension annotations (high contrast)
    label_box = dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.6)
    if show_labels:
        ax.text(0, -W/2-0.9*margin, patch_thickness*1.2, f'L = {L:.1f} mm', ha='center', fontsize=13, color='white', bbox=label_box)
        ax.text(L/2+0.9*margin, 0, patch_thickness*1.2, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=13, color='white', bbox=label_box)
    # Thickness indicator near front-left corner
    xh, yh = -sub_L/2 + 0.15*sub_L, -sub_W/2 + 0.15*sub_W
    ax.plot([xh, xh], [yh, yh], [-h, 0], color='#ff7043', linewidth=2.0)
    ax.text(xh, yh, -h*0.5, f'h = {h:.2f} mm', color='white', fontsize=13, bbox=label_box, ha='center')
    
    # Note: in-scene coordinate axes removed; a corner triad overlay is added in the GUI
    
    # Set limits and labels
    max_range = max(sub_L, sub_W, h*3) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*1.2, h*0.5])
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D Patch Antenna Geometry\n(Gold patch on blue substrate with ground plane)')
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    return fig


def draw_microstrip_patch_3d_geometry(L_m: float, W_m: float, h_m: float, 
                                       feed_direction: FeedDirection,
                                       frequency_hz: float, eps_r: float,
                                       feed_line_length_mm: float = 20.0,
                                       fig_size=(8, 6)):
    """Create a detailed 3D visualization of the microstrip-fed patch antenna geometry."""
    mm = 1e3
    L = L_m * mm  # patch length (Y direction)
    W = W_m * mm  # patch width (X direction)
    h = h_m * mm  # substrate thickness
    
    # Calculate microstrip feed line width for 50Ω
    feed_width = calculate_microstrip_width(frequency_hz, eps_r, h_m) * mm
    
    # Substrate dimensions (larger than patch + feed)
    substrate_margin = 30.0  # mm
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        substrate_W = W + 2 * substrate_margin + feed_line_length_mm
        substrate_L = L + 2 * substrate_margin
    else:
        substrate_W = W + 2 * substrate_margin
        substrate_L = L + 2 * substrate_margin + feed_line_length_mm
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (bottom layer)
    substrate_verts = [
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]],  # bottom
        [[-substrate_W/2, -substrate_L/2, 0], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]],      # top
        [[-substrate_W/2, -substrate_L/2, -h], [-substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, -h]], # left
        [[substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [substrate_W/2, substrate_L/2, -h]],     # right
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, -substrate_L/2, 0]], # front
        [[-substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]]      # back
    ]
    
    substrate = Poly3DCollection(substrate_verts, alpha=0.3, facecolor='lightblue', edgecolor='gray')
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate)
    ground_verts = [[[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.8, facecolor='silver', edgecolor='black')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    patch_thickness = 0.035  # 35 microns in mm
    patch_verts = [
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, L/2, 0], [-W/2, L/2, 0]],  # bottom
        [[-W/2, -L/2, patch_thickness], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],  # top
        [[-W/2, -L/2, 0], [-W/2, -L/2, patch_thickness], [-W/2, L/2, patch_thickness], [-W/2, L/2, 0]], # left
        [[W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [W/2, L/2, 0]],     # right
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [-W/2, -L/2, patch_thickness]], # front
        [[-W/2, L/2, 0], [W/2, L/2, 0], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]]      # back
    ]
    
    patch = Poly3DCollection(patch_verts, alpha=0.9, facecolor='gold', edgecolor='black')
    ax.add_collection3d(patch)
    
    # Microstrip feed line (based on feed direction)
    if feed_direction == FeedDirection.NEG_X:
        # Feed from negative X direction
        feed_start_x = -substrate_W/2
        feed_stop_x = -W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.POS_X:
        # Feed from positive X direction
        feed_start_x = W/2
        feed_stop_x = substrate_W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.NEG_Y:
        # Feed from negative Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = -substrate_L/2
        feed_stop_y = -L/2
    else:  # FeedDirection.POS_Y
        # Feed from positive Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = L/2
        feed_stop_y = substrate_L/2
    
    # Create microstrip feed line
    feed_verts = [
        [[feed_start_x, feed_start_y, 0], [feed_stop_x, feed_start_y, 0], [feed_stop_x, feed_stop_y, 0], [feed_start_x, feed_stop_y, 0]],  # bottom
        [[feed_start_x, feed_start_y, patch_thickness], [feed_stop_x, feed_start_y, patch_thickness], [feed_stop_x, feed_stop_y, patch_thickness], [feed_start_x, feed_stop_y, patch_thickness]],  # top
    ]
    
    feed_line = Poly3DCollection(feed_verts, alpha=0.9, facecolor='orange', edgecolor='black')
    ax.add_collection3d(feed_line)
    
    # Set equal aspect ratio and limits
    max_range = max(substrate_W, substrate_L, h*10) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*2, h*8])
    
    # Labels and styling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Microstrip-Fed Patch Antenna\nFeed: {feed_direction.value}, Line Width: {feed_width:.2f} mm')
    
    # Add dimension annotations
    ax.text(0, 0, h*4, f'{W:.1f} mm', ha='center', va='center', color='red', fontweight='bold')
    ax.text(0, 0, h*6, f'{L:.1f} mm', ha='center', va='center', color='green', fontweight='bold')
    
    # Add feed line annotation
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        feed_center_x = (feed_start_x + feed_stop_x) / 2
        ax.text(feed_center_x, 0, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    else:
        feed_center_y = (feed_start_y + feed_stop_y) / 2
        ax.text(0, feed_center_y, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    
    return fig


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
    
    ax.annotate('', xy=(L/2+margin/4, W/2), xytext=(L/2+margin/4, -W/2), 
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(L/2+margin/3, 0, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=10)

    ax.set_xlim(-sub_L / 2 - margin/2, sub_L / 2 + margin/2)
    ax.set_ylim(-sub_W / 2 - margin/2, sub_W / 2 + margin/2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Patch Geometry (Top View)")
    ax.grid(True, alpha=0.3)
    ax.legend()


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
        ax.set_rlabel_position(112)
        ax.grid(True, alpha=0.6)
        
        # Add angle labels
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180], 
                         ['0°\n(Broadside)', '30°', '60°', '90°\n(Horizon)', '120°', '150°', '180°\n(Backside)'])

    fig.suptitle("Radiation Patterns (Gain in dBi)\nShows how antenna radiates power in different directions", fontsize=14)
    fig.tight_layout()
    return fig


def draw_microstrip_patch_3d_geometry(L_m: float, W_m: float, h_m: float, 
                                       feed_direction: FeedDirection,
                                       frequency_hz: float, eps_r: float,
                                       feed_line_length_mm: float = 20.0,
                                       fig_size=(8, 6)):
    """Create a detailed 3D visualization of the microstrip-fed patch antenna geometry."""
    mm = 1e3
    L = L_m * mm  # patch length (Y direction)
    W = W_m * mm  # patch width (X direction)
    h = h_m * mm  # substrate thickness
    
    # Calculate microstrip feed line width for 50Ω
    feed_width = calculate_microstrip_width(frequency_hz, eps_r, h_m) * mm
    
    # Substrate dimensions (larger than patch + feed)
    substrate_margin = 30.0  # mm
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        substrate_W = W + 2 * substrate_margin + feed_line_length_mm
        substrate_L = L + 2 * substrate_margin
    else:
        substrate_W = W + 2 * substrate_margin
        substrate_L = L + 2 * substrate_margin + feed_line_length_mm
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (bottom layer)
    substrate_verts = [
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]],  # bottom
        [[-substrate_W/2, -substrate_L/2, 0], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]],      # top
        [[-substrate_W/2, -substrate_L/2, -h], [-substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, -h]], # left
        [[substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [substrate_W/2, substrate_L/2, -h]],     # right
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, -substrate_L/2, 0]], # front
        [[-substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]]      # back
    ]
    
    substrate = Poly3DCollection(substrate_verts, alpha=0.3, facecolor='lightblue', edgecolor='gray')
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate)
    ground_verts = [[[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.8, facecolor='silver', edgecolor='black')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    patch_thickness = 0.035  # 35 microns in mm
    patch_verts = [
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, L/2, 0], [-W/2, L/2, 0]],  # bottom
        [[-W/2, -L/2, patch_thickness], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],  # top
        [[-W/2, -L/2, 0], [-W/2, -L/2, patch_thickness], [-W/2, L/2, patch_thickness], [-W/2, L/2, 0]], # left
        [[W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [W/2, L/2, 0]],     # right
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [-W/2, -L/2, patch_thickness]], # front
        [[-W/2, L/2, 0], [W/2, L/2, 0], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]]      # back
    ]
    
    patch = Poly3DCollection(patch_verts, alpha=0.9, facecolor='gold', edgecolor='black')
    ax.add_collection3d(patch)
    
    # Microstrip feed line (based on feed direction)
    if feed_direction == FeedDirection.NEG_X:
        # Feed from negative X direction
        feed_start_x = -substrate_W/2
        feed_stop_x = -W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.POS_X:
        # Feed from positive X direction
        feed_start_x = W/2
        feed_stop_x = substrate_W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.NEG_Y:
        # Feed from negative Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = -substrate_L/2
        feed_stop_y = -L/2
    else:  # FeedDirection.POS_Y
        # Feed from positive Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = L/2
        feed_stop_y = substrate_L/2
    
    # Create microstrip feed line
    feed_verts = [
        [[feed_start_x, feed_start_y, 0], [feed_stop_x, feed_start_y, 0], [feed_stop_x, feed_stop_y, 0], [feed_start_x, feed_stop_y, 0]],  # bottom
        [[feed_start_x, feed_start_y, patch_thickness], [feed_stop_x, feed_start_y, patch_thickness], [feed_stop_x, feed_stop_y, patch_thickness], [feed_start_x, feed_stop_y, patch_thickness]],  # top
    ]
    
    feed_line = Poly3DCollection(feed_verts, alpha=0.9, facecolor='orange', edgecolor='black')
    ax.add_collection3d(feed_line)
    
    # Set equal aspect ratio and limits
    max_range = max(substrate_W, substrate_L, h*10) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*2, h*8])
    
    # Labels and styling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Microstrip-Fed Patch Antenna\nFeed: {feed_direction.value}, Line Width: {feed_width:.2f} mm')
    
    # Add dimension annotations
    ax.text(0, 0, h*4, f'{W:.1f} mm', ha='center', va='center', color='red', fontweight='bold')
    ax.text(0, 0, h*6, f'{L:.1f} mm', ha='center', va='center', color='green', fontweight='bold')
    
    # Add feed line annotation
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        feed_center_x = (feed_start_x + feed_stop_x) / 2
        ax.text(feed_center_x, 0, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    else:
        feed_center_y = (feed_start_y + feed_stop_y) / 2
        ax.text(0, feed_center_y, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    
    return fig


def _spherical_to_cart(r: np.ndarray, th: np.ndarray, ph: np.ndarray):
    x = r * np.sin(th) * np.cos(ph)
    y = r * np.sin(th) * np.sin(ph)
    z = r * np.cos(th)
    return x, y, z


def plot_3d_pattern(solver: AnalyticalPatchSolver, *, show_isotropic: bool = True, fig_size=(8, 9)):
    res = solver.compute_full_pattern(num_theta=121, num_phi=241)
    G = res.gain
    G_norm = G / max(1e-16, np.max(G))

    th, ph = np.meshgrid(res.theta, res.phi, indexing="ij")
    x, y, z = _spherical_to_cart(G_norm, th, ph)

    # Color-map by intensity for better visual interpretation
    colors = plt.cm.viridis(G_norm / max(1e-16, np.max(G_norm)))

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        facecolors=colors,
        shade=True,
        linewidth=0.2,
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )

    if show_isotropic:
        rs = np.ones_like(G_norm) * 1.0
        xs, ys, zs = _spherical_to_cart(rs, th, ph)
        ax.plot_surface(xs, ys, zs, color="#999999", alpha=0.15, linewidth=0.0)

    lim = 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Coordinate axes thicker and labeled
    ax.plot([0, 1.0], [0, 0], [0, 0], color="#2a9d8f", lw=3)
    ax.plot([0, 0], [0, 1.0], [0, 0], color="#2a9d8f", lw=3)
    ax.plot([0, 0], [0, 0], [0, 1.0], color="#2a9d8f", lw=3)

    ax.text(1.05, 0, 0, "+x")
    ax.text(0, 1.05, 0, "+y")
    ax.text(0, 0, 1.05, "+z")

    ax.set_title("3D radiation surface (normalized)\nColor = intensity, transparent sphere = isotropic")
    ax.view_init(elev=22, azim=35)
    return fig


def draw_microstrip_patch_3d_geometry(L_m: float, W_m: float, h_m: float, 
                                       feed_direction: FeedDirection,
                                       frequency_hz: float, eps_r: float,
                                       feed_line_length_mm: float = 20.0,
                                       fig_size=(8, 6)):
    """Create a detailed 3D visualization of the microstrip-fed patch antenna geometry."""
    mm = 1e3
    L = L_m * mm  # patch length (Y direction)
    W = W_m * mm  # patch width (X direction)
    h = h_m * mm  # substrate thickness
    
    # Calculate microstrip feed line width for 50Ω
    feed_width = calculate_microstrip_width(frequency_hz, eps_r, h_m) * mm
    
    # Substrate dimensions (larger than patch + feed)
    substrate_margin = 30.0  # mm
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        substrate_W = W + 2 * substrate_margin + feed_line_length_mm
        substrate_L = L + 2 * substrate_margin
    else:
        substrate_W = W + 2 * substrate_margin
        substrate_L = L + 2 * substrate_margin + feed_line_length_mm
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (bottom layer)
    substrate_verts = [
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]],  # bottom
        [[-substrate_W/2, -substrate_L/2, 0], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]],      # top
        [[-substrate_W/2, -substrate_L/2, -h], [-substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, -h]], # left
        [[substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [substrate_W/2, substrate_L/2, -h]],     # right
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, -substrate_L/2, 0]], # front
        [[-substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]]      # back
    ]
    
    substrate = Poly3DCollection(substrate_verts, alpha=0.3, facecolor='lightblue', edgecolor='gray')
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate)
    ground_verts = [[[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.8, facecolor='silver', edgecolor='black')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    patch_thickness = 0.035  # 35 microns in mm
    patch_verts = [
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, L/2, 0], [-W/2, L/2, 0]],  # bottom
        [[-W/2, -L/2, patch_thickness], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],  # top
        [[-W/2, -L/2, 0], [-W/2, -L/2, patch_thickness], [-W/2, L/2, patch_thickness], [-W/2, L/2, 0]], # left
        [[W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [W/2, L/2, 0]],     # right
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [-W/2, -L/2, patch_thickness]], # front
        [[-W/2, L/2, 0], [W/2, L/2, 0], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]]      # back
    ]
    
    patch = Poly3DCollection(patch_verts, alpha=0.9, facecolor='gold', edgecolor='black')
    ax.add_collection3d(patch)
    
    # Microstrip feed line (based on feed direction)
    if feed_direction == FeedDirection.NEG_X:
        # Feed from negative X direction
        feed_start_x = -substrate_W/2
        feed_stop_x = -W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.POS_X:
        # Feed from positive X direction
        feed_start_x = W/2
        feed_stop_x = substrate_W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.NEG_Y:
        # Feed from negative Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = -substrate_L/2
        feed_stop_y = -L/2
    else:  # FeedDirection.POS_Y
        # Feed from positive Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = L/2
        feed_stop_y = substrate_L/2
    
    # Create microstrip feed line
    feed_verts = [
        [[feed_start_x, feed_start_y, 0], [feed_stop_x, feed_start_y, 0], [feed_stop_x, feed_stop_y, 0], [feed_start_x, feed_stop_y, 0]],  # bottom
        [[feed_start_x, feed_start_y, patch_thickness], [feed_stop_x, feed_start_y, patch_thickness], [feed_stop_x, feed_stop_y, patch_thickness], [feed_start_x, feed_stop_y, patch_thickness]],  # top
    ]
    
    feed_line = Poly3DCollection(feed_verts, alpha=0.9, facecolor='orange', edgecolor='black')
    ax.add_collection3d(feed_line)
    
    # Set equal aspect ratio and limits
    max_range = max(substrate_W, substrate_L, h*10) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*2, h*8])
    
    # Labels and styling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Microstrip-Fed Patch Antenna\nFeed: {feed_direction.value}, Line Width: {feed_width:.2f} mm')
    
    # Add dimension annotations
    ax.text(0, 0, h*4, f'{W:.1f} mm', ha='center', va='center', color='red', fontweight='bold')
    ax.text(0, 0, h*6, f'{L:.1f} mm', ha='center', va='center', color='green', fontweight='bold')
    
    # Add feed line annotation
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        feed_center_x = (feed_start_x + feed_stop_x) / 2
        ax.text(feed_center_x, 0, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    else:
        feed_center_y = (feed_start_y + feed_stop_y) / 2
        ax.text(0, feed_center_y, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    
    return fig


def plot_3d_pattern_from_grid(theta: np.ndarray,
                               phi: np.ndarray,
                               intensity: np.ndarray,
                               *,
                               L_m: float | None = None,
                               W_m: float | None = None,
                               h_m: float | None = None,
                               dB_min: float | None = None,
                               dB_max: float = 0.0,
                               fig_size=(8, 9),
                               colors_db: np.ndarray | None = None,
                               clip_db: float | None = None):
    """Render a professional 3D far-field surface from (theta, phi, intensity) grids.

    - Uses scientific colormap with proper dB scaling
    - Shows patch antenna at bottom for orientation reference
    - Professional styling with clean axes and labels
    """
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    th = np.asarray(theta).reshape(-1)
    ph = np.asarray(phi).reshape(-1)
    G = np.asarray(intensity)
    if G.ndim == 1:
        # expand to grid if needed
        if G.size == th.size:
            G = np.tile(G.reshape(th.size, 1), (1, ph.size))
        elif G.size == ph.size:
            G = np.tile(G.reshape(1, ph.size), (th.size, 1))
    thg, phg = np.meshgrid(th, ph, indexing="ij")

    # Normalize radius; color source can be normalized dB or true dBi via colors_db
    G = np.maximum(1e-16, G)
    Gn = G / np.nanmax(G)
    # Colors
    if colors_db is None:
        color_db = 10.0 * np.log10(np.maximum(1e-16, Gn))
    else:
        color_db = np.asarray(colors_db, dtype=float)
        if color_db.ndim == 3:
            color_db = color_db[0]
    # Apply clip if requested
    if clip_db is not None:
        Gn = np.where(color_db < clip_db, np.nan, Gn)
    # Smart dB range
    dB_max = float(dB_max)
    if dB_min is None:
        p10 = float(np.nanpercentile(color_db, 10))
        dB_min = max(-40.0, p10)
    dB_min = float(dB_min)
    if dB_max <= dB_min:
        dB_min = min(dB_min, -10.0)
        dB_max = 0.0
    color_db_clamped = np.clip(color_db, dB_min, dB_max)
    norm = (color_db_clamped - dB_min) / max(1e-9, (dB_max - dB_min))

    # Radius equals normalized gain; colors by dB
    r = Gn
    x = r * np.sin(thg) * np.cos(phg)
    y = r * np.sin(thg) * np.sin(phg)
    z = r * np.cos(thg)

    # Create figure with professional styling
    fig = plt.figure(figsize=fig_size, facecolor='white')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('white')

    # Use professional colormap (plasma for scientific visualization)
    facecolors = cm.plasma(norm)
    surface = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=facecolors,
                             linewidth=0, edgecolor="none", antialiased=True, 
                             shade=True, alpha=0.95)

    # Overlay ZX/ZY great-circle outlines to aid orientation (no isotropic sphere)
    phi0 = 0.0
    ph90 = np.pi/2
    t = np.linspace(0, np.pi, 181)
    x_zx, y_zx, z_zx = np.sin(t)*np.cos(phi0), np.sin(t)*np.sin(phi0), np.cos(t)
    ax.plot(0.98*x_zx, 0.98*y_zx, 0.98*z_zx, color="#999", lw=1.2, alpha=0.8)
    x_zy, y_zy, z_zy = np.sin(t)*np.cos(ph90), np.sin(t)*np.sin(ph90), np.cos(t)
    ax.plot(0.98*x_zy, 0.98*y_zy, 0.98*z_zy, color="#999", lw=1.2, alpha=0.8)

    # Add reference -3/-6/-10 dB shells to give scale
    for lvl in (-3.0, -6.0, -10.0, -15.0):
        rs = 10.0**(lvl/10.0)
        xs, ys, zs = rs*np.sin(thg)*np.cos(phg), rs*np.sin(thg)*np.sin(phg), rs*np.cos(thg)
        ax.plot_wireframe(xs[::8, ::8], ys[::8, ::8], zs[::8, ::8], color="#c7c7c7", linewidth=0.5, alpha=0.6)

    # Professional patch geometry at bottom
    if L_m is not None and W_m is not None:
        L = float(L_m)
        W = float(W_m)
        if max(L, W) > 0:
            scale = 0.5 / max(L, W)  # conservative scaling
            Ls = L * scale
            Ws = W * scale
            z0 = -1.08  # clearly below the sphere
            
            # Patch with realistic copper color
            patch_verts = [
                [-Ws/2, -Ls/2, z0], [Ws/2, -Ls/2, z0], [Ws/2, Ls/2, z0], [-Ws/2, Ls/2, z0]
            ]
            patch = Poly3DCollection([patch_verts], facecolor="#CD7F32", edgecolor="#8B4513", 
                                   alpha=0.9, linewidth=1.5)
            ax.add_collection3d(patch)
            
            # Feed point indicator
            ax.scatter([-Ws/4], [0], [z0 + 0.01], color='red', s=30, alpha=0.8)
            
            # Dimension labels
            ax.text(0, -Ls/2 - 0.1, z0, f'L={L*1000:.1f}mm', ha='center', fontsize=8, color='#333')
            ax.text(Ws/2 + 0.1, 0, z0, f'W={W*1000:.1f}mm', ha='center', rotation=90, fontsize=8, color='#333')

    # Set clean limits and aspect
    lim = 1.15
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])

    # Professional coordinate axes
    axis_color = '#2E86AB'
    ax.plot([0, 0.8], [0, 0], [0, 0], color=axis_color, lw=2.5, alpha=0.8)
    ax.plot([0, 0], [0, 0.8], [0, 0], color=axis_color, lw=2.5, alpha=0.8)
    ax.plot([0, 0], [0, 0], [0, 0.8], color=axis_color, lw=2.5, alpha=0.8)
    
    # Clean axis labels
    ax.text(0.85, 0, 0, "+X", fontsize=11, color=axis_color, weight='bold')
    ax.text(0, 0.85, 0, "+Y", fontsize=11, color=axis_color, weight='bold')
    ax.text(0, 0, 0.85, "+Z\n(Broadside)", fontsize=11, color=axis_color, weight='bold', ha='center')

    # Clean axis styling
    ax.set_xlabel("X", fontsize=10, color='#333')
    ax.set_ylabel("Y", fontsize=10, color='#333')
    ax.set_zlabel("Z", fontsize=10, color='#333')
    
    # Remove ticks and grid for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # Better viewing angle for patch antennas
    ax.view_init(elev=25, azim=-45)

    # Professional colorbar with dynamic nice ticks
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cm.plasma, norm=mpl.colors.Normalize(vmin=dB_min, vmax=dB_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.1, shrink=0.8)
    cbar.set_label("Gain (dBi)" if colors_db is not None else "Normalized gain (dB)", fontsize=11, color='#333')
    cbar.ax.tick_params(labelsize=9, colors='#333')
    span_db = abs(dB_max - dB_min)
    step = 1.0
    for s in (1.0, 2.0, 3.0, 5.0):
        if span_db / s <= 8:
            step = s
            break
    tick_start = step * np.ceil(dB_min / step)
    ticks = np.arange(tick_start, dB_max + 0.01, step)
    cbar.set_ticks(ticks)

    # Peak direction arrow for readability
    try:
        idx = np.unravel_index(np.argmax(Gn), Gn.shape)
        th_pk, ph_pk = thg[idx], phg[idx]
        r_pk = 1.02
        xpk, ypk, zpk = r_pk*np.sin(th_pk)*np.cos(ph_pk), r_pk*np.sin(th_pk)*np.sin(ph_pk), r_pk*np.cos(th_pk)
        ax.plot([0, xpk], [0, ypk], [0, zpk], color="#d62728", lw=2.5, alpha=0.9)
        ax.text(xpk, ypk, zpk, " peak", color="#d62728", fontsize=10)
    except Exception:
        pass

    # Professional title
    ax.set_title("3D Radiation Pattern\nNormalized Gain (dB)", fontsize=13, color='#333', pad=20)
    
    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    return fig


def draw_microstrip_patch_3d_geometry(L_m: float, W_m: float, h_m: float, 
                                       feed_direction: FeedDirection,
                                       frequency_hz: float, eps_r: float,
                                       feed_line_length_mm: float = 20.0,
                                       fig_size=(8, 6)):
    """Create a detailed 3D visualization of the microstrip-fed patch antenna geometry."""
    mm = 1e3
    L = L_m * mm  # patch length (Y direction)
    W = W_m * mm  # patch width (X direction)
    h = h_m * mm  # substrate thickness
    
    # Calculate microstrip feed line width for 50Ω
    feed_width = calculate_microstrip_width(frequency_hz, eps_r, h_m) * mm
    
    # Substrate dimensions (larger than patch + feed)
    substrate_margin = 30.0  # mm
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        substrate_W = W + 2 * substrate_margin + feed_line_length_mm
        substrate_L = L + 2 * substrate_margin
    else:
        substrate_W = W + 2 * substrate_margin
        substrate_L = L + 2 * substrate_margin + feed_line_length_mm
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Substrate (bottom layer)
    substrate_verts = [
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]],  # bottom
        [[-substrate_W/2, -substrate_L/2, 0], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]],      # top
        [[-substrate_W/2, -substrate_L/2, -h], [-substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, -h]], # left
        [[substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [substrate_W/2, substrate_L/2, 0], [substrate_W/2, substrate_L/2, -h]],     # right
        [[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, 0], [-substrate_W/2, -substrate_L/2, 0]], # front
        [[-substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [substrate_W/2, substrate_L/2, 0], [-substrate_W/2, substrate_L/2, 0]]      # back
    ]
    
    substrate = Poly3DCollection(substrate_verts, alpha=0.3, facecolor='lightblue', edgecolor='gray')
    ax.add_collection3d(substrate)
    
    # Ground plane (bottom of substrate)
    ground_verts = [[[-substrate_W/2, -substrate_L/2, -h], [substrate_W/2, -substrate_L/2, -h], [substrate_W/2, substrate_L/2, -h], [-substrate_W/2, substrate_L/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.8, facecolor='silver', edgecolor='black')
    ax.add_collection3d(ground)
    
    # Patch (top metal layer)
    patch_thickness = 0.035  # 35 microns in mm
    patch_verts = [
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, L/2, 0], [-W/2, L/2, 0]],  # bottom
        [[-W/2, -L/2, patch_thickness], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]],  # top
        [[-W/2, -L/2, 0], [-W/2, -L/2, patch_thickness], [-W/2, L/2, patch_thickness], [-W/2, L/2, 0]], # left
        [[W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [W/2, L/2, patch_thickness], [W/2, L/2, 0]],     # right
        [[-W/2, -L/2, 0], [W/2, -L/2, 0], [W/2, -L/2, patch_thickness], [-W/2, -L/2, patch_thickness]], # front
        [[-W/2, L/2, 0], [W/2, L/2, 0], [W/2, L/2, patch_thickness], [-W/2, L/2, patch_thickness]]      # back
    ]
    
    patch = Poly3DCollection(patch_verts, alpha=0.9, facecolor='gold', edgecolor='black')
    ax.add_collection3d(patch)
    
    # Microstrip feed line (based on feed direction)
    if feed_direction == FeedDirection.NEG_X:
        # Feed from negative X direction
        feed_start_x = -substrate_W/2
        feed_stop_x = -W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.POS_X:
        # Feed from positive X direction
        feed_start_x = W/2
        feed_stop_x = substrate_W/2
        feed_start_y = -feed_width/2
        feed_stop_y = feed_width/2
    elif feed_direction == FeedDirection.NEG_Y:
        # Feed from negative Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = -substrate_L/2
        feed_stop_y = -L/2
    else:  # FeedDirection.POS_Y
        # Feed from positive Y direction
        feed_start_x = -feed_width/2
        feed_stop_x = feed_width/2
        feed_start_y = L/2
        feed_stop_y = substrate_L/2
    
    # Create microstrip feed line
    feed_verts = [
        [[feed_start_x, feed_start_y, 0], [feed_stop_x, feed_start_y, 0], [feed_stop_x, feed_stop_y, 0], [feed_start_x, feed_stop_y, 0]],  # bottom
        [[feed_start_x, feed_start_y, patch_thickness], [feed_stop_x, feed_start_y, patch_thickness], [feed_stop_x, feed_stop_y, patch_thickness], [feed_start_x, feed_stop_y, patch_thickness]],  # top
    ]
    
    feed_line = Poly3DCollection(feed_verts, alpha=0.9, facecolor='orange', edgecolor='black')
    ax.add_collection3d(feed_line)
    
    # Set equal aspect ratio and limits
    max_range = max(substrate_W, substrate_L, h*10) / 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-h*2, h*8])
    
    # Labels and styling
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Microstrip-Fed Patch Antenna\nFeed: {feed_direction.value}, Line Width: {feed_width:.2f} mm')
    
    # Add dimension annotations
    ax.text(0, 0, h*4, f'{W:.1f} mm', ha='center', va='center', color='red', fontweight='bold')
    ax.text(0, 0, h*6, f'{L:.1f} mm', ha='center', va='center', color='green', fontweight='bold')
    
    # Add feed line annotation
    if feed_direction in [FeedDirection.NEG_X, FeedDirection.POS_X]:
        feed_center_x = (feed_start_x + feed_stop_x) / 2
        ax.text(feed_center_x, 0, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    else:
        feed_center_y = (feed_start_y + feed_stop_y) / 2
        ax.text(0, feed_center_y, h*2, f'Feed: {feed_width:.1f} mm', ha='center', va='center', color='orange', fontweight='bold')
    
    return fig
