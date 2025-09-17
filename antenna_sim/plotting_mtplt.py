import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_patch_3d_geometry(L_m: float, W_m: float, h_m: float, fig_size=(8, 6), show_labels: bool = True):
    """Draw patch + substrate using a stable two-layer rendering model.

    - Main axis: substrate sides/bottom, ground, patch sides/bottom
    - Overlay axis: patch top face only (opaque), labels

    Returns a matplotlib Figure with an attribute `_overlay_ax` set to the
    transparent overlay axis. Callers can add additional copper (e.g. microstrip)
    to this axis for correct draw order.
    """
    mm = 1e3
    L = L_m * mm
    W = W_m * mm
    h = h_m * mm

    margin = max(5.0, 0.2 * max(L, W))
    sub_L = L + 2 * margin
    sub_W = W + 2 * margin

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Substrate: draw bottom + sides (top omitted). Keep sides a hair below z=0
    side_top_z = -0.02
    substrate_side_bottom = [
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]],
        [[-sub_L/2, -sub_W/2, -h], [-sub_L/2, -sub_W/2, side_top_z], [-sub_L/2, sub_W/2, side_top_z], [-sub_L/2, sub_W/2, -h]],
        [[sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, side_top_z], [sub_L/2, sub_W/2, side_top_z], [sub_L/2, sub_W/2, -h]],
        [[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, side_top_z], [-sub_L/2, -sub_W/2, side_top_z]],
        [[-sub_L/2,  sub_W/2, -h], [sub_L/2,  sub_W/2, -h], [sub_L/2,  sub_W/2, side_top_z], [-sub_L/2,  sub_W/2, side_top_z]],
    ]
    substrate = Poly3DCollection(substrate_side_bottom, alpha=0.45, facecolor='#2e7d32', edgecolor='#1b5e20', linewidth=1.0)
    try:
        substrate.set_zsort('min')
    except Exception:
        pass
    ax.add_collection3d(substrate)

    # Ground plane
    ground_verts = [[[-sub_L/2, -sub_W/2, -h], [sub_L/2, -sub_W/2, -h], [sub_L/2, sub_W/2, -h], [-sub_L/2, sub_W/2, -h]]]
    ground = Poly3DCollection(ground_verts, alpha=0.9, facecolor='#9ea7ad', edgecolor='#6b7074')
    ax.add_collection3d(ground)

    # Patch sides + bottom on main axis
    patch_thickness = max(0.08, 0.06 * h)
    patch_sides_bottom = [
        [[-L/2, -W/2, 0], [ L/2, -W/2, 0], [ L/2,  W/2, 0], [-L/2,  W/2, 0]],  # bottom
        [[-L/2, -W/2, 0], [-L/2, -W/2, patch_thickness], [-L/2, W/2, patch_thickness], [-L/2, W/2, 0]],
        [[ L/2, -W/2, 0], [ L/2, -W/2, patch_thickness], [ L/2, W/2, patch_thickness], [ L/2, W/2, 0]],
        [[-L/2, -W/2, 0], [ L/2, -W/2, 0], [ L/2, -W/2, patch_thickness], [-L/2, -W/2, patch_thickness]],
        [[-L/2,  W/2, 0], [ L/2,  W/2, 0], [ L/2,  W/2, patch_thickness], [-L/2,  W/2, patch_thickness]],
    ]
    patch_body = Poly3DCollection(patch_sides_bottom, alpha=0.98, facecolor='#d9b43a', edgecolor='#b8860b', linewidth=1.2)
    try:
        patch_body.set_zsort('max')
    except Exception:
        pass
    ax.add_collection3d(patch_body)

    # Overlay axis for patch top face + labels/copper overlays
    ax_overlay = fig.add_axes(ax.get_position(), projection='3d')
    ax_overlay.patch.set_alpha(0)
    ax_overlay.set_axis_off()
    # Prevent the overlay axis from grabbing mouse interactions
    try:
        ax_overlay.set_navigate(False)
        ax_overlay.mouse_init(rotate_btn=None, pan_btn=None)
    except Exception:
        pass
    try:
        for pane in [ax_overlay.w_xaxis.get_pane(), ax_overlay.w_yaxis.get_pane(), ax_overlay.w_zaxis.get_pane()]:
            pane.set_edgecolor((0,0,0,0))
            pane.set_facecolor((0,0,0,0))
    except Exception:
        pass
    # Sync limits & view
    max_range = max(sub_L, sub_W, h*3) / 2
    for ax_sync in (ax, ax_overlay):
        ax_sync.set_xlim([-max_range, max_range])
        ax_sync.set_ylim([-max_range, max_range])
        ax_sync.set_zlim([-h*1.2, h*0.5])
    ax_overlay.view_init(elev=getattr(ax, 'elev', 20), azim=getattr(ax, 'azim', 45))

    # Patch top face on overlay axis (opaque)
    patch_top = Poly3DCollection([[[-L/2, -W/2, patch_thickness], [L/2, -W/2, patch_thickness], [L/2, W/2, patch_thickness], [-L/2, W/2, patch_thickness]]],
                                 alpha=1.0, facecolor='#ffd24d', edgecolor='#b8860b', linewidth=1.0)
    try:
        patch_top.set_zsort('max')
    except Exception:
        pass
    patch_top.set_zorder(20)
    ax_overlay.add_collection3d(patch_top)

    # Save overlay axis on the figure for callers to add microstrip, etc.
    fig._overlay_ax = ax_overlay

    # Labels
    label_box = dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.6)
    if show_labels:
        ax_overlay.text(0, -W/2-0.9*margin, patch_thickness*1.2, f'L = {L:.1f} mm', ha='center', fontsize=13, color='white', bbox=label_box, zorder=25)
        ax_overlay.text(L/2+0.9*margin, 0, patch_thickness*1.2, f'W = {W:.1f} mm', ha='center', rotation=90, fontsize=13, color='white', bbox=label_box, zorder=25)

    # Thickness indicator on main axis
    xh, yh = -sub_L/2 + 0.15*sub_L, -sub_W/2 + 0.15*sub_W
    ax.plot([xh, xh], [yh, yh], [-h, 0], color='#ff7043', linewidth=2.0)
    ax.text(xh, yh, -h*0.5, f'h = {h:.2f} mm', color='white', fontsize=13, bbox=label_box, ha='center')

    # Axes
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title('3D Patch Antenna Geometry\n(Gold patch on green substrate with ground plane)')
    ax.view_init(elev=20, azim=45)

    # Expose base axis too for synchronization
    fig._base_ax = ax
    return fig


