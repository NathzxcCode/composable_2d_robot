"""
Collision Factor Experiment
===========================
Standalone validation of the Bhattacharyya-distance ellipsoid collision factor
defined in planning_loop.py.

Two limb segments are placed in a given configuration. A minimal GTSAM factor
graph — containing only the collision factor and four soft anchor priors — is
optimised with Levenberg-Marquardt.

The resulting figure shows:
  - Ghost (dashed, translucent): initial limb positions
  - Solid (opaque): optimised limb positions
  - Arrows at each initial endpoint: repulsion direction (negative gradient
    of the collision factor error w.r.t. endpoint position)

The key visual claim: arrows at the two endpoints of the SAME limb point in
noticeably different directions, demonstrating the rotational component of the
factor's Jacobian — not merely a translational push.

Run this script to generate one figure per configuration:
    oblique, parallel, a_frame, t_junction, scissors
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gtsam

# ── path setup ────────────────────────────────────────────────────────────────
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from planning_loop import _make_ellipsoid_collision_factor

# ── shared constants ──────────────────────────────────────────────────────────
L          = 0.15    # limb length (m)
R_DRAW     = 0.02    # visual half-width of each limb rectangle (m)
R_FACTOR   = 0.03    # Gaussian radius used inside the collision factor (m)
K_FACTOR   = 8.0     # steepness exponent in exp(-k * D_B)
COST_SIGMA = 0.05    # collision factor noise sigma
ANCHOR_POS_SIGMA   = 0.05      # positional freedom of anchor priors (m)
ANCHOR_THETA_SIGMA = 2 * np.pi # theta essentially unconstrained

ARROW_DISPLAY_LEN = 0.045  # all arrows normalised to this display length (m)

COLOR_A      = '#1f77b4'
COLOR_A_DARK = '#0d4f8a'
COLOR_B      = '#ff7f0e'
COLOR_B_DARK = '#a34d00'

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'planning_results')


# ══════════════════════════════════════════════════════════════════════════════
# Configurations
# ══════════════════════════════════════════════════════════════════════════════

def _hv(angle_deg, cx=0.0, cy=0.0):
    """Return (p1, p2) for a limb of length L centred at (cx,cy) at angle_deg."""
    a = np.deg2rad(angle_deg)
    half = 0.5 * L * np.array([np.cos(a), np.sin(a)])
    c = np.array([cx, cy])
    return c - half, c + half


def build_configs():
    """
    Returns an ordered list of (name, description, xA1, xA2, xB1, xB2).

    Configurations are chosen so the limbs start close but NOT fully
    coincident — each represents a geometrically meaningful relative posture.
    """
    configs = []

    # ── oblique crossing ──────────────────────────────────────────────────
    # Original config: limbs cross at ~40°, centroids nearly coincident.
    # Strongest rotational response (arrow angle ~168°).
    xA1, xA2 = _hv(0,  0.00,  0.00)
    xB1, xB2 = _hv(40, 0.01,  0.01)
    configs.append(('oblique',
                    'Oblique crossing  (~40°, centroids overlapping)',
                    xA1, xA2, xB1, xB2))

    # ── parallel ──────────────────────────────────────────────────────────
    # Both limbs horizontal, offset 0.05 m vertically (1 cm visual gap).
    # Predominantly translational response; moderate rotational component.
    xA1, xA2 = _hv(0, 0.00,  0.025)
    xB1, xB2 = _hv(0, 0.00, -0.025)
    configs.append(('parallel',
                    'Parallel  (same orientation, 5 cm vertical separation)',
                    xA1, xA2, xB1, xB2))

    # ── A-frame ───────────────────────────────────────────────────────────
    # Two limbs angled inward at ±70°, tips nearly touching at the top
    # (~1.4 mm gap). Bases spread apart. Strong rotational response.
    xA1, xA2 = _hv( 70, -0.025, 0.0)
    xB1, xB2 = _hv(110,  0.025, 0.0)
    configs.append(('a_frame',
                    'A-frame  (tips nearly touching, bases spread)',
                    xA1, xA2, xB1, xB2))

    # ── T-junction ────────────────────────────────────────────────────────
    # Horizontal bar (A) with vertical stem (B) approaching from below.
    # B centred 3 cm above A centre (slight x-offset for asymmetry).
    # Strong shape-term response (perpendicular covariances).
    xA1, xA2 = _hv(0,   0.000, 0.000)
    xB1, xB2 = _hv(90,  0.005, 0.030)
    configs.append(('t_junction',
                    'T-junction  (perpendicular approach, 3 cm separation)',
                    xA1, xA2, xB1, xB2))

    # ── scissors ─────────────────────────────────────────────────────────
    # Two limbs at 20° apart, centroids offset by ~3.6 cm (near crossing).
    # Intermediate rotational response; tests small-angle near-contact.
    xA1, xA2 = _hv(  0, -0.010, -0.015)
    xB1, xB2 = _hv( 20,  0.010,  0.015)
    configs.append(('scissors',
                    'Scissors  (20° angle, centroids ~3.6 cm apart)',
                    xA1, xA2, xB1, xB2))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# Standalone gradient helper  (mirrors planning_loop.py:121-156)
# ══════════════════════════════════════════════════════════════════════════════

def collision_gradients(xA1, xA2, xB1, xB2, k, r):
    """
    Evaluate the collision factor error and per-endpoint repulsion directions.

    Parameters
    ----------
    xA1, xA2, xB1, xB2 : (2,) ndarray — 2-D world positions
    k  : float — steepness exponent
    r  : float — Gaussian half-width

    Returns
    -------
    error  : float — exp(-k * D_B)
    D_B    : float — Bhattacharyya distance
    gA1..gB2 : (2,) ndarrays — repulsion direction (negative gradient)
    """
    mu_A = 0.5 * (xA1 + xA2)
    mu_B = 0.5 * (xB1 + xB2)
    v_A  = xA2 - xA1
    v_B  = xB2 - xB1

    Sigma_A   = 0.25 * np.outer(v_A, v_A) + (r ** 2) * np.eye(2)
    Sigma_B   = 0.25 * np.outer(v_B, v_B) + (r ** 2) * np.eye(2)
    Sigma     = 0.5 * (Sigma_A + Sigma_B)
    Sigma_inv = np.linalg.inv(Sigma)

    d_mu        = mu_A - mu_B
    mahalanobis = 0.125 * (d_mu @ Sigma_inv @ d_mu)
    shape_term  = 0.5 * np.log(
        np.linalg.det(Sigma) /
        np.sqrt(np.linalg.det(Sigma_A) * np.linalg.det(Sigma_B))
    )
    D_B   = mahalanobis + shape_term
    error = float(np.exp(-k * D_B))

    dE_dDB    = -k * error
    dDB_dmuA  = 0.25 * (Sigma_inv @ d_mu)
    dDB_dmuB  = -dDB_dmuA
    dE_dpos_A = dE_dDB * 0.5 * dDB_dmuA
    dE_dpos_B = dE_dDB * 0.5 * dDB_dmuB

    dDB_dSigma = (
        -0.125 * np.outer(Sigma_inv @ d_mu, Sigma_inv @ d_mu)
        + 0.5 * Sigma_inv
    )
    dE_dSigma  = dE_dDB * dDB_dSigma

    dE_dcov_A2 = np.zeros(2)
    dE_dcov_B2 = np.zeros(2)
    for idx in range(2):
        M_A = np.zeros((2, 2)); M_A[idx, :] += v_A; M_A[:, idx] += v_A
        dE_dcov_A2[idx] = np.sum(dE_dSigma * (0.5 * 0.25 * M_A))
        M_B = np.zeros((2, 2)); M_B[idx, :] += v_B; M_B[:, idx] += v_B
        dE_dcov_B2[idx] = np.sum(dE_dSigma * (0.5 * 0.25 * M_B))

    # Repulsion = negative gradient (direction optimiser moves each endpoint)
    return (
        error, D_B,
        -(dE_dpos_A - dE_dcov_A2),   # gA1
        -(dE_dpos_A + dE_dcov_A2),   # gA2
        -(dE_dpos_B - dE_dcov_B2),   # gB1
        -(dE_dpos_B + dE_dcov_B2),   # gB2
    )


# ══════════════════════════════════════════════════════════════════════════════
# GTSAM keys + noise
# ══════════════════════════════════════════════════════════════════════════════

KEY_A1 = gtsam.Symbol('a', 1).key()
KEY_A2 = gtsam.Symbol('a', 2).key()
KEY_B1 = gtsam.Symbol('b', 1).key()
KEY_B2 = gtsam.Symbol('b', 2).key()

ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([ANCHOR_POS_SIGMA, ANCHOR_POS_SIGMA, ANCHOR_THETA_SIGMA])
)


def _pose(xy, theta=0.0):
    return gtsam.Pose2(float(xy[0]), float(xy[1]), float(theta))


def run_optimisation(xA1, xA2, xB1, xB2):
    """
    Build a minimal factor graph (collision factor + 4 soft anchors) and
    return the optimised gtsam.Values.
    """
    def angle(p1, p2):
        d = p2 - p1
        return float(np.arctan2(d[1], d[0]))

    pA1 = _pose(xA1, angle(xA1, xA2))
    pA2 = _pose(xA2, angle(xA1, xA2))
    pB1 = _pose(xB1, angle(xB1, xB2))
    pB2 = _pose(xB2, angle(xB1, xB2))

    graph = gtsam.NonlinearFactorGraph()
    graph.add(_make_ellipsoid_collision_factor(
        KEY_A1, KEY_A2, KEY_B1, KEY_B2,
        k=K_FACTOR, r=R_FACTOR, cost_sigma=COST_SIGMA,
    ))
    for key, pose in [(KEY_A1, pA1), (KEY_A2, pA2),
                      (KEY_B1, pB1), (KEY_B2, pB2)]:
        graph.add(gtsam.PriorFactorPose2(key, pose, ANCHOR_NOISE))

    initial = gtsam.Values()
    initial.insert(KEY_A1, pA1); initial.insert(KEY_A2, pA2)
    initial.insert(KEY_B1, pB1); initial.insert(KEY_B2, pB2)

    result = gtsam.LevenbergMarquardtOptimizer(
        graph, initial, gtsam.LevenbergMarquardtParams()
    ).optimize()
    return result


def extract_positions(result):
    def pos(key):
        p = result.atPose2(key)
        return np.array([p.x(), p.y()])
    return pos(KEY_A1), pos(KEY_A2), pos(KEY_B1), pos(KEY_B2)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_limb(ax, p1, p2, hw, color, alpha, linestyle='solid', zorder=2):
    """Filled capsule-rectangle between p1 and p2 with half-width hw."""
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-9:
        return
    unit = d / length
    perp = np.array([-unit[1], unit[0]])
    corners = np.array([
        p1 - hw * perp,
        p1 + hw * perp,
        p2 + hw * perp,
        p2 - hw * perp,
    ])
    ax.add_patch(plt.Polygon(
        corners, closed=True,
        facecolor=color, edgecolor=color,
        alpha=alpha, linewidth=1.5, linestyle=linestyle, zorder=zorder,
    ))
    for pt in [p1, p2]:
        ax.add_patch(plt.Circle(pt, hw, color=color, alpha=alpha,
                                linewidth=0, zorder=zorder))


def draw_arrow(ax, base, direction, length, color, zorder=7):
    """Arrow normalised to `length`, base at `base`, direction from `direction`."""
    mag = np.linalg.norm(direction)
    if mag < 1e-12:
        return
    tip = base + length * (direction / mag)
    ax.annotate('', xy=tip, xytext=base,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.0, mutation_scale=14),
                zorder=zorder)


def _outward_offset(pt, all_pts, dist=0.020):
    """
    Return a label position offset outward from the centroid of all_pts.
    Falls back to a small upward offset if pt is at the centroid.
    """
    centroid = all_pts.mean(axis=0)
    diff = pt - centroid
    norm = np.linalg.norm(diff)
    if norm < 1e-9:
        return pt + np.array([0.0, dist])
    return pt + dist * (diff / norm)


# ══════════════════════════════════════════════════════════════════════════════
# Main plot function
# ══════════════════════════════════════════════════════════════════════════════

def plot_config(name, description,
                initial_pts, final_pts,
                gA1, gA2, gB1, gB2,
                D_B_init, D_B_final):
    """
    Render the collision factor figure for one configuration.

    Layers (back → front):
      1. Ghost limbs  — initial, dashed, translucent
      2. Final limbs  — optimised, solid, opaque
      3. Endpoint dots at initial positions
      4. Repulsion arrows (normalised, at initial positions)
      5. Endpoint labels (placed outward from cluster centroid)
    """
    xA1_i, xA2_i, xB1_i, xB2_i = initial_pts
    xA1_f, xA2_f, xB1_f, xB2_f = final_pts

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='#aaaaaa')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(
        f'Collision Factor — {name.replace("_", " ").title()}\n'
        f'\\fontsize{{9}}{{12}}\\selectfont {description}',
        fontsize=13, pad=14,
    )
    # Simpler title without LaTeX:
    ax.set_title(
        f'Collision Factor: {name.replace("_", " ").title()}',
        fontsize=13, pad=6,
    )
    ax.text(0.5, 1.01, description,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, color='#555555')

    # ── 1. Ghost limbs ────────────────────────────────────────────────────
    draw_limb(ax, xA1_i, xA2_i, R_DRAW, COLOR_A, alpha=0.18,
              linestyle='dashed', zorder=2)
    draw_limb(ax, xB1_i, xB2_i, R_DRAW, COLOR_B, alpha=0.18,
              linestyle='dashed', zorder=2)

    # ── 2. Final limbs ────────────────────────────────────────────────────
    draw_limb(ax, xA1_f, xA2_f, R_DRAW, COLOR_A, alpha=0.65,
              linestyle='solid', zorder=3)
    draw_limb(ax, xB1_f, xB2_f, R_DRAW, COLOR_B, alpha=0.65,
              linestyle='solid', zorder=3)

    # ── 3. Endpoint dots (initial positions) ─────────────────────────────
    dot_r = R_DRAW * 0.4
    for pt in [xA1_i, xA2_i]:
        ax.add_patch(plt.Circle(pt, dot_r, color=COLOR_A_DARK, zorder=6))
    for pt in [xB1_i, xB2_i]:
        ax.add_patch(plt.Circle(pt, dot_r, color=COLOR_B_DARK, zorder=6))

    # ── 4. Repulsion arrows ───────────────────────────────────────────────
    for base, grad, col in [
        (xA1_i, gA1, COLOR_A_DARK),
        (xA2_i, gA2, COLOR_A_DARK),
        (xB1_i, gB1, COLOR_B_DARK),
        (xB2_i, gB2, COLOR_B_DARK),
    ]:
        draw_arrow(ax, base, grad, ARROW_DISPLAY_LEN, col, zorder=7)

    # ── 5. Endpoint labels (dynamic outward placement) ────────────────────
    all_initial = np.array([xA1_i, xA2_i, xB1_i, xB2_i])
    label_data = [
        (xA1_i, 'A1', COLOR_A_DARK),
        (xA2_i, 'A2', COLOR_A_DARK),
        (xB1_i, 'B1', COLOR_B_DARK),
        (xB2_i, 'B2', COLOR_B_DARK),
    ]
    for pt, lbl, col in label_data:
        lpos = _outward_offset(pt, all_initial, dist=0.022)
        ax.text(lpos[0], lpos[1], lbl, fontsize=10, color=col,
                fontweight='bold', ha='center', va='center', zorder=8)

    # ── annotation box ────────────────────────────────────────────────────
    angle_A = np.degrees(np.arccos(np.clip(
        np.dot(gA1 / np.linalg.norm(gA1), gA2 / np.linalg.norm(gA2)), -1, 1
    )))
    angle_B = np.degrees(np.arccos(np.clip(
        np.dot(gB1 / np.linalg.norm(gB1), gB2 / np.linalg.norm(gB2)), -1, 1
    )))
    db_text = (
        f'$D_B$  initial: {D_B_init:.3f}\n'
        f'$D_B$  final:    {D_B_final:.3f}\n'
        f'A-arrow angle: {angle_A:.0f}°\n'
        f'B-arrow angle: {angle_B:.0f}°'
    )
    ax.text(
        0.97, 0.97, db_text,
        transform=ax.transAxes, fontsize=9.5,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#cccccc', alpha=0.92),
        zorder=9,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_A, alpha=0.22, edgecolor=COLOR_A,
                       linestyle='--', linewidth=1.5, label='Limb A — initial'),
        mpatches.Patch(facecolor=COLOR_A, alpha=0.70, edgecolor=COLOR_A,
                       label='Limb A — final'),
        mpatches.Patch(facecolor=COLOR_B, alpha=0.22, edgecolor=COLOR_B,
                       linestyle='--', linewidth=1.5, label='Limb B — initial'),
        mpatches.Patch(facecolor=COLOR_B, alpha=0.70, edgecolor=COLOR_B,
                       label='Limb B — final'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.003,
                            facecolor='#444444', edgecolor='#444444',
                            label='Repulsion direction (normalised)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=9, framealpha=0.9, edgecolor='#cccccc')

    # ── Axis limits: pad around all points ────────────────────────────────
    all_pts = np.vstack([all_initial,
                         np.array([xA1_f, xA2_f, xB1_f, xB2_f])])
    pad = 0.08
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'collision_factor_{name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Per-config runner
# ══════════════════════════════════════════════════════════════════════════════

def run_config(name, description, xA1, xA2, xB1, xB2):
    print(f'\n── {name} ──────────────────────────────────')
    print(f'   {description}')

    # Evaluate factor at initial config
    error_i, D_B_i, gA1, gA2, gB1, gB2 = collision_gradients(
        xA1, xA2, xB1, xB2, k=K_FACTOR, r=R_FACTOR
    )
    angle_A = np.degrees(np.arccos(np.clip(
        np.dot(gA1 / np.linalg.norm(gA1), gA2 / np.linalg.norm(gA2)), -1, 1
    )))
    print(f'   Initial: D_B={D_B_i:.3f}  error={error_i:.3f}'
          f'  A-arrow-angle={angle_A:.1f}°')

    # Optimise
    result = run_optimisation(xA1, xA2, xB1, xB2)
    xA1_f, xA2_f, xB1_f, xB2_f = extract_positions(result)

    # Evaluate at final config
    _, D_B_f, *_ = collision_gradients(
        xA1_f, xA2_f, xB1_f, xB2_f, k=K_FACTOR, r=R_FACTOR
    )
    print(f'   Final:   D_B={D_B_f:.3f}  (ΔD_B={D_B_f - D_B_i:+.3f})')

    displacements = {
        'A1': xA1_f - xA1, 'A2': xA2_f - xA2,
        'B1': xB1_f - xB1, 'B2': xB2_f - xB2,
    }
    for ep, d in displacements.items():
        print(f'   {ep}: Δ=({d[0]:+.4f}, {d[1]:+.4f})  |Δ|={np.linalg.norm(d):.4f} m')

    # Plot
    save_path = plot_config(
        name, description,
        (xA1, xA2, xB1, xB2),
        (xA1_f, xA2_f, xB1_f, xB2_f),
        gA1, gA2, gB1, gB2,
        D_B_i, D_B_f,
    )
    print(f'   Saved → {os.path.basename(save_path)}')
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot  (parallel + oblique side by side, single legend)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_panel(ax, name, initial_pts, final_pts,
                gA1, gA2, gB1, gB2, D_B_init, D_B_final, panel_label):
    """
    Draw one panel of the comparison figure into `ax`.
    Returns the four legend-proxy handles (ghost A, final A, ghost B, final B)
    so the caller can build a single shared legend.
    No legend is drawn inside the panel.
    """
    xA1_i, xA2_i, xB1_i, xB2_i = initial_pts
    xA1_f, xA2_f, xB1_f, xB2_f = final_pts

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='#aaaaaa')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)

    # Panel label in top-left corner
    ax.text(0.03, 0.97, panel_label,
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='left', zorder=10)

    # ── Ghost limbs ───────────────────────────────────────────────────────
    draw_limb(ax, xA1_i, xA2_i, R_DRAW, COLOR_A, alpha=0.18,
              linestyle='dashed', zorder=2)
    draw_limb(ax, xB1_i, xB2_i, R_DRAW, COLOR_B, alpha=0.18,
              linestyle='dashed', zorder=2)

    # ── Final limbs ───────────────────────────────────────────────────────
    draw_limb(ax, xA1_f, xA2_f, R_DRAW, COLOR_A, alpha=0.65,
              linestyle='solid', zorder=3)
    draw_limb(ax, xB1_f, xB2_f, R_DRAW, COLOR_B, alpha=0.65,
              linestyle='solid', zorder=3)

    # ── Endpoint dots ─────────────────────────────────────────────────────
    dot_r = R_DRAW * 0.4
    for pt in [xA1_i, xA2_i]:
        ax.add_patch(plt.Circle(pt, dot_r, color=COLOR_A_DARK, zorder=6))
    for pt in [xB1_i, xB2_i]:
        ax.add_patch(plt.Circle(pt, dot_r, color=COLOR_B_DARK, zorder=6))

    # ── Repulsion arrows ──────────────────────────────────────────────────
    for base, grad, col in [
        (xA1_i, gA1, COLOR_A_DARK),
        (xA2_i, gA2, COLOR_A_DARK),
        (xB1_i, gB1, COLOR_B_DARK),
        (xB2_i, gB2, COLOR_B_DARK),
    ]:
        draw_arrow(ax, base, grad, ARROW_DISPLAY_LEN, col, zorder=7)

    # ── Endpoint labels ───────────────────────────────────────────────────
    all_initial = np.array([xA1_i, xA2_i, xB1_i, xB2_i])
    for pt, lbl, col in [
        (xA1_i, 'A1', COLOR_A_DARK),
        (xA2_i, 'A2', COLOR_A_DARK),
        (xB1_i, 'B1', COLOR_B_DARK),
        (xB2_i, 'B2', COLOR_B_DARK),
    ]:
        lpos = _outward_offset(pt, all_initial, dist=0.022)
        ax.text(lpos[0], lpos[1], lbl, fontsize=10, color=col,
                fontweight='bold', ha='center', va='center', zorder=8)

    # ── annotation box ────────────────────────────────────────────────────
    angle_A = np.degrees(np.arccos(np.clip(
        np.dot(gA1 / np.linalg.norm(gA1), gA2 / np.linalg.norm(gA2)), -1, 1
    )))
    angle_B = np.degrees(np.arccos(np.clip(
        np.dot(gB1 / np.linalg.norm(gB1), gB2 / np.linalg.norm(gB2)), -1, 1
    )))
    db_text = (
        f'$D_B$  {D_B_init:.3f} → {D_B_final:.3f}\n'
        f'A-arrow angle: {angle_A:.0f}°\n'
        f'B-arrow angle: {angle_B:.0f}°'
    )
    ax.text(0.97, 0.97, db_text,
            transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#cccccc', alpha=0.92),
            zorder=9)

    # ── Axis limits: equal scale, padded around all points ────────────────
    all_pts = np.vstack([all_initial,
                         np.array([xA1_f, xA2_f, xB1_f, xB2_f])])
    cx, cy = all_pts.mean(axis=0)
    half = 0.17   # fixed half-range so both panels use the same visual scale
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    # Return proxy artists for the shared legend (created once by the caller)
    return [
        mpatches.Patch(facecolor=COLOR_A, alpha=0.22, edgecolor=COLOR_A,
                       linestyle='--', linewidth=1.5, label='Limb A — initial'),
        mpatches.Patch(facecolor=COLOR_A, alpha=0.70, edgecolor=COLOR_A,
                       label='Limb A — final'),
        mpatches.Patch(facecolor=COLOR_B, alpha=0.22, edgecolor=COLOR_B,
                       linestyle='--', linewidth=1.5, label='Limb B — initial'),
        mpatches.Patch(facecolor=COLOR_B, alpha=0.70, edgecolor=COLOR_B,
                       label='Limb B — final'),
        mpatches.FancyArrow(0, 0, 1, 0, width=0.003,
                            facecolor='#444444', edgecolor='#444444',
                            label='Repulsion direction (normalised)'),
    ]


def plot_parallel_oblique_comparison(parallel_data, oblique_data):
    """
    Two-panel figure: (a) Parallel, (b) Oblique.
    Single shared legend centred below both panels.

    Each data tuple: (initial_pts, final_pts, gA1, gA2, gB1, gB2,
                      D_B_init, D_B_final)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    legend_handles = _draw_panel(
        axes[0], 'parallel',
        *parallel_data,
        panel_label='(a) Parallel',
    )
    _draw_panel(
        axes[1], 'oblique',
        *oblique_data,
        panel_label='(b) Oblique',
    )

    # Single legend centred below both panels
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=5,
        fontsize=10,
        framealpha=0.9,
        edgecolor='#cccccc',
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        'Collision Factor: Bhattacharyya Ellipsoid Repulsion',
        fontsize=14, y=1.01,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save_path = os.path.join(
        RESULTS_DIR, 'collision_factor_comparison_parallel_oblique.png'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _gather_config_data(xA1, xA2, xB1, xB2):
    """
    Run optimisation for one config and return a data tuple suitable for
    _draw_panel / plot_parallel_oblique_comparison.

    Returns (initial_pts, final_pts, gA1, gA2, gB1, gB2, D_B_init, D_B_final)
    """
    error_i, D_B_i, gA1, gA2, gB1, gB2 = collision_gradients(
        xA1, xA2, xB1, xB2, k=K_FACTOR, r=R_FACTOR
    )
    result = run_optimisation(xA1, xA2, xB1, xB2)
    xA1_f, xA2_f, xB1_f, xB2_f = extract_positions(result)
    _, D_B_f, *_ = collision_gradients(
        xA1_f, xA2_f, xB1_f, xB2_f, k=K_FACTOR, r=R_FACTOR
    )
    return (
        (xA1, xA2, xB1, xB2),
        (xA1_f, xA2_f, xB1_f, xB2_f),
        gA1, gA2, gB1, gB2,
        D_B_i, D_B_f,
    )


def main():
    print('=== Collision Factor Experiment ===')
    print(f'L={L} m  R_draw={R_DRAW} m  R_factor={R_FACTOR} m  '
          f'k={K_FACTOR}  cost_sigma={COST_SIGMA}  anchor_sigma={ANCHOR_POS_SIGMA} m')

    configs = build_configs()
    saved = []

    # Collect data for all individual configs
    config_data = {}
    for name, description, xA1, xA2, xB1, xB2 in configs:
        path = run_config(name, description, xA1, xA2, xB1, xB2)
        saved.append(path)
        config_data[name] = (xA1, xA2, xB1, xB2)

    # Comparison figure: parallel + oblique side by side, single legend
    print('\n── comparison: parallel + oblique ──────────────────────────────────')
    parallel_pts = config_data['parallel']
    oblique_pts  = config_data['oblique']
    parallel_data = _gather_config_data(*parallel_pts)
    oblique_data  = _gather_config_data(*oblique_pts)
    comp_path = plot_parallel_oblique_comparison(parallel_data, oblique_data)
    saved.append(comp_path)
    print(f'   Saved → {os.path.basename(comp_path)}')

    print(f'\n=== Done — {len(saved)} figures saved ===')
    for p in saved:
        print(f'   {p}')


if __name__ == '__main__':
    main()
