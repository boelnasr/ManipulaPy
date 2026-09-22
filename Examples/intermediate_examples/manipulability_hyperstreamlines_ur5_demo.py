#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Manipulability Tensor Field Hyperstreamlines Demo (UR5) - ManipulaPy

A classic tensor-field visualization technique: instead of one ellipsoid
glyph per sample, integrate curves ("hyperstreamlines") that follow the
*major eigenvector* of a tensor field across a 2-D domain. An eigenvector
is only defined up to sign -- it is a *line field*, not a vector field --
so streamlines seeded around the domain boundary have no consistent
"downhill" direction to agree on near a degenerate point (where the
tensor becomes locally isotropic and every direction is equally
principal). They bend around and pile up there instead, producing the
familiar swirling, hair-like bundles seen in tensor-field-topology art
and diffusion-tensor tractography renders.

The tensor field visualized here is the joint-space pullback metric

    M(q) = J_block(q)^T @ J_block(q)     (restricted to the 2 swept joints)

for J_block = J_v (position) or J_w (orientation) -- i.e. "how much
end-effector position/orientation velocity a small motion of these two
joints produces." Its degenerate points sit inside the same near-singular
bands visible in a scalar manipulability-index heatmap
(manipulability_tensor_field_ur5_demo.py in this directory); this demo
makes that same structure visible as flowing lines instead of a color
gradient. Each streamline is then lifted from joint space into the real
UR5 workspace via forward kinematics, so the picture is the tensor field
of an actual robot, not an abstract domain.

Usage:
    python manipulability_hyperstreamlines_ur5_demo.py

Expected Output:
    - Console progress while streamlines are integrated
    - "manipulability_hyperstreamlines_ur5.png": position and orientation
      hyperstreamline bundles rendered in the real UR5 workspace
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# LaTeX-style (Computer Modern) text via matplotlib's bundled cmr10 font, so
# the figure renders the same everywhere without requiring a system LaTeX install.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

try:
    import ManipulaPy
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import ManipulaPy

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.ur5 import urdf_file

OUTPUT_DIR = Path(__file__).resolve().parent


def jacobian_blocks(robot, thetalist):
    """Return the linear (position) and angular (orientation) Jacobian blocks."""
    J = np.asarray(robot.jacobian(thetalist, frame="space"))
    return J[:3, :], J[3:, :]


def build_joint_tensor_field(robot, q_fixed, sweep_joints, ranges, resolution=90):
    """
    Sweep two joints and build the *joint-space* manipulability tensor field
    M(q) = J_block(q)^T @ J_block(q), restricted to the 2x2 block of the
    swept joints, for both the position and orientation Jacobian blocks.

    Unlike the Cartesian tensor field J_block @ J_block^T (3x3, used in
    manipulability_tensor_field_ur5_demo.py), this pullback metric lives
    on the same 2-D domain being swept, which is what a hyperstreamline /
    tensor-field-topology picture needs.

    Returns:
        dict with q1_vals, q2_vals, ee_positions (res,res,3), and
        pos_tensor / ori_tensor (res,res,2,2).
    """
    j1, j2 = sweep_joints
    q1_vals = np.linspace(ranges[0][0], ranges[0][1], resolution)
    q2_vals = np.linspace(ranges[1][0], ranges[1][1], resolution)

    ee_positions = np.zeros((resolution, resolution, 3))
    pos_tensor = np.zeros((resolution, resolution, 2, 2))
    ori_tensor = np.zeros((resolution, resolution, 2, 2))

    idx = np.ix_(sweep_joints, sweep_joints)
    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            q = q_fixed.copy()
            q[j1] = q1
            q[j2] = q2

            ee_positions[i, j] = np.asarray(robot.forward_kinematics(q))[:3, 3]

            J_v, J_w = jacobian_blocks(robot, q)
            pos_tensor[i, j] = (J_v.T @ J_v)[idx]
            ori_tensor[i, j] = (J_w.T @ J_w)[idx]

    return {
        "q1_vals": q1_vals,
        "q2_vals": q2_vals,
        "ee_positions": ee_positions,
        "pos_tensor": pos_tensor,
        "ori_tensor": ori_tensor,
    }


def _make_bilinear_sampler(q1_vals, q2_vals, values):
    """Fast manual bilinear interpolator for a (res, res, k) field on a uniform grid."""
    x0, x1 = q1_vals[0], q1_vals[-1]
    y0, y1 = q2_vals[0], q2_vals[-1]
    n1, n2 = len(q1_vals), len(q2_vals)

    def sample(px, py):
        fx = (px - x0) / (x1 - x0) * (n1 - 1)
        fy = (py - y0) / (y1 - y0) * (n2 - 1)
        i0 = int(np.clip(np.floor(fx), 0, n1 - 2))
        j0 = int(np.clip(np.floor(fy), 0, n2 - 2))
        tx = min(max(fx - i0, 0.0), 1.0)
        ty = min(max(fy - j0, 0.0), 1.0)
        v00, v10 = values[i0, j0], values[i0 + 1, j0]
        v01, v11 = values[i0, j0 + 1], values[i0 + 1, j0 + 1]
        return (
            v00 * (1 - tx) * (1 - ty)
            + v10 * tx * (1 - ty)
            + v01 * (1 - tx) * ty
            + v11 * tx * ty
        )

    return sample


def _major_eigvec_2x2(a, b, c):
    """Unit major eigenvector (direction only, sign ambiguous) of [[a, b], [b, c]]."""
    theta = 0.5 * np.arctan2(2.0 * b, a - c)
    return np.array([np.cos(theta), np.sin(theta)])


def _integrate_half_streamline(sampler, bounds, start, initial_dir, step, max_steps):
    """Integrate the major-eigenvector line field from ``start`` in ``initial_dir``."""
    (x0, x1), (y0, y1) = bounds
    pos = np.array(start, dtype=float)
    prev_dir = np.asarray(initial_dir, dtype=float)
    pts = [pos.copy()]
    for _ in range(max_steps):
        a, b, c = sampler(pos[0], pos[1])
        direction = _major_eigvec_2x2(a, b, c)
        if np.dot(direction, prev_dir) < 0.0:  # resolve the eigenvector's sign ambiguity
            direction = -direction
        pos = pos + step * direction
        if not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1):
            break
        pts.append(pos.copy())
        prev_dir = direction
    return pts


def integrate_streamline(sampler, bounds, seed, step, max_steps):
    """Integrate one full hyperstreamline through ``seed``, both ways along the line field."""
    a, b, c = sampler(seed[0], seed[1])
    v = _major_eigvec_2x2(a, b, c)
    forward = _integrate_half_streamline(sampler, bounds, seed, v, step, max_steps)
    backward = _integrate_half_streamline(sampler, bounds, seed, -v, step, max_steps)
    return np.array(list(reversed(backward)) + forward[1:])


def _ring_seeds(bounds, n_seeds, shrink=1.0):
    """Evenly spaced seed points around a rectangle inset ``shrink`` toward the domain center.

    ``shrink=1.0`` is the domain's own perimeter; smaller values give concentric
    inner rings, which fill in the streamline-free gap that a boundary-only
    seeding leaves around a degenerate point (the field has no direction to
    integrate *at* the degenerate point itself, so nothing seeded further out
    ever quite reaches it).
    """
    (x0, x1), (y0, y1) = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 * shrink, (y1 - y0) / 2 * shrink
    rx0, rx1, ry0, ry1 = cx - hx, cx + hx, cy - hy, cy + hy
    width, height = rx1 - rx0, ry1 - ry0
    perimeter = 2 * (width + height)
    seeds = []
    for k in range(n_seeds):
        s = (k / n_seeds) * perimeter
        if s < width:
            seeds.append((rx0 + s, ry0))
        elif s < width + height:
            seeds.append((rx1, ry0 + (s - width)))
        elif s < 2 * width + height:
            seeds.append((rx1 - (s - width - height), ry1))
        else:
            seeds.append((rx0, ry1 - (s - 2 * width - height)))
    return seeds


def build_hyperstreamlines(field_data, tensor_key, sweep_joints, n_seeds=90,
                            rings=(1.0, 0.75, 0.5, 0.3, 0.12),
                            steps_per_unit=260, max_steps=500):
    """Integrate a bundle of hyperstreamlines and lift each into the real UR5 workspace."""
    q1_vals, q2_vals = field_data["q1_vals"], field_data["q2_vals"]
    bounds = ((q1_vals[0], q1_vals[-1]), (q2_vals[0], q2_vals[-1]))
    span = min(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0])
    step = span / steps_per_unit

    tensor_sampler = _make_bilinear_sampler(
        q1_vals, q2_vals,
        np.stack(
            [
                field_data[tensor_key][:, :, 0, 0],
                field_data[tensor_key][:, :, 0, 1],
                field_data[tensor_key][:, :, 1, 1],
            ],
            axis=-1,
        ),
    )
    ee_sampler = _make_bilinear_sampler(q1_vals, q2_vals, field_data["ee_positions"])

    seeds = [seed for shrink in rings for seed in _ring_seeds(bounds, n_seeds, shrink)]

    streamlines_3d = []
    for seed in seeds:
        path_2d = integrate_streamline(tensor_sampler, bounds, seed, step, max_steps)
        if len(path_2d) < 2:
            continue
        path_3d = np.array([ee_sampler(p[0], p[1]) for p in path_2d])
        streamlines_3d.append(path_3d)
    return streamlines_3d


def plot_hyperstreamlines(ax, streamlines_3d, base_color, title):
    """Render a bundle of 3-D hyperstreamlines with a warm, translucent look."""
    import matplotlib.colors as mcolors

    base_rgb = np.array(mcolors.to_rgb(base_color))
    rng = np.random.default_rng(0)
    for path in streamlines_3d:
        jitter = rng.uniform(-0.06, 0.06, size=3)
        color = np.clip(base_rgb + jitter, 0.0, 1.0)
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color=color, linewidth=0.7, alpha=0.4, solid_capstyle="round",
        )

    ax.set_title(title, fontweight="bold")
    ax.set_axis_off()
    ax.set_facecolor("white")

    all_pts = np.concatenate(streamlines_3d, axis=0)
    center = all_pts.mean(axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-60)


def main() -> None:
    print("=" * 70)
    print("   ManipulaPy: Manipulability Hyperstreamlines Demo (UR5)")
    print("=" * 70)

    print(f"\nLoading UR5 URDF from: {urdf_file}")
    urdf_processor = URDFToSerialManipulator(urdf_file)
    robot = urdf_processor.serial_manipulator
    n_joints = len(np.array(robot.joint_limits))
    print(f"Loaded {n_joints}-DOF UR5 robot.")

    q_fixed = np.zeros(n_joints)
    q_fixed[3] = -np.pi / 2
    sweep_joints = (1, 2)  # shoulder_lift, elbow
    ranges = ((-np.pi + 0.05, -0.05), (-2.8, 2.8))

    print(f"\nBuilding joint-space tensor field (sweeping joints {sweep_joints})...")
    field_data = build_joint_tensor_field(
        robot, q_fixed, sweep_joints, ranges, resolution=90
    )

    fig = plt.figure(figsize=(14, 7.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(r"UR5 Manipulability Tensor Field --- Hyperstreamlines", fontsize=16, fontweight="bold")

    for pos, tensor_key, color, title in (
        (1, "pos_tensor", "#c9a227",
         r"Position tensor field $\;M_{\mathrm{pos}} = J_v^{\mathsf{T}} J_v$"),
        (2, "ori_tensor", "#b5651d",
         r"Orientation tensor field $\;M_{\mathrm{ori}} = J_\omega^{\mathsf{T}} J_\omega$"),
    ):
        print(f"Integrating hyperstreamlines for: {title}")
        streamlines_3d = build_hyperstreamlines(field_data, tensor_key, sweep_joints)
        print(f"  {len(streamlines_3d)} streamlines")

        ax = fig.add_subplot(1, 2, pos, projection="3d")
        plot_hyperstreamlines(ax, streamlines_3d, color, title)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "manipulability_hyperstreamlines_ur5.png"
    plt.savefig(save_path, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {save_path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
