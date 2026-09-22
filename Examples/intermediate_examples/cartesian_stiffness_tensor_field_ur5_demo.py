#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cartesian Stiffness Tensor Field Demo (UR5) - ManipulaPy

A third tensor field derived from the same Jacobian, this time reflecting a
chosen *joint*-space stiffness into Cartesian space -- the quantity a
compliant/impedance controller actually cares about. For a diagonal joint
stiffness K_joint = diag(k, ..., k) and the UR5's square (6-DOF, 6x6)
Jacobian J(q), the reflected Cartesian stiffness is

    K_cart(q) = J(q)^{-T} @ K_joint @ J(q)^{-1}     (6x6, SPD)

i.e. "how much end-effector force/torque a small end-effector displacement
generates, for fixed joint-level compliance." This demo renders its
position block K_cart[:3, :3] and orientation block K_cart[3:, 3:] as
tensor fields; the two blocks are coupled by the full 6x6 matrix's
off-diagonal terms, which the two separate ellipsoid fields do not show
(the same simplification manipulability_tensor_field_ur5_demo.py makes for
the velocity tensor).

Like the force/torque ellipsoid field (force_ellipsoid_tensor_field_ur5_demo.py),
K_cart blows up near a kinematic singularity (J^{-1} diverges); this demo
floors J's singular values before inverting so the field stays finite while
still growing sharply near-singular, which is the physically correct trend
(a locked-up axis reflects joint stiffness as if it were infinitely stiff).

Usage:
    python cartesian_stiffness_tensor_field_ur5_demo.py

Expected Output:
    - Console summary of the stiffness-index range for both blocks
    - "cartesian_stiffness_tensor_field_ur5.png" with four panels: scalar
      stiffness-index heatmaps and 3-D stiffness-ellipsoid tensor fields,
      anchored at the real end-effector positions
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
SINGULAR_VALUE_FLOOR = 0.05  # guards J's inverse at near-singular configurations
JOINT_STIFFNESS = 200.0      # N*m/rad, uniform across joints for this demo


def _damped_inverse(J, eps=SINGULAR_VALUE_FLOOR):
    """Invert a square Jacobian after flooring its singular values at ``eps``."""
    U, S, Vt = np.linalg.svd(J)
    S_inv = 1.0 / np.maximum(S, eps)
    return (Vt.T * S_inv) @ U.T


def build_cartesian_stiffness_field(robot, q_fixed, sweep_joints, ranges,
                                     joint_stiffness=JOINT_STIFFNESS, resolution=25):
    """
    Sweep two joints and build the position/orientation blocks of the
    reflected Cartesian stiffness tensor field K_cart(q) = J^{-T} K_joint J^{-1}.

    Returns:
        dict with q1_vals, q2_vals, ee_positions (res,res,3),
        pos_tensor_field / ori_tensor_field (res,res,3,3), and
        pos_index / ori_index (res,res) stiffness-manipulability indices.
    """
    j1, j2 = sweep_joints
    q1_vals = np.linspace(ranges[0][0], ranges[0][1], resolution)
    q2_vals = np.linspace(ranges[1][0], ranges[1][1], resolution)
    n_joints = len(q_fixed)
    K_joint = joint_stiffness * np.eye(n_joints)

    ee_positions = np.zeros((resolution, resolution, 3))
    pos_tensor_field = np.zeros((resolution, resolution, 3, 3))
    ori_tensor_field = np.zeros((resolution, resolution, 3, 3))
    pos_index = np.zeros((resolution, resolution))
    ori_index = np.zeros((resolution, resolution))

    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            q = q_fixed.copy()
            q[j1] = q1
            q[j2] = q2

            ee_positions[i, j] = np.asarray(robot.forward_kinematics(q))[:3, 3]

            J = np.asarray(robot.jacobian(q, frame="space"))  # (6, 6) for the UR5
            J_inv = _damped_inverse(J)
            K_cart = J_inv.T @ K_joint @ J_inv

            K_pos = K_cart[:3, :3]
            K_ori = K_cart[3:, 3:]
            pos_tensor_field[i, j] = K_pos
            ori_tensor_field[i, j] = K_ori
            pos_index[i, j] = np.sqrt(max(np.linalg.det(K_pos), 0.0))
            ori_index[i, j] = np.sqrt(max(np.linalg.det(K_ori), 0.0))

    return {
        "q1_vals": q1_vals,
        "q2_vals": q2_vals,
        "ee_positions": ee_positions,
        "pos_tensor_field": pos_tensor_field,
        "ori_tensor_field": ori_tensor_field,
        "pos_index": pos_index,
        "ori_index": ori_index,
    }


def _ellipsoid_surface(center, tensor, scale, sphere_res=14):
    """Build the (X, Y, Z) mesh of the 3-D ellipsoid representing ``tensor``."""
    eigvals, eigvecs = np.linalg.eigh(tensor)
    eigvals = np.maximum(eigvals, 0.0)
    radii = scale * np.sqrt(eigvals)

    u = np.linspace(0, 2 * np.pi, sphere_res)
    v = np.linspace(0, np.pi, sphere_res)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    points = eigvecs @ np.stack([x.ravel(), y.ravel(), z.ravel()])
    points += np.asarray(center).reshape(3, 1)
    shape = x.shape
    return points[0].reshape(shape), points[1].reshape(shape), points[2].reshape(shape)


def _grid_spacing(centers):
    """Mean nearest-neighbor distance between adjacent grid points, for auto-scaling."""
    d_rows = np.linalg.norm(np.diff(centers, axis=0), axis=-1).mean()
    d_cols = np.linalg.norm(np.diff(centers, axis=1), axis=-1).mean()
    return min(d_rows, d_cols)


def plot_3d_tensor_field(ax, centers, tensor_field, index_field, cmap_name,
                          stride=4, sphere_res=14, fill_frac=0.8):
    """Draw one field's tensors as 3-D ellipsoid surfaces at their real EE positions.

    Each ellipsoid's own shape (its axis ratios) is the true tensor
    anisotropy. Size is normalized per-point to a fixed target radius so
    neighboring glyphs stay legible regardless of the field's raw dynamic
    range; magnitude is instead carried by color.
    """
    colormap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=np.min(index_field), vmax=np.max(index_field) + 1e-12)

    sampled_centers = centers[::stride, ::stride]
    target_radius = 0.5 * fill_frac * _grid_spacing(sampled_centers)

    resolution = centers.shape[0]
    for i in range(0, resolution, stride):
        for j in range(0, resolution, stride):
            tensor = tensor_field[i, j]
            max_eigval = max(np.linalg.eigvalsh(tensor).max(), 1e-12)
            scale = target_radius / np.sqrt(max_eigval)
            X, Y, Z = _ellipsoid_surface(centers[i, j], tensor, scale, sphere_res=sphere_res)
            color = colormap(norm(index_field[i, j]))
            ax.plot_surface(X, Y, Z, color=color, alpha=0.85, linewidth=0, shade=True)

    ax.plot_wireframe(
        centers[..., 0], centers[..., 1], centers[..., 2],
        color="gray", linewidth=0.3, alpha=0.3,
    )

    # A near-degenerate axis (e.g. this sweep's fixed shoulder_pan keeps Y
    # almost constant) would otherwise make matplotlib auto-zoom to the
    # ellipsoids' own tiny extent along it; floor every axis's shown span so
    # the box stays legibly proportioned instead of looking clipped/broken.
    centers_flat = centers.reshape(-1, 3)
    mins, maxs = centers_flat.min(axis=0), centers_flat.max(axis=0)
    mids = (mins + maxs) / 2
    half_spans = np.maximum((maxs - mins) / 2, 2 * target_radius)
    ax.set_xlim(mids[0] - half_spans[0], mids[0] + half_spans[0])
    ax.set_ylim(mids[1] - half_spans[1], mids[1] + half_spans[1])
    ax.set_zlim(mids[2] - half_spans[2], mids[2] + half_spans[2])

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    return mappable


def plot_stiffness_tensor_fields(field_data, sweep_labels, ellipsoid_stride=4):
    """Plot scalar stiffness-index heatmaps and 3-D stiffness-ellipsoid tensor fields."""
    q1_vals, q2_vals = field_data["q1_vals"], field_data["q2_vals"]
    extent = [q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(r"UR5 Cartesian Stiffness Tensor Fields", fontsize=16, fontweight="bold")

    for pos, index_field, title, cmap in (
        (1, field_data["pos_index"],
         r"Translational stiffness index $\;w_{K,\mathrm{pos}} = \sqrt{\det(K_{\mathrm{cart}}[:3,:3])}$",
         "viridis"),
        (2, field_data["ori_index"],
         r"Rotational stiffness index $\;w_{K,\mathrm{ori}} = \sqrt{\det(K_{\mathrm{cart}}[3{:},3{:}])}$",
         "magma"),
    ):
        ax = fig.add_subplot(2, 2, pos)
        # Stiffness indices span orders of magnitude near a singularity
        # (that blow-up is the whole point), so a linear color scale would
        # wash out everything but the single hottest cell -- use a log scale.
        im = ax.imshow(index_field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        norm=LogNorm(vmin=index_field.min(), vmax=index_field.max()))
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylabel(sweep_labels[1])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Stiffness index")

    ee_positions = field_data["ee_positions"]
    for pos, tensor_field, index_field, title, cmap in (
        (3, field_data["pos_tensor_field"], field_data["pos_index"],
         r"Translational stiffness field $\left(K_{\mathrm{cart}}[:3,:3]\;@\;\mathrm{EE\ position}\right)$",
         "viridis"),
        (4, field_data["ori_tensor_field"], field_data["ori_index"],
         r"Rotational stiffness field $\left(K_{\mathrm{cart}}[3{:},3{:}]\;@\;\mathrm{EE\ position}\right)$",
         "magma"),
    ):
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        mappable = plot_3d_tensor_field(
            ax, ee_positions, tensor_field, index_field, cmap, stride=ellipsoid_stride,
        )
        ax.set_xlabel(r"$X$ (m)")
        ax.set_ylabel(r"$Y$ (m)")
        ax.set_zlabel(r"$Z$ (m)")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(mappable, ax=ax, shrink=0.6, label="Stiffness index")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "cartesian_stiffness_tensor_field_ur5.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("   ManipulaPy: Cartesian Stiffness Tensor Field Demo (UR5)")
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

    print(f"\nBuilding Cartesian stiffness tensor field (joint stiffness = "
          f"{JOINT_STIFFNESS} N*m/rad, sweeping joints {sweep_joints})...")
    field_data = build_cartesian_stiffness_field(robot, q_fixed, sweep_joints, ranges, resolution=25)

    print(f"  Translational stiffness index range: [{field_data['pos_index'].min():.2f}, "
          f"{field_data['pos_index'].max():.2f}]")
    print(f"  Rotational stiffness index range:    [{field_data['ori_index'].min():.2f}, "
          f"{field_data['ori_index'].max():.2f}]")

    plot_stiffness_tensor_fields(field_data, sweep_labels=("shoulder lift (rad)", "elbow (rad)"))

    print("\nDone.")


if __name__ == "__main__":
    main()
