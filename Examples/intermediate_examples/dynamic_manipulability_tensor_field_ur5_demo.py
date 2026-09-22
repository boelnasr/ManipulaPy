#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Dynamic (Inertia-Weighted) Manipulability Tensor Field Demo (UR5) - ManipulaPy

The velocity tensor field in manipulability_tensor_field_ur5_demo.py,
A(q) = J_block(q) J_block(q)^T, treats every joint as equally easy to
accelerate. Real joints are not: a motion that looks generous purely
kinematically can still be hard to *achieve* quickly if it has to move a
lot of inertia. Yoshikawa's dynamic manipulability ellipsoid captures that
by weighting the Jacobian with the robot's mass matrix M(q) (from
ManipulaPy.dynamics.ManipulatorDynamics.mass_matrix, exposed as
``urdf_processor.dynamics`` after loading a URDF):

    Lambda(q) = ( J_block(q) @ M(q)^{-1} @ J_block(q)^T )^{-1}

Lambda is the *reflected Cartesian inertia*: how much end-effector
force/torque is needed for a unit end-effector acceleration, for
J_block = J_v (position) or J_w (orientation). It is large where the arm
looks kinematically capable but is actually dragging a lot of inertia
around to get there (e.g. shoulder-driven motions with the elbow
extended), which the purely kinematic field cannot see at all.

Usage:
    python dynamic_manipulability_tensor_field_ur5_demo.py

Expected Output:
    - Console summary of the dynamic-index range for both fields
    - "dynamic_manipulability_tensor_field_ur5.png" with four panels:
      scalar dynamic-index heatmaps and 3-D reflected-inertia ellipsoid
      tensor fields, anchored at the real end-effector positions
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
EIGENVALUE_FLOOR = 1e-4  # guards the mass-matrix and tensor inverses


def jacobian_blocks(robot, thetalist):
    """Return the linear (position) and angular (orientation) Jacobian blocks."""
    J = np.asarray(robot.jacobian(thetalist, frame="space"))
    return J[:3, :], J[3:, :]


def _floored_inverse(A, eps=EIGENVALUE_FLOOR):
    """Invert a symmetric PSD matrix after flooring its eigenvalues at ``eps``."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T


def build_dynamic_tensor_fields(robot, dynamics, q_fixed, sweep_joints, ranges, resolution=25):
    """
    Sweep two joints and build the position and orientation dynamic
    (reflected-inertia) manipulability tensor fields
    Lambda(q) = (J_block(q) M(q)^{-1} J_block(q)^T)^{-1}.

    Returns:
        dict with q1_vals, q2_vals, ee_positions (res,res,3),
        pos_tensor_field / ori_tensor_field (res,res,3,3), and
        pos_index / ori_index (res,res) dynamic-manipulability indices.
    """
    j1, j2 = sweep_joints
    q1_vals = np.linspace(ranges[0][0], ranges[0][1], resolution)
    q2_vals = np.linspace(ranges[1][0], ranges[1][1], resolution)

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

            M = np.asarray(dynamics.mass_matrix(q))
            M_inv = _floored_inverse(M)

            J_v, J_w = jacobian_blocks(robot, q)
            Lambda_pos = _floored_inverse(J_v @ M_inv @ J_v.T)
            Lambda_ori = _floored_inverse(J_w @ M_inv @ J_w.T)

            pos_tensor_field[i, j] = Lambda_pos
            ori_tensor_field[i, j] = Lambda_ori
            pos_index[i, j] = np.sqrt(max(np.linalg.det(Lambda_pos), 0.0))
            ori_index[i, j] = np.sqrt(max(np.linalg.det(Lambda_ori), 0.0))

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


def plot_dynamic_tensor_fields(field_data, sweep_labels, ellipsoid_stride=4):
    """Plot scalar dynamic-index heatmaps and 3-D reflected-inertia ellipsoid fields."""
    q1_vals, q2_vals = field_data["q1_vals"], field_data["q2_vals"]
    extent = [q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(r"UR5 Dynamic Manipulability Tensor Fields", fontsize=16, fontweight="bold")

    for pos, index_field, title, cmap in (
        (1, field_data["pos_index"],
         r"Position dynamic index $\;w_{\Lambda,\mathrm{pos}} = "
         r"\sqrt{\det\!\left((J_v M^{-1} J_v^{\mathsf{T}})^{-1}\right)}$",
         "viridis"),
        (2, field_data["ori_index"],
         r"Orientation dynamic index $\;w_{\Lambda,\mathrm{ori}} = "
         r"\sqrt{\det\!\left((J_\omega M^{-1} J_\omega^{\mathsf{T}})^{-1}\right)}$",
         "magma"),
    ):
        ax = fig.add_subplot(2, 2, pos)
        # Dynamic-manipulability indices span orders of magnitude near a
        # singularity (that blow-up is the whole point), so a linear color
        # scale would wash out everything but the single hottest cell --
        # use a log scale.
        im = ax.imshow(index_field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        norm=LogNorm(vmin=index_field.min(), vmax=index_field.max()))
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylabel(sweep_labels[1])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Dynamic (inertia) index")

    ee_positions = field_data["ee_positions"]
    for pos, tensor_field, index_field, title, cmap in (
        (3, field_data["pos_tensor_field"], field_data["pos_index"],
         r"Reflected inertia field, position $\left(\Lambda_{\mathrm{pos}}\;@\;\mathrm{EE\ position}\right)$",
         "viridis"),
        (4, field_data["ori_tensor_field"], field_data["ori_index"],
         r"Reflected inertia field, orientation $\left(\Lambda_{\mathrm{ori}}\;@\;\mathrm{EE\ position}\right)$",
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
        fig.colorbar(mappable, ax=ax, shrink=0.6, label="Dynamic (inertia) index")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "dynamic_manipulability_tensor_field_ur5.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("   ManipulaPy: Dynamic Manipulability Tensor Field Demo (UR5)")
    print("=" * 70)

    print(f"\nLoading UR5 URDF from: {urdf_file}")
    urdf_processor = URDFToSerialManipulator(urdf_file)
    robot = urdf_processor.serial_manipulator
    dynamics = urdf_processor.dynamics
    n_joints = len(np.array(robot.joint_limits))
    print(f"Loaded {n_joints}-DOF UR5 robot (with dynamics/mass matrix).")

    q_fixed = np.zeros(n_joints)
    q_fixed[3] = -np.pi / 2
    sweep_joints = (1, 2)  # shoulder_lift, elbow
    ranges = ((-np.pi + 0.05, -0.05), (-2.8, 2.8))

    print("\nBuilding dynamic (inertia-weighted) manipulability tensor fields "
          f"(sweeping joints {sweep_joints})...")
    field_data = build_dynamic_tensor_fields(
        robot, dynamics, q_fixed, sweep_joints, ranges, resolution=25
    )

    print(f"  Position dynamic index range:    [{field_data['pos_index'].min():.4f}, "
          f"{field_data['pos_index'].max():.4f}]")
    print(f"  Orientation dynamic index range: [{field_data['ori_index'].min():.4f}, "
          f"{field_data['ori_index'].max():.4f}]")

    plot_dynamic_tensor_fields(field_data, sweep_labels=("shoulder lift (rad)", "elbow (rad)"))

    print("\nDone.")


if __name__ == "__main__":
    main()
