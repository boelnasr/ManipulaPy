#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Manipulability Index Tensor Field Demo (UR5) - ManipulaPy

Builds two *tensor fields* over a 2-D slice of the UR5's joint space:

- Position (translational) field:  A_pos(q) = J_v(q) @ J_v(q).T   (3x3 SPD tensor)
- Orientation (rotational) field:   A_ori(q) = J_w(q) @ J_w(q).T   (3x3 SPD tensor)

``J_v``/``J_w`` are the linear/angular blocks of the space-frame Jacobian
(``SerialManipulator.jacobian(thetalist, frame="space")``, rows 0:3 and
3:6 respectively -- the same split ``Singularity.manipulability_ellipsoid``
uses). At every grid point the field stores the full 3x3 tensor plus its
scalar Yoshikawa manipulability index:

    w(q) = sqrt(det(A(q)))

The scalar indices are rendered as heatmaps; the tensors themselves are
drawn as ellipse glyphs (their eigenvector/eigenvalue structure projected
onto the swept plane) so the field's *directionality* -- not just its
magnitude -- is visible, which is what makes it a tensor field rather than
a scalar one.

Usage:
    python manipulability_tensor_field_ur5_demo.py

Expected Output:
    - Console summary of the index range for both fields
    - "manipulability_tensor_field_ur5.png" with four panels: the two
      scalar-index heatmaps and the two tensor-glyph fields
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    J_v = J[:3, :]  # linear velocity block -> position manipulability
    J_w = J[3:, :]  # angular velocity block -> orientation manipulability
    return J_v, J_w


def build_index_tensor_fields(robot, q_fixed, sweep_joints, ranges, resolution=25):
    """
    Sweep two joints of ``q_fixed`` over ``ranges`` and build the position and
    orientation manipulability index tensor fields.

    Parameters:
        robot: SerialManipulator instance (already loaded with a URDF).
        q_fixed (np.ndarray): Reference configuration; all joints not swept
            stay fixed at these values.
        sweep_joints (tuple[int, int]): Indices of the two joints to sweep.
        ranges (tuple[tuple, tuple]): ((min, max), (min, max)) for the two
            swept joints, in radians.
        resolution (int): Number of samples per axis.

    Returns:
        dict with:
            q1_vals, q2_vals: 1D sweep axes (rad)
            pos_tensor_field, ori_tensor_field: (res, res, 3, 3) SPD tensors
            pos_index, ori_index: (res, res) scalar manipulability indices
    """
    j1, j2 = sweep_joints
    q1_vals = np.linspace(ranges[0][0], ranges[0][1], resolution)
    q2_vals = np.linspace(ranges[1][0], ranges[1][1], resolution)

    pos_tensor_field = np.zeros((resolution, resolution, 3, 3))
    ori_tensor_field = np.zeros((resolution, resolution, 3, 3))
    pos_index = np.zeros((resolution, resolution))
    ori_index = np.zeros((resolution, resolution))

    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            q = q_fixed.copy()
            q[j1] = q1
            q[j2] = q2

            J_v, J_w = jacobian_blocks(robot, q)
            A_pos = J_v @ J_v.T
            A_ori = J_w @ J_w.T

            pos_tensor_field[i, j] = A_pos
            ori_tensor_field[i, j] = A_ori
            pos_index[i, j] = np.sqrt(max(np.linalg.det(A_pos), 0.0))
            ori_index[i, j] = np.sqrt(max(np.linalg.det(A_ori), 0.0))

    return {
        "q1_vals": q1_vals,
        "q2_vals": q2_vals,
        "pos_tensor_field": pos_tensor_field,
        "ori_tensor_field": ori_tensor_field,
        "pos_index": pos_index,
        "ori_index": ori_index,
    }


def _tensor_glyph(ax, center, tensor_2d, scale, color):
    """Draw one tensor's 2x2 in-plane block as an ellipse glyph at ``center``."""
    eigvals, eigvecs = np.linalg.eigh(tensor_2d)
    eigvals = np.maximum(eigvals, 0.0)
    width, height = 2.0 * scale * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    ax.add_patch(
        Ellipse(
            center, width=width, height=height, angle=angle,
            facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.6,
        )
    )


def plot_index_tensor_fields(field_data, sweep_labels, glyph_stride=3):
    """Plot scalar-index heatmaps and tensor-glyph fields for position/orientation."""
    q1_vals, q2_vals = field_data["q1_vals"], field_data["q2_vals"]
    extent = [q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("UR5 Manipulability Index Tensor Fields", fontsize=15, fontweight="bold")

    # --- Scalar index heatmaps --------------------------------------------------
    for ax, index_field, title, cmap in (
        (axes[0, 0], field_data["pos_index"], "Position index  w_pos = sqrt(det(J_v J_v^T))", "viridis"),
        (axes[0, 1], field_data["ori_index"], "Orientation index  w_ori = sqrt(det(J_w J_w^T))", "magma"),
    ):
        im = ax.imshow(
            index_field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap,
        )
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylabel(sweep_labels[1])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Manipulability index")

    # --- Tensor glyph fields -----------------------------------------------------
    for ax, tensor_field, index_field, title, cmap in (
        (axes[1, 0], field_data["pos_tensor_field"], field_data["pos_index"],
         "Position tensor field (ellipse = J_v J_v^T)", "viridis"),
        (axes[1, 1], field_data["ori_tensor_field"], field_data["ori_index"],
         "Orientation tensor field (ellipse = J_w J_w^T)", "magma"),
    ):
        norm = plt.Normalize(vmin=np.min(index_field), vmax=np.max(index_field) + 1e-12)
        colormap = plt.get_cmap(cmap)

        # Glyph size in data units, scaled relative to the sweep span so
        # ellipses stay legible regardless of the raw tensor magnitude.
        span = min(q1_vals[-1] - q1_vals[0], q2_vals[-1] - q2_vals[0])
        max_axis = np.sqrt(np.max(np.linalg.eigvalsh(tensor_field)))
        glyph_scale = 0.4 * span / (len(q1_vals) / glyph_stride) / max(max_axis, 1e-9)

        for i in range(0, len(q1_vals), glyph_stride):
            for j in range(0, len(q2_vals), glyph_stride):
                tensor_2d = tensor_field[i, j][:2, :2]  # in-plane block of the 3x3 tensor
                color = colormap(norm(index_field[i, j]))
                _tensor_glyph(ax, (q1_vals[i], q2_vals[j]), tensor_2d, glyph_scale, color)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylabel(sweep_labels[1])
        ax.set_title(title, fontweight="bold")
        ax.set_facecolor("#f5f5f5")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "manipulability_tensor_field_ur5.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("   ManipulaPy: Manipulability Index Tensor Field Demo (UR5)")
    print("=" * 70)

    print(f"\nLoading UR5 URDF from: {urdf_file}")
    urdf_processor = URDFToSerialManipulator(urdf_file)
    robot = urdf_processor.serial_manipulator
    n_joints = len(np.array(robot.joint_limits))
    print(f"Loaded {n_joints}-DOF UR5 robot.")

    # UR5 joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    # Sweep shoulder_lift (1) and elbow (2): the two joints that dominate reach
    # and are most prone to the classic UR5 elbow/shoulder singularities.
    q_fixed = np.zeros(n_joints)
    q_fixed[3] = -np.pi / 2  # keep the wrist in a generic, non-degenerate pose

    sweep_joints = (1, 2)
    ranges = ((-np.pi + 0.05, -0.05), (-2.8, 2.8))

    print("\nBuilding position and orientation index tensor fields "
          f"(sweeping joints {sweep_joints})...")
    field_data = build_index_tensor_fields(
        robot, q_fixed, sweep_joints, ranges, resolution=25
    )

    print(f"  Position index range:    [{field_data['pos_index'].min():.4f}, "
          f"{field_data['pos_index'].max():.4f}]")
    print(f"  Orientation index range: [{field_data['ori_index'].min():.4f}, "
          f"{field_data['ori_index'].max():.4f}]")

    plot_index_tensor_fields(
        field_data,
        sweep_labels=("shoulder_lift (rad)", "elbow (rad)"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
