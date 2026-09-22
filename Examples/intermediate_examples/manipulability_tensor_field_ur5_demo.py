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
rendered as true 3-D ellipsoid surfaces, each placed at the robot's actual
end-effector position (from forward kinematics) for that grid point, with
the ellipsoid's own axes given by the eigenvectors/eigenvalues of the 3x3
tensor. That is what makes this a *tensor field* over the workspace rather
than a scalar one: at every sampled point in space it carries the full
directional (not just magnitude) velocity-capability information.

Usage:
    python manipulability_tensor_field_ur5_demo.py

Expected Output:
    - Console summary of the index range for both fields
    - "manipulability_tensor_field_ur5.png" with four panels: the two
      scalar-index heatmaps (top) and the two 3-D ellipsoid tensor fields,
      anchored at the real end-effector positions (bottom)
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
            ee_positions: (res, res, 3) end-effector positions (forward kinematics)
            pos_tensor_field, ori_tensor_field: (res, res, 3, 3) SPD tensors
            pos_index, ori_index: (res, res) scalar manipulability indices
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
        "ee_positions": ee_positions,
        "pos_tensor_field": pos_tensor_field,
        "ori_tensor_field": ori_tensor_field,
        "pos_index": pos_index,
        "ori_index": ori_index,
    }


def _ellipsoid_surface(center, tensor, scale, sphere_res=14):
    """
    Build the (X, Y, Z) mesh of the 3-D ellipsoid representing ``tensor``,
    scaled by ``scale`` and translated to ``center``.

    The ellipsoid's semi-axes are the eigenvectors of ``tensor``, scaled by
    the square roots of its eigenvalues -- the standard velocity-ellipsoid
    construction (``compute_ellipsoid_data`` in the singularity-analysis
    intermediate demo uses the same eigendecomposition).
    """
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
                          stride=5, sphere_res=16, fill_frac=0.8):
    """Draw one field's tensors as 3-D ellipsoid surfaces at their real EE positions.

    Each ellipsoid's own *shape* (its axis ratios) is the true tensor
    anisotropy -- flat/needle-like exactly where the manipulator is close to
    singular. Its overall size is normalized per-point to a fixed target
    radius (rather than scaled by the raw, widely-varying eigenvalues) so
    neighboring glyphs stay legible; the scalar magnitude that size would
    otherwise have encoded is instead carried by color.
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
            X, Y, Z = _ellipsoid_surface(
                centers[i, j], tensor, scale, sphere_res=sphere_res
            )
            color = colormap(norm(index_field[i, j]))
            ax.plot_surface(X, Y, Z, color=color, alpha=0.85, linewidth=0, shade=True)

    # Faint end-effector grid so the ellipsoids read as a field over the workspace.
    ax.plot_wireframe(
        centers[..., 0], centers[..., 1], centers[..., 2],
        color="gray", linewidth=0.3, alpha=0.3,
    )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array([])
    return mappable


def plot_index_tensor_fields(field_data, sweep_labels, ellipsoid_stride=4):
    """Plot scalar-index heatmaps and 3-D ellipsoid tensor fields for position/orientation."""
    q1_vals, q2_vals = field_data["q1_vals"], field_data["q2_vals"]
    extent = [q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("UR5 Manipulability Index Tensor Fields", fontsize=15, fontweight="bold")

    # --- Scalar index heatmaps (joint-space slice) -------------------------------
    for pos, index_field, title, cmap in (
        (1, field_data["pos_index"], "Position index  w_pos = sqrt(det(J_v J_v^T))", "viridis"),
        (2, field_data["ori_index"], "Orientation index  w_ori = sqrt(det(J_w J_w^T))", "magma"),
    ):
        ax = fig.add_subplot(2, 2, pos)
        im = ax.imshow(
            index_field.T, origin="lower", extent=extent, aspect="auto", cmap=cmap,
        )
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylabel(sweep_labels[1])
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Manipulability index")

    # --- 3-D ellipsoid tensor fields, anchored at the real end-effector position -
    ee_positions = field_data["ee_positions"]
    for pos, tensor_field, index_field, title, cmap in (
        (3, field_data["pos_tensor_field"], field_data["pos_index"],
         "Position tensor field (ellipsoid = J_v J_v^T @ EE position)", "viridis"),
        (4, field_data["ori_tensor_field"], field_data["ori_index"],
         "Orientation tensor field (ellipsoid = J_w J_w^T @ EE position)", "magma"),
    ):
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        mappable = plot_3d_tensor_field(
            ax, ee_positions, tensor_field, index_field, cmap, stride=ellipsoid_stride,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(mappable, ax=ax, shrink=0.6, label="Manipulability index")

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
