#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Visualization Demo: Robot Analysis and Plotting

This example demonstrates ManipulaPy's analysis-and-visualization capabilities on a
headless (matplotlib ``Agg``) backend: 3D workspace clouds, joint/end-effector
trajectory plots, manipulability and condition-number analysis, manipulability
ellipsoids, and configuration-space maps. Every figure is rendered with matplotlib
and saved to a ``plots/`` directory next to this script, so the demo runs anywhere
without a display or a GPU.

Usage:
    python visualization_basic_demo.py

Expected Output:
    - Console summary of workspace bounds, manipulability and singularity metrics
    - PNG figures saved under ./plots/

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib

# Honor a pre-set non-interactive backend (e.g. MPLBACKEND=Agg for headless runs);
# otherwise pick Agg so the demo saves figures to disk without needing a display.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.singularity import Singularity
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

    logger.info("ManipulaPy modules imported successfully")
except ImportError as e:  # pragma: no cover - import guard
    logger.error(f"Failed to import ManipulaPy: {e}")
    sys.exit(1)

# Directory next to this script for saved plots; keep the demo's outputs self-contained.
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Deterministic sampling so the reported numbers are reproducible run-to-run.
np.random.seed(0)


def yoshikawa_measure(J: np.ndarray) -> float:
    """
    Yoshikawa manipulability measure for a (possibly non-square) Jacobian.

    Uses the smaller of the two Gram matrices so the determinant stays well-defined:
    ``sqrt(det(J Jᵀ))`` for wide/square Jacobians and ``sqrt(det(Jᵀ J))`` for tall ones
    (e.g. the 6x4 Jacobian of this 4-DOF arm, where ``J Jᵀ`` is rank-deficient).

    Args:
        J: Geometric Jacobian, shape (6, n).

    Returns:
        float: Non-negative manipulability measure.
    """
    gram = J @ J.T if J.shape[0] <= J.shape[1] else J.T @ J
    return float(np.sqrt(max(0.0, float(np.linalg.det(gram)))))


def create_visualization_urdf() -> str:
    """
    Write a small 4-DOF robot URDF tailored for visualization demos.

    Returns:
        str: Absolute path to the temporary URDF file.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="visualization_demo_robot">

  <link name="base_link">
    <visual>
      <geometry><cylinder radius="0.08" length="0.1"/></geometry>
      <material name="dark_gray"><color rgba="0.3 0.3 0.3 1"/></material>
    </visual>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="base_joint" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="150" velocity="2.0"/>
  </joint>

  <link name="link_1">
    <visual>
      <geometry><cylinder radius="0.05" length="0.4"/></geometry>
      <material name="red"><color rgba="0.8 0.1 0.1 1"/></material>
    </visual>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.04" iyy="0.04" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="shoulder_joint" type="revolute">
    <parent link="link_1"/><child link="link_2"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0944" upper="2.0944" effort="100" velocity="2.0"/>
  </joint>

  <link name="link_2">
    <visual>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <material name="green"><color rgba="0.1 0.8 0.1 1"/></material>
    </visual>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.015" iyy="0.015" izz="0.005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="elbow_joint" type="revolute">
    <parent link="link_2"/><child link="link_3"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.618" upper="2.618" effort="80" velocity="2.0"/>
  </joint>

  <link name="link_3">
    <visual>
      <geometry><cylinder radius="0.035" length="0.25"/></geometry>
      <material name="blue"><color rgba="0.1 0.1 0.8 1"/></material>
    </visual>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.003" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="wrist_joint" type="revolute">
    <parent link="link_3"/><child link="end_effector"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="50" velocity="3.0"/>
  </joint>

  <link name="end_effector">
    <visual>
      <geometry><cylinder radius="0.025" length="0.1"/></geometry>
      <material name="yellow"><color rgba="1 1 0 1"/></material>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" iyy="0.002" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>"""

    urdf_file = str(Path(__file__).resolve().parent / "visualization_demo_robot.urdf")
    with open(urdf_file, "w") as f:
        f.write(urdf_content)
    logger.info(f"Created visualization demo URDF: {urdf_file}")
    return urdf_file


def plot_workspace_visualization(urdf_processor: URDFToSerialManipulator) -> None:
    """
    Sample reachable end-effector positions and render a 3D workspace cloud.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nWorkspace Visualization")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    sample_size = 1200  # kept small so the CPU run stays fast

    workspace_points = []
    for _ in range(sample_size):
        joint_config = np.array(
            [np.random.uniform(lo, hi) for lo, hi in serial_manipulator.joint_limits]
        )
        T = serial_manipulator.forward_kinematics(joint_config, frame="space")
        workspace_points.append(T[:3, 3])

    workspace_points = np.array(workspace_points)
    print(f"   Generated {len(workspace_points)} reachable workspace points")
    print("   Workspace bounds:")
    for axis, name in enumerate("XYZ"):
        col = workspace_points[:, axis]
        print(f"     {name}: [{col.min():.3f}, {col.max():.3f}] m")

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    scatter = ax1.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        c=workspace_points[:, 2],
        cmap="viridis",
        alpha=0.6,
        s=2,
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Workspace\n(Color by Z-height)")
    fig.colorbar(scatter, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(132)
    ax2.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        c=workspace_points[:, 2],
        cmap="viridis",
        alpha=0.6,
        s=2,
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("XY Projection\n(Color by Z-height)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(133)
    distances = np.linalg.norm(workspace_points, axis=1)
    ax3.hist(distances, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax3.set_xlabel("Distance from Base (m)")
    ax3.set_ylabel("Number of Points")
    ax3.set_title("Reachability Distribution")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "workspace_visualization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def plot_manipulability_analysis(urdf_processor: URDFToSerialManipulator) -> None:
    """
    Compute manipulability and Jacobian condition number for several poses and plot them.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nManipulability Analysis")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data["actuated_joints_num"]
    analyzer = Singularity(serial_manipulator)

    test_configs = [
        ("Zero", np.zeros(num_joints)),
        ("Folded", np.array([0.0, 1.2, -1.6, 0.0][:num_joints])),
        ("Extended", np.array([0.5, 0.3, 0.2, 0.0][:num_joints])),
        ("Reaching", np.array([np.pi / 3, np.pi / 4, -np.pi / 6, np.pi / 2][:num_joints])),
    ]

    names, manips, conds = [], [], []
    for config_name, joint_angles in test_configs:
        J = serial_manipulator.jacobian(joint_angles, frame="space")
        manipulability = yoshikawa_measure(J)
        cond_num = float(analyzer.condition_number(joint_angles))
        if not np.isfinite(cond_num):
            cond_num = 1e3
        is_singular = analyzer.singularity_analysis(joint_angles)
        is_near = analyzer.near_singularity_detection(joint_angles)

        print(f"\n   {config_name}:")
        print(f"     Joint angles (deg): {np.rad2deg(joint_angles).round(1)}")
        print(f"     Manipulability: {manipulability:.6f}")
        print(f"     Condition number: {cond_num:.2f}")
        print(f"     Singular: {is_singular} | Near-singular: {is_near}")

        names.append(config_name)
        manips.append(manipulability)
        conds.append(min(cond_num, 1e3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bars1 = ax1.bar(names, manips, color=colors[: len(manips)])
    ax1.set_ylabel("Manipulability Measure")
    ax1.set_title("Manipulability by Configuration")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, manips):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars2 = ax2.bar(names, conds, color=colors[: len(conds)])
    ax2.set_ylabel("Condition Number")
    ax2.set_title("Jacobian Condition Number")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    for bar, value in zip(bars2, conds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out = OUTPUT_DIR / "manipulability_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Saved {out}")


def plot_manipulability_ellipsoid(urdf_processor: URDFToSerialManipulator) -> None:
    """
    Render the linear/angular velocity manipulability ellipsoids via Singularity.

    Uses ``Singularity.manipulability_ellipsoid`` with an explicit axis so the demo
    draws directly onto its own figure and saves it (no interactive window needed).

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nManipulability Ellipsoid")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data["actuated_joints_num"]
    analyzer = Singularity(serial_manipulator)

    theta = np.array([0.5, 0.3, 0.2, 0.0][:num_joints])
    fig = plt.figure(figsize=(12, 6))
    ax_lin = fig.add_subplot(121, projection="3d")
    ax_ang = fig.add_subplot(122, projection="3d")

    # The library draws the linear part on the first axis it receives. Call once per
    # axis so each panel gets its dedicated ellipsoid surface.
    analyzer.manipulability_ellipsoid(theta, ax=ax_lin)
    analyzer.manipulability_ellipsoid(theta, ax=ax_ang)
    ax_lin.set_title("Linear Velocity Ellipsoid")
    ax_ang.set_title("Angular Velocity Ellipsoid")

    plt.suptitle(
        f"Manipulability Ellipsoids @ {np.rad2deg(theta).round(0)} deg",
        fontweight="bold",
    )
    out = OUTPUT_DIR / "manipulability_ellipsoid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def demonstrate_trajectory_plotting(urdf_processor: URDFToSerialManipulator) -> None:
    """
    Generate a quintic joint-space trajectory and plot position/velocity/acceleration.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nTrajectory Plotting and Analysis")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    dynamics = urdf_processor.dynamics
    num_joints = urdf_processor.robot_data["actuated_joints_num"]
    joint_limits = [(-np.pi, np.pi)] * num_joints

    # use_cuda=None lets the planner auto-select and fall back to the CPU path.
    planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_processor.urdf_name,
        dynamics=dynamics,
        joint_limits=joint_limits,
        use_cuda=None,
    )
    print("   Trajectory planner created")

    start_config = np.array([0.0, 0.2, -0.3, 0.0][:num_joints])
    end_config = np.array([np.pi / 3, np.pi / 4, -np.pi / 6, np.pi / 2][:num_joints])
    Tf, N = 3.0, 50

    traj = planner.joint_trajectory(start_config, end_config, Tf, N, method=5)
    positions = traj["positions"]
    velocities = traj["velocities"]
    accelerations = traj["accelerations"]
    print(f"   Trajectory shape: {positions.shape} ({Tf}s, {N} steps, quintic)")

    time_vector = np.linspace(0, Tf, N)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    series = [
        (positions, "Position (deg)", "Positions"),
        (velocities, "Velocity (deg/s)", "Velocities"),
        (accelerations, "Acceleration (deg/s^2)", "Accelerations"),
    ]
    for ax, (data, ylabel, title) in zip(axes, series):
        for i in range(num_joints):
            ax.plot(time_vector, np.rad2deg(data[:, i]), label=f"Joint {i + 1}", linewidth=2)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Joint Trajectories - {title}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    out = OUTPUT_DIR / "trajectory_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def demonstrate_end_effector_visualization(
    urdf_processor: URDFToSerialManipulator,
) -> None:
    """
    Map joint-space motions through forward kinematics and plot the 3D EE paths.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nEnd-Effector Path Visualization")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data["actuated_joints_num"]
    steps = 60
    t = np.linspace(0, 2 * np.pi, steps)

    movements = {
        "circular": np.array(
            [0.3 * np.sin(t), np.full(steps, np.pi / 6), np.full(steps, -np.pi / 4), np.zeros(steps)]
        ).T[:, :num_joints],
        "figure_eight": np.array(
            [0.4 * np.sin(t), 0.2 * np.sin(2 * t), 0.3 * np.cos(t), 0.1 * np.sin(3 * t)]
        ).T[:, :num_joints],
    }

    fig = plt.figure(figsize=(14, 5))
    for idx, (name, joint_traj) in enumerate(movements.items()):
        print(f"   Computing {name} path...")
        ee = np.array(
            [
                serial_manipulator.forward_kinematics(cfg, frame="space")[:3, 3]
                for cfg in joint_traj
            ]
        )

        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        colors = plt.cm.viridis(np.linspace(0, 1, len(ee)))
        for i in range(len(ee) - 1):
            ax.plot3D(ee[i : i + 2, 0], ee[i : i + 2, 1], ee[i : i + 2, 2], color=colors[i], linewidth=2)
        ax.scatter(*ee[0], color="green", s=80, label="Start")
        ax.scatter(*ee[-1], color="red", s=80, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"End-Effector Path\n({name.replace('_', ' ').title()})")
        ax.legend()

        path_len = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
        print(f"     Path length: {path_len:.3f} m over {len(ee)} points")

    plt.tight_layout()
    out = OUTPUT_DIR / "end_effector_paths.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def demonstrate_configuration_space_visualization(
    urdf_processor: URDFToSerialManipulator,
) -> None:
    """
    Sweep the first two joints and map reachability and manipulability over that plane.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nConfiguration Space Visualization")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data["actuated_joints_num"]
    if num_joints < 2:
        print("   Need at least 2 joints for configuration-space visualization")
        return

    j1_lo, j1_hi = serial_manipulator.joint_limits[0]
    j2_lo, j2_hi = serial_manipulator.joint_limits[1]
    resolution = 40  # 1600 FK+Jacobian evals; fast on CPU
    j1_range = np.linspace(j1_lo, j1_hi, resolution)
    j2_range = np.linspace(j2_lo, j2_hi, resolution)
    J1, J2 = np.meshgrid(j1_range, j2_range)

    heights = np.zeros_like(J1)
    manipulability = np.zeros_like(J1)
    print(f"   Grid {resolution}x{resolution} over joints 1-2; computing FK + Jacobian...")

    for i in range(resolution):
        for j in range(resolution):
            cfg = np.zeros(num_joints)
            cfg[0], cfg[1] = J1[i, j], J2[i, j]
            T = serial_manipulator.forward_kinematics(cfg, frame="space")
            heights[i, j] = T[2, 3]
            Jm = serial_manipulator.jacobian(cfg, frame="space")
            manipulability[i, j] = yoshikawa_measure(Jm)

    extent = [np.rad2deg(j1_lo), np.rad2deg(j1_hi), np.rad2deg(j2_lo), np.rad2deg(j2_hi)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(heights, extent=extent, origin="lower", cmap="coolwarm", aspect="auto")
    axes[0].set_title("End-Effector Height (Z)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(manipulability, extent=extent, origin="lower", cmap="plasma", aspect="auto")
    axes[1].set_title("Manipulability Measure")
    fig.colorbar(im1, ax=axes[1])

    # Overlay a sinusoidal sweep through the centre of the joint plane.
    t = np.linspace(0, 2 * np.pi, 40)
    traj_j1 = 0.3 * np.sin(t) + (j1_lo + j1_hi) / 2
    traj_j2 = 0.2 * np.cos(t) + (j2_lo + j2_hi) / 2
    sc = axes[2].scatter(np.rad2deg(traj_j1), np.rad2deg(traj_j2), c=t, cmap="viridis", s=40)
    axes[2].plot(np.rad2deg(traj_j1), np.rad2deg(traj_j2), "k--", alpha=0.4)
    axes[2].set_title("Sample Joint-Space Path")
    fig.colorbar(sc, ax=axes[2], label="phase")

    for ax in axes:
        ax.set_xlabel("Joint 1 (deg)")
        ax.set_ylabel("Joint 2 (deg)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "configuration_space_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def create_summary_visualization(urdf_processor: URDFToSerialManipulator) -> None:
    """
    Assemble a single multi-panel figure summarizing the robot's key properties.

    Args:
        urdf_processor: Loaded ManipulaPy URDF processor for the demo robot.
    """
    print("\nSummary Visualization")
    print("=" * 60)

    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data["actuated_joints_num"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)

    # 1. Robot parameter text panel.
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    robot_info = [
        f"Robot: {urdf_processor.robot.name}",
        f"DOF: {num_joints}",
        f"Links: {len(urdf_processor.robot.links)}",
        f"Joints: {len(urdf_processor.robot.joints)}",
        "",
        "Joint Limits (deg):",
    ]
    for i, (lo, hi) in enumerate(serial_manipulator.joint_limits):
        robot_info.append(f"  J{i + 1}: [{np.rad2deg(lo):.0f}, {np.rad2deg(hi):.0f}]")
    ax1.text(
        0.05,
        0.95,
        "\n".join(robot_info),
        transform=ax1.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
    )
    ax1.set_title("Robot Parameters", fontweight="bold")

    # 2. Joint limit bars.
    ax2 = fig.add_subplot(gs[0, 1])
    joint_numbers = list(range(1, num_joints + 1))
    lower = [np.rad2deg(l[0]) for l in serial_manipulator.joint_limits]
    upper = [np.rad2deg(l[1]) for l in serial_manipulator.joint_limits]
    ax2.barh(joint_numbers, upper, alpha=0.7, color="skyblue", label="Upper")
    ax2.barh(joint_numbers, lower, alpha=0.7, color="lightcoral", label="Lower")
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Joint")
    ax2.set_title("Joint Angle Limits")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Workspace sample.
    ax3 = fig.add_subplot(gs[0, 2:], projection="3d")
    workspace_points = np.array(
        [
            serial_manipulator.forward_kinematics(
                np.array([np.random.uniform(lo, hi) for lo, hi in serial_manipulator.joint_limits]),
                frame="space",
            )[:3, 3]
            for _ in range(300)
        ]
    )
    ax3.scatter(
        workspace_points[:, 0],
        workspace_points[:, 1],
        workspace_points[:, 2],
        alpha=0.6,
        s=3,
        c=workspace_points[:, 2],
        cmap="viridis",
    )
    ax3.set_title("Workspace Sample")

    # 4. Sample joint trajectory.
    ax4 = fig.add_subplot(gs[1, :2])
    t = np.linspace(0, 2 * np.pi, 30)
    sample_traj = np.array(
        [0.3 * np.sin(t), 0.2 * np.cos(t + np.pi / 4), 0.25 * np.sin(t + np.pi / 2), 0.15 * np.cos(t)]
    ).T[:, :num_joints]
    time_vec = np.linspace(0, 3, 30)
    colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
    for i in range(num_joints):
        ax4.plot(time_vec, np.rad2deg(sample_traj[:, i]), label=f"Joint {i + 1}", color=colors[i], linewidth=2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Joint Angle (deg)")
    ax4.set_title("Sample Joint Trajectory")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. End-effector path of the sample trajectory.
    ax5 = fig.add_subplot(gs[1, 2:], projection="3d")
    ee = np.array(
        [serial_manipulator.forward_kinematics(cfg, frame="space")[:3, 3] for cfg in sample_traj]
    )
    ax5.plot(ee[:, 0], ee[:, 1], ee[:, 2], "b-", linewidth=3, alpha=0.8)
    ax5.scatter(*ee[0], color="green", s=80, label="Start")
    ax5.scatter(*ee[-1], color="red", s=80, label="End")
    ax5.set_title("End-Effector Path")
    ax5.legend(fontsize=8)

    # 6. Manipulability bar chart.
    ax6 = fig.add_subplot(gs[2, :2])
    configs = [
        np.zeros(num_joints),
        np.array([np.pi / 6, np.pi / 4, -np.pi / 6, 0][:num_joints]),
        np.array([np.pi / 3, np.pi / 3, -np.pi / 3, np.pi / 2][:num_joints]),
        np.array([-np.pi / 4, np.pi / 6, np.pi / 4, -np.pi / 4][:num_joints]),
    ]
    config_names = ["Zero", "Config 1", "Config 2", "Config 3"]
    manips = []
    for cfg in configs:
        J = serial_manipulator.jacobian(cfg, frame="space")
        manips.append(yoshikawa_measure(J))
    bars = ax6.bar(config_names, manips, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.8)
    ax6.set_ylabel("Manipulability")
    ax6.set_title("Manipulability for Different Configurations")
    ax6.grid(True, alpha=0.3)
    for bar, value in zip(bars, manips):
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 7. Statistics text panel.
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis("off")
    max_reach = float(np.max(np.linalg.norm(workspace_points, axis=1)))
    span = workspace_points.max(axis=0) - workspace_points.min(axis=0)
    workspace_volume = float(np.prod(span))
    stats_info = [
        "Robot Statistics:",
        "",
        f"Max Reach: {max_reach:.3f} m",
        f"Workspace Bounding Volume: {workspace_volume:.3f} m^3",
        f"Avg Manipulability: {np.mean(manips):.4f}",
        f"Max Manipulability: {np.max(manips):.4f}",
        "",
        "Demonstrated Features:",
        "- Workspace sampling",
        "- Joint limit analysis",
        "- Trajectory generation",
        "- End-effector paths",
        "- Manipulability + condition number",
        "- Manipulability ellipsoids",
        "- Configuration-space mapping",
    ]
    ax7.text(
        0.05,
        0.95,
        "\n".join(stats_info),
        transform=ax7.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
    )
    ax7.set_title("Summary Statistics", fontweight="bold")

    plt.suptitle(
        "ManipulaPy Visualization Demo - Complete Robot Analysis",
        fontsize=16,
        fontweight="bold",
    )
    out = OUTPUT_DIR / "complete_robot_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved {out}")


def main() -> int:
    """Run the full ManipulaPy visualization demonstration on the demo robot."""
    print("=== ManipulaPy: Basic Visualization Demo ===")
    print("Headless analysis-and-plotting tour of ManipulaPy's visualization tools")
    print()

    urdf_file = None
    try:
        print("Step 1: Creating visualization demo URDF")
        urdf_file = create_visualization_urdf()

        print("\nStep 2: Loading robot model")
        urdf_processor = URDFToSerialManipulator(urdf_file, use_pybullet_limits=True)
        joint_info = urdf_processor.print_joint_info()
        print(f"   Robot loaded: {joint_info['num_joints']} total joints")
        print(f"   Actuated joints: {urdf_processor.robot_data['actuated_joints_num']}")

        print("\nStep 3: Workspace visualization")
        plot_workspace_visualization(urdf_processor)

        print("\nStep 4: Manipulability analysis")
        plot_manipulability_analysis(urdf_processor)

        print("\nStep 5: Manipulability ellipsoids")
        plot_manipulability_ellipsoid(urdf_processor)

        print("\nStep 6: Trajectory plotting")
        demonstrate_trajectory_plotting(urdf_processor)

        print("\nStep 7: End-effector path visualization")
        demonstrate_end_effector_visualization(urdf_processor)

        print("\nStep 8: Configuration-space visualization")
        demonstrate_configuration_space_visualization(urdf_processor)

        print("\nStep 9: Summary visualization")
        create_summary_visualization(urdf_processor)

        print("\nVisualization demo completed successfully")
        print("=" * 60)
        print(f"Figures written to: {OUTPUT_DIR}")
        for name in [
            "workspace_visualization.png",
            "manipulability_analysis.png",
            "manipulability_ellipsoid.png",
            "trajectory_analysis.png",
            "end_effector_paths.png",
            "configuration_space_analysis.png",
            "complete_robot_analysis.png",
        ]:
            print(f"   - {name}")

    except Exception as e:  # pragma: no cover - top-level guard
        logger.error(f"Visualization demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if urdf_file and os.path.exists(urdf_file):
            os.remove(urdf_file)
            logger.info(f"Cleaned up URDF file: {urdf_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
