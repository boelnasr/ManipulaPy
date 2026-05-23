#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Record the GIFs embedded in README.md.

Produces three small animated GIFs in docs/source/_static/gifs/:

    joint_trajectory.gif  - 6-DOF joint trajectory unrolling over time
    ee_path.gif           - end-effector path in 3D, traced out frame-by-frame
    workspace.gif         - Monte-Carlo reachable workspace accumulating samples

All three use matplotlib's PillowWriter, so no system tools (ffmpeg, X server,
screen recorders) are required. Each clip is ~1-2 MB; the whole script runs
in well under a minute on a laptop CPU.

Re-run after any major release so the README visuals stay in sync with the
current API.

Usage:
    python scripts/record_readme_gifs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
from ManipulaPy.urdf import URDF

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "gifs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAJ = 120     # trajectory waypoints (keeps GIFs small)
FPS = 20         # playback frame rate


def _load_ur5():
    urdf_path = get_robot_urdf("ur5")
    robot = URDF.load(urdf_path)
    return urdf_path, robot.to_serial_manipulator(), robot.to_manipulator_dynamics()


def render_joint_trajectory(serial, dynamics, urdf_path):
    """Six joint angles evolving over a quintic-timed trajectory."""
    n_joints = 6
    start = np.zeros(n_joints)
    end = np.array([0.6, -0.4, 0.8, -0.5, 0.3, -0.2])
    planner = OptimizedTrajectoryPlanning(
        serial, urdf_path, dynamics,
        joint_limits=[(-np.pi, np.pi)] * n_joints,
    )
    traj = planner.joint_trajectory(
        thetastart=start, thetaend=end,
        Tf=4.0, N=N_TRAJ, method=5,
    )
    positions = np.asarray(traj["positions"])
    t = np.linspace(0, 4.0, N_TRAJ)

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
    fig.patch.set_facecolor("white")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_joints))
    lines = [
        ax.plot([], [], lw=2.0, color=colors[j], label=f"joint {j+1}")[0]
        for j in range(n_joints)
    ]
    ax.set_xlim(0, 4.0)
    ax.set_ylim(positions.min() - 0.1, positions.max() + 0.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title("Quintic joint trajectory — ManipulaPy")
    ax.legend(loc="lower right", ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    def update(frame):
        for j, line in enumerate(lines):
            line.set_data(t[: frame + 1], positions[: frame + 1, j])
        return lines

    anim = FuncAnimation(fig, update, frames=N_TRAJ, blit=True, interval=1000 / FPS)
    out = OUT_DIR / "joint_trajectory.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out


def render_ee_path(serial, dynamics, urdf_path):
    """End-effector position traced in 3D as the trajectory unrolls."""
    n_joints = 6
    start = np.zeros(n_joints)
    end = np.array([0.6, -0.4, 0.8, -0.5, 0.3, -0.2])
    planner = OptimizedTrajectoryPlanning(
        serial, urdf_path, dynamics,
        joint_limits=[(-np.pi, np.pi)] * n_joints,
    )
    traj = planner.joint_trajectory(
        thetastart=start, thetaend=end,
        Tf=4.0, N=N_TRAJ, method=5,
    )
    positions = np.asarray(traj["positions"])

    ee = np.array([
        serial.forward_kinematics(q, frame="space")[:3, 3] for q in positions
    ])

    fig = plt.figure(figsize=(5.5, 4.5), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(ee[:, 0].min() - 0.05, ee[:, 0].max() + 0.05)
    ax.set_ylim(ee[:, 1].min() - 0.05, ee[:, 1].max() + 0.05)
    ax.set_zlim(ee[:, 2].min() - 0.05, ee[:, 2].max() + 0.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("End-effector path (UR5)")
    line, = ax.plot([], [], [], color="#2980B9", lw=2.0)
    tip, = ax.plot([], [], [], "o", color="#E74C3C", markersize=7)

    def update(frame):
        line.set_data(ee[: frame + 1, 0], ee[: frame + 1, 1])
        line.set_3d_properties(ee[: frame + 1, 2])
        tip.set_data([ee[frame, 0]], [ee[frame, 1]])
        tip.set_3d_properties([ee[frame, 2]])
        ax.view_init(elev=22, azim=-60 + frame * 0.6)
        return line, tip

    anim = FuncAnimation(fig, update, frames=N_TRAJ, interval=1000 / FPS)
    out = OUT_DIR / "ee_path.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out


def render_workspace(serial):
    """Monte-Carlo reachable workspace accumulating points."""
    rng = np.random.default_rng(0)
    n_total = 800
    joint_limits = np.array([(-np.pi, np.pi)] * 6)
    samples = rng.uniform(joint_limits[:, 0], joint_limits[:, 1], size=(n_total, 6))
    ee = np.array([
        serial.forward_kinematics(q, frame="space")[:3, 3] for q in samples
    ])

    fig = plt.figure(figsize=(5.5, 4.5), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    span = float(np.abs(ee).max()) + 0.05
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(ee[:, 2].min() - 0.05, ee[:, 2].max() + 0.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Reachable workspace (UR5, Monte-Carlo)")

    scat = ax.scatter([], [], [], s=4, c="#27AE60", alpha=0.5)
    counter = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)
    chunk = max(1, n_total // (N_TRAJ // 2))

    def update(frame):
        upto = min(n_total, (frame + 1) * chunk)
        scat._offsets3d = (ee[:upto, 0], ee[:upto, 1], ee[:upto, 2])
        counter.set_text(f"samples: {upto}/{n_total}")
        ax.view_init(elev=18, azim=-50 + frame * 0.5)
        return scat, counter

    anim = FuncAnimation(fig, update, frames=N_TRAJ // 2, interval=1000 / FPS)
    out = OUT_DIR / "workspace.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out


def main():
    print("Loading UR5...")
    urdf_path, serial, dynamics = _load_ur5()

    print("Recording joint_trajectory.gif...")
    p1 = render_joint_trajectory(serial, dynamics, urdf_path)
    print(f"  -> {p1.relative_to(Path.cwd()) if p1.is_relative_to(Path.cwd()) else p1}")

    print("Recording ee_path.gif...")
    p2 = render_ee_path(serial, dynamics, urdf_path)
    print(f"  -> {p2.relative_to(Path.cwd()) if p2.is_relative_to(Path.cwd()) else p2}")

    print("Recording workspace.gif...")
    p3 = render_workspace(serial)
    print(f"  -> {p3.relative_to(Path.cwd()) if p3.is_relative_to(Path.cwd()) else p3}")

    for p in (p1, p2, p3):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
