#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Differentiable Reach Showcase — gradient-based trajectory optimization
========================================================================

A Franka Panda has to move its end-effector from a start pose to a goal
pose. A straight line through joint space is the cheapest path — and it
plows straight through an obstacle sitting between the two. This script
fixes the path by gradient descent, using an objective ("stay clear of the
obstacle, stay smooth") that nobody hand-derived a Jacobian for.

That is the point. Before ManipulaPy v1.4, differentiating an *arbitrary*
scalar objective through forward kinematics meant deriving the gradient by
hand or paying for finite differences (n+1 extra model evaluations per
step, truncation error, no exactness). v1.4's unified backend system runs
the same ``forward_kinematics`` call on NumPy, CuPy, PyTorch or JAX, and
under the two autodiff backends every array op becomes a differentiable
one — so ``jax.grad`` of any composition of FK calls just works, exact to
machine precision, at essentially the cost of one forward pass.

This script:
  1. Builds a naive straight-line joint trajectory that collides with a
     spherical obstacle.
  2. Optimizes the interior waypoints with ``jax.value_and_grad`` through
     ``forward_kinematics`` — no closed-form Jacobian was written for this
     objective; JAX built it from the trace.
  3. Plots the before/after end-effector paths, the loss curve, and the
     clearance-to-obstacle profile.
  4. Renders the optimized motion as a GIF via PyBullet's headless
     software renderer (no GPU, no display required).

Usage:
    python showcase/differentiable_reach_showcase.py

Requires the ``[jax-cpu]`` extra for the optimization and ``[simulation]``
for the GIF; the script degrades gracefully (skips the GIF, still produces
the optimization plot) if PyBullet isn't installed.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

try:
    import ManipulaPy  # noqa: F401
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    import ManipulaPy  # noqa: F401

from ManipulaPy import ManipulaPy_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.backend import get_registered, use_backend

OUTPUT_DIR = Path(__file__).resolve().parent
PANDA_URDF = os.path.join(
    os.path.dirname(ManipulaPy_data.__file__), "franka_panda", "panda.urdf"
)

# A non-singular start and a reaching-across-the-body goal — far enough apart
# that a straight joint-space line sweeps through the space between them.
START = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.0])
GOAL = np.array([1.2, 0.2, -0.4, -1.6, 0.3, 1.9, 0.6])
N_WAYPOINTS = 30
OBSTACLE_MARGIN = 0.12  # meters — required clearance from the obstacle center
LEARNING_RATE = 0.03
MOMENTUM = 0.9
ITERATIONS = 300
OBSTACLE_WEIGHT = 120.0
SMOOTHNESS_WEIGHT = 1.0


def check_jax_available() -> bool:
    try:
        probe = get_registered("jax")
        probe.to_numpy(probe.asarray([1.0]))
        return True
    except Exception:
        return False


def naive_trajectory() -> np.ndarray:
    """Linear interpolation in joint space — the path nobody optimized."""
    t = np.linspace(0.0, 1.0, N_WAYPOINTS)[:, None]
    return START[None, :] + t * (GOAL - START)[None, :]


def ee_path(serial, trajectory: np.ndarray) -> np.ndarray:
    """End-effector positions along a joint-space trajectory (NumPy backend)."""
    return np.array([serial.forward_kinematics(q)[:3, 3] for q in trajectory])


def optimize_trajectory(serial, obstacle: np.ndarray):
    """Gradient-descend the interior waypoints away from the obstacle.

    Returns (optimized_trajectory, loss_history). Runs under a scoped JAX
    backend so the rest of the script — and the caller's own backend
    choice — is unaffected once this function returns.
    """
    import jax
    import jax.numpy as jnp

    with use_backend("jax"):

        def ee_pos(theta):
            return serial.forward_kinematics(theta)[:3, 3]

        start_j, goal_j, obstacle_j = jnp.array(START), jnp.array(GOAL), jnp.array(obstacle)

        def full_trajectory(mid):
            return jnp.concatenate([start_j[None, :], mid, goal_j[None, :]], axis=0)

        def loss_fn(mid):
            traj = full_trajectory(mid)
            ee = jax.vmap(ee_pos)(traj)
            clearance = jnp.linalg.norm(ee - obstacle_j[None, :], axis=-1)
            # Quadratic penalty only kicks in inside the safety margin — zero
            # gradient once a waypoint is clear, so the optimizer stops pushing.
            obstacle_penalty = jnp.sum(jnp.clip(OBSTACLE_MARGIN - clearance, 0.0, None) ** 2)
            smoothness = jnp.sum(jnp.sum(jnp.diff(traj, axis=0) ** 2, axis=-1))
            return OBSTACLE_WEIGHT * obstacle_penalty + SMOOTHNESS_WEIGHT * smoothness

        # jax.value_and_grad differentiates the WHOLE composition above —
        # forward kinematics through every waypoint, the obstacle penalty,
        # the smoothness term — with no hand-derived Jacobian anywhere.
        val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        mid = jnp.array(naive_trajectory()[1:-1])
        velocity = jnp.zeros_like(mid)
        loss_history = []

        print(f"   Optimizing {N_WAYPOINTS - 2} free waypoints "
              f"({(N_WAYPOINTS - 2) * 7} scalars) for {ITERATIONS} steps...")
        start_time = time.time()
        for it in range(ITERATIONS):
            loss_val, grad = val_and_grad(mid)
            velocity = MOMENTUM * velocity - LEARNING_RATE * grad
            mid = mid + velocity
            loss_history.append(float(loss_val))
            if it % 50 == 0 or it == ITERATIONS - 1:
                print(f"     step {it:4d}   loss = {loss_val:8.4f}")
        elapsed = time.time() - start_time
        print(f"   ✅ Converged in {elapsed:.2f}s "
              f"({elapsed / ITERATIONS * 1000:.2f} ms/step, JIT-compiled gradient)")

        optimized = np.array(full_trajectory(mid))

    return optimized, loss_history


def render_comparison_figure(ee_naive, ee_opt, obstacle, loss_history, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        "ManipulaPy v1.4 — Gradient-Based Trajectory Optimization Through Autodiff FK",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: 3D paths + obstacle
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.plot(*ee_naive.T, "r--", lw=2, label="naive (collides)")
    ax0.plot(*ee_opt.T, "g-", lw=2.5, label="optimized (autodiff)")
    ax0.scatter(*ee_naive[0], c="k", marker="o", s=40, label="start")
    ax0.scatter(*ee_naive[-1], c="k", marker="^", s=40, label="goal")
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    r = OBSTACLE_MARGIN
    ax0.plot_surface(
        obstacle[0] + r * np.cos(u) * np.sin(v),
        obstacle[1] + r * np.sin(u) * np.sin(v),
        obstacle[2] + r * np.cos(v),
        color="orange", alpha=0.25, linewidth=0,
    )
    ax0.set_title("End-effector path")
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]"); ax0.set_zlabel("z [m]")
    ax0.legend(loc="upper left", fontsize=8)

    # Panel 2: loss curve
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(loss_history, color="steelblue")
    ax1.set_yscale("log")
    ax1.set_xlabel("gradient step")
    ax1.set_ylabel("loss (log scale)")
    ax1.set_title("Optimization convergence")
    ax1.grid(True, alpha=0.3)

    # Panel 3: clearance profile
    ax2 = fig.add_subplot(1, 3, 3)
    d_naive = np.linalg.norm(ee_naive - obstacle[None, :], axis=-1)
    d_opt = np.linalg.norm(ee_opt - obstacle[None, :], axis=-1)
    ax2.plot(d_naive, "r--", lw=2, label="naive")
    ax2.plot(d_opt, "g-", lw=2.5, label="optimized")
    ax2.axhline(OBSTACLE_MARGIN, color="orange", linestyle=":", label="required margin")
    ax2.set_xlabel("waypoint index")
    ax2.set_ylabel("distance to obstacle [m]")
    ax2.set_title("Obstacle clearance")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved {out_path}")


def _pybullet_loadable_panda_urdf() -> str:
    """Copy the bundled Panda URDF with plain relative mesh paths.

    PyBullet's C++ URDF loader doesn't resolve the ``package://`` URIs the
    shipped file uses (ManipulaPy's own parser does — this is purely a
    PyBullet loader limitation), so this rewrites them to relative paths in
    a scratch directory. Also stamps a placeholder <inertial> block on every
    link that lacks one, only to silence PyBullet's per-link warning; the
    scene here is posed kinematically and never simulates dynamics.
    """
    src_dir = os.path.dirname(PANDA_URDF)
    tmp_dir = tempfile.mkdtemp(prefix="manipulapy_panda_pb_")
    with open(PANDA_URDF) as f:
        text = f.read().replace("package://franka_description/meshes/", "meshes/")
    inertial = (
        '<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="1"/>'
        '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
    )
    text = re.sub(r'(<link name="[^"]*">)', r"\1" + inertial, text)
    text = re.sub(r'<link name="([^"]*)"/>', r'<link name="\1">' + inertial + "</link>", text)
    path = os.path.join(tmp_dir, "panda.urdf")
    with open(path, "w") as f:
        f.write(text)
    try:
        os.symlink(os.path.join(src_dir, "meshes"), os.path.join(tmp_dir, "meshes"))
    except OSError:
        shutil.copytree(os.path.join(src_dir, "meshes"), os.path.join(tmp_dir, "meshes"))
    return path


def render_gif(trajectory: np.ndarray, obstacle: np.ndarray, out_path: Path) -> bool:
    """Render the optimized motion in headless PyBullet, save as an animated GIF."""
    try:
        import pybullet as p
    except ImportError:
        print("   ⏭️  pybullet not installed — skipping GIF "
              "(pip install \"ManipulaPy[simulation]\")")
        return False

    print("   Rendering PyBullet frames (headless, software renderer)...")
    urdf_path = _pybullet_loadable_panda_urdf()

    client = p.connect(p.DIRECT)
    try:
        saved_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=client)
            try:
                import pybullet_data

                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
                p.loadURDF(
                    "plane.urdf", basePosition=[0, 0, -0.01], physicsClientId=client,
                )
            except ImportError:
                pass
        finally:
            os.dup2(saved_fd, 2)
            os.close(devnull)
            os.close(saved_fd)

        p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, radius=OBSTACLE_MARGIN,
                rgbaColor=[1.0, 0.55, 0.0, 0.45], physicsClientId=client,
            ),
            basePosition=obstacle.tolist(), physicsClientId=client,
        )

        arm_joints = list(range(7))  # panda_joint1..7 — the actuated arm, no gripper
        width, height = 640, 480
        view = p.computeViewMatrixFromYawPitchRoll(
            [0.3, 0.15, 0.35], distance=1.15, yaw=35, pitch=-20, roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(55, width / height, 0.05, 5.0)

        frames = []
        from PIL import Image

        for q in trajectory:
            for j, angle in zip(arm_joints, q):
                p.resetJointState(robot_id, j, float(angle), physicsClientId=client)
            img = p.getCameraImage(
                width, height, view, proj,
                renderer=p.ER_TINY_RENDERER, physicsClientId=client,
            )
            rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
            frames.append(Image.fromarray(rgb))

        # Hold the final frame for a beat so the GIF doesn't loop too abruptly.
        frames += [frames[-1]] * 10
        frames[0].save(
            out_path, save_all=True, append_images=frames[1:], duration=60, loop=0,
        )
        print(f"   ✅ Saved {out_path} ({len(frames)} frames)")
        return True
    finally:
        p.disconnect(client)


def main() -> None:
    print("=" * 72)
    print("  ManipulaPy v1.4 Showcase: Differentiable Trajectory Optimization")
    print("=" * 72)

    if not check_jax_available():
        print("\n❌ This showcase needs the JAX backend: pip install \"ManipulaPy[jax-cpu]\"")
        sys.exit(1)

    print("\n🤖 Loading the bundled Franka Panda...")
    proc = URDFToSerialManipulator(PANDA_URDF)
    serial = proc.serial_manipulator

    print("\n📏 Building the naive straight-line joint trajectory...")
    naive = naive_trajectory()
    ee_naive = ee_path(serial, naive)
    obstacle = ee_naive[N_WAYPOINTS // 2].copy()
    print(f"   Obstacle placed at the naive path's midpoint: {obstacle.round(3)} — "
          f"guaranteed collision, margin {OBSTACLE_MARGIN} m")

    print("\n∂ Optimizing with jax.value_and_grad through forward_kinematics...")
    optimized, loss_history = optimize_trajectory(serial, obstacle)
    ee_opt = ee_path(serial, optimized)

    d_naive_min = float(np.min(np.linalg.norm(ee_naive - obstacle[None, :], axis=-1)))
    d_opt_min = float(np.min(np.linalg.norm(ee_opt - obstacle[None, :], axis=-1)))
    print(f"\n📊 Minimum clearance to obstacle:")
    print(f"   naive path:      {d_naive_min * 100:5.1f} cm  (< {OBSTACLE_MARGIN * 100:.0f} cm required — collides)")
    print(f"   optimized path:  {d_opt_min * 100:5.1f} cm  (>= {OBSTACLE_MARGIN * 100:.0f} cm required)")

    print("\n📈 Rendering the comparison figure...")
    render_comparison_figure(
        ee_naive, ee_opt, obstacle, loss_history,
        OUTPUT_DIR / "differentiable_reach_showcase.png",
    )

    print("\n🎬 Rendering the optimized motion as a GIF...")
    render_gif(optimized, obstacle, OUTPUT_DIR / "differentiable_reach_showcase.gif")

    print("\n" + "=" * 72)
    print("✅ Showcase complete.")
    print("📚 This objective — 'stay outside a margin, stay smooth' — has no closed-form")
    print("   Jacobian anywhere in this script. jax.grad built it from the trace of an")
    print("   ordinary ManipulaPy forward_kinematics call. Swap in torch.autograd, or")
    print("   any other differentiable objective (energy, jerk, learned cost) and the")
    print("   same pattern applies — see notebooks/12_differentiable_robotics.ipynb for")
    print("   the full contract (what's differentiable, what isn't, and why).")
    print("=" * 72)


if __name__ == "__main__":
    main()
