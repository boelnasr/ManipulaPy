#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Differentiable Reach Showcase — gradient-based trajectory optimization
========================================================================

A Franka Panda has to move its end-effector from a start pose to a goal
pose. A straight line through joint space is the cheapest path — and it
plows straight through an obstacle sitting between the two. This script
fixes the path by gradient descent, using an objective ("keep the whole
arm clear of the obstacle, stay smooth") that nobody hand-derived a
Jacobian for.

That is the point. Before ManipulaPy v1.4, differentiating an *arbitrary*
scalar objective through forward kinematics meant deriving the gradient by
hand or paying for finite differences (n+1 extra model evaluations per
step, truncation error, no exactness). v1.4's unified backend system runs
the same ``forward_kinematics`` call on NumPy, CuPy, PyTorch or JAX, and
under the two autodiff backends every array op becomes a differentiable
one — so ``jax.grad`` of any composition of FK calls just works, exact to
machine precision, at essentially the cost of one forward pass.

**Whole-arm clearance, not just the end-effector.** An earlier version of
this script only kept the *end-effector point* outside the obstacle. That
plot looked right, but the physical robot doesn't collide as a point — its
forearm, wrist and gripper have volume, and they swept right through the
obstacle even while the fingertip stayed clear. This version differentiates
through every link's origin along the kinematic chain (via
``URDF.kinematic_chain`` / ``link_fk``'s own joint-origin math, reimplemented
here in JAX so it's traceable) plus interpolated points along each link
segment, and requires *all* of them to clear a combined
robot-radius + obstacle-radius margin. It's still a capsule approximation
(a uniform per-link radius, not the real meshes) — accurate enough to make
the optimizer honest, not a substitute for a full collision mesh.

This script:
  1. Builds a naive straight-line joint trajectory whose end-effector *and*
     forearm/wrist/gripper links collide with a spherical obstacle.
  2. Optimizes the interior waypoints with ``jax.value_and_grad`` through a
     whole-chain clearance objective — no closed-form Jacobian was written
     for it; JAX built it from the trace.
  3. Plots the before/after end-effector paths (with robot-skeleton overlays
     at several waypoints), the loss curve, and the whole-arm clearance
     profile.
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
from dataclasses import dataclass
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
from ManipulaPy.urdf.types import JointType
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
N_SEGMENT_SAMPLES = 2  # interior points sampled along each link segment
LINK_RADIUS = 0.07     # meters — approximate capsule radius of the arm/gripper
OBSTACLE_RADIUS = 0.08  # meters — the obstacle's actual physical size
CLEARANCE_PAD = 0.02    # meters — extra safety margin on top of the two radii
REQUIRED_CLEARANCE = LINK_RADIUS + OBSTACLE_RADIUS + CLEARANCE_PAD
LEARNING_RATE = 0.03
MOMENTUM = 0.9
ITERATIONS = 400
OBSTACLE_WEIGHT = 120.0
SMOOTHNESS_WEIGHT = 1.0

# Links not to bother checking for collision: the two fingers (their opening
# doesn't matter for arm-vs-obstacle avoidance) and their fixed parent joint.
_IGNORED_CHAIN_LINKS = {"panda_leftfinger", "panda_rightfinger"}


@dataclass
class ChainMeta:
    """Static (non-differentiable) per-joint data for the base→hand chain.

    ``is_revolute`` and ``act_index`` are plain Python lists, not JAX arrays:
    they only ever drive a Python-level ``for i in range(n)`` unrolled at
    trace time, so keeping them as JAX arrays gains nothing and makes JAX
    treat their indexing as an abstract op instead of a compile-time
    constant — ``int(jax_array[i])`` on that path raises
    ``ConcretizationTypeError`` under ``jit``/``vmap``.
    """

    origin_mats: "object"   # jnp.ndarray (L, 4, 4)
    safe_axes: "object"     # jnp.ndarray (L, 3), never zero-norm
    is_revolute: list       # length L, bool
    act_index: list         # length L, int, -1 for non-actuated joints
    link_names: list        # length L+1, base link first


def build_chain_meta(robot, arm_joint_names) -> ChainMeta:
    """Precompute JAX-ready chain data from the URDF's own joint origins/axes.

    Mirrors ``URDF.link_fk``'s per-joint math (``origin.matrix`` composed with
    an axis-angle rotation) exactly, but with the rotation built from JAX ops
    so it's differentiable w.r.t. the joint angles. Verified against
    ``robot.link_fk`` directly (see ``showcase/tests`` — run this file's
    ``python -m pytest`` target, or compare manually: max deviation is
    ~3e-8, i.e. float32 JAX vs float64 NumPy, not a modeling error).
    """
    import jax.numpy as jnp

    chain = [j for j in robot.kinematic_chain if j.child not in _IGNORED_CHAIN_LINKS]
    idx_map = {name: i for i, name in enumerate(arm_joint_names)}

    origin_mats = jnp.stack([jnp.array(j.origin.matrix) for j in chain])
    raw_axes = jnp.stack([jnp.array(j.axis) for j in chain])
    # Fixed joints often carry a degenerate (zero) axis, since get_child_pose
    # never uses it for them. Normalizing 0/0 produces NaN that survives even
    # a `sin(0) * NaN` multiply — the same class of bug the v1.4 gradient
    # fixes addressed in MatrixLog6. Guard the norm before dividing.
    axis_norms = jnp.linalg.norm(raw_axes, axis=-1, keepdims=True)
    safe_axes = jnp.where(
        axis_norms > 1e-9,
        raw_axes / jnp.where(axis_norms > 1e-9, axis_norms, 1.0),
        jnp.array([1.0, 0.0, 0.0]),
    )
    is_revolute = [j.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS) for j in chain]
    act_index = [idx_map.get(j.name, -1) for j in chain]
    link_names = [chain[0].parent] + [j.child for j in chain]

    return ChainMeta(origin_mats, safe_axes, is_revolute, act_index, link_names)


def _rotation_from_axis_angle(axis, angle):
    """Rodrigues' formula, built from JAX ops so it differentiates through angle."""
    import jax.numpy as jnp

    zero = jnp.zeros(())
    K = jnp.array(
        [[zero, -axis[2], axis[1]], [axis[2], zero, -axis[0]], [-axis[1], axis[0], zero]]
    )
    return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)


def chain_link_positions(theta, meta: ChainMeta):
    """Every link origin along the chain, in the base frame — differentiable.

    Returns an (L+1, 3) array: the base origin followed by one point per
    joint in ``meta``, in the same order as ``meta.link_names``.
    """
    import jax.numpy as jnp

    n = meta.origin_mats.shape[0]
    T = jnp.eye(4)
    positions = [T[:3, 3]]
    for i in range(n):
        act_i = meta.act_index[i]  # plain Python int — a compile-time constant
        if act_i >= 0 and meta.is_revolute[i]:
            angle = theta[act_i]
        else:
            angle = jnp.asarray(0.0, dtype=theta.dtype)
        R = _rotation_from_axis_angle(meta.safe_axes[i], angle)
        T_joint = jnp.eye(4).at[:3, :3].set(R)
        T = T @ meta.origin_mats[i] @ T_joint
        positions.append(T[:3, 3])
    return jnp.stack(positions)


def sample_collision_points(link_positions, n_interior: int = N_SEGMENT_SAMPLES):
    """Link origins plus ``n_interior`` interpolated points per segment.

    A cheap stand-in for sampling along each link's capsule: catches an
    obstacle sitting mid-link, not just at a joint origin.
    """
    import jax.numpy as jnp

    if n_interior <= 0:
        return link_positions
    p0 = link_positions[:-1]
    p1 = link_positions[1:]
    t = jnp.linspace(0.0, 1.0, n_interior + 2)[1:-1]  # exclude the endpoints
    # (n_interior, n_segments, 3)
    interior = p0[None, :, :] + t[:, None, None] * (p1 - p0)[None, :, :]
    interior = interior.reshape(-1, 3)
    return jnp.concatenate([link_positions, interior], axis=0)


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


def whole_arm_min_clearance(trajectory: np.ndarray, meta: ChainMeta, obstacle: np.ndarray) -> np.ndarray:
    """Per-waypoint minimum distance from any sampled arm point to the obstacle."""
    import jax
    import jax.numpy as jnp

    def per_waypoint(theta):
        pts = sample_collision_points(chain_link_positions(theta, meta))
        return jnp.min(jnp.linalg.norm(pts - obstacle[None, :], axis=-1))

    return np.array(jax.vmap(per_waypoint)(jnp.array(trajectory)))


def optimize_trajectory(serial, robot, arm_joint_names, obstacle: np.ndarray):
    """Gradient-descend the interior waypoints to keep the whole arm clear.

    Returns (optimized_trajectory, loss_history). Runs under a scoped JAX
    backend so the rest of the script — and the caller's own backend
    choice — is unaffected once this function returns.
    """
    import jax
    import jax.numpy as jnp

    with use_backend("jax"):
        meta = build_chain_meta(robot, arm_joint_names)
        start_j, goal_j, obstacle_j = jnp.array(START), jnp.array(GOAL), jnp.array(obstacle)

        def full_trajectory(mid):
            return jnp.concatenate([start_j[None, :], mid, goal_j[None, :]], axis=0)

        def loss_fn(mid):
            traj = full_trajectory(mid)
            link_pos = jax.vmap(lambda th: chain_link_positions(th, meta))(traj)
            sample_pts = jax.vmap(sample_collision_points)(link_pos)  # (N, M, 3)
            clearance = jnp.linalg.norm(sample_pts - obstacle_j[None, None, :], axis=-1)
            # Quadratic penalty only kicks in inside the safety margin — zero
            # gradient once a point is clear, so the optimizer stops pushing.
            obstacle_penalty = jnp.sum(jnp.clip(REQUIRED_CLEARANCE - clearance, 0.0, None) ** 2)
            smoothness = jnp.sum(jnp.sum(jnp.diff(traj, axis=0) ** 2, axis=-1))
            return OBSTACLE_WEIGHT * obstacle_penalty + SMOOTHNESS_WEIGHT * smoothness

        # jax.value_and_grad differentiates the WHOLE composition above — every
        # link's forward kinematics through every waypoint, the obstacle
        # penalty, the smoothness term — with no hand-derived Jacobian anywhere.
        val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        mid = jnp.array(naive_trajectory()[1:-1])
        velocity = jnp.zeros_like(mid)
        loss_history = []

        n_points = meta.origin_mats.shape[0] + 1
        n_samples = n_points + (n_points - 1) * N_SEGMENT_SAMPLES
        print(f"   Optimizing {N_WAYPOINTS - 2} free waypoints "
              f"({(N_WAYPOINTS - 2) * 7} scalars) against {n_samples} sampled points "
              f"per waypoint along the whole arm, for {ITERATIONS} steps...")
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
        meta_out = meta

    return optimized, loss_history, meta_out


def _skeleton_points(theta: np.ndarray, meta: ChainMeta) -> np.ndarray:
    import jax.numpy as jnp

    return np.array(chain_link_positions(jnp.array(theta), meta))


def render_comparison_figure(
    ee_naive, ee_opt, naive_clearance, opt_clearance, obstacle, loss_history,
    naive_traj, opt_traj, meta: ChainMeta, out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(
        "ManipulaPy v1.4 — Gradient-Based Trajectory Optimization Through Autodiff FK "
        "(whole-arm clearance)",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: 3D end-effector paths + obstacle + robot-skeleton overlays
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.plot(*ee_naive.T, "r--", lw=1.5, label="naive EE path (collides)")
    ax0.plot(*ee_opt.T, "g-", lw=2.5, label="optimized EE path")
    ax0.scatter(*ee_naive[0], c="k", marker="o", s=40, label="start")
    ax0.scatter(*ee_naive[-1], c="k", marker="^", s=40, label="goal")

    # A few full-arm skeletons along the optimized path make the whole-body
    # clearance visible, not just the fingertip curve. The start/goal poses
    # are fixed (identical for both trajectories) and happen to sit closer
    # to the obstacle than the detour does, so the interesting waypoint is
    # not "minimum optimized clearance" — it's the waypoint where the NAIVE
    # path was deepest inside the obstacle; showing the optimized pose there
    # is the direct "this is what replaced the collision" comparison.
    collision_idx = int(np.argmin(naive_clearance))
    interior = np.linspace(2, len(opt_traj) - 3, 3).astype(int)
    skeleton_indices = sorted(set(interior.tolist() + [collision_idx]))
    cmap = plt.get_cmap("Blues")
    for k, idx in enumerate(skeleton_indices):
        skel = _skeleton_points(opt_traj[idx], meta)
        if idx == collision_idx:
            color, lw, label = "crimson", 2.2, f"at naive collision point (wp {idx})"
        else:
            color = cmap(0.35 + 0.5 * k / max(len(skeleton_indices) - 1, 1))
            lw, label = 1.4, None
        ax0.plot(*skel.T, "-o", color=color, lw=lw, ms=3, alpha=0.9, label=label)

    # And the naive (colliding) pose at that same waypoint, for direct contrast.
    naive_skel = _skeleton_points(naive_traj[collision_idx], meta)
    ax0.plot(*naive_skel.T, "--", color="firebrick", lw=1.6, alpha=0.8,
             label=f"naive pose (wp {collision_idx}, collides)")

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    r = OBSTACLE_RADIUS
    ax0.plot_surface(
        obstacle[0] + r * np.cos(u) * np.sin(v),
        obstacle[1] + r * np.sin(u) * np.sin(v),
        obstacle[2] + r * np.cos(v),
        color="orange", alpha=0.35, linewidth=0,
    )
    ax0.set_title("End-effector path + arm skeletons")
    ax0.set_xlabel("x [m]"); ax0.set_ylabel("y [m]"); ax0.set_zlabel("z [m]")
    ax0.legend(loc="upper left", fontsize=7)

    # Panel 2: loss curve
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.plot(loss_history, color="steelblue")
    ax1.set_yscale("log")
    ax1.set_xlabel("gradient step")
    ax1.set_ylabel("loss (log scale)")
    ax1.set_title("Optimization convergence")
    ax1.grid(True, alpha=0.3)

    # Panel 3: whole-arm clearance profile (minimum over ALL sampled points,
    # not just the end-effector) — this is the number that actually answers
    # "does the robot's body avoid the obstacle?"
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(naive_clearance, "r--", lw=2, label="naive (whole-arm min)")
    ax2.plot(opt_clearance, "g-", lw=2.5, label="optimized (whole-arm min)")
    ax2.axhline(REQUIRED_CLEARANCE, color="orange", linestyle=":", label="required clearance")
    ax2.axhline(0.0, color="gray", linestyle="-", lw=0.8)
    ax2.set_xlabel("waypoint index")
    ax2.set_ylabel("min distance, any arm point → obstacle center [m]")
    ax2.set_title("Whole-arm obstacle clearance")
    ax2.legend(fontsize=7)
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

        # The obstacle's drawn size matches its physical radius; the extra
        # LINK_RADIUS the optimizer keeps clear of is the robot's own
        # (approximate) body, not part of the obstacle itself.
        p.createMultiBody(
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, radius=OBSTACLE_RADIUS,
                rgbaColor=[1.0, 0.55, 0.0, 0.65], physicsClientId=client,
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
    robot = proc.robot
    arm_joint_names = [j.name for j in robot.actuated_joints if j.name != "panda_finger_joint1"]

    print("\n📏 Building the naive straight-line joint trajectory...")
    naive = naive_trajectory()
    ee_naive = ee_path(serial, naive)
    obstacle = ee_naive[N_WAYPOINTS // 2].copy()
    print(f"   Obstacle placed at the naive path's midpoint: {obstacle.round(3)}")
    print(f"   Required clearance = link radius ({LINK_RADIUS} m) + obstacle radius "
          f"({OBSTACLE_RADIUS} m) + pad ({CLEARANCE_PAD} m) = {REQUIRED_CLEARANCE:.2f} m")

    print("\n∂ Optimizing with jax.value_and_grad over the whole kinematic chain...")
    optimized, loss_history, meta = optimize_trajectory(serial, robot, arm_joint_names, obstacle)
    ee_opt = ee_path(serial, optimized)

    naive_clearance = whole_arm_min_clearance(naive, meta, obstacle)
    opt_clearance = whole_arm_min_clearance(optimized, meta, obstacle)

    print(f"\n📊 Minimum whole-arm clearance to obstacle (any link point, not just the fingertip):")
    print(f"   naive path:      {naive_clearance.min() * 100:5.1f} cm  "
          f"(< {REQUIRED_CLEARANCE * 100:.0f} cm required — the arm collides)")
    print(f"   optimized path:  {opt_clearance.min() * 100:5.1f} cm  "
          f"(>= {REQUIRED_CLEARANCE * 100:.0f} cm required)")

    print("\n📈 Rendering the comparison figure...")
    render_comparison_figure(
        ee_naive, ee_opt, naive_clearance, opt_clearance, obstacle, loss_history,
        naive, optimized, meta, OUTPUT_DIR / "differentiable_reach_showcase.png",
    )

    print("\n🎬 Rendering the optimized motion as a GIF...")
    render_gif(optimized, obstacle, OUTPUT_DIR / "differentiable_reach_showcase.gif")

    print("\n" + "=" * 72)
    print("✅ Showcase complete.")
    print("📚 This objective — 'keep every sampled point along the arm outside a margin,")
    print("   stay smooth' — has no closed-form Jacobian anywhere in this script. jax.grad")
    print("   built it from the trace of ordinary ManipulaPy forward-kinematics math. Swap")
    print("   in torch.autograd, or any other differentiable objective (energy, jerk, a")
    print("   learned cost), and the same pattern applies — see")
    print("   notebooks/12_differentiable_robotics.ipynb for the full differentiability")
    print("   contract (what's guaranteed, what isn't, and why).")
    print("=" * 72)


if __name__ == "__main__":
    main()
