#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced Optimal Control Demo: Trajectory-Tracking Control Showdown - ManipulaPy

This demo puts ManipulaPy's control suite to work on a realistic
trajectory-tracking task and compares several strategies head-to-head:

- Computed-torque control (full model-based feedback linearization)
- PD + inverse-dynamics feedforward control
- PID control whose Kd/Ki ratios come from the library's Ziegler-Nichols helper
- Gravity-compensated PD (a lightweight model-free baseline)

Every controller drives the same simulated xArm6 plant, integrated with the
library's own ``ManipulatorDynamics.forward_dynamics``. A constant joint-space
disturbance torque is injected into the plant but hidden from the controllers,
so the laws with integral / feedforward action can demonstrate their
disturbance-rejection advantage. Gains are scaled by the manipulator inertia
(the home-configuration mass matrix is badly conditioned, cond ~6e3), which
keeps the light wrist joints well-behaved.

The demo reports per-controller RMS / max / steady-state tracking error plus an
IAE score, names the winner on each metric, and saves comparison plots next to
this file. It auto-detects CUDA and degrades gracefully to the CPU fallback
path, so it runs cleanly on machines without a GPU.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; never opens a window
import matplotlib.pyplot as plt
import numpy as np

from ManipulaPy.control import ManipulatorController
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
from ManipulaPy.urdf_processor import URDFToSerialManipulator

try:
    from ManipulaPy.cuda_kernels import check_cuda_availability
except Exception:  # pragma: no cover - cuda_kernels import is optional here

    def check_cuda_availability() -> bool:
        """Fallback used when cuda_kernels cannot be imported."""
        return False


HERE = os.path.dirname(os.path.abspath(__file__))
GRAVITY = np.array([0.0, 0.0, -9.81])

# Closed-loop bandwidth / damping used to derive inertia-scaled gains.
NATURAL_FREQ = 8.0  # rad/s
DAMPING = 1.0  # critically damped


def build_robot() -> Tuple[object, object, np.ndarray, np.ndarray]:
    """
    Load the bundled xArm6 model and return its kinematics, dynamics, and limits.

    Returns:
        Tuple of (serial_manipulator, dynamics, joint_limits, torque_limits).
    """
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file

    processor = URDFToSerialManipulator(urdf_file)
    robot = processor.serial_manipulator
    dynamics = processor.dynamics

    joint_limits = np.asarray(robot.joint_limits, dtype=float)
    torque_limits = np.tile(np.array([-150.0, 150.0]), (joint_limits.shape[0], 1))
    return robot, dynamics, joint_limits, torque_limits


def build_reference(
    robot: object,
    dynamics: object,
    joint_limits: np.ndarray,
    torque_limits: np.ndarray,
    Tf: float,
    N: int,
) -> Dict[str, np.ndarray]:
    """
    Generate a smooth quintic joint-space reference trajectory to track.

    Uses ``OptimizedTrajectoryPlanning`` with ``use_cuda=None`` so it
    auto-selects the GPU when present and falls back to CPU otherwise.

    Returns:
        Dict with keys ``positions``, ``velocities``, ``accelerations``,
        each shaped ``(N, n_joints)``.
    """
    planner = OptimizedTrajectoryPlanning(
        robot, "", dynamics, joint_limits, torque_limits, use_cuda=None
    )
    theta_start = np.zeros(joint_limits.shape[0])
    theta_end = np.array([0.6, -0.5, 0.4, 0.3, -0.4, 0.2])
    # method=5 -> quintic time scaling (zero vel/acc at the endpoints).
    return planner.joint_trajectory(theta_start, theta_end, Tf, N, 5)


def design_gains(
    controller: ManipulatorController, inertia: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build inertia-scaled gain sets for every controller under test.

    PD-style gains follow a critically damped second-order target,
    ``Kp = J * wn^2`` and ``Kd = J * 2*zeta*wn``, where ``J`` is the per-joint
    inertia from the home-configuration mass matrix. The PID gain *ratios* are
    derived from the library's Ziegler-Nichols helper and then scaled by the
    same inertia so the badly conditioned wrist joints stay stable.

    Returns:
        Dict mapping controller name to its ``Kp``/``Kd`` (and optional ``Ki``)
        gain vectors.
    """
    Kp_pd = inertia * NATURAL_FREQ**2
    Kd_pd = inertia * 2.0 * DAMPING * NATURAL_FREQ

    # Ziegler-Nichols supplies textbook Kd/Ki *ratios* from an ultimate
    # gain/period pair; we keep its proportions and re-scale per joint.
    Ku, Tu = 1.0, 2.0 * np.pi / NATURAL_FREQ
    zn_kp, zn_ki, zn_kd = controller.ziegler_nichols_tuning(Ku, Tu, kind="PID")
    ki_ratio = zn_ki / zn_kp  # integral-to-proportional ratio
    kd_ratio = zn_kd / zn_kp  # derivative-to-proportional ratio
    print(
        f"  Ziegler-Nichols PID ratios (per unit Kp): "
        f"Ki/Kp={ki_ratio:.3f}, Kd/Kp={kd_ratio:.3f}"
    )

    Kp_pid = inertia * NATURAL_FREQ**2
    Ki_pid = Kp_pid * ki_ratio
    Kd_pid = Kp_pid * kd_ratio

    return {
        "Computed Torque": {
            "Kp": np.full_like(inertia, 120.0),
            "Kd": np.full_like(inertia, 22.0),
            "Ki": np.full_like(inertia, 8.0),
        },
        "PD + Feedforward": {"Kp": Kp_pd, "Kd": Kd_pd},
        "PID (Ziegler-Nichols)": {"Kp": Kp_pid, "Kd": Kd_pid, "Ki": Ki_pid},
        "Gravity-comp PD": {"Kp": Kp_pd, "Kd": Kd_pd},
    }


def simulate_controller(
    name: str,
    dynamics: object,
    controller: ManipulatorController,
    reference: Dict[str, np.ndarray],
    dt: float,
    gains: Dict[str, np.ndarray],
    disturbance_torque: np.ndarray,
    substeps: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Closed-loop simulation of one control law tracking the reference.

    The plant is the library's own ``forward_dynamics`` integrated with a
    semi-implicit (symplectic) Euler scheme. To stay stable with stiff feedback,
    the plant is advanced with ``substeps`` fine integration steps per control
    update (zero-order hold on the torque). A constant joint-space disturbance
    torque is added to the plant but hidden from each controller, so the laws
    with integral or feedforward action must reject it through feedback.

    Args:
        name: Controller label (selects the control law below).
        dynamics: ManipulatorDynamics instance (the plant + the model).
        controller: ManipulatorController bound to ``dynamics``.
        reference: Reference trajectory dict (positions/velocities/accelerations).
        dt: Control update step.
        gains: Dict of gain vectors (``Kp``, ``Kd`` and optionally ``Ki``).
        disturbance_torque: Constant joint-space torque applied to the plant.
        substeps: Number of fine plant-integration steps per control update.

    Returns:
        Dict with ``theta`` (achieved positions) and ``tau`` (commanded torques),
        each shaped ``(N, n_joints)``.
    """
    pos = reference["positions"]
    vel = reference["velocities"]
    acc = reference["accelerations"]
    N, n = pos.shape

    theta = pos[0].copy()
    dtheta = np.zeros(n)
    controller.eint = np.zeros(n)  # reset integral memory between runs

    theta_hist = np.zeros((N, n))
    tau_hist = np.zeros((N, n))

    Kp = gains["Kp"]
    Kd = gains["Kd"]
    Ki = gains.get("Ki")
    h = dt / substeps

    for k in range(N):
        thetad, dthetad, ddthetad = pos[k], vel[k], acc[k]

        if name == "Computed Torque":
            tau = controller.computed_torque_control(
                thetad, dthetad, ddthetad, theta, dtheta, GRAVITY, dt, Kp, Ki, Kd
            )
        elif name == "PD + Feedforward":
            tau = controller.pd_feedforward_control(
                thetad, dthetad, ddthetad, theta, dtheta, Kp, Kd, GRAVITY, np.zeros(6)
            )
        elif name == "PID (Ziegler-Nichols)":
            tau = controller.pid_control(
                thetad, dthetad, theta, dtheta, dt, Kp, Ki, Kd
            )
            # PID is model-free; add gravity compensation so it can hold pose.
            tau = tau + dynamics.gravity_forces(theta, GRAVITY)
        else:  # Gravity-compensated PD baseline
            tau = controller.pd_control(thetad, dthetad, theta, dtheta, Kp, Kd)
            tau = tau + dynamics.gravity_forces(theta, GRAVITY)

        # Plant step: integrate forward dynamics with the hidden disturbance
        # torque, using fine sub-steps (zero-order hold on the command).
        for _ in range(substeps):
            ddtheta = dynamics.forward_dynamics(
                theta, dtheta, tau + disturbance_torque, GRAVITY, np.zeros(6)
            )
            dtheta = dtheta + ddtheta * h
            theta = theta + dtheta * h

        theta_hist[k] = theta
        tau_hist[k] = tau

    final_err = np.linalg.norm(theta_hist[-1] - pos[-1])
    print(f"  {name:<24s} steady-state error = {final_err:.4f} rad")
    return {"theta": theta_hist, "tau": tau_hist}


def tracking_metrics(
    achieved: np.ndarray, reference: np.ndarray, dt: float
) -> Dict[str, float]:
    """
    Compute summary tracking-error metrics for one controller.

    Returns:
        Dict with RMS error, max joint error, final (steady-state) error norm,
        and the integral of the absolute error (IAE) summed across joints.
    """
    err = achieved - reference
    rms = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))
    final_norm = float(np.linalg.norm(err[-1]))
    iae = float(np.sum(np.abs(err)) * dt)
    return {"rms": rms, "max": max_err, "final": final_norm, "iae": iae}


def plot_results(
    reference: Dict[str, np.ndarray],
    runs: Dict[str, Dict[str, np.ndarray]],
    metrics: Dict[str, Dict[str, float]],
    t: np.ndarray,
) -> Tuple[str, str]:
    """
    Save tracking-comparison and error-metric figures next to this script.

    Returns:
        Tuple of absolute paths to the two saved PNG files.
    """
    pos = reference["positions"]

    # --- Figure 1: joint-1 tracking + error-norm time series -----------------
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    ax_top.plot(t, pos[:, 0], "k--", lw=2, label="Reference")
    for name, run in runs.items():
        ax_top.plot(t, run["theta"][:, 0], lw=1.5, label=name)
    ax_top.set_title("Joint 1 trajectory tracking (constant disturbance torque)")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Joint angle (rad)")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    for name, run in runs.items():
        err_norm = np.linalg.norm(run["theta"] - pos, axis=1)
        ax_bot.plot(t, err_norm, lw=1.5, label=name)
    ax_bot.set_title("Tracking error norm over time")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("||theta - theta_d|| (rad)")
    ax_bot.legend(loc="best", fontsize=8)
    ax_bot.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(HERE, "optimal_control_tracking.png")
    fig.savefig(path1, dpi=110)
    plt.close(fig)

    # --- Figure 2: RMS / max / final / IAE bar comparison --------------------
    names = list(metrics.keys())
    rms = [metrics[k]["rms"] for k in names]
    mx = [metrics[k]["max"] for k in names]
    final = [metrics[k]["final"] for k in names]
    iae = [metrics[k]["iae"] for k in names]
    x = np.arange(len(names))
    w = 0.2

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, rms, w, label="RMS error (rad)")
    ax.bar(x - 0.5 * w, mx, w, label="Max error (rad)")
    ax.bar(x + 0.5 * w, final, w, label="Steady-state error (rad)")
    ax.bar(x + 1.5 * w, iae, w, label="IAE (rad·s)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Error metric")
    ax.set_title("Controller tracking-accuracy comparison")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    path2 = os.path.join(HERE, "optimal_control_error_metrics.png")
    fig2.savefig(path2, dpi=110)
    plt.close(fig2)

    return path1, path2


def main() -> None:
    """Run the trajectory-tracking control comparison end-to-end."""
    print("=== ManipulaPy: Advanced Optimal Control Demo ===")
    cuda = bool(check_cuda_availability())
    print(f"CUDA available: {cuda} (CPU fallback is used automatically when False)")

    robot, dynamics, joint_limits, torque_limits = build_robot()
    n = joint_limits.shape[0]
    print(f"Loaded xArm6 model: {n} joints")

    # Reference trajectory (keep N modest so the CPU run stays fast).
    Tf, N = 3.0, 100
    dt = Tf / (N - 1)
    t = np.linspace(0.0, Tf, N)
    reference = build_reference(robot, dynamics, joint_limits, torque_limits, Tf, N)
    print(f"Generated quintic reference: {N} samples over {Tf:.1f}s")

    controller = ManipulatorController(dynamics)

    # Per-joint inertia from the home-configuration mass matrix. The matrix is
    # badly conditioned (the wrist joint inertia is ~1e-4), so gains and the
    # disturbance are scaled by it to keep the light joints stable.
    inertia = np.diag(dynamics.mass_matrix(np.zeros(n)))
    print(f"Mass-matrix condition number: {np.linalg.cond(dynamics.mass_matrix(np.zeros(n))):.1f}")

    print("\nDesigning inertia-scaled gains:")
    gain_sets = design_gains(controller, inertia)

    # Unmodelled disturbance: a constant joint-space torque (e.g. an unmodelled
    # payload / friction bias), scaled by inertia so it perturbs every joint by
    # a comparable acceleration.
    disturbance = inertia * np.array([6.0, -5.0, 4.0, 3.0, -3.0, 2.0])
    print(f"\nInjected constant disturbance torque (N·m): {np.round(disturbance, 3).tolist()}")

    print("\nRunning closed-loop simulations:")
    runs: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for name, gains in gain_sets.items():
        run = simulate_controller(
            name, dynamics, controller, reference, dt, gains, disturbance
        )
        runs[name] = run
        metrics[name] = tracking_metrics(run["theta"], reference["positions"], dt)

    # --- Report --------------------------------------------------------------
    print("\n--- Tracking accuracy summary ---")
    header = (
        f"{'Controller':<24s}{'RMS':>10s}{'Max':>10s}"
        f"{'Steady':>10s}{'IAE':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name in runs:
        m = metrics[name]
        print(
            f"{name:<24s}{m['rms']:>10.4f}{m['max']:>10.4f}"
            f"{m['final']:>10.4f}{m['iae']:>10.4f}"
        )

    best_rms = min(metrics, key=lambda k: metrics[k]["rms"])
    best_ss = min(metrics, key=lambda k: metrics[k]["final"])
    print(f"\nBest RMS tracking accuracy:   {best_rms} ({metrics[best_rms]['rms']:.4f} rad)")
    print(
        f"Best disturbance rejection:   {best_ss} "
        f"(steady-state {metrics[best_ss]['final']:.4f} rad)"
    )

    p1, p2 = plot_results(reference, runs, metrics, t)
    print(f"\nSaved tracking plot:       {p1}")
    print(f"Saved error-metric plot:   {p2}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
