#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Intermediate Control Comparison Demo - ManipulaPy

This demo puts three of ManipulaPy's control laws head-to-head on the *same*
joint-space tracking task and quantifies how they differ:

- PID control                  (model-free feedback)
- Computed-torque control      (full inverse-dynamics feedback linearization)
- Feedforward + PID            (inverse-dynamics feedforward with a PID corrector)

All three drive the closed loop through the robot's real rigid-body dynamics
(`ManipulatorDynamics.forward_dynamics`), so the comparison reflects the actual
nonlinear, gravity-loaded plant rather than a toy double integrator. The demo
auto-tunes PID gains with the library's Ziegler-Nichols helper, runs each loop,
prints a metrics table (tracking RMSE, settling time, overshoot, steady-state
error, control effort), and saves comparison plots next to this script.

The example is CUDA-aware but degrades gracefully to CPU: no GPU is required.

Usage:
    python control_comparison_intermediate_demo.py

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
except ImportError as exc:  # pragma: no cover - import guard
    print(f"Error importing ManipulaPy modules: {exc}")
    print("Please ensure ManipulaPy is properly installed.")
    raise SystemExit(1)

# CUDA is optional and only used opportunistically; never required.
try:
    from ManipulaPy.cuda_kernels import check_cuda_availability

    CUDA_AVAILABLE = bool(check_cuda_availability())
except Exception:  # pragma: no cover - defensive
    CUDA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
GRAVITY = np.array([0.0, 0.0, -9.81])


class ControlComparisonDemo:
    """Compare PID, computed-torque, and feedforward+PID control on one task."""

    def __init__(self) -> None:
        """Load the XArm model and build the controller + simulation horizon."""
        logger.info("Loading XArm6 model from built-in URDF...")
        processor = URDFToSerialManipulator(str(urdf_file))
        self.robot = processor.serial_manipulator
        self.dynamics = processor.dynamics
        self.controller = ManipulatorController(self.dynamics)
        self.n_joints = int(self.dynamics.S_list.shape[1])

        # Simulation horizon: kept small so the CPU run stays well under a minute
        # while still resolving the transient response of each controller.
        self.dt = 0.01
        self.substeps = 4  # plant integration sub-steps per control update
        self.duration = 2.0
        self.steps = int(self.duration / self.dt)
        self.time = np.arange(self.steps) * self.dt

        # Tracking task: hold at the home pose, then step each joint to a target.
        self.theta_start = np.zeros(self.n_joints)
        self.theta_goal = np.array([0.5, -0.4, 0.6, -0.3, 0.4, -0.5])[: self.n_joints]

        logger.info("Model ready: %d-DOF arm, %d sim steps @ dt=%.3fs",
                    self.n_joints, self.steps, self.dt)

    # ------------------------------------------------------------------ tuning
    def report_ziegler_nichols(self) -> None:
        """Run the library's relay/Ziegler-Nichols helper for reference.

        The ultimate-gain search and ``tune_controller`` are showcased here as an
        informational starting point. For a 6-DOF gravity-loaded arm the relay
        estimate alone is conservative, so the comparison below uses the shared,
        well-conditioned feedback gains from :meth:`feedback_gains` to keep every
        strategy stable and the contest fair.
        """
        logger.info("Probing ultimate gain/period (Ziegler-Nichols helper)...")
        try:
            Ku, Tu, _, _ = self.controller.find_ultimate_gain_and_period(
                self.theta_start.copy(), self.theta_goal.copy(), self.dt, max_steps=400
            )
            Kp, Ki, Kd = self.controller.tune_controller(Ku, Tu, kind="PID")
            logger.info("Z-N suggestion -> Ku=%.4g Tu=%.4g | Kp=%.4g Ki=%.4g Kd=%.4g",
                        float(np.ravel(Ku)[0]), float(np.ravel(Tu)[0]),
                        float(np.ravel(Kp)[0]), float(np.ravel(Ki)[0]),
                        float(np.ravel(Kd)[0]))
        except Exception as exc:  # pragma: no cover - plant-dependent
            logger.warning("Ziegler-Nichols probe skipped (%s)", exc)

    def feedback_gains(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared per-joint PID feedback gains used by every controller.

        The XArm6 inertia matrix is strongly anisotropic (the wrist joints carry
        a thousand-fold smaller effective inertia than the shoulder). A single
        flat gain would make the light joints violently underdamped, so the gains
        are scaled by each joint's diagonal inertia to target a comparable
        closed-loop bandwidth/damping across the arm. This is exactly the kind of
        plant knowledge that computed-torque control captures *automatically*.

        Returns:
            (Kp, Ki, Kd) gain vectors sized to the arm's DOF.
        """
        m = np.clip(np.diag(self.dynamics.mass_matrix(self.theta_start)), 1e-3, None)
        wn = 6.0  # target natural frequency (rad/s)
        zeta = 1.0  # critically damped
        Kp = m * wn ** 2
        Kd = 2.0 * zeta * m * wn
        Ki = 0.1 * Kp
        return Kp, Ki, Kd

    # ------------------------------------------------------------- simulation
    def _step_plant(
        self, theta: np.ndarray, dtheta: np.ndarray, tau: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance the rigid-body plant one control step (held torque).

        The torque is held constant over ``self.substeps`` semi-implicit Euler
        sub-steps; the finer integration keeps the explicit scheme stable for a
        stiff, gravity-loaded manipulator without shrinking the control rate.

        Args:
            theta: Current joint positions.
            dtheta: Current joint velocities.
            tau: Applied joint torques (zero-order held across sub-steps).

        Returns:
            Updated (theta, dtheta).
        """
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            ddtheta = self.dynamics.forward_dynamics(
                theta, dtheta, tau, GRAVITY, np.zeros(6)
            )
            dtheta = dtheta + ddtheta * h
            theta = theta + dtheta * h
        return theta, dtheta

    def run_controller(
        self, name: str, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Simulate one closed loop for the named control strategy.

        Args:
            name: One of ``"PID"``, ``"Computed-Torque"``, ``"Feedforward+PID"``.
            Kp, Ki, Kd: Feedback gain vectors.

        Returns:
            Dict with ``positions`` (steps x n), ``torques`` (steps x n), and
            ``error`` (steps x n) trajectories.
        """
        theta = self.theta_start.copy()
        dtheta = np.zeros(self.n_joints)
        thetad = self.theta_goal
        dthetad = np.zeros(self.n_joints)
        ddthetad = np.zeros(self.n_joints)

        # Reset controller integral state between runs.
        self.controller.eint = np.zeros(self.n_joints)

        positions = np.zeros((self.steps, self.n_joints))
        torques = np.zeros((self.steps, self.n_joints))

        for k in range(self.steps):
            if name == "PID":
                tau = self.controller.pid_control(
                    thetad, dthetad, theta, dtheta, self.dt, Kp, Ki, Kd,
                    i_clamp=5.0,
                )
            elif name == "Computed-Torque":
                tau = self.controller.computed_torque_control(
                    thetad, dthetad, ddthetad, theta, dtheta,
                    GRAVITY, self.dt, Kp, Ki, Kd, i_clamp=5.0,
                )
            elif name == "Feedforward+PID":
                tau_ff = self.controller.feedforward_control(
                    thetad, dthetad, ddthetad, GRAVITY, np.zeros(6)
                )
                tau_fb = self.controller.pid_control(
                    thetad, dthetad, theta, dtheta, self.dt, Kp, Ki, Kd,
                    i_clamp=5.0,
                )
                tau = tau_ff + tau_fb
            else:  # pragma: no cover - guarded by caller
                raise ValueError(f"unknown controller {name!r}")

            positions[k] = theta
            torques[k] = tau
            theta, dtheta = self._step_plant(theta, dtheta, tau)

        return {
            "positions": positions,
            "torques": torques,
            "error": positions - thetad,
        }

    # ---------------------------------------------------------------- metrics
    def compute_metrics(self, result: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Aggregate tracking-quality metrics for a simulated run.

        Uses the controller's own settling/overshoot/steady-state helpers on the
        worst-tracking joint (largest commanded step), plus an effort/RMSE
        summary across all joints.
        """
        positions = result["positions"]
        torques = result["torques"]
        err = result["error"]

        # Pick the joint with the largest commanded motion for transient metrics.
        steps_size = np.abs(self.theta_goal - self.theta_start)
        j = int(np.argmax(steps_size))
        resp = positions[:, j]
        set_point = float(self.theta_goal[j])

        rmse = float(np.sqrt(np.mean(err ** 2)))
        final_err = float(np.linalg.norm(positions[-1] - self.theta_goal))
        effort = float(np.sum(np.abs(torques)) * self.dt)

        settling = self.controller.calculate_settling_time(
            self.time, resp, set_point, tolerance=0.02
        )
        overshoot = self.controller.calculate_percent_overshoot(resp, set_point)
        ss_error = self.controller.calculate_steady_state_error(resp, set_point)

        return {
            "RMSE (rad)": rmse,
            "Final |err| (rad)": final_err,
            "Settling (s)": float(settling) if np.isfinite(settling) else float("nan"),
            "Overshoot (%)": float(overshoot),
            "SS error (rad)": float(ss_error),
            "Effort (N*m*s)": effort,
        }

    # ----------------------------------------------------------------- output
    def print_table(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Print a side-by-side metrics table for all controllers."""
        names = list(metrics.keys())
        rows = list(next(iter(metrics.values())).keys())
        col_w = max(len(n) for n in names) + 2
        head = f"{'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in names)
        print("\n" + "=" * len(head))
        print("CONTROL STRATEGY COMPARISON")
        print("=" * len(head))
        print(head)
        print("-" * len(head))
        for r in rows:
            line = f"{r:<20}"
            for n in names:
                line += f"{metrics[n][r]:>{col_w}.4f}"
            print(line)
        print("=" * len(head))

    def plot_comparison(self, results: Dict[str, Dict[str, np.ndarray]]) -> Path:
        """Plot tracking response and control effort for the dominant joint."""
        steps_size = np.abs(self.theta_goal - self.theta_start)
        j = int(np.argmax(steps_size))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for name, res in results.items():
            axes[0].plot(self.time, res["positions"][:, j], label=name, lw=2)
            axes[1].plot(self.time, res["torques"][:, j], label=name, lw=2)

        axes[0].axhline(self.theta_goal[j], color="k", ls="--", lw=1, label="setpoint")
        axes[0].set_title(f"Joint {j + 1} tracking response")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("position (rad)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title(f"Joint {j + 1} control torque")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("torque (N*m)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.suptitle("PID vs Computed-Torque vs Feedforward+PID", fontweight="bold")
        fig.tight_layout()
        out = SCRIPT_DIR / "control_comparison.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    def plot_error_bars(self, metrics: Dict[str, Dict[str, float]]) -> Path:
        """Bar chart of RMSE and control effort across strategies."""
        names = list(metrics.keys())
        rmse = [metrics[n]["RMSE (rad)"] for n in names]
        effort = [metrics[n]["Effort (N*m*s)"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(names, rmse, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[0].set_title("Tracking RMSE (lower is better)")
        axes[0].set_ylabel("rad")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(names, effort, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[1].set_title("Total control effort (lower is better)")
        axes[1].set_ylabel("N*m*s")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out = SCRIPT_DIR / "control_metrics.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out


def main() -> None:
    """Run the full PID / computed-torque / feedforward comparison."""
    print("=== ManipulaPy: Intermediate Control Comparison Demo ===")
    print(f"CUDA available: {CUDA_AVAILABLE} (CPU fallback is always safe)\n")

    demo = ControlComparisonDemo()
    demo.report_ziegler_nichols()
    Kp, Ki, Kd = demo.feedback_gains()

    strategies = ["PID", "Computed-Torque", "Feedforward+PID"]
    results: Dict[str, Dict[str, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name in strategies:
        logger.info("Simulating %s controller...", name)
        res = demo.run_controller(name, Kp, Ki, Kd)
        results[name] = res
        metrics[name] = demo.compute_metrics(res)

    demo.print_table(metrics)

    p1 = demo.plot_comparison(results)
    p2 = demo.plot_error_bars(metrics)
    print(f"\nSaved plots:\n  {p1}\n  {p2}")

    # Highlight the model-based advantage the comparison is meant to reveal.
    best = min(metrics, key=lambda n: metrics[n]["RMSE (rad)"])
    print(f"\nLowest tracking RMSE: {best} "
          f"({metrics[best]['RMSE (rad)']:.4f} rad)")
    print("Demo complete.")


if __name__ == "__main__":
    main()
