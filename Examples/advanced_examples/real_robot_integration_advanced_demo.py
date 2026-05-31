#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced Real Robot Integration Demo - ManipulaPy

This demo walks through a realistic end-to-end integration pipeline, the same
sequence of steps you would follow when bringing up a physical manipulator:

    1. Load a URDF via ``URDFToSerialManipulator`` and build the kinematic
       (``SerialManipulator``) and dynamic (``ManipulatorDynamics``) models.
    2. Plan a smooth joint-space trajectory with ``OptimizedTrajectoryPlanning``
       (CPU fallback when CUDA is unavailable).
    3. Close a computed-torque control loop (``ManipulatorController``) around a
       forward-dynamics "virtual plant" so the controller actually has to track
       the planned reference under model dynamics.
    4. Monitor joint-limit and near-singularity safety conditions on every step.
    5. Report tracking metrics, save diagnostic plots, and optionally replay the
       commanded motion in PyBullet (DIRECT mode, skipped cleanly when PyBullet
       is unavailable).

The CUDA path is auto-detected and the example degrades gracefully to CPU, so it
runs headless without a GPU.

Usage:
    python real_robot_integration_advanced_demo.py

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.singularity import Singularity
    from ManipulaPy.cuda_kernels import CUDA_AVAILABLE, check_cuda_availability
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file as XARM_URDF
except ImportError as exc:  # pragma: no cover - import guard
    print(f"Error importing ManipulaPy modules: {exc}")
    print("Please ensure ManipulaPy is properly installed.")
    raise SystemExit(1)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Keep all artefacts next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

GRAVITY = np.array([0.0, 0.0, -9.81])


class RealRobotIntegrationDemo:
    """End-to-end model -> plan -> control -> safety integration showcase.

    The class mirrors a hardware bring-up: every stage produces an artefact the
    next stage consumes, and a safety monitor watches the commanded motion.
    """

    def __init__(self, urdf_path: str, save_plots: bool = True) -> None:
        """Load the robot models from ``urdf_path`` and build the controllers."""
        self.urdf_path = urdf_path
        self.save_plots = save_plots

        print("[1/5] Loading URDF and building kinematic + dynamic models ...")
        processor = URDFToSerialManipulator(urdf_path)
        self.robot = processor.serial_manipulator
        self.dynamics = processor.dynamics
        self.controller = ManipulatorController(self.dynamics)
        self.singularity = Singularity(self.robot)

        self.joint_limits = np.asarray(self.robot.joint_limits, dtype=float)
        self.num_joints = self.joint_limits.shape[0]
        # Conservative torque envelope used for safety clamping.
        self.torque_limits = np.column_stack(
            [-60.0 * np.ones(self.num_joints), 60.0 * np.ones(self.num_joints)]
        )

        self.planner = OptimizedTrajectoryPlanning(
            self.robot,
            urdf_path,
            self.dynamics,
            self.joint_limits,
            torque_limits=self.torque_limits,
            use_cuda=None,  # auto-detect; falls back to CPU on this box
        )
        print(f"      Robot loaded: {self.num_joints} DOF, "
              f"CUDA available: {CUDA_AVAILABLE}")

    # ------------------------------------------------------------------ #
    # Stage 2: trajectory planning
    # ------------------------------------------------------------------ #
    def plan_trajectory(
        self, theta_start: np.ndarray, theta_goal: np.ndarray, Tf: float, N: int
    ) -> Dict[str, np.ndarray]:
        """Plan a quintic joint-space trajectory between two configurations."""
        print("[2/5] Planning joint-space trajectory "
              f"(Tf={Tf}s, N={N}, quintic) ...")
        traj = self.planner.joint_trajectory(
            theta_start, theta_goal, Tf, N, method=5
        )
        # Confirm the planned path respects joint limits before we command it.
        pos = traj["positions"]
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        within = np.all((pos >= lo - 1e-6) & (pos <= hi + 1e-6))
        print(f"      Planned {pos.shape[0]} waypoints; "
              f"within joint limits: {within}")
        return traj

    # ------------------------------------------------------------------ #
    # Stage 3 + 4: closed-loop control with a forward-dynamics plant
    # ------------------------------------------------------------------ #
    def run_control_loop(
        self, traj: Dict[str, np.ndarray], dt: float
    ) -> Dict[str, np.ndarray]:
        """Track the reference with computed-torque control over a virtual plant.

        The "plant" is the robot's own forward dynamics, integrated with the
        commanded torque each step. A safety monitor clamps torques, flags
        joint-limit excursions, and detects near-singular configurations.
        """
        print("[3/5] Running computed-torque control loop with safety "
              "monitoring ...")
        qd = traj["positions"]
        dqd = traj["velocities"]
        ddqd = traj["accelerations"]
        steps = qd.shape[0]

        Kp = 120.0 * np.ones(self.num_joints)
        Ki = 1.0 * np.ones(self.num_joints)
        Kd = 28.0 * np.ones(self.num_joints)

        q = qd[0].copy()
        dq = np.zeros(self.num_joints)
        # Reset the controller's integral accumulator for a clean run.
        self.controller.eint = np.zeros(self.num_joints)

        q_hist = np.zeros((steps, self.num_joints))
        tau_hist = np.zeros((steps, self.num_joints))
        err_hist = np.zeros((steps, self.num_joints))
        cond_hist = np.zeros(steps)

        limit_violations = 0
        torque_saturations = 0
        singular_steps = 0

        for k in range(steps):
            tau = self.controller.computed_torque_control(
                qd[k], dqd[k], ddqd[k], q, dq, GRAVITY, dt, Kp, Ki, Kd
            )

            # --- safety monitor -------------------------------------- #
            tau_clamped = np.clip(
                tau, self.torque_limits[:, 0], self.torque_limits[:, 1]
            )
            if not np.allclose(tau_clamped, tau):
                torque_saturations += 1
            tau = tau_clamped

            cond = self.singularity.condition_number(q)
            cond_hist[k] = cond
            # A large Jacobian condition number means the arm is near a
            # singularity (loss of a Cartesian DOF). The zero pose is itself
            # singular, so the early steps are expected to flag here.
            if self.singularity.near_singularity_detection(q, threshold=1e3):
                singular_steps += 1

            # --- plant: integrate forward dynamics ------------------- #
            ddq = self.dynamics.forward_dynamics(
                q, dq, tau, GRAVITY, np.zeros(6)
            )
            dq = dq + ddq * dt
            q = q + dq * dt

            # Joint-limit excursion check on the realised state.
            if np.any(q < self.joint_limits[:, 0]) or np.any(
                q > self.joint_limits[:, 1]
            ):
                limit_violations += 1
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

            q_hist[k] = q
            tau_hist[k] = tau
            err_hist[k] = qd[k] - q

        final_err = err_hist[-1]
        rms_err = float(np.sqrt(np.mean(err_hist ** 2)))
        print(f"      Final position error (rad): "
              f"{np.array2string(final_err, precision=4)}")
        print(f"      RMS tracking error: {rms_err:.5f} rad")
        print(f"      Safety: torque saturations={torque_saturations}, "
              f"joint-limit excursions={limit_violations}, "
              f"near-singular steps={singular_steps}")

        return {
            "qd": qd,
            "q": q_hist,
            "tau": tau_hist,
            "err": err_hist,
            "cond": cond_hist,
            "rms_err": rms_err,
            "torque_saturations": torque_saturations,
            "limit_violations": limit_violations,
            "singular_steps": singular_steps,
        }

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def plot_results(
        self, result: Dict[str, np.ndarray], dt: float
    ) -> Optional[str]:
        """Plot reference vs. realised tracking, errors, torques, conditioning."""
        if not self.save_plots:
            return None
        steps = result["q"].shape[0]
        t = np.arange(steps) * dt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            "Real Robot Integration: Computed-Torque Tracking",
            fontsize=14, fontweight="bold",
        )

        ax = axes[0, 0]
        for j in range(self.num_joints):
            ax.plot(t, result["qd"][:, j], "--", lw=1, alpha=0.7)
            ax.plot(t, result["q"][:, j], "-", lw=1.5, label=f"J{j + 1}")
        ax.set_title("Reference (dashed) vs. realised (solid) joint angles")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [rad]")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for j in range(self.num_joints):
            ax.plot(t, result["err"][:, j], lw=1.2, label=f"J{j + 1}")
        ax.set_title("Tracking error per joint")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [rad]")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for j in range(self.num_joints):
            ax.plot(t, result["tau"][:, j], lw=1.2, label=f"J{j + 1}")
        ax.set_title("Commanded joint torques (post safety clamp)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque [N·m]")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        # The zero start pose is singular (cond -> very large); use a log axis
        # and clip the spike so the well-conditioned remainder stays readable.
        cond = np.clip(result["cond"], 1.0, 1e6)
        ax.semilogy(t, cond, color="darkred", lw=1.5)
        ax.axhline(1e3, color="gray", ls="--", lw=1, label="near-singular threshold")
        ax.set_title("Jacobian condition number (singularity proximity)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("cond(J) [log]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = os.path.join(HERE, "real_robot_integration_tracking.png")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"      Saved tracking diagnostics to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    # Stage 5: optional PyBullet replay (skipped cleanly if unavailable)
    # ------------------------------------------------------------------ #
    def try_pybullet_replay(self, traj: Dict[str, np.ndarray]) -> bool:
        """Replay the commanded trajectory in PyBullet DIRECT mode if possible.

        Returns ``True`` if the replay executed, ``False`` if it was skipped
        because PyBullet or a usable physics client was unavailable.
        """
        print("[5/5] Attempting PyBullet replay (DIRECT, headless-safe) ...")
        try:
            import pybullet as p

            client = p.connect(p.DIRECT)
            p.resetSimulation()
            p.setGravity(0.0, 0.0, -9.81)
            p.setTimeStep(0.01)
            robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
            revolute = [
                i
                for i in range(p.getNumJoints(robot_id))
                if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED
            ]
            n_dof = min(len(revolute), self.num_joints)
            # Replay a handful of planned waypoints under position control.
            for idx in np.linspace(0, traj["positions"].shape[0] - 1, 5).astype(int):
                target = traj["positions"][idx]
                for col, jidx in enumerate(revolute[:n_dof]):
                    p.setJointMotorControl2(
                        robot_id, jidx, p.POSITION_CONTROL,
                        targetPosition=float(target[col]), force=60.0,
                    )
                for _ in range(20):
                    p.stepSimulation()
            realised = np.array(
                [p.getJointState(robot_id, j)[0] for j in revolute[:n_dof]]
            )
            p.disconnect(client)
            print(f"      PyBullet replay OK over {n_dof} joints; final state "
                  f"{np.array2string(realised, precision=3)}")
            return True
        except Exception as exc:  # noqa: BLE001 - graceful skip on any failure
            print(f"      PyBullet replay skipped ({type(exc).__name__}: {exc})")
            return False


def build_motion(num_joints: int, joint_limits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return start/goal configurations safely inside the joint limits."""
    mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
    span = joint_limits[:, 1] - joint_limits[:, 0]
    theta_start = np.zeros(num_joints)
    # A modest reach: 15% of each joint's range away from the midpoint.
    theta_goal = np.clip(
        mid + 0.15 * span * np.array([1, -1, 1, -1, 1, -1][:num_joints]),
        joint_limits[:, 0] + 1e-3,
        joint_limits[:, 1] - 1e-3,
    )
    return theta_start, theta_goal


def main() -> None:
    """Run the full integration pipeline end to end."""
    print("=== ManipulaPy: Advanced Real Robot Integration Demo ===")
    if not CUDA_AVAILABLE:
        check_cuda_availability()  # surface the diagnostic once, then use CPU

    demo = RealRobotIntegrationDemo(XARM_URDF, save_plots=True)

    theta_start, theta_goal = build_motion(demo.num_joints, demo.joint_limits)

    Tf, N, dt = 3.0, 120, 3.0 / 120
    traj = demo.plan_trajectory(theta_start, theta_goal, Tf, N)
    result = demo.run_control_loop(traj, dt)

    print("[4/5] Generating diagnostics ...")
    demo.plot_results(result, dt)

    demo.try_pybullet_replay(traj)

    converged = result["rms_err"] < 0.05 and result["limit_violations"] == 0
    print("=== Pipeline complete ===")
    print(f"    Tracking converged: {converged} "
          f"(RMS error {result['rms_err']:.5f} rad)")


if __name__ == "__main__":
    main()
