#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced Collision Avoidance Demo - ManipulaPy

This demo showcases ManipulaPy's potential-field collision avoidance on a real
6-DOF xArm robot loaded from its bundled URDF:

- Configuration-space planning with ``potential_field.PotentialField``
  (attractive + repulsive potentials and analytic gradient descent).
- URDF-driven self-collision checking with ``potential_field.CollisionChecker``.
- Cartesian-space inspection of planned paths via the manipulator's forward
  kinematics (validates the CM-1..CM-6 SE(3) math fixes end-to-end).
- A lean random-sampling baseline for context, plus comprehensive plots and a
  text report.

It auto-detects CUDA and degrades gracefully to CPU, so it runs headless without
a GPU. All artifacts are written next to this file.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; never opens a window
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.potential_field import PotentialField, CollisionChecker
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.cuda_kernels import CUDA_AVAILABLE, check_cuda_availability
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file as XARM_URDF
except ImportError as e:  # pragma: no cover - import guard
    print(f"Error importing ManipulaPy modules: {e}")
    print("Please ensure ManipulaPy is properly installed.")
    raise SystemExit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keep all generated artifacts next to this script.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class AdvancedCollisionAvoidanceDemo:
    """
    Demonstrate potential-field collision avoidance on a real xArm6 robot.

    The robot model, joint limits and collision geometry all come from the
    bundled xArm6 URDF, so the forward kinematics and self-collision queries
    exercise the real ManipulaPy pipeline rather than a hand-rolled stand-in.
    """

    def __init__(self, save_plots: bool = True, use_gpu: Optional[bool] = None) -> None:
        """
        Args:
            save_plots: Whether to write generated plots/report to ``OUTPUT_DIR``.
            use_gpu: Force GPU usage; ``None`` auto-detects (and falls back to CPU).
        """
        self.save_plots = save_plots
        self.use_gpu = bool(use_gpu) if use_gpu is not None else check_cuda_availability()
        self.setup_robot()
        self.setup_environment()
        self.setup_planner()
        logger.info("Demo initialized - GPU available: %s", self.use_gpu)

    # ------------------------------------------------------------------ setup

    def setup_robot(self) -> None:
        """Load the xArm6 model (kinematics + dynamics) from its URDF."""
        self.urdf_path = XARM_URDF
        processor = URDFToSerialManipulator(self.urdf_path)
        self.robot = processor.serial_manipulator
        self.dynamics = processor.dynamics
        self.joint_limits = np.array(self.robot.joint_limits, dtype=float)
        self.num_joints = self.joint_limits.shape[0]
        logger.info("Loaded %d-DOF robot from %s", self.num_joints, os.path.basename(self.urdf_path))

    def setup_environment(self) -> None:
        """
        Build a configuration-space environment for potential-field planning.

        Obstacles are represented directly as joint-space configurations that the
        planner should keep away from -- this is exactly the representation
        ``PotentialField`` operates on.
        """
        rng = np.random.default_rng(0)
        # A handful of forbidden configurations scattered across the joint space.
        self.obstacle_configs = [
            self._random_config(rng) for _ in range(8)
        ]

        # ManipulaPy's potential field: attractive pull to the goal + repulsive
        # push away from each obstacle configuration within the influence radius.
        self.potential_field = PotentialField(
            attractive_gain=1.0,
            repulsive_gain=12.0,
            influence_distance=1.0,
        )

        # URDF-driven self-collision checker (builtin backend, no meshes needed).
        try:
            self.collision_checker: Optional[CollisionChecker] = CollisionChecker(
                self.urdf_path, backend="builtin", load_meshes=False
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("CollisionChecker unavailable (%s); skipping self-collision checks", exc)
            self.collision_checker = None

        logger.info("Environment ready with %d obstacle configurations", len(self.obstacle_configs))

    def setup_planner(self) -> None:
        """Create the trajectory planner (CPU fallback when CUDA is absent)."""
        try:
            self.planner: Optional[OptimizedTrajectoryPlanning] = OptimizedTrajectoryPlanning(
                serial_manipulator=self.robot,
                urdf_path=self.urdf_path,
                dynamics=self.dynamics,
                joint_limits=self.joint_limits,
                use_cuda=self.use_gpu if self.use_gpu else None,
                cuda_threshold=10_000,  # keep small runs on the CPU path
                enable_profiling=False,
            )
            logger.info("Trajectory planner created")
        except Exception as exc:
            logger.warning("Could not create trajectory planner: %s", exc)
            self.planner = None

    # --------------------------------------------------------------- helpers

    def _random_config(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniform random joint configuration within the joint limits."""
        return rng.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _clip_to_limits(self, config: np.ndarray) -> np.ndarray:
        """Clamp a configuration to the robot's joint limits."""
        return np.clip(config, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def ee_position(self, config: np.ndarray) -> np.ndarray:
        """End-effector position via the robot's forward kinematics."""
        return self.robot.forward_kinematics(config)[:3, 3]

    def is_self_collision(self, config: np.ndarray) -> bool:
        """Report whether a configuration is in self-collision (False if unknown)."""
        if self.collision_checker is None:
            return False
        try:
            return bool(self.collision_checker.check_collision(config))
        except Exception:
            return False

    # --------------------------------------------------------------- planning

    def potential_field_planning(
        self, start: np.ndarray, goal: np.ndarray, max_iterations: int = 400
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Plan a joint-space path with ManipulaPy's potential field.

        Gradient descent on ``PotentialField.compute_gradient`` drives the robot
        toward the goal while obstacle configurations push it away.

        Returns:
            ``(path, info)`` where ``path`` is a list of configurations and
            ``info`` holds convergence diagnostics.
        """
        start_time = time.time()
        path: List[np.ndarray] = [start.copy()]
        current = start.copy()

        step_size = 0.05
        tolerance = 0.05
        stuck_threshold = 1e-4
        stuck_counter = 0
        rng = np.random.default_rng(1)

        info: Dict = {
            "iterations": 0,
            "success": False,
            "final_distance": float(np.linalg.norm(goal - start)),
            "path_length": 0.0,
            "computation_time": 0.0,
            "self_collisions": 0,
        }

        for iteration in range(max_iterations):
            info["iterations"] = iteration + 1
            if np.linalg.norm(current - goal) < tolerance:
                info["success"] = True
                break

            gradient = self.potential_field.compute_gradient(
                current, goal, self.obstacle_configs
            )
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-9:
                break

            next_config = self._clip_to_limits(current - step_size * gradient / grad_norm)

            # Escape shallow local minima with an occasional random kick.
            if np.linalg.norm(next_config - current) < stuck_threshold:
                stuck_counter += 1
                if stuck_counter > 25:
                    next_config = self._clip_to_limits(
                        current + rng.normal(0.0, 0.1, self.num_joints)
                    )
                    stuck_counter = 0
            else:
                stuck_counter = 0

            if self.is_self_collision(next_config):
                info["self_collisions"] += 1

            current = next_config
            path.append(current.copy())

        info["final_distance"] = float(np.linalg.norm(current - goal))
        info["computation_time"] = time.time() - start_time
        info["path_length"] = float(
            sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))
        )
        return path, info

    def random_baseline_planning(
        self, start: np.ndarray, goal: np.ndarray, max_iterations: int = 400
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Greedy random-sampling baseline used purely for comparison.

        At each step it samples a few candidate moves and keeps the one that most
        reduces the distance to the goal -- a deliberately simple foil that shows
        the value of the analytic potential-field gradient.
        """
        start_time = time.time()
        path: List[np.ndarray] = [start.copy()]
        current = start.copy()
        rng = np.random.default_rng(2)
        step_size = 0.05
        tolerance = 0.05

        info: Dict = {
            "iterations": 0,
            "success": False,
            "final_distance": float(np.linalg.norm(goal - start)),
            "path_length": 0.0,
            "computation_time": 0.0,
        }

        for iteration in range(max_iterations):
            info["iterations"] = iteration + 1
            if np.linalg.norm(current - goal) < tolerance:
                info["success"] = True
                break

            best = current
            best_dist = np.linalg.norm(current - goal)
            for _ in range(8):
                candidate = self._clip_to_limits(
                    current + rng.normal(0.0, step_size, self.num_joints)
                )
                dist = np.linalg.norm(candidate - goal)
                if dist < best_dist:
                    best, best_dist = candidate, dist
            current = best
            path.append(current.copy())

        info["final_distance"] = float(np.linalg.norm(current - goal))
        info["computation_time"] = time.time() - start_time
        info["path_length"] = float(
            sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))
        )
        return path, info

    def run_comparative_analysis(self, num_scenarios: int = 6) -> Dict:
        """Run both planners over several random start/goal pairs and tally stats."""
        logger.info("Running comparative analysis with %d scenarios", num_scenarios)
        rng = np.random.default_rng(7)

        results: Dict = {
            "potential_field": {"success": [], "time": [], "path_length": [], "iterations": []},
            "random_baseline": {"success": [], "time": [], "path_length": [], "iterations": []},
            "scenarios": [],
        }

        for scenario in range(num_scenarios):
            start = self._random_config(rng)
            goal = self._random_config(rng)

            pf_path, pf_info = self.potential_field_planning(start, goal)
            rb_path, rb_info = self.random_baseline_planning(start, goal)

            for key, info in (("potential_field", pf_info), ("random_baseline", rb_info)):
                results[key]["success"].append(info["success"])
                results[key]["time"].append(info["computation_time"])
                results[key]["path_length"].append(info["path_length"])
                results[key]["iterations"].append(info["iterations"])

            results["scenarios"].append(
                {
                    "start": start,
                    "goal": goal,
                    "pf_path": pf_path,
                    "pf_info": pf_info,
                    "rb_path": rb_path,
                    "rb_info": rb_info,
                }
            )
            logger.info(
                "Scenario %d/%d: PF success=%s (%d it), baseline success=%s (%d it)",
                scenario + 1,
                num_scenarios,
                pf_info["success"],
                pf_info["iterations"],
                rb_info["success"],
                rb_info["iterations"],
            )

        for algorithm in ("potential_field", "random_baseline"):
            data = results[algorithm]
            if data["success"]:
                data["success_rate"] = float(np.mean(data["success"]))
                data["avg_time"] = float(np.mean(data["time"]))
                data["avg_iterations"] = float(np.mean(data["iterations"]))
                successful = [length for length, ok in zip(data["path_length"], data["success"]) if ok]
                data["avg_path_length"] = float(np.mean(successful)) if successful else 0.0

        return results

    def planned_trajectory(self, start: np.ndarray, goal: np.ndarray, N: int = 60) -> Dict:
        """Generate a smooth quintic joint trajectory between two configurations."""
        if self.planner is None:
            return {}
        return self.planner.joint_trajectory(start, goal, Tf=3.0, N=N, method=5)

    # ------------------------------------------------------------ visualization

    def _save(self, name: str) -> None:
        """Save the current figure into ``OUTPUT_DIR`` if plotting is enabled."""
        if self.save_plots:
            path = os.path.join(OUTPUT_DIR, name)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", name)

    def visualize_potential_field(
        self, config: np.ndarray, goal: np.ndarray, resolution: int = 40
    ) -> plt.Figure:
        """Plot 2D slices of the total potential through pairs of joints."""
        joint_pairs = [(0, 1), (2, 3), (4, 5)][: max(1, self.num_joints // 2)]
        fig, axes = plt.subplots(1, len(joint_pairs), figsize=(6 * len(joint_pairs), 5), squeeze=False)

        for idx, (j1, j2) in enumerate(joint_pairs):
            ax = axes[0, idx]
            r1 = np.linspace(self.joint_limits[j1, 0], self.joint_limits[j1, 1], resolution)
            r2 = np.linspace(self.joint_limits[j2, 0], self.joint_limits[j2, 1], resolution)
            J1, J2 = np.meshgrid(r1, r2)
            grid = np.zeros_like(J1)

            for i in range(resolution):
                for j in range(resolution):
                    test = config.copy()
                    test[j1] = J1[i, j]
                    test[j2] = J2[i, j]
                    grid[i, j] = self.potential_field.compute_attractive_potential(
                        test, goal
                    ) + self.potential_field.compute_repulsive_potential(
                        test, self.obstacle_configs
                    )

            contour = ax.contourf(J1, J2, grid, levels=20, cmap="viridis")
            ax.plot(config[j1], config[j2], "ro", markersize=9, label="Start")
            ax.plot(goal[j1], goal[j2], "w*", markersize=14, label="Goal")
            ax.set_xlabel(f"Joint {j1 + 1} (rad)")
            ax.set_ylabel(f"Joint {j2 + 1} (rad)")
            ax.set_title(f"Potential field: joints {j1 + 1}-{j2 + 1}")
            ax.legend(loc="upper right")
            fig.colorbar(contour, ax=ax, label="Potential")

        fig.tight_layout()
        self._save("potential_field_visualization.png")
        return fig

    def visualize_trajectory(self, scenario: Dict, smooth_traj: Dict) -> plt.Figure:
        """Show planned paths in joint space and the Cartesian end-effector path."""
        fig = plt.figure(figsize=(15, 5))

        # End-effector path (Cartesian) for the potential-field solution.
        ax1 = fig.add_subplot(131, projection="3d")
        pf_ee = np.array([self.ee_position(c) for c in scenario["pf_path"]])
        ax1.plot(pf_ee[:, 0], pf_ee[:, 1], pf_ee[:, 2], "b-", linewidth=2, label="Potential field")
        ax1.scatter(*pf_ee[0], color="green", s=70, label="Start")
        ax1.scatter(*pf_ee[-1], color="red", s=70, label="End")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("End-effector path (FK)")
        ax1.legend()

        # Joint-space paths: potential field vs baseline.
        ax2 = fig.add_subplot(132)
        pf_arr = np.array(scenario["pf_path"])
        rb_arr = np.array(scenario["rb_path"])
        for joint in range(min(3, self.num_joints)):
            ax2.plot(pf_arr[:, joint], "-", alpha=0.8, label=f"PF J{joint + 1}")
            ax2.plot(rb_arr[:, joint], "--", alpha=0.6, label=f"Base J{joint + 1}")
        ax2.set_xlabel("Path point")
        ax2.set_ylabel("Joint angle (rad)")
        ax2.set_title("Joint-space paths")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Smooth quintic trajectory from the planner (real positions output).
        ax3 = fig.add_subplot(133)
        if smooth_traj:
            positions = smooth_traj["positions"]
            for joint in range(min(3, self.num_joints)):
                ax3.plot(positions[:, joint], label=f"Joint {joint + 1}")
            ax3.set_title("Quintic joint trajectory")
        else:
            ax3.set_title("Quintic joint trajectory (planner unavailable)")
        ax3.set_xlabel("Sample")
        ax3.set_ylabel("Joint angle (rad)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save("trajectory_comparison.png")
        return fig

    def visualize_performance(self, results: Dict) -> plt.Figure:
        """Summarize success rate, timing and convergence across scenarios."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        algorithms = ["Potential Field", "Random Baseline"]
        keys = ["potential_field", "random_baseline"]

        rates = [results[k].get("success_rate", 0.0) * 100 for k in keys]
        bars = axes[0].bar(algorithms, rates, color=["steelblue", "indianred"], alpha=0.8)
        axes[0].set_ylabel("Success rate (%)")
        axes[0].set_title("Planning success rate")
        axes[0].set_ylim(0, 100)
        for bar, rate in zip(bars, rates):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.0f}%", ha="center")

        axes[1].boxplot([results[k]["iterations"] for k in keys], tick_labels=algorithms)
        axes[1].set_ylabel("Iterations to converge")
        axes[1].set_title("Iteration count")
        axes[1].grid(True, alpha=0.3)

        # Convergence of distance-to-goal for the first scenario.
        if results["scenarios"]:
            sc = results["scenarios"][0]
            goal = sc["goal"]
            pf_d = [np.linalg.norm(c - goal) for c in sc["pf_path"]]
            rb_d = [np.linalg.norm(c - goal) for c in sc["rb_path"]]
            axes[2].plot(pf_d, "b-", label="Potential field", linewidth=2)
            axes[2].plot(rb_d, "r--", label="Random baseline", linewidth=2)
            axes[2].set_yscale("log")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Distance to goal")
        axes[2].set_title("Convergence (scenario 1)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        self._save("performance_statistics.png")
        return fig

    # ------------------------------------------------------------------- report

    def generate_report(self, results: Dict, smooth_traj: Dict) -> str:
        """Build and optionally save a human-readable summary report."""
        pf = results["potential_field"]
        rb = results["random_baseline"]
        lines = [
            "=" * 64,
            "        ADVANCED COLLISION AVOIDANCE ANALYSIS REPORT",
            "=" * 64,
            "",
            "1. SYSTEM CONFIGURATION",
            "-----------------------",
            f"Robot: {os.path.basename(self.urdf_path)} ({self.num_joints}-DOF)",
            f"CUDA available: {CUDA_AVAILABLE}",
            f"Obstacle configurations: {len(self.obstacle_configs)}",
            "",
            "2. POTENTIAL-FIELD vs BASELINE",
            "------------------------------",
            f"Scenarios tested: {len(results['scenarios'])}",
            "",
            "Potential field (ManipulaPy.potential_field.PotentialField):",
            f"  Success rate:   {pf.get('success_rate', 0.0) * 100:.1f}%",
            f"  Avg iterations: {pf.get('avg_iterations', 0.0):.1f}",
            f"  Avg time:       {pf.get('avg_time', 0.0) * 1000:.2f} ms",
            f"  Avg path len:   {pf.get('avg_path_length', 0.0):.3f}",
            "",
            "Random baseline:",
            f"  Success rate:   {rb.get('success_rate', 0.0) * 100:.1f}%",
            f"  Avg iterations: {rb.get('avg_iterations', 0.0):.1f}",
            f"  Avg time:       {rb.get('avg_time', 0.0) * 1000:.2f} ms",
            "",
            "3. SMOOTH TRAJECTORY (quintic)",
            "------------------------------",
        ]
        if smooth_traj:
            lines.append(f"Samples: {smooth_traj['positions'].shape[0]}")
            lines.append(f"Peak |velocity|: {np.abs(smooth_traj['velocities']).max():.4f} rad/s")
            lines.append(f"Peak |accel|:    {np.abs(smooth_traj['accelerations']).max():.4f} rad/s^2")
        else:
            lines.append("Planner unavailable; trajectory skipped.")
        lines += [
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 64,
        ]
        report = "\n".join(lines)

        if self.save_plots:
            path = os.path.join(OUTPUT_DIR, "collision_avoidance_analysis_report.txt")
            with open(path, "w") as fh:
                fh.write(report + "\n")
            logger.info("Saved collision_avoidance_analysis_report.txt")
        return report


def main() -> None:
    """Run the full collision-avoidance demonstration end to end."""
    print("=" * 70)
    print("   MANIPULAPY ADVANCED COLLISION AVOIDANCE DEMONSTRATION")
    print("=" * 70)

    demo = AdvancedCollisionAvoidanceDemo(save_plots=True, use_gpu=None)

    print("\n1. Running comparative planning analysis...")
    results = demo.run_comparative_analysis(num_scenarios=6)

    scenario = results["scenarios"][0]
    print("\n2. Generating a smooth quintic trajectory for scenario 1...")
    smooth_traj = demo.planned_trajectory(scenario["start"], scenario["goal"])

    print("\n3. Producing visualizations...")
    demo.visualize_potential_field(scenario["start"], scenario["goal"])
    demo.visualize_trajectory(scenario, smooth_traj)
    demo.visualize_performance(results)
    plt.close("all")

    print("\n4. Writing report...")
    report = demo.generate_report(results, smooth_traj)
    print(report)

    print("\nGenerated files:")
    for name in (
        "potential_field_visualization.png",
        "trajectory_comparison.png",
        "performance_statistics.png",
        "collision_avoidance_analysis_report.txt",
    ):
        ok = os.path.exists(os.path.join(OUTPUT_DIR, name))
        print(f"  [{'x' if ok else ' '}] {name}")

    if demo.planner is not None:
        try:
            demo.planner.cleanup_gpu_memory()
        except Exception:
            pass

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
