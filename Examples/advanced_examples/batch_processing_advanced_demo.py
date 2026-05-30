#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Batch Processing Advanced Demo - ManipulaPy

This demo showcases large-scale batch trajectory processing and scaling
analysis through ManipulaPy's :class:`OptimizedTrajectoryPlanning` batch API:

- Vectorized batch joint-trajectory generation (``batch_joint_trajectory``)
  versus a sequential loop over ``joint_trajectory``.
- Performance- and throughput-scaling analysis across growing batch sizes.
- Statistical analysis of the generated batch (path lengths, per-joint
  velocity/acceleration profiles) computed directly from real planner output.

The planner auto-selects GPU acceleration when CUDA is available and falls
back to a CPU implementation otherwise, so this demo runs unchanged on a
headless CPU-only machine.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.cuda_kernels import CUDA_AVAILABLE, check_cuda_availability
except ImportError as e:  # pragma: no cover - import guard for standalone runs
    print(f"Error importing ManipulaPy modules: {e}")
    print("Please ensure ManipulaPy is properly installed.")
    raise SystemExit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# The planner logs an INFO line per trajectory; quiet it so the scaling sweep
# does not flood stdout with thousands of identical messages.
logging.getLogger("ManipulaPy.path_planning").setLevel(logging.WARNING)

# Keep generated artifacts next to this example, regardless of CWD.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class BatchProcessingDemo:
    """Demonstrate ManipulaPy's batch trajectory-processing strengths.

    The demo builds a 6-DOF manipulator, then drives its
    :class:`OptimizedTrajectoryPlanning` instance through batched and
    sequential trajectory generation to compare scaling behaviour and to
    analyse the resulting motion statistics.
    """

    def __init__(self, save_plots: bool = True) -> None:
        """Initialise the demo.

        Args:
            save_plots: Whether to save generated plots/report to disk.
        """
        self.save_plots = save_plots
        self.use_gpu = check_cuda_availability()
        self.num_joints = 6

        self._setup_robot()
        logger.info(
            "Batch demo initialised - GPU acceleration available: %s", self.use_gpu
        )

    # ------------------------------------------------------------------ setup
    def _setup_robot(self) -> None:
        """Build a 6-DOF serial manipulator, dynamics, and a batch planner."""
        self.joint_limits = [
            (-np.pi, np.pi),
            (-np.pi / 2, np.pi / 2),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi / 2, np.pi / 2),
            (-np.pi, np.pi),
        ]
        self.torque_limits = [
            (-100, 100),
            (-80, 80),
            (-60, 60),
            (-40, 40),
            (-30, 30),
            (-20, 20),
        ]

        link_lengths = [0.3, 0.4, 0.35, 0.2, 0.15, 0.1]
        M = np.eye(4)
        M[0, 3] = sum(link_lengths)

        S_list = np.zeros((6, self.num_joints))
        positions = np.zeros((3, self.num_joints))
        omega_list = np.zeros((3, self.num_joints))
        for i in range(self.num_joints):
            omega = np.array([0, 0, 1]) if i % 2 == 0 else np.array([0, 1, 0])
            omega_list[:, i] = omega
            S_list[:3, i] = omega
            position = np.array([sum(link_lengths[: i + 1]) * 0.7, 0.0, 0.1 * i])
            positions[:, i] = position
            S_list[3:, i] = np.cross(-omega, position)
        B_list = S_list.copy()
        G_list = [np.eye(6) for _ in range(self.num_joints)]

        self.robot = SerialManipulator(
            M_list=M,
            omega_list=omega_list,
            r_list=positions,
            b_list=positions,
            S_list=S_list,
            B_list=B_list,
            G_list=G_list,
            joint_limits=self.joint_limits,
        )
        self.dynamics = ManipulatorDynamics(
            M_list=M,
            omega_list=omega_list,
            r_list=positions,
            b_list=positions,
            S_list=S_list,
            B_list=B_list,
            Glist=G_list,
        )
        # use_cuda=None lets the planner auto-detect and fall back to CPU.
        self.planner = OptimizedTrajectoryPlanning(
            serial_manipulator=self.robot,
            urdf_path="batch_demo_robot.urdf",
            dynamics=self.dynamics,
            joint_limits=self.joint_limits,
            torque_limits=self.torque_limits,
            use_cuda=None,
            cuda_threshold=10,
        )
        logger.info("Robot kinematics, dynamics, and batch planner initialised")

    # ------------------------------------------------------------- batch data
    def generate_batch_data(self, batch_size: int) -> Dict:
        """Sample random valid start/end joint configurations for a batch.

        Args:
            batch_size: Number of trajectories to generate configurations for.

        Returns:
            Dictionary with ``start_configs`` and ``end_configs`` arrays of
            shape ``(batch_size, num_joints)`` plus the batch ``size``.
        """
        lows = np.array([lo for lo, _ in self.joint_limits], dtype=np.float32)
        highs = np.array([hi for _, hi in self.joint_limits], dtype=np.float32)
        rng = np.random.default_rng(0)
        start = rng.uniform(lows, highs, size=(batch_size, self.num_joints))
        end = rng.uniform(lows, highs, size=(batch_size, self.num_joints))
        return {
            "start_configs": start.astype(np.float32),
            "end_configs": end.astype(np.float32),
            "size": batch_size,
        }

    # ------------------------------------------------------ trajectory drivers
    def batch_trajectory(
        self, batch_data: Dict, N: int, method: int = 3
    ) -> Tuple[Dict, float]:
        """Generate a whole batch of trajectories in one vectorized call.

        Args:
            batch_data: Output of :meth:`generate_batch_data`.
            N: Number of samples per trajectory.
            method: Time-scaling order (3 = cubic, 5 = quintic).

        Returns:
            Tuple of (results dict with positions/velocities/accelerations,
            wall-clock seconds).
        """
        start = time.time()
        results = self.planner.batch_joint_trajectory(
            batch_data["start_configs"],
            batch_data["end_configs"],
            2.0,
            N,
            method,
        )
        return results, time.time() - start

    def sequential_trajectory(
        self, batch_data: Dict, N: int, method: int = 3
    ) -> Tuple[Dict, float]:
        """Generate the same batch with a sequential loop over ``joint_trajectory``.

        Args:
            batch_data: Output of :meth:`generate_batch_data`.
            N: Number of samples per trajectory.
            method: Time-scaling order (3 = cubic, 5 = quintic).

        Returns:
            Tuple of (results dict, wall-clock seconds).
        """
        batch_size = batch_data["size"]
        positions = np.zeros((batch_size, N, self.num_joints), dtype=np.float32)
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        start = time.time()
        for i in range(batch_size):
            traj = self.planner.joint_trajectory(
                batch_data["start_configs"][i],
                batch_data["end_configs"][i],
                2.0,
                N,
                method,
            )
            positions[i] = traj["positions"]
            velocities[i] = traj["velocities"]
            accelerations[i] = traj["accelerations"]
        elapsed = time.time() - start

        return {
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
        }, elapsed

    # -------------------------------------------------------------- analysis
    def run_scaling_analysis(self, batch_sizes: List[int], N: int = 50) -> Dict:
        """Compare batch versus sequential generation across batch sizes.

        Args:
            batch_sizes: Batch sizes to sweep.
            N: Samples per trajectory.

        Returns:
            Dictionary of timing, throughput, and speedup curves.
        """
        logger.info("Running batch vs. sequential scaling analysis")
        # Warm up the trajectory backend once so the first measured batch is
        # not skewed by one-time JIT/threading-layer initialisation.
        warmup = self.generate_batch_data(2)
        self.sequential_trajectory(warmup, N)
        self.batch_trajectory(warmup, N)

        results: Dict[str, List[float]] = {
            "batch_sizes": list(batch_sizes),
            "batch_times": [],
            "sequential_times": [],
            "speedup": [],
            "throughput_batch": [],
            "throughput_sequential": [],
            "memory_mb": [],
        }

        for batch_size in batch_sizes:
            batch_data = self.generate_batch_data(batch_size)

            _, seq_time = self.sequential_trajectory(batch_data, N)
            _, batch_time = self.batch_trajectory(batch_data, N)

            results["sequential_times"].append(seq_time)
            results["batch_times"].append(batch_time)
            results["speedup"].append(seq_time / batch_time if batch_time > 0 else 0.0)
            results["throughput_sequential"].append(
                batch_size / seq_time if seq_time > 0 else 0.0
            )
            results["throughput_batch"].append(
                batch_size / batch_time if batch_time > 0 else 0.0
            )
            # 3 float32 arrays of shape (batch_size, N, num_joints).
            results["memory_mb"].append(
                batch_size * N * self.num_joints * 4 * 3 / (1024 * 1024)
            )
            logger.info(
                "batch=%4d  sequential=%.3fs  batch=%.3fs  speedup=%.2fx",
                batch_size,
                seq_time,
                batch_time,
                results["speedup"][-1],
            )

        return results

    def analyze_batch_results(self, batch_results: Dict) -> Dict:
        """Compute trajectory- and joint-level statistics from real output.

        Args:
            batch_results: Output of :meth:`batch_trajectory`.

        Returns:
            Nested dictionary of trajectory and per-joint statistics.
        """
        positions = batch_results["positions"]
        velocities = batch_results["velocities"]
        accelerations = batch_results["accelerations"]

        path_lengths = np.array(
            [
                np.sum(np.linalg.norm(np.diff(positions[i], axis=0), axis=1))
                for i in range(positions.shape[0])
            ]
        )
        max_velocities = np.max(np.linalg.norm(velocities, axis=2), axis=1)
        max_accelerations = np.max(np.linalg.norm(accelerations, axis=2), axis=1)

        def _summary(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        joint_statistics = {}
        for joint in range(self.num_joints):
            joint_statistics[f"joint_{joint + 1}"] = {
                "position_range": float(
                    np.max(positions[:, :, joint]) - np.min(positions[:, :, joint])
                ),
                "max_velocity": float(np.max(np.abs(velocities[:, :, joint]))),
                "max_acceleration": float(np.max(np.abs(accelerations[:, :, joint]))),
                "rms_velocity": float(np.sqrt(np.mean(velocities[:, :, joint] ** 2))),
                "rms_acceleration": float(
                    np.sqrt(np.mean(accelerations[:, :, joint] ** 2))
                ),
            }

        return {
            "path_lengths": path_lengths,
            "trajectory_statistics": {
                "path_lengths": _summary(path_lengths),
                "max_velocities": _summary(max_velocities),
                "max_accelerations": _summary(max_accelerations),
            },
            "joint_statistics": joint_statistics,
        }

    # ----------------------------------------------------------- visualisation
    def visualize_scaling(self, scaling: Dict) -> None:
        """Plot timing, throughput, speedup, and memory scaling curves.

        Args:
            scaling: Output of :meth:`run_scaling_analysis`.
        """
        batch_sizes = scaling["batch_sizes"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.plot(batch_sizes, scaling["sequential_times"], "b-o", label="Sequential")
        ax.plot(batch_sizes, scaling["batch_times"], "r-^", label="Batch API")
        ax.set(xlabel="Batch Size", ylabel="Time (s)", title="Generation Time Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(
            batch_sizes, scaling["throughput_sequential"], "b-o", label="Sequential"
        )
        ax.plot(batch_sizes, scaling["throughput_batch"], "r-^", label="Batch API")
        ax.set(
            xlabel="Batch Size",
            ylabel="Throughput (trajectories/s)",
            title="Throughput Scaling",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(batch_sizes, scaling["speedup"], "g-s", linewidth=2)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set(
            xlabel="Batch Size",
            ylabel="Speedup Factor",
            title="Batch vs. Sequential Speedup",
        )
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(batch_sizes, scaling["memory_mb"], "m-d", linewidth=2)
        ax.set(
            xlabel="Batch Size",
            ylabel="Trajectory Buffer (MB)",
            title="Batch Memory Footprint",
        )
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if self.save_plots:
            path = os.path.join(OUTPUT_DIR, "batch_scaling_analysis.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", path)
        plt.close(fig)

    def visualize_statistics(self, analysis: Dict, batch_size: int) -> None:
        """Plot trajectory and per-joint motion statistics from real output.

        Args:
            analysis: Output of :meth:`analyze_batch_results`.
            batch_size: Batch size that produced the analysis.
        """
        joint_stats = analysis["joint_statistics"]
        joint_names = list(joint_stats.keys())
        labels = [f"J{i + 1}" for i in range(len(joint_names))]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.hist(
            analysis["path_lengths"], bins=20, color="cyan", edgecolor="black", alpha=0.8
        )
        mean_len = analysis["trajectory_statistics"]["path_lengths"]["mean"]
        ax.axvline(mean_len, color="red", linestyle="--", label=f"Mean: {mean_len:.2f}")
        ax.set(
            xlabel="Joint-space Path Length (rad)",
            ylabel="Count",
            title=f"Path Length Distribution (n={batch_size})",
        )
        ax.legend()

        ax = axes[0, 1]
        ax.bar(labels, [joint_stats[j]["max_velocity"] for j in joint_names],
               color="orange", alpha=0.8)
        ax.set(xlabel="Joint", ylabel="Max Velocity (rad/s)",
               title="Maximum Joint Velocities")

        ax = axes[1, 0]
        ax.bar(labels, [joint_stats[j]["max_acceleration"] for j in joint_names],
               color="purple", alpha=0.8)
        ax.set(xlabel="Joint", ylabel="Max Acceleration (rad/s^2)",
               title="Maximum Joint Accelerations")

        ax = axes[1, 1]
        x = np.arange(len(joint_names))
        ax.bar(x - 0.2, [joint_stats[j]["rms_velocity"] for j in joint_names],
               0.4, label="RMS Velocity", color="blue", alpha=0.8)
        ax.bar(x + 0.2, [joint_stats[j]["rms_acceleration"] for j in joint_names],
               0.4, label="RMS Acceleration", color="red", alpha=0.8)
        ax.set(xlabel="Joint", ylabel="RMS Value", title="RMS Velocity vs. Acceleration")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()
        if self.save_plots:
            path = os.path.join(OUTPUT_DIR, f"batch_statistics_analysis_{batch_size}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", path)
        plt.close(fig)

    # ---------------------------------------------------------------- report
    def generate_report(self, scaling: Dict, analysis: Dict) -> str:
        """Build and optionally save a text report of the analysis.

        Args:
            scaling: Output of :meth:`run_scaling_analysis`.
            analysis: Output of :meth:`analyze_batch_results`.

        Returns:
            The report as a string.
        """
        lines = [
            "=" * 72,
            "                 BATCH PROCESSING ANALYSIS REPORT",
            "=" * 72,
            "",
            "1. CONFIGURATION",
            "-" * 16,
            f"Robot DOF:            {self.num_joints}",
            f"CUDA available:       {CUDA_AVAILABLE}",
            f"GPU acceleration:     {'ENABLED' if self.use_gpu else 'DISABLED (CPU)'}",
            "",
            "2. SCALING (batch_joint_trajectory vs. sequential joint_trajectory)",
            "-" * 66,
            f"{'batch':>7} {'seq (s)':>10} {'batch (s)':>10} {'speedup':>9} {'thr/s':>9}",
        ]
        for i, bs in enumerate(scaling["batch_sizes"]):
            lines.append(
                f"{bs:>7} {scaling['sequential_times'][i]:>10.3f} "
                f"{scaling['batch_times'][i]:>10.3f} {scaling['speedup'][i]:>8.2f}x "
                f"{scaling['throughput_batch'][i]:>9.1f}"
            )
        best_speedup = max(scaling["speedup"])
        lines += [
            "",
            f"Best batch-vs-sequential speedup: {best_speedup:.2f}x",
            "",
            "3. TRAJECTORY STATISTICS (from real planner output)",
            "-" * 51,
        ]
        for metric, stats in analysis["trajectory_statistics"].items():
            lines.append(
                f"  {metric.replace('_', ' ').title():<18} "
                f"mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
                f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
            )
        lines += ["", "4. PER-JOINT MOTION", "-" * 19]
        for joint, stats in analysis["joint_statistics"].items():
            lines.append(
                f"  {joint:<9} max_vel={stats['max_velocity']:.3f} rad/s  "
                f"max_acc={stats['max_acceleration']:.3f} rad/s^2  "
                f"range={stats['position_range']:.3f} rad"
            )
        lines += [
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 72,
            "",
        ]
        report = "\n".join(lines)

        if self.save_plots:
            path = os.path.join(OUTPUT_DIR, "batch_processing_analysis_report.txt")
            with open(path, "w") as f:
                f.write(report)
            logger.info("Saved %s", path)
        return report


def main() -> None:
    """Run the batch-processing demonstration end to end."""
    print("=" * 72)
    print("   MANIPULAPY BATCH PROCESSING DEMONSTRATION")
    print("=" * 72)

    demo = BatchProcessingDemo(save_plots=True)

    # Keep batch sizes modest so the CPU run stays well under ~90s.
    batch_sizes = [5, 10, 25, 50, 100]

    print("\n1. Running batch vs. sequential scaling analysis...")
    scaling = demo.run_scaling_analysis(batch_sizes, N=50)

    print("\n2. Generating a representative batch and analysing it...")
    batch_data = demo.generate_batch_data(100)
    batch_results, _ = demo.batch_trajectory(batch_data, N=50)
    analysis = demo.analyze_batch_results(batch_results)

    print("\n3. Generating visualisations...")
    demo.visualize_scaling(scaling)
    demo.visualize_statistics(analysis, batch_data["size"])

    print("\n4. Generating analysis report...")
    report = demo.generate_report(scaling, analysis)
    print(report)

    print("=" * 72)
    print("                    ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Best batch-vs-sequential speedup: {max(scaling['speedup']):.2f}x")
    print("\nGenerated files:")
    for filename in (
        "batch_scaling_analysis.png",
        f"batch_statistics_analysis_{batch_data['size']}.png",
        "batch_processing_analysis_report.txt",
    ):
        marker = "[ok]" if os.path.exists(os.path.join(OUTPUT_DIR, filename)) else "[--]"
        print(f"  {marker} {filename}")
    print("=" * 72)


if __name__ == "__main__":
    main()
