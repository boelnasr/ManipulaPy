#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
TRAC-IK Benchmark Suite - ManipulaPy

Comprehensive benchmarking tool comparing TRAC-IK performance against other IK solvers.
Uses xArm 6-DOF robot model for realistic testing.

Features:
- Speed comparison: TRAC-IK vs iterative vs smart vs robust IK
- Accuracy analysis: position and orientation errors
- Success rate comparison across methods
- Statistical analysis with percentiles
- Workspace coverage testing
- Real-time performance evaluation

Usage:
    python trac_ik_benchmark.py                     # Run with defaults (30 tests)
    python trac_ik_benchmark.py --num-tests 100     # More tests for statistics
    python trac_ik_benchmark.py --timeout 0.1       # Custom TRAC-IK timeout
    python trac_ik_benchmark.py --plot              # Generate plots
    python trac_ik_benchmark.py --save-results      # Save JSON results

Performance Targets:
    - TRAC-IK: 80%+ within 10mm, ~500ms solve time (parallel speedup)
    - Iterative IK: ~10-40% within 10mm, ~200ms solve time
    - Smart IK: ~40-60% within 10mm, ~500-800ms solve time
    - Robust IK: ~10-50% within 10mm, ~1500ms solve time

Author: ManipulaPy Development Team
Date: January 2026
"""

import numpy as np
import time
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Add ManipulaPy to path if needed
try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
    from ManipulaPy import utils
    from ManipulaPy.trac_ik import TracIKSolver
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
    from ManipulaPy import utils
    from ManipulaPy.trac_ik import TracIKSolver

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IKResult:
    """Single IK solve result."""
    success: bool
    solve_time_ms: float
    position_error_mm: float
    orientation_error_mrad: float
    iterations: int = 0


@dataclass
class MethodStats:
    """Statistics for an IK method."""
    name: str
    num_tests: int
    success_rate: float
    near_success_rate: float  # Within 10mm/10mrad
    avg_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    min_time_ms: float
    max_time_ms: float
    avg_pos_error_mm: float
    avg_ori_error_mrad: float
    median_pos_error_mm: float
    successful_times: List[float] = None


class TracIKBenchmark:
    """
    TRAC-IK Benchmark Suite.

    Compares TRAC-IK performance against other IK methods using xArm 6-DOF robot.
    """

    def __init__(
        self,
        num_tests: int = 30,
        trac_ik_timeout: float = 0.5,
        tolerance_position: float = 2e-3,
        tolerance_orientation: float = 2e-3,
        output_dir: str = "trac_ik_benchmark_results",
        verbose: bool = False
    ):
        """
        Initialize benchmark suite.

        Args:
            num_tests: Number of test poses to evaluate
            trac_ik_timeout: TRAC-IK timeout in seconds (default: 500ms)
            tolerance_position: Position tolerance in meters (default: 2mm)
            tolerance_orientation: Orientation tolerance in radians (default: 2mrad)
            output_dir: Directory for results
            verbose: Enable verbose logging
        """
        self.num_tests = num_tests
        self.trac_ik_timeout = trac_ik_timeout
        self.tol_pos = tolerance_position
        self.tol_ori = tolerance_orientation
        self.output_dir = output_dir
        self.verbose = verbose

        os.makedirs(output_dir, exist_ok=True)

        self.robot = None
        self.joint_limits = None
        self.dof = None
        self.test_poses = []
        self.test_configs = []
        self.results = {}

        logger.info(f"TRAC-IK Benchmark: {num_tests} tests, timeout={trac_ik_timeout*1000:.0f}ms")

    def load_robot(self) -> bool:
        """Load xArm 6-DOF robot from URDF."""
        logger.info("Loading xArm 6-DOF robot...")

        try:
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator

            # Get joint limits
            if hasattr(urdf_processor, 'robot_data') and 'joint_limits' in urdf_processor.robot_data:
                self.joint_limits = urdf_processor.robot_data['joint_limits']
            else:
                # xArm 6 DOF default limits
                self.joint_limits = [
                    (-6.28, 6.28), (-2.09, 2.09), (-6.28, 6.28),
                    (-6.28, 6.28), (-6.28, 6.28), (-6.28, 6.28)
                ]

            # Update robot's joint limits
            self.robot.joint_limits = self.joint_limits
            self.dof = len(self.joint_limits)

            logger.info(f"Robot loaded: {self.dof} DOF, xArm 6")
            return True

        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return False

    def generate_test_poses(self) -> int:
        """
        Generate reachable test poses using FK on random configurations.
        Avoids singular configurations by checking Jacobian condition number.

        Returns:
            Number of valid test poses generated
        """
        logger.info(f"Generating {self.num_tests} test poses (avoiding singularities)...")

        self.test_poses = []
        self.test_configs = []

        # Generate poses from random valid configurations
        attempts = 0
        max_attempts = self.num_tests * 20

        while len(self.test_poses) < self.num_tests and attempts < max_attempts:
            attempts += 1

            # Random configuration within joint limits (with margin)
            config = []
            for low, high in self.joint_limits:
                margin = (high - low) * 0.15  # 15% margin from limits
                config.append(np.random.uniform(low + margin, high - margin))
            config = np.array(config)

            try:
                # Check for singularity using Jacobian condition number
                J = self.robot.jacobian(config, frame="space")
                cond = np.linalg.cond(J)

                # Skip if near singularity (condition number > 100)
                if cond > 100:
                    continue

                # Check Jacobian rank
                rank = np.linalg.matrix_rank(J)
                if rank < self.dof:
                    continue

                T = self.robot.forward_kinematics(config, frame="space")

                # Verify it's a valid transformation
                if np.isnan(T).any() or np.isinf(T).any():
                    continue

                self.test_poses.append(T)
                self.test_configs.append(config)

            except Exception:
                continue

        logger.info(f"Generated {len(self.test_poses)} valid non-singular test poses "
                   f"(checked {attempts} configurations)")
        return len(self.test_poses)

    def compute_error(
        self,
        T_result: np.ndarray,
        T_target: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute position and orientation error.

        Returns:
            Tuple of (position_error_mm, orientation_error_mrad)
        """
        # Position error
        pos_error = np.linalg.norm(T_result[:3, 3] - T_target[:3, 3]) * 1000  # mm

        # Orientation error using rotation matrix difference
        R_err = T_result[:3, :3].T @ T_target[:3, :3]
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        ori_error = np.arccos(trace) * 1000  # mrad

        return pos_error, ori_error

    def benchmark_trac_ik(self) -> List[IKResult]:
        """Benchmark TRAC-IK solver."""
        logger.info("Benchmarking TRAC-IK...")
        results = []

        for i, (T_target, true_config) in enumerate(zip(self.test_poses, self.test_configs)):
            # Solve IK
            theta, success, solve_time = self.robot.trac_ik(
                T_target,
                timeout=self.trac_ik_timeout,
                eomg=self.tol_ori,
                ev=self.tol_pos,
                num_restarts=3
            )

            # Compute errors
            T_result = self.robot.forward_kinematics(theta, frame="space")
            pos_err, ori_err = self.compute_error(T_result, T_target)

            results.append(IKResult(
                success=success,
                solve_time_ms=solve_time * 1000,
                position_error_mm=pos_err,
                orientation_error_mrad=ori_err
            ))

            if self.verbose and (i + 1) % 10 == 0:
                logger.debug(f"  TRAC-IK: {i+1}/{len(self.test_poses)} tests complete")

        return results

    def benchmark_iterative_ik(self) -> List[IKResult]:
        """Benchmark iterative IK solver."""
        logger.info("Benchmarking Iterative IK...")
        results = []

        for i, T_target in enumerate(self.test_poses):
            # Random initial guess
            theta0 = np.zeros(self.dof)

            start = time.perf_counter()
            theta, success, iters = self.robot.iterative_inverse_kinematics(
                T_target,
                theta0,
                eomg=self.tol_ori,
                ev=self.tol_pos,
                max_iterations=500,
                damping=0.02,
                step_cap=0.3
            )
            solve_time = time.perf_counter() - start

            # Compute errors
            T_result = self.robot.forward_kinematics(theta, frame="space")
            pos_err, ori_err = self.compute_error(T_result, T_target)

            results.append(IKResult(
                success=success,
                solve_time_ms=solve_time * 1000,
                position_error_mm=pos_err,
                orientation_error_mrad=ori_err,
                iterations=iters
            ))

            if self.verbose and (i + 1) % 10 == 0:
                logger.debug(f"  Iterative: {i+1}/{len(self.test_poses)} tests complete")

        return results

    def benchmark_smart_ik(self) -> List[IKResult]:
        """Benchmark smart IK solver with workspace heuristic."""
        logger.info("Benchmarking Smart IK (workspace_heuristic)...")
        results = []

        for i, T_target in enumerate(self.test_poses):
            start = time.perf_counter()
            theta, success, iters = self.robot.smart_inverse_kinematics(
                T_target,
                strategy='workspace_heuristic',
                eomg=self.tol_ori,
                ev=self.tol_pos,
                max_iterations=500,
                damping=0.02,
                step_cap=0.3
            )
            solve_time = time.perf_counter() - start

            # Compute errors
            T_result = self.robot.forward_kinematics(theta, frame="space")
            pos_err, ori_err = self.compute_error(T_result, T_target)

            results.append(IKResult(
                success=success,
                solve_time_ms=solve_time * 1000,
                position_error_mm=pos_err,
                orientation_error_mrad=ori_err,
                iterations=iters
            ))

            if self.verbose and (i + 1) % 10 == 0:
                logger.debug(f"  Smart IK: {i+1}/{len(self.test_poses)} tests complete")

        return results

    def benchmark_robust_ik(self) -> List[IKResult]:
        """Benchmark robust IK solver (multi-start)."""
        logger.info("Benchmarking Robust IK (multi-start)...")
        results = []

        for i, T_target in enumerate(self.test_poses):
            start = time.perf_counter()
            theta, success, iters, strategy = self.robot.robust_inverse_kinematics(
                T_target,
                max_attempts=5,  # Reduced for speed
                eomg=self.tol_ori,
                ev=self.tol_pos,
                max_iterations=300
            )
            solve_time = time.perf_counter() - start

            # Compute errors
            T_result = self.robot.forward_kinematics(theta, frame="space")
            pos_err, ori_err = self.compute_error(T_result, T_target)

            results.append(IKResult(
                success=success,
                solve_time_ms=solve_time * 1000,
                position_error_mm=pos_err,
                orientation_error_mrad=ori_err,
                iterations=iters
            ))

            if self.verbose and (i + 1) % 10 == 0:
                logger.debug(f"  Robust IK: {i+1}/{len(self.test_poses)} tests complete")

        return results

    def compute_stats(self, name: str, results: List[IKResult]) -> MethodStats:
        """Compute statistics for a method's results."""
        successes = [r for r in results if r.success]
        # Near success: within 10mm position error
        near_successes = [r for r in results if r.position_error_mm < 10.0]

        times = [r.solve_time_ms for r in results]
        success_times = [r.solve_time_ms for r in successes] if successes else [0]

        # Compute errors for ALL results (not just successes)
        all_pos_errors = [r.position_error_mm for r in results]
        all_ori_errors = [r.orientation_error_mrad for r in results]

        return MethodStats(
            name=name,
            num_tests=len(results),
            success_rate=len(successes) / len(results) * 100,
            near_success_rate=len(near_successes) / len(results) * 100,
            avg_time_ms=np.mean(times),
            median_time_ms=np.median(times),
            p95_time_ms=np.percentile(times, 95),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            avg_pos_error_mm=np.mean(all_pos_errors),
            avg_ori_error_mrad=np.mean(all_ori_errors),
            median_pos_error_mm=np.median(all_pos_errors),
            successful_times=success_times
        )

    def run_benchmark(self, methods: List[str] = None) -> Dict[str, MethodStats]:
        """
        Run complete benchmark suite.

        Args:
            methods: List of methods to test. Default: all methods.
                     Options: 'trac_ik', 'iterative', 'smart', 'robust'

        Returns:
            Dictionary mapping method names to their statistics
        """
        if methods is None:
            methods = ['trac_ik', 'iterative', 'smart', 'robust']

        # Load robot and generate test poses
        if not self.load_robot():
            raise RuntimeError("Failed to load robot")

        if not self.generate_test_poses():
            raise RuntimeError("Failed to generate test poses")

        logger.info("=" * 60)
        logger.info("Starting benchmark...")
        logger.info("=" * 60)

        results = {}

        # Run each method
        if 'trac_ik' in methods:
            trac_results = self.benchmark_trac_ik()
            results['trac_ik'] = self.compute_stats('TRAC-IK', trac_results)

        if 'iterative' in methods:
            iter_results = self.benchmark_iterative_ik()
            results['iterative'] = self.compute_stats('Iterative IK', iter_results)

        if 'smart' in methods:
            smart_results = self.benchmark_smart_ik()
            results['smart'] = self.compute_stats('Smart IK', smart_results)

        if 'robust' in methods:
            robust_results = self.benchmark_robust_ik()
            results['robust'] = self.compute_stats('Robust IK', robust_results)

        self.results = results
        return results

    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "=" * 90)
        print("TRAC-IK BENCHMARK RESULTS")
        print("=" * 90)
        print(f"Robot: xArm 6-DOF | Tests: {self.num_tests} | "
              f"Tolerance: {self.tol_pos*1000:.1f}mm / {self.tol_ori*1000:.1f}mrad")
        print("=" * 90)

        # Header
        print(f"\n{'Method':<16} {'Success':>8} {'<10mm':>8} {'Avg Time':>10} {'Med Time':>10} "
              f"{'Avg Err':>10} {'Med Err':>10}")
        print("-" * 90)

        # Results
        for name, stats in self.results.items():
            print(f"{stats.name:<16} {stats.success_rate:>7.1f}% {stats.near_success_rate:>7.1f}% "
                  f"{stats.avg_time_ms:>8.1f}ms "
                  f"{stats.median_time_ms:>8.1f}ms "
                  f"{stats.avg_pos_error_mm:>8.2f}mm "
                  f"{stats.median_pos_error_mm:>8.2f}mm")

        print("-" * 90)

        # Comparison summary
        if 'trac_ik' in self.results:
            trac = self.results['trac_ik']
            print(f"\nTRAC-IK Performance Summary:")
            print(f"  - Success Rate: {trac.success_rate:.1f}%")
            print(f"  - Average Solve Time: {trac.avg_time_ms:.2f}ms")
            print(f"  - 95th Percentile Time: {trac.p95_time_ms:.2f}ms")
            print(f"  - Real-time capable: {'Yes' if trac.p95_time_ms < 10 else 'No'} (target: <10ms)")

            # Speedup comparison
            for name, stats in self.results.items():
                if name != 'trac_ik' and stats.avg_time_ms > 0:
                    speedup = stats.avg_time_ms / trac.avg_time_ms
                    print(f"  - Speedup vs {stats.name}: {speedup:.1f}x")

        print("=" * 80 + "\n")

    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            filename = os.path.join(self.output_dir, "trac_ik_benchmark_results.json")

        # Convert to serializable format
        output = {
            'config': {
                'num_tests': self.num_tests,
                'trac_ik_timeout_ms': self.trac_ik_timeout * 1000,
                'tolerance_position_mm': self.tol_pos * 1000,
                'tolerance_orientation_mrad': self.tol_ori * 1000,
                'robot': 'xArm 6-DOF'
            },
            'results': {}
        }

        for name, stats in self.results.items():
            output['results'][name] = {
                'name': stats.name,
                'success_rate_pct': stats.success_rate,
                'near_success_rate_pct': stats.near_success_rate,
                'avg_time_ms': stats.avg_time_ms,
                'median_time_ms': stats.median_time_ms,
                'p95_time_ms': stats.p95_time_ms,
                'min_time_ms': stats.min_time_ms,
                'max_time_ms': stats.max_time_ms,
                'avg_pos_error_mm': stats.avg_pos_error_mm,
                'median_pos_error_mm': stats.median_pos_error_mm,
                'avg_ori_error_mrad': stats.avg_ori_error_mrad
            }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {filename}")

    def generate_plots(self):
        """Generate benchmark visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return

        if not self.results:
            logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = list(self.results.keys())

        # 1. Success Rate (strict and near-success)
        ax1 = axes[0]
        success_rates = [self.results[m].success_rate for m in methods]
        near_rates = [self.results[m].near_success_rate for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax1.bar(x - width/2, success_rates, width, label='Strict', color='#2ecc71')
        bars2 = ax1.bar(x + width/2, near_rates, width, label='<10mm', color='#3498db')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('IK Success Rate Comparison')
        ax1.set_ylim(0, 105)
        ax1.set_xticks(x)
        ax1.set_xticklabels([self.results[m].name for m in methods], fontsize=8)
        ax1.legend()
        for bar, rate in zip(bars1, success_rates):
            if rate > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)

        # 2. Solve Time
        ax2 = axes[1]
        avg_times = [self.results[m].avg_time_ms for m in methods]
        p95_times = [self.results[m].p95_time_ms for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        ax2.bar(x - width/2, avg_times, width, label='Average', color='#3498db')
        ax2.bar(x + width/2, p95_times, width, label='95th percentile', color='#9b59b6')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('IK Solve Time Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([self.results[m].name for m in methods])
        ax2.legend()
        ax2.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='10ms target')

        # 3. Accuracy
        ax3 = axes[2]
        pos_errors = [self.results[m].avg_pos_error_mm for m in methods]
        ori_errors = [self.results[m].avg_ori_error_mrad for m in methods]
        x = np.arange(len(methods))
        ax3.bar(x - width/2, pos_errors, width, label='Position (mm)', color='#e74c3c')
        ax3.bar(x + width/2, ori_errors, width, label='Orientation (mrad)', color='#f39c12')
        ax3.set_ylabel('Error')
        ax3.set_title('IK Accuracy (Successful Solves)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([self.results[m].name for m in methods])
        ax3.legend()

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, "trac_ik_benchmark_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        logger.info(f"Plots saved to {plot_path}")


def main():
    """Main entry point for benchmark."""
    parser = argparse.ArgumentParser(
        description="TRAC-IK Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python trac_ik_benchmark.py                     # Basic benchmark
    python trac_ik_benchmark.py --num-tests 100     # More tests
    python trac_ik_benchmark.py --timeout 0.1       # Longer timeout
    python trac_ik_benchmark.py --plot --save       # Save plots and results
    python trac_ik_benchmark.py --methods trac_ik iterative  # Compare specific methods
        """
    )

    parser.add_argument('--num-tests', type=int, default=30,
                       help='Number of test poses (default: 30)')
    parser.add_argument('--timeout', type=float, default=0.5,
                       help='TRAC-IK timeout in seconds (default: 0.5)')
    parser.add_argument('--tolerance', type=float, default=2e-3,
                       help='Position/orientation tolerance (default: 2e-3 = 2mm/2mrad)')
    parser.add_argument('--methods', nargs='+',
                       choices=['trac_ik', 'iterative', 'smart', 'robust'],
                       default=None,
                       help='Methods to benchmark (default: all)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save', '--save-results', action='store_true',
                       help='Save results to JSON')
    parser.add_argument('--output-dir', type=str, default='trac_ik_benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Create and run benchmark
    benchmark = TracIKBenchmark(
        num_tests=args.num_tests,
        trac_ik_timeout=args.timeout,
        tolerance_position=args.tolerance,
        tolerance_orientation=args.tolerance,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    try:
        benchmark.run_benchmark(methods=args.methods)
        benchmark.print_results()

        if args.save:
            benchmark.save_results()

        if args.plot:
            benchmark.generate_plots()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
