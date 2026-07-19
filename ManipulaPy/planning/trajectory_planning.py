#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Optimized Path Planning Module - ManipulaPy

This module provides highly optimized trajectory planning capabilities including
joint and Cartesian space trajectory generation with CUDA acceleration and collision
avoidance.

Key optimizations:
- Adaptive grid sizing for optimal GPU occupancy
- Memory pooling to reduce allocation overhead
- Batch processing for multiple trajectories
- Fused kernels to minimize memory bandwidth
- Intelligent fallback to CPU when beneficial
- 2D parallelization for better GPU utilization
- Advanced kernel selection for 40x+ speedups

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import sys as _sys
from types import ModuleType as _ModuleType

from . import _kernels as _runtime
from ._kernels import *  # noqa: F401,F403

from .trajectory import (  # noqa: F401
    _GenerationMixin,
    _trajectory_cpu_fallback,
    _traj_cpu_njit,
)
from .trajectory_dynamics import _DynamicsMixin
from .collision_host import _CollisionMixin
from ._plotting import _PlottingMixin


_FORWARDED_RUNTIME_NAMES = frozenset(
    {
        "CUDA_AVAILABLE",
        "_best_2d_config",
        "_h2d_pinned",
        "auto_select_optimal_kernel",
        "batch_trajectory_kernel",
        "benchmark_kernel_performance",
        "cartesian_trajectory_kernel",
        "check_cuda_availability",
        "cuda",
        "forward_dynamics_kernel",
        "fused_potential_gradient_kernel",
        "get_backend",
        "get_cuda_array",
        "get_gpu_properties",
        "get_memory_pool_stats",
        "get_optimal_kernel_config",
        "inverse_dynamics_kernel",
        "make_1d_grid",
        "make_2d_grid",
        "make_2d_grid_optimized",
        "optimized_batch_trajectory_generation",
        "optimized_trajectory_generation",
        "optimized_trajectory_generation_monitored",
        "print_performance_recommendations",
        "profile_start",
        "profile_stop",
        "return_cuda_array",
        "setup_cuda_environment_for_40x_speedup",
        "trajectory_kernel",
        "trajectory_kernel_cache_friendly",
        "trajectory_kernel_memory_optimized",
        "trajectory_kernel_vectorized",
        "trajectory_kernel_warp_optimized",
    }
)


class _PlanningCompatibilityModule(_ModuleType):
    """Forward historical mutable patch points to the shared runtime module."""

    def __getattribute__(self, name):
        if name in _FORWARDED_RUNTIME_NAMES:
            return getattr(_runtime, name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in _FORWARDED_RUNTIME_NAMES:
            setattr(_runtime, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _FORWARDED_RUNTIME_NAMES and hasattr(_runtime, name):
            delattr(_runtime, name)
        super().__delattr__(name)


_sys.modules[__name__].__class__ = _PlanningCompatibilityModule


class OptimizedTrajectoryPlanning(
    _GenerationMixin, _DynamicsMixin, _CollisionMixin, _PlottingMixin
):
    """
    Highly optimized trajectory planning class with adaptive GPU/CPU execution,
    memory pooling, and batch processing capabilities for 40x+ speedups.
    """

    def __init__(
        self,
        serial_manipulator,
        urdf_path,
        dynamics,
        joint_limits,
        torque_limits=None,
        *,  # ――― everything after * is keyword-only ―――
        use_cuda: Optional[bool] = None,
        cuda_threshold: int = 10,
        memory_pool_size_mb: Optional[int] = None,
        enable_profiling: bool = False,
        auto_optimize: bool = True,
        kernel_type: str = "auto",
        target_speedup: float = 40.0,
    ) -> None:
        """
        Enhanced trajectory planner with advanced CUDA optimizations.

        Parameters
        ----------
        serial_manipulator : SerialManipulator
        urdf_path          : str
        dynamics           : ManipulatorDynamics
        joint_limits       : list[tuple[float,float]]
        torque_limits      : list[tuple[float,float]], optional

        use_cuda           : None | bool
            • None  → auto-detect (default)
            • True  → force GPU (raise if CUDA absent)
            • False → force CPU

        cuda_threshold     : int
            Min. (N × joints) before we bother launching the GPU.

        memory_pool_size_mb: int | None
            If set, resize the global CUDA memory pool (in MB).

        enable_profiling   : bool
            Enable CUDA profiling for performance analysis.

        auto_optimize      : bool
            Automatically setup CUDA environment for maximum performance.

        kernel_type        : str
            Kernel selection strategy: "auto", "standard", "vectorized",
            "memory_optimized", "warp_optimized", "cache_friendly", "auto_tune"

        target_speedup     : float
            Target speedup over CPU (used for recommendations).
        """
        # ------------------------------------------------------------
        # FIRST: Set all basic attributes to prevent AttributeError
        # ------------------------------------------------------------
        self.kernel_type = kernel_type if kernel_type is not None else "auto"
        self.target_speedup = target_speedup if target_speedup is not None else 40.0
        self.enable_profiling = (
            enable_profiling if enable_profiling is not None else False
        )

        # Initialize all caches and tracking attributes immediately
        self._gpu_arrays = {}
        self._kernel_cache = {}
        self._last_cpu_time = 0.0

        # Initialize performance stats early
        self.performance_stats = {
            "gpu_calls": 0,
            "cpu_calls": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "memory_transfers": 0,
            "kernel_launches": 0,
            "speedup_achieved": 0.0,
            "best_kernel_used": "none",
        }

        # ------------------------------------------------------------
        # Auto-optimization setup
        # ------------------------------------------------------------
        if auto_optimize and _runtime.CUDA_AVAILABLE:
            _runtime.setup_cuda_environment_for_40x_speedup()

        # ------------------------------------------------------------
        # basic data
        # ------------------------------------------------------------
        self.serial_manipulator = serial_manipulator
        self.dynamics = dynamics
        self.joint_limits = np.asarray(joint_limits, dtype=np.float32)
        self.torque_limits = (
            np.asarray(torque_limits, dtype=np.float32)
            if torque_limits is not None
            else np.array([[-np.inf, np.inf]] * len(joint_limits), dtype=np.float32)
        )

        # Store optimization parameters
        self.kernel_type = kernel_type
        self.target_speedup = target_speedup

        # ------------------------------------------------------------
        # collision-checking helpers
        # ------------------------------------------------------------
        try:
            self.collision_checker = CollisionChecker(urdf_path)
            self.potential_field = PotentialField()
        except Exception as exc:
            logger.warning("Could not initialise collision checker: %s", exc)
            self.collision_checker = None
            self.potential_field = None

        # ------------------------------------------------------------
        # CUDA feature flags
        # ------------------------------------------------------------
        detected_cuda = _runtime.check_cuda_availability()
        if use_cuda is None:
            self.cuda_available = detected_cuda
        elif use_cuda and not detected_cuda:
            raise RuntimeError("use_cuda=True requested but CUDA is not available.")
        else:
            self.cuda_available = bool(use_cuda)

        self.gpu_properties = (
            _runtime.get_gpu_properties() if self.cuda_available else None
        )

        # Adaptive threshold based on target speedup
        if self.cuda_available and self.gpu_properties:
            sm_count = self.gpu_properties["multiprocessor_count"]
            # Calculate threshold for target speedup
            min_elements_per_sm = 1000 if target_speedup >= 40 else 500
            self.cpu_threshold = max(
                cuda_threshold, int(sm_count * min_elements_per_sm / len(joint_limits))
            )
        else:
            self.cpu_threshold = cuda_threshold

        # optionally resize a global memory-pool
        if memory_pool_size_mb is not None and self.cuda_available:
            from ..cuda_kernels import _cuda_memory_pool

            _cuda_memory_pool.max_pool_size = (
                memory_pool_size_mb * 1024 * 1024 // 4  # entries of float32
            )

        # ------------------------------------------------------------
        # performance bookkeeping
        # ------------------------------------------------------------
        self.performance_stats = {
            "gpu_calls": 0,
            "cpu_calls": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "memory_transfers": 0,
            "kernel_launches": 0,
            "speedup_achieved": 0.0,
            "best_kernel_used": "none",
        }

        # Enable profiling if requested (after all attributes are initialized)
        if self.enable_profiling and self.cuda_available:
            _runtime.profile_start()

        # Print performance recommendations on initialization
        if self.cuda_available:
            num_joints = len(joint_limits)
            logger.info(
                f"🚀 OptimizedTrajectoryPlanning initialized for {num_joints} joints"
            )
            if target_speedup >= 40:
                min_N_for_target = self.cpu_threshold // num_joints
                logger.info(
                    f"💡 For {target_speedup}x speedup, use N ≥ "
                    f"{min_N_for_target:,} trajectory points"
                )

        logger.info(
            "Optimised planner – CUDA enabled: %s (threshold %d, kernel: %s)",
            self.cuda_available,
            self.cpu_threshold,
            self.kernel_type,
        )
        if self.gpu_properties:
            logger.info(
                "GPU: %d SMs, %d max threads/block",
                self.gpu_properties["multiprocessor_count"],
                self.gpu_properties["max_threads_per_block"],
            )

    def _get_or_resize_gpu_array(
        self, array_name: str, shape: Tuple[int, ...], dtype: Any = np.float32
    ) -> Any:
        """Return a pooled CUDA array with the requested shape / dtype.

        Looks up a previously pooled device array by ``array_name`` and reuses
        it when its shape and dtype match; otherwise the old array (if any) is
        returned to the pool and a freshly sized one is acquired and cached.

        Args:
            array_name (str): Cache key identifying the pooled array slot.
            shape (Tuple[int, ...]): Desired device array shape.
            dtype (Any): Desired element dtype. Defaults to ``np.float32``.

        Returns:
            Any: The pooled CUDA device array of the requested shape/dtype, or
            ``None`` if CUDA is unavailable.
        """
        if not self.cuda_available:
            return None

        arr = self._gpu_arrays.get(array_name)

        if (arr is None) or (arr.shape != shape) or (arr.dtype != dtype):
            if arr is not None:
                _runtime.return_cuda_array(arr)

            arr = _runtime.get_cuda_array(shape, dtype)
            self._gpu_arrays[array_name] = arr

        return arr

    def _should_use_gpu(self, N: int, num_joints: int) -> bool:
        """Decide whether to dispatch a problem to the GPU.

        Returns ``False`` when CUDA is unavailable or the total work
        (``N * num_joints``) is below ``self.cpu_threshold``. Otherwise logs a
        debug hint when the per-SM element count is unlikely to reach the
        configured target speedup, then approves GPU execution.

        Args:
            N (int): Number of trajectory points / timesteps.
            num_joints (int): Number of joints (work units per point).

        Returns:
            bool: ``True`` if the GPU path should be used, ``False`` otherwise.
        """
        if not self.cuda_available:
            return False

        total_work = N * num_joints
        if total_work < self.cpu_threshold:
            return False

        # Additional checks for memory and performance
        if self.gpu_properties:
            sm_count = self.gpu_properties["multiprocessor_count"]
            elements_per_sm = total_work / sm_count

            # Check if we can achieve target speedup
            target_speedup_value = getattr(self, "target_speedup", 40.0)
            if target_speedup_value >= 40 and elements_per_sm < 10000:
                logger.debug(
                    f"Problem size may not achieve {target_speedup_value}x speedup. "
                    f"Elements per SM: {elements_per_sm:.0f}, recommended: ≥10,000"
                )

        return True

    def _get_optimal_kernel_config(
        self, N: int, num_joints: int
    ) -> Optional[Dict[str, Any]]:
        """Get or compute the optimal kernel launch configuration with caching.

        Resolves the effective kernel type (auto-selecting one when
        ``self.kernel_type == "auto"``) and computes a launch configuration for
        the given problem size, memoising results by
        ``(N, num_joints, kernel_type)``.

        Args:
            N (int): Number of trajectory points / timesteps.
            num_joints (int): Number of joints.

        Returns:
            Optional[Dict[str, Any]]: Kernel configuration dict (grid/block,
            kernel type, etc.), or ``None`` if no configuration is available.
        """
        # Ensure required attributes exist
        if not hasattr(self, "_kernel_cache"):
            self._kernel_cache = {}
        if not hasattr(self, "kernel_type"):
            self.kernel_type = "auto"

        cache_key = (N, num_joints, self.kernel_type)

        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        if self.kernel_type == "auto":
            kernel_type = _runtime.auto_select_optimal_kernel(N, num_joints)
        else:
            kernel_type = self.kernel_type

        config = _runtime.get_optimal_kernel_config(N, num_joints, kernel_type)
        self._kernel_cache[cache_key] = config

        return config

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Enhanced performance statistics with speedup analysis.

        Returns:
            dict: Comprehensive performance statistics
        """
        stats = self.performance_stats.copy()

        if stats["gpu_calls"] > 0:
            stats["avg_gpu_time"] = stats["total_gpu_time"] / stats["gpu_calls"]
        else:
            stats["avg_gpu_time"] = 0.0

        if stats["cpu_calls"] > 0:
            stats["avg_cpu_time"] = stats["total_cpu_time"] / stats["cpu_calls"]
        else:
            stats["avg_cpu_time"] = 0.0

        total_calls = stats["gpu_calls"] + stats["cpu_calls"]
        if total_calls > 0:
            stats["gpu_usage_percent"] = (stats["gpu_calls"] / total_calls) * 100
        else:
            stats["gpu_usage_percent"] = 0.0

        # Calculate overall speedup
        total_gpu_time = stats["total_gpu_time"]
        total_cpu_time = stats["total_cpu_time"]
        if total_gpu_time > 0 and total_cpu_time > 0:
            stats["overall_speedup"] = total_cpu_time / total_gpu_time
        else:
            stats["overall_speedup"] = 0.0

        # Add memory pool statistics
        if self.cuda_available:
            stats["memory_pool_stats"] = _runtime.get_memory_pool_stats()

        # Simple EWMA auto-tune for adaptive threshold
        if stats["avg_gpu_time"] > 0 and stats["avg_cpu_time"] > 0:
            efficiency_ratio = stats["avg_cpu_time"] / stats["avg_gpu_time"]
            self.cpu_threshold = int(
                0.9 * self.cpu_threshold + 0.1 * efficiency_ratio * self.cpu_threshold
            )
            self.cpu_threshold = max(
                50, min(self.cpu_threshold, 5000)
            )  # Keep within reasonable bounds

        return stats

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "gpu_calls": 0,
            "cpu_calls": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "memory_transfers": 0,
            "kernel_launches": 0,
            "speedup_achieved": 0.0,
            "best_kernel_used": "none",
        }

    def cleanup_gpu_memory(self) -> None:
        """Enhanced GPU memory cleanup."""
        if self.cuda_available:
            # Clean up per-instance cache
            if hasattr(self, "_gpu_arrays"):
                for array in self._gpu_arrays.values():
                    if array is not None:
                        _runtime.return_cuda_array(array)
                self._gpu_arrays.clear()

            # Clear kernel cache
            if hasattr(self, "_kernel_cache"):
                self._kernel_cache.clear()

            # Clear global memory pool
            from ..cuda_kernels import _cuda_memory_pool

            _cuda_memory_pool.clear()

            # Synchronize and clean up CUDA context
            _runtime.cuda.synchronize()

            logger.info("GPU memory cleaned up")

    def benchmark_all_kernels(
        self, N: int = 5000, num_joints: int = 6, num_runs: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive benchmarking of all available kernels.

        Args:
            N (int): Number of trajectory points for benchmarking
            num_joints (int): Number of joints
            num_runs (int): Number of benchmark runs per kernel

        Returns:
            dict: Benchmark results for all kernels
        """
        if not self.cuda_available:
            logger.warning("CUDA not available for benchmarking")
            return {}

        logger.info(
            f"🔬 Benchmarking all kernels: N={N}, joints={num_joints}, runs={num_runs}"
        )

        # Generate test data
        thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
        thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)

        kernel_types = [
            "standard",
            "vectorized",
            "memory_optimized",
            "warp_optimized",
            "cache_friendly",
        ]
        results = {}

        for kernel_type in kernel_types:
            logger.info(f"📊 Testing {kernel_type} kernel...")

            # Reset stats for clean measurement
            self.reset_performance_stats()

            times = []
            for run in range(num_runs):
                start_time = time.time()

                try:
                    trajectory = self.joint_trajectory(
                        thetastart,
                        thetaend,
                        2.0,
                        N,
                        5,
                        kernel_type=kernel_type,
                        enable_monitoring=False,
                    )
                    elapsed = time.time() - start_time
                    times.append(elapsed)

                except Exception as e:
                    logger.warning(f"Kernel {kernel_type} failed: {e}")
                    times.append(float("inf"))

            if times and min(times) < float("inf"):
                results[kernel_type] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "all_times": times,
                    "success_rate": sum(1 for t in times if t < float("inf"))
                    / len(times),
                }
            else:
                results[kernel_type] = {
                    "mean_time": float("inf"),
                    "std_time": 0,
                    "min_time": float("inf"),
                    "max_time": float("inf"),
                    "all_times": times,
                    "success_rate": 0,
                }

        # Find best kernel
        best_kernel = min(results.keys(), key=lambda k: results[k]["mean_time"])
        best_time = results[best_kernel]["mean_time"]

        logger.info(f"🏆 Best kernel: {best_kernel} ({best_time*1000:.2f}ms)")

        # Print comparison table
        print("\n📋 Kernel Performance Comparison:")
        print("=" * 70)
        print(f"{'Kernel':<20} {'Mean (ms)':<12} {'Min (ms)':<12} {'Success':<10}")
        print("-" * 70)

        for kernel_type, stats in results.items():
            mean_ms = (
                stats["mean_time"] * 1000
                if stats["mean_time"] < float("inf")
                else float("inf")
            )
            min_ms = (
                stats["min_time"] * 1000
                if stats["min_time"] < float("inf")
                else float("inf")
            )
            success = f"{stats['success_rate']*100:.0f}%"

            marker = "🏆" if kernel_type == best_kernel else "  "
            print(
                f"{marker}{kernel_type:<18} {mean_ms:<12.2f} "
                f"{min_ms:<12.2f} {success:<10}"
            )

        return results

    def __del__(self) -> None:
        """Enhanced destructor with better error handling."""
        try:
            if (
                hasattr(self, "enable_profiling")
                and self.enable_profiling
                and hasattr(self, "cuda_available")
                and self.cuda_available
            ):
                _runtime.profile_stop()
            if hasattr(self, "cleanup_gpu_memory"):
                self.cleanup_gpu_memory()
        except Exception:
            pass  # Ignore errors during cleanup

    def benchmark_performance(
        self, test_cases=None, include_cpu_comparison: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Enhanced performance benchmarking with detailed analysis.

        Args:
            test_cases (list, optional): List of test cases to benchmark.
            include_cpu_comparison (bool): Whether to include CPU vs GPU comparison.

        Returns:
            dict: Comprehensive benchmark results
        """
        if test_cases is None:
            test_cases = [
                {"N": 100, "joints": 6, "name": "Small"},
                {"N": 1000, "joints": 6, "name": "Medium"},
                {"N": 5000, "joints": 6, "name": "Large"},
                {"N": 10000, "joints": 6, "name": "Very Large"},
                {"N": 1000, "joints": 12, "name": "Many joints"},
                {"N": 5000, "joints": 12, "name": "Large + Many joints"},
            ]

        results = {}

        print("\n🚀 Enhanced Performance Benchmarking")
        print("=" * 60)

        for test_case in test_cases:
            N = test_case["N"]
            joints = test_case["joints"]
            name = test_case["name"]

            logger.info(f"Benchmarking {name} case: N={N}, joints={joints}")

            # Generate test data
            thetastart = np.random.uniform(-1, 1, joints).astype(np.float32)
            thetaend = np.random.uniform(-1, 1, joints).astype(np.float32)

            # Reset stats
            self.reset_performance_stats()

            # Test trajectory generation with multiple runs for accuracy
            times = []
            for run in range(3):  # Multiple runs for statistical accuracy
                start_time = time.time()
                trajectory = self.joint_trajectory(thetastart, thetaend, 2.0, N, 5)
                end_time = time.time()
                times.append(end_time - start_time)

            mean_time = np.mean(times)
            std_time = np.std(times)

            # Get performance stats
            stats = self.get_performance_stats()

            results[name] = {
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min(times),
                "max_time": max(times),
                "N": N,
                "joints": joints,
                "stats": stats,
                "used_gpu": stats["gpu_calls"] > 0,
                "trajectory_shape": trajectory["positions"].shape,
                "speedup_achieved": stats.get("speedup_achieved", 0),
                "kernel_used": stats.get("best_kernel_used", "unknown"),
                "elements_per_second": (N * joints) / mean_time,
            }

            # CPU comparison if requested
            if include_cpu_comparison and self.cuda_available:
                # Force CPU execution
                old_threshold = self.cpu_threshold
                self.cpu_threshold = float("inf")  # Force CPU

                cpu_start = time.time()
                cpu_trajectory = self.joint_trajectory(thetastart, thetaend, 2.0, N, 5)
                cpu_time = time.time() - cpu_start

                self.cpu_threshold = old_threshold  # Restore threshold

                if mean_time > 0:
                    actual_speedup = cpu_time / mean_time
                    results[name]["cpu_time"] = cpu_time
                    results[name]["actual_speedup"] = actual_speedup
                else:
                    results[name]["actual_speedup"] = 0

            # Print summary
            gpu_indicator = "🚀 GPU" if results[name]["used_gpu"] else "🖥️  CPU"
            speedup_str = ""
            if (
                "actual_speedup" in results[name]
                and results[name]["actual_speedup"] > 1
            ):
                speedup_str = f" ({results[name]['actual_speedup']:.1f}x speedup)"

            print(
                f"{gpu_indicator} {name}: {mean_time*1000:.2f}"
                f"±{std_time*1000:.2f}ms{speedup_str}"
            )

            logger.info(
                f"{name} benchmark: {mean_time:.4f}s, GPU: {results[name]['used_gpu']}"
            )

        # Print summary table
        print("\n📊 Benchmark Summary:")
        print("-" * 80)
        print(
            f"{'Test Case':<20} {'Time (ms)':<12} {'GPU':<6} "
            f"{'Speedup':<10} {'Throughput':<15}"
        )
        print("-" * 80)

        for name, result in results.items():
            time_ms = result["mean_time"] * 1000
            gpu_used = "✓" if result["used_gpu"] else "✗"
            speedup = (
                f"{result.get('actual_speedup', 0):.1f}x"
                if result.get("actual_speedup", 0) > 1
                else "-"
            )
            throughput = f"{result['elements_per_second']/1e6:.2f} M/s"

            print(
                f"{name:<20} {time_ms:<12.2f} {gpu_used:<6} "
                f"{speedup:<10} {throughput:<15}"
            )

        return results


# Maintain backward compatibility with original class name
class TrajectoryPlanning(OptimizedTrajectoryPlanning):
    """
    Backward compatibility alias for OptimizedTrajectoryPlanning.

    This ensures existing code continues to work while providing
    access to all optimizations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the compatibility wrapper with the optimized planner."""
        super().__init__(*args, **kwargs)
        logger.info("Using OptimizedTrajectoryPlanning (backward compatibility mode)")


# Enhanced utility functions for advanced users
def create_optimized_planner(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    torque_limits=None,
    target_speedup=40.0,
    gpu_memory_mb=None,
    enable_profiling=False,
    kernel_type="auto",
) -> OptimizedTrajectoryPlanning:
    """
    Enhanced factory function to create an optimized trajectory planner.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        torque_limits: Torque limits (optional)
        target_speedup: Target speedup over CPU (default: 40x)
        gpu_memory_mb: GPU memory pool size in MB (optional)
        enable_profiling: Enable CUDA profiling (optional)
        kernel_type: Kernel selection strategy (optional)

    Returns:
        OptimizedTrajectoryPlanning: Configured planner instance
    """
    # Auto-detect optimal settings
    cuda_available = _runtime.check_cuda_availability()

    # Adaptive threshold based on target speedup and problem size
    num_joints = len(joint_limits)

    if cuda_available:
        gpu_props = _runtime.get_gpu_properties()
        if gpu_props:
            sm_count = gpu_props["multiprocessor_count"]
            if target_speedup >= 40:
                threshold = max(50, int(sm_count * 10000 / num_joints))
            elif target_speedup >= 20:
                threshold = max(50, int(sm_count * 5000 / num_joints))
            else:
                threshold = max(50, int(sm_count * 1000 / num_joints))
        else:
            threshold = 1000
    else:
        threshold = float("inf")  # Never use GPU if not available

    # Create planner with optimized settings
    planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        use_cuda=None,  # Auto-detect
        cuda_threshold=threshold,
        memory_pool_size_mb=gpu_memory_mb,
        enable_profiling=enable_profiling,
        auto_optimize=True,
        kernel_type=kernel_type,
        target_speedup=target_speedup,
    )

    logger.info(
        f"Created optimized planner for {num_joints} joints, "
        f"target: {target_speedup}x speedup, CUDA: {cuda_available}"
    )

    return planner


def compare_implementations(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    test_params=None,
    detailed_analysis=True,
) -> Dict[str, Any]:
    """
    Enhanced implementation comparison with detailed kernel analysis.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_params: Test parameters (optional)
        detailed_analysis: Whether to perform detailed kernel comparison

    Returns:
        dict: Comprehensive comparison results
    """
    if test_params is None:
        test_params = {"N": 5000, "Tf": 2.0, "method": 5, "num_runs": 5}

    # Create CPU-only planner
    cpu_planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        use_cuda=False,
    )

    # Create GPU planner (if available)
    gpu_planner = None
    if _runtime.check_cuda_availability():
        gpu_planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits,
            use_cuda=True,
            cuda_threshold=0,  # Force GPU usage
            kernel_type="auto_tune",
        )

    # Generate test data
    num_joints = len(joint_limits)
    thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)

    results = {"cpu": {}, "gpu": {}}

    # Test CPU implementation
    logger.info("Testing CPU implementation...")
    cpu_times = []
    for run in range(test_params.get("num_runs", 3)):
        start_time = time.time()
        cpu_result = cpu_planner.joint_trajectory(
            thetastart,
            thetaend,
            test_params["Tf"],
            test_params["N"],
            test_params["method"],
        )
        cpu_times.append(time.time() - start_time)

    cpu_mean_time = np.mean(cpu_times)
    results["cpu"] = {
        "mean_time": cpu_mean_time,
        "std_time": np.std(cpu_times),
        "min_time": np.min(cpu_times),
        "max_time": np.max(cpu_times),
        "result_shape": cpu_result["positions"].shape,
        "stats": cpu_planner.get_performance_stats(),
    }

    # Test GPU implementation (if available)
    if gpu_planner is not None:
        logger.info("Testing GPU implementation...")

        # Test different kernels if detailed analysis requested
        if detailed_analysis:
            kernel_results = gpu_planner.benchmark_all_kernels(
                N=test_params["N"],
                num_joints=num_joints,
                num_runs=test_params.get("num_runs", 3),
            )
            results["kernel_comparison"] = kernel_results

        # Test best configuration
        gpu_times = []
        for run in range(test_params.get("num_runs", 3)):
            start_time = time.time()
            gpu_result = gpu_planner.joint_trajectory(
                thetastart,
                thetaend,
                test_params["Tf"],
                test_params["N"],
                test_params["method"],
            )
            gpu_times.append(time.time() - start_time)

        gpu_mean_time = np.mean(gpu_times)
        speedup = cpu_mean_time / gpu_mean_time if gpu_mean_time > 0 else 0

        results["gpu"] = {
            "mean_time": gpu_mean_time,
            "std_time": np.std(gpu_times),
            "min_time": np.min(gpu_times),
            "max_time": np.max(gpu_times),
            "result_shape": gpu_result["positions"].shape,
            "stats": gpu_planner.get_performance_stats(),
            "speedup": speedup,
        }

        # Compare accuracy
        pos_diff = np.abs(cpu_result["positions"] - gpu_result["positions"])
        vel_diff = np.abs(cpu_result["velocities"] - gpu_result["velocities"])
        acc_diff = np.abs(cpu_result["accelerations"] - gpu_result["accelerations"])

        results["accuracy"] = {
            "max_pos_diff": np.max(pos_diff),
            "max_vel_diff": np.max(vel_diff),
            "max_acc_diff": np.max(acc_diff),
            "mean_pos_diff": np.mean(pos_diff),
            "mean_vel_diff": np.mean(vel_diff),
            "mean_acc_diff": np.mean(acc_diff),
        }

        # Print comprehensive results
        print("\n🚀 Implementation Comparison Results:")
        print("=" * 50)
        print(
            f"CPU Time: {cpu_mean_time*1000:.2f} ± "
            f"{results['cpu']['std_time']*1000:.2f} ms"
        )
        print(
            f"GPU Time: {gpu_mean_time*1000:.2f} ± "
            f"{results['gpu']['std_time']*1000:.2f} ms"
        )
        print(f"Speedup: {speedup:.1f}x")
        print(f"Max Position Error: {results['accuracy']['max_pos_diff']:.2e}")
        print(f"Mean Position Error: {results['accuracy']['mean_pos_diff']:.2e}")

        if speedup >= 40:
            print("🎯 Achieved 40x+ speedup target!")
        elif speedup >= 20:
            print("⚡ Good speedup achieved!")
        elif speedup >= 5:
            print("✅ Moderate speedup achieved")
        else:
            print("⚠️  Limited speedup - consider larger problem sizes")

        logger.info(f"GPU speedup: {speedup:.2f}x")
    else:
        results["gpu"] = {"available": False}
        logger.info("GPU not available for comparison")

    return results


def benchmark_kernel_performance_comprehensive(
    serial_manipulator, urdf_path, dynamics, joint_limits, test_sizes=None, num_runs=5
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive kernel performance benchmarking across multiple problem sizes.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_sizes: List of (N, joints) tuples to test
        num_runs: Number of runs per test

    Returns:
        dict: Comprehensive benchmark results
    """
    if not _runtime.check_cuda_availability():
        logger.warning("CUDA not available for comprehensive benchmarking")
        return {}

    if test_sizes is None:
        test_sizes = [
            (1000, 6),
            (5000, 6),
            (10000, 6),
            (20000, 6),
            (1000, 12),
            (5000, 12),
            (10000, 12),
        ]

    print("\n🔬 Comprehensive Kernel Performance Benchmarking")
    print("=" * 60)

    all_results = {}

    for N, joints in test_sizes:
        logger.info(f"Testing N={N}, joints={joints}")

        # Create optimized planner
        planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits[:joints],  # Use subset of joints
            use_cuda=True,
            cuda_threshold=0,
            kernel_type="auto_tune",
        )

        # Benchmark all kernels for this problem size
        kernel_results = planner.benchmark_all_kernels(
            N=N, num_joints=joints, num_runs=num_runs
        )

        all_results[f"N{N}_J{joints}"] = {
            "N": N,
            "joints": joints,
            "total_elements": N * joints,
            "kernel_results": kernel_results,
        }

        # Find best kernel for this size
        if kernel_results:
            best_kernel = min(
                kernel_results.keys(), key=lambda k: kernel_results[k]["mean_time"]
            )
            best_time = kernel_results[best_kernel]["mean_time"]
            throughput = (N * joints) / best_time / 1e6

            print(
                f"N={N:5d}, J={joints:2d}: Best={best_kernel:<15} "
                f"Time={best_time*1000:6.2f}ms Throughput={throughput:6.1f}M/s"
            )

    return all_results


del _GenerationMixin, _DynamicsMixin, _CollisionMixin, _PlottingMixin


__all__ = [
    "OptimizedTrajectoryPlanning",
    "TrajectoryPlanning",  # Backward compatibility
    "create_optimized_planner",
    "compare_implementations",
    "benchmark_kernel_performance_comprehensive",
]
