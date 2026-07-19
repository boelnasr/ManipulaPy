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

# CPU helpers are defined in ``trajectory`` to keep Numba compilation isolated,
# then registered here so historical facade patches share one runtime source.
_runtime._trajectory_cpu_fallback = _trajectory_cpu_fallback
_runtime._traj_cpu_njit = _traj_cpu_njit


_FORWARDED_RUNTIME_NAMES = frozenset(
    {
        "CUDA_AVAILABLE",
        "CubicTimeScaling",
        "MatrixExp3",
        "MatrixLog3",
        "QuinticTimeScaling",
        "TransToRp",
        "_best_2d_config",
        "_h2d_pinned",
        "_traj_cpu_njit",
        "_trajectory_cpu_fallback",
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
        "logger",
        "make_1d_grid",
        "make_2d_grid",
        "make_2d_grid_optimized",
        "optimized_batch_trajectory_generation",
        "optimized_trajectory_generation",
        "optimized_trajectory_generation_monitored",
        "np",
        "plt",
        "print_performance_recommendations",
        "profile_start",
        "profile_stop",
        "return_cuda_array",
        "setup_cuda_environment_for_40x_speedup",
        "time",
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
        # Split the fixed physical probe from the backend-dependent routing
        # decision: the probe result cannot change over the planner's lifetime,
        # but the active backend can, so unfrozen decision points re-evaluate
        # the routing predicate live against the stored probe.
        physical_cuda = _runtime.check_cuda_availability()
        detected_cuda = _runtime._cuda_routing_enabled(physical_cuda)
        if auto_optimize and detected_cuda:
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
            _runtime.logger.warning("Could not initialise collision checker: %s", exc)
            self.collision_checker = None
            self.potential_field = None

        # ------------------------------------------------------------
        # CUDA feature flags
        # ------------------------------------------------------------
        # Kernel routing is owned by cuda_kernels so direct wrapper calls and
        # planner calls cannot disagree about backend/device capability.
        self._physical_cuda = physical_cuda
        self._forced_cpu = use_cuda is False
        if use_cuda is None:
            self.cuda_available = detected_cuda
        elif use_cuda and not detected_cuda:
            raise RuntimeError(
                "use_cuda=True requested but no GPU-capable backend with CUDA "
                "is active. Select the CuPy backend on a CUDA device."
            )
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
            _runtime.logger.info(
                f"🚀 OptimizedTrajectoryPlanning initialized for {num_joints} joints"
            )
            if target_speedup >= 40:
                min_N_for_target = self.cpu_threshold // num_joints
                _runtime.logger.info(
                    f"💡 For {target_speedup}x speedup, use N ≥ "
                    f"{min_N_for_target:,} trajectory points"
                )

        _runtime.logger.info(
            "Optimised planner – CUDA enabled: %s (threshold %d, kernel: %s)",
            self.cuda_available,
            self.cpu_threshold,
            self.kernel_type,
        )
        if self.gpu_properties:
            _runtime.logger.info(
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
        # Re-consult the central routing predicate so a backend switched after
        # construction cannot leave a stale GPU decision in either direction:
        # the physical probe is fixed at construction, the active backend is
        # read live, and use_cuda=False stays a hard CPU pin. Bare instances
        # built without __init__ fall back to their cuda_available flag.
        if getattr(self, "_forced_cpu", False):
            return False
        physical = getattr(self, "_physical_cuda", self.cuda_available)
        if not _runtime._cuda_routing_enabled(physical):
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
                _runtime.logger.debug(
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

            _runtime.logger.info("GPU memory cleaned up")

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
        # Live routing check: a stale constructor flag must not let CPU
        # timings be reported as per-kernel GPU benchmarks.
        if getattr(self, "_forced_cpu", False) or not _runtime._cuda_routing_enabled(
            getattr(self, "_physical_cuda", self.cuda_available)
        ):
            _runtime.logger.warning("CUDA not available for benchmarking")
            return {}

        _runtime.logger.info(
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
            _runtime.logger.info(f"📊 Testing {kernel_type} kernel...")

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
                    _runtime.logger.warning(f"Kernel {kernel_type} failed: {e}")
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

        _runtime.logger.info(f"🏆 Best kernel: {best_kernel} ({best_time*1000:.2f}ms)")

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

            _runtime.logger.info(f"Benchmarking {name} case: N={N}, joints={joints}")

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

            # CPU comparison if requested (live routing check, mirroring
            # _should_use_gpu: a stale flag or a use_cuda=False planner must
            # not produce a CPU-vs-CPU "speedup" comparison)
            if (
                include_cpu_comparison
                and not getattr(self, "_forced_cpu", False)
                and _runtime._cuda_routing_enabled(
                    getattr(self, "_physical_cuda", self.cuda_available)
                )
            ):
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

            _runtime.logger.info(
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
        _runtime.logger.info(
            "Using OptimizedTrajectoryPlanning (backward compatibility mode)"
        )


from .benchmarks import (  # noqa: E402
    benchmark_kernel_performance_comprehensive,
    compare_implementations,
    create_optimized_planner,
)


del _GenerationMixin, _DynamicsMixin, _CollisionMixin, _PlottingMixin


__all__ = [
    "OptimizedTrajectoryPlanning",
    "TrajectoryPlanning",  # Backward compatibility
    "create_optimized_planner",
    "compare_implementations",
    "benchmark_kernel_performance_comprehensive",
]
