#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernel implementation split by runtime concern."""

from functools import lru_cache
import math
import os
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np

from ._runtime import (
    CUDA_AVAILABLE,
    CUPY_AVAILABLE,
    _get_backend,
    cp,
    cuda,
    float32,
    logger,
)
from .memory import get_memory_pool_stats
from .trajectory_kernels import (
    batch_trajectory_kernel,
    optimized_batch_trajectory_generation,
    optimized_trajectory_generation_monitored,
    trajectory_kernel,
    trajectory_kernel_cache_friendly,
    trajectory_kernel_memory_optimized,
    trajectory_kernel_vectorized,
    trajectory_kernel_warp_optimized,
)
from .field_kernels import optimized_potential_field

_cuda_error: Any = None


def check_cuda_availability() -> bool:
    """Enhanced CUDA availability check with detailed diagnostics."""
    if CUDA_AVAILABLE:
        try:
            devices = cuda.list_devices()
            device = cuda.get_current_device()

            print(f"✅ CUDA is fully operational!")
            print(f"✅ Devices: {len(devices)}")
            for i, dev in enumerate(devices):
                print(f"   Device {i}: {dev}")

            sm_count = device.MULTIPROCESSOR_COUNT
            max_threads = device.MAX_THREADS_PER_BLOCK
            shared_mem = device.MAX_SHARED_MEMORY_PER_BLOCK

            print(f"✅ Current device specs:")
            print(f"   SMs: {sm_count}")
            print(f"   Max threads/block: {max_threads}")
            print(f"   Shared memory: {shared_mem//1024}KB")

            # Performance recommendations
            min_N_for_40x = sm_count * 256 * 4  # 4 blocks per SM, 256 threads each
            print(f"💡 For 40x+ speedup, use N ≥ {min_N_for_40x:,} trajectory points")

            return True

        except Exception as e:
            print(f"⚠️ CUDA available but device query failed: {e}")
            return True
    else:
        print(f"❌ CUDA not available: {_cuda_error}")

        # Provide specific diagnostic help
        if "CUDA_ERROR_NO_DEVICE" in str(_cuda_error):
            print("\n🔧 No CUDA devices found:")
            print("1. Check GPU connection: nvidia-smi")
            print("2. Reinstall drivers: sudo apt install nvidia-driver-535")
            print("3. Reboot system")
        elif "import" in str(_cuda_error).lower():
            print("\n🔧 Installation issue:")
            print("1. Update numba: pip install --upgrade numba")
            print("2. Install CUDA toolkit matching your driver")
            print("3. Install ManipulaPy with GPU: pip install ManipulaPy[gpu-cuda12]")

        return False


def check_cupy_availability() -> bool:
    """Check CuPy availability for additional GPU operations."""
    if not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Install with: pip install cupy-cuda12x",
            UserWarning,
            stacklevel=2,
        )
    return CUPY_AVAILABLE


def make_1d_grid(
    size: int, threads: int = 256
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Create optimal 1D grid for maximum GPU utilization.

    Args:
        size: Total number of elements to cover with one thread each.
        threads: Initial thread-block size; overridden internally based on
            ``size`` for better occupancy.

    Returns:
        Tuple[Tuple[int, ...], Tuple[int, ...]]: ``(blocks, threads)`` launch
        configuration, each a 1-tuple suitable for ``kernel[blocks, threads]``.
    """
    if size <= 0:
        return (1,), (1,)

    # Use larger block sizes for better occupancy
    if size >= 10000:
        threads = 256  # Optimal for most GPUs
    elif size >= 1000:
        threads = 128
    else:
        threads = max(32, 2 ** int(math.log2(size)))

    blocks = (size + threads - 1) // threads
    return (blocks,), (threads,)


def make_2d_grid(
    N: int, num_joints: int, block_size: Tuple[int, int] = (128, 8)
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Create 2D grid configuration for CUDA kernel launch (backward compatibility).

    This is the original function maintained for compatibility.
    For optimal performance, use make_2d_grid_optimized().

    Args:
        N: Number of trajectory time steps (X dimension of the grid).
        num_joints: Number of joints (Y dimension of the grid).
        block_size: Initial ``(threads_x, threads_y)`` block shape; shrunk for
            tiny problems and adjusted to reach a minimum block count.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: ``(grid, block)`` 2D launch
        configuration. Returns ``((1, 1), (1, 1))`` when CUDA is unavailable.
    """
    if not CUDA_AVAILABLE:
        return ((1, 1), (1, 1))

    # Original logic for backward compatibility
    threads_x, threads_y = block_size

    # Shrink block if the problem is tiny
    threads_x = max(4, 1 << int(math.log2(max(1, min(threads_x, N)))))
    threads_y = max(4, 1 << int(math.log2(max(1, min(threads_y, num_joints)))))

    def grid_dims(tx: int, ty: int) -> Tuple[int, int]:
        """Compute block counts for a candidate 2D thread shape.

        Args:
            tx: Threads per block along the X (time) dimension.
            ty: Threads per block along the Y (joint) dimension.

        Returns:
            Tuple[int, int]: Number of blocks ``(blocks_x, blocks_y)`` needed
            to cover ``N`` time steps and ``num_joints`` joints.
        """
        return ((N + tx - 1) // tx, (num_joints + ty - 1) // ty)

    blocks_x, blocks_y = grid_dims(threads_x, threads_y)
    total_blocks = blocks_x * blocks_y

    # Target ≥ 2 × SM blocks for decent load
    try:
        sm_count = (
            cuda.get_current_device().MULTIPROCESSOR_COUNT if CUDA_AVAILABLE else 16
        )
        max_threads_per_block = (
            cuda.get_current_device().MAX_THREADS_PER_BLOCK if CUDA_AVAILABLE else 1024
        )
    except Exception:
        sm_count = 16  # Fallback
        max_threads_per_block = 1024

    min_blocks = sm_count * 2

    # Keep halving X and Y until we hit the target
    toggle = 0
    while total_blocks < min_blocks:
        if toggle == 0 and threads_x > 4:
            threads_x //= 2
        elif toggle == 1 and threads_y > 4:
            threads_y //= 2
        else:
            break
        toggle ^= 1

        # Keep within HW limit
        if threads_x * threads_y > max_threads_per_block:
            if threads_x >= threads_y and threads_x > 4:
                threads_x //= 2
            elif threads_y > 4:
                threads_y //= 2

        blocks_x, blocks_y = grid_dims(threads_x, threads_y)
        total_blocks = blocks_x * blocks_y

    return (blocks_x, blocks_y), (threads_x, threads_y)


def make_2d_grid_optimized(
    N: int, num_joints: int, target_occupancy: float = 0.75
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Create optimal 2D grid configuration targeting specific occupancy for 40x+ speedup.

    Args:
        N: Number of trajectory points
        num_joints: Number of joints
        target_occupancy: Target GPU occupancy (0.5-1.0)

    Returns:
        tuple: ((blocks_x, blocks_y), (threads_x, threads_y))
    """
    if not CUDA_AVAILABLE:
        return ((1, 1), (1, 1))

    device = cuda.get_current_device()
    sm_count = device.MULTIPROCESSOR_COUNT
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK

    # Calculate optimal block size based on problem characteristics
    total_work = N * num_joints

    if total_work >= 1000000:  # Large problems (1M+ elements)
        # Use maximum threads per block for best throughput
        threads_x = min(256, N)
        threads_y = min(max_threads_per_block // threads_x, num_joints)
    elif total_work >= 100000:  # Medium-large problems
        threads_x = min(128, N)
        threads_y = min(max_threads_per_block // threads_x, num_joints)
    elif total_work >= 10000:  # Medium problems
        threads_x = min(64, N)
        threads_y = min(max_threads_per_block // threads_x, num_joints)
    else:  # Small problems
        threads_x = min(32, N)
        threads_y = min(max_threads_per_block // threads_x, num_joints)

    # Ensure threads are multiples of warp size (32) for optimal performance
    threads_x = max(32, (threads_x // 32) * 32)
    threads_y = max(1, threads_y)

    # Recalculate if we exceed max threads
    while threads_x * threads_y > max_threads_per_block:
        if threads_x > threads_y and threads_x > 32:
            threads_x = max(32, threads_x - 32)
        elif threads_y > 1:
            threads_y -= 1
        else:
            break

    # Calculate grid dimensions
    blocks_x = (N + threads_x - 1) // threads_x
    blocks_y = (num_joints + threads_y - 1) // threads_y

    # Ensure sufficient blocks for target occupancy
    total_blocks = blocks_x * blocks_y
    min_blocks_needed = int(sm_count * target_occupancy * 4)  # 4 blocks per SM target

    if total_blocks < min_blocks_needed:
        # Adjust block size to increase block count
        scale_factor = math.sqrt(min_blocks_needed / total_blocks)
        threads_x = max(32, int(threads_x / scale_factor))
        threads_y = max(1, int(threads_y / scale_factor))

        # Recalculate
        blocks_x = (N + threads_x - 1) // threads_x
        blocks_y = (num_joints + threads_y - 1) // threads_y

    return ((blocks_x, blocks_y), (threads_x, threads_y))


def get_gpu_properties() -> Optional[Dict[str, Any]]:
    """Get comprehensive GPU properties for optimization."""
    if not CUDA_AVAILABLE:
        return None

    try:
        device = cuda.get_current_device()
        return {
            "multiprocessor_count": device.MULTIPROCESSOR_COUNT,
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
            "max_block_dim_x": device.MAX_BLOCK_DIM_X,
            "max_block_dim_y": device.MAX_BLOCK_DIM_Y,
            "warp_size": getattr(device, "WARP_SIZE", 32),
            "compute_capability": getattr(device, "COMPUTE_CAPABILITY", (6, 0)),
            "memory_bandwidth_peak_gb_s": 500,  # Approximate, varies by GPU
        }
    except Exception:
        return None


if CUDA_AVAILABLE:

    class CUDAPerformanceMonitor:
        """Advanced performance monitoring for CUDA kernels."""

        def __init__(self) -> None:
            """Initialize empty kernel and memory statistics dictionaries."""
            self.kernel_stats = {}
            self.memory_stats = {}

        def record_kernel_launch(
            self,
            kernel_name: str,
            grid: Tuple[int, ...],
            block: Tuple[int, ...],
            shared_mem: int = 0,
        ) -> None:
            """Accumulate launch counts, block/thread totals, and shared memory for a kernel.

            Args:
                kernel_name: Identifier under which to aggregate statistics.
                grid: Grid dimensions of the launch (1D or 2D tuple of block counts).
                block: Block dimensions of the launch (threads per block per axis).
                shared_mem: Bytes of dynamic shared memory used by the launch.
            """
            if kernel_name not in self.kernel_stats:
                self.kernel_stats[kernel_name] = {
                    "launches": 0,
                    "total_blocks": 0,
                    "total_threads": 0,
                    "total_shared_mem": 0,
                }

            stats = self.kernel_stats[kernel_name]
            stats["launches"] += 1
            stats["total_blocks"] += grid[0] * grid[1] if len(grid) > 1 else grid[0]
            stats["total_threads"] += (
                grid[0] * grid[1] * block[0] * block[1]
                if len(grid) > 1
                else grid[0] * block[0]
            )
            stats["total_shared_mem"] += shared_mem

        def get_stats(self) -> Dict[str, Any]:
            """Return aggregated kernel launch statistics and memory pool statistics."""
            return {
                "kernel_stats": self.kernel_stats,
                "memory_pool_stats": get_memory_pool_stats(),
            }

    _perf_monitor = CUDAPerformanceMonitor()

    # KERNEL CONFIGURATION OPTIMIZATION
    def get_optimal_kernel_config(
        N: int, num_joints: int, kernel_type: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically select optimal kernel and configuration for 40x+ speedup.

        Args:
            N: Number of trajectory points
            num_joints: Number of joints
            kernel_type: "auto", "standard", "vectorized", "memory_optimized",
                        "warp_optimized", or "cache_friendly"

        Returns:
            Configuration dictionary with kernel function and launch parameters
        """
        if not CUDA_AVAILABLE:
            return None

        device = cuda.get_current_device()
        sm_count = device.MULTIPROCESSOR_COUNT
        max_threads = device.MAX_THREADS_PER_BLOCK
        total_work = N * num_joints

        # Auto-select kernel based on problem characteristics
        if kernel_type == "auto":
            if total_work < 50000:
                kernel_type = "standard"
            elif total_work < 500000:
                kernel_type = "vectorized"
            elif total_work < 2000000:
                kernel_type = "memory_optimized"
            else:
                kernel_type = "warp_optimized"

        # Configure based on selected kernel type
        if kernel_type == "vectorized":
            vector_size = 8
            effective_N = (N + vector_size - 1) // vector_size
            threads_x = min(256, max(32, effective_N))
            threads_y = min(max_threads // threads_x, num_joints)
            blocks_x = (effective_N + threads_x - 1) // threads_x
            blocks_y = (num_joints + threads_y - 1) // threads_y
            kernel_func = trajectory_kernel_vectorized

        elif kernel_type == "memory_optimized":
            threads_x = min(128, max(64, N // (sm_count * 2)))
            threads_y = min(max_threads // threads_x, min(16, num_joints))
            blocks_x = min(sm_count * 4, (N + threads_x - 1) // threads_x)
            blocks_y = min(sm_count * 4, (num_joints + threads_y - 1) // threads_y)
            kernel_func = trajectory_kernel_memory_optimized

        elif kernel_type == "warp_optimized":
            # Optimize for warp-level execution
            threads_x = 32  # One warp
            threads_y = min(max_threads // 32, num_joints)
            blocks_x = (N + 31) // 32  # Each block processes 32 time steps
            blocks_y = (num_joints + threads_y - 1) // threads_y
            kernel_func = trajectory_kernel_warp_optimized

        elif kernel_type == "cache_friendly":
            # Use tile-based approach
            threads_x = 64
            threads_y = 8
            blocks_x = (N + 63) // 64
            blocks_y = (num_joints + 7) // 8
            kernel_func = trajectory_kernel_cache_friendly

        else:  # standard
            if num_joints <= 8:
                threads_x, threads_y = 128, min(8, num_joints)
            elif num_joints <= 16:
                threads_x, threads_y = 64, min(16, num_joints)
            else:
                threads_x, threads_y = 32, min(32, num_joints)

            while threads_x * threads_y > max_threads:
                if threads_x > threads_y and threads_x > 32:
                    threads_x = max(32, threads_x - 32)
                elif threads_y > 1:
                    threads_y -= 1
                else:
                    break

            blocks_x = (N + threads_x - 1) // threads_x
            blocks_y = (num_joints + threads_y - 1) // threads_y
            kernel_func = trajectory_kernel

        # Calculate performance metrics
        total_blocks = blocks_x * blocks_y
        theoretical_occupancy = min(100, (total_blocks / (sm_count * 4)) * 100)

        # Estimate performance potential
        elements_per_sm = total_work / sm_count
        expected_speedup_range = (20, 60) if elements_per_sm > 10000 else (5, 20)

        return {
            "kernel_func": kernel_func,
            "kernel_type": kernel_type,
            "grid": (blocks_x, blocks_y),
            "block": (threads_x, threads_y),
            "total_blocks": total_blocks,
            "threads_per_block": threads_x * threads_y,
            "theoretical_occupancy": theoretical_occupancy,
            "expected_speedup_range": expected_speedup_range,
            "elements_per_sm": elements_per_sm,
            "recommended_for_40x": elements_per_sm > 10000,
        }

    # AUTO-TUNING FOR MAXIMUM PERFORMANCE
    @lru_cache(maxsize=64)
    def _best_2d_config(N: int, J: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Auto-tune 2D CUDA kernel launch configuration for optimal performance.

        This function is maintained for backward compatibility with path_planning.py.

        Args:
            N: Number of trajectory time steps (X dimension).
            J: Number of joints (Y dimension).

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: ``(grid, block)`` 2D launch
            configuration. Returns ``((1, 1), (1, 1))`` when CUDA is unavailable.
        """
        if not CUDA_AVAILABLE:
            return ((1, 1), (1, 1))

        # Use the optimized configuration function
        config = get_optimal_kernel_config(N, J, "auto")
        if config:
            return config["grid"], config["block"]

        # Fallback to basic configuration
        return make_2d_grid(N, J)

    def auto_select_optimal_kernel(N: int, num_joints: int) -> str:
        """
        Automatically select the best kernel type for maximum performance.

        Returns kernel type string for get_optimal_kernel_config().

        Args:
            N: Number of trajectory time steps.
            num_joints: Number of joints.

        Returns:
            str: Kernel type name ("standard", "vectorized", "memory_optimized",
            "warp_optimized", or "cache_friendly") chosen from the per-SM work
            load and GPU multiprocessor count.
        """
        total_work = N * num_joints
        device_props = get_gpu_properties()

        if not device_props:
            return "standard"

        sm_count = device_props["multiprocessor_count"]
        elements_per_sm = total_work / sm_count

        # Decision tree based on extensive benchmarking
        if elements_per_sm < 1000:
            return "standard"  # Small problems
        elif elements_per_sm < 10000:
            return "vectorized"  # Medium problems
        elif elements_per_sm < 50000:
            return "memory_optimized"  # Large problems
        elif sm_count >= 40:  # High-end GPUs
            return "warp_optimized"
        else:
            return "cache_friendly"  # Memory-bound scenarios

    # PROFILING AND BENCHMARKING UTILITIES
    def profile_start() -> None:
        """Start CUDA profiling with enhanced monitoring."""
        try:
            cuda.profile_start()
            _perf_monitor.kernel_stats.clear()
        except Exception:
            pass

    def profile_stop() -> Dict[str, Any]:
        """Stop CUDA profiling and return statistics."""
        try:
            cuda.profile_stop()
            return _perf_monitor.get_stats()
        except Exception:
            return {}

    def benchmark_kernel_performance(
        kernel_name: str, *args: Any, num_runs: int = 10, warmup_runs: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Enhanced kernel benchmarking with detailed statistics.

        Args:
            kernel_name: Which high-level routine to benchmark: "trajectory",
                "potential_field", or "batch_trajectory".
            *args: Positional arguments forwarded to the selected routine.
            num_runs: Number of timed runs to average over.
            warmup_runs: Number of untimed warm-up runs to discard JIT/transfer
                overhead.

        Returns:
            Optional[Dict[str, Any]]: Timing statistics (mean/avg, std, min, max,
            median time in seconds, raw timings, and memory pool stats), or None
            when CUDA is unavailable.
        """
        if not _cuda_routing_enabled():
            print(f"Cannot benchmark {kernel_name} - CUDA not available")
            return None

        # Warmup runs
        for _ in range(warmup_runs):
            if kernel_name == "trajectory":
                optimized_trajectory_generation_monitored(
                    *args, enable_monitoring=False
                )
            elif kernel_name == "potential_field":
                optimized_potential_field(*args)
            elif kernel_name == "batch_trajectory":
                optimized_batch_trajectory_generation(*args)
            cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = perf_counter()

            if kernel_name == "trajectory":
                result = optimized_trajectory_generation_monitored(
                    *args, enable_monitoring=False
                )
            elif kernel_name == "potential_field":
                result = optimized_potential_field(*args)
            elif kernel_name == "batch_trajectory":
                result = optimized_batch_trajectory_generation(*args)

            cuda.synchronize()
            times.append(perf_counter() - start)

        # Calculate statistics
        times = np.array(times)
        mean_time = float(np.mean(times))
        stats = {
            # avg_time is an alias for mean_time kept for compatibility
            # with pre-v1.3.2 callers (and tests) that expected the
            # "avg_time" key.
            "avg_time": mean_time,
            "mean_time": mean_time,
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "median_time": float(np.median(times)),
            "all_times": times.tolist(),
            "memory_pool_stats": get_memory_pool_stats(),
        }

        print(f"📊 {kernel_name} benchmark results ({num_runs} runs):")
        print(
            f"   Mean: {stats['mean_time']*1000:.2f} ± {stats['std_time']*1000:.2f} ms"
        )
        print(
            f"   Range: {stats['min_time']*1000:.2f} - {stats['max_time']*1000:.2f} ms"
        )
        print(
            f"   Memory pool hit rate: {stats['memory_pool_stats']['cache_hit_rate']*100:.1f}%"
        )

        return stats

else:

    class CUDAPerformanceMonitor:
        """CPU-only no-op performance monitor used when CUDA is unavailable."""

        def __init__(self) -> None:
            """No-op initializer for the CPU-only performance monitor stub."""
            pass

        def record_kernel_launch(self, *args: Any) -> None:
            """No-op stub since no CUDA kernels are launched on CPU."""
            pass

        def get_stats(self) -> Dict[str, Any]:
            """Return empty statistics for the CPU-only performance monitor stub."""
            return {}

    _perf_monitor = CUDAPerformanceMonitor()

    def get_optimal_kernel_config(
        *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Return no kernel configuration when CUDA is unavailable."""
        return None

    def auto_select_optimal_kernel(*args: Any, **kwargs: Any) -> str:
        """Report that no CUDA kernel can be selected."""
        return "none"

    def _best_2d_config(
        *args: Any, **kwargs: Any
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return a minimal launch shape when CUDA is unavailable."""
        return ((1, 1), (1, 1))

    def profile_start() -> None:
        """No-op CUDA profiler start for CPU-only environments."""
        pass

    def profile_stop() -> Dict[str, Any]:
        """Return empty CUDA profiler stats in CPU-only environments."""
        return {}

    def benchmark_kernel_performance(
        *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Report that CUDA benchmarking is unavailable."""
        print("CUDA benchmarking not available")
        return None


def _cuda_routing_enabled(cuda_available: Optional[bool] = None) -> bool:
    """Return whether the active backend may launch the Numba CUDA kernels."""
    physical_cuda = CUDA_AVAILABLE if cuda_available is None else cuda_available
    return bool(physical_cuda and getattr(_get_backend(), "gpu_capable", False))


def print_performance_recommendations(N: int, num_joints: int) -> None:
    """Print recommendations for achieving 40x+ speedup.

    Args:
        N: Number of trajectory time steps in the target problem.
        num_joints: Number of joints in the target problem.
    """
    total_work = N * num_joints

    print("🚀 ManipulaPy CUDA Performance Recommendations")
    print("=" * 50)
    print(f"Current problem size: {total_work:,} elements ({N:,} × {num_joints})")

    if not CUDA_AVAILABLE:
        print("❌ CUDA not available")
        print("📋 To enable 40x+ speedups:")
        print("   1. Install NVIDIA GPU drivers: nvidia-smi")
        print("   2. Install CUDA toolkit (11.8+ or 12.0+)")
        print("   3. Install GPU support: pip install ManipulaPy[gpu-cuda12]")
        return

    device_props = get_gpu_properties()
    if device_props:
        sm_count = device_props["multiprocessor_count"]
        elements_per_sm = total_work / sm_count

        print(f"✅ GPU detected: {sm_count} SMs")
        print(f"📊 Elements per SM: {elements_per_sm:,.0f}")

        if elements_per_sm > 10000:
            print("✅ Problem size OPTIMAL for 40x+ speedup!")
        elif elements_per_sm > 1000:
            print("⚠️  Good for 10-20x speedup. For 40x+:")
            recommended_N = int(10000 * sm_count / num_joints)
            print(f"   📈 Use N ≥ {recommended_N:,} trajectory points")
        else:
            print("⚠️  Problem too small for maximum speedup:")
            min_N_for_40x = int(10000 * sm_count / num_joints)
            print(f"   📈 For 40x speedup: N ≥ {min_N_for_40x:,}")
            print(f"   📈 For 10x speedup: N ≥ {min_N_for_40x//10:,}")

        print(f"\n💡 Optimization tips:")
        print(f"   🔧 Use quintic trajectories (method=5) for more work per thread")
        print(f"   🔧 Enable pinned memory (use_pinned=True)")
        print(f"   🔧 Use batch processing for multiple trajectories")
        print(f"   🔧 Enable auto-tuning (kernel_type='auto_tune')")

        optimal_kernel = auto_select_optimal_kernel(N, num_joints)
        print(f"   🎯 Recommended kernel: {optimal_kernel}")


def setup_cuda_environment_for_40x_speedup() -> None:
    """Setup CUDA environment variables for maximum performance."""
    import os

    print("🔧 Setting up CUDA environment for 40x+ speedup...")

    # CUDA environment optimizations — setdefault so we never clobber a
    # value the user (or a test harness) explicitly set, only fill in defaults.
    os.environ.setdefault("CUDA_CACHE_DISABLE", "0")  # Enable kernel caching
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Enable async execution
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # Stable device ordering

    # Numba optimizations
    os.environ.setdefault("NUMBA_CUDA_CACHE_SIZE", "2048")  # Larger cache
    os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")  # Reduce warnings

    if CUPY_AVAILABLE and CUDA_AVAILABLE:
        try:
            import cupy as cp

            # Setup CuPy memory pool for optimal allocation.
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2**30)  # 1GB limit
            print("✅ CuPy memory pool configured")
        except Exception as exc:
            print(f"⚠️  CuPy memory pool not configured: {exc}")

    print("✅ CUDA environment optimized for maximum performance")
