#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Enhanced CUDA Kernels Module - ManipulaPy

This module provides highly optimized CUDA-accelerated functions for trajectory planning and dynamics
computation optimized for 40x+ speedups over CPU implementations.

Key optimizations:
- Advanced 2D/3D grid parallelization with optimal occupancy
- Shared memory utilization for 6x6 matrix operations
- Register-optimized computations avoiding memory spills
- Vectorized kernels processing multiple elements per thread
- Memory-bandwidth optimized kernels for large problems
- Adaptive launch configurations based on problem size
- Pinned memory transfers for improved PCIe bandwidth
- Fused kernels reducing memory bandwidth requirements
- Cache-friendly memory access patterns
- Warp-level optimizations for maximum throughput

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import logging
import math
import os
import warnings
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np
from numba import config as _nb_cfg

# Configure numba for optimal performance
_nb_cfg.CUDA_CACHE_SIZE = "2048"  # Increased cache size for better compilation reuse
_nb_cfg.CUDA_LOW_OCCUPANCY_WARNINGS = False  # Disable warnings for specialized kernels

# Environment toggle for fast math operations
FAST_MATH = bool(int(os.getenv("MANIPULAPY_FASTMATH", "1")))

# Setup logging
logger = logging.getLogger(__name__)


# ENHANCED CUDA DETECTION WITH COMPREHENSIVE ERROR HANDLING
def _cuda_safe_to_probe() -> bool:
    """Check whether the CUDA driver can be initialized without crashing.

    A mismatched or broken NVIDIA driver can raise a hardware-level
    ``SIGSEGV`` *inside* numba's C driver call (e.g. ``cuCtxGetCurrent``),
    which a Python ``try``/``except`` in this process cannot catch — it would
    abort the whole interpreter at import time. To stay safe we run the risky
    initialization in a throwaway subprocess: if the child segfaults or hangs,
    only the child dies and we fall back to CPU instead of crashing the import.

    Set ``MANIPULAPY_SKIP_CUDA_PROBE=1`` to skip this check (e.g. when the
    subprocess cost is undesirable and the driver is known good).

    Returns:
        bool: ``True`` if a child process initialized CUDA cleanly, else ``False``.
    """
    if os.getenv("NUMBA_DISABLE_CUDA", "0") == "1":
        return False
    if os.getenv("MANIPULAPY_SKIP_CUDA_PROBE", "0") == "1":
        return True
    import subprocess
    import sys

    probe = (
        "from numba import cuda\n"
        "import numpy as np\n"
        "assert cuda.is_available()\n"
        "cuda.list_devices()\n"
        "cuda.get_current_device()\n"
        "d = cuda.device_array(8, dtype=np.float32)\n"
        "cuda.synchronize()\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _detect_cuda_capability() -> Tuple[bool, Any, Any, Any, Optional[str]]:
    """
    Comprehensive CUDA detection with detailed diagnostics and error handling.

    Returns:
        tuple: (cuda_available, cuda_module, float32, int32, error_message)
    """
    try:
        # Step 0: Probe the driver in a subprocess first. A broken driver can
        # SIGSEGV inside numba's C call, which try/except here cannot catch, so
        # we never touch it in-process unless a sacrificial child survived.
        if not _cuda_safe_to_probe():
            return (
                False,
                None,
                None,
                None,
                "CUDA driver probe failed (unavailable or driver crash) - using CPU",
            )

        # Step 1: Import numba.cuda with proper error handling
        from numba import cuda, float32, int32

        # Step 2: Check basic CUDA availability
        try:
            cuda_available = cuda.is_available()
            if not cuda_available:
                return (
                    False,
                    None,
                    None,
                    None,
                    "CUDA runtime not available - likely no GPU or driver issues",
                )
        except Exception as e:
            return False, None, None, None, f"CUDA availability check failed: {e}"

        # Step 3: Verify device detection
        try:
            devices = cuda.list_devices()
            if not devices:
                return False, None, None, None, "No CUDA devices detected"
        except Exception as e:
            return False, None, None, None, f"Device enumeration failed: {e}"

        # Step 4: Test basic GPU operations
        try:
            # Test memory allocation
            test_array = cuda.device_array(100, dtype=np.float32)

            # Test basic kernel compilation
            @cuda.jit
            def test_kernel(arr) -> None:
                """Write each thread's index into the array to validate execution.

                Args:
                    arr: 1D device array (in-place output buffer); each element
                        ``arr[idx]`` is set to ``float(idx)`` for the thread's
                        global grid index.
                """
                idx = cuda.grid(1)
                if idx < arr.shape[0]:
                    arr[idx] = float32(idx)

            # Get device properties
            current_device = cuda.get_current_device()
            sm_count = current_device.MULTIPROCESSOR_COUNT
            max_threads = current_device.MAX_THREADS_PER_BLOCK

            # Test kernel execution
            test_kernel[1, 64](test_array)
            cuda.synchronize()

            # Verify results
            result = test_array.copy_to_host()
            if not np.allclose(result[:10], np.arange(10, dtype=np.float32)):
                return False, None, None, None, "CUDA kernel execution test failed"

            del test_array

            logger.info(
                f"✅ CUDA fully operational: {len(devices)} device(s), {sm_count} SMs, {max_threads} max threads/block"
            )
            return True, cuda, float32, int32, None

        except Exception as e:
            return False, None, None, None, f"CUDA functionality test failed: {e}"

    except ImportError as e:
        return False, None, None, None, f"numba.cuda import failed: {e}"
    except Exception as e:
        return False, None, None, None, f"Unexpected CUDA detection error: {e}"


# Perform CUDA detection
CUDA_AVAILABLE, cuda, float32, int32, _cuda_error = _detect_cuda_capability()

# Mock CUDA objects for graceful degradation
if not CUDA_AVAILABLE:

    class MockCuda:
        @staticmethod
        def jit(func=None, device=False, inline=False, fastmath=False) -> Any:
            """Return a stub decorator whose wrapped kernel raises on call.

            Args:
                func: Kernel function to wrap, or None when used with arguments
                    as a decorator factory.
                device: Ignored stub flag mirroring ``numba.cuda.jit`` for a
                    device function.
                inline: Ignored stub flag mirroring ``numba.cuda.jit`` inlining.
                fastmath: Ignored stub flag mirroring ``numba.cuda.jit`` fast-math.

            Returns:
                A wrapper callable that raises ``RuntimeError`` on invocation, or
                the same wrapper already applied to ``func`` when ``func`` is given.
            """
            def wrapper(*args, **kwargs) -> NoReturn:
                """Raise because no CUDA device is available to run the kernel."""
                raise RuntimeError(
                    f"CUDA not available: {_cuda_error}\n"
                    "For 40x+ speedups, install CUDA support:\n"
                    "1. Install NVIDIA drivers: nvidia-smi\n"
                    "2. Install CUDA toolkit (11.8+ or 12.0+)\n"
                    "3. Install ManipulaPy with GPU support:\n"
                    "   pip install ManipulaPy[gpu-cuda12]\n"
                    "4. Verify: python -c 'from numba import cuda; print(cuda.is_available())'"
                )

            return wrapper if func is None else wrapper(func)

        @staticmethod
        def grid(dim) -> int:
            """Return 0 as the thread index since no real grid exists.

            Args:
                dim: Grid dimensionality requested (1, 2, or 3); ignored by the
                    CUDA-less stub.

            Returns:
                int: Always 0, the only valid index in the degenerate fallback.
            """
            return 0

        @staticmethod
        def device_array(*args, **kwargs) -> NoReturn:
            """Raise because device memory cannot be allocated without CUDA."""
            raise RuntimeError(f"CUDA not available: {_cuda_error}")

        @staticmethod
        def to_device(*args, **kwargs) -> NoReturn:
            """Raise because host-to-device transfer needs an unavailable CUDA device."""
            raise RuntimeError(f"CUDA not available: {_cuda_error}")

        @staticmethod
        def pinned_array(*args, **kwargs) -> NoReturn:
            """Raise because pinned host memory cannot be allocated without CUDA."""
            raise RuntimeError(f"CUDA not available: {_cuda_error}")

        @staticmethod
        def is_available() -> bool:
            """Report that CUDA is not available."""
            return False

        @staticmethod
        def list_devices() -> list:
            """Return an empty device list since no CUDA device exists."""
            return []

        @staticmethod
        def synchronize() -> None:
            """No-op synchronization stub for the CUDA-less fallback."""
            pass

        @staticmethod
        def get_current_device() -> Any:
            """Return a mock device exposing minimal hardware property defaults."""
            class MockDevice:
                MULTIPROCESSOR_COUNT = 1
                MAX_THREADS_PER_BLOCK = 1024
                MAX_SHARED_MEMORY_PER_BLOCK = 48 * 1024
                MAX_BLOCK_DIM_X = 1024
                MAX_BLOCK_DIM_Y = 1024
                WARP_SIZE = 32
                COMPUTE_CAPABILITY = (6, 0)

            return MockDevice()

        @staticmethod
        def shared() -> Any:
            """Return a mock shared-memory namespace whose array() raises."""
            class SharedMock:
                @staticmethod
                def array(*args, **kwargs) -> NoReturn:
                    """Raise because shared memory needs an unavailable CUDA device."""
                    raise RuntimeError(f"CUDA not available: {_cuda_error}")

            return SharedMock()

        blockIdx = type("blockIdx", (), {"x": 0, "y": 0, "z": 0})()
        blockDim = type("blockDim", (), {"x": 1, "y": 1, "z": 1})()
        threadIdx = type("threadIdx", (), {"x": 0, "y": 0, "z": 0})()

        @staticmethod
        def syncthreads() -> None:
            """No-op thread-barrier stub for the CUDA-less fallback."""
            pass

    cuda = MockCuda()
    if float32 is None:
        float32 = np.float32
    if int32 is None:
        int32 = np.int32

# Check CuPy availability
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Use float32 for optimal GPU performance
float_t = float32


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


# Pinned-memory opt-in: numba.cuda.pinned_array() segfaults (SIGSEGV, not a
# catchable Python exception) on certain numba+driver combinations — e.g.
# numba 0.65 + NVIDIA driver 580 (CUDA 13 ABI). The crash happens in
# numba/cuda/api.py during cuMemHostRegister, before the try/except below
# can ever fire. Keep the path off by default so users on broken combos
# don't lose their Python process; opt back in with
# MANIPULAPY_USE_PINNED_MEMORY=1 when the combo is known good (numba <=
# 0.59 with driver 535, or numba 0.66+ with the upstream fix landed).
_PINNED_MEMORY_OPT_IN = os.environ.get("MANIPULAPY_USE_PINNED_MEMORY", "0").lower() in (
    "1",
    "true",
    "yes",
)


# ENHANCED MEMORY MANAGEMENT
def _h2d_pinned(arr: np.ndarray) -> Any:
    """Host-to-device transfer with optional pinned-memory acceleration.

    Pinned memory delivers ~3x peak transfer bandwidth on large arrays, but
    ``cuda.pinned_array`` is currently incompatible with several modern
    numba+driver combinations (see ``_PINNED_MEMORY_OPT_IN`` above). Plain
    ``cuda.to_device`` is correct on every supported configuration; pinned
    transfers are a pure performance optimisation that must be opted in.

    Args:
        arr: Host ndarray to copy to the device. Forced to C-contiguous layout
            if it is not already.

    Returns:
        A numba CUDA device array holding a copy of ``arr``.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    # Ensure contiguous memory layout
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # Safe default: skip pinned memory entirely unless explicitly enabled.
    if not _PINNED_MEMORY_OPT_IN:
        return cuda.to_device(arr)

    # Opt-in pinned-memory path (~3x bandwidth on supported configs).
    try:
        pinned_arr = cuda.pinned_array(arr.shape, dtype=arr.dtype)
        pinned_arr[:] = arr
        return cuda.to_device(pinned_arr)
    except Exception:
        # If pinned_array raised a real Python exception we can still fall
        # back; segfaults bypass this branch (process is already gone).
        return cuda.to_device(arr)


# OPTIMAL GRID CONFIGURATION FOR 40x+ SPEEDUP
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


# CPU FALLBACK IMPLEMENTATION
def trajectory_cpu_fallback(
    thetastart: np.ndarray,
    thetaend: np.ndarray,
    Tf: float,
    N: int,
    method: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized CPU fallback using NumPy vectorization.

    Args:
        thetastart: (num_joints,) ndarray of starting joint angles, radians.
        thetaend: (num_joints,) ndarray of ending joint angles, radians.
        Tf: Total trajectory duration, seconds. Values <= 0 collapse to the
            start configuration with zero velocity and acceleration.
        N: Number of trajectory time steps. Values <= 1 collapse to the start
            configuration.
        method: Time-scaling polynomial order: 3 for cubic, 5 for quintic, any
            other value (e.g. 1) for linear.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
        traj_acc)``, each an ``(N, num_joints)`` float32 ndarray of joint
        positions (radians), velocities (radians/s), and accelerations
        (radians/s^2).
    """
    num_joints = len(thetastart)

    # Degenerate inputs collapse to "sit at start": s = s_dot = s_ddot = 0.
    # Matches the GPU kernels' N<=1 guard and avoids the divide-by-zero
    # RuntimeWarning when callers pass Tf=0 (regression coverage in
    # test_zero_time_trajectory accepts NaN but the warning is noisy).
    if N <= 1 or Tf <= 0.0:
        s = np.zeros(N, dtype=np.float32)
        s_dot = np.zeros(N, dtype=np.float32)
        s_ddot = np.zeros(N, dtype=np.float32)
    else:
        # Vectorized time computation
        t = np.linspace(0, Tf, N, dtype=np.float32)
        tau = t / Tf

        # Vectorized time scaling
        if method == 3:  # Cubic
            s = 3.0 * tau**2 - 2.0 * tau**3
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:  # Quintic
            tau2 = tau**2
            tau3 = tau**3
            tau4 = tau**4
            tau5 = tau**5
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
            s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = np.ones_like(tau) / Tf
            s_ddot = np.zeros_like(tau)

    # Vectorized trajectory computation
    delta = thetaend - thetastart
    traj_pos = thetastart[np.newaxis, :] + s[:, np.newaxis] * delta[np.newaxis, :]
    traj_vel = s_dot[:, np.newaxis] * delta[np.newaxis, :]
    traj_acc = s_ddot[:, np.newaxis] * delta[np.newaxis, :]

    return (
        traj_pos.astype(np.float32),
        traj_vel.astype(np.float32),
        traj_acc.astype(np.float32),
    )


# CUDA KERNELS FOR 40x+ SPEEDUP
if CUDA_AVAILABLE:

    jit_kwargs = {"fastmath": FAST_MATH}

    @cuda.jit(device=True, inline=True, **jit_kwargs)
    def matrix_vector_multiply_6x6(M, v, result) -> None:
        """Optimized 6x6 matrix-vector multiplication using registers.

        Args:
            M: (6, 6) device array, the matrix operand.
            v: (6,) device array, the vector operand.
            result: (6,) device array, in-place output buffer set to ``M @ v``.
        """
        # Unrolled for maximum performance
        result[0] = (
            M[0, 0] * v[0]
            + M[0, 1] * v[1]
            + M[0, 2] * v[2]
            + M[0, 3] * v[3]
            + M[0, 4] * v[4]
            + M[0, 5] * v[5]
        )
        result[1] = (
            M[1, 0] * v[0]
            + M[1, 1] * v[1]
            + M[1, 2] * v[2]
            + M[1, 3] * v[3]
            + M[1, 4] * v[4]
            + M[1, 5] * v[5]
        )
        result[2] = (
            M[2, 0] * v[0]
            + M[2, 1] * v[1]
            + M[2, 2] * v[2]
            + M[2, 3] * v[3]
            + M[2, 4] * v[4]
            + M[2, 5] * v[5]
        )
        result[3] = (
            M[3, 0] * v[0]
            + M[3, 1] * v[1]
            + M[3, 2] * v[2]
            + M[3, 3] * v[3]
            + M[3, 4] * v[4]
            + M[3, 5] * v[5]
        )
        result[4] = (
            M[4, 0] * v[0]
            + M[4, 1] * v[1]
            + M[4, 2] * v[2]
            + M[4, 3] * v[3]
            + M[4, 4] * v[4]
            + M[4, 5] * v[5]
        )
        result[5] = (
            M[5, 0] * v[0]
            + M[5, 1] * v[1]
            + M[5, 2] * v[2]
            + M[5, 3] * v[3]
            + M[5, 4] * v[4]
            + M[5, 5] * v[5]
        )

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """Each thread computes its own time scaling — no shared memory race.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= N or j_idx >= thetastart.shape[0]:
            return

        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

        if method == 3:  # Cubic
            tau2 = tau * tau
            tau3 = tau2 * tau
            s = 3.0 * tau2 - 2.0 * tau3
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:  # Quintic
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
            s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
        else:  # Linear
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        start_angle = thetastart[j_idx]
        delta_angle = thetaend[j_idx] - start_angle

        traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
        traj_vel[t_idx, j_idx] = s_dot * delta_angle
        traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_vectorized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Vectorized trajectory kernel with correct 8-parameter signature.
        Each thread processes multiple time steps for better throughput.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        VECTOR_SIZE = 8  # Each thread processes 8 time steps

        t_base = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) * VECTOR_SIZE
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if j_idx >= thetastart.shape[0]:
            return

        # Shared memory for joint data
        shared_joints = cuda.shared.array((32, 2), dtype=float32)

        # Load joint data to shared memory
        if cuda.threadIdx.x == 0 and j_idx < 32:
            start_val = thetastart[j_idx]
            shared_joints[j_idx, 0] = start_val
            shared_joints[j_idx, 1] = thetaend[j_idx] - start_val

        cuda.syncthreads()

        # Get joint data
        if j_idx < 32:
            start_angle = shared_joints[j_idx, 0]
            delta_angle = shared_joints[j_idx, 1]
        else:
            start_angle = thetastart[j_idx]
            delta_angle = thetaend[j_idx] - start_angle

        # Process VECTOR_SIZE time steps
        for i in range(VECTOR_SIZE):
            t_idx = t_base + i
            if t_idx >= N:
                break

            # Compute time scaling
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic - optimized computation
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            # Store results
            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_memory_optimized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Memory-bandwidth optimized kernel with correct 8-parameter signature.
        Uses grid-stride loops for better memory utilization.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        # Grid-stride loop for better memory utilization
        stride_t = cuda.gridDim.x * cuda.blockDim.x
        stride_j = cuda.gridDim.y * cuda.blockDim.y

        t_start = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_start = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Shared memory for coefficients and joint data
        shared_data = cuda.shared.array(
            (32, 4), dtype=float32
        )  # [start, delta, s, s_dot]

        # Process joints in chunks
        for j_idx in range(j_start, thetastart.shape[0], stride_j):
            # Load joint data to shared memory
            local_j = cuda.threadIdx.y
            if local_j < 32 and j_idx < thetastart.shape[0]:
                start_val = thetastart[j_idx]
                shared_data[local_j, 0] = start_val
                shared_data[local_j, 1] = thetaend[j_idx] - start_val

            cuda.syncthreads()

            # Process time steps with grid-stride
            for t_idx in range(t_start, N, stride_t):
                # Compute time scaling
                tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

                if method == 5:  # Quintic
                    tau_sq = tau * tau
                    tau_cb = tau_sq * tau
                    s = tau_cb * (10.0 + tau * (-15.0 + 6.0 * tau))
                    s_dot = tau_sq * (30.0 + tau * (-60.0 + 30.0 * tau)) / Tf
                    s_ddot = tau * (60.0 + tau * (-180.0 + 120.0 * tau)) / (Tf * Tf)
                elif method == 3:  # Cubic
                    tau_sq = tau * tau
                    s = tau_sq * (3.0 - 2.0 * tau)
                    s_dot = 6.0 * tau * (1.0 - tau) / Tf
                    s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
                else:  # Linear
                    s = tau
                    s_dot = 1.0 / Tf
                    s_ddot = 0.0

                # Use shared memory data if available
                if local_j < 32 and j_idx < thetastart.shape[0]:
                    start_angle = shared_data[local_j, 0]
                    delta_angle = shared_data[local_j, 1]
                else:
                    start_angle = (
                        thetastart[j_idx] if j_idx < thetastart.shape[0] else 0.0
                    )
                    delta_angle = (
                        (thetaend[j_idx] - start_angle)
                        if j_idx < thetastart.shape[0]
                        else 0.0
                    )

                # Store results
                if j_idx < thetastart.shape[0]:
                    traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
                    traj_vel[t_idx, j_idx] = s_dot * delta_angle
                    traj_acc[t_idx, j_idx] = s_ddot * delta_angle

            cuda.syncthreads()

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_warp_optimized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Warp-level optimized kernel with correct 8-parameter signature.
        Uses warp-level primitives for maximum throughput.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        # Warp-level indexing
        warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
        lane_id = cuda.threadIdx.x % 32

        # Each warp processes 32 consecutive time steps
        t_base = warp_id * 32
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if j_idx >= thetastart.shape[0]:
            return

        # Load joint data (broadcast across warp)
        start_angle = thetastart[j_idx]
        delta_angle = thetaend[j_idx] - start_angle

        # Each thread in warp processes one time step
        t_idx = t_base + lane_id

        if t_idx < N:
            # Optimized time scaling computation
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            # Coalesced memory writes
            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_cache_friendly(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Cache-friendly kernel with correct 8-parameter signature.
        Uses tiled computation to maximize cache utilization.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        TILE_SIZE_T = 64  # Time tile size
        TILE_SIZE_J = 8  # Joint tile size

        # Block-level tiling
        t_tile_start = cuda.blockIdx.x * TILE_SIZE_T
        j_tile_start = cuda.blockIdx.y * TILE_SIZE_J

        # Thread indices within tile
        t_local = cuda.threadIdx.x
        j_local = cuda.threadIdx.y

        # Global indices
        t_idx = t_tile_start + t_local
        j_idx = j_tile_start + j_local

        # Shared memory for tile data
        shared_joints = cuda.shared.array((TILE_SIZE_J, 2), dtype=float32)
        shared_time = cuda.shared.array((TILE_SIZE_T, 3), dtype=float32)

        # Load joint data to shared memory
        if t_local == 0 and j_idx < thetastart.shape[0]:
            start_val = thetastart[j_idx]
            shared_joints[j_local, 0] = start_val
            shared_joints[j_local, 1] = thetaend[j_idx] - start_val

        # Load time scaling data to shared memory
        if j_local == 0 and t_idx < N:
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            shared_time[t_local, 0] = s
            shared_time[t_local, 1] = s_dot
            shared_time[t_local, 2] = s_ddot

        cuda.syncthreads()

        # Compute trajectory using shared memory data
        if t_idx < N and j_idx < thetastart.shape[0]:
            start_angle = shared_joints[j_local, 0]
            delta_angle = shared_joints[j_local, 1]
            s = shared_time[t_local, 0]
            s_dot = shared_time[t_local, 1]
            s_ddot = shared_time[t_local, 2]

            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    # DYNAMICS KERNELS - FIXED SIGNATURES
    @cuda.jit(**jit_kwargs)
    def inverse_dynamics_kernel(
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector,
        Ftip,
        Glist,
        Slist,
        M,
        torques_trajectory,
        torque_limits,
    ) -> None:
        """
        FIXED: Inverse dynamics kernel with correct 10-parameter signature.
        Removed the problematic 'stream' parameter that was causing the mismatch.

        Uses a simplified per-joint dynamics model (diagonal inertia, linear
        Coriolis term, scalar gravity contribution) rather than full recursive
        Newton-Euler.

        Args:
            thetalist_trajectory: (N, num_joints) device array of joint angles,
                radians.
            dthetalist_trajectory: (N, num_joints) device array of joint
                velocities, radians/s.
            ddthetalist_trajectory: (N, num_joints) device array of joint
                accelerations, radians/s^2.
            gravity_vector: (3,) device array, gravitational acceleration; only
                the z component is used.
            Ftip: External wrench at the tip (unused in this simplified kernel).
            Glist: (num_joints, *, *) device array of spatial inertia matrices;
                its diagonal supplies the effective inertia term.
            Slist: (>=num_joints, >=num_joints) device array of screw axes; its
                diagonal supplies the velocity-coupling term.
            M: Home configuration matrix (unused in this simplified kernel).
            torques_trajectory: (N, num_joints) device array, in-place output
                buffer for computed joint torques, clamped to ``torque_limits``.
            torque_limits: (num_joints, 2) device array of ``[min, max]`` torque
                bounds per joint.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if (
            t_idx >= thetalist_trajectory.shape[0]
            or j_idx >= thetalist_trajectory.shape[1]
        ):
            return

        # Load trajectory data
        theta_j = thetalist_trajectory[t_idx, j_idx]
        dtheta_j = dthetalist_trajectory[t_idx, j_idx]
        ddtheta_j = ddthetalist_trajectory[t_idx, j_idx]

        # Simplified dynamics computation with bounds checking
        M_contrib = (
            Glist[j_idx, j_idx, j_idx]
            if (
                j_idx < Glist.shape[0]
                and j_idx < Glist.shape[1]
                and j_idx < Glist.shape[2]
            )
            else 1.0
        )

        c_j = (
            Slist[j_idx, j_idx] * dtheta_j
            if (j_idx < Slist.shape[0] and j_idx < Slist.shape[1])
            else 0.0
        )

        g_j = gravity_vector[2] * 0.1 if gravity_vector.shape[0] > 2 else 0.0

        # Compute torque
        tau = M_contrib * ddtheta_j + c_j + g_j

        # Apply torque limits
        if j_idx < torque_limits.shape[0]:
            tau = max(torque_limits[j_idx, 0], min(tau, torque_limits[j_idx, 1]))

        torques_trajectory[t_idx, j_idx] = tau

    @cuda.jit(**jit_kwargs)
    def forward_dynamics_kernel(
        thetalist,
        dthetalist,
        taumat,
        g,
        Ftipmat,
        dt,
        intRes,
        Glist,
        Slist,
        M,
        thetamat,
        dthetamat,
        ddthetamat,
        joint_limits,
    ) -> None:
        """Forward dynamics kernel.

        Each thread integrates from the initial state up to its own ``t_idx``,
        avoiding the temporal data race in the previous version (which read
        ``thetamat[t_idx-1]`` while parallel threads at lower ``t_idx`` may
        not have written that row yet). Cost is O(t_idx * intRes) per
        thread instead of O(intRes), but correctness no longer depends on
        warp scheduling.

        Uses a simplified per-joint dynamics model (diagonal inertia, scalar
        gravity) rather than full recursive Newton-Euler.

        Args:
            thetalist: (num_joints,) device array of initial joint angles, radians.
            dthetalist: (num_joints,) device array of initial joint velocities,
                radians/s.
            taumat: (N, num_joints) device array of applied joint torques per
                time step; ``taumat[i]`` advances state into row ``i``.
            g: (3,) device array, gravitational acceleration; only the z
                component is used.
            Ftipmat: External tip wrench per step (unused in this simplified
                kernel).
            dt: Time step between trajectory rows, seconds.
            intRes: Number of Euler sub-integration steps per ``dt``.
            Glist: (num_joints, *, *) device array of spatial inertia matrices;
                its diagonal supplies the inverse-inertia term.
            Slist: Screw axes (unused in this simplified kernel).
            M: Home configuration matrix (unused in this simplified kernel).
            thetamat: (N, num_joints) device array, in-place output buffer for
                integrated joint angles, radians.
            dthetamat: (N, num_joints) device array, in-place output buffer for
                integrated joint velocities, radians/s.
            ddthetamat: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            joint_limits: (num_joints, 2) device array of ``[min, max]`` angle
                limits used to clamp integrated positions.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= taumat.shape[0] or j_idx >= thetalist.shape[0]:
            return

        current_theta = thetalist[j_idx]
        current_dtheta = dthetalist[j_idx]
        dt_step = dt / intRes
        ddtheta = 0.0

        if t_idx == 0:
            thetamat[t_idx, j_idx] = current_theta
            dthetamat[t_idx, j_idx] = current_dtheta
            ddthetamat[t_idx, j_idx] = 0.0
            return

        # Walk from t=1 to this thread's t_idx, matching the CPU path's
        # convention that row 0 is the initial state and taumat[i] advances
        # state row i. This remains independent per thread.
        for step in range(1, t_idx + 1):
            tau = taumat[step, j_idx]
            for _ in range(intRes):
                M_inv = (
                    1.0 / Glist[j_idx, j_idx, j_idx]
                    if (
                        j_idx < Glist.shape[0]
                        and j_idx < Glist.shape[1]
                        and j_idx < Glist.shape[2]
                        and Glist[j_idx, j_idx, j_idx] != 0.0
                    )
                    else 1.0
                )
                g_force = g[2] * 0.1 if g.shape[0] > 2 else 0.0
                ddtheta = (tau - g_force) * M_inv

                current_dtheta += ddtheta * dt_step
                current_theta += current_dtheta * dt_step

                if j_idx < joint_limits.shape[0]:
                    current_theta = max(
                        joint_limits[j_idx, 0],
                        min(current_theta, joint_limits[j_idx, 1]),
                    )

        thetamat[t_idx, j_idx] = current_theta
        dthetamat[t_idx, j_idx] = current_dtheta
        ddthetamat[t_idx, j_idx] = ddtheta

    @cuda.jit(**jit_kwargs)
    def cartesian_trajectory_kernel(
        pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """Cartesian trajectory kernel.

        Each thread computes its own time scaling (no shared memory) so the
        scaling matches its own ``t_idx``. Quintic acceleration uses the
        full ``60 tau (1 - tau) (1 - 2 tau) / Tf^2`` form, and the linear
        method (1) is no longer silently zeroed.

        Args:
            pstart: (3,) device array, starting Cartesian position.
            pend: (3,) device array, ending Cartesian position.
            traj_pos: (N, 3) device array, in-place output buffer for Cartesian
                positions.
            traj_vel: (N, 3) device array, in-place output buffer for Cartesian
                velocities.
            traj_acc: (N, 3) device array, in-place output buffer for Cartesian
                accelerations.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        coord_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= N or coord_idx >= 3:
            return

        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)
        if method == 3:  # Cubic
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
        elif method == 5:  # Quintic
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
            s_ddot = 60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        dp = pend[coord_idx] - pstart[coord_idx]

        traj_pos[t_idx, coord_idx] = s * dp + pstart[coord_idx]
        traj_vel[t_idx, coord_idx] = s_dot * dp
        traj_acc[t_idx, coord_idx] = s_ddot * dp

    @cuda.jit(**jit_kwargs)
    def fused_potential_gradient_kernel(
        positions, goal, obstacles, potential, gradient, influence_distance
    ) -> None:
        """
        FIXED: Fused potential gradient kernel with correct 6-parameter signature.
        Removed the problematic 'stream' parameter.

        Computes a combined attractive (toward goal) and repulsive (away from
        obstacles within ``influence_distance``) potential field and its
        gradient for each query position.

        Args:
            positions: (N, 3) device array of query point positions.
            goal: (3,) device array, attractive goal position.
            obstacles: (num_obstacles, 3) device array of obstacle positions.
            potential: (N,) device array, in-place output buffer for the total
                potential at each position.
            gradient: (N, >=3) device array, in-place output buffer for the
                potential gradient (x, y, z) at each position.
            influence_distance: Repulsive influence radius; obstacles farther
                than this contribute nothing.
        """
        idx = cuda.grid(1)
        if idx >= positions.shape[0]:
            return

        influence_distance_inv = (
            1.0 / influence_distance if influence_distance > 0.0 else 0.0
        )

        # Load position
        pos_x = positions[idx, 0]
        pos_y = positions[idx, 1]
        pos_z = positions[idx, 2]

        # Attractive potential
        diff_x = pos_x - goal[0]
        diff_y = pos_y - goal[1]
        diff_z = pos_z - goal[2]

        attractive_pot = 0.5 * (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
        grad_x = diff_x
        grad_y = diff_y
        grad_z = diff_z

        # Repulsive potential
        repulsive_pot = 0.0
        for obs in range(obstacles.shape[0]):
            obs_diff_x = pos_x - obstacles[obs, 0]
            obs_diff_y = pos_y - obstacles[obs, 1]
            obs_diff_z = pos_z - obstacles[obs, 2]

            dist_sq = (
                obs_diff_x * obs_diff_x
                + obs_diff_y * obs_diff_y
                + obs_diff_z * obs_diff_z
            )

            if dist_sq > 0.0 and dist_sq < influence_distance * influence_distance:
                # math.rsqrt is supported as a CUDA intrinsic in numba <=
                # 0.59 but was dropped in 0.65. 1.0 / math.sqrt(...) is
                # portable across all numba versions and lowers to the
                # same PTX (rsqrt.approx.f32) under -ffast-math.
                dist_inv = 1.0 / math.sqrt(dist_sq)
                influence_term = dist_inv - influence_distance_inv
                repulsive_term = 0.5 * influence_term * influence_term
                repulsive_pot += repulsive_term

                # ∇U_rep = (1/d - 1/d_0) * (-1/d^3) * (pos - obstacle).
                # Force = -∇U_rep then points pos -> away_from_obstacle, which
                # is what a repulsive potential field is meant to produce. The
                # previous code dropped the leading minus, so the resulting
                # gradient pulled the robot toward obstacles.
                grad_factor = -influence_term * dist_inv * dist_inv * dist_inv
                grad_x += grad_factor * obs_diff_x
                grad_y += grad_factor * obs_diff_y
                grad_z += grad_factor * obs_diff_z

        potential[idx] = attractive_pot + repulsive_pot

        if idx < gradient.shape[0] and gradient.shape[1] >= 3:
            gradient[idx, 0] = grad_x
            gradient[idx, 1] = grad_y
            gradient[idx, 2] = grad_z

    @cuda.jit(**jit_kwargs)
    def batch_trajectory_kernel(
        thetastart_batch,  # (batch_size, num_joints)
        thetaend_batch,  # (batch_size, num_joints)
        traj_pos_batch,  # (batch_size, N, num_joints)
        traj_vel_batch,  # (batch_size, N, num_joints)
        traj_acc_batch,  # (batch_size, N, num_joints)
        Tf,
        N,
        method,
        batch_size,
    ) -> None:
        """Generate position, velocity, and acceleration for a batch of trajectories.

        Args:
            thetastart_batch: (batch_size, num_joints) device array of starting
                joint angles, radians.
            thetaend_batch: (batch_size, num_joints) device array of ending
                joint angles, radians.
            traj_pos_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint positions, radians.
            traj_vel_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint velocities, radians/s.
            traj_acc_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
            batch_size: Number of trajectories in the batch.
        """
        # Compute global indices
        batch_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        t_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        j_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        # Bounds check—use shape[1] for num_joints
        if batch_idx >= batch_size or t_idx >= N or j_idx >= thetastart_batch.shape[1]:
            return

        # Per-thread time-scaling computation. Previous version wrote scaling
        # for thread (0,0,0)'s t_idx into shared memory and let every other
        # thread read it — so threads at different t_idx got the wrong scaling.
        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)
        if method == 3:
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = 30.0 * tau2 * (1 - 2 * tau + tau2) / Tf
            s_ddot = 60.0 * tau * (1 - tau) * (1 - 2 * tau) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        # Compute delta for this trajectory
        dtheta = thetaend_batch[batch_idx, j_idx] - thetastart_batch[batch_idx, j_idx]

        # Write results
        traj_pos_batch[batch_idx, t_idx, j_idx] = (
            s * dtheta + thetastart_batch[batch_idx, j_idx]
        )
        traj_vel_batch[batch_idx, t_idx, j_idx] = s_dot * dtheta
        traj_acc_batch[batch_idx, t_idx, j_idx] = s_ddot * dtheta

    # MEMORY POOL MANAGEMENT
    class _GlobalCudaMemoryPool:
        """Enhanced memory pool with size tracking and performance optimization."""

        def __init__(self) -> None:
            """Initialize the memory pool with empty caches and statistics counters."""
            self.pool = {}
            self.max_pool_size = 200  # Increased for better caching
            self.total_allocated = 0
            self.cache_hits = 0
            self.cache_misses = 0

        def get_array(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
            """Return a pooled device array of the given shape/dtype, allocating on a cache miss.

            Args:
                shape: Shape of the requested device array.
                dtype: Element dtype of the requested array (default float32).

            Returns:
                A numba CUDA device array of the requested shape and dtype, reused
                from the pool on a cache hit or freshly allocated otherwise.
            """
            key = (shape, dtype)
            if key in self.pool and len(self.pool[key]) > 0:
                self.cache_hits += 1
                return self.pool[key].pop()
            else:
                self.cache_misses += 1
                self.total_allocated += np.prod(shape) * np.dtype(dtype).itemsize
                return cuda.device_array(shape, dtype=dtype)

        def return_array(self, array: Any) -> None:
            """Return a device array to the pool for reuse, up to the pool size limit.

            Args:
                array: Device array to return to the pool, keyed by its shape and
                    dtype. Dropped if the per-key pool is already at
                    ``max_pool_size``.
            """
            key = (array.shape, array.dtype)
            if key not in self.pool:
                self.pool[key] = []

            if len(self.pool[key]) < self.max_pool_size:
                self.pool[key].append(array)

        def get_stats(self) -> Dict[str, Any]:
            """Return cache hit rate, total allocated memory, and per-shape pool sizes."""
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                "cache_hit_rate": hit_rate,
                "total_allocated_mb": self.total_allocated / (1024 * 1024),
                "pool_sizes": {str(k): len(v) for k, v in self.pool.items()},
            }

        def clear(self) -> None:
            """Empty the pool and reset allocation and cache statistics."""
            self.pool.clear()
            self.total_allocated = 0
            self.cache_hits = 0
            self.cache_misses = 0

    _cuda_memory_pool = _GlobalCudaMemoryPool()

    def get_cuda_array(shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
        """Get optimized CUDA array from memory pool.

        Args:
            shape: Shape of the requested device array.
            dtype: Element dtype of the requested array (default float32).

        Returns:
            A pooled or newly allocated numba CUDA device array.
        """
        return _cuda_memory_pool.get_array(shape, dtype)

    def return_cuda_array(array: Any) -> None:
        """Return CUDA array to memory pool.

        Args:
            array: Device array to release back into the shared memory pool for
                later reuse.
        """
        _cuda_memory_pool.return_array(array)

    def get_memory_pool_stats() -> Dict[str, Any]:
        """Get memory pool performance statistics."""
        return _cuda_memory_pool.get_stats()

    # PERFORMANCE MONITORING
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
    def _best_2d_config(
        N: int, J: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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

    @lru_cache(maxsize=64)
    def _auto_tune_kernel_config(
        N: int, num_joints: int
    ) -> Optional[Dict[str, Any]]:
        """Auto-tune kernel configuration for specific problem size.

        Benchmarks each candidate kernel type on small test arrays and returns
        the fastest configuration. Results are memoized via ``lru_cache``.

        Args:
            N: Number of trajectory time steps for the target problem.
            num_joints: Number of joints for the target problem.

        Returns:
            Optional[Dict[str, Any]]: The best-performing kernel configuration
            dict (as returned by ``get_optimal_kernel_config``), or None when
            CUDA is unavailable.
        """
        if not CUDA_AVAILABLE:
            return None

        configs_to_test = [
            ("standard", {}),
            ("vectorized", {}),
            ("memory_optimized", {}),
            ("warp_optimized", {}),
            ("cache_friendly", {}),
        ]

        best_config = None
        best_time = float("inf")

        # Create small test arrays
        test_N = min(N, 1000)
        test_joints = min(num_joints, 8)

        try:
            d_start = cuda.device_array(test_joints, dtype=float32)
            d_end = cuda.device_array(test_joints, dtype=float32)
            d_pos = cuda.device_array((test_N, test_joints), dtype=float32)
            d_vel = cuda.device_array((test_N, test_joints), dtype=float32)
            d_acc = cuda.device_array((test_N, test_joints), dtype=float32)

            for kernel_type, params in configs_to_test:
                try:
                    config = get_optimal_kernel_config(test_N, test_joints, kernel_type)
                    if not config:
                        continue

                    kernel_func = config["kernel_func"]
                    grid = config["grid"]
                    block = config["block"]

                    # Warm-up
                    kernel_func[grid, block](
                        d_start, d_end, d_pos, d_vel, d_acc, 1.0, test_N, 3
                    )
                    cuda.synchronize()

                    # Timed run
                    start_time = perf_counter()
                    kernel_func[grid, block](
                        d_start, d_end, d_pos, d_vel, d_acc, 1.0, test_N, 3
                    )
                    cuda.synchronize()
                    elapsed = perf_counter() - start_time

                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = config

                except Exception:
                    continue

            return best_config or get_optimal_kernel_config(N, num_joints, "standard")

        except Exception:
            return get_optimal_kernel_config(N, num_joints, "standard")

    # HIGH-LEVEL OPTIMIZED FUNCTIONS
    def optimized_trajectory_generation_monitored(
        thetastart: Any,
        thetaend: Any,
        Tf: float,
        N: int,
        method: int,
        use_pinned: bool = True,
        kernel_type: str = "auto",
        enable_monitoring: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectory with comprehensive performance monitoring.

        This is the main function for achieving 40x+ speedups. Falls back to the
        optimized CPU implementation if CUDA is unavailable or GPU execution
        fails.

        Args:
            thetastart: (num_joints,) array-like of starting joint angles, radians.
            thetaend: (num_joints,) array-like of ending joint angles, radians.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
            use_pinned: If True, use pinned host memory for host/device transfers.
            kernel_type: Kernel selection strategy: "auto", "auto_tune", or an
                explicit kernel name ("standard", "vectorized",
                "memory_optimized", "warp_optimized", "cache_friendly").
            enable_monitoring: If True, log launch configuration and throughput
                and record per-kernel launch statistics.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
            traj_acc)``, each an ``(N, num_joints)`` float32 ndarray of joint
            positions (radians), velocities (radians/s), and accelerations
            (radians/s^2).
        """
        if not CUDA_AVAILABLE:
            return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

        num_joints = len(thetastart)
        total_work = N * num_joints

        # Performance recommendations
        if total_work < 50000:
            logger.warning(
                f"Problem size {total_work:,} may not achieve 40x speedup. "
                f"Recommend N ≥ {50000 // num_joints:,} for better GPU utilization."
            )

        try:
            # Convert to optimal data types
            thetastart_arr = np.ascontiguousarray(thetastart, dtype=np.float32)
            thetaend_arr = np.ascontiguousarray(thetaend, dtype=np.float32)

            # Get optimal configuration
            if kernel_type == "auto_tune":
                config = _auto_tune_kernel_config(N, num_joints)
            else:
                config = get_optimal_kernel_config(N, num_joints, kernel_type)

            if not config:
                raise RuntimeError("Failed to get kernel configuration")

            if enable_monitoring:
                logger.info(f"🚀 Using {config['kernel_type']} kernel:")
                logger.info(f"   Grid: {config['grid']}, Block: {config['block']}")
                logger.info(f"   Occupancy: {config['theoretical_occupancy']:.1f}%")
                logger.info(
                    f"   Expected speedup: {config['expected_speedup_range'][0]}-{config['expected_speedup_range'][1]}x"
                )
                if config.get("recommended_for_40x"):
                    logger.info("   ✅ Problem size optimal for 40x+ speedup!")
                else:
                    logger.info("   ⚠️  Consider larger N for maximum speedup")

            # Setup GPU memory with optimal transfers
            if use_pinned:
                d_thetastart = _h2d_pinned(thetastart_arr)
                d_thetaend = _h2d_pinned(thetaend_arr)
            else:
                d_thetastart = cuda.to_device(thetastart_arr)
                d_thetaend = cuda.to_device(thetaend_arr)

            # Allocate output arrays
            d_traj_pos = get_cuda_array((N, num_joints), dtype=np.float32)
            d_traj_vel = get_cuda_array((N, num_joints), dtype=np.float32)
            d_traj_acc = get_cuda_array((N, num_joints), dtype=np.float32)

            try:
                # Extract kernel configuration
                kernel_func = config["kernel_func"]
                grid = config["grid"]
                block = config["block"]

                # Record performance metrics
                if enable_monitoring:
                    _perf_monitor.record_kernel_launch(
                        config["kernel_type"], grid, block
                    )

                # Warm-up for large problems to eliminate JIT overhead
                if total_work > 100000:
                    warm_N = min(1000, N)
                    warm_grid = ((warm_N + block[0] - 1) // block[0], grid[1])
                    kernel_func[warm_grid, block](
                        d_thetastart,
                        d_thetaend,
                        d_traj_pos,
                        d_traj_vel,
                        d_traj_acc,
                        Tf,
                        warm_N,
                        method,
                    )
                    cuda.synchronize()

                # Main kernel launch - FIXED: Using 8 parameters instead of 9
                start_time = perf_counter()
                kernel_func[grid, block](
                    d_thetastart,
                    d_thetaend,
                    d_traj_pos,
                    d_traj_vel,
                    d_traj_acc,
                    Tf,
                    N,
                    method,
                )
                cuda.synchronize()
                gpu_time = perf_counter() - start_time

                # Copy results back with optimal memory transfer
                if use_pinned:
                    # Use pinned host arrays for faster transfer
                    traj_pos_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )
                    traj_vel_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )
                    traj_acc_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )

                    d_traj_pos.copy_to_host(traj_pos_pinned)
                    d_traj_vel.copy_to_host(traj_vel_pinned)
                    d_traj_acc.copy_to_host(traj_acc_pinned)

                    # Convert to regular numpy arrays
                    traj_pos = np.array(traj_pos_pinned)
                    traj_vel = np.array(traj_vel_pinned)
                    traj_acc = np.array(traj_acc_pinned)
                else:
                    traj_pos = d_traj_pos.copy_to_host()
                    traj_vel = d_traj_vel.copy_to_host()
                    traj_acc = d_traj_acc.copy_to_host()

                if enable_monitoring:
                    throughput = (
                        total_work / gpu_time / 1e6
                    )  # Million elements per second
                    logger.info(f"⚡ GPU execution: {gpu_time*1000:.2f}ms")
                    logger.info(f"📊 Throughput: {throughput:.1f} M elements/sec")

                return traj_pos, traj_vel, traj_acc

            finally:
                # Always return arrays to pool
                return_cuda_array(d_traj_pos)
                return_cuda_array(d_traj_vel)
                return_cuda_array(d_traj_acc)

        except Exception as e:
            logger.warning(f"GPU trajectory generation failed: {e}")
            logger.info("Falling back to optimized CPU implementation")
            return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

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
        if not CUDA_AVAILABLE:
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
    # CPU-only fallback implementations
    class _MockMemoryPool:
        def get_array(self, *args: Any, **kwargs: Any) -> Any:
            """Stub that raises because the CUDA memory pool is unavailable on CPU."""
            raise RuntimeError("CUDA memory pool not available")

        def return_array(self, *args: Any, **kwargs: Any) -> None:
            """Stub that raises because the CUDA memory pool is unavailable on CPU."""
            raise RuntimeError("CUDA memory pool not available")

        def clear(self) -> None:
            """No-op stub for the unavailable CUDA memory pool."""
            pass

        def get_stats(self) -> Dict[str, Any]:
            """Return empty statistics for the unavailable CUDA memory pool."""
            return {}

    _cuda_memory_pool = _MockMemoryPool()

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

    # Mock functions for API compatibility
    def trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA trajectory kernel is unavailable."""
        raise RuntimeError("CUDA trajectory kernel not available")

    def trajectory_kernel_vectorized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA vectorized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA vectorized trajectory kernel not available")

    def trajectory_kernel_memory_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory-optimized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA memory-optimized trajectory kernel not available")

    def trajectory_kernel_warp_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA warp-optimized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA warp-optimized trajectory kernel not available")

    def trajectory_kernel_cache_friendly(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA cache-friendly trajectory kernel is unavailable."""
        raise RuntimeError("CUDA cache-friendly trajectory kernel not available")

    def inverse_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA inverse dynamics kernel is unavailable."""
        raise RuntimeError("CUDA inverse dynamics kernel not available")

    def forward_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA forward dynamics kernel is unavailable."""
        raise RuntimeError("CUDA forward dynamics kernel not available")

    def cartesian_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA Cartesian trajectory kernel is unavailable."""
        raise RuntimeError("CUDA Cartesian trajectory kernel not available")

    def fused_potential_gradient_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA potential field kernel is unavailable."""
        raise RuntimeError("CUDA potential field kernel not available")

    def batch_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA batch trajectory kernel is unavailable."""
        raise RuntimeError("CUDA batch trajectory kernel not available")

    def get_cuda_array(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory pool is unavailable."""
        raise RuntimeError("CUDA memory pool not available")

    def return_cuda_array(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory pool is unavailable."""
        raise RuntimeError("CUDA memory pool not available")

    def get_memory_pool_stats() -> Dict[str, Any]:
        """Return empty memory-pool stats when CUDA is unavailable."""
        return {}

    def get_optimal_kernel_config(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Return no kernel configuration when CUDA is unavailable."""
        return None

    def auto_select_optimal_kernel(*args: Any, **kwargs: Any) -> str:
        """Report that no CUDA kernel can be selected."""
        return "none"

    def optimized_trajectory_generation_monitored(
        *args: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use the CPU trajectory fallback when CUDA is unavailable."""
        return trajectory_cpu_fallback(args[0], args[1], args[2], args[3], args[4])

    def _best_2d_config(*args: Any, **kwargs: Any) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return a minimal launch shape when CUDA is unavailable."""
        return ((1, 1), (1, 1))

    def profile_start() -> None:
        """No-op CUDA profiler start for CPU-only environments."""
        pass

    def profile_stop() -> Dict[str, Any]:
        """Return empty CUDA profiler stats in CPU-only environments."""
        return {}

    def benchmark_kernel_performance(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Report that CUDA benchmarking is unavailable."""
        print("CUDA benchmarking not available")
        return None


# HIGH-LEVEL WRAPPER FUNCTIONS
def optimized_trajectory_generation(
    thetastart: Any,
    thetaend: Any,
    Tf: float,
    N: int,
    method: int,
    use_pinned: bool = True,
    kernel_type: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main entry point for optimized trajectory generation.

    This function automatically selects the best kernel and configuration
    for maximum performance and 40x+ speedups.

    Args:
        thetastart, thetaend: Start and end joint angles
        Tf: Final time
        N: Number of trajectory points
        method: Time scaling method (3=cubic, 5=quintic)
        use_pinned: Use pinned memory for faster transfers
        kernel_type: Kernel selection ("auto", "standard", "vectorized", etc.)
    """
    return optimized_trajectory_generation_monitored(
        thetastart,
        thetaend,
        Tf,
        N,
        method,
        use_pinned,
        kernel_type,
        enable_monitoring=True,
    )


def optimized_potential_field(
    positions: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    influence_distance: float,
    use_pinned: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized potential field computation with CUDA acceleration.

    Args:
        positions: (N, 3) ndarray of query point positions.
        goal: (3,) ndarray, attractive goal position.
        obstacles: (num_obstacles, 3) ndarray of obstacle positions.
        influence_distance: Repulsive influence radius; obstacles farther than
            this contribute nothing.
        use_pinned: If True, use pinned host memory for host-to-device transfers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(potential, gradient)`` where
        ``potential`` is an ``(N,)`` float32 array of total potential values and
        ``gradient`` is an ``(N, 3)`` float32 array of potential gradients.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for potential field computation")

    N = positions.shape[0]

    # Use pinned memory for faster transfers
    if use_pinned:
        d_positions = _h2d_pinned(np.ascontiguousarray(positions, dtype=np.float32))
        d_goal = _h2d_pinned(np.ascontiguousarray(goal, dtype=np.float32))
        d_obstacles = _h2d_pinned(np.ascontiguousarray(obstacles, dtype=np.float32))
    else:
        d_positions = cuda.to_device(np.ascontiguousarray(positions, dtype=np.float32))
        d_goal = cuda.to_device(np.ascontiguousarray(goal, dtype=np.float32))
        d_obstacles = cuda.to_device(np.ascontiguousarray(obstacles, dtype=np.float32))

    # Allocate output arrays
    d_potential = get_cuda_array((N,), dtype=np.float32)
    d_gradient = get_cuda_array((N, 3), dtype=np.float32)

    try:
        # Launch fused kernel - FIXED: Using 6 parameters instead of 7
        grid, block = make_1d_grid(N)

        fused_potential_gradient_kernel[grid, block](
            d_positions,
            d_goal,
            d_obstacles,
            d_potential,
            d_gradient,
            influence_distance,
        )

        # Copy results back
        potential = d_potential.copy_to_host()
        gradient = d_gradient.copy_to_host()

        return potential, gradient

    finally:
        # Return arrays to pool
        return_cuda_array(d_potential)
        return_cuda_array(d_gradient)


def optimized_batch_trajectory_generation(
    thetastart_batch: np.ndarray,
    thetaend_batch: np.ndarray,
    Tf: float,
    N: int,
    method: int,
    use_pinned: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized batch trajectory generation for multiple trajectories.

    Args:
        thetastart_batch: (batch_size, num_joints) ndarray of starting joint
            angles, radians.
        thetaend_batch: (batch_size, num_joints) ndarray of ending joint angles,
            radians.
        Tf: Total trajectory duration, seconds.
        N: Number of trajectory time steps.
        method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        use_pinned: If True, use pinned host memory for host-to-device transfers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos_batch,
        traj_vel_batch, traj_acc_batch)``, each a ``(batch_size, N, num_joints)``
        float32 ndarray of joint positions (radians), velocities (radians/s), and
        accelerations (radians/s^2).

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for batch trajectory generation")

    batch_size, num_joints = thetastart_batch.shape

    # Use pinned memory for faster transfers
    if use_pinned:
        d_thetastart_batch = _h2d_pinned(
            np.ascontiguousarray(thetastart_batch, dtype=np.float32)
        )
        d_thetaend_batch = _h2d_pinned(
            np.ascontiguousarray(thetaend_batch, dtype=np.float32)
        )
    else:
        d_thetastart_batch = cuda.to_device(
            np.ascontiguousarray(thetastart_batch, dtype=np.float32)
        )
        d_thetaend_batch = cuda.to_device(
            np.ascontiguousarray(thetaend_batch, dtype=np.float32)
        )

    # Allocate output arrays
    d_traj_pos_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_vel_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_acc_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)

    try:
        # 3D grid for batch processing
        grid = ((batch_size + 7) // 8, (N + 15) // 16, (num_joints + 7) // 8)
        block = (8, 16, 8)

        # Launch batch kernel - FIXED: Using 9 parameters instead of 10
        batch_trajectory_kernel[grid, block](
            d_thetastart_batch,
            d_thetaend_batch,
            d_traj_pos_batch,
            d_traj_vel_batch,
            d_traj_acc_batch,
            Tf,
            N,
            method,
            batch_size,
        )

        # Copy results back
        traj_pos_batch = d_traj_pos_batch.copy_to_host()
        traj_vel_batch = d_traj_vel_batch.copy_to_host()
        traj_acc_batch = d_traj_acc_batch.copy_to_host()

        return traj_pos_batch, traj_vel_batch, traj_acc_batch

    finally:
        # Return arrays to pool
        return_cuda_array(d_traj_pos_batch)
        return_cuda_array(d_traj_vel_batch)
        return_cuda_array(d_traj_acc_batch)


# LEGACY COMPATIBILITY FUNCTIONS
def attractive_potential_kernel(*args: Any, **kwargs: Any) -> NoReturn:
    """Legacy function - use fused_potential_gradient_kernel instead."""
    raise RuntimeError(
        "Legacy attractive_potential_kernel is deprecated.\n"
        "Use fused_potential_gradient_kernel for better performance."
    )


def repulsive_potential_kernel(*args: Any, **kwargs: Any) -> NoReturn:
    """Legacy function - use fused_potential_gradient_kernel instead."""
    raise RuntimeError(
        "Legacy repulsive_potential_kernel is deprecated.\n"
        "Use fused_potential_gradient_kernel for better performance."
    )


def gradient_kernel(*args: Any, **kwargs: Any) -> NoReturn:
    """Legacy function - use fused_potential_gradient_kernel instead."""
    raise RuntimeError(
        "Legacy gradient_kernel is deprecated.\n"
        "Use fused_potential_gradient_kernel for better performance."
    )


# PERFORMANCE UTILITIES
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


# COMPREHENSIVE EXPORT LIST
__all__ = [
    # Core availability checks
    "CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "check_cuda_availability",
    "check_cupy_availability",
    # Standard kernels
    "trajectory_kernel",
    "inverse_dynamics_kernel",
    "forward_dynamics_kernel",
    "cartesian_trajectory_kernel",
    "fused_potential_gradient_kernel",
    "batch_trajectory_kernel",
    # Advanced kernels for 40x+ speedup
    "trajectory_kernel_vectorized",
    "trajectory_kernel_memory_optimized",
    "trajectory_kernel_warp_optimized",
    "trajectory_kernel_cache_friendly",
    # High-level optimized functions
    "optimized_trajectory_generation",
    "optimized_trajectory_generation_monitored",
    "optimized_potential_field",
    "optimized_batch_trajectory_generation",
    # Kernel configuration and optimization
    "get_optimal_kernel_config",
    "auto_select_optimal_kernel",
    "_best_2d_config",
    # Memory management
    "get_cuda_array",
    "return_cuda_array",
    "get_memory_pool_stats",
    # Performance monitoring
    "CUDAPerformanceMonitor",
    "profile_start",
    "profile_stop",
    "benchmark_kernel_performance",
    # Grid configuration utilities (backward compatibility)
    "make_1d_grid",
    "make_2d_grid",
    "make_2d_grid_optimized",
    "get_gpu_properties",
    "trajectory_cpu_fallback",
    # Performance optimization helpers
    "print_performance_recommendations",
    "setup_cuda_environment_for_40x_speedup",
    # Legacy compatibility
    "attractive_potential_kernel",
    "repulsive_potential_kernel",
    "gradient_kernel",
]
