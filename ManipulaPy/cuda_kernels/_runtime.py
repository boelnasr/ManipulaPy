#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernel implementation split by runtime concern."""

import logging
import math
import os
import sys
from types import ModuleType
import warnings
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, NoReturn, Optional, Tuple

import numpy as np
from numba import config as _nb_cfg

from ..backend import get_backend as _get_backend

# Configure numba for optimal performance
_nb_cfg.CUDA_CACHE_SIZE = "2048"  # Increased cache size for better compilation reuse
_nb_cfg.CUDA_LOW_OCCUPANCY_WARNINGS = False  # Disable warnings for specialized kernels

# Environment toggle for fast math operations
FAST_MATH = bool(int(os.getenv("MANIPULAPY_FASTMATH", "1")))

# Setup logging
logger = logging.getLogger("ManipulaPy.cuda_kernels")


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


_OWNERS = {}
_SUBSCRIBERS = ()


class _CompatibilityModule(ModuleType):
    """Module type that keeps historical package assignments synchronized."""

    def __setattr__(self, name, value):
        owner = _OWNERS.get(name)
        if owner is None:
            return super().__setattr__(name, value)
        owner.__dict__[name] = value
        for module in _SUBSCRIBERS:
            if name in module.__dict__:
                module.__dict__[name] = value


def register_module(module):
    """Register a compatibility subscriber."""
    global _SUBSCRIBERS
    if module not in _SUBSCRIBERS:
        _SUBSCRIBERS += (module,)


def register_owner(name, module):
    """Register the canonical module for a historical binding."""
    _OWNERS[name] = module


def set_compat_value(module, name, value):
    """Install an initial compatibility value without invoking proxies."""
    module.__dict__[name] = value


def activate_proxies(*modules):
    """Activate assignment synchronization after package initialization."""
    for module in modules:
        register_module(module)
    for module in modules:
        module.__class__ = _CompatibilityModule


def initialize_compat(package, memory, trajectory, field, registry):
    """Wire relocated globals and install historical assignment forwarding."""
    bindings = {
        memory: {
            "CUDA_AVAILABLE": CUDA_AVAILABLE,
            "cuda": cuda,
            "_cuda_error": _cuda_error,
            "_PINNED_MEMORY_OPT_IN": _PINNED_MEMORY_OPT_IN,
        },
        field: {
            "CUDA_AVAILABLE": CUDA_AVAILABLE,
            "cuda": cuda,
            "_h2d_pinned": memory._h2d_pinned,
            "get_cuda_array": memory.get_cuda_array,
            "return_cuda_array": memory.return_cuda_array,
            "_cuda_routing_enabled": registry._cuda_routing_enabled,
            "make_1d_grid": registry.make_1d_grid,
        },
        trajectory: {
            "CUDA_AVAILABLE": CUDA_AVAILABLE,
            "cuda": cuda,
            "float32": float32,
            "FAST_MATH": FAST_MATH,
            "logger": logger,
            "_h2d_pinned": memory._h2d_pinned,
            "get_cuda_array": memory.get_cuda_array,
            "return_cuda_array": memory.return_cuda_array,
            "get_optimal_kernel_config": registry.get_optimal_kernel_config,
            "_perf_monitor": registry._perf_monitor,
            "_cuda_routing_enabled": registry._cuda_routing_enabled,
            "get_gpu_properties": registry.get_gpu_properties,
        },
        registry: {
            "CUDA_AVAILABLE": CUDA_AVAILABLE,
            "CUPY_AVAILABLE": CUPY_AVAILABLE,
            "_cuda_error": _cuda_error,
            "cuda": cuda,
            "cp": cp,
            "float32": float32,
            "logger": logger,
            "_get_backend": _get_backend,
            "get_memory_pool_stats": memory.get_memory_pool_stats,
            "trajectory_kernel": trajectory.trajectory_kernel,
            "trajectory_kernel_vectorized": trajectory.trajectory_kernel_vectorized,
            "trajectory_kernel_memory_optimized": trajectory.trajectory_kernel_memory_optimized,
            "trajectory_kernel_warp_optimized": trajectory.trajectory_kernel_warp_optimized,
            "trajectory_kernel_cache_friendly": trajectory.trajectory_kernel_cache_friendly,
            "optimized_trajectory_generation_monitored": trajectory.optimized_trajectory_generation_monitored,
            "optimized_potential_field": field.optimized_potential_field,
            "optimized_batch_trajectory_generation": trajectory.optimized_batch_trajectory_generation,
        },
    }
    for module, values in bindings.items():
        module.__dict__.update(values)

    owner_names = {
        memory: (
            "_h2d_pinned",
            "_GlobalCudaMemoryPool",
            "_MockMemoryPool",
            "_cuda_memory_pool",
            "get_cuda_array",
            "return_cuda_array",
            "get_memory_pool_stats",
        ),
        trajectory: (
            "trajectory_cpu_fallback",
            "jit_kwargs",
            "matrix_vector_multiply_6x6",
            "trajectory_kernel",
            "trajectory_kernel_vectorized",
            "trajectory_kernel_memory_optimized",
            "trajectory_kernel_warp_optimized",
            "trajectory_kernel_cache_friendly",
            "inverse_dynamics_kernel",
            "forward_dynamics_kernel",
            "cartesian_trajectory_kernel",
            "batch_trajectory_kernel",
            "_auto_tune_kernel_config",
            "_optimized_trajectory_generation_monitored_cuda",
            "optimized_trajectory_generation_monitored",
            "optimized_trajectory_generation",
            "optimized_batch_trajectory_generation",
        ),
        field: (
            "fused_potential_gradient_kernel",
            "optimized_potential_field",
            "attractive_potential_kernel",
            "repulsive_potential_kernel",
            "gradient_kernel",
        ),
        registry: (
            "check_cuda_availability",
            "check_cupy_availability",
            "make_1d_grid",
            "make_2d_grid",
            "make_2d_grid_optimized",
            "get_gpu_properties",
            "CUDAPerformanceMonitor",
            "_perf_monitor",
            "get_optimal_kernel_config",
            "_best_2d_config",
            "auto_select_optimal_kernel",
            "profile_start",
            "profile_stop",
            "benchmark_kernel_performance",
            "_cuda_routing_enabled",
            "print_performance_recommendations",
            "setup_cuda_environment_for_40x_speedup",
        ),
    }
    for name in (
        "CUDA_AVAILABLE",
        "CUPY_AVAILABLE",
        "cuda",
        "cp",
        "float32",
        "int32",
        "_cuda_error",
        "FAST_MATH",
        "logger",
        "float_t",
        "_PINNED_MEMORY_OPT_IN",
        "_get_backend",
        "_cuda_safe_to_probe",
        "_detect_cuda_capability",
        "MockCuda",
    ):
        if name in globals():
            register_owner(name, sys.modules[__name__])
    for module, names in owner_names.items():
        for name in names:
            if name in module.__dict__:
                register_owner(name, module)
    activate_proxies(
        package, sys.modules[__name__], memory, trajectory, field, registry
    )
