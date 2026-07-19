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

import time
import warnings
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from ..cuda_kernels import (
    CUDA_AVAILABLE,
    _best_2d_config,
    _h2d_pinned,
    auto_select_optimal_kernel,
    benchmark_kernel_performance,
    check_cuda_availability,
    get_cuda_array,
    get_gpu_properties,
    get_memory_pool_stats,
    get_optimal_kernel_config,
    make_1d_grid,
    make_2d_grid,
    make_2d_grid_optimized,
    optimized_batch_trajectory_generation,
    optimized_trajectory_generation,
    optimized_trajectory_generation_monitored,
    print_performance_recommendations,
    profile_start,
    profile_stop,
    return_cuda_array,
    setup_cuda_environment_for_40x_speedup,
)
from ..backend import get_backend
from ..utils import (
    CubicTimeScaling,
    MatrixExp3,
    MatrixLog3,
    QuinticTimeScaling,
    TransToRp,
)

# Import CUDA functions only if available
if CUDA_AVAILABLE:
    from numba import cuda

    from ..cuda_kernels import (
        batch_trajectory_kernel,
        cartesian_trajectory_kernel,
        forward_dynamics_kernel,
        fused_potential_gradient_kernel,
        inverse_dynamics_kernel,
        trajectory_kernel,
        trajectory_kernel_cache_friendly,
        trajectory_kernel_memory_optimized,
        trajectory_kernel_vectorized,
        trajectory_kernel_warp_optimized,
    )
else:
    # Create dummy functions for when CUDA is not available
    def trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for the standard trajectory kernel."""
        raise RuntimeError("CUDA not available")

    def trajectory_kernel_vectorized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for the vectorized trajectory kernel."""
        raise RuntimeError("CUDA not available")

    def trajectory_kernel_memory_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for the memory-optimized kernel."""
        raise RuntimeError("CUDA not available")

    def trajectory_kernel_warp_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for the warp-optimized kernel."""
        raise RuntimeError("CUDA not available")

    def trajectory_kernel_cache_friendly(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for the cache-friendly kernel."""
        raise RuntimeError("CUDA not available")

    def inverse_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for inverse dynamics kernels."""
        raise RuntimeError("CUDA not available")

    def forward_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for forward dynamics kernels."""
        raise RuntimeError("CUDA not available")

    def cartesian_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for Cartesian trajectory kernels."""
        raise RuntimeError("CUDA not available")

    def fused_potential_gradient_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for potential field kernels."""
        raise RuntimeError("CUDA not available")

    def batch_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise a CUDA availability error for batch trajectory kernels."""
        raise RuntimeError("CUDA not available")

    class MockCuda:
        @staticmethod
        def to_device(*args: Any, **kwargs: Any) -> NoReturn:
            """Raise a CUDA availability error for device transfers."""
            raise RuntimeError("CUDA not available")

        @staticmethod
        def device_array(*args: Any, **kwargs: Any) -> NoReturn:
            """Raise a CUDA availability error for device allocations."""
            raise RuntimeError("CUDA not available")

        @staticmethod
        def synchronize() -> None:
            """No-op synchronization placeholder when CUDA is unavailable."""
            pass

    cuda = MockCuda()

import logging

from ..potential_field import CollisionChecker, PotentialField

# Module-level logger; leave handler configuration to the host application
logger = logging.getLogger("ManipulaPy.planning.trajectory_planning")
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)


__all__ = [
    "Any",
    "Dict",
    "List",
    "NoReturn",
    "Optional",
    "Tuple",
    "np",
    "plt",
    "time",
    "warnings",
    "njit",
    "prange",
    "logger",
    "logging",
    "get_backend",
    "CollisionChecker",
    "PotentialField",
    "CubicTimeScaling",
    "MatrixExp3",
    "MatrixLog3",
    "QuinticTimeScaling",
    "TransToRp",
    "cuda",
    "CUDA_AVAILABLE",
    "_best_2d_config",
    "_h2d_pinned",
    "auto_select_optimal_kernel",
    "benchmark_kernel_performance",
    "check_cuda_availability",
    "get_cuda_array",
    "get_gpu_properties",
    "get_memory_pool_stats",
    "get_optimal_kernel_config",
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
    "trajectory_kernel_vectorized",
    "trajectory_kernel_memory_optimized",
    "trajectory_kernel_warp_optimized",
    "trajectory_kernel_cache_friendly",
    "inverse_dynamics_kernel",
    "forward_dynamics_kernel",
    "cartesian_trajectory_kernel",
    "fused_potential_gradient_kernel",
    "batch_trajectory_kernel",
]

# MockCuda exists only when CUDA is unavailable; export it so the
# historical module namespace is preserved on non-CUDA hosts.
if not CUDA_AVAILABLE:
    __all__.append("MockCuda")
