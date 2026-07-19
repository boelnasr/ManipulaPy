#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernels compatibility facade."""

from . import _runtime
from ._runtime import (
    Any,
    Dict,
    NoReturn,
    Optional,
    Tuple,
    FAST_MATH,
    CUDA_AVAILABLE,
    CUPY_AVAILABLE,
    _PINNED_MEMORY_OPT_IN,
    _cuda_error,
    _cuda_safe_to_probe,
    _detect_cuda_capability,
    _get_backend,
    _nb_cfg,
    cp,
    cuda,
    float32,
    float_t,
    int32,
    logger,
    logging,
    lru_cache,
    math,
    np,
    os,
    perf_counter,
    warnings,
)
from . import memory
from .memory import (
    _h2d_pinned,
    _cuda_memory_pool,
    get_cuda_array,
    get_memory_pool_stats,
    return_cuda_array,
)
from . import trajectory_kernels
from . import field_kernels
from .trajectory_kernels import (
    _optimized_trajectory_generation_monitored_cuda,
    batch_trajectory_kernel,
    cartesian_trajectory_kernel,
    forward_dynamics_kernel,
    inverse_dynamics_kernel,
    optimized_batch_trajectory_generation,
    optimized_trajectory_generation,
    optimized_trajectory_generation_monitored,
    trajectory_cpu_fallback,
    trajectory_kernel,
    trajectory_kernel_cache_friendly,
    trajectory_kernel_memory_optimized,
    trajectory_kernel_vectorized,
    trajectory_kernel_warp_optimized,
)
from .field_kernels import (
    attractive_potential_kernel,
    fused_potential_gradient_kernel,
    gradient_kernel,
    optimized_potential_field,
    repulsive_potential_kernel,
)
from . import registry
from .registry import (
    CUDAPerformanceMonitor,
    _best_2d_config,
    _cuda_routing_enabled,
    _perf_monitor,
    auto_select_optimal_kernel,
    benchmark_kernel_performance,
    check_cuda_availability,
    check_cupy_availability,
    get_gpu_properties,
    get_optimal_kernel_config,
    make_1d_grid,
    make_2d_grid,
    make_2d_grid_optimized,
    print_performance_recommendations,
    profile_start,
    profile_stop,
    setup_cuda_environment_for_40x_speedup,
)

if CUDA_AVAILABLE:
    from .memory import _GlobalCudaMemoryPool
    from .trajectory_kernels import (
        _auto_tune_kernel_config,
        jit_kwargs,
        matrix_vector_multiply_6x6,
    )
else:
    from ._runtime import MockCuda
    from .memory import _MockMemoryPool

__all__ = [
    "CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "check_cuda_availability",
    "check_cupy_availability",
    "trajectory_kernel",
    "inverse_dynamics_kernel",
    "forward_dynamics_kernel",
    "cartesian_trajectory_kernel",
    "fused_potential_gradient_kernel",
    "batch_trajectory_kernel",
    "trajectory_kernel_vectorized",
    "trajectory_kernel_memory_optimized",
    "trajectory_kernel_warp_optimized",
    "trajectory_kernel_cache_friendly",
    "optimized_trajectory_generation",
    "optimized_trajectory_generation_monitored",
    "optimized_potential_field",
    "optimized_batch_trajectory_generation",
    "get_optimal_kernel_config",
    "auto_select_optimal_kernel",
    "_best_2d_config",
    "get_cuda_array",
    "return_cuda_array",
    "get_memory_pool_stats",
    "CUDAPerformanceMonitor",
    "profile_start",
    "profile_stop",
    "benchmark_kernel_performance",
    "make_1d_grid",
    "make_2d_grid",
    "make_2d_grid_optimized",
    "get_gpu_properties",
    "trajectory_cpu_fallback",
    "print_performance_recommendations",
    "setup_cuda_environment_for_40x_speedup",
    "attractive_potential_kernel",
    "repulsive_potential_kernel",
    "gradient_kernel",
]

_runtime.initialize_compat(
    __import__(__name__, fromlist=["*"]),
    memory,
    trajectory_kernels,
    field_kernels,
    registry,
)
