#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernel implementation split by runtime concern."""

from typing import Any, Dict, NoReturn, Tuple

import numpy as np

from ._runtime import CUDA_AVAILABLE, _PINNED_MEMORY_OPT_IN, _cuda_error, cuda


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


if CUDA_AVAILABLE:

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

else:

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

    def get_cuda_array(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory pool is unavailable."""
        raise RuntimeError("CUDA memory pool not available")

    def return_cuda_array(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory pool is unavailable."""
        raise RuntimeError("CUDA memory pool not available")

    def get_memory_pool_stats() -> Dict[str, Any]:
        """Return empty memory-pool stats when CUDA is unavailable."""
        return {}
