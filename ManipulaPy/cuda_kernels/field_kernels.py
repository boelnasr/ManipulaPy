#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernel implementation split by runtime concern."""

import math
from typing import Any, NoReturn, Tuple

import numpy as np

from . import _runtime
from ._runtime import CUDA_AVAILABLE, cuda
from .memory import _h2d_pinned, get_cuda_array, return_cuda_array

_cuda_routing_enabled: Any = None
make_1d_grid: Any = None

if CUDA_AVAILABLE:
    jit_kwargs = _runtime.jit_kwargs

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

else:

    def fused_potential_gradient_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA potential field kernel is unavailable."""
        raise RuntimeError("CUDA potential field kernel not available")


def potential_field_cpu_fallback(
    positions: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    influence_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the fused attractive/repulsive field with NumPy.

    This is the CPU reference for ``fused_potential_gradient_kernel``. Its
    equations and zero-distance handling intentionally mirror the raw Numba
    kernel so registry dispatch can compare the two paths directly.
    """
    positions_arr = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    goal_arr = np.ascontiguousarray(goal, dtype=np.float32).reshape(3)
    obstacles_arr = np.ascontiguousarray(obstacles, dtype=np.float32).reshape(-1, 3)

    differences = positions_arr - goal_arr
    potential = 0.5 * np.sum(differences * differences, axis=1)
    gradient = differences.copy()

    influence_inverse = (
        np.float32(1.0 / influence_distance)
        if influence_distance > 0.0
        else np.float32(0.0)
    )
    influence_squared = np.float32(influence_distance * influence_distance)

    for obstacle in obstacles_arr:
        obstacle_difference = positions_arr - obstacle
        distance_squared = np.sum(
            obstacle_difference * obstacle_difference, axis=1
        )
        influenced = (distance_squared > 0.0) & (
            distance_squared < influence_squared
        )
        if not np.any(influenced):
            continue

        distance_inverse = np.float32(1.0) / np.sqrt(distance_squared[influenced])
        influence_term = distance_inverse - influence_inverse
        potential[influenced] += np.float32(0.5) * influence_term * influence_term
        gradient_factor = (
            -influence_term * distance_inverse * distance_inverse * distance_inverse
        )
        gradient[influenced] += (
            gradient_factor[:, np.newaxis] * obstacle_difference[influenced]
        )

    return potential.astype(np.float32), gradient.astype(np.float32)


def _optimized_potential_field_cuda(
    positions: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    influence_distance: float,
    use_pinned: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Launch the fused potential-field CUDA implementation."""

    N = positions.shape[0]

    # The fused kernel indexes ``obstacles[obs, 0..2]``, so obstacles must be a
    # 2-D ``(M, 3)`` array. Callers (e.g. the planner's no-obstacle path) may
    # pass an empty list -> ``np.array([])`` which is 1-D ``(0,)``; a 1-D type
    # breaks Numba's nopython type inference and aborts the kernel launch.
    # Normalise to ``(M, 3)`` (empty -> ``(0, 3)``).
    obstacles = np.ascontiguousarray(obstacles, dtype=np.float32).reshape(-1, 3)

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


def optimized_potential_field(
    positions: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    influence_distance: float,
    use_pinned: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a fused potential field through registered backend dispatch."""
    from .registry import execute_registered_kernel

    return execute_registered_kernel(
        "potential_field.fused",
        positions,
        goal,
        obstacles,
        influence_distance,
        use_pinned=use_pinned,
    )


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
