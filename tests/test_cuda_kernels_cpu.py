#!/usr/bin/env python3
"""
CPU-side coverage for cuda_kernels.py.

These tests exercise the NumPy fallback paths and ensure GPU-only
entry points raise clearly when CUDA is unavailable.
"""

import numpy as np
import pytest

from ManipulaPy import cuda_kernels
from ManipulaPy.cuda_kernels import (
    trajectory_cpu_fallback,
    optimized_trajectory_generation,
    optimized_potential_field,
    optimized_batch_trajectory_generation,
    auto_select_optimal_kernel,
    get_optimal_kernel_config,
    check_cuda_availability,
)


def test_trajectory_cpu_fallback_linear_matches_expected():
    thetastart = np.array([0.0, 0.0], dtype=np.float32)
    thetaend = np.array([1.0, -1.0], dtype=np.float32)
    Tf, N, method = 1.0, 5, 1  # linear

    pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Positions should interpolate linearly, vel constant, acc zero
    t = np.linspace(0, 1.0, N, dtype=np.float32)[:, None]
    expected_pos = thetastart + t * (thetaend - thetastart)
    expected_vel = np.full_like(expected_pos, (thetaend - thetastart) / Tf)
    expected_acc = np.zeros_like(expected_pos)

    assert pos.shape == (N, thetastart.size)
    assert np.allclose(pos, expected_pos)
    assert np.allclose(vel, expected_vel)
    assert np.allclose(acc, expected_acc)


def test_trajectory_cpu_fallback_quintic_endpoints_exact():
    thetastart = np.array([0.5], dtype=np.float32)
    thetaend = np.array([1.5], dtype=np.float32)
    Tf, N, method = 2.0, 11, 5  # quintic

    pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    assert np.isclose(pos[0, 0], thetastart[0])
    assert np.isclose(pos[-1, 0], thetaend[0])
    # Quintic should start/end with zero velocity/acceleration
    assert np.isclose(vel[0, 0], 0.0, atol=1e-6)
    assert np.isclose(vel[-1, 0], 0.0, atol=1e-6)
    assert np.isclose(acc[0, 0], 0.0, atol=1e-6)
    assert np.isclose(acc[-1, 0], 0.0, atol=1e-6)


def test_optimized_trajectory_generation_uses_cpu_when_no_cuda(monkeypatch):
    # Force CUDA unavailable
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", False)
    result_pos, result_vel, result_acc = optimized_trajectory_generation(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        1.0,
        4,
        3,
        use_pinned=False,
    )
    cpu_pos, cpu_vel, cpu_acc = trajectory_cpu_fallback(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        1.0,
        4,
        3,
    )
    assert np.allclose(result_pos, cpu_pos)
    assert np.allclose(result_vel, cpu_vel)
    assert np.allclose(result_acc, cpu_acc)


def test_gpu_only_entrypoints_raise_when_no_cuda():
    assert check_cuda_availability() is False

    positions = np.zeros((2, 3), dtype=np.float32)
    goal = np.zeros(3, dtype=np.float32)
    obstacles = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(RuntimeError):
        optimized_potential_field(positions, goal, obstacles, influence_distance=1.0, use_pinned=False)

    with pytest.raises(RuntimeError):
        optimized_batch_trajectory_generation(
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            Tf=1.0,
            N=8,
            method=3,
            use_pinned=False,
        )


def test_kernel_selection_fallbacks_when_no_cuda():
    assert auto_select_optimal_kernel(100, 6) == "none"
    assert get_optimal_kernel_config(100, 6) is None
