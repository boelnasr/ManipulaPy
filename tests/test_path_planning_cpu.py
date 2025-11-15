#!/usr/bin/env python3
"""
Lightweight CPU-only tests for path_planning._trajectory_cpu_fallback.
These exercise the numba-jitted CPU trajectory generator without requiring CUDA.
"""

import numpy as np
import pytest

from ManipulaPy.path_planning import _trajectory_cpu_fallback, _traj_cpu_njit


def test_trajectory_cpu_fallback_cubic_endpoints():
    thetastart = np.array([0.0, 1.0], dtype=np.float32)
    thetaend = np.array([1.0, 3.0], dtype=np.float32)
    Tf, N, method = 1.0, 3, 3  # cubic: endpoints should match, vel/acc zero at ends

    pos, vel, acc = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Endpoints hit start/end exactly
    assert np.allclose(pos[0], thetastart)
    assert np.allclose(pos[-1], thetaend)
    # Cubic scaling yields zero velocity at boundaries
    assert np.allclose(vel[0], 0.0)
    assert np.allclose(vel[-1], 0.0)


def test_trajectory_cpu_fallback_quintic_midpoint_values():
    thetastart = np.array([0.0], dtype=np.float32)
    thetaend = np.array([1.0], dtype=np.float32)
    Tf, N, method = 2.0, 5, 5  # quintic

    pos, vel, acc = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Midpoint tau = 0.5 uses the implemented polynomial: 10*t^3 - 9*t^4
    assert np.isclose(pos[N // 2, 0], 0.6875, atol=1e-4)
    # Velocities should stay finite
    assert np.all(np.isfinite(vel))


def test_trajectory_cpu_fallback_unsupported_method_returns_constant():
    thetastart = np.array([1.0, 2.0], dtype=np.float32)
    thetaend = np.array([3.0, 4.0], dtype=np.float32)
    Tf, N, method = 1.0, 4, 7  # unsupported → s=0

    pos, vel, acc = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Positions remain at start, velocities/accels zero
    assert np.allclose(pos, thetastart)
    assert np.allclose(vel, 0.0)
    assert np.allclose(acc, 0.0)


def test_traj_cpu_njit_matches_fallback():
    thetastart = np.array([0.5], dtype=np.float32)
    thetaend = np.array([1.5], dtype=np.float32)
    Tf, N, method = 1.5, 6, 3

    pos1, vel1, acc1 = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
    pos2, vel2, acc2 = _traj_cpu_njit(thetastart, thetaend, Tf, N, method)

    assert np.allclose(pos1, pos2)
    assert np.allclose(vel1, vel2)
    assert np.allclose(acc1, acc2)
