#!/usr/bin/env python3
"""
Lightweight CPU-only tests for path_planning._trajectory_cpu_fallback.
These exercise the numba-jitted CPU trajectory generator without requiring CUDA.
"""

import numpy as np
import pytest

from ManipulaPy.path_planning import (
    OptimizedTrajectoryPlanning,
    _traj_cpu_njit,
    _trajectory_cpu_fallback,
)
from ManipulaPy.potential_field import PotentialField


def test_trajectory_cpu_fallback_cubic_endpoints() -> None:
    """Cubic fallback hits start/end exactly with zero boundary velocity."""
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


def test_trajectory_cpu_fallback_quintic_midpoint_values() -> None:
    """Quintic fallback reaches the expected midpoint value with finite velocities."""
    thetastart = np.array([0.0], dtype=np.float32)
    thetaend = np.array([1.0], dtype=np.float32)
    Tf, N, method = 2.0, 5, 5  # quintic

    pos, vel, acc = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Quintic time-scaling: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    # At tau = 0.5: s = 1.25 - 0.9375 + 0.1875 = 0.5
    # (Previously asserted 0.6875, which was anchored to the buggy
    # 10*t^3 - 9*t^4 reduction; fixed in v1.3.2 — see 842a004.)
    assert np.isclose(pos[N // 2, 0], 0.5, atol=1e-4)
    # Velocities should stay finite
    assert np.all(np.isfinite(vel))


def test_trajectory_cpu_fallback_unsupported_method_returns_constant() -> None:
    """An unsupported method holds position constant with zero velocity and acceleration."""
    thetastart = np.array([1.0, 2.0], dtype=np.float32)
    thetaend = np.array([3.0, 4.0], dtype=np.float32)
    Tf, N, method = 1.0, 4, 7  # unsupported → s=0

    pos, vel, acc = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Positions remain at start, velocities/accels zero
    assert np.allclose(pos, thetastart)
    assert np.allclose(vel, 0.0)
    assert np.allclose(acc, 0.0)


def test_traj_cpu_njit_matches_fallback() -> None:
    """The njit-compiled trajectory matches the pure-Python fallback output."""
    thetastart = np.array([0.5], dtype=np.float32)
    thetaend = np.array([1.5], dtype=np.float32)
    Tf, N, method = 1.5, 6, 3

    pos1, vel1, acc1 = _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
    pos2, vel2, acc2 = _traj_cpu_njit(thetastart, thetaend, Tf, N, method)

    assert np.allclose(pos1, pos2)
    assert np.allclose(vel1, vel2)
    assert np.allclose(acc1, acc2)


class _CollideIfFirstJointLarge:
    """Fake collision checker: a config collides while joint 0 exceeds 0.5 rad."""

    def check_collision(self, q: np.ndarray) -> bool:
        """Report a collision when the first joint angle is above the band."""
        return bool(abs(q[0]) > 0.5)


def _make_planner(cuda_available: bool) -> OptimizedTrajectoryPlanning:
    """Build a planner stub for collision-avoidance tests without a URDF.

    Bypasses __init__ (which needs a URDF / CUDA) and wires only the attributes
    the collision-avoidance routines touch.
    """
    planner = OptimizedTrajectoryPlanning.__new__(OptimizedTrajectoryPlanning)
    planner.cuda_available = cuda_available
    planner.collision_checker = _CollideIfFirstJointLarge()
    planner.potential_field = PotentialField()
    return planner


def test_gpu_collision_avoidance_handles_joint_space_and_matches_cpu() -> None:
    """The CUDA-dispatched collision avoidance works on N-DOF joint data.

    Regression: the GPU path previously fed joint-space configurations to the
    3-D Cartesian ``optimized_potential_field`` kernel, producing a wrong
    gradient and a 6-vs-3 broadcast error on the ``step -= gradient`` update.
    It must now handle a 6-DOF trajectory without error and agree with the
    joint-space CPU routine.
    """
    thetaend = np.zeros(6, dtype=float)
    # Joint 0 starts at 1.0 (> 0.5 -> colliding) and is pulled toward the goal.
    traj = np.tile(np.linspace(1.0, 0.0, 5)[:, None], (1, 6)).astype(float)

    gpu_result = _make_planner(True)._apply_collision_avoidance_gpu(
        traj.copy(), thetaend
    )
    cpu_result = _make_planner(False)._apply_collision_avoidance_cpu(
        traj.copy(), thetaend
    )

    assert gpu_result.shape == traj.shape
    # GPU dispatch now uses the joint-space routine, so results are identical.
    assert np.allclose(gpu_result, cpu_result)
    # Every adjusted configuration is nudged out of collision.
    assert not any(_CollideIfFirstJointLarge().check_collision(q) for q in gpu_result)
