#!/usr/bin/env python3
"""
CPU-only unit tests for key methods in path_planning. No CUDA or URDF required.
"""

import numpy as np
import pytest

from ManipulaPy.path_planning import OptimizedTrajectoryPlanning, _traj_cpu_njit


class StubManipulator:
    """Placeholder serial manipulator (not used in tested code paths)."""
    pass


class MockDynamics:
    def __init__(self, n=2):
        self.n = n
        self.Glist = np.ones((n, n, n), dtype=np.float32)
        self.S_list = np.eye(6, n, dtype=np.float32)
        self.M_list = np.eye(4, dtype=np.float32)

    def inverse_dynamics(self, q, dq, ddq, g, Ftip):
        # Simple affine model: ddq + gravity bias
        return np.asarray(ddq, dtype=np.float32) + 0.1

    def forward_dynamics(self, q, dq, tau, g, Ftip):
        # Simple linear response: tau minus small damping
        return np.asarray(tau, dtype=np.float32) - 0.05


@pytest.fixture
def planner():
    joint_limits = [(-1.0, 1.0), (-2.0, 2.0)]
    return OptimizedTrajectoryPlanning(
        StubManipulator(), "nonexistent.urdf", MockDynamics(n=2), joint_limits, use_cuda=False
    )


def test_joint_trajectory_cpu_path(planner: OptimizedTrajectoryPlanning):
    result = planner.joint_trajectory(
        thetastart=[0.0, 0.5],
        thetaend=[1.0, -0.5],
        Tf=1.0,
        N=4,
        method=3,  # cubic
    )
    pos = result["positions"]
    vel = result["velocities"]
    acc = result["accelerations"]

    assert pos.shape == (4, 2)
    # Endpoints respected after clipping
    assert np.allclose(pos[0], [0.0, 0.5])
    assert np.allclose(pos[-1], [1.0, -0.5])
    # CPU path used
    assert planner.performance_stats["cpu_calls"] >= 1
    assert planner.performance_stats["gpu_calls"] == 0


def test_batch_joint_trajectory_cpu(planner: OptimizedTrajectoryPlanning):
    start_batch = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    end_batch = np.array([[1.0, 1.0], [-0.5, 0.5]], dtype=np.float32)
    res = planner.batch_joint_trajectory(start_batch, end_batch, Tf=1.0, N=3, method=3)
    pos = res["positions"]
    assert pos.shape == (2, 3, 2)
    # Compare first trajectory to direct CPU generator
    expected, _, _ = _traj_cpu_njit(start_batch[0], end_batch[0], 1.0, 3, 3)
    assert np.allclose(pos[0], expected)


def test_inverse_dynamics_cpu_clips_to_limits():
    joint_limits = [(-1.0, 1.0)]
    torque_limits = [(-0.2, 0.2)]
    planner = OptimizedTrajectoryPlanning(
        StubManipulator(), "nonexistent.urdf", MockDynamics(n=1), joint_limits, torque_limits, use_cuda=False
    )
    q = np.zeros((2, 1), dtype=np.float32)
    dq = np.zeros_like(q)
    ddq = np.ones_like(q) * 5.0  # large accel to trigger clipping
    torques = planner.inverse_dynamics_trajectory(q, dq, ddq)
    assert np.all(torques <= 0.2 + 1e-6)
    assert np.all(torques >= -0.2 - 1e-6)


def test_forward_dynamics_cpu(planner: OptimizedTrajectoryPlanning):
    thetalist = np.zeros(2, dtype=np.float32)
    dthetalist = np.zeros_like(thetalist)
    taumat = np.zeros((3, 2), dtype=np.float32)
    g = np.array([0, 0, -9.81], dtype=np.float32)
    Ftip = np.zeros((3, 6), dtype=np.float32)

    res = planner.forward_dynamics_trajectory(thetalist, dthetalist, taumat, g, Ftip, dt=0.1, intRes=1)
    assert res["positions"].shape == (3, 2)
    assert res["velocities"].shape == (3, 2)
    assert res["accelerations"].shape == (3, 2)


def test_cartesian_trajectory_cpu(planner: OptimizedTrajectoryPlanning):
    Xstart = np.eye(4, dtype=np.float32)
    Xend = np.eye(4, dtype=np.float32)
    Xend[:3, 3] = np.array([1.0, 0.0, 0.0])
    res = planner.cartesian_trajectory(Xstart, Xend, Tf=1.0, N=5, method=3)
    positions = res["positions"]
    velocities = res["velocities"]
    accelerations = res["accelerations"]
    # Endpoints at start/end translation
    assert np.allclose(positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(positions[-1], [1.0, 0.0, 0.0])
    # Shapes check
    assert velocities.shape == (5, 3)
    assert accelerations.shape == (5, 3)
