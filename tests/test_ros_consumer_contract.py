#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Consumer-side smoke test for the manipulapy_ros return contract.

This asserts the *behavioral* half of the public-API freeze: the exact
NumPy return type / shape / dtype that the external ``manipulapy_ros``
MoveIt2 wrapper depends on. Where the signature-snapshot test (Task 0.B)
proves the surface did not change shape, this proves a real call on a real
URDF still hands back plain ``numpy.ndarray`` / ``float64`` values.

The consumer path is deliberate: URDF -> objects via the public
``URDFToSerialManipulator``, never hand-built screw lists, because that is
how a downstream package actually constructs the robot.

IMPORTANT: this test must NEVER import or call ``ManipulaPy.backend.set_backend``.
The ROS wrapper pins ``ManipulaPy>=1.3.2`` and never selects a backend, so it
always sees the frozen default (NumPy) contract. Keeping this test on the
default backend means it fails loudly the day the default starts returning a
torch.Tensor / JAX array / float32 instead of a NumPy float64.
"""

import numpy as np
import pytest

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.singularity import Singularity
from ManipulaPy.urdf_processor import URDFToSerialManipulator


@pytest.fixture(scope="module")
def panda():
    """Load the bundled Franka Panda the way a downstream consumer would.

    NOTE on DOF: the bundled ``franka_panda/panda.urdf`` parses to 8 actuated
    joints (the arm's 7 plus one extra joint in the URDF), so the model's DOF
    is 8, not the bare-arm 7. ``n`` is read back from the constructed model
    rather than hard-coded so the shape assertions stay internally consistent
    with whatever the public URDF path produces.
    """
    processor = URDFToSerialManipulator(get_robot_urdf("panda"), load_meshes=False)
    sm = processor.serial_manipulator
    dyn = processor.dynamics
    n = sm.S_list.shape[1]
    return sm, dyn, n


def _q(n):
    """Deterministic, non-singular joint configuration of length ``n``."""
    return np.linspace(0.1, 0.7, n)


class TestRosConsumerContract:
    """Assert NumPy type/shape/dtype under the DEFAULT backend only."""

    def test_forward_kinematics_space(self, panda):
        sm, _, n = panda
        T = sm.forward_kinematics(_q(n), frame="space")
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert T.dtype == np.float64

    def test_forward_kinematics_body(self, panda):
        sm, _, n = panda
        T = sm.forward_kinematics(_q(n), frame="body")
        assert isinstance(T, np.ndarray)
        assert T.shape == (4, 4)
        assert T.dtype == np.float64

    def test_jacobian(self, panda):
        sm, _, n = panda
        J = sm.jacobian(_q(n))
        assert isinstance(J, np.ndarray)
        assert J.shape == (6, n)
        assert J.dtype == np.float64

    def test_iterative_inverse_kinematics(self, panda):
        sm, _, n = panda
        q = _q(n)
        T_desired = sm.forward_kinematics(q, frame="space")
        # Short iteration budget: convergence is irrelevant to the contract,
        # only the returned array's type/dtype/shape matter.
        sol, success, iters = sm.iterative_inverse_kinematics(
            T_desired, q, max_iterations=5
        )
        assert isinstance(sol, np.ndarray)
        assert sol.shape == (n,)
        assert sol.dtype == np.float64
        assert isinstance(success, bool)
        assert isinstance(iters, int)

    def test_mass_matrix(self, panda):
        _, dyn, n = panda
        M = dyn.mass_matrix(_q(n))
        assert isinstance(M, np.ndarray)
        assert M.shape == (n, n)
        assert M.dtype == np.float64

    def test_inverse_dynamics(self, panda):
        _, dyn, n = panda
        tau = dyn.inverse_dynamics(
            _q(n), np.zeros(n), np.zeros(n), [0.0, 0.0, -9.81], np.zeros(6)
        )
        assert isinstance(tau, np.ndarray)
        assert tau.shape == (n,)
        assert tau.dtype == np.float64

    def test_gravity_forces(self, panda):
        _, dyn, n = panda
        g = dyn.gravity_forces(_q(n))
        assert isinstance(g, np.ndarray)
        assert g.shape == (n,)
        assert g.dtype == np.float64

    def test_velocity_quadratic_forces(self, panda):
        _, dyn, n = panda
        c = dyn.velocity_quadratic_forces(_q(n), np.zeros(n))
        assert isinstance(c, np.ndarray)
        assert c.shape == (n,)
        assert c.dtype == np.float64

    def test_condition_number(self, panda):
        sm, _, n = panda
        sing = Singularity(sm)
        cond = sing.condition_number(_q(n))
        # Task 0.B recorded this scalar as numpy.float64, not a Python float.
        assert isinstance(cond, np.float64)

    def test_near_singularity_detection(self, panda):
        sm, _, n = panda
        sing = Singularity(sm)
        near = sing.near_singularity_detection(_q(n))
        # Task 0.B recorded this scalar as numpy.bool_, not a Python bool.
        assert isinstance(near, np.bool_)
