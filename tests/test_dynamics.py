#!/usr/bin/env python3

"""
test_dynamics.py

A test suite for ManipulaPy's ManipulatorDynamics class,
mirroring the style used in test_kinematics.py. Here, we
explicitly provide S_list, B_list, Glist, etc., so no calls to
utils.extract_screw_list(...) are necessary.
Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest
import numpy as np
from math import pi
from ManipulaPy.dynamics import ManipulatorDynamics


class TestDynamics(unittest.TestCase):
    def setUp(self):
        """
        Build a 6-DOF manipulator for testing.
        We'll explicitly pass S_list, B_list, and Glist
        so the constructor doesn't auto-build from (omega, r).
        """
        # 1) S_list shape => (6,6)
        self.S_list = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, -1, 0, -0.089, 0, 0],
                [0, -1, 0, -0.089, 0, 0.425],
                [0, -1, 0, -0.089, 0, 0.817],
                [1, 0, 0, 0, 0.109, 0],
                [0, -1, 0, -0.089, 0, 0.817],
            ]
        ).T  # shape => (6,6)

        # 2) B_list also shape => (6,6).
        # For simplicity, let’s just copy S_list here, or you can do another approach
        self.B_list = np.copy(self.S_list)

        # 3) M_list => the manipulator's home pose (4x4)
        self.M_list = np.array(
            [[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]]
        )

        # 4) Glist => inertia for each of 6 joints, shape => (6, 6, 6)
        self.Glist = []
        for i in range(6):
            Im = np.eye(6, dtype=float)
            Im[0, 0] = 1.0 + i * 0.1  # vary slightly
            self.Glist.append(Im)
        self.Glist = np.stack(self.Glist, axis=0)  # shape => (6,6,6)

        # Create the manipulator
        self.dynamics = ManipulatorDynamics(
            M_list=self.M_list,
            omega_list=None,  # we won't rely on the auto-building
            r_list=None,
            b_list=None,
            S_list=self.S_list,
            B_list=self.B_list,
            Glist=self.Glist,
        )

        self.n_joints = 6

    def test_mass_matrix(self):
        """
        Check that mass_matrix(...) returns an NxN matrix and it's symmetric.
        """
        thetalist = np.zeros(self.n_joints)
        M = self.dynamics.mass_matrix(thetalist)
        self.assertEqual(
            M.shape, (self.n_joints, self.n_joints), "Mass matrix should be NxN."
        )
        np.testing.assert_array_almost_equal(
            M, M.T, decimal=5, err_msg="Mass matrix is not symmetric."
        )

    def test_velocity_quadratic_forces(self):
        """
        With zero velocity, Coriolis/centrifugal forces c should be zero.
        """
        thetalist = np.zeros(self.n_joints)
        dthetalist = np.zeros(self.n_joints)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        self.assertEqual(
            c.shape,
            (self.n_joints,),
            "Returned shape for velocity_quad_forces must be (N,).",
        )
        np.testing.assert_allclose(
            c,
            0.0,
            atol=1e-8,
            err_msg="With zero velocity, velocity_quadratic_forces(...) should be 0.",
        )

    def test_gravity_forces(self):
        """
        If there's nontrivial inertia, gravity_forces(...) should yield a nonzero vector.
        """
        thetalist = np.zeros(self.n_joints)
        g = [0, 0, -9.81]
        gf = self.dynamics.gravity_forces(thetalist, g)
        self.assertEqual(
            gf.shape, (self.n_joints,), "Gravity forces must be shape (N,)."
        )
        self.assertTrue(
            np.linalg.norm(gf) > 1e-7,
            "Gravity forces should not be zero if mass is nontrivial.",
        )

    def test_inverse_forward_dynamics_consistency(self):
        """
        If we use inverse_dynamics to get torques for a desired ddtheta,
        then forward_dynamics with those torques should yield ddtheta within tolerance.
        """
        thetalist = np.random.uniform(-pi, pi, self.n_joints)
        dthetalist = np.zeros(self.n_joints)
        ddthetalist_des = np.random.uniform(-1, 1, self.n_joints)
        g_vec = [0, 0, -9.81]
        Ftip = np.zeros(6)

        # 1) get torques from inverse_dynamics
        tau = self.dynamics.inverse_dynamics(
            thetalist, dthetalist, ddthetalist_des, g_vec, Ftip
        )

        # 2) feed them to forward_dynamics
        ddthetalist = self.dynamics.forward_dynamics(
            thetalist, dthetalist, tau, g_vec, Ftip
        )

        np.testing.assert_allclose(
            ddthetalist,
            ddthetalist_des,
            atol=1e-3,
            err_msg="Forward dynamics does not match the desired accelerations from inverse_dynamics.",
        )


if __name__ == "__main__":
    unittest.main()
