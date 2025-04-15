#!/usr/bin/env python3

import unittest
import numpy as np
from ManipulaPy.kinematics import SerialManipulator


class TestKinematics(unittest.TestCase):
    """
    Tests for the kinematics module in ManipulaPy.
    Using standard PoE: at zero angles, forward kinematics => M in both frames.
    Also, a simple IK test that tries to achieve T_desired = M.
    """

    def setUp(self):
        # 1) Screw axes in space frame (6,6)
        self.Slist = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, -1, 0, -0.089, 0, 0],
                [0, -1, 0, -0.089, 0, 0.425],
                [0, -1, 0, -0.089, 0, 0.817],
                [1, 0, 0, 0, 0.109, 0],
                [0, -1, 0, -0.089, 0, 0.817],
            ]
        ).T  # shape => (6,6)

        # 2) Home configuration (M)
        self.M = np.array(
            [[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]]
        )

        # 3) Omega list from top 3 rows (for the constructor)
        self.omega_list = self.Slist[:3, :]

        # 4) Body frame screw axes B_list
        # Typically, B_list is S_list converted via M, but for test, let's use a placeholder
        self.B_list = np.copy(self.Slist)

        # 5) Create the SerialManipulator
        self.robot = SerialManipulator(
            M_list=self.M,
            omega_list=self.omega_list,
            S_list=self.Slist,
            B_list=self.B_list,
        )

    def test_forward_kinematics_space(self):
        """
        In standard PoE, zero angles => product of exponentials is identity,
        then multiply by M => final T(0) = M.
        """
        thetalist = np.zeros(6)
        T_space = self.robot.forward_kinematics(thetalist, frame="space")
        np.testing.assert_array_almost_equal(
            T_space,
            self.M,
            decimal=4,
            err_msg="Space frame at zero angles should match M.",
        )

    def test_forward_kinematics_body(self):
        """
        In standard PoE, zero angles => T(0) = M (body frame).
        Usually T(θ) = M * exp( B1θ1 ) ... exp( Bnθn ).
        """
        thetalist = np.zeros(6)
        T_body = self.robot.forward_kinematics(thetalist, frame="body")
        np.testing.assert_array_almost_equal(
            T_body,
            self.M,
            decimal=4,
            err_msg="Body frame at zero angles should match M.",
        )

    def test_simple_inverse_kinematics(self):
        """
        Simple IK test: target T_desired = M.
        The solver should converge to zero angles (or an equivalent solution).
        """
        target_pose = np.copy(self.M)
        init_guess = np.zeros(6)

        solution, success, _ = self.robot.iterative_inverse_kinematics(
            T_desired=target_pose,
            thetalist0=init_guess,
            eomg=1e-5,
            ev=1e-5,
            max_iterations=500,
        )
        self.assertTrue(success, "IK solver did not converge to a solution for M.")

        final_pose = self.robot.forward_kinematics(solution, frame="space")
        np.testing.assert_array_almost_equal(
            final_pose,
            target_pose,
            decimal=3,
            err_msg="IK solution's forward kinematics does not match M.",
        )


if __name__ == "__main__":
    unittest.main()
