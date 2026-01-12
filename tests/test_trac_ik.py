#!/usr/bin/env python3

"""
TRAC-IK Solver Tests - ManipulaPy

Tests for the TRAC-IK style parallel IK solver.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest
import numpy as np
import time
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.trac_ik import TracIKSolver, trac_ik_solve


class TestTracIK(unittest.TestCase):
    """Tests for TRAC-IK solver."""

    def setUp(self):
        """Set up test fixtures with a standard 6-DOF robot configuration."""
        # Screw axes in space frame (6,6)
        self.Slist = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, -0.089, 0, 0],
            [0, -1, 0, -0.089, 0, 0.425],
            [0, -1, 0, -0.089, 0, 0.817],
            [1, 0, 0, 0, 0.109, 0],
            [0, -1, 0, -0.089, 0, 0.817],
        ]).T

        # Home configuration (M)
        self.M = np.array([
            [1, 0, 0, 0.817],
            [0, 1, 0, 0],
            [0, 0, 1, 0.191],
            [0, 0, 0, 1]
        ])

        # Omega list from top 3 rows
        self.omega_list = self.Slist[:3, :]

        # Body frame screw axes
        self.B_list = np.copy(self.Slist)

        # Joint limits
        self.joint_limits = [(-np.pi, np.pi)] * 6

        # Create the SerialManipulator
        self.robot = SerialManipulator(
            M_list=self.M,
            omega_list=self.omega_list,
            S_list=self.Slist,
            B_list=self.B_list,
            joint_limits=self.joint_limits,
        )

    def test_trac_ik_solver_creation(self):
        """Test that TracIKSolver can be created."""
        solver = TracIKSolver(
            fk_func=lambda th: self.robot.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: self.robot.jacobian(th, frame="space"),
            joint_limits=self.joint_limits,
            n_joints=6
        )
        self.assertIsNotNone(solver)

    def test_trac_ik_returns_valid_result(self):
        """Test that TRAC-IK returns a valid result structure."""
        # Use home position (trivial case)
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        theta, success, solve_time = self.robot.trac_ik(
            T_target,
            timeout=0.1,
            eomg=1e-3,
            ev=1e-3
        )

        # Check return types
        self.assertEqual(len(theta), 6, "Should return 6 joint angles")
        self.assertIsInstance(success, bool, "Success should be boolean")
        self.assertIsInstance(solve_time, float, "Solve time should be float")
        self.assertGreater(solve_time, 0, "Solve time should be positive")

    def test_trac_ik_trivial_case(self):
        """Test TRAC-IK on trivial home position case."""
        # Home position is trivially reachable with zero angles
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        theta, success, solve_time = self.robot.trac_ik(
            T_target,
            theta0=np.zeros(6),  # Start at solution
            timeout=0.1,
            eomg=1e-2,
            ev=1e-2
        )

        # For trivial case starting at solution, should succeed
        self.assertTrue(success, "TRAC-IK should succeed for trivial case")

    def test_trac_ik_respects_timeout(self):
        """Test that TRAC-IK respects the timeout parameter."""
        test_angles = np.array([0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
        T_target = self.robot.forward_kinematics(test_angles, frame="space")

        start = time.perf_counter()
        theta, success, solve_time = self.robot.trac_ik(
            T_target,
            timeout=0.05,  # 50ms timeout
            num_restarts=1
        )
        elapsed = time.perf_counter() - start

        # Should complete within reasonable time (timeout + overhead)
        self.assertLess(elapsed, 0.2, "Should complete within timeout + overhead")
        self.assertEqual(len(theta), 6, "Should return valid joint angles")

    def test_trac_ik_respects_joint_limits(self):
        """Test that TRAC-IK solutions respect joint limits."""
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        theta, success, solve_time = self.robot.trac_ik(
            T_target,
            timeout=0.1,
            eomg=1e-2,
            ev=1e-2
        )

        # Check all joints are within limits
        for i, (mn, mx) in enumerate(self.joint_limits):
            if mn is not None:
                self.assertGreaterEqual(theta[i], mn - 1e-6,
                    f"Joint {i} below lower limit")
            if mx is not None:
                self.assertLessEqual(theta[i], mx + 1e-6,
                    f"Joint {i} above upper limit")

    def test_trac_ik_convenience_function(self):
        """Test the trac_ik_solve convenience function."""
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        theta, success, solve_time = trac_ik_solve(
            self.robot,
            T_target,
            theta0=np.zeros(6),
            timeout=0.1
        )

        self.assertEqual(len(theta), 6, "Should return 6 joint angles")
        self.assertIsInstance(success, bool)
        self.assertIsInstance(solve_time, float)

    def test_trac_ik_parallel_execution(self):
        """Test that parallel execution works without errors."""
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        solver = TracIKSolver(
            fk_func=lambda th: self.robot.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: self.robot.jacobian(th, frame="space"),
            joint_limits=self.joint_limits,
            n_joints=6
        )

        # Parallel execution
        theta_par, success_par, time_par = solver.solve(
            T_target, theta0=np.zeros(6), timeout=0.1, use_parallel=True
        )

        # Sequential execution
        theta_seq, success_seq, time_seq = solver.solve(
            T_target, theta0=np.zeros(6), timeout=0.1, use_parallel=False
        )

        # Both should return valid results
        self.assertEqual(len(theta_par), 6)
        self.assertEqual(len(theta_seq), 6)

    def test_trac_ik_multiple_restarts(self):
        """Test that multiple restarts don't cause errors."""
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        # Single restart
        theta1, success1, time1 = self.robot.trac_ik(
            T_target, timeout=0.05, num_restarts=1
        )

        # Multiple restarts
        theta3, success3, time3 = self.robot.trac_ik(
            T_target, timeout=0.1, num_restarts=5
        )

        # Both should return valid results
        self.assertEqual(len(theta1), 6)
        self.assertEqual(len(theta3), 6)


class TestTracIKIntegration(unittest.TestCase):
    """Integration tests comparing TRAC-IK with existing solvers."""

    def setUp(self):
        """Set up test fixtures."""
        self.Slist = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, -0.089, 0, 0],
            [0, -1, 0, -0.089, 0, 0.425],
            [0, -1, 0, -0.089, 0, 0.817],
            [1, 0, 0, 0, 0.109, 0],
            [0, -1, 0, -0.089, 0, 0.817],
        ]).T

        self.M = np.array([
            [1, 0, 0, 0.817],
            [0, 1, 0, 0],
            [0, 0, 1, 0.191],
            [0, 0, 0, 1]
        ])

        self.omega_list = self.Slist[:3, :]
        self.B_list = np.copy(self.Slist)
        self.joint_limits = [(-np.pi, np.pi)] * 6

        self.robot = SerialManipulator(
            M_list=self.M,
            omega_list=self.omega_list,
            S_list=self.Slist,
            B_list=self.B_list,
            joint_limits=self.joint_limits,
        )

    def test_trac_ik_method_exists(self):
        """Test that trac_ik method exists on SerialManipulator."""
        self.assertTrue(hasattr(self.robot, 'trac_ik'))
        self.assertTrue(callable(self.robot.trac_ik))

    def test_trac_ik_vs_iterative_ik_interface(self):
        """Test that TRAC-IK has a compatible interface."""
        T_target = self.robot.forward_kinematics(np.zeros(6), frame="space")

        # TRAC-IK returns (theta, success, time)
        result = self.robot.trac_ik(T_target, timeout=0.05)
        self.assertEqual(len(result), 3, "TRAC-IK should return 3 values")

        theta, success, solve_time = result
        self.assertIsInstance(theta, np.ndarray)
        self.assertIsInstance(success, bool)
        self.assertIsInstance(solve_time, float)


if __name__ == '__main__':
    unittest.main()
