#!/usr/bin/env python3

"""
Extended tests for utils module — fills coverage gaps.

Covers: extract_screw_list edge cases, rotation_logm, MatrixLog6,
MatrixLog3 branches, logm_to_twist, rotation_matrix_to_euler_angles
gimbal-lock, and extract_r_list.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest

import numpy as np

from ManipulaPy import utils


class TestExtractScrewListEdgeCases(unittest.TestCase):
    """Cover edge cases in extract_screw_list."""

    def test_1d_omega_1d_r(self):
        """1D arrays that are multiples of 3 should be reshaped."""
        omega = np.array([0, 0, 1, 0, 1, 0])  # 6 elements → 2 joints
        r = np.array([0.1, 0.2, 0.0, 0.3, 0.4, 0.0])
        S = utils.extract_screw_list(omega, r)
        self.assertEqual(S.shape, (6, 2))

    def test_single_joint_1d(self):
        """Single joint with 1D r → reshaped to (3,1)."""
        omega = np.array([[0], [0], [1]])
        r = np.array([0.1, 0.2, 0.0])  # 1D, size 3
        S = utils.extract_screw_list(omega, r)
        self.assertEqual(S.shape, (6, 1))

    def test_empty_r_list(self):
        """Empty r_list should default to zeros."""
        omega = np.array([[0, 0], [0, 1], [1, 0]])
        r = np.array([])
        S = utils.extract_screw_list(omega, r)
        self.assertEqual(S.shape, (6, 2))

    def test_broadcast_r_single_col(self):
        """Single-column r_list should broadcast to match omega."""
        omega = np.array([[0, 0], [0, 1], [1, 0]])
        r = np.array([[0.1], [0.2], [0.0]])
        S = utils.extract_screw_list(omega, r)
        self.assertEqual(S.shape, (6, 2))

    def test_broadcast_omega_single_col(self):
        """Single-column omega should broadcast to match r."""
        omega = np.array([[0], [0], [1]])
        r = np.array([[0.1, 0.2], [0.0, 0.1], [0.0, 0.0]])
        S = utils.extract_screw_list(omega, r)
        self.assertEqual(S.shape, (6, 2))

    def test_mismatched_shapes_raises(self):
        omega = np.array([[0, 0], [0, 1], [1, 0]])
        r = np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            utils.extract_screw_list(omega, r)

    def test_non_3_row_raises(self):
        omega = np.array([[0, 0], [0, 1]])  # Only 2 rows
        r = np.array([[0.1, 0.2], [0.0, 0.1]])
        with self.assertRaises(ValueError):
            utils.extract_screw_list(omega, r)

    def test_1d_not_multiple_of_3_raises(self):
        omega = np.array([0, 0, 1])
        r = np.array([0.1, 0.2])  # Size 2, not multiple of 3
        with self.assertRaises(ValueError):
            utils.extract_screw_list(omega, r)

    def test_1d_omega_not_multiple_of_3_raises(self):
        omega = np.array([0, 0])  # Size 2
        r = np.array([[0.1], [0.2], [0.0]])
        with self.assertRaises(ValueError):
            utils.extract_screw_list(omega, r)


class TestRotationLogm(unittest.TestCase):
    """Cover rotation_logm edge cases."""

    def test_identity(self):
        omega, theta = utils.rotation_logm(np.eye(3))
        self.assertAlmostEqual(theta, 0.0, places=5)

    def test_90_deg_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        omega, theta = utils.rotation_logm(R)
        self.assertAlmostEqual(theta, np.pi / 2, places=5)
        # Axis should be roughly [0, 0, 1]
        np.testing.assert_array_almost_equal(omega, [0, 0, 1], decimal=3)

    def test_non_3x3_raises(self):
        with self.assertRaises(ValueError):
            utils.rotation_logm(np.eye(4))


class TestMatrixLog6(unittest.TestCase):
    """Cover MatrixLog6 branches."""

    def test_identity_transform(self):
        """Identity should give zero 4x4 matrix logarithm."""
        T = np.eye(4)
        result = utils.MatrixLog6(T)
        np.testing.assert_array_almost_equal(result, np.zeros((4, 4)), decimal=5)

    def test_pure_translation(self):
        """Pure translation (R=I) exercises omega≈0 branch."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = utils.MatrixLog6(T)
        self.assertEqual(result.shape, (4, 4))

    def test_rotation_and_translation(self):
        """General SE(3) should produce a valid 4x4 matrix logarithm."""
        # 90° rotation around z + translation
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        T[:3, 3] = [0.5, 0.3, 0.1]
        result = utils.MatrixLog6(T)
        self.assertEqual(result.shape, (4, 4))
        # Upper-left 3x3 should be skew-symmetric (omega_hat * theta)
        omega_hat = result[:3, :3]
        np.testing.assert_array_almost_equal(omega_hat, -omega_hat.T, decimal=10)


class TestMatrixLog3Branches(unittest.TestCase):
    """Cover MatrixLog3 branches: identity, 180°, and general."""

    def test_identity(self):
        result = utils.MatrixLog3(np.eye(3))
        np.testing.assert_array_almost_equal(result, np.zeros((3, 3)))

    def test_180_z(self):
        """180° around z: acosinput ≈ -1."""
        R = np.diag([-1.0, -1.0, 1.0])
        result = utils.MatrixLog3(R)
        self.assertEqual(result.shape, (3, 3))

    def test_180_y(self):
        """180° around y: 1+R[2][2] ≈ 0, uses second branch."""
        R = np.diag([-1.0, 1.0, -1.0])
        result = utils.MatrixLog3(R)
        self.assertEqual(result.shape, (3, 3))

    def test_180_x(self):
        """180° around x: 1+R[2][2] and 1+R[1][1] ≈ 0, uses third branch."""
        R = np.diag([1.0, -1.0, -1.0])
        result = utils.MatrixLog3(R)
        self.assertEqual(result.shape, (3, 3))

    def test_general_rotation(self):
        """Non-trivial rotation."""
        angle = 0.7
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        result = utils.MatrixLog3(R)
        self.assertEqual(result.shape, (3, 3))
        # Should be skew-symmetric
        np.testing.assert_array_almost_equal(result, -result.T, decimal=10)


class TestLogmToTwist(unittest.TestCase):
    """Cover logm_to_twist."""

    def test_zero_matrix(self):
        logm = np.zeros((4, 4))
        twist = utils.logm_to_twist(logm)
        np.testing.assert_array_almost_equal(twist, np.zeros(6))

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            utils.logm_to_twist(np.zeros((3, 3)))

    def test_general_logm(self):
        logm = np.zeros((4, 4))
        logm[0, 1] = -0.5
        logm[1, 0] = 0.5
        logm[0, 3] = 1.0
        logm[1, 3] = 2.0
        logm[2, 3] = 3.0
        twist = utils.logm_to_twist(logm)
        self.assertEqual(twist.shape, (6,))
        # omega from skew part, v from last column
        self.assertAlmostEqual(twist[2], 0.5)  # omega_z
        self.assertAlmostEqual(twist[3], 1.0)  # v_x


class TestRotationMatrixToEulerGimbalLock(unittest.TestCase):
    """Cover the gimbal-lock (singular) branch."""

    def test_gimbal_lock(self):
        """R[2,0] ≈ ±1 → sy < 1e-6 → singular branch."""
        # Pitch = -90° → gimbal lock
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.0]])
        angles = utils.rotation_matrix_to_euler_angles(R)
        self.assertEqual(angles.shape, (3,))

    def test_normal_rotation(self):
        R = np.array(
            [
                [np.cos(0.3), -np.sin(0.3), 0],
                [np.sin(0.3), np.cos(0.3), 0],
                [0, 0, 1],
            ]
        )
        angles = utils.rotation_matrix_to_euler_angles(R)
        self.assertEqual(angles.shape, (3,))


class TestExtractRList(unittest.TestCase):
    """Cover extract_r_list."""

    def test_6xN_screw_list(self):
        S = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        r = utils.extract_r_list(S)
        # Returns list of r vectors, one per joint
        self.assertEqual(len(r), 2)

    def test_none_returns_empty(self):
        """None input returns empty array."""
        result = utils.extract_r_list(None)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
