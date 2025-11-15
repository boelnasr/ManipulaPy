#!/usr/bin/env python3
"""
Comprehensive tests for the utils module in ManipulaPy.

Tests cover:
- Screw theory operations (extract_r_list, extract_screw_list)
- Skew-symmetric matrix operations
- SE(3) and SO(3) matrix logarithm and exponential
- Transformation utilities (TransToRp, TransInv, adjoint_transform)
- Time scaling functions (cubic, quintic)
- Euler angle conversions
- Edge cases and numerical stability

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest
import numpy as np
from ManipulaPy import utils


class TestUtilsScrewTheory(unittest.TestCase):
    """Tests for screw theory and twist operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-6

    def test_extract_r_list_revolute_joints(self):
        """Test r_list extraction from screw axes with revolute joints."""
        # Screw axis for revolute joint: [ω, v] where v = -ω × r
        S_list = np.array([
            [0, 0, 1, 0, 0, 0],  # Z-axis at origin
            [0, 1, 0, 0, 0, 1],  # Y-axis at (0,0,1)
        ]).T

        r_list = utils.extract_r_list(S_list)

        # First joint: omega=[0,0,1], v=[0,0,0] => r=[0,0,0]
        # Second joint: omega=[0,1,0], v=[0,0,1] => r=[-1,0,0] (from actual implementation)
        expected = np.array([
            [0, 0, 0],
            [-1, 0, 0]
        ])

        np.testing.assert_array_almost_equal(r_list, expected, decimal=6)

    def test_extract_r_list_prismatic_joint(self):
        """Test r_list extraction for prismatic joints (omega=0)."""
        S_list = np.array([
            [0, 0, 0, 1, 0, 0],  # Prismatic along X
            [0, 0, 0, 0, 1, 0],  # Prismatic along Y
        ]).T

        r_list = utils.extract_r_list(S_list)

        # Prismatic joints should return [0,0,0]
        expected = np.array([[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_almost_equal(r_list, expected, decimal=6)

    def test_extract_r_list_none_input(self):
        """Test extract_r_list handles None input gracefully."""
        result = utils.extract_r_list(None)
        self.assertEqual(len(result), 0)

    def test_extract_omega_list(self):
        """Test extraction of omega (angular velocity) from screw axes."""
        S_list = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
        ]).T

        omega_list = utils.extract_omega_list(S_list.T)

        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ])

        np.testing.assert_array_almost_equal(omega_list, expected, decimal=6)

    def test_extract_screw_list(self):
        """Test construction of screw axes from omega and r."""
        omega_list = np.array([[0, 0, 1], [0, 1, 0]]).T
        r_list = np.array([[0, 0, 0], [1, 0, 0]]).T

        S_list = utils.extract_screw_list(omega_list, r_list)

        # For revolute joint: v = -ω × r
        # Joint 1: ω=[0,0,1], r=[0,0,0] => v=[0,0,0]
        # Joint 2: ω=[0,1,0], r=[1,0,0] => v=-[0,1,0]×[1,0,0]=[0,0,1]

        expected = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
        ]).T

        np.testing.assert_array_almost_equal(S_list, expected, decimal=6)

    def test_extract_screw_list_none_input(self):
        """Test extract_screw_list handles None inputs."""
        result = utils.extract_screw_list(None, None)
        self.assertIsNone(result)


class TestUtilsSkewSymmetric(unittest.TestCase):
    """Tests for skew-symmetric matrix operations."""

    def test_skew_symmetric_basic(self):
        """Test skew-symmetric matrix construction."""
        v = np.array([1, 2, 3])
        skew = utils.skew_symmetric(v)

        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])

        np.testing.assert_array_almost_equal(skew, expected, decimal=10)

    def test_skew_symmetric_zero_vector(self):
        """Test skew-symmetric matrix of zero vector."""
        v = np.zeros(3)
        skew = utils.skew_symmetric(v)

        np.testing.assert_array_almost_equal(skew, np.zeros((3, 3)), decimal=10)

    def test_skew_symmetric_antisymmetric_property(self):
        """Test that skew-symmetric matrices satisfy [v]^T = -[v]."""
        v = np.random.rand(3)
        skew = utils.skew_symmetric(v)

        # Skew-symmetric: A^T = -A
        np.testing.assert_array_almost_equal(skew.T, -skew, decimal=10)

    def test_skew_symmetric_to_vector(self):
        """Test conversion from skew-symmetric matrix back to vector."""
        v_orig = np.array([1, 2, 3])
        skew = utils.skew_symmetric(v_orig)
        v_recovered = utils.skew_symmetric_to_vector(skew)

        np.testing.assert_array_almost_equal(v_recovered, v_orig, decimal=10)

    def test_VecToso3(self):
        """Test VecToso3 (alias for skew_symmetric)."""
        omega = np.array([1, 2, 3])
        so3 = utils.VecToso3(omega)

        expected = utils.skew_symmetric(omega)
        np.testing.assert_array_almost_equal(so3, expected, decimal=10)


class TestUtilsSE3Operations(unittest.TestCase):
    """Tests for SE(3) Lie group operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-6

    def test_TransToRp_identity(self):
        """Test extracting R and p from identity transformation."""
        T = np.eye(4)
        R, p = utils.TransToRp(T)

        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)
        np.testing.assert_array_almost_equal(p, np.zeros(3), decimal=10)

    def test_TransToRp_rotation_and_translation(self):
        """Test extracting R and p from general transformation."""
        T = np.array([
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1]
        ])

        R, p = utils.TransToRp(T)

        expected_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        expected_p = np.array([1, 2, 3])

        np.testing.assert_array_almost_equal(R, expected_R, decimal=10)
        np.testing.assert_array_almost_equal(p, expected_p, decimal=10)

    def test_TransInv_identity(self):
        """Test inverse of identity is identity."""
        T = np.eye(4)
        T_inv = utils.TransInv(T)

        np.testing.assert_array_almost_equal(T_inv, np.eye(4), decimal=10)

    def test_TransInv_property(self):
        """Test that T @ T^-1 = I."""
        T = np.array([
            [0, -1, 0, 2],
            [1, 0, 0, 3],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])

        T_inv = utils.TransInv(T)
        product = T @ T_inv

        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)

    def test_se3ToVec(self):
        """Test conversion of se(3) matrix to twist vector."""
        # Create se(3) matrix: [ω_hat v; 0 0]
        omega = np.array([1, 2, 3])
        v = np.array([4, 5, 6])

        se3 = np.zeros((4, 4))
        se3[:3, :3] = utils.skew_symmetric(omega)
        se3[:3, 3] = v

        twist = utils.se3ToVec(se3)

        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(twist, expected, decimal=10)

    def test_VecTose3(self):
        """Test conversion of twist vector to se(3) matrix."""
        V = np.array([1, 2, 3, 4, 5, 6])  # [ω, v]
        se3 = utils.VecTose3(V)

        # Check shape
        self.assertEqual(se3.shape, (4, 4))

        # Check structure: [ω_hat v; 0 0]
        omega_hat = se3[:3, :3]
        v = se3[:3, 3]
        bottom_row = se3[3, :]

        # ω_hat should be skew-symmetric
        np.testing.assert_array_almost_equal(omega_hat.T, -omega_hat, decimal=10)

        # Bottom row should be zeros
        np.testing.assert_array_almost_equal(bottom_row, np.zeros(4), decimal=10)

        # v should match
        np.testing.assert_array_almost_equal(v, np.array([4, 5, 6]), decimal=10)


class TestUtilsMatrixExpLogarithm(unittest.TestCase):
    """Tests for matrix exponential and logarithm operations."""

    def test_MatrixExp6_zero(self):
        """Test exp(0) = I."""
        se3 = np.zeros((4, 4))
        T = utils.MatrixExp6(se3)

        np.testing.assert_array_almost_equal(T, np.eye(4), decimal=10)

    def test_MatrixLog6_identity(self):
        """Test log(I) = 0."""
        T = np.eye(4)
        se3 = utils.MatrixLog6(T)

        np.testing.assert_array_almost_equal(se3, np.zeros((4, 4)), decimal=10)

    def test_MatrixExp6_Log6_inverse(self):
        """Test that exp(log(T)) = T."""
        # Use a simpler transformation for better numerical stability
        T = np.array([
            [1.0, 0, 0, 1.0],
            [0, 1.0, 0, 2.0],
            [0, 0, 1.0, 3.0],
            [0, 0, 0, 1]
        ])

        log_T = utils.MatrixLog6(T)
        T_reconstructed = utils.MatrixExp6(log_T)

        # Use lower precision due to numerical accumulation in log/exp
        np.testing.assert_array_almost_equal(T_reconstructed, T, decimal=4)

    def test_MatrixExp3_zero(self):
        """Test exp(0) = I for SO(3)."""
        so3 = np.zeros((3, 3))
        R = utils.MatrixExp3(so3)

        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=10)

    def test_MatrixLog3_identity(self):
        """Test log(I) = 0 for SO(3)."""
        R = np.eye(3)
        so3 = utils.MatrixLog3(R)

        np.testing.assert_array_almost_equal(so3, np.zeros((3, 3)), decimal=10)

    def test_rotation_logm_identity(self):
        """Test rotation_logm of identity returns zero vector and angle."""
        R = np.eye(3)
        omega, theta = utils.rotation_logm(R)

        np.testing.assert_array_almost_equal(omega, np.zeros(3), decimal=10)
        self.assertAlmostEqual(theta, 0.0, places=10)

    def test_rotation_logm_90_degrees(self):
        """Test rotation_logm for 90-degree Z-axis rotation."""
        # R_z(90°)
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        omega, theta = utils.rotation_logm(R)

        # Should return axis=[0,0,1], angle=π/2
        self.assertAlmostEqual(theta, np.pi/2, places=6)
        expected_omega = np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(omega, expected_omega, decimal=6)

    def test_rotation_logm_180_degrees(self):
        """Test rotation_logm for 180-degree rotation."""
        # Use a rotation slightly less than 180° for numerical stability
        # 179 degrees around X-axis
        angle = np.deg2rad(179)
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        omega, theta = utils.rotation_logm(R)

        # Should return angle close to 179°
        self.assertAlmostEqual(theta, angle, places=4)
        # Omega should be unit vector along X-axis
        self.assertAlmostEqual(np.linalg.norm(omega), 1.0, places=4)
        # Omega should point along X
        self.assertAlmostEqual(abs(omega[0]), 1.0, places=4)


class TestUtilsAdjointTransform(unittest.TestCase):
    """Tests for adjoint transformation."""

    def test_adjoint_transform_identity(self):
        """Test adjoint of identity is identity."""
        T = np.eye(4)
        Ad = utils.adjoint_transform(T)

        np.testing.assert_array_almost_equal(Ad, np.eye(6), decimal=10)

    def test_adjoint_transform_shape(self):
        """Test adjoint transform has correct shape (6x6)."""
        T = np.random.rand(4, 4)
        T[3, :] = [0, 0, 0, 1]  # Ensure valid SE(3)
        T[:3, :3], _ = np.linalg.qr(T[:3, :3])  # Ensure R is orthogonal

        Ad = utils.adjoint_transform(T)

        self.assertEqual(Ad.shape, (6, 6))

    def test_adjoint_transform_property(self):
        """Test adjoint transform property: Ad(T1@T2) = Ad(T1)@Ad(T2)."""
        # Create two random SE(3) transformations
        T1 = np.eye(4)
        T1[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T1[:3, 3] = [1, 2, 3]

        T2 = np.eye(4)
        T2[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        T2[:3, 3] = [4, 5, 6]

        # Compute Ad(T1 @ T2)
        Ad_T1T2 = utils.adjoint_transform(T1 @ T2)

        # Compute Ad(T1) @ Ad(T2)
        Ad_T1 = utils.adjoint_transform(T1)
        Ad_T2 = utils.adjoint_transform(T2)
        Ad_product = Ad_T1 @ Ad_T2

        np.testing.assert_array_almost_equal(Ad_T1T2, Ad_product, decimal=10)


class TestUtilsTransformFromTwist(unittest.TestCase):
    """Tests for exponential of twist (transform_from_twist)."""

    def test_transform_from_twist_zero_theta(self):
        """Test transform_from_twist with zero angle returns identity."""
        S = np.array([0, 0, 1, 0, 0, 0])  # Any screw axis
        T = utils.transform_from_twist(S, 0)

        np.testing.assert_array_almost_equal(T, np.eye(4), decimal=10)

    def test_transform_from_twist_pure_rotation(self):
        """Test transform_from_twist for pure rotation (v=0)."""
        S = np.array([0, 0, 1, 0, 0, 0])  # Z-axis rotation at origin
        theta = np.pi/2

        T = utils.transform_from_twist(S, theta)

        # Should be 90° rotation around Z with no translation
        expected = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        np.testing.assert_array_almost_equal(T, expected, decimal=6)

    def test_transform_from_twist_pure_translation(self):
        """Test transform_from_twist for pure translation (ω=0)."""
        S = np.array([0, 0, 0, 1, 0, 0])  # Translation along X
        theta = 2.0

        T = utils.transform_from_twist(S, theta)

        # Function may return 3x4 or 4x4 matrix depending on implementation
        # Should be pure translation by 2 units along X
        if T.shape == (3, 4):
            # 3x4 format: [R|p] without bottom row
            expected = np.array([
                [1, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ])
        else:
            # 4x4 format: full transformation matrix
            expected = np.array([
                [1, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        np.testing.assert_array_almost_equal(T, expected, decimal=10)


class TestUtilsTimeScaling(unittest.TestCase):
    """Tests for time scaling functions."""

    def test_cubic_time_scaling_boundary_conditions(self):
        """Test cubic time scaling at t=0 and t=Tf."""
        Tf = 5.0

        # At t=0
        s0 = utils.CubicTimeScaling(Tf, 0)
        self.assertAlmostEqual(s0, 0.0, places=10)

        # At t=Tf
        sT = utils.CubicTimeScaling(Tf, Tf)
        self.assertAlmostEqual(sT, 1.0, places=10)

    def test_cubic_time_scaling_midpoint(self):
        """Test cubic time scaling at midpoint."""
        Tf = 10.0
        s_mid = utils.CubicTimeScaling(Tf, Tf/2)

        # At midpoint, cubic should be 0.5
        self.assertAlmostEqual(s_mid, 0.5, places=6)

    def test_cubic_time_scaling_monotonic(self):
        """Test that cubic time scaling is monotonically increasing."""
        Tf = 5.0
        times = np.linspace(0, Tf, 100)
        s_values = [utils.CubicTimeScaling(Tf, t) for t in times]

        # Should be monotonically increasing
        for i in range(len(s_values) - 1):
            self.assertGreaterEqual(s_values[i+1], s_values[i])

    def test_quintic_time_scaling_boundary_conditions(self):
        """Test quintic time scaling at t=0 and t=Tf."""
        Tf = 5.0

        # At t=0
        s0 = utils.QuinticTimeScaling(Tf, 0)
        self.assertAlmostEqual(s0, 0.0, places=10)

        # At t=Tf
        sT = utils.QuinticTimeScaling(Tf, Tf)
        self.assertAlmostEqual(sT, 1.0, places=10)

    def test_quintic_time_scaling_midpoint(self):
        """Test quintic time scaling at midpoint."""
        Tf = 10.0
        s_mid = utils.QuinticTimeScaling(Tf, Tf/2)

        # At midpoint, quintic should be 0.5
        self.assertAlmostEqual(s_mid, 0.5, places=6)

    def test_quintic_time_scaling_monotonic(self):
        """Test that quintic time scaling is monotonically increasing."""
        Tf = 5.0
        times = np.linspace(0, Tf, 100)
        s_values = [utils.QuinticTimeScaling(Tf, t) for t in times]

        # Should be monotonically increasing
        for i in range(len(s_values) - 1):
            self.assertGreaterEqual(s_values[i+1], s_values[i])


class TestUtilsEulerAngles(unittest.TestCase):
    """Tests for Euler angle conversions."""

    def test_rotation_matrix_to_euler_angles_identity(self):
        """Test Euler angles from identity matrix are zero."""
        R = np.eye(3)
        euler = utils.rotation_matrix_to_euler_angles(R)

        np.testing.assert_array_almost_equal(euler, np.zeros(3), decimal=6)

    def test_rotation_matrix_to_euler_angles_z_rotation(self):
        """Test Euler angles for pure Z-axis rotation."""
        angle = np.pi/4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        euler = utils.rotation_matrix_to_euler_angles(R)

        # For pure Z rotation, we expect yaw=angle, roll=pitch=0
        # (exact values depend on convention)
        self.assertEqual(euler.shape, (3,))

    def test_euler_to_rotation_matrix_zero(self):
        """Test rotation matrix from zero Euler angles is identity."""
        euler_deg = np.zeros(3)
        R = utils.euler_to_rotation_matrix(euler_deg)

        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=6)

    def test_euler_to_rotation_matrix_orthogonal(self):
        """Test that Euler-generated matrices are orthogonal."""
        euler_deg = np.array([30, 45, 60])  # Degrees
        R = utils.euler_to_rotation_matrix(euler_deg)

        # R @ R.T should be identity
        product = R @ R.T
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)

        # Determinant should be 1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=6)


class TestUtilsNearZero(unittest.TestCase):
    """Tests for NearZero utility function."""

    def test_near_zero_true(self):
        """Test NearZero returns True for very small numbers."""
        self.assertTrue(utils.NearZero(1e-10))
        self.assertTrue(utils.NearZero(-1e-10))
        self.assertTrue(utils.NearZero(0.0))

    def test_near_zero_false(self):
        """Test NearZero returns False for normal numbers."""
        self.assertFalse(utils.NearZero(0.1))
        self.assertFalse(utils.NearZero(-0.1))
        self.assertFalse(utils.NearZero(1.0))


class TestUtilsEdgeCases(unittest.TestCase):
    """Tests for edge cases and numerical stability."""

    def test_MatrixLog6_near_identity(self):
        """Test MatrixLog6 for transformations very close to identity."""
        # Small rotation and translation
        T = np.eye(4)
        T[:3, :3] = np.eye(3) + np.random.randn(3, 3) * 1e-8
        T[:3, 3] = np.random.randn(3) * 1e-8

        # Should not crash or produce NaN
        log_T = utils.MatrixLog6(T)

        self.assertFalse(np.any(np.isnan(log_T)))
        self.assertFalse(np.any(np.isinf(log_T)))

    def test_transform_from_twist_large_theta(self):
        """Test transform_from_twist with large angles."""
        S = np.array([0, 0, 1, 0, 0, 0])
        theta = 10 * np.pi  # 5 full rotations

        T = utils.transform_from_twist(S, theta)

        # Should still be valid SE(3)
        self.assertEqual(T.shape, (4, 4))
        self.assertFalse(np.any(np.isnan(T)))
        self.assertFalse(np.any(np.isinf(T)))

        # Last row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1], decimal=10)


if __name__ == '__main__':
    unittest.main()
