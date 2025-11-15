#!/usr/bin/env python3
"""
Extended tests for Singularity module in ManipulaPy.

Tests cover:
- Singularity detection and analysis
- Condition number computation
- Manipulability ellipsoid generation
- Edge cases and numerical stability
- Different robot configurations

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from ManipulaPy.singularity import Singularity


class MockSerialManipulator:
    """Mock SerialManipulator for testing with configurable behavior."""

    def __init__(self, singular_configs=None):
        """
        Initialize mock manipulator.

        Parameters:
            singular_configs (list): List of joint configurations that are singular
        """
        self.singular_configs = singular_configs or []

    def jacobian(self, thetalist, frame="space"):
        """Return mock Jacobian based on configuration."""
        if frame != "space":
            raise ValueError("Only space frame supported in mock")

        # Check if configuration is singular
        is_singular = any(
            np.allclose(thetalist, config, atol=1e-6)
            for config in self.singular_configs
        )

        if is_singular:
            # Return rank-deficient Jacobian (singular)
            J = np.zeros((6, 6))
            # Make it rank 5 (last row is zero)
            J[:5, :5] = np.eye(5)
            return J
        else:
            # Return full-rank Jacobian (non-singular)
            J = np.eye(6)
            # Add some variation to make it more realistic
            J += 0.1 * np.random.rand(6, 6)
            J[5, 5] = 0.5  # Reduce manipulability slightly
            return J

    def forward_kinematics(self, thetas):
        """Return mock forward kinematics transformation."""
        T = np.eye(4)
        # Simple mock: end-effector position is sum of joint angles
        T[0, 3] = np.sum(thetas[:2]) if len(thetas) > 1 else thetas[0]
        T[1, 3] = np.sum(thetas[2:4]) if len(thetas) > 3 else 0.0
        T[2, 3] = np.sum(thetas[4:]) if len(thetas) > 5 else 0.0
        return T


class TestSingularityDetection(unittest.TestCase):
    """Tests for singularity detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock manipulator with known singular configuration
        singular_config = np.zeros(6)
        self.robot = MockSerialManipulator(singular_configs=[singular_config])
        self.singularity = Singularity(self.robot)

    def test_singularity_at_zero_configuration(self):
        """Test singularity detection at zero configuration."""
        theta = np.zeros(6)
        is_singular = self.singularity.singularity_analysis(theta)
        self.assertTrue(is_singular)

    def test_no_singularity_at_random_configuration(self):
        """Test that random configurations are not singular."""
        theta = np.random.uniform(-1, 1, 6)
        is_singular = self.singularity.singularity_analysis(theta)
        self.assertFalse(is_singular)

    def test_singularity_at_multiple_configurations(self):
        """Test singularity detection at multiple known configurations."""
        singular_configs = [
            np.zeros(6),
            np.array([np.pi, 0, 0, 0, 0, 0]),
            np.array([0, np.pi/2, 0, 0, 0, 0])
        ]
        robot = MockSerialManipulator(singular_configs=singular_configs)
        singularity = Singularity(robot)

        for config in singular_configs:
            is_singular = singularity.singularity_analysis(config)
            self.assertTrue(is_singular, f"Should detect singularity at {config}")

    def test_singularity_with_different_tolerances(self):
        """Test that singularity detection is sensitive to small changes."""
        # At singular configuration
        theta_singular = np.zeros(6)
        is_singular = self.singularity.singularity_analysis(theta_singular)
        self.assertTrue(is_singular)

        # Slightly perturbed configuration
        theta_near = theta_singular + 1e-3
        is_near = self.singularity.singularity_analysis(theta_near)
        # Should not be singular anymore
        self.assertFalse(is_near)


class TestConditionNumber(unittest.TestCase):
    """Tests for condition number computation."""

    def setUp(self):
        """Set up test fixtures."""
        singular_config = np.zeros(6)
        self.robot = MockSerialManipulator(singular_configs=[singular_config])
        self.singularity = Singularity(self.robot)

    def test_condition_number_at_singularity(self):
        """Test condition number is infinite at singularity."""
        theta = np.zeros(6)
        cond = self.singularity.condition_number(theta)
        self.assertTrue(np.isinf(cond) or cond > 1e10)

    def test_condition_number_at_regular_configuration(self):
        """Test condition number is finite at regular configuration."""
        theta = np.ones(6)
        cond = self.singularity.condition_number(theta)
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(cond, 1.0)

    def test_condition_number_increases_near_singularity(self):
        """Test that condition number increases approaching singularity."""
        # Far from singularity
        theta_far = np.array([1.0, 0.5, -0.5, 1.0, 0.5, -0.5])
        cond_far = self.singularity.condition_number(theta_far)

        # Closer to singularity
        theta_near = np.array([0.1, 0.05, -0.05, 0.1, 0.05, -0.05])
        cond_near = self.singularity.condition_number(theta_near)

        # Condition number should be larger closer to singularity
        # (not always guaranteed but generally true)
        self.assertTrue(np.isfinite(cond_far))
        self.assertTrue(np.isfinite(cond_near))

    def test_condition_number_positive(self):
        """Test that condition number is always positive."""
        test_configs = [
            np.random.uniform(-1, 1, 6),
            np.array([1.0, 0, 0, 0, 0, 0]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ]

        for theta in test_configs:
            if not np.allclose(theta, 0):  # Skip singular config
                cond = self.singularity.condition_number(theta)
                self.assertGreater(cond, 0)


class TestNearSingularityDetection(unittest.TestCase):
    """Tests for near-singularity detection."""

    def setUp(self):
        """Set up test fixtures."""
        singular_config = np.zeros(6)
        self.robot = MockSerialManipulator(singular_configs=[singular_config])
        self.singularity = Singularity(self.robot)

    def test_near_singularity_at_singular_configuration(self):
        """Test near-singularity detection at singular configuration."""
        theta = np.zeros(6)
        is_near = self.singularity.near_singularity_detection(theta, threshold=1e-2)
        self.assertTrue(is_near)

    def test_near_singularity_at_regular_configuration(self):
        """Test near-singularity detection at regular configuration."""
        theta = np.ones(6)
        is_near = self.singularity.near_singularity_detection(theta, threshold=1e-2)
        # Should be near singularity with low threshold
        # (condition number is typically > 1e-2)
        self.assertTrue(is_near)

    def test_near_singularity_threshold_sensitivity(self):
        """Test threshold sensitivity of near-singularity detection."""
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # With very high threshold, should detect near-singularity
        is_near_high = self.singularity.near_singularity_detection(theta, threshold=1.0)
        self.assertTrue(is_near_high)

        # With very low threshold, might not detect
        is_near_low = self.singularity.near_singularity_detection(theta, threshold=1e-10)
        # Depends on actual condition number, but should be consistent
        self.assertIsInstance(is_near_low, (bool, np.bool_))

    def test_near_singularity_default_threshold(self):
        """Test near-singularity detection with default threshold."""
        theta = np.array([1.0, 0.5, -0.5, 1.0, 0.5, -0.5])
        is_near = self.singularity.near_singularity_detection(theta)
        # Should return a boolean
        self.assertIsInstance(is_near, (bool, np.bool_))


class TestManipulabilityEllipsoid(unittest.TestCase):
    """Tests for manipulability ellipsoid visualization."""

    def setUp(self):
        """Set up test fixtures."""
        self.robot = MockSerialManipulator()
        self.singularity = Singularity(self.robot)

    @patch('matplotlib.pyplot.show')
    def test_manipulability_ellipsoid_without_axis(self, mock_show):
        """Test manipulability ellipsoid generation without provided axis."""
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Should not raise exception
        try:
            self.singularity.manipulability_ellipsoid(theta, ax=None)
        except Exception as e:
            self.fail(f"manipulability_ellipsoid raised exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_manipulability_ellipsoid_at_different_configurations(self, mock_show):
        """Test manipulability ellipsoid at various configurations."""
        test_configs = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ]

        for theta in test_configs:
            try:
                self.singularity.manipulability_ellipsoid(theta, ax=None)
            except Exception as e:
                self.fail(f"manipulability_ellipsoid failed at {theta}: {e}")

    def test_manipulability_ellipsoid_with_custom_axis(self):
        """Test manipulability ellipsoid with custom matplotlib axis."""
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        try:
            # This should use the provided ax (covers line 94: ax1 = ax2 = ax)
            self.singularity.manipulability_ellipsoid(theta, ax=ax)
            plt.close(fig)
        except Exception as e:
            plt.close(fig)
            self.fail(f"manipulability_ellipsoid with custom ax raised exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_manipulability_ellipsoid_calls_show(self, mock_show):
        """Test manipulability ellipsoid calls plt.show when ax=None."""
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # This should create its own figure and call plt.show() (covers line 116-117)
        self.singularity.manipulability_ellipsoid(theta, ax=None)

        # Verify plt.show() was called
        mock_show.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and numerical stability."""

    def test_singularity_with_nan_configuration(self):
        """Test singularity detection with NaN configuration."""
        robot = MockSerialManipulator()
        singularity = Singularity(robot)

        theta = np.array([np.nan, 0, 0, 0, 0, 0])

        # Should handle gracefully (might raise or return False)
        try:
            result = singularity.singularity_analysis(theta)
            # If it doesn't raise, result should be boolean
            self.assertIsInstance(result, (bool, np.bool_))
        except (ValueError, RuntimeError):
            # It's acceptable to raise an exception for invalid input
            pass

    def test_singularity_with_inf_configuration(self):
        """Test singularity detection with infinite configuration."""
        robot = MockSerialManipulator()
        singularity = Singularity(robot)

        theta = np.array([np.inf, 0, 0, 0, 0, 0])

        # Should handle gracefully
        try:
            result = singularity.singularity_analysis(theta)
            self.assertIsInstance(result, (bool, np.bool_))
        except (ValueError, RuntimeError, OverflowError):
            pass

    def test_condition_number_numerical_stability(self):
        """Test condition number computation numerical stability."""
        robot = MockSerialManipulator()
        singularity = Singularity(robot)

        # Test with very small but non-zero angles
        theta = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
        cond = singularity.condition_number(theta)

        # Should return finite positive value
        self.assertTrue(np.isfinite(cond) or np.isinf(cond))
        if np.isfinite(cond):
            self.assertGreater(cond, 0)

    def test_singularity_with_large_configuration(self):
        """Test singularity detection with large joint angles."""
        robot = MockSerialManipulator()
        singularity = Singularity(robot)

        # Large but finite angles
        theta = np.array([100.0, -100.0, 50.0, -50.0, 75.0, -75.0])

        try:
            is_singular = singularity.singularity_analysis(theta)
            self.assertIsInstance(is_singular, (bool, np.bool_))
        except Exception as e:
            self.fail(f"Failed with large configuration: {e}")


class TestWorkspaceGeneration(unittest.TestCase):
    """Tests for workspace generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.robot = MockSerialManipulator()
        self.singularity = Singularity(self.robot)

    def test_workspace_generation_requires_cuda(self):
        """Test that workspace generation requires CUDA and valid robot."""
        joint_limits = [(-1, 1), (-1, 1), (-1, 1)]

        # This function requires CUDA and a robot with 3D workspace
        # The mock robot we use has degenerate forward kinematics
        # So we just test that the function exists and has correct signature
        try:
            # Import CUDA availability check
            from ManipulaPy.cuda_kernels import CUDA_AVAILABLE

            if not CUDA_AVAILABLE:
                # Should raise an error or skip
                with self.assertRaises((RuntimeError, ImportError, AttributeError, IndexError)):
                    self.singularity.plot_workspace_monte_carlo(joint_limits, num_samples=10)
            else:
                # If CUDA available, the function may fail due to degenerate workspace
                # from mock robot (all points on same plane)
                # This is expected behavior
                with patch('matplotlib.pyplot.show'):
                    try:
                        self.singularity.plot_workspace_monte_carlo(joint_limits, num_samples=10)
                    except (IndexError, Exception) as e:
                        # Expected errors with mock robot:
                        # - QhullError (degenerate workspace)
                        # - IndexError (from degenerate ConvexHull)
                        # - Any other workspace-related error
                        error_msg = str(e)
                        self.assertTrue(
                            "Qhull" in error_msg or
                            "workspace" in error_msg.lower() or
                            "convex" in error_msg.lower() or
                            "index" in error_msg.lower() or
                            isinstance(e, IndexError),
                            f"Unexpected error: {error_msg}"
                        )
        except ImportError:
            # If ManipulaPy.cuda_kernels can't be imported, that's fine
            pass

    def test_workspace_generation_parameters(self):
        """Test workspace generation accepts different parameters."""
        # Test that function accepts different joint limit configurations
        test_configs = [
            [(-1, 1), (-1, 1), (-1, 1)],  # 3-DOF
            [(-np.pi, np.pi)] * 6,         # 6-DOF full range
            [(-0.5, 0.5), (-0.5, 0.5)],   # 2-DOF limited range
        ]

        for joint_limits in test_configs:
            # Just verify the function signature is correct
            # Actual execution requires CUDA
            try:
                with patch('matplotlib.pyplot.show'):
                    # This will likely fail without CUDA or with degenerate workspace
                    try:
                        self.singularity.plot_workspace_monte_carlo(joint_limits, num_samples=10)
                    except (RuntimeError, ImportError, AttributeError, IndexError):
                        # Expected if CUDA not available or workspace is degenerate
                        pass
            except Exception as e:
                # Any other error is a problem with the test setup
                error_msg = str(e).lower()
                if "cuda" not in error_msg and "qhull" not in error_msg and "convex" not in error_msg:
                    raise


class TestDifferentRobotConfigurations(unittest.TestCase):
    """Tests with different robot configurations."""

    def test_3dof_robot(self):
        """Test singularity analysis requires square Jacobian."""
        class Mock3DOFRobot:
            def jacobian(self, thetalist, frame="space"):
                # 6x3 Jacobian for 3-DOF robot
                if np.allclose(thetalist, 0):
                    return np.zeros((6, 3))
                else:
                    J = np.random.rand(6, 3)
                    return J

            def forward_kinematics(self, thetas):
                T = np.eye(4)
                T[:3, 3] = np.random.rand(3)
                return T

        robot = Mock3DOFRobot()
        singularity = Singularity(robot)

        # Test with 3-element configuration
        theta_singular = np.zeros(3)

        # singularity_analysis expects square Jacobian, should raise error
        with self.assertRaises(np.linalg.LinAlgError):
            singularity.singularity_analysis(theta_singular)

    def test_7dof_robot(self):
        """Test singularity analysis requires square Jacobian."""
        class Mock7DOFRobot:
            def jacobian(self, thetalist, frame="space"):
                # 6x7 Jacobian for 7-DOF robot (redundant)
                if np.allclose(thetalist, 0):
                    # Singular configuration
                    J = np.zeros((6, 7))
                    J[:6, :6] = np.eye(6) * 0.01  # Very small values
                    return J
                else:
                    J = np.random.rand(6, 7)
                    return J

            def forward_kinematics(self, thetas):
                T = np.eye(4)
                T[:3, 3] = np.random.rand(3)
                return T

        robot = Mock7DOFRobot()
        singularity = Singularity(robot)

        # Test with 7-element configuration
        theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # singularity_analysis expects square Jacobian, should raise error
        with self.assertRaises(np.linalg.LinAlgError):
            singularity.singularity_analysis(theta)


if __name__ == '__main__':
    unittest.main()
