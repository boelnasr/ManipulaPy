#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for Singularity module in ManipulaPy.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock,patch
from ManipulaPy.cuda_kernels import CUDA_AVAILABLE
from ManipulaPy.singularity import Singularity
from ManipulaPy.cuda_kernels import CUDA_AVAILABLE

class MockSerialManipulator:
    """Mock class for SerialManipulator to provide a test Jacobian."""

    def jacobian(self, thetalist, frame="space"):
        assert frame == "space"

        # Return an arbitrary 6x6 Jacobian
        if np.allclose(thetalist, 0):
            # Singular configuration (det = 0)
            return np.zeros((6, 6))
        else:
            # Non-singular Jacobian
            J = np.eye(6)
            J[5, 5] = 0.5  # Reduce manipulability
            return J

    def forward_kinematics(self, thetas):
        T = np.eye(4)
        T[:3, 3] = np.sum(thetas)
        return T


@pytest.fixture
def singularity_analyzer():
    robot = MockSerialManipulator()
    return Singularity(robot)


def test_singularity_detection(singularity_analyzer):
    assert singularity_analyzer.singularity_analysis(np.zeros(6)) == True
    assert singularity_analyzer.singularity_analysis(np.ones(6)) == False


def test_condition_number(singularity_analyzer):
    cond_singular = singularity_analyzer.condition_number(np.zeros(6))
    cond_regular = singularity_analyzer.condition_number(np.ones(6))

    assert np.isinf(cond_singular)
    assert cond_regular > 1.0


def test_near_singularity_detection(singularity_analyzer):
    assert singularity_analyzer.near_singularity_detection(np.zeros(6))
    assert singularity_analyzer.near_singularity_detection(np.ones(6))



def test_manipulability_ellipsoid_plot(singularity_analyzer):
    try:
        # Just test it doesn't crash (plotting optional)
        singularity_analyzer.manipulability_ellipsoid(np.ones(6))
    except Exception as e:
        pytest.fail(f"manipulability_ellipsoid raised exception: {e}")

    def jacobian(self, thetalist, frame="space"):
        assert frame == "space"
        if np.allclose(thetalist, 0):
            return np.zeros((6, 6))  # Singular
        else:
            J = np.eye(6)
            J[5, 5] = 0.5
            return J

    def forward_kinematics(self, thetas):
        T = np.eye(4)
        T[:3, 3] = np.full(3, np.sum(thetas))
        return T


    @pytest.fixture
    def singularity_analyzer():
        return Singularity(MockSerialManipulator())


    def test_singularity_detection(singularity_analyzer):
        assert singularity_analyzer.singularity_analysis(np.zeros(6))
        assert not singularity_analyzer.singularity_analysis(np.ones(6))


    def test_condition_number(singularity_analyzer):
        cond_singular = singularity_analyzer.condition_number(np.zeros(6))
        cond_regular = singularity_analyzer.condition_number(np.ones(6))

        assert np.isinf(cond_singular)
        assert cond_regular > 1.0


    def test_near_singularity_detection(singularity_analyzer):
        assert singularity_analyzer.near_singularity_detection(np.zeros(6))
        assert singularity_analyzer.near_singularity_detection(np.ones(6))


    @patch("matplotlib.pyplot.show")  # Suppress actual plot
    def test_manipulability_ellipsoid_plot(mock_show, singularity_analyzer):
        try:
            singularity_analyzer.manipulability_ellipsoid(np.ones(6))
        except Exception as e:
            pytest.fail(f"manipulability_ellipsoid raised: {e}")


    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @patch("matplotlib.pyplot.show")
    def test_plot_workspace_monte_carlo(mock_show, singularity_analyzer):
        joint_limits = [(-1, 1), (-2, 2), (-np.pi, np.pi)]
        try:
            singularity_analyzer.plot_workspace_monte_carlo(joint_limits, num_samples=500)
        except Exception as e:
            pytest.fail(f"plot_workspace_monte_carlo raised: {e}")
