#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for Singularity module in ManipulaPy.
Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ManipulaPy.cuda_kernels import CUDA_AVAILABLE
from ManipulaPy.singularity import Singularity


class MockSerialManipulator:
    """Mock class for SerialManipulator to provide a test Jacobian."""

    def jacobian(self, thetalist, frame: str = "space") -> np.ndarray:
        """Return a mock 6x6 Jacobian, singular at the zero configuration."""
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

    def forward_kinematics(self, thetas) -> np.ndarray:
        """Return a mock homogeneous transform built from the joint sum."""
        T = np.eye(4)
        T[:3, 3] = np.sum(thetas)
        return T


@pytest.fixture
def singularity_analyzer() -> Singularity:
    """Provide a Singularity analyzer backed by a mock manipulator."""
    robot = MockSerialManipulator()
    return Singularity(robot)


def test_singularity_detection(singularity_analyzer) -> None:
    """Detect singular and non-singular configurations."""
    assert singularity_analyzer.singularity_analysis(np.zeros(6)) == True
    assert singularity_analyzer.singularity_analysis(np.ones(6)) == False


def test_condition_number(singularity_analyzer) -> None:
    """Condition number is infinite at a singularity and finite otherwise."""
    cond_singular = singularity_analyzer.condition_number(np.zeros(6))
    cond_regular = singularity_analyzer.condition_number(np.ones(6))

    assert np.isinf(cond_singular)
    assert cond_regular > 1.0


def test_near_singularity_detection(singularity_analyzer) -> None:
    """Singular J (cond = inf) is flagged; a well-conditioned J (cond = 2) is not."""
    assert singularity_analyzer.near_singularity_detection(np.zeros(6))
    assert not singularity_analyzer.near_singularity_detection(np.ones(6))


class _RankDeficientManipulator:
    """Non-zero rank-5 Jacobian: sigma_min = 0 exactly."""

    def jacobian(self, thetalist, frame: str = "space") -> np.ndarray:
        J = np.eye(6)
        J[5] = J[4]
        return J


def test_near_singularity_rank_deficient_without_warnings() -> None:
    import warnings

    analyzer = Singularity(_RankDeficientManipulator())
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        assert analyzer.near_singularity_detection(np.zeros(6))


@pytest.fixture(scope="module")
def panda_analyzer(tmp_path_factory) -> Singularity:
    """Panda arm (panda_joint1..7, tip panda_hand) from an isolated URDF copy.

    The copy keeps package:// discovery from scanning the checkout.
    """
    import shutil
    import warnings
    from pathlib import Path

    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.urdf import URDF

    src = (
        Path(__file__).resolve().parent.parent
        / "ManipulaPy"
        / "ManipulaPy_data"
        / "franka_panda"
        / "panda.urdf"
    )
    path = shutil.copy(src, tmp_path_factory.mktemp("urdf") / "panda.urdf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = URDF.load(path).extract_screw_axes(tip_link="panda_hand")
    S = params["S_list"]
    robot = SerialManipulator(
        M_list=params["M"],
        omega_list=S[:3, :],
        S_list=S,
        B_list=params["B_list"],
        joint_limits=params["joint_limits"],
    )
    return Singularity(robot)


PANDA_HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])


def test_panda_home_is_not_near_singular(panda_analyzer) -> None:
    assert panda_analyzer.condition_number(PANDA_HOME) == pytest.approx(9.40, abs=0.01)
    assert not panda_analyzer.near_singularity_detection(PANDA_HOME)


def test_panda_straight_arm_is_near_singular(panda_analyzer) -> None:
    """q = 0: straight arm (q4 = 0 is just past its -0.0698 rad limit; math check)."""
    q = np.zeros(7)
    J = np.asarray(panda_analyzer.serial_manipulator.jacobian(q, frame="space"))
    assert np.linalg.svd(J, compute_uv=False)[-1] < 1e-9  # confirm the case first
    assert panda_analyzer.near_singularity_detection(q)


def test_manipulability_ellipsoid_plot(singularity_analyzer) -> None:
    """Plotting the manipulability ellipsoid does not raise."""
    try:
        # Just test it doesn't crash (plotting optional)
        singularity_analyzer.manipulability_ellipsoid(np.ones(6))
    except Exception as e:
        pytest.fail(f"manipulability_ellipsoid raised exception: {e}")

    def jacobian(self, thetalist, frame: str = "space") -> np.ndarray:
        """Return a mock 6x6 Jacobian, singular at the zero configuration."""
        assert frame == "space"
        if np.allclose(thetalist, 0):
            return np.zeros((6, 6))  # Singular
        else:
            J = np.eye(6)
            J[5, 5] = 0.5
            return J

    def forward_kinematics(self, thetas) -> np.ndarray:
        """Return a mock homogeneous transform built from the joint sum."""
        T = np.eye(4)
        T[:3, 3] = np.full(3, np.sum(thetas))
        return T

    @pytest.fixture
    def singularity_analyzer() -> Singularity:
        """Provide a Singularity analyzer backed by a mock manipulator."""
        return Singularity(MockSerialManipulator())

    def test_singularity_detection(singularity_analyzer) -> None:
        """Detect singular and non-singular configurations."""
        assert singularity_analyzer.singularity_analysis(np.zeros(6))
        assert not singularity_analyzer.singularity_analysis(np.ones(6))

    def test_condition_number(singularity_analyzer) -> None:
        """Condition number is infinite at a singularity and finite otherwise."""
        cond_singular = singularity_analyzer.condition_number(np.zeros(6))
        cond_regular = singularity_analyzer.condition_number(np.ones(6))

        assert np.isinf(cond_singular)
        assert cond_regular > 1.0

    def test_near_singularity_detection(singularity_analyzer) -> None:
        """Near-singularity detection flags both test configurations."""
        assert singularity_analyzer.near_singularity_detection(np.zeros(6))
        assert singularity_analyzer.near_singularity_detection(np.ones(6))

    @patch("matplotlib.pyplot.show")  # Suppress actual plot
    def test_manipulability_ellipsoid_plot(mock_show, singularity_analyzer) -> None:
        """Plotting the manipulability ellipsoid does not raise."""
        try:
            singularity_analyzer.manipulability_ellipsoid(np.ones(6))
        except Exception as e:
            pytest.fail(f"manipulability_ellipsoid raised: {e}")

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @patch("matplotlib.pyplot.show")
    def test_plot_workspace_monte_carlo(mock_show, singularity_analyzer) -> None:
        """Monte Carlo workspace plotting runs without raising."""
        joint_limits = [(-1, 1), (-2, 2), (-np.pi, np.pi)]
        try:
            singularity_analyzer.plot_workspace_monte_carlo(
                joint_limits, num_samples=500
            )
        except Exception as e:
            pytest.fail(f"plot_workspace_monte_carlo raised: {e}")
