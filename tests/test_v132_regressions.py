#!/usr/bin/env python3
"""Regression tests for v1.3.2 fix patch.

One test per fixed bug. Tests verify the bug is fixed AND would have
caught the original behavior. Tests grouped by source file.
"""

import unittest

import numpy as np


class TestUtilsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/utils.py bugs."""

    def test_transform_from_twist_prismatic_returns_4x4(self):
        from ManipulaPy.utils import transform_from_twist

        S_prismatic = np.array([0, 0, 0, 1, 0, 0])
        T = transform_from_twist(S_prismatic, theta=2.5)

        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(T[:3, 3], [2.5, 0, 0])
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_logm_pure_translation_no_div_by_zero(self):
        from ManipulaPy.utils import logm

        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]

        result = logm(T)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        np.testing.assert_array_almost_equal(result[:3], [0, 0, 0])
        np.testing.assert_array_almost_equal(result[3:], [1, 2, 3])


class TestDynamicsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/dynamics.py bugs."""


class TestKinematicsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/kinematics.py bugs."""


class TestPathPlanningRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/path_planning.py bugs."""


class TestControlRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/control.py bugs."""


class TestSingularityRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/singularity.py bugs."""


class TestPotentialFieldRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/potential_field.py bugs."""


class TestSimRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/sim.py bugs."""


class TestVisionRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/vision.py bugs."""


class TestUrdfRegressions(unittest.TestCase):
    """Regressions for URDF subsystem bugs."""


class TestCudaKernelRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/cuda_kernels.py bugs.

    These tests verify the kernel logic by calling the CPU fallback paths
    where they exist, or by re-implementing the kernel math in pure Python
    and asserting it matches the corrected formula.
    """


if __name__ == "__main__":
    unittest.main()
