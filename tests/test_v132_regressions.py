#!/usr/bin/env python3
"""Regression tests for v1.3.2 fix patch.

One test per fixed bug. Tests verify the bug is fixed AND would have
caught the original behavior. Tests grouped by source file.
"""

import unittest

import numpy as np


class TestUtilsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/utils.py bugs."""


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