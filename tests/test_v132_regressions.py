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
    def test_mass_matrix_2r_planar_arm_against_analytical(self):
        """Verify mass matrix against analytical 2R planar arm.

        For a 2R arm with point masses m1, m2 at distances L1, L2 from their
        respective joints, the analytical mass matrix is:
            M[0,0] = m1*L1^2 + m2*(L1^2 + L2^2 + 2*L1*L2*cos(theta2))
            M[0,1] = M[1,0] = m2*(L2^2 + L1*L2*cos(theta2))
            M[1,1] = m2*L2^2

        Source: Murray, Li, Sastry "A Mathematical Introduction to Robotic
        Manipulation" Example 4.3.
        """
        from ManipulaPy.dynamics import ManipulatorDynamics

        L1 = L2 = 1.0
        m1 = m2 = 1.0

        # 2R planar arm in space frame: joints rotate about z axis
        omega_list = np.array([[0, 0, 1], [0, 0, 1]]).T  # (3, 2)
        r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T     # joint locations
        M_list = np.eye(4)
        M_list[0, 3] = L1 + L2  # End-effector at full extension

        # Per-link CoM transforms in zero config: link 1's CoM at (L1, 0, 0),
        # link 2's CoM at (L1+L2, 0, 0)
        M_link1_com = np.eye(4); M_link1_com[0, 3] = L1
        M_link2_com = np.eye(4); M_link2_com[0, 3] = L1 + L2
        Mlist_per_link = [M_link1_com, M_link2_com]

        # Spatial inertia: point mass m has G = diag(0, 0, 0, m, m, m) in body frame
        Glist = []
        for m in (m1, m2):
            G = np.zeros((6, 6))
            G[3, 3] = G[4, 4] = G[5, 5] = m
            Glist.append(G)
        Glist = np.array(Glist)

        dyn = ManipulatorDynamics(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            b_list=None,
            S_list=None,
            B_list=None,
            Glist=Glist,
            Mlist_per_link=Mlist_per_link,
        )

        # Test at theta = (0, pi/2)
        theta = np.array([0.0, np.pi / 2])
        M = dyn.mass_matrix(theta)

        c2 = np.cos(theta[1])
        M_expected = np.array([
            [m1 * L1**2 + m2 * (L1**2 + L2**2 + 2 * L1 * L2 * c2),
            m2 * (L2**2 + L1 * L2 * c2)],
            [m2 * (L2**2 + L1 * L2 * c2),
            m2 * L2**2],
        ])

        np.testing.assert_array_almost_equal(M, M_expected, decimal=4)
        self.assertTrue(np.allclose(M, M.T, atol=1e-10), "Mass matrix not symmetric")


    def test_mass_matrix_legacy_path_emits_warning(self):
        """Constructing ManipulatorDynamics without Mlist_per_link should warn."""
        import warnings
        from ManipulaPy.dynamics import ManipulatorDynamics

        omega_list = np.array([[0, 0, 1]]).T
        r_list = np.array([[0, 0, 0]]).T
        M_list = np.eye(4)
        Glist = np.array([np.eye(6)])

        dyn = ManipulatorDynamics(
            M_list=M_list, omega_list=omega_list, r_list=r_list,
            b_list=None, S_list=None, B_list=None, Glist=Glist,
            # Mlist_per_link omitted intentionally
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dyn.mass_matrix(np.array([0.0]))
            self.assertTrue(any("legacy approximation" in str(wi.message) for wi in w))


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
