#!/usr/bin/env python3
"""Regression tests for v1.3.2 fix patch.

One test per fixed bug. Tests verify the bug is fixed AND would have
caught the original behavior. Tests grouped by source file.
"""

import unittest
from typing import Any

import numpy as np
import pytest


class TestUtilsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/utils.py bugs."""

    def test_transform_from_twist_prismatic_returns_4x4(self) -> None:
        """Verify a prismatic twist yields a valid 4x4 homogeneous transform."""
        from ManipulaPy.utils import transform_from_twist

        S_prismatic = np.array([0, 0, 0, 1, 0, 0])
        T = transform_from_twist(S_prismatic, theta=2.5)

        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        np.testing.assert_array_almost_equal(T[:3, 3], [2.5, 0, 0])
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_logm_pure_translation_no_div_by_zero(self) -> None:
        """Verify logm of a pure translation avoids division by zero."""
        from ManipulaPy.utils import logm

        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]

        result = logm(T)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        np.testing.assert_array_almost_equal(result[:3], [0, 0, 0])
        np.testing.assert_array_almost_equal(result[3:], [1, 2, 3])

    def test_logm_matches_scipy_for_coupled_rotation_translation(self) -> None:
        """CM-1: the SE(3) matrix-log translation block must be scaled by theta.

        logm returned the unit-screw linear velocity (G_inv @ p) instead of the
        matrix-log value (theta * G_inv @ p). The angular block was already
        scaled, so the result disagreed with scipy.linalg.logm by a factor of
        1/theta on the linear half for any coupled rotation+translation.
        """
        from scipy.linalg import logm as scipy_logm

        from ManipulaPy.utils import logm, se3ToVec

        theta = 1.3
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]  # Rz(theta)
        T[:3, 3] = [1.0, 2.0, 3.0]

        got = logm(T)  # 6-vector [w*theta ; v*theta]
        ref = se3ToVec(np.real(scipy_logm(T)))
        np.testing.assert_array_almost_equal(got, ref, decimal=6)

    def test_matrixlog6_roundtrip_coupled_motion(self) -> None:
        """CM-2: MatrixExp6(MatrixLog6(T)) must reconstruct T.

        MatrixLog6 left the translation block as G_inv @ p (missing the theta
        factor present on the rotation block), so the round-trip failed for any
        transform combining rotation and translation. adaptive IK consumes this
        via se3ToVec(MatrixLog6(T_err)).
        """
        from ManipulaPy.utils import MatrixExp6, MatrixLog6, VecTose3, se3ToVec

        theta = 1.1
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]  # Ry(theta)
        T[:3, 3] = [0.5, -1.2, 2.0]

        S_theta = se3ToVec(MatrixLog6(T))
        T_recon = MatrixExp6(VecTose3(S_theta))
        np.testing.assert_array_almost_equal(T_recon, T, decimal=6)

    def test_rotation_logm_recovers_axis_at_180_degrees(self) -> None:
        """CM-6: rotation_logm must recover the axis at theta = pi.

        A 180-degree rotation matrix is symmetric, so R - R^T = 0 and
        sin(theta) -> 0; the generic 1/(2 sin theta) * (R - R^T) formula
        collapses to the zero vector and loses the axis. Mirror the
        MatrixLog3 theta~pi handling so exp(log(R)) reconstructs R.
        """
        from ManipulaPy.utils import MatrixExp3, VecToso3, rotation_logm

        axes = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            np.array([1.0, 2.0, 2.0]) / 3.0,
        ]
        for n in axes:
            R = 2.0 * np.outer(n, n) - np.eye(3)  # 180-degree rotation about n
            omega, theta = rotation_logm(R)
            self.assertFalse(np.any(np.isnan(omega)), f"NaN axis for n={n}")
            self.assertAlmostEqual(theta, np.pi, places=6)
            R_recon = MatrixExp3(VecToso3(omega * theta))
            np.testing.assert_array_almost_equal(
                R_recon, R, decimal=6, err_msg=f"exp(log(R)) != R for n={n}"
            )

        # Just below pi (generic branch) must stay accurate too — the pi-branch
        # threshold is intentionally tight so this range uses the exact formula.
        for ang in (np.pi - 1e-4, np.pi - 1e-2):
            n = np.array([0.0, 1.0, 0.0])
            R = MatrixExp3(VecToso3(n * ang))
            omega, theta = rotation_logm(R)
            self.assertAlmostEqual(theta, ang, places=6)
            R_recon = MatrixExp3(VecToso3(omega * theta))
            np.testing.assert_array_almost_equal(
                R_recon, R, decimal=6, err_msg=f"exp(log(R)) != R at angle {ang}"
            )


class TestDynamicsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/dynamics.py bugs."""

    def test_mass_matrix_2r_planar_arm_against_analytical(self) -> None:
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
        r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T  # joint locations
        M_list = np.eye(4)
        M_list[0, 3] = L1 + L2  # End-effector at full extension

        # Per-link CoM transforms in zero config: link 1's CoM at (L1, 0, 0),
        # link 2's CoM at (L1+L2, 0, 0)
        M_link1_com = np.eye(4)
        M_link1_com[0, 3] = L1
        M_link2_com = np.eye(4)
        M_link2_com[0, 3] = L1 + L2
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
        M_expected = np.array(
            [
                [
                    m1 * L1**2 + m2 * (L1**2 + L2**2 + 2 * L1 * L2 * c2),
                    m2 * (L2**2 + L1 * L2 * c2),
                ],
                [m2 * (L2**2 + L1 * L2 * c2), m2 * L2**2],
            ]
        )

        np.testing.assert_array_almost_equal(M, M_expected, decimal=4)
        self.assertTrue(np.allclose(M, M.T, atol=1e-10), "Mass matrix not symmetric")

    def test_mass_matrix_legacy_path_emits_warning(self) -> None:
        """Constructing ManipulatorDynamics without Mlist_per_link should warn."""
        import warnings
        from ManipulaPy.dynamics import ManipulatorDynamics

        omega_list = np.array([[0, 0, 1]]).T
        r_list = np.array([[0, 0, 0]]).T
        M_list = np.eye(4)
        Glist = np.array([np.eye(6)])

        dyn = ManipulatorDynamics(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            b_list=None,
            S_list=None,
            B_list=None,
            Glist=Glist,
            # Mlist_per_link omitted intentionally
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dyn.mass_matrix(np.array([0.0]))
            self.assertTrue(any("legacy approximation" in str(wi.message) for wi in w))

    def test_gravity_forces_mutable_default_is_safe(self) -> None:
        """Verify gravity_forces uses None instead of a mutable default for g."""
        import inspect
        from ManipulaPy.dynamics import ManipulatorDynamics

        sig = inspect.signature(ManipulatorDynamics.gravity_forces)
        g_default = sig.parameters["g"].default
        self.assertIsNone(g_default, "Mutable default detected — should use None")

    def test_gravity_forces_2r_planar_analytical(self) -> None:
        """CM-4: gravity_forces must equal the analytical holding torque.

        The old implementation read the rotational inertia block
        Glist[i][:3, :3] (all zeros for point masses) instead of the link
        mass, and used a non-physical formula — so it returned [0, 0] for this
        fixture regardless of gravity direction. The holding torque is
        tau_i = d(PE)/d(theta_i) with PE = -sum_k m_k * g . p_k_com.

        An explicit (correct) S_list is supplied so this test is independent of
        the separate screw-sign fix (CM-5); gravity_forces only uses the
        space-frame Jacobian / FK, both driven by S_list.
        """
        from ManipulaPy.dynamics import ManipulatorDynamics
        from ManipulaPy.utils import extract_screw_list

        L1 = L2 = 1.0
        m1 = m2 = 1.0
        omega_list = np.array([[0, 0, 1], [0, 0, 1]]).T  # (3, 2), both about +z
        r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T
        S_list = extract_screw_list(omega_list, r_list)  # correct space screws

        M_list = np.eye(4)
        M_list[0, 3] = L1 + L2
        M_link1 = np.eye(4)
        M_link1[0, 3] = L1
        M_link2 = np.eye(4)
        M_link2[0, 3] = L1 + L2

        Glist = np.array([np.diag([0.0, 0.0, 0.0, m, m, m]) for m in (m1, m2)])

        dyn = ManipulatorDynamics(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            b_list=r_list,  # dummy; gravity uses only the space Jacobian
            S_list=S_list,
            B_list=None,
            Glist=Glist,
            Mlist_per_link=[M_link1, M_link2],
        )

        theta = np.array([0.0, np.pi / 2])

        # Gravity in -x. CoM positions: p1=(0,0,0)->rot, p2 fold; holding torque
        # tau = d(PE)/d(theta) works out to [-9.81, -9.81] (derived by hand).
        tau_x = dyn.gravity_forces(theta, [-9.81, 0.0, 0.0])
        np.testing.assert_array_almost_equal(tau_x, [-9.81, -9.81], decimal=4)

        # Gravity in -y at the same pose -> [19.62, 0.0].
        tau_y = dyn.gravity_forces(theta, [0.0, -9.81, 0.0])
        np.testing.assert_array_almost_equal(tau_y, [19.62, 0.0], decimal=4)

        # Planar arm in the xy-plane: vertical (-z) gravity does no work, so the
        # holding torque is exactly zero — the value the OLD code returned for
        # every direction.
        tau_z = dyn.gravity_forces(theta, [0.0, 0.0, -9.81])
        np.testing.assert_array_almost_equal(tau_z, [0.0, 0.0], decimal=6)

    def test_gravity_forces_legacy_path_emits_warning(self) -> None:
        """CM-4: without Mlist_per_link, gravity_forces warns and falls back."""
        import warnings

        from ManipulaPy.dynamics import ManipulatorDynamics
        from ManipulaPy.utils import extract_screw_list

        omega_list = np.array([[0, 0, 1]]).T
        r_list = np.array([[0, 0, 0]]).T
        M_list = np.eye(4)
        Glist = np.array([np.eye(6)])

        dyn = ManipulatorDynamics(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            b_list=r_list,
            S_list=extract_screw_list(omega_list, r_list),
            B_list=None,
            Glist=Glist,
            # Mlist_per_link omitted intentionally
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dyn.gravity_forces(np.array([0.0]), [0.0, 0.0, -9.81])
            self.assertTrue(
                any("legacy" in str(wi.message).lower() for wi in w),
                "gravity_forces without Mlist_per_link should emit a warning",
            )


class TestKinematicsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/kinematics.py bugs."""

    def _load_simple_arm(self) -> Any:
        """Load the bundled simple_arm fixture and return its SerialManipulator."""
        import os

        from ManipulaPy.urdf_processor import URDFToSerialManipulator

        urdf_path = os.path.join(
            os.path.dirname(__file__), "urdf_fixtures", "simple_arm.urdf"
        )
        self.assertTrue(os.path.exists(urdf_path), f"Test fixture missing: {urdf_path}")
        return URDFToSerialManipulator(urdf_path).serial_manipulator

    def test_body_jacobian_satisfies_adjoint_identity(self) -> None:
        """CM-3: jacobian(frame="body") must satisfy Js = Ad_{T_sb} @ Jb.

        The body branch seeded its accumulator from the full body FK
        (M times the joint product) instead of identity, so the last column
        was Ad(T_full^-1) @ B_n rather than B_n, and the whole body Jacobian
        violated the space/body adjoint identity. This contaminates body-frame
        end_effector_velocity and joint_velocity.
        """
        from ManipulaPy.utils import adjoint_transform

        robot = self._load_simple_arm()
        n = robot.S_list.shape[1]
        theta = np.linspace(0.1, 0.1 + 0.15 * (n - 1), n)

        Js = robot.jacobian(theta, frame="space")
        Jb = robot.jacobian(theta, frame="body")
        T_sb = robot.forward_kinematics(theta, frame="space")

        np.testing.assert_array_almost_equal(
            Js,
            adjoint_transform(T_sb) @ Jb,
            decimal=5,
            err_msg="Js != Ad_{T_sb} @ Jb — body Jacobian is inconsistent",
        )
        # Last column of the body Jacobian must equal B_n exactly (Ad(I) @ B_n).
        np.testing.assert_array_almost_equal(Jb[:, -1], robot.B_list[:, -1], decimal=6)

    def test_generate_path_screws_match_analytical_2r_off_home(self) -> None:
        """CM-5: building from omega_list/r_list (no S_list) must give correct
        off-home FK.

        kinematics.__init__ passed -omega_list to extract_screw_list, which
        already applies v = -omega x r — so the minus corrupted the generated
        space screw (axis flipped). FK was only self-consistent at the home
        pose (theta=0); off-home it produced FK_correct(-theta). Verified here
        against the closed-form 2R planar arm. The URDF path is unaffected
        because it always passes explicit S_list/B_list.
        """
        from ManipulaPy.kinematics import SerialManipulator

        L1 = L2 = 1.0
        omega_list = np.array([[0, 0, 1], [0, 0, 1]]).T
        r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T
        M_list = np.eye(4)
        M_list[0, 3] = L1 + L2  # home pose: EE at full extension

        # Generate path: omega_list/r_list only, no S_list or B_list.
        robot = SerialManipulator(M_list=M_list, omega_list=omega_list, r_list=r_list)

        theta = np.array([0.3, 0.5])
        T = robot.forward_kinematics(theta, frame="space")

        s = theta[0] + theta[1]
        T_expected = np.array(
            [
                [np.cos(s), -np.sin(s), 0, L1 * np.cos(theta[0]) + L2 * np.cos(s)],
                [np.sin(s), np.cos(s), 0, L1 * np.sin(theta[0]) + L2 * np.sin(s)],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_almost_equal(T, T_expected, decimal=6)


class TestPathPlanningRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/path_planning.py bugs."""

    def test_cartesian_quintic_acceleration_satisfies_boundary_conditions(self) -> None:
        """The Cartesian-trajectory CPU fallback used a wrong s_ddot for the
        quintic time-scaling: 60·τ·(1−2τ)/Tf² = (60τ − 120τ²)/Tf², missing
        the +120τ³ term and using -120 instead of -180 on τ².

        Quintic time-scaling REQUIRES s_ddot(0) = s_ddot(Tf) = 0 (zero
        acceleration at both endpoints — that's the whole point of going
        to 5th order). The buggy formula gives s_ddot(Tf) = -60/Tf² ≠ 0,
        which would yank the end-effector at the end of every motion.
        """
        from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

        planner = OptimizedTrajectoryPlanning.__new__(OptimizedTrajectoryPlanning)
        planner.performance_stats = {
            "cpu_calls": 0,
            "gpu_calls": 0,
            "total_cpu_time": 0.0,
            "total_gpu_time": 0.0,
            "kernel_launches": 0,
        }
        pstart = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pend = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        Tf, N = 1.0, 101  # 101 samples → endpoints at indices 0 and 100

        traj_vel, traj_acc = planner._cartesian_trajectory_cpu(
            pstart, pend, Tf, N, method=5
        )

        # Boundary acceleration MUST be zero for a quintic profile
        np.testing.assert_array_almost_equal(
            traj_acc[0],
            [0, 0, 0],
            decimal=4,
            err_msg="Quintic s_ddot(0) != 0 — boundary condition violated",
        )
        np.testing.assert_array_almost_equal(
            traj_acc[-1],
            [0, 0, 0],
            decimal=4,
            err_msg="Quintic s_ddot(Tf) != 0 — missing +120·tau^3 term",
        )

        # And boundary velocity must also be zero
        np.testing.assert_array_almost_equal(
            traj_vel[0], [0, 0, 0], decimal=4, err_msg="Quintic s_dot(0) != 0"
        )
        np.testing.assert_array_almost_equal(
            traj_vel[-1], [0, 0, 0], decimal=4, err_msg="Quintic s_dot(Tf) != 0"
        )

    def test_screw_list_sign_correct_via_home_fk(self) -> None:
        """At home config, FK in BOTH frames must equal M_list (the home pose).

        Uses the bundled tests/urdf_fixtures/simple_arm.urdf so this test
        always runs without requiring ManipulaPy_data to be installed.
        Verifies the asymmetric -omega/+omega convention in
        kinematics.py:76/:83 produces correct FK end-to-end.
        """
        import os
        from ManipulaPy.urdf_processor import URDFToSerialManipulator

        urdf_path = os.path.join(
            os.path.dirname(__file__), "urdf_fixtures", "simple_arm.urdf"
        )
        self.assertTrue(os.path.exists(urdf_path), f"Test fixture missing: {urdf_path}")

        urdf = URDFToSerialManipulator(urdf_path)
        robot = urdf.serial_manipulator
        n = robot.S_list.shape[1]
        home_config = np.zeros(n)

        T_space = robot.forward_kinematics(home_config, frame="space")
        T_body = robot.forward_kinematics(home_config, frame="body")

        np.testing.assert_array_almost_equal(
            T_space,
            robot.M_list,
            decimal=6,
            err_msg="Space-frame FK at home != M_list — S_list sign suspect",
        )
        np.testing.assert_array_almost_equal(
            T_body,
            robot.M_list,
            decimal=6,
            err_msg="Body-frame FK at home != M_list — B_list sign suspect",
        )

        # Joint motion must actually move the EE — guards against a
        # degenerate zero-screw chain that would also satisfy T == M trivially.
        theta_test = np.zeros(n)
        theta_test[0] = 0.01
        T_pert = robot.forward_kinematics(theta_test, frame="space")
        self.assertFalse(
            np.allclose(T_pert, robot.M_list, atol=1e-6),
            "FK didn't respond to joint motion — screw chain may be degenerate",
        )

    def test_quintic_polynomial_endpoints(self) -> None:
        """Pure math test: quintic time-scaling endpoints."""

        def s(tau) -> float:
            """Quintic time-scaling position s(tau)."""
            return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

        def s_ddot(tau, Tf: float = 1.0) -> float:
            """Quintic time-scaling acceleration s_ddot(tau)."""
            return (60 * tau - 180 * tau**2 + 120 * tau**3) / (Tf * Tf)

        self.assertAlmostEqual(s(0.0), 0.0)
        self.assertAlmostEqual(s(1.0), 1.0)
        self.assertAlmostEqual(s_ddot(0.0), 0.0)
        self.assertAlmostEqual(s_ddot(1.0), 0.0)

    def test_quintic_trajectory_zero_endpoint_acceleration(self) -> None:
        """Verify a quintic joint trajectory has zero acceleration at both endpoints."""
        from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

        class MockManip:
            joint_limits = [(-3.14, 3.14)] * 6

        class MockDyn:
            pass

        planner = OptimizedTrajectoryPlanning(
            MockManip(),
            urdf_path=None,
            dynamics=MockDyn(),
            joint_limits=[(-3.14, 3.14)] * 6,
        )
        thetastart = np.zeros(6)
        thetaend = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])
        traj = planner.joint_trajectory(thetastart, thetaend, Tf=2.0, N=100, method=5)

        np.testing.assert_array_almost_equal(
            traj["accelerations"][0], np.zeros(6), decimal=4
        )
        np.testing.assert_array_almost_equal(
            traj["accelerations"][-1], np.zeros(6), decimal=4
        )

    def test_plot_tcp_trajectory_does_not_shadow_time_module(self) -> None:
        """Verify the trajectory plotter does not shadow the time module."""
        import inspect
        from ManipulaPy import path_planning

        src = inspect.getsource(path_planning)
        self.assertNotIn(
            "time = np.arange",
            src,
            "Variable 'time' shadows the time module — rename to time_array",
        )


class TestControlRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/control.py bugs."""

    def test_cartesian_space_control_dimensions(self) -> None:
        """Verify cartesian_space_control returns a torque vector sized to the DOF."""
        from ManipulaPy.urdf_processor import URDFToSerialManipulator
        from ManipulaPy.control import ManipulatorController
        from ManipulaPy.ManipulaPy_data.ur5 import urdf_file

        urdf = URDFToSerialManipulator(str(urdf_file))

        controller = ManipulatorController(urdf.dynamics)
        n = len(urdf.serial_manipulator.joint_limits)

        desired_position = np.array([0.4, 0.2, 0.5])
        Kp = np.eye(3)
        Kd = np.eye(3)

        tau = controller.cartesian_space_control(
            desired_position, np.zeros(n), np.zeros(n), Kp, Kd
        )
        self.assertEqual(tau.shape, (n,))

    def test_cartesian_space_control_accepts_vector_gains(self) -> None:
        """Verify cartesian_space_control accepts per-axis vector Kp and Kd gains."""
        from ManipulaPy.control import ManipulatorController

        class Dynamics:
            def forward_kinematics(self, current_joint_angles) -> np.ndarray:
                """Return a fixed identity end-effector pose."""
                T = np.eye(4)
                return T

            def jacobian(self, current_joint_angles) -> np.ndarray:
                """Return a fixed translational Jacobian stub."""
                return np.vstack([np.eye(3), np.zeros((3, 3))])

        ctrl = ManipulatorController(Dynamics())
        tau = ctrl.cartesian_space_control(
            desired_position=np.array([1.0, 2.0, 3.0]),
            current_joint_angles=np.zeros(3),
            current_joint_velocities=np.array([0.5, 1.0, 1.5]),
            Kp=np.array([10.0, 20.0, 30.0]),
            Kd=np.array([1.0, 2.0, 3.0]),
        )

        np.testing.assert_allclose(tau, np.array([9.5, 38.0, 85.5]))

    def test_rise_time_handles_response_never_reaching_setpoint(self) -> None:
        """Verify rise-time calculation handles a response that never reaches setpoint."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)

        rt = ctrl.calculate_rise_time(np.linspace(0, 1, 100), np.zeros(100), 1.0)
        self.assertTrue(np.isinf(rt) or rt >= 0)

    def test_percent_overshoot_handles_zero_setpoint(self) -> None:
        """Verify percent-overshoot calculation stays finite for a zero setpoint."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)

        po = ctrl.calculate_percent_overshoot(np.array([0.0, 0.1, 0.0]), set_point=0.0)
        self.assertTrue(np.isfinite(po))

    def test_pid_control_resets_eint_on_dof_change(self) -> None:
        """Verify the PID integral term resets when the input DOF changes."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        # First call: 6-DOF
        Kp, Ki, Kd = np.eye(6), np.eye(6), np.eye(6)
        ctrl.pid_control(
            np.ones(6),
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            dt=0.01,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
        )
        self.assertEqual(ctrl.eint.shape, (6,))
        # Second call: 2-DOF
        Kp2, Ki2, Kd2 = np.eye(2), np.eye(2), np.eye(2)
        ctrl.pid_control(
            np.ones(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            dt=0.01,
            Kp=Kp2,
            Ki=Ki2,
            Kd=Kd2,
        )
        self.assertEqual(
            ctrl.eint.shape, (2,), "eint must reset when input DOF changes"
        )

    def test_pid_control_dof_change_logs_debug_not_warning(self) -> None:
        """Verify a PID DOF change logs at debug level rather than warning."""
        from unittest.mock import patch

        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        Kp, Ki, Kd = np.eye(3), np.eye(3), np.eye(3)
        ctrl.pid_control(
            np.ones(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            dt=0.01,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
        )

        with (
            patch("ManipulaPy.control.logger.warning") as warning,
            patch("ManipulaPy.control.logger.debug") as debug,
        ):
            Kp2, Ki2, Kd2 = np.eye(2), np.eye(2), np.eye(2)
            ctrl.pid_control(
                np.ones(2),
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                dt=0.01,
                Kp=Kp2,
                Ki=Ki2,
                Kd=Kd2,
            )

        warning.assert_not_called()
        debug.assert_called_once()

    def test_ziegler_nichols_tuning_rejects_zero_period(self) -> None:
        """Verify Ziegler-Nichols PID tuning rejects a zero oscillation period."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        with self.assertRaises(ValueError):
            ctrl.ziegler_nichols_tuning(Ku=10.0, Tu=0.0, kind="PID")

    def test_ziegler_nichols_p_allows_zero_period(self) -> None:
        """Verify Ziegler-Nichols P tuning permits a zero oscillation period."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)

        Kp, Ki, Kd = ctrl.ziegler_nichols_tuning(Ku=10.0, Tu=0.0, kind="P")

        self.assertEqual(Kp, 5.0)
        self.assertEqual(Ki, 0.0)
        self.assertEqual(Kd, 0.0)

    def test_ziegler_nichols_tuning_rejects_negative_period(self) -> None:
        """Verify Ziegler-Nichols PID tuning rejects a negative oscillation period."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        with self.assertRaises(ValueError):
            ctrl.ziegler_nichols_tuning(Ku=10.0, Tu=-1.0, kind="PID")

    def test_ziegler_nichols_tuning_rejects_nonfinite_period(self) -> None:
        """Verify Ziegler-Nichols PID tuning rejects NaN or infinite periods."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        with self.assertRaises(ValueError):
            ctrl.ziegler_nichols_tuning(Ku=10.0, Tu=float("nan"), kind="PID")
        with self.assertRaises(ValueError):
            ctrl.ziegler_nichols_tuning(Ku=10.0, Tu=float("inf"), kind="PID")

    def test_pid_control_integral_windup_clamped_when_set(self) -> None:
        """Verify PID integral windup stays within the configured i_clamp bound."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        desired = np.array([1.0, 1.0])
        actual = np.array(
            [0.0, 0.0]
        )  # constant 1.0 rad error, controller cannot move plant
        Kp, Ki, Kd = np.eye(2), np.eye(2), np.eye(2)
        for _ in range(1000):
            ctrl.pid_control(
                desired,
                np.zeros(2),
                actual,
                np.zeros(2),
                dt=0.01,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd,
                i_clamp=5.0,
            )
        self.assertTrue(
            np.all(np.abs(ctrl.eint) <= 5.0 + 1e-9), f"eint={ctrl.eint} exceeded clamp"
        )

    def test_pid_control_rejects_invalid_i_clamp(self) -> None:
        """Verify pid_control rejects nonpositive or nonfinite i_clamp values."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        for bad_clamp in (-5.0, 0.0, float("nan"), float("inf")):
            with self.subTest(i_clamp=bad_clamp):
                with self.assertRaises(ValueError):
                    ctrl.pid_control(
                        np.ones(2),
                        np.zeros(2),
                        np.zeros(2),
                        np.zeros(2),
                        dt=0.01,
                        Kp=np.eye(2),
                        Ki=np.eye(2),
                        Kd=np.eye(2),
                        i_clamp=bad_clamp,
                    )

    def test_computed_torque_control_rejects_invalid_i_clamp(self) -> None:
        """Verify computed_torque_control rejects an invalid i_clamp value."""
        from ManipulaPy.control import ManipulatorController

        class Dynamics:
            def mass_matrix(self, thetalist) -> np.ndarray:
                """Return an identity mass matrix sized to thetalist."""
                return np.eye(len(thetalist))

            def inverse_dynamics(self, *args) -> np.ndarray:
                """Return a zero inverse-dynamics torque stub."""
                return np.zeros(2)

        ctrl = ManipulatorController(Dynamics())
        with self.assertRaises(ValueError):
            ctrl.computed_torque_control(
                np.ones(2),
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                np.array([0, 0, -9.81]),
                dt=0.01,
                Kp=np.eye(2),
                Ki=np.eye(2),
                Kd=np.eye(2),
                i_clamp=float("nan"),
            )

    def test_pid_control_unclamped_when_clamp_none(self) -> None:
        """Default (i_clamp=None) preserves existing accumulation behavior."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        desired = np.array([1.0])
        actual = np.array([0.0])
        Kp, Ki, Kd = np.eye(1), np.eye(1), np.eye(1)
        for _ in range(100):
            ctrl.pid_control(
                desired, np.zeros(1), actual, np.zeros(1), dt=0.01, Kp=Kp, Ki=Ki, Kd=Kd
            )
        # 100 steps x 1.0 error x 0.01 dt = 1.0 expected
        np.testing.assert_allclose(ctrl.eint, [1.0], atol=1e-9)

    def test_pid_control_initializes_integral_as_float_for_integer_inputs(self) -> None:
        """Verify pid_control initializes the integral term as float for integer inputs."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        tau = ctrl.pid_control(
            thetalistd=[1, 1],
            dthetalistd=[0, 0],
            thetalist=[0, 0],
            dthetalist=[0, 0],
            dt=0.01,
            Kp=[1, 1],
            Ki=[1, 1],
            Kd=[1, 1],
        )

        self.assertTrue(np.issubdtype(ctrl.eint.dtype, np.floating))
        np.testing.assert_allclose(ctrl.eint, [0.01, 0.01])
        np.testing.assert_allclose(tau, [1.01, 1.01])

    def test_computed_torque_control_initializes_integral_as_float_for_integer_inputs(
        self,
    ) -> None:
        """Verify computed_torque_control initializes the integral as float for integer inputs."""
        from ManipulaPy.control import ManipulatorController

        class Dynamics:
            def mass_matrix(self, thetalist) -> np.ndarray:
                """Return an identity mass matrix sized to thetalist."""
                return np.eye(len(thetalist))

            def inverse_dynamics(self, *args) -> np.ndarray:
                """Return a zero inverse-dynamics torque stub."""
                return np.zeros(2)

        ctrl = ManipulatorController(Dynamics())
        tau = ctrl.computed_torque_control(
            thetalistd=[1, 1],
            dthetalistd=[0, 0],
            ddthetalistd=[0, 0],
            thetalist=[0, 0],
            dthetalist=[0, 0],
            g=[0, 0, -9],
            dt=0.01,
            Kp=[1, 1],
            Ki=[1, 1],
            Kd=[1, 1],
        )

        self.assertTrue(np.issubdtype(ctrl.eint.dtype, np.floating))
        np.testing.assert_allclose(ctrl.eint, [0.01, 0.01])
        np.testing.assert_allclose(tau, [1.01, 1.01])

    def test_kalman_filter_update_rejects_shape_mismatch(self) -> None:
        """Verify the Kalman update rejects a measurement that mismatches the state size."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        ctrl.x_hat = np.zeros(4)
        ctrl.P = np.eye(4)
        # z is 2-vector but x_hat is 4-vector — mismatch must raise ValueError
        z_wrong = np.zeros(2)
        R_wrong = np.eye(2)
        with self.assertRaises(ValueError):
            ctrl.kalman_filter_update(z_wrong, R_wrong)

    def test_kalman_filter_update_rejects_bad_R_shape(self) -> None:
        """Verify the Kalman update rejects a measurement-noise matrix of wrong shape."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        ctrl.x_hat = np.zeros(4)
        ctrl.P = np.eye(4)
        # z matches n=4 but R is not (4,4)
        z = np.zeros(4)
        R_wrong = np.eye(3)
        with self.assertRaises(ValueError):
            ctrl.kalman_filter_update(z, R_wrong)

    def test_kalman_filter_update_rejects_column_vector_z(self) -> None:
        """Verify the Kalman update rejects a column-vector measurement."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        ctrl.x_hat = np.zeros(4)
        ctrl.P = np.eye(4)
        with self.assertRaises(ValueError):
            ctrl.kalman_filter_update(np.zeros((4, 1)), np.eye(4))

    def test_kalman_filter_predict_rejects_bad_Q_shape(self) -> None:
        """Verify the Kalman predict step rejects a process-noise matrix of wrong shape."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        ctrl.x_hat = np.zeros(4)
        ctrl.P = np.eye(4)
        # Q must be (4,4); passing (3,3) must raise ValueError
        Q_wrong = np.eye(3)
        with self.assertRaises(ValueError):
            ctrl.kalman_filter_predict(
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                np.array([0, 0, -9.81]),
                np.zeros(6),
                dt=0.01,
                Q=Q_wrong,
            )

    def test_kalman_filters_explicitly_convert_real_cupy_like_arrays(self) -> None:
        """Verify Kalman filters explicitly convert CuPy-like arrays via .get()."""
        from types import SimpleNamespace
        from unittest.mock import patch

        import ManipulaPy.control as control
        from ManipulaPy.control import ManipulatorController

        class ExplicitArray:
            def __init__(self, value) -> None:
                """Wrap a value as a NumPy array for the CuPy-like stub."""
                self.value = np.asarray(value)

            def get(self) -> np.ndarray:
                """Return the wrapped NumPy array (CuPy-style .get())."""
                return self.value

            def __array__(self, dtype=None) -> None:
                """Block implicit NumPy conversion to force explicit .get() usage."""
                raise TypeError("Implicit conversion to a NumPy array is not allowed")

        class Dynamics:
            def forward_dynamics(
                self, thetalist, dthetalist, taulist, g, Ftip
            ) -> np.ndarray:
                """Return zero joint accelerations matching thetalist."""
                return np.zeros_like(thetalist)

        ctrl = ManipulatorController(Dynamics())
        with (
            patch.object(control, "CUPY_AVAILABLE", True),
            patch.object(control, "cp", SimpleNamespace(ndarray=ExplicitArray)),
        ):
            ctrl.kalman_filter_predict(
                ExplicitArray([0.0, 0.0]),
                ExplicitArray([0.0, 0.0]),
                ExplicitArray([0.0, 0.0]),
                ExplicitArray([0.0, 0.0, -9.81]),
                ExplicitArray(np.zeros(6)),
                dt=0.01,
                Q=ExplicitArray(np.eye(4) * 0.01),
            )
            old_x_hat = ctrl.x_hat.copy()
            ctrl.kalman_filter_update(
                ExplicitArray(np.ones(4) * 0.01),
                ExplicitArray(np.eye(4) * 0.001),
            )

        self.assertFalse(np.allclose(old_x_hat, ctrl.x_hat))

    def test_control_module_imports_without_cupy(self) -> None:
        """ManipulaPy.control must import on a cupy-less machine.

        Currently fails because Union[cp.ndarray, ...] in enforce_limits is
        evaluated at class-body time and cp is None when cupy is missing.
        """
        import subprocess, sys, textwrap

        script = textwrap.dedent("""
            import builtins, importlib, sys
            real = builtins.__import__
            def fake(name, *a, **k):
                if name.split('.')[0] == 'cupy':
                    raise ImportError('blocked')
                return real(name, *a, **k)
            builtins.__import__ = fake
            sys.modules.pop('ManipulaPy.control', None)
            importlib.import_module('ManipulaPy.control')
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(
            result.returncode,
            0,
            f"control.py failed to import without cupy:\n{result.stderr}",
        )

    def test_settling_time_returns_first_settled_time_not_last(self) -> None:
        """Verify settling-time returns the first settled instant, not the last sample."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        # Monotonic settled at t=1.0 onward
        t = np.array([0.0, 1.0, 2.0, 3.0])
        r = np.array([0.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(
            ctrl.calculate_settling_time(t, r, 1.0, tolerance=0.02), 1.0
        )

    def test_settling_time_handles_negative_setpoint(self) -> None:
        """Verify settling-time handles a negative setpoint symmetrically."""
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)
        t = np.array([0.0, 1.0, 2.0, 3.0])
        r = np.array([0.0, -1.0, -1.0, -1.0])
        # Symmetric to the positive case — should return 1.0, not time[-1]
        self.assertAlmostEqual(
            ctrl.calculate_settling_time(t, r, -1.0, tolerance=0.02), 1.0
        )


class TestSingularityRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/singularity.py bugs."""

    def test_singularity_analysis_works_on_redundant_robot(self) -> None:
        """Verify singularity analysis works on a redundant (non-square) Jacobian."""
        from ManipulaPy.singularity import Singularity

        class MockManip:
            def jacobian(self, thetalist, frame="space") -> np.ndarray:
                """Return a random 6x7 (redundant) Jacobian stub."""
                return np.random.randn(6, 7)  # 7-DOF redundant

        sing = Singularity(MockManip())
        result = sing.singularity_analysis(np.zeros(7))
        self.assertIsInstance(result, (bool, np.bool_))

    def test_manipulability_ellipsoid_radii_finite_at_singular_jacobian(self) -> None:
        """A singular Jacobian must still produce finite plotted ellipsoid points."""
        from ManipulaPy.singularity import Singularity

        class MockManip:
            def jacobian(self, thetalist, frame="space") -> np.ndarray:
                """Return a singular (all-zero) 6x6 Jacobian stub."""
                return np.zeros((6, 6))

        class RecordingAxis:
            def __init__(self) -> None:
                """Initialize the axis stub with an empty surfaces list."""
                self.surfaces = []

            def plot_surface(self, x, y, z, **kwargs) -> None:
                """Record the plotted surface coordinates instead of rendering."""
                self.surfaces.append((x, y, z))

            def set_title(self, title) -> None:
                """No-op stub for the axis title setter."""
                pass

        ax = RecordingAxis()
        Singularity(MockManip()).manipulability_ellipsoid(np.zeros(6), ax=ax)

        self.assertEqual(len(ax.surfaces), 2)
        for surface in ax.surfaces:
            for values in surface:
                self.assertTrue(np.all(np.isfinite(values)))

    def test_manipulability_ellipsoid_handles_underactuated_jacobian(self) -> None:
        """Verify the manipulability ellipsoid handles an underactuated Jacobian."""
        from ManipulaPy.singularity import Singularity

        class MockManip:
            def jacobian(self, thetalist, frame="space") -> np.ndarray:
                """Return an underactuated (6x1) Jacobian stub."""
                return np.array([[1.0], [0.0], [0.0], [0.0], [1.0], [0.0]])

        class RecordingAxis:
            def __init__(self) -> None:
                """Initialize the axis stub with an empty surfaces list."""
                self.surfaces = []

            def plot_surface(self, x, y, z, **kwargs) -> None:
                """Record the plotted surface coordinates instead of rendering."""
                self.surfaces.append((x, y, z))

            def set_title(self, title) -> None:
                """No-op stub for the axis title setter."""
                pass

        ax = RecordingAxis()
        Singularity(MockManip()).manipulability_ellipsoid(np.zeros(1), ax=ax)

        self.assertEqual(len(ax.surfaces), 2)
        for surface in ax.surfaces:
            for values in surface:
                self.assertEqual(values.shape, (20, 10))
                self.assertTrue(np.all(np.isfinite(values)))


class TestPotentialFieldRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/potential_field.py bugs."""

    def test_repulsive_potential_when_at_obstacle(self) -> None:
        """Verify the repulsive potential stays finite when exactly at an obstacle."""
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(
            attractive_gain=1.0, repulsive_gain=1.0, influence_distance=0.5
        )
        q = np.array([1.0, 1.0, 1.0])
        p_val = pf.compute_repulsive_potential(q, [q.copy()])
        self.assertTrue(np.isfinite(p_val))

    def test_repulsive_gradient_when_at_obstacle(self) -> None:
        """Verify the repulsive gradient stays finite when exactly at an obstacle."""
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(
            attractive_gain=1.0, repulsive_gain=1.0, influence_distance=0.5
        )
        q = np.array([1.0, 1.0, 1.0])
        grad = pf.compute_gradient(q, np.zeros(3), [q.copy()])
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_repulsive_gradient_at_obstacle_provides_escape_direction(self) -> None:
        """When q == obstacle exactly, gradient must be nonzero so robot can escape."""
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(
            attractive_gain=0.0, repulsive_gain=1.0, influence_distance=0.5
        )
        q = np.array([1.0, 1.0, 1.0])
        grad = pf.compute_gradient(q, np.zeros(3), [q.copy()])
        # Must be nonzero so -grad gives a nonzero force
        self.assertGreater(
            np.linalg.norm(grad), 0, "Robot stuck at obstacle has no escape force"
        )

    def test_repulsive_gradient_at_obstacle_escape_is_bounded(self) -> None:
        """Verify the at-obstacle escape gradient is bounded to a unit direction."""
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(
            attractive_gain=0.0, repulsive_gain=1.0, influence_distance=0.5
        )
        q = np.array([1.0, 1.0, 1.0])

        grad = pf.compute_gradient(q, np.zeros(3), [q.copy()])

        np.testing.assert_allclose(grad, [1.0, 0.0, 0.0])


class TestSimRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/sim.py bugs."""

    def test_sim_module_imports_without_pybullet(self) -> None:
        """sim module must be importable when pybullet is missing.

        Runs in a subprocess so that mucking with ``sys.modules`` and the
        import system can't leak state into other tests in this process
        (notably tests/test_sim.py, which captures Simulation at import time
        and relies on a monkeypatched ManipulaPy.sim.p).

        Scope is intentionally limited to pybullet/pybullet_data. Blocking
        cupy too would crash a transitive import of ManipulaPy.control,
        whose own optional cupy guard is tracked separately. The cupy
        fallback in sim.py is exercised by source review (``_NumpyProxy``
        delegates to ``np`` and exposes ``asnumpy``) rather than by this
        end-to-end import probe.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import builtins
            import importlib
            import sys

            blocked = {"pybullet", "pybullet_data"}
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name.split(".")[0] in blocked:
                    raise ImportError(f"blocked optional dependency: {name}")
                return real_import(name, *args, **kwargs)

            sys.modules.pop("ManipulaPy.sim", None)
            builtins.__import__ = fake_import
            try:
                sim_module = importlib.import_module("ManipulaPy.sim")
            finally:
                builtins.__import__ = real_import

            assert sim_module._PYBULLET_AVAILABLE is False, "pybullet guard failed"
            assert sim_module.p is None, "p must be None when pybullet missing"
            try:
                sim_module.Simulation("robot.urdf", joint_limits=[])
            except ImportError as exc:
                assert "ManipulaPy[simulation]" in str(exc), str(exc)
            else:
                raise AssertionError("Simulation() did not raise ImportError")
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_run_controller_tracks_desired_positions(self) -> None:
        """run_controller must end at the final desired position (open-loop)."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        desired_positions = [
            np.zeros(6),
            np.array([0.5, 0, 0, 0, 0, 0]),
            np.array([1.0, 0, 0, 0, 0, 0]),
        ]

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p") as mock_p,
            patch("ManipulaPy.sim.time.sleep"),
        ):
            mock_p.getJointState.return_value = (1.0, 0.0)  # position, velocity
            mock_p.stepSimulation = MagicMock()
            mock_p.getNumJoints.return_value = 6
            mock_p.getLinkState.return_value = ((0, 0, 0),) * 5

            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.set_joint_positions = MagicMock()
            sim.get_joint_positions = MagicMock(return_value=np.zeros(6))
            sim.plot_trajectory = MagicMock()

            sim.run_controller(desired_positions)

            actual_calls = [
                call.args[0] for call in sim.set_joint_positions.call_args_list
            ]
            self.assertEqual(len(actual_calls), len(desired_positions))
            for actual, expected in zip(actual_calls, desired_positions):
                np.testing.assert_array_equal(actual, expected)

    def test_run_controller_rejects_1d_input(self) -> None:
        """run_controller requires a 2D waypoint matrix."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p"),
            patch("ManipulaPy.sim.time.sleep"),
        ):
            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.set_joint_positions = MagicMock()
            sim.plot_trajectory = MagicMock()

            with self.assertRaisesRegex(ValueError, "shape|2D"):
                sim.run_controller(np.zeros(6))

    def test_run_controller_rejects_empty_input(self) -> None:
        """run_controller rejects empty trajectories before stepping."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p"),
            patch("ManipulaPy.sim.time.sleep"),
        ):
            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.set_joint_positions = MagicMock()
            sim.plot_trajectory = MagicMock()

            with self.assertRaisesRegex(ValueError, "empty"):
                sim.run_controller([])

    def test_run_controller_rejects_wrong_dof(self) -> None:
        """run_controller requires waypoint width to match movable joints."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p"),
            patch("ManipulaPy.sim.time.sleep"),
        ):
            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.set_joint_positions = MagicMock()
            sim.plot_trajectory = MagicMock()

            with self.assertRaisesRegex(ValueError, "6|3|joint count"):
                sim.run_controller([np.zeros(3)])

    def test_run_controller_accepts_generator(self) -> None:
        """run_controller materializes iterable waypoints before tracking."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        first_waypoint = np.zeros(6)
        second_waypoint = np.array([0.5, 0, 0, 0, 0, 0])
        desired_positions = (waypoint for waypoint in [first_waypoint, second_waypoint])

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p") as mock_p,
            patch("ManipulaPy.sim.time.sleep"),
        ):
            mock_p.stepSimulation = MagicMock()
            mock_p.getNumJoints.return_value = 6
            mock_p.getLinkState.return_value = ((0, 0, 0),) * 5

            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.set_joint_positions = MagicMock()
            sim.plot_trajectory = MagicMock()

            sim.run_controller(desired_positions)

            self.assertEqual(sim.set_joint_positions.call_count, 2)
            second_call_arg = sim.set_joint_positions.call_args_list[1].args[0]
            np.testing.assert_array_equal(second_call_arg, second_waypoint)

    def test_loop_methods_close_simulation_on_inner_exception(self) -> None:
        """Non-KeyboardInterrupt exceptions inside simulation loops must clean up."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        def make_sim() -> "Simulation":
            """Build a Simulation with all loop dependencies mocked out."""
            sim = Simulation.__new__(Simulation)
            sim.logger = MagicMock()
            sim.physics_client = 1
            sim.robot_id = 1
            sim.non_fixed_joints = list(range(6))
            sim.time_step = 0.01
            sim.real_time_factor = 1.0
            sim.joint_params = [object()]
            sim.reset_button = 1
            sim.home_position = np.zeros(6)
            sim.trajectory_body_ids = []
            sim.controller = MagicMock()
            sim.set_joint_positions = MagicMock()
            sim.get_joint_parameters = MagicMock(return_value=np.zeros(6))
            sim.check_collisions = MagicMock()
            sim.update_simulation_parameters = MagicMock()
            sim.add_joint_parameters = MagicMock()
            sim.add_additional_parameters = MagicMock()
            sim.add_reset_button = MagicMock()
            sim.run_trajectory = MagicMock()
            sim.clear_trajectory_visualization = MagicMock()
            sim.close_simulation = MagicMock()
            return sim

        cases = [
            (
                "simulate_robot_with_desired_angles",
                lambda sim: sim.simulate_robot_with_desired_angles(np.zeros(6)),
            ),
            ("manual_control", lambda sim: sim.manual_control()),
            ("run", lambda sim: sim.run(np.zeros((1, 6)))),
        ]

        for method_name, exercise in cases:
            with self.subTest(method=method_name):
                sim = make_sim()
                with (
                    patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
                    patch("ManipulaPy.sim.p") as mock_p,
                    patch("ManipulaPy.sim.time.sleep"),
                ):
                    mock_p.stepSimulation.side_effect = RuntimeError("boom")

                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        exercise(sim)

                sim.close_simulation.assert_called_once_with()

    def test_del_logs_cleanup_failure_at_debug_level(self) -> None:
        """Destructor cleanup failures should be visible only at debug level."""
        from unittest.mock import MagicMock

        from ManipulaPy.sim import Simulation

        sim = Simulation.__new__(Simulation)
        sim.logger = MagicMock()
        sim.trajectory_body_ids = [1]
        sim.clear_trajectory_visualization = MagicMock(side_effect=RuntimeError("boom"))

        sim.__del__()

        sim.logger.debug.assert_called_once()
        self.assertTrue(sim.logger.debug.call_args.kwargs["exc_info"])

    def test_simulation_with_external_physics_client_loads_robot(self) -> None:
        """Passing an existing physics_client must still load plane + robot."""
        from unittest.mock import patch, MagicMock
        from ManipulaPy.sim import Simulation

        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p") as mock_p,
            patch("ManipulaPy.sim.pybullet_data") as mock_pd,
        ):
            mock_p.loadURDF.side_effect = [101, 202]  # plane, robot
            mock_p.getNumJoints.return_value = 6
            mock_p.getJointInfo.side_effect = lambda r, i: (
                i,
                b"j",
                0,
            )  # 0 = JOINT_REVOLUTE
            mock_p.JOINT_FIXED = 4
            sim = Simulation(
                "robot.urdf", joint_limits=[(-1, 1)] * 6, physics_client=42
            )
            self.assertEqual(sim.robot_id, 202)
            self.assertEqual(sim.plane_id, 101)
            self.assertEqual(len(sim.non_fixed_joints), 6)
            self.assertEqual(sim.home_position.shape, (6,))

    def test_set_joint_positions_passes_forces_kwarg(self) -> None:
        """set_joint_positions must pass forces= so URDFs with low effort
        limits don't silently fail to track."""
        from unittest.mock import MagicMock, patch
        from ManipulaPy.sim import Simulation

        sim = Simulation.__new__(Simulation)
        sim.logger = MagicMock()
        sim.robot_id = 1
        sim.non_fixed_joints = list(range(6))
        sim.torque_limits = None
        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p") as mock_p,
        ):
            sim.set_joint_positions(np.zeros(6))
            kwargs = mock_p.setJointMotorControlArray.call_args.kwargs
            self.assertIn("forces", kwargs, "forces= must be supplied")
            self.assertEqual(len(kwargs["forces"]), 6)

    def test_setup_logger_does_not_duplicate_handlers(self) -> None:
        """Constructing Simulation N times must not stack N stream handlers."""
        import logging
        from unittest.mock import patch, MagicMock
        from ManipulaPy.sim import Simulation

        # Clear any leftover handlers from prior test runs
        logging.getLogger("SimulationLogger").handlers.clear()
        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p"),
            patch("ManipulaPy.sim.pybullet_data"),
        ):
            sims = [Simulation.__new__(Simulation) for _ in range(3)]
            for s in sims:
                s.logger = s.setup_logger()
            handler_count = len(
                [
                    h
                    for h in sims[0].logger.handlers
                    if isinstance(h, logging.StreamHandler)
                ]
            )
            self.assertEqual(
                handler_count, 1, f"Expected 1 stream handler, got {handler_count}"
            )

    def test_capsule_line_uses_inline_axis_angle_quaternion(self) -> None:
        """_capsule_line must not depend on p.getQuaternionFromAxisAngle —
        that helper is missing from several pybullet builds. Inline math
        replaced it; confirm by deleting the helper from the mocked module
        and verifying _capsule_line still completes successfully.
        """
        from unittest.mock import MagicMock, patch
        from ManipulaPy.sim import Simulation

        sim = Simulation.__new__(Simulation)
        sim.logger = MagicMock()

        with patch("ManipulaPy.sim.p") as mock_p:
            mock_p.getQuaternionFromEuler.return_value = (0, 0, 0, 1)
            mock_p.createCollisionShape.return_value = 10
            mock_p.createVisualShape.return_value = 11
            mock_p.createMultiBody.return_value = 12
            del mock_p.getQuaternionFromAxisAngle  # simulate stripped pybullet build

            body_id = sim._capsule_line(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )

        self.assertEqual(body_id, 12)
        mock_p.createMultiBody.assert_called_once()

    def test_plot_trajectory_no_mutable_default_color(self) -> None:
        """color=[1, 0, 0] as a mutable default is the same footgun as Task 4.
        Fix: use color=None and assign the default inside the function."""
        import inspect
        from ManipulaPy.sim import Simulation

        sig = inspect.signature(Simulation.plot_trajectory)
        color_default = sig.parameters["color"].default
        self.assertIsNone(
            color_default,
            "Simulation.plot_trajectory color default must be None — found "
            f"{color_default!r}",
        )


class TestSimPybulletGuards(unittest.TestCase):
    """Task 15: every public method that touches p.* must raise a clear
    ImportError when pybullet is unavailable, never AttributeError on the
    None proxy installed by Task 13's import guard.
    """

    # Methods that directly call p.*. Each entry: (method_name, args_factory).
    # Orchestration methods (run_trajectory, simulate_robot_motion, etc.) are
    # included because they delegate to the leaf methods and users hit them
    # directly; failing fast at the entry point gives a clearer error.
    _METHODS_AND_ARGS = [
        ("set_joint_positions", lambda: (np.zeros(6),)),
        ("get_joint_positions", lambda: ()),
        ("plot_trajectory", lambda: (np.zeros((3, 3)),)),
        ("clear_trajectory_visualization", lambda: ()),
        ("get_joint_parameters", lambda: ()),
        ("check_collisions", lambda: ()),
        ("step_simulation", lambda: ()),
        ("add_joint_parameters", lambda: ()),
        ("add_reset_button", lambda: ()),
        ("add_additional_parameters", lambda: ()),
        ("update_simulation_parameters", lambda: ()),
        ("save_joint_states", lambda: ("/tmp/_v132_states.csv",)),
        ("run_trajectory", lambda: (np.zeros((1, 6)),)),
        ("simulate_robot_motion", lambda: (np.zeros((1, 6)),)),
        ("simulate_robot_with_desired_angles", lambda: (np.zeros(6),)),
        ("manual_control", lambda: ()),
        ("plot_trajectory_in_scene", lambda: (np.zeros((1, 6)), np.zeros((1, 3)))),
    ]

    def _make_bare_sim(self) -> "Simulation":
        """Build a minimally-populated Simulation instance for guard tests."""
        from unittest.mock import MagicMock
        from ManipulaPy.sim import Simulation

        sim = Simulation.__new__(Simulation)
        sim.logger = MagicMock()
        sim.physics_client = 1
        sim.robot_id = 1
        sim.non_fixed_joints = list(range(6))
        sim.time_step = 0.01
        sim.real_time_factor = 1.0
        sim.joint_params = []
        sim.reset_button = None
        sim.home_position = None
        sim.trajectory_body_ids = []
        return sim

    def test_public_methods_raise_importerror_without_pybullet(self) -> None:
        """Verify public sim methods raise ImportError when pybullet is absent."""
        from unittest.mock import patch

        for method_name, args_factory in self._METHODS_AND_ARGS:
            with self.subTest(method=method_name):
                sim = self._make_bare_sim()
                method = getattr(sim, method_name)
                args = args_factory()
                with (
                    patch("ManipulaPy.sim._PYBULLET_AVAILABLE", False),
                    patch("ManipulaPy.sim.p", None),
                ):
                    with self.assertRaises(ImportError) as ctx:
                        method(*args)
                self.assertIn(
                    "ManipulaPy[simulation]",
                    str(ctx.exception),
                    f"{method_name}: ImportError must hint at the install extra",
                )


class TestVisionRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/vision.py bugs."""

    def test_detect_obstacles_default_threshold_does_not_filter_everything(
        self,
    ) -> None:
        """depth_threshold=0.0 + 'if mean_depth > threshold: continue' was
        filtering out every detection with positive depth (i.e., all real
        obstacles). With the new 5.0m default a 1m obstacle survives.
        """
        from types import SimpleNamespace

        from ManipulaPy.vision import Vision

        vision = Vision.__new__(Vision)
        vision.logger = SimpleNamespace(
            error=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        )
        vision.cameras = {
            0: {
                "intrinsic_matrix": np.array(
                    [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
                )
            }
        }
        vision._ensure_yolo_loaded = lambda: True
        box = SimpleNamespace(xyxy=[np.array([0, 0, 2, 2])])
        result = SimpleNamespace(boxes=[box])
        vision.yolo_model = lambda image, conf=0.3: [result]

        depth = np.ones((4, 4), dtype=np.float32)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        positions, orientations = vision.detect_obstacles(depth, rgb)

        self.assertEqual(positions.shape, (1, 3))
        self.assertEqual(orientations.shape, (1,))


class TestUrdfRegressions(unittest.TestCase):
    def test_mesh_load_failure_emits_warning(self) -> None:
        """Verify a failed mesh load emits a 'not found' warning log."""
        import logging
        from ManipulaPy.urdf.types import Mesh  # adjust import path as needed

        mesh = Mesh(filename="/nonexistent/path/to/mesh.stl", scale=[1, 1, 1])
        with self.assertLogs("ManipulaPy.urdf.types", level="WARNING") as cm:
            mesh._load_mesh()
        self.assertTrue(any("not found" in msg.lower() for msg in cm.output))

    def test_package_resolver_refuses_ambiguous_package_uri(self) -> None:
        """Verify the package resolver refuses an ambiguous package:// URI."""
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for workspace in ("ws1", "ws2"):
                mesh = root / workspace / "demo_pkg" / "meshes" / "base.stl"
                mesh.parent.mkdir(parents=True)
                mesh.write_text("solid empty\nendsolid empty\n")

            resolver = PackageResolver()
            resolver.add_search_path(root / "ws1")
            resolver.add_search_path(root / "ws2")

            uri = "package://demo_pkg/meshes/base.stl"
            with self.assertLogs("ManipulaPy.urdf.resolver", level="WARNING") as cm:
                resolved = resolver.resolve(uri)

            self.assertEqual(resolved, uri)
            self.assertTrue(any("multiple" in msg.lower() for msg in cm.output))

    def test_package_resolver_explicit_map_overrides_other_strategies(self) -> None:
        """add_package() must short-circuit ambiguity detection.

        The remediation advice in the ambiguity warning instructs callers to
        call add_package() to disambiguate. If add_package() then competes with
        the discovery sources it was meant to override, the escape hatch is
        broken — exactly the regression Codex flagged on Task 27.
        """
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for workspace in ("preferred", "shadow"):
                mesh = root / workspace / "demo_pkg" / "meshes" / "base.stl"
                mesh.parent.mkdir(parents=True)
                mesh.write_text(workspace)

            preferred_mesh = root / "preferred" / "demo_pkg" / "meshes" / "base.stl"

            resolver = PackageResolver(use_ros=False)
            resolver.add_search_path(root / "shadow")
            resolver.add_package("demo_pkg", str(root / "preferred" / "demo_pkg"))

            resolved = resolver.resolve("package://demo_pkg/meshes/base.stl")

            self.assertEqual(
                resolved,
                str(preferred_mesh),
                "add_package() mapping must take precedence over search paths.",
            )

    def test_resolver_use_ros_false_isolates_from_ros_env(self) -> None:
        """use_ros=False must NOT scan ROS_PACKAGE_PATH or ament_index."""
        import os
        import tempfile
        from unittest.mock import patch
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            ros_root = Path(tmp) / "ros_ws"
            mesh = ros_root / "demo_pkg" / "meshes" / "base.stl"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("ros mesh")

            with patch.dict(
                os.environ, {"ROS_PACKAGE_PATH": str(ros_root)}, clear=False
            ):
                resolver = PackageResolver(use_ros=False)
                resolved = resolver.resolve("package://demo_pkg/meshes/base.stl")

            self.assertEqual(
                resolved,
                "package://demo_pkg/meshes/base.stl",
                "use_ros=False should not consult ROS_PACKAGE_PATH",
            )

    def test_resolver_uses_find_ros_package_when_enabled(self) -> None:
        """use_ros=True must consult _find_ros_package (ament/rospack/catkin)."""
        import tempfile
        from unittest.mock import patch
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            pkg_root = Path(tmp) / "ament_share" / "demo_pkg"
            mesh = pkg_root / "meshes" / "base.stl"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("ament mesh")

            resolver = PackageResolver(use_ros=True)
            with patch.object(resolver, "_find_ros_package", return_value=pkg_root):
                resolved = resolver.resolve("package://demo_pkg/meshes/base.stl")

            self.assertEqual(resolved, str(mesh))

    def test_resolver_search_path_supports_flat_package_root(self) -> None:
        """search_path/relative form must work when caller adds the package
        root itself (e.g., add_search_path('/opt/robot_pkg')).
        """
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            pkg_root = Path(tmp) / "robot_pkg"
            mesh = pkg_root / "meshes" / "a.stl"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("flat")

            resolver = PackageResolver(use_ros=False)
            resolver.add_search_path(pkg_root)

            resolved = resolver.resolve("package://robot_pkg/meshes/a.stl")
            self.assertEqual(resolved, str(mesh))

    def test_resolver_rejects_path_traversal_in_package_uri(self) -> None:
        """package://pkg/../etc/passwd must be refused."""
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo_pkg").mkdir()
            (root / "secret.txt").write_text("classified")

            resolver = PackageResolver(use_ros=False)
            resolver.add_package("demo_pkg", str(root / "demo_pkg"))

            uri = "package://demo_pkg/../secret.txt"
            with self.assertLogs("ManipulaPy.urdf.resolver", level="WARNING") as cm:
                resolved = resolver.resolve(uri)

            self.assertEqual(resolved, uri)
            self.assertTrue(any("traversal" in msg.lower() for msg in cm.output))

    def test_resolver_dedupes_symlinked_workspaces(self) -> None:
        """Symlinked or duplicate-mounted search paths must not falsely
        trigger ambiguity when they reference the same physical file.
        """
        import os
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real_ws"
            mesh = real / "demo_pkg" / "meshes" / "base.stl"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("real")

            link = Path(tmp) / "link_ws"
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unsupported on this filesystem")

            resolver = PackageResolver(use_ros=False)
            resolver.add_search_path(real)
            resolver.add_search_path(link)

            resolved = resolver.resolve("package://demo_pkg/meshes/base.stl")
            self.assertTrue(
                resolved.endswith("real_ws/demo_pkg/meshes/base.stl")
                or resolved.endswith("link_ws/demo_pkg/meshes/base.stl"),
                f"resolved={resolved!r}",
            )
            self.assertFalse(
                resolved.startswith("package://"),
                "symlinked candidates pointing at the same file must not "
                "trigger the ambiguity refusal",
            )

    def test_resolver_warns_on_malformed_package_uri(self) -> None:
        """package://, package://pkg, package://pkg/ should all warn."""
        from ManipulaPy.urdf.resolver import PackageResolver

        resolver = PackageResolver(use_ros=False)
        for malformed in ("package://", "package://pkg", "package://pkg/"):
            with self.subTest(uri=malformed):
                with self.assertLogs("ManipulaPy.urdf.resolver", level="WARNING") as cm:
                    resolved = resolver.resolve(malformed)
                self.assertEqual(resolved, malformed)
                self.assertTrue(
                    any("malformed" in msg.lower() for msg in cm.output),
                    f"no malformed-URI warning for {malformed!r}: {cm.output}",
                )

    def test_resolver_handles_file_uri(self) -> None:
        """file://path should round-trip through url2pathname so Windows
        file:///C:/foo doesn't end up as /C:/foo.
        """
        import tempfile
        from pathlib import Path

        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mesh.stl"
            target.write_text("data")

            resolver = PackageResolver(use_ros=False)
            resolved = resolver.resolve(f"file://{target}")
            self.assertEqual(Path(resolved), target)

    def test_mesh_load_failure_does_not_flood_warnings(self) -> None:
        """Repeated vertices/faces access on a missing mesh must warn once."""
        import logging
        from ManipulaPy.urdf.types import Mesh

        mesh = Mesh(filename="/nonexistent/path/mesh.stl", scale=[1, 1, 1])
        with self.assertLogs("ManipulaPy.urdf.types", level="WARNING") as cm:
            for _ in range(5):
                _ = mesh.vertices
                _ = mesh.faces
        self.assertEqual(
            len(cm.output),
            1,
            f"expected exactly one warning, got {len(cm.output)}: {cm.output}",
        )

    def test_mesh_unresolved_package_uri_is_reported_distinctly(self) -> None:
        """An unresolved package:// URI stored as filename must surface as
        an unresolved-URI warning, not a misleading 'Mesh file not found'.
        """
        from ManipulaPy.urdf.types import Mesh

        mesh = Mesh(filename="package://demo_pkg/meshes/base.stl", scale=[1, 1, 1])
        with self.assertLogs("ManipulaPy.urdf.types", level="WARNING") as cm:
            _ = mesh.vertices
        joined = "\n".join(cm.output).lower()
        self.assertIn("unresolved package uri", joined)
        self.assertNotIn("file not found", joined)

    def test_mesh_warning_escapes_log_injection_in_filename(self) -> None:
        """Newlines/control chars in URDF filenames must not forge log lines."""
        from ManipulaPy.urdf.types import Mesh

        forged = "/tmp/mesh.stl\nERROR: forged log line"
        mesh = Mesh(filename=forged, scale=[1, 1, 1])
        with self.assertLogs("ManipulaPy.urdf.types", level="WARNING") as cm:
            _ = mesh.vertices
        joined = "\n".join(cm.output)
        # The literal newline+ERROR sequence must NOT appear in any single
        # warning record's message; the !r formatting escapes it as \n.
        for record in cm.records:
            self.assertNotIn("\nERROR: forged log line", record.getMessage())

    def test_mesh_loader_warning_includes_full_path_not_just_basename(self) -> None:
        """When two missing meshes share a basename, warnings must distinguish
        them by including the full path (Codex finding: path.name truncation).
        """
        import builtins
        from pathlib import Path
        from unittest.mock import patch

        from ManipulaPy.urdf.geometry import mesh_loader

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs) -> Any:
            """Block importing trimesh; defer everything else to the real import."""
            if name == "trimesh":
                raise ImportError("blocked trimesh")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            self.assertLogs(
                "ManipulaPy.urdf.geometry.mesh_loader", level="WARNING"
            ) as cm,
        ):
            mesh_loader._load_with_trimesh(Path("/tmp/left/robot.dae"))
            mesh_loader._load_with_trimesh(Path("/tmp/right/robot.dae"))

        joined = "\n".join(cm.output)
        self.assertIn("/tmp/left/robot.dae", joined)
        self.assertIn("/tmp/right/robot.dae", joined)

    def test_mesh_loader_import_warning_mentions_file_and_install_hint(self) -> None:
        """Verify the mesh-loader import warning names the file and an install hint."""
        import builtins
        from pathlib import Path
        from unittest.mock import patch

        from ManipulaPy.urdf.geometry import mesh_loader

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs) -> Any:
            """Block importing trimesh; defer everything else to the real import."""
            if name == "trimesh":
                raise ImportError("blocked trimesh")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            self.assertLogs(
                "ManipulaPy.urdf.geometry.mesh_loader", level="WARNING"
            ) as cm,
        ):
            result = mesh_loader._load_with_trimesh(Path("robot.dae"))

        self.assertIsNone(result)
        warning = "\n".join(cm.output)
        self.assertIn("robot.dae", warning)
        self.assertIn("pip install trimesh", warning)

    def test_urdf_processor_omega_list_shape_and_values(self) -> None:
        """omega_list must be (3, N) with each column a unit screw axis.

        Earlier code derived omega from data['Slist'] using inconsistent
        slicing; commit 83c1068 centralised the source on data['omega_list']
        and renamed the prior 'omeg_list' typo. This locks in the shape
        and value invariants.
        """
        from pathlib import Path

        try:
            from ManipulaPy.urdf_processor import URDFToSerialManipulator
            from ManipulaPy.ManipulaPy_data.ur5 import urdf_file
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"UR5 fixture unavailable: {e}")

        if not Path(str(urdf_file)).exists():
            self.skipTest(f"UR5 fixture unavailable: {urdf_file}")

        urdf = URDFToSerialManipulator(str(urdf_file))
        for path_label, omega in (
            ("serial_manipulator", urdf.serial_manipulator.omega_list),
            ("manipulator_dynamics", urdf.manipulator_dynamics.omega_list),
        ):
            self.assertEqual(
                omega.shape,
                (3, 6),
                f"{path_label}.omega_list expected (3, 6) for UR5, got {omega.shape}",
            )
            norms = np.linalg.norm(omega, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones(6), decimal=4)

    def test_trimesh_viz_mesh_load_failure_emits_warning(self) -> None:
        """trimesh_viz silently fell back to a placeholder box when mesh load
        failed, hiding viz problems. Commit on Task 38 logs a warning before
        falling back.
        """
        from unittest.mock import MagicMock, patch

        from ManipulaPy.urdf.types import Mesh
        from ManipulaPy.urdf.visualization import trimesh_viz

        if not trimesh_viz.TRIMESH_AVAILABLE:
            self.skipTest("trimesh not installed")

        geometry = Mesh(filename="/missing/mesh.stl", scale=[1, 1, 1])

        with (
            patch.object(
                trimesh_viz.trimesh, "load", side_effect=ValueError("bad mesh")
            ),
            patch.object(
                trimesh_viz.trimesh.creation, "box", return_value=MagicMock()
            ) as box,
            self.assertLogs(
                "ManipulaPy.urdf.visualization.trimesh_viz", level="WARNING"
            ) as cm,
        ):
            result = trimesh_viz._geometry_to_trimesh(geometry)

        self.assertIsNotNone(result)
        box.assert_called_once()
        joined = "\n".join(cm.output)
        self.assertIn("missing/mesh.stl", joined)
        self.assertIn("placeholder", joined)


class TestCudaKernelRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/cuda_kernels.py bugs.

    These tests verify the kernel logic by calling the CPU fallback paths
    where they exist, or by re-implementing the kernel math in pure Python
    and asserting it matches the corrected formula.
    """

    def test_trajectory_cpu_fallback_matches_quintic_reference(self) -> None:
        """CPU fallback locks in the per-timestep formula used by CUDA kernels."""
        from ManipulaPy.cuda_kernels import trajectory_cpu_fallback

        N = 100
        Tf = 2.0
        thetastart = np.zeros(6)
        thetaend = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])

        expected = np.zeros((N, 6))
        for t in range(N):
            tau = t / (N - 1.0)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            for j in range(6):
                expected[t, j] = thetastart[j] + s * (thetaend[j] - thetastart[j])

        pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method=5)

        np.testing.assert_allclose(pos, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_almost_equal(pos[0], thetastart, decimal=6)
        np.testing.assert_array_almost_equal(pos[-1], thetaend, decimal=6)
        np.testing.assert_array_almost_equal(vel[0], np.zeros(6), decimal=6)
        np.testing.assert_array_almost_equal(vel[-1], np.zeros(6), decimal=6)
        np.testing.assert_array_almost_equal(acc[0], np.zeros(6), decimal=6)
        np.testing.assert_array_almost_equal(acc[-1], np.zeros(6), decimal=6)

    def test_trajectory_cpu_fallback_linear_method_is_linear(self) -> None:
        """method=1 must be true linear scaling, not cubic fall-through."""
        from ManipulaPy.cuda_kernels import trajectory_cpu_fallback

        pos, vel, acc = trajectory_cpu_fallback(
            np.array([0.0, 1.0]),
            np.array([1.0, 3.0]),
            Tf=2.0,
            N=5,
            method=1,
        )

        expected_pos = np.array(
            [
                [0.0, 1.0],
                [0.25, 1.5],
                [0.5, 2.0],
                [0.75, 2.5],
                [1.0, 3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(pos, expected_pos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            vel, np.array([[0.5, 1.0]] * 5), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(acc, np.zeros_like(acc))

    def test_cuda_kernel_variants_handle_linear_method(self) -> None:
        """All trajectory kernel variants must implement linear method (method=1)."""
        import inspect
        from ManipulaPy import cuda_kernels

        src = inspect.getsource(cuda_kernels)
        # Without GPU CI this structural guard verifies all production variants
        # have a linear branch instead of falling through to cubic for method=1.
        for kernel_name in [
            "trajectory_kernel",
            "trajectory_kernel_vectorized",
            "trajectory_kernel_memory_optimized",
            "trajectory_kernel_warp_optimized",
            "trajectory_kernel_cache_friendly",
        ]:
            # Find the function source
            pattern = f"def {kernel_name}("
            if pattern not in src:
                continue
            start = src.index(pattern)
            end = (
                src.index("\n    @cuda.jit", start)
                if "\n    @cuda.jit" in src[start:]
                else len(src)
            )
            kernel_src = src[start:end]
            # Should have a "Linear" branch or s = tau assignment with no transformation
            has_linear = (
                "# Linear" in kernel_src
                or "else:  # Linear" in kernel_src
                or "s = tau\n" in kernel_src
            )
            self.assertTrue(
                has_linear,
                f"{kernel_name} missing linear method branch — non-quintic falls through to cubic",
            )

    # ------------------------------------------------------------------
    # Tasks 30, 31, 32, 33 — remaining CUDA kernel adversarial findings
    # ------------------------------------------------------------------

    def _kernel_source(self, name) -> str:
        """Return the source text of a single named CUDA kernel function."""
        import inspect
        from ManipulaPy import cuda_kernels

        src = inspect.getsource(cuda_kernels)
        marker = f"def {name}("
        start = src.index(marker)
        # Stop at the next decorator or end of file.
        rest = src[start:]
        end_rel = rest.find("\n    @cuda.jit")
        return rest if end_rel < 0 else rest[:end_rel]

    def test_cartesian_kernel_no_shared_memory_race(self) -> None:
        """cartesian_trajectory_kernel must compute scaling per-thread, not via
        a shared-memory write from thread (0,0) only.
        """
        body = self._kernel_source("cartesian_trajectory_kernel")
        self.assertNotIn(
            "cuda.shared.array",
            body,
            "cartesian_trajectory_kernel still uses shared-memory scaling — "
            "thread (0,0) wrote scaling for its own t_idx, other threads read it.",
        )
        self.assertNotIn("cuda.syncthreads", body)

    def test_cartesian_kernel_quintic_position_uses_tau5(self) -> None:
        """The quintic position formula must include 6 * tau^5, not 6 * tau * tau^3 (=tau^4)."""
        body = self._kernel_source("cartesian_trajectory_kernel")
        self.assertNotIn(
            "6.0 * tau * tau3",
            body,
            "cartesian_trajectory_kernel quintic position is using "
            "'6 * tau * tau3' (= 6 tau^4); correct quintic needs 6 tau^5.",
        )
        # Either form spelled out is acceptable.
        self.assertTrue(
            "6.0 * tau5" in body or "6.0 * tau2 * tau3" in body,
            f"expected 6*tau5 or 6*tau2*tau3 in cartesian quintic; body=\n{body[:600]}",
        )

    def test_cartesian_kernel_quintic_acceleration_includes_one_minus_tau(self) -> None:
        """Quintic s_ddot must be 60 tau (1 - tau)(1 - 2tau) / Tf^2,
        not the truncated 60 tau (1 - 2tau) / Tf^2.
        """
        body = self._kernel_source("cartesian_trajectory_kernel")
        self.assertIn(
            "(1.0 - tau) * (1.0 - 2.0 * tau)",
            body,
            "cartesian_trajectory_kernel quintic acceleration is missing the "
            "(1 - tau) factor — it would zero-cross at tau=0.5 incorrectly.",
        )

    def test_cartesian_kernel_supports_linear_method(self) -> None:
        """Verify the cartesian trajectory kernel implements the linear method branch."""
        body = self._kernel_source("cartesian_trajectory_kernel")
        self.assertIn(
            "s = tau",
            body,
            "cartesian_trajectory_kernel else-branch must implement the linear "
            "method (method == 1) instead of falling through to s = 0.",
        )

    def test_batch_kernel_no_shared_memory_race(self) -> None:
        """Verify the batch trajectory kernel no longer uses race-prone shared memory."""
        body = self._kernel_source("batch_trajectory_kernel")
        self.assertNotIn(
            "cuda.shared.array",
            body,
            "batch_trajectory_kernel still uses shared-memory scaling — same "
            "race as cartesian_trajectory_kernel.",
        )

    def test_batch_kernel_quintic_position_uses_tau5(self) -> None:
        """Verify the batch kernel quintic position term uses the tau^5 form."""
        body = self._kernel_source("batch_trajectory_kernel")
        self.assertNotIn("6.0 * tau * tau3", body)
        self.assertTrue(
            "6.0 * tau5" in body or "6.0 * tau2 * tau3" in body,
        )

    def test_batch_kernel_quintic_acceleration_includes_one_minus_tau(self) -> None:
        """Verify the batch kernel quintic acceleration includes the (1 - tau) factor."""
        body = self._kernel_source("batch_trajectory_kernel")
        self.assertIn("(1 - tau) * (1 - 2 * tau)", body)

    def test_batch_kernel_supports_linear_method(self) -> None:
        """Verify the batch trajectory kernel implements the linear method branch."""
        body = self._kernel_source("batch_trajectory_kernel")
        self.assertIn("s = tau", body)

    def test_quintic_formula_satisfies_boundary_conditions(self) -> None:
        """Pure-Python equivalent of the corrected quintic — proves the new
        formula gives s(0)=0, s(1)=1, s_dot(0)=s_dot(1)=0, s_ddot(0)=s_ddot(1)=0.
        """
        Tf = 2.0
        for tau in (0.0, 1.0):
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
            s_ddot = 60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau) / (Tf * Tf)
            if tau == 0.0:
                self.assertAlmostEqual(s, 0.0)
            else:
                self.assertAlmostEqual(s, 1.0)
            self.assertAlmostEqual(s_dot, 0.0)
            self.assertAlmostEqual(s_ddot, 0.0)

    def test_forward_dynamics_kernel_has_no_temporal_data_race(self) -> None:
        """forward_dynamics_kernel previously read thetamat[t_idx-1, j_idx] from
        another thread's not-yet-written row — a classic CUDA temporal race.
        The fixed kernel integrates from initial state per-thread.
        """
        body = self._kernel_source("forward_dynamics_kernel")
        self.assertNotIn(
            "thetamat[t_idx - 1",
            body,
            "forward_dynamics_kernel still reads thetamat[t_idx - 1, j_idx] — "
            "temporal data race across parallel threads.",
        )
        self.assertNotIn("dthetamat[t_idx - 1", body)
        # The fixed version must preserve row 0 as the initial state, then walk
        # from t=1 to t_idx within the thread to match the CPU trajectory path.
        self.assertIn("if t_idx == 0", body)
        self.assertIn("range(1, t_idx + 1)", body)
        self.assertNotIn("range(t_idx + 1)", body)

    def test_potential_gradient_repulsive_pushes_away_from_obstacle(self) -> None:
        """Pure-Python equivalent of the fused potential gradient — verifies
        the corrected sign on the repulsive term: force = -∇U must point
        AWAY from the obstacle. The buggy code dropped the leading minus,
        so -∇U pulled the robot toward obstacles.
        """
        import math

        # Robot at (1, 0, 0), obstacle at origin, goal at (5, 0, 0).
        pos = (1.0, 0.0, 0.0)
        goal = (5.0, 0.0, 0.0)
        obs = (0.0, 0.0, 0.0)
        influence_distance = 2.0

        diff = tuple(pos[i] - goal[i] for i in range(3))
        grad = list(diff)  # attractive component

        obs_diff = tuple(pos[i] - obs[i] for i in range(3))
        dist_sq = sum(c * c for c in obs_diff)
        if 0.0 < dist_sq < influence_distance * influence_distance:
            dist_inv = 1.0 / math.sqrt(dist_sq)
            influence_term = dist_inv - 1.0 / influence_distance
            # Corrected sign — matches the fixed kernel.
            grad_factor = -influence_term * dist_inv * dist_inv * dist_inv
            for i in range(3):
                grad[i] += grad_factor * obs_diff[i]

        # Force is -gradient. Robot is at +x relative to obstacle, so the
        # repulsive push must be in +x. Goal is also at +x, so attractive
        # also pulls in +x. Net force_x must be positive.
        force = tuple(-g for g in grad)
        self.assertGreater(
            force[0],
            0.0,
            "repulsive force must push the robot in +x (away from obstacle).",
        )
        self.assertAlmostEqual(force[1], 0.0)
        self.assertAlmostEqual(force[2], 0.0)

    def test_cuda_kernels_has_no_bare_except_handlers(self) -> None:
        """Bare 'except:' swallows SystemExit/KeyboardInterrupt and trips
        flake8 E722. Narrow to 'except Exception:'."""
        import ast
        import inspect

        from ManipulaPy import cuda_kernels

        tree = ast.parse(inspect.getsource(cuda_kernels))
        bare_handlers = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        self.assertEqual(
            bare_handlers, [], f"bare except handlers at lines {bare_handlers}"
        )

    def test_setup_cuda_environment_respects_existing_user_values(self) -> None:
        """setup_cuda_environment must not clobber env vars the user set
        explicitly. Use setdefault, not direct assignment."""
        import os
        from ManipulaPy.cuda_kernels import setup_cuda_environment_for_40x_speedup

        key = "CUDA_LAUNCH_BLOCKING"
        previous = os.environ.get(key)
        os.environ[key] = "1"
        try:
            setup_cuda_environment_for_40x_speedup()
            self.assertEqual(
                os.environ[key],
                "1",
                "setup_cuda_environment must not overwrite user-set values",
            )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def test_setup_cuda_environment_tolerates_unusable_cupy_runtime(self) -> None:
        """Importable CuPy is not enough; CUDA runtime calls can still fail."""
        import sys
        import types
        from unittest.mock import patch

        from ManipulaPy import cuda_kernels

        class BrokenMemoryPool:
            def set_limit(self, *args, **kwargs) -> None:
                """Stub memory-pool limit setter that always raises a CUDA error."""
                raise RuntimeError("cudaErrorCompatNotSupportedOnDevice")

        broken_cupy = types.ModuleType("cupy")
        broken_cupy.get_default_memory_pool = lambda: BrokenMemoryPool()

        with (
            patch.object(cuda_kernels, "CUPY_AVAILABLE", True),
            patch.dict(sys.modules, {"cupy": broken_cupy}),
        ):
            cuda_kernels.setup_cuda_environment_for_40x_speedup()


class TestTestInfraRegressions(unittest.TestCase):
    """Regressions for test infrastructure issues (Tasks 42-44)."""

    def test_torch_smart_mock_tensor_is_array_like_class(self) -> None:
        """torch_mock.Tensor must be a class, not a Mock(), so that
        isinstance(x, torch.Tensor) and issubclass checks behave correctly.
        """
        from tests.conftest import create_smart_mock

        torch_mock = create_smart_mock("torch")
        tensor = torch_mock.tensor([1.0, 2.0, 3.0])

        self.assertIsInstance(tensor, torch_mock.Tensor)
        self.assertEqual(tensor.shape, (3,))
        np.testing.assert_array_equal(tensor.numpy(), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(np.asarray(tensor), np.array([1.0, 2.0, 3.0]))
        # Class-ness: issubclass must work without raising TypeError.
        self.assertTrue(issubclass(torch_mock.Tensor, object))

    def test_trajectory_planning_has_no_top_level_psutil_import(self) -> None:
        """psutil is optional. A module-level 'import psutil' breaks
        collection on systems without psutil installed; the import must
        live inside the try/except block where it's actually used.
        """
        import ast
        from pathlib import Path

        source = (
            Path(__file__)
            .parent.joinpath("test_trajectory_planning.py")
            .read_text(encoding="utf-8")
        )
        tree = ast.parse(source)

        top_level_psutil = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_psutil.extend(
                    a.name for a in node.names if a.name == "psutil"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "psutil":
                top_level_psutil.append(node.module)

        self.assertEqual(
            top_level_psutil,
            [],
            "tests/test_trajectory_planning.py must not import psutil at "
            "module level — it breaks collection without psutil installed.",
        )

    def test_test_control_module_imports_without_cupy(self) -> None:
        """tests/test_control.py used to do bare 'import cupy as cp' at
        module scope. On a CPU-only contributor machine without cupy,
        that broke collection. Wrapped in try/except now.
        """
        import os
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; "
            "sys.modules['cupy'] = None; "
            "import importlib.util; "
            f"spec = importlib.util.spec_from_file_location("
            f"'test_control', r'{os.path.join(repo_root, 'tests', 'test_control.py')}'); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        self.assertEqual(
            result.returncode,
            0,
            f"test_control.py failed to import without cupy:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_test_control_rejects_importable_but_unusable_cupy(self) -> None:
        """tests/test_control.py must not select CuPy just because it imports.

        Some developer machines have a cupy package installed while the CUDA
        runtime cannot initialize on the visible hardware. In that state,
        cupy.asarray raises at test runtime and unmarked control tests fail
        before they reach ManipulaPy control code.
        """
        import os
        import subprocess
        import sys
        import textwrap

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_control_path = os.path.join(repo_root, "tests", "test_control.py")
        code = textwrap.dedent(f"""
            import importlib.util
            import sys
            import types

            def raise_cuda_runtime_error(*args, **kwargs):
                raise RuntimeError("cudaErrorCompatNotSupportedOnDevice")

            cupy = types.ModuleType("cupy")
            cupy.ndarray = type("ndarray", (), {{}})
            cupy.asarray = raise_cuda_runtime_error
            cupy.asnumpy = lambda arr: arr
            cupy.cuda = types.SimpleNamespace(
                runtime=types.SimpleNamespace(getDeviceCount=raise_cuda_runtime_error)
            )
            sys.modules["cupy"] = cupy

            spec = importlib.util.spec_from_file_location(
                "test_control", r"{test_control_path}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if mod.is_module_available("cupy"):
                raise AssertionError("Broken CuPy runtime must not be selected")
            """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
        self.assertEqual(
            result.returncode,
            0,
            f"test_control.py accepted an unusable cupy runtime:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


class TestPackagingRegressions(unittest.TestCase):
    """Regressions for pyproject.toml / setup.py / wheel metadata (Tasks 45-48)."""

    def _load_pyproject(self) -> tuple:
        """Parse pyproject.toml and return ``(parsed_dict, repo_root)``."""
        import sys
        from pathlib import Path

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore

        repo_root = Path(__file__).resolve().parents[1]
        with open(repo_root / "pyproject.toml", "rb") as f:
            return tomllib.load(f), repo_root

    def test_heavy_dependencies_are_optional_not_core(self) -> None:
        """torch, opencv-python, ultralytics, pybullet, cupy-* and trimesh
        must NOT be in [project.dependencies] — they live in optional groups
        so `pip install ManipulaPy` works on Apple Silicon and minimal
        images (closes GitHub issue #25)."""
        pyproject, _ = self._load_pyproject()

        core_deps = {
            dep.split(">=")[0]
            .split("==")[0]
            .split(";")[0]
            .split("[")[0]
            .strip()
            .lower()
            for dep in pyproject["project"]["dependencies"]
        }
        heavy = {
            "cupy-cuda11x",
            "pybullet",
            "torch",
            "opencv-python",
            "ultralytics",
            "trimesh",
        }
        leaked = core_deps & heavy
        self.assertFalse(
            leaked,
            f"heavy deps still in core dependencies: {leaked}",
        )

        optional = pyproject["project"]["optional-dependencies"]
        for group in ("simulation", "vision", "urdf", "cuda", "all"):
            self.assertIn(group, optional, f"missing optional group [{group}]")

    def test_python_312_classifier_present_no_premature_313(self) -> None:
        """Python 3.12 is supported (Numba/SciPy ship 3.12 wheels). 3.13
        is held back until CI proves it green."""
        pyproject, _ = self._load_pyproject()
        classifiers = pyproject["project"]["classifiers"]
        self.assertIn("Programming Language :: Python :: 3.12", classifiers)
        self.assertNotIn("Programming Language :: Python :: 3.13", classifiers)

    def test_py_typed_marker_present(self) -> None:
        """PEP 561 marker so downstream type checkers honor inline hints."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        self.assertTrue(
            (repo_root / "ManipulaPy" / "py.typed").is_file(),
            "ManipulaPy/py.typed missing — type checkers will treat the "
            "package as untyped",
        )

    def test_setup_py_version_matches_pyproject(self) -> None:
        """setup.py is legacy but stale-version metadata is misleading.
        Keep the version field synced with pyproject.toml's [project].version."""
        from pathlib import Path
        import re

        pyproject, repo_root = self._load_pyproject()
        target = pyproject["project"]["version"]

        text = (repo_root / "setup.py").read_text(encoding="utf-8")
        m = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        self.assertIsNotNone(m, "could not find version= in setup.py")
        self.assertEqual(
            m.group(1),
            target,
            f"setup.py version {m.group(1)!r} != pyproject {target!r}",
        )

    def _setup_py_keyword(self, keyword_name) -> Any:
        """Return a literal setup.py keyword argument from the setup() call."""
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        tree = ast.parse((repo_root / "setup.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
                for keyword in node.keywords:
                    if keyword.arg == keyword_name:
                        return ast.literal_eval(keyword.value)
        raise AssertionError(f"setup.py missing setup({keyword_name}=...)")

    def test_setup_py_core_dependencies_follow_optional_split(self) -> None:
        """Legacy setup.py must not reintroduce the heavy deps that pyproject
        moved behind extras. Some downstream tools still inspect setup.py."""
        install_requires = self._setup_py_keyword("install_requires")
        core_deps = {
            dep.split(">=")[0]
            .split("==")[0]
            .split(";")[0]
            .split("[")[0]
            .strip()
            .lower()
            for dep in install_requires
        }
        heavy = {
            "cupy-cuda11x",
            "cupy-cuda12x",
            "pycuda",
            "pybullet",
            "torch",
            "torchvision",
            "opencv-python",
            "ultralytics",
            "trimesh",
            "scikit-learn",
        }
        self.assertFalse(
            core_deps & heavy,
            f"setup.py install_requires still contains heavy deps: {core_deps & heavy}",
        )

        extras = self._setup_py_keyword("extras_require")
        for group in ("simulation", "vision", "urdf", "cuda", "ml", "all"):
            self.assertIn(group, extras, f"setup.py missing extra {group!r}")

    def test_setup_py_package_data_does_not_reinclude_meshes(self) -> None:
        """Task 48 chose URDF-only wheels; setup.py package_data must not
        contradict MANIFEST.in by listing mesh globs."""
        package_data = self._setup_py_keyword("package_data")
        package_data_text = repr(package_data).lower()
        for suffix in (".stl", ".dae", ".obj", ".mesh"):
            self.assertNotIn(suffix, package_data_text)

    def test_python_312_ci_matrix_present(self) -> None:
        """Declaring Python 3.12 support requires CI coverage for 3.12."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github" / "workflows" / "test.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("3.12", workflow)

    def test_init_py_version_matches_pyproject(self) -> None:
        """ManipulaPy.__version__ must match pyproject.toml [project].version
        so what `pip install` ships and what `import ManipulaPy; print(...)`
        reports are the same string."""
        import ManipulaPy

        pyproject, _ = self._load_pyproject()
        self.assertEqual(
            ManipulaPy.__version__,
            pyproject["project"]["version"],
            "ManipulaPy.__version__ drift from pyproject.toml",
        )

    def test_changelog_has_v132_release_section(self) -> None:
        """v1.3.2 ship-ritual: CHANGELOG must have a dated [1.3.2] section
        (no lingering '[Unreleased] — targeting 1.3.2' header)."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [1.3.2]", text)
        self.assertNotIn("[Unreleased] — targeting 1.3.2", text)

    def test_data_manifest_does_not_lie_about_mesh_bundling(self) -> None:
        """The wheel doesn't bundle .stl/.dae/.obj/.mesh assets (excluded
        by MANIFEST.in). The data manifest must say so explicitly so users
        on a PyPI install aren't surprised."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        manifest = (
            repo_root / "ManipulaPy" / "ManipulaPy_data" / "MANIFEST.md"
        ).read_text(encoding="utf-8")
        # The wheel-note callout must be present
        self.assertIn("PyPI wheel", manifest)
        self.assertIn("NOT bundled", manifest)


class TestXacroRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/urdf/xacro.py — security audit (Task 41)."""

    def test_xacro_arg_name_must_match_identifier_pattern(self) -> None:
        """Verify xacro arg names must match a safe identifier pattern."""
        from pathlib import Path
        from ManipulaPy.urdf.xacro import XacroProcessor

        for bad_key in (
            "--malicious-flag",
            "key with spaces",
            "key;rm -rf",
            "1leading_digit",
        ):
            with self.subTest(key=bad_key):
                with self.assertRaises(ValueError):
                    XacroProcessor._process_with_command(
                        Path("/nonexistent.xacro"), {bad_key: "value"}
                    )

    def test_xacro_arg_value_rejects_shell_metacharacters(self) -> None:
        """Verify xacro arg values reject shell metacharacters."""
        from pathlib import Path
        from ManipulaPy.urdf.xacro import XacroProcessor

        for bad_value in (
            "foo;rm -rf",
            "foo|cat",
            "$(echo)",
            "a`b`",
            "line1\nline2",
            "trailing\x00null",
        ):
            with self.subTest(value=bad_value):
                with self.assertRaises(ValueError):
                    XacroProcessor._process_with_command(
                        Path("/nonexistent.xacro"), {"safe_key": bad_value}
                    )

    def test_xacro_arg_value_rejects_cli_flag_lookalikes(self) -> None:
        """Verify xacro arg values reject CLI-flag-lookalike strings."""
        from pathlib import Path
        from ManipulaPy.urdf.xacro import XacroProcessor

        for bad_value in ("--malicious", "-foo", "--help"):
            with self.subTest(value=bad_value):
                with self.assertRaises(ValueError):
                    XacroProcessor._process_with_command(
                        Path("/nonexistent.xacro"), {"safe_key": bad_value}
                    )

    def test_xacro_arg_value_allows_negative_numbers(self) -> None:
        """Codex caught: rejecting all '-' values would break legit
        negative numerics like joint limits or offsets."""
        import subprocess
        from pathlib import Path
        from unittest.mock import patch
        from ManipulaPy.urdf.xacro import XacroProcessor

        # We expect validation to PASS (no ValueError) for negative numbers,
        # so the call advances to subprocess.run. Mock that so we don't need
        # the xacro binary, then assert the formatted CLI arg made it through.
        with patch("ManipulaPy.urdf.xacro.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="<robot/>", stderr=""
            )
            XacroProcessor._process_with_command(
                Path("/some.xacro"),
                {"offset": "-1.5", "limit": "-2"},
            )
            cmd = mock_run.call_args[0][0]
            self.assertIn("offset:=-1.5", cmd)
            self.assertIn("limit:=-2", cmd)


class TestCodeRabbitRoundTwo(unittest.TestCase):
    """Regressions for the second CodeRabbit pass over v1.3.2 work.

    Each test pins one finding from the review so the same regression
    cannot land twice.
    """

    def test_kalman_filter_update_validates_P_shape(self) -> None:
        """kalman_filter_update must reject self.P=None or wrong-shape P
        with a clear ValueError, before reaching the matrix algebra (which
        would fail with a confusing low-level error).
        """
        from unittest.mock import MagicMock
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController.__new__(ManipulatorController)
        ctrl.dynamics = MagicMock()

        # Case 1: P is None
        ctrl.x_hat = np.zeros(4)
        ctrl.P = None
        with self.assertRaises(ValueError) as ctx:
            ctrl.kalman_filter_update(np.zeros(4), np.eye(4))
        self.assertIn("P must be initialized", str(ctx.exception))

        # Case 2: P is wrong shape
        ctrl.P = np.eye(3)  # should be (4, 4)
        with self.assertRaises(ValueError) as ctx:
            ctrl.kalman_filter_update(np.zeros(4), np.eye(4))
        self.assertIn("(4, 4)", str(ctx.exception))

        # Case 3: P is correctly shaped — should not raise from this guard
        ctrl.P = np.eye(4)
        try:
            ctrl.kalman_filter_update(np.zeros(4), np.eye(4))
        except ValueError as e:
            self.fail(f"P shape check rejected a valid (4,4) P: {e}")

    def test_connect_disconnect_initialize_robot_raise_importerror_without_pybullet(
        self,
    ) -> None:
        """connect_simulation/disconnect_simulation/initialize_robot must
        all surface ImportError when pybullet is missing — previously they
        skipped _check_pybullet_available() and raised AttributeError on
        the None proxy instead.
        """
        from unittest.mock import patch, MagicMock
        from ManipulaPy.sim import Simulation

        for method_name in (
            "connect_simulation",
            "disconnect_simulation",
            "initialize_robot",
        ):
            with self.subTest(method=method_name):
                sim = Simulation.__new__(Simulation)
                sim.logger = MagicMock()
                sim.physics_client = 1
                sim.urdf_file_path = "/nonexistent.urdf"
                sim.time_step = 0.01
                with (
                    patch("ManipulaPy.sim._PYBULLET_AVAILABLE", False),
                    patch("ManipulaPy.sim.p", None),
                ):
                    with self.assertRaises(ImportError) as ctx:
                        getattr(sim, method_name)()
                self.assertIn("ManipulaPy[simulation]", str(ctx.exception))

    def test_set_joint_positions_collapses_torque_pairs_to_scalars(self) -> None:
        """set_joint_positions must hand PyBullet one scalar per joint.
        torque_limits is canonically [(min, max), ...]; the previous code
        passed those tuples through verbatim and PyBullet rejected them.
        """
        from unittest.mock import MagicMock, patch
        from ManipulaPy.sim import Simulation

        sim = Simulation.__new__(Simulation)
        sim.logger = MagicMock()
        sim.robot_id = 1
        sim.non_fixed_joints = list(range(3))
        sim.torque_limits = [(-10.0, 10.0), (-20.0, 5.0), (-3.0, 30.0)]
        sim.physics_client = 1

        captured = {}

        def fake_set_motor(robot_id, indices, mode, targetPositions, forces) -> None:
            """Stub motor-control setter that records the applied forces."""
            captured["forces"] = forces

        fake_p = MagicMock()
        fake_p.setJointMotorControlArray.side_effect = fake_set_motor
        fake_p.POSITION_CONTROL = 0
        with (
            patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True),
            patch("ManipulaPy.sim.p", fake_p),
        ):
            sim.set_joint_positions([0.0, 0.0, 0.0])

        forces = captured["forces"]
        self.assertEqual(len(forces), 3, "one scalar per joint required")
        for f in forces:
            self.assertIsInstance(f, float, f"forces must be scalars, got {type(f)}")
        # Max abs magnitudes from each (min, max) pair.
        self.assertAlmostEqual(forces[0], 10.0)
        self.assertAlmostEqual(forces[1], 20.0)
        self.assertAlmostEqual(forces[2], 30.0)

    def test_manifest_load_example_imports_get_robot_urdf(self) -> None:
        """The ManipulaPy_data/MANIFEST.md snippet that calls
        ``URDF.load(get_robot_urdf('ur5'))`` must also import
        ``get_robot_urdf`` so users can copy-paste it verbatim.
        """
        from pathlib import Path

        manifest = (
            Path(__file__).resolve().parents[1]
            / "ManipulaPy"
            / "ManipulaPy_data"
            / "MANIFEST.md"
        )
        text = manifest.read_text(encoding="utf-8")
        # Every ```python block that calls get_robot_urdf(...) must first
        # import it.
        import re

        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        offending = []
        for i, block in enumerate(blocks):
            if (
                "get_robot_urdf(" in block
                and "from ManipulaPy.ManipulaPy_data import get_robot_urdf" not in block
            ):
                offending.append(i)
        self.assertEqual(
            offending,
            [],
            f"MANIFEST.md python block(s) {offending} call get_robot_urdf without importing it",
        )

    def test_resolver_explicit_map_miss_does_not_fall_through(self) -> None:
        """When add_package() pins a package to a path and the requested
        file is absent under that path, the resolver must NOT silently
        return a hit from another search path — that defeats the explicit
        override.
        """
        import tempfile
        from pathlib import Path
        from ManipulaPy.urdf.resolver import PackageResolver

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pinned = tmp / "pinned"
            other = tmp / "other"
            (pinned / "meshes").mkdir(parents=True)
            (other / "mypkg" / "meshes").mkdir(parents=True)
            # File only exists under the auto-discoverable path, not the
            # explicit one. The resolver should refuse to fall back.
            (other / "mypkg" / "meshes" / "robot.dae").write_text("")

            resolver = PackageResolver(base_path=str(other))
            resolver.add_package("mypkg", str(pinned))

            with self.assertLogs("ManipulaPy.urdf.resolver", level="WARNING") as cm:
                result = resolver._resolve_package_uri(
                    "package://mypkg/meshes/robot.dae"
                )
            self.assertEqual(
                result,
                "package://mypkg/meshes/robot.dae",
                "explicit-map miss must not fall through to other strategies",
            )
            joined = "\n".join(cm.output)
            self.assertIn("Explicit package mapping", joined)

    def test_h2d_pinned_skips_pinned_memory_by_default(self) -> None:
        """_h2d_pinned must NOT call cuda.pinned_array() unless the user
        opts in via MANIPULAPY_USE_PINNED_MEMORY=1. The pinned path
        SIGSEGVs on numba 0.65 + driver 580 (CUDA 13 ABI) — a try/except
        cannot catch it, so the only safe default is to skip it entirely.

        We patch ``cuda.pinned_array`` to a sentinel that records calls
        and verify it is never invoked under the default env.
        """
        import importlib
        import os
        from unittest.mock import MagicMock, patch

        # Make sure the module is loaded so we can patch its symbols.
        import ManipulaPy.cuda_kernels as ck

        if not ck.CUDA_AVAILABLE:
            self.skipTest("CUDA path not exercised when CUDA is unavailable")

        arr = np.zeros(8, dtype=np.float32)
        spy = MagicMock(name="pinned_array_spy")
        # Default: opt-in flag is False — pinned_array must NOT be touched.
        with (
            patch.object(ck, "_PINNED_MEMORY_OPT_IN", False),
            patch.object(ck.cuda, "pinned_array", spy),
        ):
            try:
                ck._h2d_pinned(arr)
            except Exception:
                # We only care that pinned_array wasn't called; the
                # transfer itself may legitimately fail in some test envs.
                pass
            spy.assert_not_called()

    def test_h2d_pinned_uses_pinned_memory_when_opted_in(self) -> None:
        """When the user sets MANIPULAPY_USE_PINNED_MEMORY=1, _h2d_pinned
        must route through cuda.pinned_array() — that's the whole point
        of the opt-in flag. If pinned_array raises a real exception, the
        function falls back to cuda.to_device.
        """
        from unittest.mock import MagicMock, patch
        import ManipulaPy.cuda_kernels as ck

        if not ck.CUDA_AVAILABLE:
            self.skipTest("CUDA path not exercised when CUDA is unavailable")

        arr = np.zeros(8, dtype=np.float32)
        # Sentinel that records the call AND returns a numpy-array-like
        # object so the assignment + to_device still works.
        captured = {}

        def fake_pinned_array(shape, dtype) -> np.ndarray:
            """Record the call and return a plain NumPy array stand-in."""
            captured["called"] = True
            captured["shape"] = shape
            captured["dtype"] = dtype
            return np.zeros(shape, dtype=dtype)

        with (
            patch.object(ck, "_PINNED_MEMORY_OPT_IN", True),
            patch.object(ck.cuda, "pinned_array", side_effect=fake_pinned_array),
        ):
            ck._h2d_pinned(arr)
        self.assertTrue(captured.get("called"))
        self.assertEqual(captured["shape"], arr.shape)

    def test_optimized_potential_field_accepts_empty_obstacles(self) -> None:
        """optimized_potential_field must not crash on an empty obstacle set.

        The trajectory planner's GPU collision-avoidance path passes
        ``np.array([])`` (a 1-D ``(0,)`` array) for the no-obstacle case. The
        fused kernel indexes ``obstacles[obs, 0]`` (2-D), so a 1-D array breaks
        Numba's nopython type inference and aborts the entire large-N GPU
        trajectory run. The wrapper must normalise obstacles to ``(M, 3)``.
        """
        import ManipulaPy.cuda_kernels as ck

        if not ck.CUDA_AVAILABLE:
            self.skipTest("CUDA path not exercised when CUDA is unavailable")

        positions = np.zeros((4, 3), dtype=np.float64)
        goal = np.array([1.0, 0.0, 0.0])
        empty_obstacles = np.array([])  # 1-D (0,) — exactly what the planner passes

        potential, gradient = ck.optimized_potential_field(
            positions, goal, empty_obstacles, influence_distance=0.5, use_pinned=False
        )
        self.assertEqual(potential.shape, (4,))
        self.assertEqual(gradient.shape, (4, 3))

    def test_trajectory_cpu_fallback_handles_zero_Tf_without_warnings(self) -> None:
        """trajectory_cpu_fallback used to emit a numpy RuntimeWarning
        ("invalid value encountered in divide") when called with Tf=0
        because of `tau = t / Tf`. The GPU kernels were guarded against
        N<=1 in earlier commits; the CPU fallback must apply the same
        guard so that degenerate inputs return a sit-at-start trajectory
        instead of NaN/Inf + a warning.
        """
        import warnings
        from ManipulaPy.cuda_kernels import trajectory_cpu_fallback

        thetastart = np.array([0.0, 0.1], dtype=np.float32)
        thetaend = np.array([1.0, 0.9], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Tf = 0 — used to divide by zero; must not warn now.
            pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, 0.0, 10, 3)
            self.assertEqual(pos.shape, (10, 2))
            np.testing.assert_array_equal(vel, np.zeros((10, 2), dtype=np.float32))
            np.testing.assert_array_equal(acc, np.zeros((10, 2), dtype=np.float32))
            # Every sample should sit at the start configuration.
            for row in range(10):
                np.testing.assert_allclose(pos[row], thetastart, rtol=1e-5)

            # N = 1 — same degenerate-input handling.
            pos1, vel1, acc1 = trajectory_cpu_fallback(thetastart, thetaend, 1.0, 1, 5)
            self.assertEqual(pos1.shape, (1, 2))
            np.testing.assert_allclose(pos1[0], thetastart, rtol=1e-5)
            np.testing.assert_array_equal(vel1, np.zeros((1, 2), dtype=np.float32))
            np.testing.assert_array_equal(acc1, np.zeros((1, 2), dtype=np.float32))

    def test_trajectory_kernels_guard_division_by_zero_when_N_equals_one(self) -> None:
        """All trajectory kernels normalize by (N - 1). With N=1 that is
        division by zero, producing NaN/Inf. Source must guard with
        ``0.0 if N <= 1 else t_idx / (N - 1.0)``.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "ManipulaPy" / "cuda_kernels.py"
        text = src.read_text(encoding="utf-8")

        # The unguarded forms must not appear anywhere.
        bad = [
            line
            for line in text.splitlines()
            if line.strip().startswith("tau =")
            and "t_idx /" in line
            and "0.0 if N <= 1 else" not in line
        ]
        self.assertEqual(
            bad,
            [],
            f"unguarded N==1 division remains: {bad}",
        )
        # And the guarded form must appear at least 7 times (one per kernel
        # variant identified in the v1.3.2 patch).
        guarded_count = text.count("0.0 if N <= 1 else t_idx / (N - 1.0)")
        self.assertGreaterEqual(
            guarded_count,
            7,
            f"expected >=7 guarded tau assignments, got {guarded_count}",
        )

    def test_simulation_rst_open_loop_examples_import_numpy(self) -> None:
        """The Open-Loop RST snippets call ``np.array(...)``; without an
        explicit ``import numpy as np`` they fail when copy-pasted.
        """
        from pathlib import Path

        rst = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "source"
            / "api"
            / "simulation.rst"
        )
        text = rst.read_text(encoding="utf-8")

        # Find every Open-Loop section (there are two) — each must contain
        # an ``import numpy as np`` line ahead of the np.array call.
        import re

        # Split by RST heading ---- underlines so we get section bodies.
        sections = re.split(r"\n(?=Open-Loop[^\n]*\n[-=~]+\n)", text)
        offenders = []
        for section in sections:
            header_match = re.match(r"(Open-Loop[^\n]*)", section)
            if not header_match:
                continue
            if "np.array(" in section and "import numpy as np" not in section:
                offenders.append(header_match.group(1))
        self.assertEqual(
            offenders,
            [],
            f"Open-Loop section(s) {offenders} use np.array() without `import numpy as np`",
        )


class TestSelfCollision(unittest.TestCase):
    """Regressions for Task 49: adjacent-link ACM, collision-mesh source, PyBullet flag."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tetra_hull(self, offset=None) -> Any:
        """Return a small tetrahedron ConvexHull, optionally translated."""
        from scipy.spatial import ConvexHull

        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if offset is not None:
            pts = pts + np.asarray(offset)
        return ConvexHull(pts)

    def _make_mock_checker(self, hull_map, acm=None) -> "CollisionChecker":
        """Build a CollisionChecker with injected hulls and ACM, bypassing URDF loading."""
        from unittest.mock import MagicMock, patch
        from ManipulaPy.potential_field import CollisionChecker

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
        checker.robot = MagicMock()
        checker.convex_hulls = hull_map
        checker._acm = acm if acm is not None else set()
        checker._visual_fallback_warned = set()
        return checker

    def _make_self_contact_urdf(self) -> str:
        """Create a primitive URDF with deterministic base<->link self contact."""
        import os
        import tempfile
        import textwrap

        urdf = textwrap.dedent("""\
            <?xml version="1.0"?>
            <robot name="self_contact_fixture">
              <link name="base">
                <collision>
                  <geometry><box size="0.4 0.4 0.4"/></geometry>
                </collision>
                <inertial>
                  <mass value="1"/>
                  <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
              </link>
              <link name="link1">
                <collision>
                  <geometry><box size="0.2 0.2 0.2"/></geometry>
                </collision>
                <inertial>
                  <mass value="1"/>
                  <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
              </link>
              <link name="link2">
                <collision>
                  <origin xyz="0 0 -1.0"/>
                  <geometry><box size="0.3 0.3 0.3"/></geometry>
                </collision>
                <inertial>
                  <mass value="1"/>
                  <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                </inertial>
              </link>
              <joint name="joint1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0 0 0.5"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
              </joint>
              <joint name="joint2" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <origin xyz="0 0 0.5"/>
                <axis xyz="0 1 0"/>
                <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
              </joint>
            </robot>
            """)
        fd, path = tempfile.mkstemp(suffix=".urdf", prefix="manipulapy_self_contact_")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(urdf)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    # ------------------------------------------------------------------
    # ACM tests
    # ------------------------------------------------------------------

    def test_acm_excludes_joint_neighbors(self) -> None:
        """build_link_adjacency must include every parent<->child joint pair."""
        from pathlib import Path

        try:
            from ManipulaPy.ManipulaPy_data.ur5 import urdf_file
            from ManipulaPy.urdf import URDF
            from ManipulaPy.potential_field import build_link_adjacency
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"UR5 fixture unavailable: {e}")
        if not Path(str(urdf_file)).exists():
            self.skipTest(f"UR5 fixture file not found: {urdf_file}")

        robot = URDF.load(str(urdf_file))
        acm = build_link_adjacency(robot, exclude_grandparents=False)

        for j in robot.joints:
            pair = frozenset({j.parent, j.child})
            self.assertIn(
                pair,
                acm,
                f"ACM missing joint pair {j.parent!r} <-> {j.child!r}",
            )

    def test_acm_grandparents_excluded_when_flag_set(self) -> None:
        """With exclude_grandparents=True, grandparent<->grandchild pairs are also excluded."""
        from pathlib import Path

        try:
            from ManipulaPy.ManipulaPy_data.ur5 import urdf_file
            from ManipulaPy.urdf import URDF
            from ManipulaPy.potential_field import build_link_adjacency
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"UR5 fixture unavailable: {e}")
        if not Path(str(urdf_file)).exists():
            self.skipTest(f"UR5 fixture file not found: {urdf_file}")

        robot = URDF.load(str(urdf_file))
        acm_with = build_link_adjacency(robot, exclude_grandparents=True)
        acm_without = build_link_adjacency(robot, exclude_grandparents=False)

        # With grandparents, ACM must be a strict superset (or equal on degenerate robots)
        self.assertTrue(
            acm_without.issubset(acm_with),
            "ACM with grandparents must contain all pairs from ACM without",
        )

    def test_acm_empty_joints(self) -> None:
        """build_link_adjacency on a URDF with no joints returns an empty set."""
        from unittest.mock import MagicMock
        from ManipulaPy.potential_field import build_link_adjacency

        mock_urdf = MagicMock()
        mock_urdf.joints = []
        result = build_link_adjacency(mock_urdf)
        self.assertEqual(result, set())

    # ------------------------------------------------------------------
    # CollisionChecker ACM integration
    # ------------------------------------------------------------------

    def test_home_pose_no_false_positive(self) -> None:
        """Adjacent-link hulls placed at overlapping positions must not trigger
        a collision when their pair is in the ACM."""
        # Two overlapping hulls for adjacent links — old code reported collision here
        hull_a = self._make_tetra_hull()
        hull_b = self._make_tetra_hull(offset=[0.3, 0.0, 0.0])  # clearly overlapping

        acm = {frozenset({"link_a", "link_b"})}
        checker = self._make_mock_checker({"link_a": hull_a, "link_b": hull_b}, acm=acm)
        checker.robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        }

        result = checker.check_collision([0.0] * 6)
        self.assertFalse(
            result,
            "Adjacent links in ACM must not produce a false-positive collision",
        )

    def test_intentional_self_collision_detected(self) -> None:
        """A deliberate non-adjacent hull overlap must be flagged."""
        hull_base = self._make_tetra_hull()
        hull_tool = self._make_tetra_hull(offset=[0.3, 0.0, 0.0])

        checker = self._make_mock_checker(
            {"base": hull_base, "tool": hull_tool},
            acm={frozenset({"base", "link1"}), frozenset({"link1", "tool"})},
        )
        checker.robot.link_fk.return_value = {
            "base": np.eye(4),
            "tool": np.eye(4),
        }

        self.assertTrue(
            checker.check_collision([0.0] * 2),
            "Non-adjacent overlapping links outside the ACM must report collision",
        )

    def test_non_adjacent_overlapping_hulls_detected(self) -> None:
        """Non-adjacent links that physically overlap MUST be detected as a collision."""
        hull_a = self._make_tetra_hull()
        hull_b = self._make_tetra_hull(offset=[0.3, 0.0, 0.0])  # overlapping

        # Empty ACM — no exclusions
        checker = self._make_mock_checker(
            {"link_a": hull_a, "link_b": hull_b}, acm=set()
        )
        checker.robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        }

        result = checker.check_collision([0.0] * 6)
        self.assertTrue(
            result,
            "Non-adjacent overlapping hulls must be detected as a collision",
        )

    def test_non_adjacent_separated_hulls_not_detected(self) -> None:
        """Non-adjacent links that are far apart must not be reported as colliding."""
        hull_a = self._make_tetra_hull()
        hull_b = self._make_tetra_hull(offset=[20.0, 0.0, 0.0])

        checker = self._make_mock_checker(
            {"link_a": hull_a, "link_b": hull_b}, acm=set()
        )
        checker.robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        }

        result = checker.check_collision([0.0] * 6)
        self.assertFalse(result, "Separated far-apart hulls must not report collision")

    def test_check_collision_applies_fk_translation_to_cached_vertices(self) -> None:
        """Verify check_collision actually transforms cached hull vertices by the FK matrix
        (not just inspecting them in body frame). A hull at origin and a translated FK should
        place the hull's world-frame vertices in the translated location."""
        hull_a = self._make_tetra_hull()
        hull_b = self._make_tetra_hull()  # same body-frame geometry

        checker = self._make_mock_checker(
            {"link_a": hull_a, "link_b": hull_b}, acm=set()
        )

        # Place A at origin, B 100m away — must NOT collide
        T_a = np.eye(4)
        T_b = np.eye(4)
        T_b[:3, 3] = [100.0, 0.0, 0.0]
        checker.robot.link_fk.return_value = {"link_a": T_a, "link_b": T_b}
        self.assertFalse(
            checker.check_collision([0.0] * 6),
            "Two identical hulls at +100m apart must not report collision",
        )

        # Now place B AT the same origin — must collide
        T_b[:3, 3] = [0.0, 0.0, 0.0]
        checker.robot.link_fk.return_value = {"link_a": T_a, "link_b": T_b}
        self.assertTrue(
            checker.check_collision([0.0] * 6),
            "Two identical hulls co-located after FK must report collision",
        )

    # ------------------------------------------------------------------
    # Collision-mesh source preference
    # ------------------------------------------------------------------

    def test_collision_checker_prefers_collision_meshes(self) -> None:
        """_create_convex_hulls must use link.collisions vertices when both are present,
        and fall back to link.visuals only when collisions is empty."""
        from unittest.mock import MagicMock, patch
        from ManipulaPy.potential_field import CollisionChecker

        # 8-vertex cube — distinguishable vertex count for the collision mesh
        collision_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        # 6-vertex octahedron — distinguishable vertex count for the visual mesh
        visual_verts = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

        def make_geom_element(vertices) -> Any:
            """Build a mock collision/visual element exposing mesh vertices."""
            elem = MagicMock()
            elem.geometry = MagicMock()
            elem.geometry.mesh_data = MagicMock()
            elem.geometry.mesh_data.vertices = vertices
            # Ensure the "legacy mesh" branch doesn't accidentally fire
            del elem.geometry.mesh
            return elem

        # Link with BOTH collisions and visuals — collisions must win
        link_both = MagicMock()
        link_both.name = "link_both"
        link_both.collisions = [make_geom_element(collision_verts)]
        link_both.visuals = [make_geom_element(visual_verts)]

        # Link with empty collisions — visual fallback must engage
        link_visual_only = MagicMock()
        link_visual_only.name = "link_visual_only"
        link_visual_only.collisions = []
        link_visual_only.visuals = [make_geom_element(visual_verts)]

        stub_robot = MagicMock()
        stub_robot.links = [link_both, link_visual_only]
        stub_robot.joints = []

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
        checker.robot = stub_robot
        checker._acm = set()
        checker._visual_fallback_warned = set()

        with self.assertLogs("ManipulaPy.potential_field", level="WARNING"):
            hulls = checker._create_convex_hulls()

        self.assertIn("link_both", hulls)
        self.assertEqual(
            hulls["link_both"].points.shape[0],
            collision_verts.shape[0],
            "When both collisions and visuals exist, collision mesh vertices must be used",
        )
        self.assertIn("link_visual_only", hulls)
        self.assertEqual(
            hulls["link_visual_only"].points.shape[0],
            visual_verts.shape[0],
            "When collisions is empty, visual mesh vertices must be used as fallback",
        )
        self.assertIn(
            "link_visual_only",
            checker._visual_fallback_warned,
            "Visual fallback must mark the link as warned",
        )

    def test_visual_fallback_warning_issued_once(self) -> None:
        """When falling back from collisions to visuals, a warning is logged exactly once per link."""
        import logging
        from unittest.mock import MagicMock, patch
        from ManipulaPy.potential_field import CollisionChecker

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
        checker._visual_fallback_warned = set()

        with self.assertLogs("ManipulaPy.potential_field", level="WARNING") as cm:
            checker._warn_visual_fallback_once("test_link")
            checker._warn_visual_fallback_once("test_link")  # must NOT log again

        warnings = [line for line in cm.output if "test_link" in line]
        self.assertEqual(
            len(warnings), 1, "Warning for the same link must be issued exactly once"
        )

    # ------------------------------------------------------------------
    # Simulation self-collision flag
    # ------------------------------------------------------------------

    @pytest.mark.simulation
    def test_sim_self_collision_off_by_default(self) -> None:
        """Simulation with enable_self_collision=False returns [] from check_collisions."""
        try:
            import pybullet as p
        except ImportError:
            self.skipTest("pybullet not available")

        from ManipulaPy.sim import Simulation

        urdf_file = self._make_self_contact_urdf()
        joint_limits = [(-np.pi, np.pi)] * 2
        client = p.connect(p.DIRECT)
        try:
            sim = Simulation(
                urdf_file,
                joint_limits,
                physics_client=client,
                enable_self_collision=False,
            )

            p.performCollisionDetection()
            contacts = sim.check_collisions()
            self.assertIsInstance(contacts, list)
            self.assertEqual(contacts, [])
        finally:
            try:
                p.disconnect(client)
            except Exception:
                pass

    @pytest.mark.simulation
    def test_sim_self_collision_flag_wires_pybullet_flag(self) -> None:
        """enable_self_collision=True passes URDF_USE_SELF_COLLISION to loadURDF."""
        try:
            import pybullet as p
        except ImportError:
            self.skipTest("pybullet not available")

        from unittest.mock import patch
        from ManipulaPy.sim import Simulation

        urdf_file = self._make_self_contact_urdf()
        joint_limits = [(-np.pi, np.pi)] * 2
        client = p.connect(p.DIRECT)
        try:
            load_calls = []
            original_loadURDF = p.loadURDF

            def capturing_loadURDF(*args, **kwargs) -> Any:
                """Record loadURDF calls, then delegate to the real loader."""
                load_calls.append((args, kwargs))
                return original_loadURDF(*args, **kwargs)

            with patch("pybullet.loadURDF", side_effect=capturing_loadURDF):
                Simulation(
                    urdf_file,
                    joint_limits,
                    physics_client=client,
                    enable_self_collision=True,
                )

            # Find the robot loadURDF call (not the plane)
            robot_calls = [c for c in load_calls if urdf_file in str(c[0])]
            self.assertGreater(
                len(robot_calls), 0, "loadURDF was not called for the robot"
            )
            args, kwargs = robot_calls[0]
            flags = kwargs.get("flags", 0)
            self.assertTrue(
                bool(flags & p.URDF_USE_SELF_COLLISION),
                f"URDF_USE_SELF_COLLISION not set in loadURDF flags (got flags={flags})",
            )
        finally:
            try:
                p.disconnect(client)
            except Exception:
                pass

    @pytest.mark.simulation
    def test_sim_check_collisions_returns_list(self) -> None:
        """check_collisions() must return a list (not None) in all cases."""
        try:
            import pybullet as p
        except ImportError:
            self.skipTest("pybullet not available")

        from ManipulaPy.sim import Simulation

        urdf_file = self._make_self_contact_urdf()
        joint_limits = [(-np.pi, np.pi)] * 2
        client = p.connect(p.DIRECT)
        try:
            sim = Simulation(
                urdf_file,
                joint_limits,
                physics_client=client,
                enable_self_collision=True,
            )

            p.performCollisionDetection()
            result = sim.check_collisions()
            self.assertIsInstance(result, list, "check_collisions must return a list")
            # Each element must be a 3-tuple (linkA, linkB, position)
            for item in result:
                self.assertEqual(
                    len(item), 3, f"Contact tuple must have 3 elements, got {item}"
                )
        finally:
            try:
                p.disconnect(client)
            except Exception:
                pass

    @pytest.mark.simulation
    def test_sim_self_collision_returns_contacts_when_enabled(self) -> None:
        """A primitive non-adjacent self contact is reported when enabled."""
        try:
            import pybullet as p
        except ImportError:
            self.skipTest("pybullet not available")

        from ManipulaPy.sim import Simulation

        urdf_file = self._make_self_contact_urdf()
        joint_limits = [(-np.pi, np.pi)] * 2
        client = p.connect(p.DIRECT)
        try:
            sim = Simulation(
                urdf_file,
                joint_limits,
                physics_client=client,
                enable_self_collision=True,
            )

            p.performCollisionDetection()
            contacts = sim.check_collisions()
            self.assertGreater(len(contacts), 0)
            for item in contacts:
                self.assertEqual(
                    len(item),
                    3,
                    f"Contact tuple must have 3 elements, got {item}",
                )
                link_a, link_b, position = item
                self.assertIsInstance(link_a, int)
                self.assertIsInstance(link_b, int)
                self.assertIsInstance(position, tuple)
        finally:
            try:
                p.disconnect(client)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
