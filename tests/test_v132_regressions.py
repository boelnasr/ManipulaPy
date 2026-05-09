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
    
    def test_gravity_forces_mutable_default_is_safe(self):
        import inspect
        from ManipulaPy.dynamics import ManipulatorDynamics

        sig = inspect.signature(ManipulatorDynamics.gravity_forces)
        g_default = sig.parameters["g"].default
        self.assertIsNone(g_default, "Mutable default detected — should use None")


class TestKinematicsRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/kinematics.py bugs."""


class TestPathPlanningRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/path_planning.py bugs."""

    def test_cartesian_quintic_acceleration_satisfies_boundary_conditions(self):
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
            "cpu_calls": 0, "gpu_calls": 0,
            "total_cpu_time": 0.0, "total_gpu_time": 0.0,
            "kernel_launches": 0,
        }
        pstart = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pend   = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        Tf, N  = 1.0, 101  # 101 samples → endpoints at indices 0 and 100

        traj_vel, traj_acc = planner._cartesian_trajectory_cpu(
            pstart, pend, Tf, N, method=5
        )

        # Boundary acceleration MUST be zero for a quintic profile
        np.testing.assert_array_almost_equal(
            traj_acc[0], [0, 0, 0], decimal=4,
            err_msg="Quintic s_ddot(0) != 0 — boundary condition violated")
        np.testing.assert_array_almost_equal(
            traj_acc[-1], [0, 0, 0], decimal=4,
            err_msg="Quintic s_ddot(Tf) != 0 — missing +120·tau^3 term")

        # And boundary velocity must also be zero
        np.testing.assert_array_almost_equal(
            traj_vel[0], [0, 0, 0], decimal=4,
            err_msg="Quintic s_dot(0) != 0")
        np.testing.assert_array_almost_equal(
            traj_vel[-1], [0, 0, 0], decimal=4,
            err_msg="Quintic s_dot(Tf) != 0")

    def test_screw_list_sign_correct_via_home_fk(self):
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
        self.assertTrue(os.path.exists(urdf_path),
                        f"Test fixture missing: {urdf_path}")

        urdf = URDFToSerialManipulator(urdf_path)
        robot = urdf.serial_manipulator
        n = robot.S_list.shape[1]
        home_config = np.zeros(n)

        T_space = robot.forward_kinematics(home_config, frame="space")
        T_body = robot.forward_kinematics(home_config, frame="body")

        np.testing.assert_array_almost_equal(
            T_space, robot.M_list, decimal=6,
            err_msg="Space-frame FK at home != M_list — S_list sign suspect")
        np.testing.assert_array_almost_equal(
            T_body, robot.M_list, decimal=6,
            err_msg="Body-frame FK at home != M_list — B_list sign suspect")

        # Joint motion must actually move the EE — guards against a
        # degenerate zero-screw chain that would also satisfy T == M trivially.
        theta_test = np.zeros(n)
        theta_test[0] = 0.01
        T_pert = robot.forward_kinematics(theta_test, frame="space")
        self.assertFalse(
            np.allclose(T_pert, robot.M_list, atol=1e-6),
            "FK didn't respond to joint motion — screw chain may be degenerate")
        

    def test_quintic_polynomial_endpoints(self):
        """Pure math test: quintic time-scaling endpoints."""
        def s(tau):
            return 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        def s_ddot(tau, Tf=1.0):
            return (60 * tau - 180 * tau**2 + 120 * tau**3) / (Tf * Tf)

        self.assertAlmostEqual(s(0.0), 0.0)
        self.assertAlmostEqual(s(1.0), 1.0)
        self.assertAlmostEqual(s_ddot(0.0), 0.0)
        self.assertAlmostEqual(s_ddot(1.0), 0.0)
    def test_plot_tcp_trajectory_does_not_shadow_time_module(self):
        """plot_tcp_trajectory must not bind a local 'time' that shadows
        the imported 'time' module — doing so breaks the timing branch
        with AttributeError mid-execution."""
        import inspect
        from ManipulaPy import path_planning

        src = inspect.getsource(path_planning)
        self.assertNotIn(
            "time = np.arange", src,
            "Variable 'time' shadows the time module — rename to time_array",
        )

    def test_quintic_trajectory_zero_endpoint_acceleration(self):
        from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

        class MockManip:
            joint_limits = [(-3.14, 3.14)] * 6

        class MockDyn:
            pass

        planner = OptimizedTrajectoryPlanning(
            MockManip(), urdf_path=None, dynamics=MockDyn(),
            joint_limits=[(-3.14, 3.14)] * 6,
        )
        thetastart = np.zeros(6)
        thetaend = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])
        traj = planner.joint_trajectory(thetastart, thetaend, Tf=2.0, N=100, method=5)

        np.testing.assert_array_almost_equal(traj["accelerations"][0], np.zeros(6), decimal=4)
        np.testing.assert_array_almost_equal(traj["accelerations"][-1], np.zeros(6), decimal=4)
    
    def test_plot_tcp_trajectory_does_not_shadow_time_module(self):
        import inspect
        from ManipulaPy import path_planning

        src = inspect.getsource(path_planning)
        self.assertNotIn(
            "time = np.arange", src,
            "Variable 'time' shadows the time module — rename to time_array",
        )


class TestControlRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/control.py bugs."""
    def test_cartesian_space_control_dimensions(self):
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

    def test_rise_time_handles_response_never_reaching_setpoint(self):
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)

        rt = ctrl.calculate_rise_time(np.linspace(0, 1, 100), np.zeros(100), 1.0)
        self.assertTrue(np.isinf(rt) or rt >= 0)

    def test_percent_overshoot_handles_zero_setpoint(self):
        from ManipulaPy.control import ManipulatorController

        ctrl = ManipulatorController(None)

        po = ctrl.calculate_percent_overshoot(np.array([0.0, 0.1, 0.0]), set_point=0.0)
        self.assertTrue(np.isfinite(po))


class TestSingularityRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/singularity.py bugs."""

    def test_singularity_analysis_works_on_redundant_robot(self):
        from ManipulaPy.singularity import Singularity

        class MockManip:
            def jacobian(self, thetalist, frame="space"):
                return np.random.randn(6, 7)  # 7-DOF redundant

        sing = Singularity(MockManip())
        result = sing.singularity_analysis(np.zeros(7))
        self.assertIsInstance(result, (bool, np.bool_))

    def test_manipulability_ellipsoid_radii_finite_at_singular_jacobian(self):
        """A singular Jacobian must still produce finite plotted ellipsoid points."""
        from ManipulaPy.singularity import Singularity

        class MockManip:
            def jacobian(self, thetalist, frame="space"):
                return np.zeros((6, 6))

        class RecordingAxis:
            def __init__(self):
                self.surfaces = []

            def plot_surface(self, x, y, z, **kwargs):
                self.surfaces.append((x, y, z))

            def set_title(self, title):
                pass

        ax = RecordingAxis()
        Singularity(MockManip()).manipulability_ellipsoid(np.zeros(6), ax=ax)

        self.assertEqual(len(ax.surfaces), 2)
        for surface in ax.surfaces:
            for values in surface:
                self.assertTrue(np.all(np.isfinite(values)))


class TestPotentialFieldRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/potential_field.py bugs."""

    def test_repulsive_potential_when_at_obstacle(self):
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(attractive_gain=1.0, repulsive_gain=1.0, influence_distance=0.5)
        q = np.array([1.0, 1.0, 1.0])
        p_val = pf.compute_repulsive_potential(q, [q.copy()])
        self.assertTrue(np.isfinite(p_val))

    def test_repulsive_gradient_when_at_obstacle(self):
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(attractive_gain=1.0, repulsive_gain=1.0, influence_distance=0.5)
        q = np.array([1.0, 1.0, 1.0])
        grad = pf.compute_gradient(q, np.zeros(3), [q.copy()])
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_repulsive_gradient_at_obstacle_provides_escape_direction(self):
        """When q == obstacle exactly, gradient must be nonzero so robot can escape."""
        from ManipulaPy.potential_field import PotentialField

        pf = PotentialField(attractive_gain=0.0, repulsive_gain=1.0, influence_distance=0.5)
        q = np.array([1.0, 1.0, 1.0])
        grad = pf.compute_gradient(q, np.zeros(3), [q.copy()])
        # Must be nonzero so -grad gives a nonzero force
        self.assertGreater(np.linalg.norm(grad), 0, "Robot stuck at obstacle has no escape force")


class TestSimRegressions(unittest.TestCase):
    """Regressions for ManipulaPy/sim.py bugs."""

    def test_sim_module_imports_without_cupy_or_pybullet(self):
        """sim module must be importable even when optional deps are missing.

        Runs in a subprocess so that mucking with ``sys.modules`` and the
        import system can't leak state into other tests in this process
        (notably tests/test_sim.py, which captures Simulation at import time
        and relies on a monkeypatched ManipulaPy.sim.p).

        Only pybullet/pybullet_data are blocked here. Blocking cupy too would
        crash a transitive import of ManipulaPy.control, whose own optional
        cupy guard is tracked separately. The cupy fallback in sim.py is
        verified by source: ``cp = np`` and ``cp.asnumpy = np.asarray``.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
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
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}",
        )

    def test_run_controller_tracks_desired_positions(self):
        """run_controller must end at the final desired position (open-loop)."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        desired_positions = [
            np.zeros(6),
            np.array([0.5, 0, 0, 0, 0, 0]),
            np.array([1.0, 0, 0, 0, 0, 0]),
        ]

        with patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True), \
             patch("ManipulaPy.sim.p") as mock_p, \
             patch("ManipulaPy.sim.time.sleep"):
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

            actual_calls = [call.args[0] for call in sim.set_joint_positions.call_args_list]
            self.assertEqual(len(actual_calls), len(desired_positions))
            for actual, expected in zip(actual_calls, desired_positions):
                np.testing.assert_array_equal(actual, expected)

    def test_run_controller_rejects_1d_input(self):
        """run_controller requires a 2D waypoint matrix."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True), \
             patch("ManipulaPy.sim.p"), \
             patch("ManipulaPy.sim.time.sleep"):
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

    def test_run_controller_rejects_empty_input(self):
        """run_controller rejects empty trajectories before stepping."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True), \
             patch("ManipulaPy.sim.p"), \
             patch("ManipulaPy.sim.time.sleep"):
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

    def test_run_controller_rejects_wrong_dof(self):
        """run_controller requires waypoint width to match movable joints."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        with patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True), \
             patch("ManipulaPy.sim.p"), \
             patch("ManipulaPy.sim.time.sleep"):
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

    def test_run_controller_accepts_generator(self):
        """run_controller materializes iterable waypoints before tracking."""
        from unittest.mock import MagicMock, patch

        from ManipulaPy.sim import Simulation

        first_waypoint = np.zeros(6)
        second_waypoint = np.array([0.5, 0, 0, 0, 0, 0])
        desired_positions = (waypoint for waypoint in [first_waypoint, second_waypoint])

        with patch("ManipulaPy.sim._PYBULLET_AVAILABLE", True), \
             patch("ManipulaPy.sim.p") as mock_p, \
             patch("ManipulaPy.sim.time.sleep"):
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
