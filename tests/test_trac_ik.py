#!/usr/bin/env python3

"""
Tests for the TRAC-IK solver module.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import threading
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.trac_ik import TracIKSolver, trac_ik_solve


class _RobotFixtureMixin:
    """Shared 6-DOF robot fixture for trac_ik tests."""

    def _make_robot(self):
        Slist = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, -1, 0, -0.089, 0, 0],
                [0, -1, 0, -0.089, 0, 0.425],
                [0, -1, 0, -0.089, 0, 0.817],
                [1, 0, 0, 0, 0.109, 0],
                [0, -1, 0, -0.089, 0, 0.817],
            ]
        ).T
        M = np.array(
            [[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]]
        )
        B_list = np.copy(Slist)
        joint_limits = [(-np.pi, np.pi)] * 6

        return SerialManipulator(
            M_list=M,
            omega_list=Slist[:3, :],
            S_list=Slist,
            B_list=B_list,
            joint_limits=joint_limits,
        )

    def _make_solver(self, robot=None):
        if robot is None:
            robot = self._make_robot()
        return TracIKSolver(
            fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: robot.jacobian(th, frame="space"),
            joint_limits=robot.joint_limits,
            n_joints=len(robot.joint_limits),
        )

    def _reachable_target(self, robot=None):
        """Return a T_desired known to be reachable."""
        if robot is None:
            robot = self._make_robot()
        theta_known = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
        return robot.forward_kinematics(theta_known, frame="space"), theta_known


# =====================================================================
# TracIKSolver construction
# =====================================================================
class TestTracIKSolverInit(unittest.TestCase, _RobotFixtureMixin):
    """Tests for TracIKSolver.__init__."""

    def test_init_stores_attributes(self):
        solver = self._make_solver()
        self.assertEqual(solver.n_joints, 6)
        self.assertEqual(len(solver.joint_limits), 6)
        self.assertEqual(len(solver.bounds), 6)

    def test_init_none_limits_use_defaults(self):
        """Joints with None limits should get ±2π bounds."""
        robot = self._make_robot()
        solver = TracIKSolver(
            fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: robot.jacobian(th, frame="space"),
            joint_limits=[(None, None), (-1.0, 1.0), (None, 2.0)],
            n_joints=3,
        )
        lb0, ub0 = solver.bounds[0]
        self.assertAlmostEqual(lb0, -2 * np.pi)
        self.assertAlmostEqual(ub0, 2 * np.pi)
        self.assertAlmostEqual(solver.bounds[1][0], -1.0)
        self.assertAlmostEqual(solver.bounds[2][0], -2 * np.pi)
        self.assertAlmostEqual(solver.bounds[2][1], 2.0)

    def test_custom_error_func(self):
        """Custom error_func should replace default."""
        robot = self._make_robot()
        custom_fn = MagicMock(return_value=(np.zeros(6), 0.0, 0.0))
        solver = TracIKSolver(
            fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: robot.jacobian(th, frame="space"),
            joint_limits=robot.joint_limits,
            n_joints=6,
            error_func=custom_fn,
        )
        self.assertIs(solver.error_func, custom_fn)


# =====================================================================
# solve() — parallel and sequential
# =====================================================================
class TestTracIKSolverSolve(unittest.TestCase, _RobotFixtureMixin):
    """Tests for TracIKSolver.solve (high-level API)."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_solve_parallel_success(self):
        """Parallel solve should find a solution for a reachable target."""
        theta, success, solve_time = self.solver.solve(
            self.T_desired, theta0=self.theta_known, timeout=2.0, use_parallel=True
        )
        self.assertTrue(success)
        self.assertEqual(theta.shape, (6,))
        self.assertGreater(solve_time, 0.0)

    def test_solve_sequential_success(self):
        """Sequential solve should also succeed."""
        theta, success, solve_time = self.solver.solve(
            self.T_desired,
            theta0=self.theta_known,
            timeout=2.0,
            use_parallel=False,
        )
        self.assertTrue(success)

    def test_solve_without_initial_guess(self):
        """Solve without theta0 should use workspace heuristic."""
        theta, success, solve_time = self.solver.solve(
            self.T_desired, theta0=None, timeout=2.0
        )
        # May or may not succeed without a good starting point, but should not crash
        self.assertEqual(theta.shape, (6,))
        self.assertGreater(solve_time, 0.0)

    def test_solve_returns_best_on_failure(self):
        """Unreachable target should return best attempt, success=False."""
        T_far = np.eye(4)
        T_far[:3, 3] = [100.0, 100.0, 100.0]  # Very far away
        theta, success, _ = self.solver.solve(T_far, timeout=0.1)
        self.assertFalse(success)
        self.assertEqual(theta.shape, (6,))

    def test_solve_respects_tolerances(self):
        """Tight tolerances should be honoured when success=True."""
        eomg, ev = 1e-4, 1e-4
        theta, success, _ = self.solver.solve(
            self.T_desired, theta0=self.theta_known, eomg=eomg, ev=ev, timeout=2.0
        )
        if success:
            T_result = self.robot.forward_kinematics(theta, frame="space")
            pos_err = np.linalg.norm(T_result[:3, 3] - self.T_desired[:3, 3])
            R_err = T_result[:3, :3].T @ self.T_desired[:3, :3]
            rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            self.assertLess(pos_err, ev)
            self.assertLess(rot_err, eomg)

    def test_solve_num_restarts(self):
        """Different num_restarts values should not crash."""
        for n in [1, 2, 5]:
            theta, _, _ = self.solver.solve(
                self.T_desired, theta0=self.theta_known, num_restarts=n, timeout=1.0
            )
            self.assertEqual(theta.shape, (6,))


# =====================================================================
# _generate_initial_guesses
# =====================================================================
class TestInitialGuesses(unittest.TestCase, _RobotFixtureMixin):
    """Tests for _generate_initial_guesses."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)
        self.T_desired, _ = self._reachable_target(self.robot)

    def test_with_theta0(self):
        theta0 = np.ones(6) * 0.5
        guesses = self.solver._generate_initial_guesses(
            self.T_desired, theta0, num_restarts=5
        )
        # First guess should be copy of theta0
        np.testing.assert_array_almost_equal(guesses[0], theta0)
        self.assertGreaterEqual(len(guesses), 4)

    def test_without_theta0(self):
        guesses = self.solver._generate_initial_guesses(
            self.T_desired, None, num_restarts=6
        )
        self.assertGreaterEqual(len(guesses), 4)
        # First should be workspace heuristic, second midpoint
        self.assertEqual(guesses[0].shape, (6,))
        self.assertEqual(guesses[1].shape, (6,))

    def test_minimum_restarts(self):
        """With few restarts, should still generate core guesses."""
        guesses = self.solver._generate_initial_guesses(
            self.T_desired, None, num_restarts=1
        )
        # Always adds workspace heuristic + midpoint + zero + flipped
        self.assertGreaterEqual(len(guesses), 1)


# =====================================================================
# SVD-robust Jacobian solve (tested via _dls_solver internals)
# =====================================================================
class TestSVDRobustSolve(unittest.TestCase, _RobotFixtureMixin):
    """Test SVD-robust solve behavior via _dls_solver."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_well_conditioned(self):
        """DLS solver should converge from a good starting point."""
        stop_event = threading.Event()
        theta, success, error = self.solver._dls_solver(
            self.T_desired, self.theta_known, eomg=1e-3, ev=1e-3,
            timeout=5.0, stop_event=stop_event,
        )
        self.assertTrue(success)
        self.assertLess(error, 0.01)

    def test_near_singular(self):
        """DLS should handle configurations near singularities gracefully."""
        # Start from a nearly-singular configuration (extended arm)
        theta_singular = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        stop_event = threading.Event()
        theta, success, error = self.solver._dls_solver(
            self.T_desired, theta_singular, eomg=1e-2, ev=1e-2,
            timeout=2.0, stop_event=stop_event,
        )
        # Should return finite values regardless of success
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_svd_failure_returns_zero(self):
        """If SVD raises LinAlgError, DLS should still return a result via fallback."""
        stop_event = threading.Event()
        # Patch SVD to fail, forcing fallback to normal equations
        with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError):
            theta, success, error = self.solver._dls_solver(
                self.T_desired, self.theta_known, eomg=1e-3, ev=1e-3,
                timeout=0.5, stop_event=stop_event,
            )
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_damping_effect(self):
        """DLS solver uses adaptive damping — verify it doesn't diverge."""
        stop_event = threading.Event()
        T_difficult = np.eye(4)
        T_difficult[:3, 3] = [0.5, 0.5, 0.5]
        theta, success, error = self.solver._dls_solver(
            T_difficult, np.zeros(6), eomg=1e-2, ev=1e-2,
            timeout=2.0, stop_event=stop_event,
        )
        self.assertTrue(np.all(np.isfinite(theta)))
        self.assertTrue(np.all(np.isfinite(error)))


# =====================================================================
# _detect_oscillation
# =====================================================================
class TestDetectOscillation(unittest.TestCase, _RobotFixtureMixin):
    """Tests for _detect_oscillation."""

    def setUp(self):
        self.solver = self._make_solver()

    def test_short_history_returns_false(self):
        history = deque([1.0, 2.0], maxlen=10)
        stag, osc = self.solver._detect_oscillation(history, window_size=5)
        self.assertFalse(stag)
        self.assertFalse(osc)

    def test_stagnation_detected(self):
        """Constant error should be detected as stagnation."""
        history = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], maxlen=10)
        stag, osc = self.solver._detect_oscillation(history, window_size=5)
        self.assertTrue(stag)

    def test_oscillation_detected(self):
        """Alternating error should be detected as oscillation."""
        history = deque([0.5, 0.6, 0.5, 0.6, 0.5, 0.6], maxlen=10)
        stag, osc = self.solver._detect_oscillation(history, window_size=5)
        self.assertTrue(osc)

    def test_decreasing_error_no_flags(self):
        """Steadily decreasing error should not flag anything."""
        history = deque([1.0, 0.8, 0.6, 0.4, 0.2, 0.1], maxlen=10)
        stag, osc = self.solver._detect_oscillation(history, window_size=5)
        self.assertFalse(stag)
        self.assertFalse(osc)

    def test_near_zero_stagnation(self):
        """Very small errors that are constant should stagnate."""
        history = deque([1e-12, 1e-12, 1e-12, 1e-12, 1e-12], maxlen=10)
        stag, _ = self.solver._detect_oscillation(history, window_size=5)
        self.assertTrue(stag)


# =====================================================================
# DLS solver perturbation recovery
# =====================================================================
class TestDLSPerturbationRecovery(unittest.TestCase, _RobotFixtureMixin):
    """Test DLS solver's perturbation recovery from stagnation."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)

    def test_perturbation_limit(self):
        """DLS with max_perturbations=0 should give up sooner."""
        T_far = np.eye(4)
        T_far[:3, 3] = [50.0, 50.0, 50.0]
        stop_event = threading.Event()
        theta, success, error = self.solver._dls_solver(
            T_far, np.zeros(6), eomg=1e-8, ev=1e-8,
            timeout=1.0, stop_event=stop_event, max_perturbations=0,
        )
        self.assertFalse(success)

    def test_perturbation_recovery_attempts(self):
        """DLS with higher max_perturbations should try harder."""
        stop_event = threading.Event()
        T_difficult = np.eye(4)
        T_difficult[:3, 3] = [0.3, 0.3, 0.3]
        theta, success, error = self.solver._dls_solver(
            T_difficult, np.zeros(6), eomg=1e-2, ev=1e-2,
            timeout=2.0, stop_event=stop_event, max_perturbations=5,
        )
        self.assertTrue(np.all(np.isfinite(theta)))


# =====================================================================
# _dls_solver
# =====================================================================
class TestDLSSolver(unittest.TestCase, _RobotFixtureMixin):
    """Tests for _dls_solver."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_dls_converges(self):
        stop_event = threading.Event()
        theta, success, error = self.solver._dls_solver(
            self.T_desired, self.theta_known, eomg=1e-3, ev=1e-3,
            timeout=5.0, stop_event=stop_event,
        )
        self.assertTrue(success)
        self.assertLess(error, 0.01)

    def test_dls_respects_stop_event(self):
        """Should stop early when stop_event is set."""
        stop_event = threading.Event()
        stop_event.set()  # Already signalled
        theta, success, error = self.solver._dls_solver(
            self.T_desired, np.zeros(6), eomg=1e-6, ev=1e-6,
            timeout=10.0, stop_event=stop_event,
        )
        # Should exit immediately without converging
        self.assertFalse(success)

    def test_dls_respects_timeout(self):
        """Should return within reasonable time after timeout."""
        stop_event = threading.Event()
        T_far = np.eye(4)
        T_far[:3, 3] = [50.0, 50.0, 50.0]
        theta, success, error = self.solver._dls_solver(
            T_far, np.zeros(6), eomg=1e-8, ev=1e-8,
            timeout=0.05, stop_event=stop_event,
        )
        self.assertFalse(success)


# =====================================================================
# _sqp_solver
# =====================================================================
class TestSQPSolver(unittest.TestCase, _RobotFixtureMixin):
    """Tests for _sqp_solver."""

    def setUp(self):
        self.robot = self._make_robot()
        self.solver = self._make_solver(self.robot)
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_sqp_converges(self):
        stop_event = threading.Event()
        theta, success, error = self.solver._sqp_solver(
            self.T_desired, self.theta_known, eomg=1e-3, ev=1e-3,
            timeout=5.0, stop_event=stop_event,
        )
        self.assertTrue(success)

    def test_sqp_respects_stop_event(self):
        stop_event = threading.Event()
        stop_event.set()
        theta, success, error = self.solver._sqp_solver(
            self.T_desired, np.zeros(6), eomg=1e-8, ev=1e-8,
            timeout=10.0, stop_event=stop_event,
        )
        # SQP objective returns 1e10 when stopped, so it won't converge
        self.assertFalse(success)


# =====================================================================
# _default_error_func — all three branches
# =====================================================================
class TestDefaultErrorFunc(unittest.TestCase, _RobotFixtureMixin):
    """Tests for _default_error_func."""

    def setUp(self):
        self.solver = self._make_solver()

    def test_identical_poses_zero_error(self):
        """Identical poses should give ~zero error."""
        T = np.eye(4)
        V_err, rot_err, trans_err = self.solver._default_error_func(T, T)
        self.assertAlmostEqual(rot_err, 0.0, places=5)
        self.assertAlmostEqual(trans_err, 0.0, places=5)
        np.testing.assert_array_almost_equal(V_err, np.zeros(6), decimal=5)

    def test_pure_translation_error(self):
        """Only position differs — rotation error should be ~0."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        T_des[:3, 3] = [0.1, 0.2, 0.3]
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertAlmostEqual(rot_err, 0.0, places=5)
        expected_dist = np.linalg.norm([0.1, 0.2, 0.3])
        self.assertAlmostEqual(trans_err, expected_dist, places=5)

    def test_small_rotation(self):
        """Small rotation (angle < 1e-6) — exercises small-angle branch."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        # Tiny rotation around z
        angle = 1e-8
        T_des[:3, :3] = np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle), np.cos(angle), 0],
             [0, 0, 1]]
        )
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertLess(rot_err, 1e-5)

    def test_90_degree_rotation(self):
        """90° rotation — exercises general case branch."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        # 90° rotation around z-axis
        T_des[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertAlmostEqual(rot_err, np.pi / 2, places=3)
        self.assertAlmostEqual(trans_err, 0.0, places=5)

    def test_180_degree_rotation(self):
        """180° rotation — exercises near-pi branch."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        # 180° around z-axis
        T_des[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertAlmostEqual(rot_err, np.pi, places=3)

    def test_180_degree_around_x(self):
        """180° around x — exercises near-pi branch with k=2 (z-axis dominant)."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        T_des[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertAlmostEqual(rot_err, np.pi, places=3)

    def test_180_degree_around_y(self):
        """180° around y — exercises near-pi branch with k=0 or k=2."""
        T_curr = np.eye(4)
        T_des = np.eye(4)
        T_des[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]])
        V_err, rot_err, trans_err = self.solver._default_error_func(T_curr, T_des)
        self.assertAlmostEqual(rot_err, np.pi, places=3)


# =====================================================================
# Delegating helpers (_workspace_heuristic, _midpoint, _random, _clip)
# =====================================================================
class TestDelegatingHelpers(unittest.TestCase, _RobotFixtureMixin):
    """Verify the helpers delegate to ik_helpers correctly."""

    def setUp(self):
        self.solver = self._make_solver()
        self.T_desired, _ = self._reachable_target()

    def test_workspace_heuristic_shape(self):
        guess = self.solver._workspace_heuristic(self.T_desired)
        self.assertEqual(guess.shape, (6,))

    def test_midpoint_guess_shape(self):
        guess = self.solver._midpoint_guess()
        self.assertEqual(guess.shape, (6,))
        # For symmetric limits (-pi, pi), midpoint should be 0
        np.testing.assert_array_almost_equal(guess, np.zeros(6))

    def test_random_guess_within_limits(self):
        for _ in range(10):
            guess = self.solver._random_guess()
            for i, (mn, mx) in enumerate(self.solver.joint_limits):
                if mn is not None:
                    self.assertGreaterEqual(guess[i], mn)
                if mx is not None:
                    self.assertLessEqual(guess[i], mx)

    def test_clip_to_limits(self):
        theta = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        clipped = self.solver._clip_to_limits(theta)
        self.assertAlmostEqual(clipped[0], np.pi)
        self.assertAlmostEqual(clipped[1], -np.pi)


# =====================================================================
# trac_ik_solve convenience function
# =====================================================================
class TestTracIKSolveConvenience(unittest.TestCase, _RobotFixtureMixin):
    """Tests for the module-level trac_ik_solve() function."""

    def setUp(self):
        self.robot = self._make_robot()
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_convenience_function_success(self):
        theta, success, solve_time = trac_ik_solve(
            self.robot, self.T_desired, theta0=self.theta_known, timeout=2.0
        )
        self.assertTrue(success)
        self.assertEqual(theta.shape, (6,))

    def test_convenience_function_no_theta0(self):
        theta, success, solve_time = trac_ik_solve(
            self.robot, self.T_desired, timeout=2.0
        )
        self.assertEqual(theta.shape, (6,))
        self.assertGreater(solve_time, 0.0)


# =====================================================================
# SerialManipulator.trac_ik() integration
# =====================================================================
class TestSerialManipulatorTracIK(unittest.TestCase, _RobotFixtureMixin):
    """Test the trac_ik() method on SerialManipulator."""

    def setUp(self):
        self.robot = self._make_robot()
        self.T_desired, self.theta_known = self._reachable_target(self.robot)

    def test_method_returns_tuple(self):
        result = self.robot.trac_ik(self.T_desired, theta0=self.theta_known, timeout=2.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_method_success(self):
        theta, success, solve_time = self.robot.trac_ik(
            self.T_desired, theta0=self.theta_known, timeout=2.0
        )
        self.assertTrue(success)

    def test_method_with_list_theta0(self):
        """Should accept list as theta0."""
        theta, success, _ = self.robot.trac_ik(
            self.T_desired, theta0=[0.1, 0.2, -0.3, 0.4, -0.5, 0.6], timeout=2.0
        )
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
