#!/usr/bin/env python3

"""
Extended tests for ik_helpers module — fills coverage gaps.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest

import numpy as np

from ManipulaPy.ik_helpers import (
    IKInitialGuessCache,
    _clip_to_limits,
    adaptive_multi_start_ik,
    midpoint_of_limits,
    random_in_limits,
    workspace_heuristic_guess,
)


class TestWorkspaceHeuristicBranches(unittest.TestCase):
    """Cover all branch conditions in workspace_heuristic_guess."""

    def _limits(self, n):
        return [(-np.pi, np.pi)] * n

    def test_1_joint(self):
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.2, 0.4]
        theta = workspace_heuristic_guess(T, 1, self._limits(1))
        self.assertEqual(theta.shape, (1,))
        self.assertAlmostEqual(theta[0], np.arctan2(0.2, 0.3), places=5)

    def test_2_joints(self):
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.2, 0.4]
        theta = workspace_heuristic_guess(T, 2, self._limits(2))
        self.assertEqual(theta.shape, (2,))

    def test_3_joints(self):
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.2, 0.4]
        theta = workspace_heuristic_guess(T, 3, self._limits(3))
        self.assertAlmostEqual(theta[2], np.pi / 4)

    def test_4_joints_normal(self):
        """4 joints, non-gimbal-lock rotation."""
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.2, 0.4]
        # Rotation with R[2,2] != ±1
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        theta = workspace_heuristic_guess(T, 4, self._limits(4))
        self.assertEqual(theta.shape, (4,))

    def test_5_joints_normal(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        theta = workspace_heuristic_guess(T, 5, self._limits(5))
        self.assertEqual(theta.shape, (5,))

    def test_6_joints_gimbal_lock(self):
        """R[2,2] ≈ 1 triggers gimbal lock branch."""
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.3]
        # Identity rotation → R[2,2] = 1 → gimbal lock
        theta = workspace_heuristic_guess(T, 6, self._limits(6))
        self.assertEqual(theta.shape, (6,))
        # In gimbal lock, theta[4] and theta[5] should be 0
        self.assertAlmostEqual(theta[4], 0.0)
        self.assertAlmostEqual(theta[5], 0.0)

    def test_xy_at_origin(self):
        """r_xy < 1e-6 triggers else branch for joint 2."""
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.0, 1.0]  # Directly above → r_xy = 0
        theta = workspace_heuristic_guess(T, 6, self._limits(6))
        self.assertAlmostEqual(theta[1], 0.0)


class TestRandomInLimitsBranches(unittest.TestCase):
    """Cover all branches in random_in_limits."""

    def test_both_limits(self):
        limits = [(-1.0, 1.0)]
        theta = random_in_limits(limits)
        self.assertGreaterEqual(theta[0], -1.0)
        self.assertLessEqual(theta[0], 1.0)

    def test_only_min(self):
        limits = [(-1.0, None)]
        theta = random_in_limits(limits)
        self.assertGreaterEqual(theta[0], -1.0)

    def test_only_max(self):
        limits = [(None, 1.0)]
        theta = random_in_limits(limits)
        self.assertLessEqual(theta[0], 1.0)

    def test_no_limits(self):
        limits = [(None, None)]
        theta = random_in_limits(limits)
        self.assertGreaterEqual(theta[0], -np.pi)
        self.assertLessEqual(theta[0], np.pi)


class TestMidpointOfLimits(unittest.TestCase):
    def test_none_limits_stay_zero(self):
        limits = [(None, None), (None, 2.0), (-1.0, None)]
        theta = midpoint_of_limits(limits)
        self.assertAlmostEqual(theta[0], 0.0)
        self.assertAlmostEqual(theta[1], 0.0)
        self.assertAlmostEqual(theta[2], 0.0)

    def test_empty_limits(self):
        theta = midpoint_of_limits([])
        self.assertEqual(len(theta), 0)


class TestClipToLimits(unittest.TestCase):
    def test_clips_min(self):
        theta = np.array([-5.0])
        clipped = _clip_to_limits(theta, [(-1.0, 1.0)])
        self.assertAlmostEqual(clipped[0], -1.0)

    def test_clips_max(self):
        theta = np.array([5.0])
        clipped = _clip_to_limits(theta, [(-1.0, 1.0)])
        self.assertAlmostEqual(clipped[0], 1.0)

    def test_none_limits_no_clip(self):
        theta = np.array([100.0])
        clipped = _clip_to_limits(theta, [(None, None)])
        self.assertAlmostEqual(clipped[0], 100.0)

    def test_theta_longer_than_limits(self):
        """More joints than limits should clip only what's available."""
        theta = np.array([5.0, 5.0, 5.0])
        clipped = _clip_to_limits(theta, [(-1.0, 1.0)])
        self.assertAlmostEqual(clipped[0], 1.0)
        self.assertAlmostEqual(clipped[1], 5.0)  # Untouched

    def test_limits_longer_than_theta(self):
        """bounds check: i < len(theta_clipped)."""
        theta = np.array([5.0])
        clipped = _clip_to_limits(theta, [(-1.0, 1.0), (-2.0, 2.0)])
        self.assertAlmostEqual(clipped[0], 1.0)


class TestIKInitialGuessCacheExtended(unittest.TestCase):
    """Extended cache tests for uncovered branches."""

    def test_get_nearest_empty_cache(self):
        cache = IKInitialGuessCache()
        self.assertIsNone(cache.get_nearest(np.eye(4)))

    def test_fifo_eviction(self):
        cache = IKInitialGuessCache(max_size=3)
        for i in range(5):
            T = np.eye(4)
            T[0, 3] = float(i)
            cache.add(T, np.array([float(i)]))
        self.assertEqual(cache.size(), 3)

    def test_get_nearest_with_residual(self):
        """Low-residual entries should be preferred."""
        cache = IKInitialGuessCache()
        T = np.eye(4)
        cache.add(T, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), residual=1e-4)
        result = cache.get_nearest(T, k=1, joint_limits=[(-np.pi, np.pi)] * 6)
        self.assertIsNotNone(result)

    def test_get_nearest_high_quality_direct_return(self):
        """If best_quality < 1e-3, return directly without averaging."""
        cache = IKInitialGuessCache()
        T = np.eye(4)
        theta = np.array([0.1, 0.2, 0.3])
        cache.add(T, theta, residual=1e-5)
        result = cache.get_nearest(T, k=3)
        np.testing.assert_array_almost_equal(result, theta)

    def test_get_nearest_averaging(self):
        """Multiple entries with high residual should be averaged."""
        cache = IKInitialGuessCache()
        T = np.eye(4)
        cache.add(T, np.array([1.0, 0.0, 0.0]), residual=0.5)
        cache.add(T, np.array([0.0, 1.0, 0.0]), residual=0.5)
        cache.add(T, np.array([0.0, 0.0, 1.0]), residual=0.5)
        result = cache.get_nearest(T, k=3)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3,))

    def test_get_nearest_no_joint_limits(self):
        """Without joint_limits, no clipping should occur."""
        cache = IKInitialGuessCache()
        T = np.eye(4)
        cache.add(T, np.array([100.0]), residual=0.5)
        result = cache.get_nearest(T, k=1, joint_limits=None)
        self.assertIsNotNone(result)

    def test_clear(self):
        cache = IKInitialGuessCache()
        cache.add(np.eye(4), np.zeros(3))
        cache.clear()
        self.assertEqual(cache.size(), 0)

    def test_pose_distance(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 1.0
        dist = IKInitialGuessCache._pose_distance(T1, T2)
        self.assertAlmostEqual(dist, 1.0, places=5)


class TestAdaptiveMultiStartIK(unittest.TestCase):
    """Test adaptive_multi_start_ik with mock solver."""

    def test_success_on_first_attempt(self):
        """Mock solver succeeds immediately."""
        mock_solver = lambda T, **kw: (np.zeros(6), True, 10)
        theta, success, iters, strategy = adaptive_multi_start_ik(
            mock_solver, np.eye(4), max_attempts=3
        )
        self.assertTrue(success)
        self.assertEqual(iters, 10)

    def test_all_attempts_fail(self):
        """Mock solver always fails — should return best solution."""
        mock_solver = lambda T, **kw: (np.ones(6), False, 100)
        theta, success, iters, strategy = adaptive_multi_start_ik(
            mock_solver, np.eye(4), max_attempts=3
        )
        self.assertFalse(success)
        self.assertEqual(strategy, "none (failed)")
        self.assertEqual(iters, 300)

    def test_exception_in_solver(self):
        """Exception in solver should be caught and skipped."""
        call_count = [0]

        def failing_solver(T, **kw):
            call_count[0] += 1
            raise RuntimeError("solver error")

        theta, success, iters, strategy = adaptive_multi_start_ik(
            failing_solver, np.eye(4), max_attempts=3
        )
        self.assertFalse(success)
        self.assertEqual(call_count[0], 3)

    def test_all_exceptions_returns_zeros(self):
        """If best_solution is None (all exceptions), should return midpoint of []."""
        def always_fails(T, **kw):
            raise RuntimeError("fail")

        theta, success, iters, strategy = adaptive_multi_start_ik(
            always_fails, np.eye(4), max_attempts=2
        )
        self.assertFalse(success)
        self.assertIsNotNone(theta)

    def test_verbose_mode(self, ):
        """Verbose mode should not crash."""
        mock_solver = lambda T, **kw: (np.zeros(6), False, 10)
        theta, success, iters, strategy = adaptive_multi_start_ik(
            mock_solver, np.eye(4), max_attempts=2, verbose=True
        )
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
