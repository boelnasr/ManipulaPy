#!/usr/bin/env python3

"""
Extended tests for potential_field module — fills coverage gaps.

Covers: CollisionChecker with mock URDF, _transform_convex_hull,
check_collision, _hulls_intersect, and _create_convex_hulls branches.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
from scipy.spatial import ConvexHull

from ManipulaPy.potential_field import PotentialField


class TestPotentialFieldExtended(unittest.TestCase):
    """Additional coverage for PotentialField gradient and edge cases."""

    def test_repulsive_outside_influence(self):
        """Obstacle outside influence distance → 0 repulsive potential."""
        pf = PotentialField(influence_distance=0.5)
        q = np.array([0.0, 0.0])
        obstacle = np.array([10.0, 10.0])
        pot = pf.compute_repulsive_potential(q, [obstacle])
        self.assertAlmostEqual(pot, 0.0)

    def test_gradient_far_obstacle(self):
        """Far obstacle contributes no repulsive gradient."""
        pf = PotentialField(influence_distance=0.5)
        q = np.array([0.0, 0.0])
        q_goal = np.array([1.0, 1.0])
        obstacle = np.array([10.0, 10.0])
        grad = pf.compute_gradient(q, q_goal, [obstacle])
        # Should be purely attractive
        expected = pf.attractive_gain * (q - q_goal)
        np.testing.assert_array_almost_equal(grad, expected)

    def test_gradient_near_obstacle(self):
        """Close obstacle adds repulsive gradient component."""
        pf = PotentialField(influence_distance=2.0, repulsive_gain=100.0)
        q = np.array([0.1, 0.0])
        q_goal = np.array([1.0, 0.0])
        obstacle = np.array([0.0, 0.0])
        grad_with = pf.compute_gradient(q, q_goal, [obstacle])
        grad_without = pf.compute_gradient(q, q_goal, [])
        # The repulsive gradient should differ from the purely attractive one
        self.assertGreater(np.linalg.norm(grad_with - grad_without), 0.1)

    def test_multiple_obstacles(self):
        """Multiple obstacles within influence."""
        pf = PotentialField(influence_distance=5.0)
        q = np.array([0.0, 0.0])
        obstacles = [np.array([0.1, 0.0]), np.array([0.0, 0.1])]
        pot = pf.compute_repulsive_potential(q, obstacles)
        self.assertGreater(pot, 0.0)


class TestCollisionCheckerMocked(unittest.TestCase):
    """Test CollisionChecker with mocked URDF objects."""

    def _make_convex_hull(self, offset=None):
        """Create a small tetrahedron convex hull."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if offset is not None:
            pts += offset
        return ConvexHull(pts)

    def test_hulls_intersect_overlapping(self):
        """Two overlapping hulls should intersect."""
        from ManipulaPy.potential_field import CollisionChecker

        hull1 = self._make_convex_hull()
        hull2 = self._make_convex_hull(offset=np.array([0.5, 0.5, 0.5]))

        # Create a minimal CollisionChecker with mocked URDF
        mock_robot = MagicMock()
        mock_robot.links = []

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot
            checker.convex_hulls = {}

        self.assertTrue(checker._hulls_intersect(hull1, hull2))

    def test_hulls_intersect_separated(self):
        """Two well-separated hulls should not intersect."""
        from ManipulaPy.potential_field import CollisionChecker

        hull1 = self._make_convex_hull()
        hull2 = self._make_convex_hull(offset=np.array([10.0, 10.0, 10.0]))

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = MagicMock()
            checker.convex_hulls = {}

        self.assertFalse(checker._hulls_intersect(hull1, hull2))

    def test_transform_convex_hull(self):
        """Transformation should shift hull vertices."""
        from ManipulaPy.potential_field import CollisionChecker

        hull = self._make_convex_hull()
        T = np.eye(4)
        T[:3, 3] = [5.0, 0.0, 0.0]  # Translate by 5 in x

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)

        transformed = checker._transform_convex_hull(hull, T)
        # All x coordinates should be >= 5.0
        self.assertGreater(np.min(transformed.points[:, 0]), 4.9)

    def test_check_collision_with_mock_fk(self):
        """check_collision with mocked link_fk and convex hulls."""
        from ManipulaPy.potential_field import CollisionChecker

        hull = self._make_convex_hull()

        mock_robot = MagicMock()
        mock_robot.links = []
        # Overlapping transforms → collision
        mock_robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        }

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot
            checker.convex_hulls = {"link_a": hull, "link_b": hull}

        self.assertTrue(checker.check_collision(np.zeros(6)))

    def test_check_collision_no_collision(self):
        """Separated links → no collision."""
        from ManipulaPy.potential_field import CollisionChecker

        hull1 = self._make_convex_hull()
        hull2 = self._make_convex_hull()

        T_far = np.eye(4)
        T_far[:3, 3] = [100.0, 100.0, 100.0]

        mock_robot = MagicMock()
        mock_robot.links = []
        mock_robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": T_far,
        }

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot
            checker.convex_hulls = {"link_a": hull1, "link_b": hull2}

        self.assertFalse(checker.check_collision(np.zeros(6)))

    def test_check_collision_missing_hull(self):
        """Links without convex hulls should be skipped."""
        from ManipulaPy.potential_field import CollisionChecker

        mock_robot = MagicMock()
        mock_robot.links = []
        mock_robot.link_fk.return_value = {
            "link_a": np.eye(4),
            "link_b": np.eye(4),
        }

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot
            checker.convex_hulls = {}  # No hulls

        self.assertFalse(checker.check_collision(np.zeros(6)))

    def test_create_convex_hulls_mesh_data(self):
        """_create_convex_hulls with mesh_data attribute."""
        from ManipulaPy.potential_field import CollisionChecker

        # Create mock URDF with mesh_data
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        ).astype(float)

        mock_geom = MagicMock()
        mock_geom.mesh_data = MagicMock()
        mock_geom.mesh_data.vertices = vertices

        mock_visual = MagicMock()
        mock_visual.geometry = mock_geom

        mock_link = MagicMock()
        mock_link.name = "test_link"
        mock_link.visuals = [mock_visual]

        mock_robot = MagicMock()
        mock_robot.links = [mock_link]

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot

        hulls = checker._create_convex_hulls()
        self.assertIn("test_link", hulls)

    def test_create_convex_hulls_legacy_mesh(self):
        """_create_convex_hulls with legacy mesh attribute."""
        from ManipulaPy.potential_field import CollisionChecker

        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        ).astype(float)

        mock_mesh = MagicMock()
        mock_mesh.vertices = vertices

        mock_geom = MagicMock(spec=[])  # No mesh_data attr
        mock_geom.mesh = mock_mesh

        mock_visual = MagicMock()
        mock_visual.geometry = mock_geom

        mock_link = MagicMock()
        mock_link.name = "legacy_link"
        mock_link.visuals = [mock_visual]

        mock_robot = MagicMock()
        mock_robot.links = [mock_link]

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot

        hulls = checker._create_convex_hulls()
        self.assertIn("legacy_link", hulls)

    def test_create_convex_hulls_no_geometry(self):
        """Visual with geometry=None should be skipped."""
        from ManipulaPy.potential_field import CollisionChecker

        mock_visual = MagicMock()
        mock_visual.geometry = None

        mock_link = MagicMock()
        mock_link.name = "empty_link"
        mock_link.visuals = [mock_visual]

        mock_robot = MagicMock()
        mock_robot.links = [mock_link]

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot

        hulls = checker._create_convex_hulls()
        self.assertEqual(len(hulls), 0)

    def test_create_convex_hulls_no_visuals(self):
        """Link with no visuals should be skipped."""
        from ManipulaPy.potential_field import CollisionChecker

        mock_link = MagicMock()
        mock_link.name = "no_vis_link"
        mock_link.visuals = []

        mock_robot = MagicMock()
        mock_robot.links = [mock_link]

        with patch.object(CollisionChecker, "__init__", lambda self, *a, **kw: None):
            checker = CollisionChecker.__new__(CollisionChecker)
            checker.robot = mock_robot

        hulls = checker._create_convex_hulls()
        self.assertEqual(len(hulls), 0)


if __name__ == "__main__":
    unittest.main()
