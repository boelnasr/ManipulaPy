#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Potential Field Module - ManipulaPy

This module provides potential field path planning capabilities including attractive
and repulsive potential computations, gradient calculations, and collision checking
for robotic manipulator motion planning in cluttered environments.

Copyright (c) 2025 Mohamed Aboelnasr

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""
import itertools
import logging

import numpy as np
from scipy.spatial import ConvexHull

from .urdf import URDF  # Use native parser

_logger = logging.getLogger(__name__)


def build_link_adjacency(urdf, exclude_grandparents: bool = True) -> set:
    """Pairs of link names excluded from self-collision checks.

    Always includes parent<->child pairs joined by any URDF joint. Optionally
    extends to grandparent<->grandchild, matching SRDF convention for arms whose
    successive-link geometry overlaps slightly even when not coincident.

    Returns a set of frozenset({name_a, name_b}) so callers can do
    order-independent ``frozenset(pair) in acm`` checks.
    """
    excluded = set()
    parent_of = {j.child: j.parent for j in urdf.joints}
    for j in urdf.joints:
        excluded.add(frozenset({j.parent, j.child}))
    if exclude_grandparents:
        for child, parent in parent_of.items():
            grandparent = parent_of.get(parent)
            if grandparent:
                excluded.add(frozenset({grandparent, child}))
    return excluded


class PotentialField:
    def __init__(
        self, attractive_gain=1.0, repulsive_gain=100.0, influence_distance=0.5
    ):
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance

    def compute_attractive_potential(self, q, q_goal):
        """
        Compute the attractive potential.
        """
        return 0.5 * self.attractive_gain * np.sum((q - q_goal) ** 2)

    def compute_repulsive_potential(self, q, obstacles):
        """
        Compute the repulsive potential.
        """
        repulsive_potential = 0
        for obstacle in obstacles:
            d = np.linalg.norm(q - obstacle)
            if d <= self.influence_distance:
                d_safe = max(d, 1e-10)
                repulsive_potential += (
                    2
                    * self.repulsive_gain
                    * (1.0 / d_safe - 1.0 / self.influence_distance) ** 2
                )
        return 10 * repulsive_potential

    def compute_gradient(self, q, q_goal, obstacles):
        """
        Compute the gradient of the potential field.
        """
        # Compute attractive gradient
        attractive_gradient = self.attractive_gain * (q - q_goal)

        # Compute repulsive gradient
        # Derivative of: 10 * 2 * gain * (1/d - 1/d0)^2
        # = 10 * 2 * gain * 2 * (1/d - 1/d0) * d(1/d)/dq
        # = 40 * gain * (1/d - 1/d0) * (-(q-obs)/d^3)
        repulsive_gradient = np.zeros_like(q)
        for obstacle in obstacles:
            diff = q - obstacle
            d = np.linalg.norm(diff)
            if d <= self.influence_distance:
                d_safe = max(d, 1e-10)
                # If q is exactly at obstacle, pick an arbitrary escape direction
                # (otherwise robot is stuck because gradient is zero)
                if d < 1e-10:
                    direction = np.zeros_like(q)
                    direction[0] = 1.0  # Escape along +x (arbitrary but consistent)
                    repulsive_gradient += self.repulsive_gain * direction
                    continue
                else:
                    direction = diff
                repulsive_gradient += (
                    -40
                    * self.repulsive_gain
                    * (1.0 / d_safe - 1.0 / self.influence_distance)
                    * (1.0 / (d_safe**3))
                    * direction
                )

        # Total gradient
        total_gradient = attractive_gradient + repulsive_gradient
        return total_gradient


class CollisionChecker:
    """
    Collision checker using URDF visual/collision geometry and convex hulls.

    Supports multiple URDF parser backends:
        - "builtin": Native ManipulaPy parser (NumPy 2.0 compatible, default)
        - "pybullet": PyBullet-based parser (requires pybullet)
    """

    def __init__(self, urdf_path, backend: str = "builtin", load_meshes: bool = True):
        """
        Initializes a CollisionChecker object.

        Args:
            urdf_path (str): The path to the URDF file.
            backend (str): Parser backend - "builtin" (default) or "pybullet"
            load_meshes (bool): Whether to load mesh geometry data (default: True)
        """
        self.robot = URDF.load(urdf_path, backend=backend, load_meshes=load_meshes)
        # ACM derived from URDF topology to avoid adjacent-link false positives
        self._acm = build_link_adjacency(self.robot, exclude_grandparents=True)
        self._visual_fallback_warned = set()
        self.convex_hulls = self._create_convex_hulls()

    def _warn_visual_fallback_once(self, link_name: str) -> None:
        if link_name not in self._visual_fallback_warned:
            self._visual_fallback_warned.add(link_name)
            _logger.warning(
                "Link %r has no collision geometry; falling back to visual geometry "
                "for collision checking — results may be inaccurate.",
                link_name,
            )

    def _create_convex_hulls(self):
        """
        Creates a dictionary of convex hulls for each link, preferring collision
        geometry and falling back to visual geometry with a one-shot warning.

        Returns:
            dict: A dictionary where the keys are the names of the robot links
                  and the values are the corresponding convex hulls.
        """
        convex_hulls = {}
        for link in self.robot.links:
            # Prefer collision geometry; fall back to visuals with a warning
            sources = link.collisions if link.collisions else None
            if sources is None and link.visuals:
                self._warn_visual_fallback_once(link.name)
                sources = link.visuals
            if not sources:
                continue

            for geom_element in sources:
                if geom_element.geometry is None:
                    continue

                geom = geom_element.geometry

                # Check if it's a mesh with loaded vertices
                if hasattr(geom, "mesh_data") and geom.mesh_data is not None:
                    vertices = geom.mesh_data.vertices
                    if vertices is not None and len(vertices) >= 4:
                        try:
                            convex_hull = ConvexHull(vertices)
                            convex_hulls[link.name] = convex_hull
                        except Exception:
                            pass
                # Fallback for legacy mesh attribute
                elif hasattr(geom, "mesh") and geom.mesh is not None:
                    mesh = geom.mesh
                    if hasattr(mesh, "vertices") and mesh.vertices is not None:
                        vertices = np.array(mesh.vertices)
                        if len(vertices) >= 4:
                            try:
                                convex_hull = ConvexHull(vertices)
                                convex_hulls[link.name] = convex_hull
                            except Exception:
                                pass

        return convex_hulls

    def _transform_convex_hull(self, convex_hull, transform):
        """Apply a 4x4 transform to a ConvexHull and return a NEW ConvexHull.

        Retained for backwards compatibility with existing tests
        (test_potential_field_coverage.py, test_potential_field_extended.py)
        that call this method directly. Internal callers in check_collision
        transform cached vertices directly via matrix multiply and skip the
        ConvexHull rebuild — prefer that path for new code.
        """
        transformed_points = transform[:3, :3] @ convex_hull.points.T + transform[
            :3, 3
        ].reshape(-1, 1)
        return ConvexHull(transformed_points.T)

    def check_collision(self, thetalist):
        """
        Check for self-collision at a given joint configuration.

        Args:
            thetalist: Joint configuration (array or dict)

        Returns:
            bool: True if collision detected, False otherwise
        """
        fk_results = self.robot.link_fk(cfg=thetalist, use_names=True)

        hull_names = [n for n in self.convex_hulls if n in fk_results]
        for name_a, name_b in itertools.combinations(hull_names, 2):
            # Skip adjacent-link pairs — they always overlap at joints
            if frozenset({name_a, name_b}) in self._acm:
                continue

            T_a = fk_results[name_a]
            T_b = fk_results[name_b]
            pts_a = (T_a[:3, :3] @ self.convex_hulls[name_a].points.T + T_a[:3, 3:4]).T
            pts_b = (T_b[:3, :3] @ self.convex_hulls[name_b].points.T + T_b[:3, 3:4]).T
            if self._points_intersect(pts_a, pts_b):
                return True
        return False

    def _points_intersect(self, pts_a, pts_b):
        """
        Check if two point clouds' bounding boxes intersect.

        This is a simplified check — for production use, consider a proper
        collision detection library like fcl or trimesh.

        Args:
            pts_a: (N, 3) array of points
            pts_b: (M, 3) array of points

        Returns:
            bool: True if bounding boxes overlap
        """
        min_a = np.min(pts_a, axis=0)
        max_a = np.max(pts_a, axis=0)
        min_b = np.min(pts_b, axis=0)
        max_b = np.max(pts_b, axis=0)
        return bool(np.all(max_a >= min_b) and np.all(max_b >= min_a))

    def _hulls_intersect(self, hull1, hull2):
        """
        Check if two convex hulls intersect.

        Thin wrapper around _points_intersect for backwards compatibility
        with external callers that pass ConvexHull objects.

        Args:
            hull1: First ConvexHull
            hull2: Second ConvexHull

        Returns:
            bool: True if hulls potentially intersect
        """
        return self._points_intersect(hull1.points, hull2.points)
