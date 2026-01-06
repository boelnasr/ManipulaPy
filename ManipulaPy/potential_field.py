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
import numpy as np
from scipy.spatial import ConvexHull

from .urdf import URDF  # Use native parser


# Import CUDA kernel functions (assuming these are defined in cuda_kernels.py)


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
                repulsive_potential += (
                    2
                    * self.repulsive_gain
                    * (1.0 / d - 1.0 / self.influence_distance) ** 2
                )
        return 10*repulsive_potential

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
            d = np.linalg.norm(q - obstacle)
            if d <= self.influence_distance:
                repulsive_gradient += (
                    -40*self.repulsive_gain
                    * (1.0 / d - 1.0 / self.influence_distance)
                    * (1.0 / (d**3))
                    * (q - obstacle)
                )

        # Total gradient
        total_gradient = attractive_gradient + repulsive_gradient
        return total_gradient


class CollisionChecker:
    """
    Collision checker using URDF visual/collision geometry and convex hulls.

    Supports multiple URDF parser backends:
        - "builtin": Native ManipulaPy parser (NumPy 2.0 compatible, default)
        - "urchin": Legacy urchin parser (requires urchin, not NumPy 2.0 compatible)
        - "pybullet": PyBullet-based parser (requires pybullet)
    """

    def __init__(self, urdf_path, backend: str = "builtin", load_meshes: bool = True):
        """
        Initializes a CollisionChecker object.

        Args:
            urdf_path (str): The path to the URDF file.
            backend (str): Parser backend - "builtin" (default), "urchin", or "pybullet"
            load_meshes (bool): Whether to load mesh geometry data (default: True)
        """
        self.robot = URDF.load(urdf_path, backend=backend, load_meshes=load_meshes)
        self.convex_hulls = self._create_convex_hulls()

    def _create_convex_hulls(self):
        """
        Creates a dictionary of convex hulls for each visual mesh in the robot's links.

        Returns:
            dict: A dictionary where the keys are the names of the robot links
                  and the values are the corresponding convex hulls.
        """
        convex_hulls = {}
        for link in self.robot.links:
            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry is None:
                        continue

                    # Handle different geometry types
                    geom = visual.geometry

                    # Check if it's a mesh with loaded vertices
                    if hasattr(geom, 'mesh_data') and geom.mesh_data is not None:
                        vertices = geom.mesh_data.vertices
                        if vertices is not None and len(vertices) >= 4:
                            try:
                                convex_hull = ConvexHull(vertices)
                                convex_hulls[link.name] = convex_hull
                            except Exception:
                                # Skip if convex hull fails (degenerate geometry)
                                pass
                    # Fallback for legacy mesh attribute
                    elif hasattr(geom, 'mesh') and geom.mesh is not None:
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
        """
        Transform convex hull vertices by a 4x4 transformation matrix.

        Args:
            convex_hull: ConvexHull object
            transform: 4x4 transformation matrix

        Returns:
            ConvexHull: Transformed convex hull
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
        # Use native parser's link_fk with use_names=True for string keys
        fk_results = self.robot.link_fk(cfg=thetalist, use_names=True)

        for link_name, transform in fk_results.items():
            if link_name in self.convex_hulls:
                transformed_hull = self._transform_convex_hull(
                    self.convex_hulls[link_name], transform
                )
                for other_link_name, other_transform in fk_results.items():
                    if (
                        link_name != other_link_name
                        and other_link_name in self.convex_hulls
                    ):
                        other_transformed_hull = self._transform_convex_hull(
                            self.convex_hulls[other_link_name], other_transform
                        )
                        # Check intersection using convex hull overlap
                        if self._hulls_intersect(transformed_hull, other_transformed_hull):
                            return True
        return False

    def _hulls_intersect(self, hull1, hull2):
        """
        Check if two convex hulls intersect using separating axis theorem.

        This is a simplified check - for production use, consider using
        a proper collision detection library like fcl or trimesh.

        Args:
            hull1: First ConvexHull
            hull2: Second ConvexHull

        Returns:
            bool: True if hulls potentially intersect
        """
        # Simple bounding box overlap check
        min1 = np.min(hull1.points, axis=0)
        max1 = np.max(hull1.points, axis=0)
        min2 = np.min(hull2.points, axis=0)
        max2 = np.max(hull2.points, axis=0)

        # Check if bounding boxes overlap
        return np.all(max1 >= min2) and np.all(max2 >= min1)
