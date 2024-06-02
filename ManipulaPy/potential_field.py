#!/usr/bin/env python3

import numpy as np
from numba import cuda
from urchin.urdf import URDF
from scipy.spatial import ConvexHull



# Import CUDA kernel functions (assuming these are defined in cuda_kernels.py)
from .cuda_kernels import attractive_potential_kernel, repulsive_potential_kernel, gradient_kernel

class PotentialField:
    def __init__(self, attractive_gain=1.0, repulsive_gain=100.0, influence_distance=0.5):
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
                repulsive_potential += 0.5 * self.repulsive_gain * (1.0 / d - 1.0 / self.influence_distance) ** 2
        return repulsive_potential

    def compute_gradient(self, q, q_goal, obstacles):
        """
        Compute the gradient of the potential field.
        """
        # Compute attractive gradient
        attractive_gradient = self.attractive_gain * (q - q_goal)

        # Compute repulsive gradient
        repulsive_gradient = np.zeros_like(q)
        for obstacle in obstacles:
            d = np.linalg.norm(q - obstacle)
            if d <= self.influence_distance:
                repulsive_gradient += self.repulsive_gain * (1.0 / d - 1.0 / self.influence_distance) * (1.0 / (d ** 3)) * (q - obstacle)

        # Total gradient
        total_gradient = attractive_gradient + repulsive_gradient
        return total_gradient

class CollisionChecker:
    def __init__(self, urdf_path):
        """
        Initializes a CollisionChecker object.

        Args:
            urdf_path (str): The path to the URDF file.
        """
        self.robot = URDF.load(urdf_path)
        self.convex_hulls = self._create_convex_hulls()

    def _create_convex_hulls(self):
        """
        Creates a dictionary of convex hulls for each visual mesh in the robot's links.

        Returns:
            dict: A dictionary where the keys are the names of the robot links and the values are the corresponding convex hulls.
        """
        convex_hulls = {}
        for link in self.robot.links:
            if link.visuals:
                for visual in link.visuals:
                    mesh = visual.geometry.mesh
                    if hasattr(mesh, 'vertices'):
                        vertices = np.array(mesh.vertices)
                        convex_hull = ConvexHull(vertices)
                        convex_hulls[link.name] = convex_hull
        return convex_hulls

    def _transform_convex_hull(self, convex_hull, transform):
        transformed_points = transform[:3, :3] @ convex_hull.points.T + transform[:3, 3].reshape(-1, 1)
        return ConvexHull(transformed_points.T)

    def check_collision(self, thetalist):
        fk_results = self.robot.link_fk(cfg=thetalist)
        for link_name, transform in fk_results.items():
            if link_name in self.convex_hulls:
                transformed_hull = self._transform_convex_hull(self.convex_hulls[link_name], transform)
                for other_link_name, other_transform in fk_results.items():
                    if link_name != other_link_name and other_link_name in self.convex_hulls:
                        other_transformed_hull = self._transform_convex_hull(self.convex_hulls[other_link_name], other_transform)
                        if transformed_hull.intersects(other_transformed_hull):
                            return True
        return False
