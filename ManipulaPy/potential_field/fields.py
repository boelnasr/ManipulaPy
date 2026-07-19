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

import itertools  # noqa: F401
import logging
from typing import Any, Dict, Iterable, Set  # noqa: F401

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull  # noqa: F401

from ..backend import get_backend
from ..urdf import URDF  # noqa: F401

_logger = logging.getLogger(__name__)


class PotentialField:
    """Artificial potential field for attractive and repulsive joint-space costs."""

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 100.0,
        influence_distance: float = 0.5,
    ) -> None:
        """
        Initialize potential gains and obstacle influence distance.

        Args:
            attractive_gain: Weight for attraction toward the goal.
            repulsive_gain: Weight for obstacle repulsion.
            influence_distance: Distance threshold for obstacle repulsion.
        """
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance

    def compute_attractive_potential(
        self, q: NDArray[np.float64], q_goal: NDArray[np.float64]
    ) -> float:
        """
        Compute the attractive potential.
        """
        backend = get_backend()
        q = backend.asarray(q)
        q_goal = backend.asarray(q_goal)
        diff = (q - q_goal) * 1.0
        math_dtype = diff.dtype
        half = backend.asarray(0.5, dtype=math_dtype)
        attractive_gain = backend.asarray(self.attractive_gain, dtype=math_dtype)
        return half * attractive_gain * backend.sum(diff**2)

    def compute_repulsive_potential(
        self, q: NDArray[np.float64], obstacles: Iterable[NDArray[np.float64]]
    ) -> float:
        """
        Compute the repulsive potential.
        """
        backend = get_backend()
        q = backend.asarray(q)
        repulsive_potential = 0
        has_obstacles = False
        for obstacle in obstacles:
            has_obstacles = True
            obstacle = backend.asarray(obstacle)
            diff = (q - obstacle) * 1.0
            d = backend.norm(diff)
            math_dtype = d.dtype
            zero = backend.asarray(0.0, dtype=math_dtype)
            one = backend.asarray(1.0, dtype=math_dtype)
            two = backend.asarray(2.0, dtype=math_dtype)
            epsilon = backend.asarray(1e-10, dtype=math_dtype)
            repulsive_gain = backend.asarray(self.repulsive_gain, dtype=math_dtype)
            influence_distance = backend.asarray(
                self.influence_distance, dtype=math_dtype
            )
            influence_safe = backend.maximum(influence_distance, epsilon)
            d_safe = backend.maximum(d, epsilon)
            contribution = (
                two * repulsive_gain * (one / d_safe - one / influence_safe) ** 2
            )
            repulsive_potential = repulsive_potential + backend.where(
                d <= influence_distance, contribution, zero
            )
        if not has_obstacles:
            return 0
        ten = backend.asarray(10.0, dtype=repulsive_potential.dtype)
        return ten * repulsive_potential

    def compute_gradient(
        self,
        q: NDArray[np.float64],
        q_goal: NDArray[np.float64],
        obstacles: Iterable[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Compute the gradient of the potential field.
        """
        backend = get_backend()
        q = backend.asarray(q)
        q_goal = backend.asarray(q_goal)
        attractive_diff = (q - q_goal) * 1.0
        attractive_dtype = attractive_diff.dtype
        attractive_gain = backend.asarray(self.attractive_gain, dtype=attractive_dtype)

        # Compute attractive gradient
        attractive_gradient = attractive_gain * attractive_diff

        # Compute repulsive gradient
        # Derivative of: 10 * 2 * gain * (1/d - 1/d0)^2
        # = 10 * 2 * gain * 2 * (1/d - 1/d0) * d(1/d)/dq
        # = 40 * gain * (1/d - 1/d0) * (-(q-obs)/d^3)
        repulsive_gradient = backend.zeros(q.shape, dtype=attractive_dtype)
        for obstacle in obstacles:
            obstacle = backend.asarray(obstacle)
            diff = (q - obstacle) * 1.0
            d = backend.norm(diff)
            math_dtype = d.dtype
            d = backend.asarray(d, dtype=math_dtype)
            zero = backend.asarray(0.0, dtype=math_dtype)
            one = backend.asarray(1.0, dtype=math_dtype)
            forty = backend.asarray(40.0, dtype=math_dtype)
            epsilon = backend.asarray(1e-10, dtype=math_dtype)
            repulsive_gain = backend.asarray(self.repulsive_gain, dtype=math_dtype)
            influence_distance = backend.asarray(
                self.influence_distance, dtype=math_dtype
            )
            escape_direction = backend.asarray(
                [1.0] + [0.0] * (q.shape[0] - 1), dtype=math_dtype
            )
            influence_safe = backend.maximum(influence_distance, epsilon)
            d_safe = backend.maximum(d, epsilon)
            exact_obstacle = d < epsilon
            regular_d = backend.where(exact_obstacle, one, d_safe)
            regular_contribution = (
                -forty
                * repulsive_gain
                * (one / regular_d - one / influence_safe)
                * (one / (regular_d**3))
                * diff
            )
            contribution = backend.where(
                exact_obstacle,
                repulsive_gain * escape_direction,
                regular_contribution,
            )
            repulsive_gradient = repulsive_gradient + backend.where(
                d <= influence_distance, contribution, zero
            )

        # Total gradient
        total_gradient = attractive_gradient + repulsive_gradient
        return total_gradient


from .adjacency import build_link_adjacency  # noqa: E402,F401
from .collision import CollisionChecker, _to_host_numpy  # noqa: E402,F401
