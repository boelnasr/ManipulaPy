#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Transformations Module - ManipulaPy

This module provides essential geometric transformation utilities for robotics applications
including rotation matrices for X, Y, and Z axes, Euler angle conversions, and homogeneous
transformation matrix construction for pose representation and manipulation.

The module supports:
- Elementary rotation matrices around coordinate axes
- Euler angle to rotation matrix conversion (ZYX convention)
- Pose vector to homogeneous transformation matrix conversion
- Type-annotated functions for better code clarity and IDE support

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


class Transformations:
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the X-axis."""
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the Y-axis."""
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the Z-axis."""
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def vector_2_matrix(vector: np.ndarray) -> np.ndarray:
        """
        Calculate the pose from the position and Euler angles (ZYX order).

        Args:
            vector (np.ndarray): A 6-element array where the first 3 elements are x, y, z translation,
                                and the last 3 elements are rotation angles.

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        translation_component = vector[:3]
        rotation_component = vector[3:]

        # Rotation matrices for each Euler angle
        Rz = Transformations.rotation_matrix_z(rotation_component[0])
        Ry = Transformations.rotation_matrix_y(rotation_component[1])
        Rx = Transformations.rotation_matrix_x(rotation_component[2])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Construct the transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = translation_component

        return transform_matrix
