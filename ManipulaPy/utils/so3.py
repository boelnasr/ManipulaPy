#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SO(3) Utilities - ManipulaPy

Rotation-group math: skew-symmetric (hat/vee) operators, the SO(3) matrix
exponential and logarithm, axis-angle extraction, and rotation <-> Euler
angle conversions.

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

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


def NearZero(z: float) -> bool:
    """
    Determines if a given number is near zero.

    Parameters:
        z (float): The number to check.

    Returns:
        bool: True if the number is near zero, False otherwise.
    """
    return abs(z) < 1e-6


def skew_symmetric(v) -> NDArray[np.float64]:
    """
    Returns the skew symmetric matrix of a 3D vector.

    Parameters:
        v (array-like): A 3D vector.

    Returns:
        np.ndarray: The corresponding skew symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_logm(R) -> Tuple[NDArray[np.float64], float]:
    """
    Computes the logarithm of a rotation matrix.

    Parameters:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: A tuple containing the rotation vector and the angle.
    """
    # Validate input shape
    R = np.asarray(R)
    if R.shape != (3, 3):
        raise ValueError(
            f"rotation_logm requires a 3x3 rotation matrix, got shape {R.shape}. "
            f"Matrix:\n{R}"
        )

    # Ensure we're computing with scalar values
    trace_val = np.trace(R)
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_theta = np.clip((trace_val - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Convert to Python scalar (handles numpy scalars, 0-d arrays, and 1-d arrays)
    try:
        theta_scalar = float(theta)
    except (TypeError, ValueError):
        # Handle cases where theta is an array
        theta_arr = np.asarray(theta).flatten()
        if theta_arr.size == 0:
            theta_scalar = 0.0
        else:
            theta_scalar = float(theta_arr[0])

    # Check if rotation is very small (identity or near-identity)
    if theta_scalar < 1e-6:
        return np.zeros(3), theta_scalar
    elif theta_scalar > np.pi - 1e-6:
        # theta ~ pi: R is (near-)symmetric so R - R^T -> 0 while 1/(2 sin theta)
        # blows up, collapsing the generic formula to the zero vector. Extract
        # the axis from the most positive diagonal term instead (mirrors
        # MatrixLog3 / Modern Robotics). The threshold is kept tight (1e-6):
        # the generic formula stays accurate until extremely close to pi, and
        # the diagonal extraction assumes exactly theta = pi, so widening the
        # band would trade a smaller floating-point error for a larger
        # axis-approximation error.
        if not NearZero(1 + R[2, 2]):
            omega = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array(
                [R[0, 2], R[1, 2], 1 + R[2, 2]]
            )
        elif not NearZero(1 + R[1, 1]):
            omega = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array(
                [R[0, 1], 1 + R[1, 1], R[2, 1]]
            )
        else:
            omega = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array(
                [1 + R[0, 0], R[1, 0], R[2, 0]]
            )
        return omega, theta_scalar
    else:
        omega = (
            1
            / (2 * np.sin(theta_scalar))
            * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        )
        return omega, theta_scalar


def skew_symmetric_to_vector(skew_symmetric) -> NDArray[np.float64]:
    """
    Convert a skew-symmetric matrix to a vector.

    Parameters:
        skew_symmetric (np.ndarray): A 3x3 skew-symmetric matrix.

    Returns:
        np.ndarray: The corresponding vector.
    """
    return np.array([skew_symmetric[2, 1], skew_symmetric[0, 2], skew_symmetric[1, 0]])


def MatrixLog3(R) -> NDArray[np.float64]:
    """
    Computes the matrix logarithm of a rotation matrix.

    Parameters:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: The matrix logarithm of the rotation matrix.
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omega = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array(
                [R[0][2], R[1][2], 1 + R[2][2]]
            )
        elif not NearZero(1 + R[1][1]):
            omega = (1.0 / np.sqrt(2 * (1 + R[1][1]))) * np.array(
                [R[0][1], 1 + R[1][1], R[2][1]]
            )
        else:
            omega = (1.0 / np.sqrt(2 * (1 + R[0][0]))) * np.array(
                [1 + R[0][0], R[1][0], R[2][0]]
            )
        return VecToso3(np.pi * omega)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)


def VecToso3(omega) -> NDArray[np.float64]:
    """
    Converts a 3D angular velocity vector to a skew-symmetric matrix.

    Parameters:
        omega (array-like): A 3D angular velocity vector.

    Returns:
        np.ndarray: The corresponding skew-symmetric matrix.
    """
    return np.array(
        [[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]]
    )


def MatrixExp3(so3mat) -> NDArray[np.float64]:
    """
    Computes the matrix exponential of a matrix in so(3).

    Parameters:
        so3mat (np.ndarray): A 3x3 skew-symmetric matrix.

    Returns:
        np.ndarray: The corresponding 3x3 rotation matrix in SO(3).
    """
    return expm(so3mat)


def rotation_matrix_to_euler_angles(R) -> NDArray[np.float64]:
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).

    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        numpy.ndarray: A 3-element array representing the Euler angles (roll, pitch, yaw).
    """
    assert R.shape == (3, 3), f"Expected 3x3 rotation matrix, got shape {R.shape}"
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_to_rotation_matrix(euler_deg) -> NDArray[np.float64]:
    """
    Convert Euler angles (roll_deg, pitch_deg, yaw_deg) in degrees
    to a 3x3 rotation matrix.
    ZYX convention is typical in robotics: yaw -> pitch -> roll,
    but adapt as needed.

    Parameters:
        euler_deg (array-like): [roll_deg, pitch_deg, yaw_deg]

    Returns:
        np.ndarray: A 3x3 rotation matrix (float64 by default).
    """
    roll_deg, pitch_deg, yaw_deg = euler_deg
    # Convert degrees to radians
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    # Example Z-Y-X convention (yaw→pitch→roll).
    # If your code uses X→Y→Z or another sequence, adapt these multiplications.
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )

    # Multiply in the correct order for your convention.
    R = Rz @ Ry @ Rx
    return R
