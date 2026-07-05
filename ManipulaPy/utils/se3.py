#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SE(3) Utilities - ManipulaPy

Homogeneous-transform math: rotation/position splitting, transform inverse,
the adjoint, the SE(3) matrix exponential and logarithm, and se(3) <-> matrix
conversions.

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

from .so3 import MatrixExp3, VecToso3, rotation_logm, skew_symmetric


def transform_from_twist(S, theta: float) -> NDArray[np.float64]:
    """
    Computes the transformation matrix from a twist and a joint angle.

    Parameters:
        S (array-like): A 6D twist vector.
        theta (float): The joint angle.

    Returns:
        np.ndarray: The corresponding transformation matrix.
    """
    omega = S[:3]
    v = S[3:]
    if np.linalg.norm(omega) == 0:  # Prismatic joint
        T = np.eye(4)
        T[:3, 3] = v * theta
        return T
    else:  # Revolute joint
        skew_omega = skew_symmetric(omega)
        R = (
            np.eye(3)
            + np.sin(theta) * skew_omega
            + (1 - np.cos(theta)) * np.dot(skew_omega, skew_omega)
        )
        p = np.dot(
            np.eye(3) * theta
            + (1 - np.cos(theta)) * skew_omega
            + (theta - np.sin(theta)) * np.dot(skew_omega, skew_omega),
            v,
        )
        return np.vstack((np.hstack((R, p.reshape(-1, 1))), [0, 0, 0, 1]))


def adjoint_transform(T) -> NDArray[np.float64]:
    """
    Computes the adjoint transformation matrix for a given transformation matrix.

    Parameters:
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: The corresponding adjoint transformation matrix.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    skew_p = skew_symmetric(p)
    return np.vstack((np.hstack((R, np.zeros((3, 3)))), np.hstack((skew_p @ R, R))))


def logm(T) -> NDArray[np.float64]:
    """
    Computes the logarithm of a transformation matrix.

    Parameters:
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: The logarithm of the transformation matrix.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    omega, theta = rotation_logm(R)
    if theta < 1e-6:
        return np.hstack((np.zeros(3), p))
    G_inv = (
        1 / theta * np.eye(3)
        - 0.5 * skew_symmetric(omega)
        + (1 / theta - 0.5 / np.tan(theta / 2))
        * np.dot(skew_symmetric(omega), skew_symmetric(omega))
    )
    v = theta * np.dot(G_inv, p)
    return np.hstack((omega * theta, v))


def se3ToVec(se3_matrix) -> NDArray[np.float64]:
    """
    Convert an se(3) matrix to a twist vector.

    Parameters:
        se3_matrix (np.ndarray): A 4x4 matrix from the se(3) Lie algebra.

    Returns:
        np.ndarray: A 6-dimensional twist vector.
    """
    if se3_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    omega = np.array([se3_matrix[2, 1], se3_matrix[0, 2], se3_matrix[1, 0]])
    v = se3_matrix[0:3, 3]
    return np.hstack((omega, v))


def TransToRp(T) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Converts a homogeneous transformation matrix into a rotation matrix and position vector.

    Parameters:
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        tuple: A tuple containing the rotation matrix and position vector.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p


def TransInv(T) -> NDArray[np.float64]:
    """
    Inverts a homogeneous transformation matrix.

    Parameters:
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: The inverse of the transformation matrix.
    """
    R, p = TransToRp(T)
    Rt = R.T
    return np.vstack((np.hstack((Rt, -Rt @ p.reshape(-1, 1))), [0, 0, 0, 1]))


def MatrixLog6(T) -> NDArray[np.float64]:
    """
    Compute the matrix logarithm of a given transformation matrix T.

    Parameters:
        T (np.ndarray): The transformation matrix of shape (4, 4).

    Returns:
        np.ndarray: The matrix logarithm of T, with shape (4, 4).
    """
    R, p = TransToRp(T)
    omega, theta = rotation_logm(R)

    # Pure translation or extremely small rotation
    if abs(theta) < 1e-6:
        return np.vstack(
            (np.hstack((np.zeros((3, 3)), p.reshape(-1, 1))), [0, 0, 0, 0])
        )

    # Use the standard G^{-1}(theta) from Modern Robotics (Eq. 3.88):
    # G_inv = I/theta - 0.5*w_hat + (1/theta - 0.5*cot(theta/2)) * w_hat^2
    w_hat = skew_symmetric(omega)
    G_inv = (
        (np.eye(3) / theta)
        - 0.5 * w_hat
        + (1 / theta - 0.5 / np.tan(theta / 2)) * (w_hat @ w_hat)
    )

    # se(3) log has w_hat*theta in the rotation block
    omega_mat_scaled = theta * w_hat
    v = theta * (G_inv @ p)
    return np.vstack((np.hstack((omega_mat_scaled, v.reshape(-1, 1))), [0, 0, 0, 0]))


def MatrixExp6(se3mat) -> NDArray[np.float64]:
    """
    Computes the matrix exponential of a matrix in se(3).

    Parameters:
        se3mat (np.ndarray): A 4x4 matrix representing a twist in se(3).

    Returns:
        np.ndarray: The corresponding 4x4 transformation matrix in SE(3).
    """
    if se3mat.shape != (4, 4):
        raise ValueError("Input matrix must be of shape (4, 4)")

    # Extract rotation (so(3)) and translation components
    omega_theta_vec = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    v = se3mat[0:3, 3]
    theta = np.linalg.norm(omega_theta_vec)

    # Pure translation case
    if theta < 1e-6:
        T = np.eye(4)
        T[:3, 3] = v
        return T

    # Unit rotation axis hat matrix
    omega_hat = se3mat[0:3, 0:3] / theta

    # Rotation component
    R = MatrixExp3(se3mat[0:3, 0:3])

    # Compute the matrix G(theta) that maps body velocity to translation
    G = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * omega_hat
        + (theta - np.sin(theta)) * (omega_hat @ omega_hat)
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (G @ v) / theta
    return T


def VecTose3(V) -> NDArray[np.float64]:
    """
    Converts a spatial velocity vector to an se(3) matrix.

    Parameters:
        V (array-like): A 6D spatial velocity vector.

    Returns:
        np.ndarray: The corresponding 4x4 matrix in se(3).
    """
    return np.r_[np.c_[VecToso3(V[:3]), V[3:].reshape(3, 1)], np.zeros((1, 4))]
