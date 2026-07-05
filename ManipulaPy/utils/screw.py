#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Screw/Twist Utilities - ManipulaPy

Screw-axis bookkeeping: extracting rotation-axis positions and angular parts
from a screw list, assembling screw-axis matrices, and converting a matrix
logarithm to a twist vector.

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

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .so3 import skew_symmetric_to_vector


def extract_r_list(Slist) -> NDArray[np.float64]:
    """
    Extracts the r_list from the given Slist.

    Parameters:
        Slist (list): A list of S vectors representing the joint screws.

    Returns:
        np.ndarray: An array of r vectors.
    """
    # Handle None or improperly shaped input
    if Slist is None or not hasattr(np.array(Slist), "T"):
        return np.array([])

    # Continue with the original function
    r_list = []
    for S in np.array(Slist).T:
        omega = S[:3]
        v = S[3:]
        if np.linalg.norm(omega) != 0:
            r = -np.cross(omega, v) / np.linalg.norm(omega) ** 2
            r_list.append(r)
        else:
            r_list.append([0, 0, 0])  # For prismatic joints
    return np.array(r_list)


def extract_omega_list(Slist) -> NDArray[np.float64]:
    """
    Extracts the first three elements from each sublist in the given list and returns them as a numpy array.

    Parameters:
        Slist (list): A list of sublists.

    Returns:
        np.array: A numpy array containing the first three elements from each sublist.
    """
    return np.array(Slist)[:, :3]


def extract_screw_list(omega_list, r_list) -> Optional[NDArray[np.float64]]:
    """
    Build a 6xn screw-axis matrix from (3xn) angular velocities 'omega_list'
    and (3xn) positions 'r_list'.

    For each column ``i``, ``S[:3, i]`` is set to ``omega_list[:, i]`` and
    ``S[3:, i]`` is set to ``-(omega x r)``. Returns a 6xn array of
    ``[wx, wy, wz, vx, vy, vz]`` in each column.
    """
    if omega_list is None or r_list is None:
        return None

    # Convert to numpy arrays if not already
    omega_list = np.asarray(omega_list)
    r_list = np.asarray(r_list)

    # Handle case where r_list is empty or 1D
    if r_list.size == 0:
        # Create a default r_list with zeros
        if omega_list.ndim == 2:
            n_joints = omega_list.shape[1]
        else:
            n_joints = omega_list.shape[0] // 3 if omega_list.ndim == 1 else 1
        r_list = np.zeros((3, n_joints))
    elif r_list.ndim == 1:
        # If r_list is 1D, reshape or handle appropriately
        if r_list.size == 0:
            if omega_list.ndim == 2:
                n_joints = omega_list.shape[1]
            else:
                n_joints = 1
            r_list = np.zeros((3, n_joints))
        elif r_list.size == 3:
            # Single position vector, reshape to (3, 1)
            r_list = r_list.reshape(3, 1)
        else:
            # Multiple positions in 1D array, try to reshape
            if r_list.size % 3 == 0:
                n_positions = r_list.size // 3
                r_list = r_list.reshape(3, n_positions)
            else:
                raise ValueError(
                    f"Cannot reshape r_list of size {r_list.size} into (3, n) format"
                )

    # Ensure omega_list is also 2D
    if omega_list.ndim == 1:
        if omega_list.size % 3 == 0:
            n_joints = omega_list.size // 3
            omega_list = omega_list.reshape(3, n_joints)
        else:
            raise ValueError(
                f"Cannot reshape omega_list of size {omega_list.size} into (3, n) format"
            )

    w_rows, w_cols = omega_list.shape
    r_rows, r_cols = r_list.shape

    if w_rows != 3 or r_rows != 3:
        raise ValueError("omega_list and r_list must each have 3 rows.")
    if w_cols != r_cols:
        # Try to broadcast if one has only one column
        if r_cols == 1 and w_cols > 1:
            r_list = np.tile(r_list, (1, w_cols))
            r_cols = w_cols
        elif w_cols == 1 and r_cols > 1:
            omega_list = np.tile(omega_list, (1, r_cols))
            w_cols = r_cols
        else:
            raise ValueError(
                f"omega_list and r_list must have the same number of columns. Got {w_cols} and {r_cols}."
            )

    S = np.zeros((6, w_cols), dtype=float)
    for i in range(w_cols):
        w = omega_list[:, i]
        r = r_list[:, i]
        v = np.cross(-w, r)
        S[:3, i] = w
        S[3:, i] = v
    return S


def logm_to_twist(logm) -> NDArray[np.float64]:
    """
    Convert the logarithm of a transformation matrix to a twist vector.

    Parameters:
        logm (np.ndarray): The logarithm of a transformation matrix.

    Returns:
        np.ndarray: The corresponding twist vector.
    """
    if logm.shape != (4, 4):
        raise ValueError("logm must be a 4x4 matrix.")

    omega_matrix = logm[0:3, 0:3]
    omega = skew_symmetric_to_vector(omega_matrix)
    v = logm[0:3, 3]
    return np.hstack((omega, v))
