#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screw-axis and twist bookkeeping utilities.

Copyright (c) 2025 Mohamed Aboelnasr. Licensed under AGPL-3.0-or-later.
"""

import math
from typing import Optional

from numpy.typing import NDArray

from ManipulaPy.backend import get_backend

from .so3 import skew_symmetric_to_vector


def extract_r_list(Slist) -> NDArray:
    """Extract representative axis positions from a screw-axis matrix."""
    b = get_backend()
    if Slist is None:
        return b.array([])
    screws = b.asarray(Slist).T
    omega, velocity = screws[:, :3], screws[:, 3:]
    norm_sq = b.sum(omega * omega, axis=1)
    safe_norm_sq = b.where(norm_sq != 0, norm_sq, b.asarray(1.0))
    positions = -b.cross(omega, velocity) / safe_norm_sq.reshape(-1, 1)
    return b.where((norm_sq != 0).reshape(-1, 1), positions, b.zeros(positions.shape))


def extract_omega_list(Slist) -> NDArray:
    """Extract the first three entries from each input screw."""
    return get_backend().asarray(Slist)[:, :3]


def _repeat_columns(array, count: int):
    b = get_backend()
    return b.concatenate(tuple(array for _ in range(count)), axis=1)


def _num_elements(array) -> int:
    """Element count of a backend array.

    NumPy exposes this as the ``.size`` attribute, but on a Torch tensor
    ``.size`` is a *method*, so reading it yields a bound method instead of an
    int. Deriving the count from ``.shape`` is backend-neutral.
    """
    return math.prod(array.shape)


def extract_screw_list(omega_list, r_list) -> Optional[NDArray]:
    """Build a 6xn screw-axis matrix from angular velocities and positions."""
    if omega_list is None or r_list is None:
        return None
    b = get_backend()
    omega_list, r_list = b.asarray(omega_list), b.asarray(r_list)

    r_size = _num_elements(r_list)
    if r_size == 0:
        n_joints = omega_list.shape[1] if omega_list.ndim == 2 else omega_list.shape[0] // 3
        r_list = b.zeros((3, n_joints))
    elif r_list.ndim == 1:
        if r_size % 3:
            raise ValueError(f"Cannot reshape r_list of size {r_size} into (3, n) format")
        r_list = r_list.reshape(3, r_size // 3)

    if omega_list.ndim == 1:
        omega_size = _num_elements(omega_list)
        if omega_size % 3:
            raise ValueError(
                f"Cannot reshape omega_list of size {omega_size} into (3, n) format"
            )
        omega_list = omega_list.reshape(3, omega_size // 3)

    w_rows, w_cols = omega_list.shape
    r_rows, r_cols = r_list.shape
    if w_rows != 3 or r_rows != 3:
        raise ValueError("omega_list and r_list must each have 3 rows.")
    if w_cols != r_cols:
        if r_cols == 1 and w_cols > 1:
            r_list = _repeat_columns(r_list, w_cols)
        elif w_cols == 1 and r_cols > 1:
            omega_list = _repeat_columns(omega_list, r_cols)
            w_cols = r_cols
        else:
            raise ValueError(
                "omega_list and r_list must have the same number of columns. "
                f"Got {w_cols} and {r_cols}."
            )

    linear = b.cross(-omega_list.T, r_list.T).T
    return b.concatenate((omega_list, linear), axis=0)


def logm_to_twist(logm) -> NDArray:
    """Convert a 4x4 matrix logarithm to a twist vector."""
    if logm.shape != (4, 4):
        raise ValueError("logm must be a 4x4 matrix.")
    b = get_backend()
    return b.concatenate((skew_symmetric_to_vector(logm[:3, :3]), logm[:3, 3]))
