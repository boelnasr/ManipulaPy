#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Control Module - ManipulaPy

This module provides various control algorithms for robotic manipulators including
PID, computed torque, adaptive, and robust control methods.

Note: All control methods use CPU-based NumPy computation to avoid GPU-CPU transfer
overhead. Since the dynamics module operates on NumPy arrays, keeping everything on
the CPU is significantly more efficient than repeated PCIe transfers between GPU and
CPU memory spaces.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

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

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Optional CuPy import for defensive array handling
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _to_numpy(arr: Any) -> NDArray[Any]:
    """
    Safely convert array to NumPy, handling both NumPy and CuPy arrays.

    Args:
        arr: Input array (can be NumPy array, CuPy array, or list)

    Returns:
        NumPy array

    Note:
        This is necessary because np.asarray() does not work with CuPy arrays.
        CuPy raises "Implicit conversion to a NumPy array is not allowed"
        to prevent accidental performance issues. We must explicitly call .get()
        to transfer CuPy arrays from GPU to CPU.
    """
    if CUPY_AVAILABLE and cp is not None:
        try:
            if isinstance(arr, cp.ndarray):
                # CuPy array: explicitly transfer from GPU to CPU
                return arr.get()
        except (TypeError, AttributeError):
            # cp.ndarray may not be a real type when CuPy is mocked; treat as non-CuPy
            pass

    # NumPy array, list, or other: convert to NumPy
    return np.asarray(arr)


def _validate_i_clamp(i_clamp: Optional[float]) -> Optional[float]:
    """Return a scalar positive finite integral clamp, or None if disabled."""
    if i_clamp is None:
        return None

    clamp = np.asarray(_to_numpy(i_clamp), dtype=float)
    if clamp.size != 1:
        raise ValueError(f"i_clamp must be a scalar, got shape {clamp.shape}")

    clamp_value = float(clamp.reshape(-1)[0])
    if not np.isfinite(clamp_value) or clamp_value <= 0:
        raise ValueError(
            f"i_clamp must be positive and finite when set, got {i_clamp!r}"
        )
    return clamp_value


from .manipulator_controller import ManipulatorController, logger
