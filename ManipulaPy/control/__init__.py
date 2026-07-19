#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Control Module - ManipulaPy

This module provides various control algorithms for robotic manipulators including
PID, computed torque, adaptive, and robust control methods.

Control inputs, guarded dynamics results, and persistent controller state are
normalized with the caller-selected active array backend; NumPy remains the process
default. Plotting and tuning utilities convert only at declared host boundaries, and
response metrics return their public Python scalars after backend-native operations.

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

from ManipulaPy.backend import get_backend


def _as_backend_array(value: Any) -> Any:
    """Normalize an input or numerical result with the active backend."""
    return get_backend().asarray(value)


def _to_host_array(value: Any) -> NDArray[Any]:
    """Convert a backend-native value at an explicit host-only boundary."""
    if isinstance(value, np.ndarray):
        return value
    backend = get_backend()
    if backend.is_backend_array(value):
        return backend.to_numpy(value)
    return np.asarray(value)


def _validate_i_clamp(i_clamp: Optional[float]) -> Optional[float]:
    """Return a scalar positive finite integral clamp, or None if disabled."""
    if i_clamp is None:
        return None

    clamp = np.asarray(_to_host_array(i_clamp), dtype=float)
    if clamp.size != 1:
        raise ValueError(f"i_clamp must be a scalar, got shape {clamp.shape}")

    clamp_value = float(clamp.reshape(-1)[0])
    if not np.isfinite(clamp_value) or clamp_value <= 0:
        raise ValueError(
            f"i_clamp must be positive and finite when set, got {i_clamp!r}"
        )
    return clamp_value


from .manipulator_controller import ManipulatorController, logger
