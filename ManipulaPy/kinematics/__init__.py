#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Kinematics Module - ManipulaPy

This module provides classes and functions for performing kinematic analysis
and computations for serial manipulators, including forward and inverse
kinematics, Jacobian calculations, and end-effector velocity calculations.

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

from typing import Any, List, Optional, Tuple, Union  # noqa: F401

import numpy as np  # noqa: F401
from numpy.typing import NDArray  # noqa: F401

from .. import utils  # noqa: F401
from .serial_manipulator import SerialManipulator
from .trac_ik import TracIKSolver, trac_ik_solve
from .ik_helpers import (
    IKInitialGuessCache,
    adaptive_multi_start_ik,
    extrapolate_from_current,
    midpoint_of_limits,
    random_in_limits,
    workspace_heuristic_guess,
)

__all__ = [
    "SerialManipulator",
    "TracIKSolver",
    "trac_ik_solve",
    "workspace_heuristic_guess",
    "extrapolate_from_current",
    "random_in_limits",
    "midpoint_of_limits",
    "IKInitialGuessCache",
    "adaptive_multi_start_ik",
]

# The concern modules are implementation details, not additions to the historical
# package namespace. Importing serial_manipulator registers them on this package.
for _concern_module in ("fk", "jacobian", "velocity", "ik"):
    globals().pop(_concern_module, None)
del _concern_module
