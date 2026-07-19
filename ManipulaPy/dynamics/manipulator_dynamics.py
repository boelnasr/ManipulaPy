#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Dynamics Module - ManipulaPy

This module provides classes and functions for manipulator dynamics analysis
including mass matrix computation, Coriolis forces, gravity compensation, and
inverse/forward dynamics.

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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..backend import get_backend  # noqa: F401
from ..kinematics import SerialManipulator
from ..utils import adjoint_transform as ad  # noqa: F401
from .cache import _CacheConcern
from .forces import _ForcesConcern
from .id_fd import _InverseForwardDynamicsConcern
from .mass_matrix import _MassMatrixConcern


class ManipulatorDynamics(SerialManipulator):
    """Serial manipulator model with mass, Coriolis, gravity, and dynamics APIs."""

    def __init__(
        self,
        M_list: NDArray[np.float64],
        omega_list: Union[NDArray[np.float64], List[float]],
        r_list: Union[NDArray[np.float64], List[float]],
        b_list: Union[NDArray[np.float64], List[float]],
        S_list: NDArray[np.float64],
        B_list: NDArray[np.float64],
        Glist: Union[List[NDArray[np.float64]], NDArray[np.float64]],
        Mlist_per_link: Optional[List[NDArray[np.float64]]] = None,  # New
    ) -> None:
        """
        Initialize manipulator dynamics data and finite-difference caches.

        Args:
            M_list: Home transforms for the manipulator.
            omega_list: Joint angular velocity axes.
            r_list: Space-frame screw-axis position vectors.
            b_list: Body-frame screw-axis position vectors.
            S_list: Space-frame screw axes.
            B_list: Body-frame screw axes.
            Glist: Spatial inertia matrices per link.
            Mlist_per_link: Optional CoM transforms per link.
        """
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist
        self.Mlist_per_link = Mlist_per_link  # NEW

        self._mass_matrix_cache: Dict[Tuple[Any, ...], Any] = {}
        self._mass_matrix_derivative_cache: Dict[Tuple[Any, ...], Any] = {}

    _concrete_cache_key = _CacheConcern.__dict__["_concrete_cache_key"]
    _mass_matrix_derivatives = _CacheConcern.__dict__["_mass_matrix_derivatives"]
    mass_matrix = _MassMatrixConcern.__dict__["mass_matrix"]
    _mass_matrix_legacy = _MassMatrixConcern.__dict__["_mass_matrix_legacy"]
    partial_derivative = _ForcesConcern.__dict__["partial_derivative"]
    velocity_quadratic_forces = _ForcesConcern.__dict__["velocity_quadratic_forces"]
    gravity_forces = _ForcesConcern.__dict__["gravity_forces"]
    _gravity_forces_legacy = _ForcesConcern.__dict__["_gravity_forces_legacy"]
    inverse_dynamics = _InverseForwardDynamicsConcern.__dict__["inverse_dynamics"]
    forward_dynamics = _InverseForwardDynamicsConcern.__dict__["forward_dynamics"]
