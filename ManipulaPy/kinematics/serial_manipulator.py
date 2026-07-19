#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Kinematics Module - ManipulaPy

This module provides classes and functions for performing kinematic analysis and computations
for serial manipulators, including forward and inverse kinematics, Jacobian calculations,
and end-effector velocity calculations.

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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .. import utils
from ..backend import get_backend
from .fk import _ForwardKinematicsMixin
from .ik import _InverseKinematicsMixin
from .jacobian import _JacobianMixin
from .velocity import _VelocityMixin


class SerialManipulator(
    _ForwardKinematicsMixin,
    _JacobianMixin,
    _VelocityMixin,
    _InverseKinematicsMixin,
):
    """Kinematic model for serial manipulators using screw-axis operations."""

    def __init__(
        self,
        M_list: NDArray[np.float64],
        omega_list: Union[NDArray[np.float64], List[float]],
        r_list: Optional[Union[NDArray[np.float64], List[float]]] = None,
        b_list: Optional[Union[NDArray[np.float64], List[float]]] = None,
        S_list: Optional[NDArray[np.float64]] = None,
        B_list: Optional[NDArray[np.float64]] = None,
        G_list: Optional[Union[NDArray[np.float64], List[NDArray[np.float64]]]] = None,
        joint_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    ) -> None:
        """
        Initialize the class with the given parameters.

        Parameters:
            M_list (list): A list of M values.
            omega_list (list): A list of omega values.
            r_list (list, optional): A list of r values. Defaults to None.
            b_list (list, optional): A list of b values. Defaults to None.
            S_list (list, optional): A list of S values. Defaults to None.
            B_list (list, optional): A list of B values. Defaults to None.
            G_list (list, optional): A list of G values. Defaults to None.
            joint_limits (list, optional): A list of joint limits. Defaults to None.
        """
        self.M_list = M_list
        self.G_list = G_list
        self.omega_list = omega_list

        # Extract r_list from S_list if not provided
        self.r_list = r_list if r_list is not None else utils.extract_r_list(S_list)
        # Extract b_list from B_list if not provided
        self.b_list = b_list if b_list is not None else utils.extract_r_list(B_list)

        # Generate S_list if not provided. extract_screw_list already applies
        # the standard v = -omega x r, so omega_list is passed positively;
        # negating it here corrupted the generated space screw (axis flipped),
        # making FK correct only at the home pose.
        self.S_list = (
            S_list
            if S_list is not None
            else utils.extract_screw_list(omega_list, self.r_list)
        )

        # Generate B_list if not provided
        self.B_list = (
            B_list
            if B_list is not None
            else utils.extract_screw_list(omega_list, self.b_list)
        )

        # Determine number of joints for joint limits
        if joint_limits is not None:
            self.joint_limits = joint_limits
        else:
            # Try to infer number of joints from available data
            if hasattr(omega_list, "shape"):
                if omega_list.ndim == 2:
                    n_joints = omega_list.shape[1]
                else:
                    n_joints = (
                        len(omega_list) // 3
                        if len(omega_list) % 3 == 0
                        else len(omega_list)
                    )
            elif hasattr(M_list, "shape"):
                n_joints = 6  # Default assumption for 6-DOF robot
            else:
                n_joints = 6  # Default fallback

            self.joint_limits = [(None, None)] * n_joints

        # Cache whether M_list is a list of poses (3D array) for FK hot path
        m_shape = (
            self.M_list.shape
            if hasattr(self.M_list, "shape")
            else np.shape(self.M_list)
        )
        self._m_list_is_array_of_poses = len(m_shape) > 2

    def update_state(
        self,
        joint_positions: Union[NDArray[np.float64], List[float]],
        joint_velocities: Optional[Union[NDArray[np.float64], List[float]]] = None,
    ) -> None:
        """
        Updates the internal state of the manipulator.

        Args:
            joint_positions (np.ndarray): Current joint positions.
            joint_velocities (np.ndarray, optional): Current joint velocities. Default is None.
        """
        backend = get_backend()
        self.joint_positions = backend.asarray(joint_positions)
        if joint_velocities is not None:
            self.joint_velocities = backend.asarray(joint_velocities)
        else:
            self.joint_velocities = backend.zeros(
                self.joint_positions.shape, dtype=self.joint_positions.dtype
            )


class _KinematicsCompatibilityModule(__import__("types").ModuleType):
    """Keep historical runtime patches visible to extracted mixins."""

    _forwarded_names = {"get_backend", "np", "utils"}
    _concern_modules = (
        "ManipulaPy.kinematics.fk",
        "ManipulaPy.kinematics.jacobian",
        "ManipulaPy.kinematics.velocity",
        "ManipulaPy.kinematics.ik",
    )

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._forwarded_names:
            for module_name in self._concern_modules:
                module = __import__(module_name, fromlist=["*"])
                if name in vars(module):
                    setattr(module, name, value)


__import__("sys").modules[__name__].__class__ = _KinematicsCompatibilityModule

del (
    _ForwardKinematicsMixin,
    _InverseKinematicsMixin,
    _JacobianMixin,
    _VelocityMixin,
    _KinematicsCompatibilityModule,
)
