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

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from . import serial_manipulator as _runtime


class _ForwardKinematicsConcern:
    def forward_kinematics(
        self, thetalist: Union[NDArray[np.float64], List[float]], frame: str = "space"
    ) -> NDArray[np.float64]:
        """
        Compute the forward kinematics of a robotic arm using the product of exponentials method.

        Args:
            thetalist (numpy.ndarray): A 1D array of joint angles in radians.
            frame (str, optional): The frame in which to compute the forward kinematics.
                Either 'space' or 'body'.

        Returns:
            numpy.ndarray: The 4x4 transformation matrix representing the end-effector's pose.
        """
        backend = _runtime.get_backend()
        theta_input = thetalist
        thetalist = backend.asarray(theta_input)
        dtype_kind = getattr(thetalist.dtype, "kind", None)
        if not backend.is_backend_array(theta_input) or dtype_kind in ("b", "i", "u"):
            thetalist = backend.asarray(theta_input, dtype=backend.float64)
        S_list = backend.asarray(self.S_list)
        B_list = backend.asarray(self.B_list)
        if frame == "space":
            # T(θ) = e^[S1θ1] e^[S2θ2] ... e^[Snθn] * M
            T = backend.eye(4, dtype=thetalist.dtype)
            for i, theta in enumerate(thetalist):
                T = backend.matmul(
                    T,
                    _runtime.utils.transform_from_twist(S_list[:, i], theta),
                )
            M = self.M_list[-1] if self._m_list_is_array_of_poses else self.M_list
            T = backend.matmul(T, backend.asarray(M))

        elif frame == "body":
            # T(θ) = M * e^[B1θ1] e^[B2θ2] ... e^[Bnθn]
            T = backend.eye(4, dtype=thetalist.dtype)
            for i, theta in enumerate(thetalist):
                T = backend.matmul(
                    T,
                    _runtime.utils.transform_from_twist(B_list[:, i], theta),
                )
            M = self.M_list[-1] if self._m_list_is_array_of_poses else self.M_list
            T = backend.matmul(backend.asarray(M), T)

        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")

        return T

    def end_effector_pose(
        self, thetalist: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        Computes the end-effector's position and orientation given joint angles.

        Parameters:
            thetalist (numpy.ndarray): A 1D array of joint angles in radians.

        Returns:
            numpy.ndarray: A 6x1 vector representing the position and orientation (Euler angles) of the end-effector.
        """
        T = self.forward_kinematics(thetalist)
        R, p = _runtime.utils.TransToRp(T)
        orientation = _runtime.utils.rotation_matrix_to_euler_angles(R)
        return _runtime.np.concatenate((p, orientation))
