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

from .. import utils
from ..backend import get_backend


class _JacobianMixin:
    def jacobian(
        self, thetalist: Union[NDArray[np.float64], List[float]], frame: str = "space"
    ) -> NDArray[np.float64]:
        """
        Calculate the Jacobian matrix for the given joint angles.

        Parameters:
            thetalist (list): A list of joint angles.
            frame (str): The reference frame for the Jacobian calculation.
                        Valid values are 'space' or 'body'. Defaults to 'space'.

        Returns:
            numpy.ndarray: The Jacobian matrix of shape (6, len(thetalist)).
        """
        backend = get_backend()
        theta_input = thetalist
        thetalist = backend.asarray(theta_input)
        dtype_kind = getattr(thetalist.dtype, "kind", None)
        if not backend.is_backend_array(theta_input) or dtype_kind in ("b", "i", "u"):
            thetalist = backend.asarray(theta_input, dtype=backend.float64)
        S_list = backend.asarray(self.S_list)
        B_list = backend.asarray(self.B_list)
        T = backend.eye(4, dtype=thetalist.dtype)
        if frame == "space":
            if len(thetalist) == 0:
                return backend.zeros((6, 0), dtype=thetalist.dtype)
            columns = []
            for i in range(len(thetalist)):
                columns.append(backend.matmul(utils.adjoint_transform(T), S_list[:, i]))
                T = backend.matmul(
                    T, utils.transform_from_twist(S_list[:, i], thetalist[i])
                )
        elif frame == "body":
            # Modern Robotics JacobianBody: start from identity, accumulate
            # e^{-[B_{i+1}]theta_{i+1}} to the right. The last column is B_n
            # (Ad(I) @ B_n); each earlier column is Ad of the trailing product.
            n = len(thetalist)
            columns = [None] * n
            columns[n - 1] = B_list[:, n - 1]
            for i in range(n - 2, -1, -1):
                T = backend.matmul(
                    T,
                    utils.transform_from_twist(B_list[:, i + 1], -thetalist[i + 1]),
                )
                columns[i] = backend.matmul(utils.adjoint_transform(T), B_list[:, i])
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return backend.stack(columns, axis=1)
