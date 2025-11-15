#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Dynamics Module - ManipulaPy

This module provides classes and functions for manipulator dynamics analysis including
mass matrix computation, Coriolis forces, gravity compensation, and inverse/forward dynamics.

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
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from numpy.typing import NDArray
from .kinematics import SerialManipulator
from .utils import adjoint_transform as ad


class ManipulatorDynamics(SerialManipulator):
    def __init__(
        self,
        M_list: NDArray[np.float64],
        omega_list: Union[NDArray[np.float64], List[float]],
        r_list: Union[NDArray[np.float64], List[float]],
        b_list: Union[NDArray[np.float64], List[float]],
        S_list: NDArray[np.float64],
        B_list: NDArray[np.float64],
        Glist: Union[List[NDArray[np.float64]], NDArray[np.float64]]
    ) -> None:
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist
        self._mass_matrix_cache: Dict[Tuple[float, ...], NDArray[np.float64]] = {}
        self._mass_matrix_derivative_cache: Dict[Tuple[Any, ...], NDArray[np.float64]] = {}

    def mass_matrix(self, thetalist: Union[NDArray[np.float64], List[float]]) -> NDArray[np.float64]:
        thetalist_key = tuple(thetalist)
        if thetalist_key in self._mass_matrix_cache:
            return self._mass_matrix_cache[thetalist_key]

        n = len(thetalist)
        M = np.zeros((n, n), dtype=np.float64)

        # Precompute the "space" transforms for each link
        AdT = np.zeros((6, 6, n + 1))
        AdT[:, :, 0] = np.eye(6)
        for j in range(n):
            T = self.forward_kinematics(thetalist[:j + 1], frame="space")
            AdT[:, :, j + 1] = ad(T)

        # Full space Jacobian at the end-effector
        J_full = self.jacobian(thetalist, frame="space")  # shape (6, n)

        # Implement the correct formula: M(θ) = Σᵢ JᵢᵀI_i Jᵢ
        # where Jᵢ is the spatial Jacobian up to link i, and I_i is link i's spatial inertia in base frame
        for i in range(n):
            for j in range(n):
                # Transform the i-th link's inertia into the base frame
                Ii_base = AdT[:, :, i + 1].T @ self.Glist[i] @ AdT[:, :, i + 1]
                
                # Get the spatial Jacobian columns for joints i and j
                Ji = J_full[:, i]  # i-th column of Jacobian
                Jj = J_full[:, j]  # j-th column of Jacobian
                
                # Accumulate M[i,j] = Jᵢᵀ I_i Jⱼ for each link
                M[i, j] += Ji.T @ Ii_base @ Jj

        # Ensure exact symmetry (guard against tiny floating-point drift)
        M = 0.5 * (M + M.T)
        self._mass_matrix_cache[thetalist_key] = M
        return M

    def _mass_matrix_derivatives(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        epsilon: float = 1e-6
    ) -> NDArray[np.float64]:
        """
        Central finite-difference approximation of dM/dtheta_k for
        every joint angle, cached so repeated calls avoid recomputing
        full mass matrices inside tight loops.
        """
        theta_key = tuple(np.asarray(thetalist, dtype=np.float64))
        cache_key = (theta_key, float(epsilon))
        if cache_key in self._mass_matrix_derivative_cache:
            return self._mass_matrix_derivative_cache[cache_key]

        n = len(thetalist)
        derivatives = np.zeros((n, n, n), dtype=np.float64)
        for k in range(n):
            thetalist_plus = np.array(thetalist, dtype=np.float64)
            thetalist_plus[k] += epsilon
            thetalist_minus = np.array(thetalist, dtype=np.float64)
            thetalist_minus[k] -= epsilon

            M_plus = self.mass_matrix(thetalist_plus)
            M_minus = self.mass_matrix(thetalist_minus)
            derivatives[:, :, k] = (M_plus - M_minus) / (2.0 * epsilon)

        self._mass_matrix_derivative_cache[cache_key] = derivatives
        return derivatives

    def partial_derivative(
        self,
        i: int,
        j: int,
        k: int,
        thetalist: Union[NDArray[np.float64], List[float]]
    ) -> float:
        """
        Keep public API but serve results from the cached tensor so a
        single derivative never re-triggers mass matrix evaluation.
        """
        dM = self._mass_matrix_derivatives(thetalist)
        return dM[i, j, k]

    def velocity_quadratic_forces(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        n = len(thetalist)
        dtheta = np.asarray(dthetalist, dtype=np.float64)
        if np.allclose(dtheta, 0.0):
            return np.zeros(n, dtype=np.float64)

        dM = self._mass_matrix_derivatives(thetalist)
        c = np.zeros(n, dtype=np.float64)

        for i in range(n):
            accum = 0.0
            for j in range(n):
                for k in range(n):
                    gamma = 0.5 * (
                        dM[i, j, k] + dM[i, k, j] - dM[j, k, i]
                    )
                    accum += gamma * dtheta[j] * dtheta[k]
            c[i] = accum
        return c

    def gravity_forces(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]] = [0, 0, -9.81]
    ) -> NDArray[np.float64]:
        n = len(thetalist)
        grav = np.zeros(n)
        G = np.array(g)
        for i in range(n):
            AdT = ad(self.forward_kinematics(thetalist[: i + 1], "space"))
            grav[i] = np.dot(AdT.T[:3, :3], G[:3]).dot(
                self.Glist[i][:3, :3].sum(axis=0)
            )
        return grav

    def inverse_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        ddthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        n = len(thetalist)
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        taulist = np.dot(M, ddthetalist) + c + g_forces + np.dot(J_transpose, Ftip)
        return taulist

    def forward_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        taulist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = taulist - c - g_forces - np.dot(J_transpose, Ftip)
        ddthetalist = np.linalg.solve(M, rhs)
        return ddthetalist
