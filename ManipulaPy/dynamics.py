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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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
        Glist: Union[List[NDArray[np.float64]], NDArray[np.float64]],
        Mlist_per_link: Optional[List[NDArray[np.float64]]] = None, # New
    ) -> None:
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist
        self.Mlist_per_link = Mlist_per_link  # NEW

        self._mass_matrix_cache: Dict[Tuple[float, ...], NDArray[np.float64]] = {}
        self._mass_matrix_derivative_cache: Dict[
            Tuple[Any, ...], NDArray[np.float64]
        ] = {}

    def mass_matrix(
        self, thetalist: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """Compute mass matrix using per-link body Jacobians.

        M(θ) = Σ_k J_k^T G_k J_k

        Where J_k is the body Jacobian of link k's CoM (6×n), and G_k is link
        k's spatial inertia in body frame. For joints i > k, J_k[:, i] = 0
        (link k doesn't depend on those joints).

        Reference: Modern Robotics §8.3 / Murray, Li, Sastry §4.3.

        If Mlist_per_link is None (legacy path), falls back to the previous
        EE-Jacobian approximation with a deprecation warning.
        """
        from ManipulaPy.utils import adjoint_transform as _ad

        thetalist_key = tuple(thetalist)
        if thetalist_key in self._mass_matrix_cache:
            return self._mass_matrix_cache[thetalist_key]

        n = len(thetalist)

        if self.Mlist_per_link is None:
            # Legacy fallback (still wrong, but preserves old behavior for
            # callers constructing ManipulatorDynamics manually without per-link M)
            import warnings
            warnings.warn(
                "mass_matrix called without Mlist_per_link — using legacy "
                "approximation (incorrect for non-trivial robots). Construct "
                "ManipulatorDynamics via URDFToSerialManipulator to get accurate "
                "mass matrix.",
                stacklevel=2,
            )
            return self._mass_matrix_legacy(thetalist)

        M = np.zeros((n, n), dtype=np.float64)

        # Spatial Jacobian columns J_s[:, i] = Ad(prefix_i) @ S_i, built once
        # via the canonical incremental formula in kinematics.jacobian. Body
        # twist of link k is then J_b_k[:, i] = Ad(T_k_com^-1) @ J_s[:, i].
        J_s = self.jacobian(thetalist, frame="space")  # (6, n)

        for k in range(n):
            # T_k_com(θ): base → link k CoM at the current configuration.
            # Joints k+1..n don't move link k, so truncating thetalist to
            # k+1 entries gives the correct link pose; the inv(M_list)
            # @ Mlist_per_link[k] offset shifts from link frame to CoM frame.
            T_k_zero = self.Mlist_per_link[k]
            T_k = self.forward_kinematics(thetalist[: k + 1], frame="space")
            T_k_at_zero = self.forward_kinematics(np.zeros(k + 1), frame="space")
            T_link_to_com = np.linalg.inv(T_k_at_zero) @ T_k_zero
            T_k_com = T_k @ T_link_to_com

            # Convert spatial → body for link k. Columns i > k stay zero
            # because joint i is downstream of link k and doesn't move it.
            Ad_inv_T_k_com = _ad(np.linalg.inv(T_k_com))
            J_k = np.zeros((6, n), dtype=np.float64)
            J_k[:, : k + 1] = Ad_inv_T_k_com @ J_s[:, : k + 1]

            M += J_k.T @ self.Glist[k] @ J_k

        # Symmetrize against floating-point drift
        M = 0.5 * (M + M.T)
        self._mass_matrix_cache[thetalist_key] = M
        return M

    def _mass_matrix_legacy(self, thetalist):
        """Legacy mass matrix (incorrect, kept for backward compat). DO NOT USE."""
        thetalist_key = tuple(thetalist)
        n = len(thetalist)
        M = np.zeros((n, n), dtype=np.float64)
        AdT = np.zeros((6, 6, n + 1))
        AdT[:, :, 0] = np.eye(6)
        for j in range(n):
            T = self.forward_kinematics(thetalist[: j + 1], frame="space")
            AdT[:, :, j + 1] = ad(T)
        J_full = self.jacobian(thetalist, frame="space")
        for i in range(n):
            for j in range(n):
                Ii_base = AdT[:, :, i + 1].T @ self.Glist[i] @ AdT[:, :, i + 1]
                Ji = J_full[:, i]
                Jj = J_full[:, j]
                M[i, j] += Ji.T @ Ii_base @ Jj
        M = 0.5 * (M + M.T)
        self._mass_matrix_cache[thetalist_key] = M
        return M

    def _mass_matrix_derivatives(
        self, thetalist: Union[NDArray[np.float64], List[float]], epsilon: float = 1e-6
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
        self, i: int, j: int, k: int, thetalist: Union[NDArray[np.float64], List[float]]
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
        dthetalist: Union[NDArray[np.float64], List[float]],
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
                    gamma = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                    accum += gamma * dtheta[j] * dtheta[k]
            c[i] = accum
        return c

    def gravity_forces(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]] = [0, 0, -9.81],
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
        Ftip: Union[NDArray[np.float64], List[float]],
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
        Ftip: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = taulist - c - g_forces - np.dot(J_transpose, Ftip)
        ddthetalist = np.linalg.solve(M, rhs)
        return ddthetalist
