#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Dynamics Module - ManipulaPy

This module provides classes and functions for manipulator dynamics analysis including
mass matrix computation, Coriolis forces, gravity compensation, and inverse/forward dynamics.

Copyright (c) 2025 Mohamed Aboelnar
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
from .kinematics import SerialManipulator
from .utils import adjoint_transform as ad


class ManipulatorDynamics(SerialManipulator):
    def __init__(self, M_list, omega_list, r_list, b_list, S_list, B_list, Glist):
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist
        self._mass_matrix_cache = {}

    def mass_matrix(self, thetalist):
        thetalist_key = tuple(thetalist)
        if thetalist_key in self._mass_matrix_cache:
            return self._mass_matrix_cache[thetalist_key]

        n = len(thetalist)
        M = np.zeros((n, n))
        AdT = np.zeros((6, 6, n + 1))
        AdT[:, :, 0] = np.eye(6)

        for i in range(n):
            T = self.forward_kinematics(thetalist[: i + 1], "space")
            AdT[:, :, i + 1] = ad(T)

        for i in range(n):
            F = np.zeros(6)
            for j in range(i, n):
                AdTi = AdT[:, :, j + 1].T
                I = self.Glist[j]
                Ia = np.dot(AdTi, np.dot(I, AdT[:, :, j + 1]))
                dV = np.zeros(6)
                dV[5] = 1 if i == j else 0
                F += np.dot(Ia, dV)
            M[i, i:] = np.dot(self.jacobian(thetalist).T, F)[i:]

        M = M + M.T - np.diag(np.diag(M))
        self._mass_matrix_cache[thetalist_key] = M
        return M

    def partial_derivative(self, i, j, k, thetalist):
        epsilon = 1e-6
        thetalist_plus = np.array(thetalist)
        thetalist_plus[k] += epsilon
        M_plus = self.mass_matrix(thetalist_plus)

        thetalist_minus = np.array(thetalist)
        thetalist_minus[k] -= epsilon
        M_minus = self.mass_matrix(thetalist_minus)

        return (M_plus[i, j] - M_minus[i, j]) / (2 * epsilon)

    def velocity_quadratic_forces(self, thetalist, dthetalist):
        n = len(thetalist)
        c = np.zeros(n)
        J = self.jacobian(thetalist)
        for i in range(n):
            c[i] = sum(
                [
                    self.partial_derivative(i, j, k, thetalist)
                    * dthetalist[j]
                    * dthetalist[k]
                    for j in range(n)
                    for k in range(n)
                ]
            )
        return c

    def gravity_forces(self, thetalist, g=[0, 0, -9.81]):
        n = len(thetalist)
        grav = np.zeros(n)
        G = np.array(g)
        for i in range(n):
            AdT = ad(self.forward_kinematics(thetalist[: i + 1], "space"))
            grav[i] = np.dot(AdT.T[:3, :3], G[:3]).dot(
                self.Glist[i][:3, :3].sum(axis=0)
            )
        return grav

    def inverse_dynamics(self, thetalist, dthetalist, ddthetalist, g, Ftip):
        n = len(thetalist)
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        taulist = np.dot(M, ddthetalist) + c + g_forces + np.dot(J_transpose, Ftip)
        return taulist

    def forward_dynamics(self, thetalist, dthetalist, taulist, g, Ftip):
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = taulist - c - g_forces - np.dot(J_transpose, Ftip)
        ddthetalist = np.linalg.solve(M, rhs)
        return ddthetalist
