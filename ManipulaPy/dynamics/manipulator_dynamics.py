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

from ..backend import get_backend
from ..kinematics import SerialManipulator
from ..utils import adjoint_transform as ad


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

    @staticmethod
    def _concrete_cache_key(backend: Any, thetalist: Any) -> Tuple[Any, ...]:
        """Return a backend-scoped, host-normalized key for concrete arrays."""
        theta_host = backend.to_numpy(
            backend.asarray(thetalist, dtype=backend.float64)
        )
        theta_key = tuple(float(value) for value in theta_host)
        return (backend.cache_token(), theta_key)

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
        backend = get_backend()
        # The tuple key hashes only for concrete (NumPy/CuPy) arrays; a traced
        # tensor would hash by identity (silent never-hit) or need a host sync
        # that breaks trace-safety, so the cache is skipped for such backends.
        use_cache = backend.is_concrete
        if use_cache:
            thetalist_key = self._concrete_cache_key(backend, thetalist)
            cached = self._mass_matrix_cache.get(thetalist_key)
            if cached is not None:
                return cached

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

        # Spatial Jacobian columns J_s[:, i] = Ad(prefix_i) @ S_i, built once
        # via the canonical incremental formula in kinematics.jacobian. Body
        # twist of link k is then J_b_k[:, i] = Ad(T_k_com^-1) @ J_s[:, i].
        J_s = self.jacobian(thetalist, frame="space")  # (6, n)

        # Functional accumulation (M = M + term) keeps the hot path free of
        # in-place writes so a future traced backend needs no further changes.
        M = backend.zeros((n, n), dtype=backend.float64)
        for k in range(n):
            # T_k_com(θ): base → link k CoM at the current configuration.
            # Joints k+1..n don't move link k, so truncating thetalist to
            # k+1 entries gives the correct link pose; the inv(M_list)
            # @ Mlist_per_link[k] offset shifts from link frame to CoM frame.
            T_k_zero = backend.asarray(self.Mlist_per_link[k])
            T_k = self.forward_kinematics(thetalist[: k + 1], frame="space")
            T_k_at_zero = self.forward_kinematics(
                backend.zeros((k + 1,), dtype=backend.float64), frame="space"
            )
            T_link_to_com = backend.matmul(backend.inv(T_k_at_zero), T_k_zero)
            T_k_com = backend.matmul(T_k, T_link_to_com)

            # Convert spatial → body for link k. Columns i > k stay zero
            # because joint i is downstream of link k and doesn't move it.
            Ad_inv_T_k_com = ad(backend.inv(T_k_com))
            J_k_active = backend.matmul(Ad_inv_T_k_com, J_s[:, : k + 1])
            J_k = backend.concatenate(
                (
                    J_k_active,
                    backend.zeros((6, n - (k + 1)), dtype=J_k_active.dtype),
                ),
                axis=1,
            )

            G_k = backend.asarray(self.Glist[k])
            M = M + backend.matmul(backend.matmul(J_k.T, G_k), J_k)

        # Symmetrize against floating-point drift
        M = 0.5 * (M + M.T)
        if use_cache:
            self._mass_matrix_cache[thetalist_key] = M
        return M

    def _mass_matrix_legacy(
        self, thetalist: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """Legacy mass matrix (incorrect, kept for backward compat). DO NOT USE."""
        backend = get_backend()
        use_cache = backend.is_concrete
        n = len(thetalist)

        # Per-link spatial inertia in the base frame: I_base[i] = Ad_i^T G_i Ad_i.
        eye_n = backend.eye(n, dtype=backend.float64)
        I_base = []
        for i in range(n):
            AdT_i = ad(self.forward_kinematics(thetalist[: i + 1], frame="space"))
            G_i = backend.asarray(self.Glist[i])
            I_base.append(backend.matmul(backend.matmul(AdT_i.T, G_i), AdT_i))

        J_full = self.jacobian(thetalist, frame="space")

        # Row i of M is (J_i^T I_base[i]) @ J_full. Placing each row via a
        # one-hot column keeps construction functional (no M[i, j] writes).
        M = backend.zeros((n, n), dtype=backend.float64)
        for i in range(n):
            Ji = J_full[:, i]
            row_i = backend.matmul(backend.matmul(Ji, I_base[i]), J_full)
            M = M + eye_n[i][:, None] * row_i
        M = 0.5 * (M + M.T)
        if use_cache:
            cache_key = self._concrete_cache_key(backend, thetalist)
            self._mass_matrix_cache[cache_key] = M
        return M

    def _mass_matrix_derivatives(
        self, thetalist: Union[NDArray[np.float64], List[float]], epsilon: float = 1e-6
    ) -> NDArray[np.float64]:
        """
        Central finite-difference approximation of dM/dtheta_k for
        every joint angle, cached so repeated calls avoid recomputing
        full mass matrices inside tight loops.
        """
        backend = get_backend()
        use_cache = backend.is_concrete
        if use_cache:
            cache_key = self._concrete_cache_key(backend, thetalist) + (
                float(epsilon),
            )
            cached = self._mass_matrix_derivative_cache.get(cache_key)
            if cached is not None:
                return cached

        n = len(thetalist)
        theta = backend.asarray(thetalist, dtype=backend.float64)
        eye_n = backend.eye(n, dtype=backend.float64)

        # Perturb one joint at a time via a one-hot column (theta +/- eps*e_k)
        # and place the slice at dM[:, :, k] with the same one-hot, avoiding
        # in-place writes so the finite-difference stays trace-safe.
        derivatives = backend.zeros((n, n, n), dtype=backend.float64)
        for k in range(n):
            e_k = eye_n[k]
            M_plus = self.mass_matrix(theta + epsilon * e_k)
            M_minus = self.mass_matrix(theta - epsilon * e_k)
            slice_k = (M_plus - M_minus) / (2.0 * epsilon)
            derivatives = derivatives + slice_k[:, :, None] * e_k

        if use_cache:
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
        """
        Compute Coriolis and centripetal generalized forces.

        Args:
            thetalist: Joint angles.
            dthetalist: Joint velocities.

        Returns:
            Joint-space velocity quadratic force vector.
        """
        backend = get_backend()
        n = len(thetalist)
        dtheta = backend.asarray(dthetalist, dtype=backend.float64)

        dM = self._mass_matrix_derivatives(thetalist)
        eye_n = backend.eye(n, dtype=backend.float64)

        # c_i = dtheta^T Gamma_i dtheta with the Christoffel matrix
        # Gamma_i[j, k] = 0.5 (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]).
        # This is the quadratic-form of the original triple sum; at zero
        # velocity every term vanishes, so the old zero-velocity short circuit
        # is unnecessary and would branch on tensor values.
        c = backend.zeros((n,), dtype=backend.float64)
        for i in range(n):
            dM_i = dM[i]
            gamma_i = 0.5 * (dM_i + dM_i.T - dM[:, :, i])
            c_i = backend.matmul(dtheta, backend.matmul(gamma_i, dtheta))
            c = c + c_i * eye_n[i]
        return c

    def gravity_forces(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        g: Optional[Union[NDArray[np.float64], List[float]]] = None,
    ) -> NDArray[np.float64]:
        """
        Compute joint torques needed to compensate gravity.

        Args:
            thetalist: Joint angles.
            g: Gravity vector. Defaults to ``[0, 0, -9.81]``.

        Returns:
            Joint-space gravity compensation torques.
        """
        backend = get_backend()
        if g is None:
            g = [0.0, 0.0, -9.81]
        g = backend.asarray(g, dtype=backend.float64)
        n = len(thetalist)

        if self.Mlist_per_link is None:
            # Legacy fallback for callers that build ManipulatorDynamics
            # manually without per-link CoM data. Mirrors mass_matrix.
            import warnings

            warnings.warn(
                "gravity_forces called without Mlist_per_link — using legacy "
                "approximation (incorrect for non-trivial robots). Construct "
                "ManipulatorDynamics via URDFToSerialManipulator to get accurate "
                "gravity compensation.",
                stacklevel=2,
            )
            return self._gravity_forces_legacy(thetalist, g)

        # g(θ)_i = Σ_k (J_k^T F_k)_i, where J_k is the body Jacobian of link k's
        # CoM and F_k = [0; m_k R_k^T (-g)] is the gravity-balancing wrench in
        # that CoM frame (Modern Robotics §8.3 / base accelerated by -g). The
        # per-link CoM Jacobian construction matches mass_matrix exactly.
        J_s = self.jacobian(thetalist, frame="space")  # (6, n)

        grav = backend.zeros((n,), dtype=backend.float64)
        for k in range(n):
            T_k_zero = backend.asarray(self.Mlist_per_link[k])
            T_k = self.forward_kinematics(thetalist[: k + 1], frame="space")
            T_k_at_zero = self.forward_kinematics(
                backend.zeros((k + 1,), dtype=backend.float64), frame="space"
            )
            T_link_to_com = backend.matmul(backend.inv(T_k_at_zero), T_k_zero)
            T_k_com = backend.matmul(T_k, T_link_to_com)

            # Columns i > k stay zero: joint i is downstream of link k.
            J_k_active = backend.matmul(ad(backend.inv(T_k_com)), J_s[:, : k + 1])
            J_k = backend.concatenate(
                (
                    J_k_active,
                    backend.zeros((6, n - (k + 1)), dtype=J_k_active.dtype),
                ),
                axis=1,
            )

            # Pure force m_k * R_k^T (-g) in the CoM body frame; no moment,
            # since the force acts through the CoM origin. [moment; force]
            # ordering pairs with the [omega; v] twist of J_k. The -g sign is
            # the v1.3.2 gravity correction (base accelerated by -g).
            m_k = backend.asarray(self.Glist[k])[3, 3]
            force = m_k * backend.matmul(T_k_com[:3, :3].T, -g)
            F = backend.concatenate((backend.zeros((3,), dtype=force.dtype), force))
            grav = grav + backend.matmul(J_k.T, F)

        return grav

    def _gravity_forces_legacy(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """Legacy gravity approximation (incorrect, kept for backward compat)."""
        backend = get_backend()
        n = len(thetalist)
        G = backend.asarray(g, dtype=backend.float64)
        eye_n = backend.eye(n, dtype=backend.float64)
        grav = backend.zeros((n,), dtype=backend.float64)
        for i in range(n):
            AdT = ad(self.forward_kinematics(thetalist[: i + 1], "space"))
            G_i = backend.asarray(self.Glist[i])
            val = backend.matmul(
                backend.matmul(AdT.T[:3, :3], G[:3]),
                backend.sum(G_i[:3, :3], axis=0),
            )
            grav = grav + val * eye_n[i]
        return grav

    def inverse_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        ddthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        Compute joint torques for the requested motion and end-effector wrench.

        Args:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            ddthetalist: Joint accelerations.
            g: Gravity vector.
            Ftip: End-effector wrench.

        Returns:
            Required joint torques.
        """
        backend = get_backend()
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        taulist = (
            backend.matmul(M, backend.asarray(ddthetalist))
            + c
            + g_forces
            + backend.matmul(J_transpose, backend.asarray(Ftip))
        )
        return taulist

    def forward_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        taulist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        Solve joint accelerations from applied torques and external loads.

        Args:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            taulist: Applied joint torques.
            g: Gravity vector.
            Ftip: End-effector wrench.

        Returns:
            Joint accelerations.
        """
        backend = get_backend()
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = (
            backend.asarray(taulist)
            - c
            - g_forces
            - backend.matmul(J_transpose, backend.asarray(Ftip))
        )
        ddthetalist = backend.solve(M, rhs)
        return ddthetalist
