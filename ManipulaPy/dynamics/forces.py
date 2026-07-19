#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Velocity- and gravity-force computation for manipulator dynamics."""

from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import manipulator_dynamics as _runtime


class _ForcesConcern:
    """Generalized-force methods installed on the public dynamics class."""

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
        backend = _runtime.get_backend()
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
        backend = _runtime.get_backend()
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
            J_k_active = backend.matmul(
                _runtime.ad(backend.inv(T_k_com)), J_s[:, : k + 1]
            )
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
        backend = _runtime.get_backend()
        n = len(thetalist)
        G = backend.asarray(g, dtype=backend.float64)
        eye_n = backend.eye(n, dtype=backend.float64)
        grav = backend.zeros((n,), dtype=backend.float64)
        for i in range(n):
            AdT = _runtime.ad(self.forward_kinematics(thetalist[: i + 1], "space"))
            G_i = backend.asarray(self.Glist[i])
            val = backend.matmul(
                backend.matmul(AdT.T[:3, :3], G[:3]),
                backend.sum(G_i[:3, :3], axis=0),
            )
            grav = grav + val * eye_n[i]
        return grav
