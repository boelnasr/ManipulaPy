#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mass-matrix computation for manipulator dynamics."""

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from . import manipulator_dynamics as _runtime


class _MassMatrixConcern:
    """Mass-matrix methods installed on the public dynamics class."""

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
        backend = _runtime.get_backend()
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
            Ad_inv_T_k_com = _runtime.ad(backend.inv(T_k_com))
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
        backend = _runtime.get_backend()
        use_cache = backend.is_concrete
        n = len(thetalist)

        # Per-link spatial inertia in the base frame: I_base[i] = Ad_i^T G_i Ad_i.
        eye_n = backend.eye(n, dtype=backend.float64)
        I_base = []
        for i in range(n):
            AdT_i = _runtime.ad(
                self.forward_kinematics(thetalist[: i + 1], frame="space")
            )
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
