#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Concrete-backend cache helpers for manipulator dynamics."""

from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import manipulator_dynamics as _runtime


class _CacheConcern:
    """Methods that key and populate finite-difference dynamics caches."""

    @staticmethod
    def _concrete_cache_key(backend: Any, thetalist: Any) -> Tuple[Any, ...]:
        """Return a backend-scoped, host-normalized key for concrete arrays."""
        theta_host = backend.to_numpy(backend.asarray(thetalist, dtype=backend.float64))
        theta_key = tuple(float(value) for value in theta_host)
        return (backend.cache_token(), theta_key)

    def _mass_matrix_derivatives(
        self, thetalist: Union[NDArray[np.float64], List[float]], epsilon: float = 1e-6
    ) -> NDArray[np.float64]:
        """
        Central finite-difference approximation of dM/dtheta_k for
        every joint angle, cached so repeated calls avoid recomputing
        full mass matrices inside tight loops.
        """
        backend = _runtime.get_backend()
        use_cache = backend.is_concrete
        if use_cache:
            cache_key = self._concrete_cache_key(backend, thetalist) + (float(epsilon),)
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
