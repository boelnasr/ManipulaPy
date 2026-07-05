#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
NumPy array backend - ManipulaPy

The default :class:`ArrayBackend` implementation. Every method is a thin
forward to the corresponding NumPy primitive, preserving NumPy semantics so
call sites can dispatch through the backend without behavioural change.

Copyright (c) 2025 Mohamed Aboelnasr

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

from typing import Any, Optional, Tuple

import numpy as np

from .base import ArrayBackend


class NumpyBackend(ArrayBackend):
    """Host-CPU backend backed by NumPy."""

    float32 = np.float32
    float64 = np.float64

    # -- construction ---------------------------------------------------
    def array(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return np.array(obj, dtype=dtype)

    def asarray(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return np.asarray(obj, dtype=dtype)

    def zeros(self, shape: Any, dtype: Optional[Any] = None) -> Any:
        return np.zeros(shape, dtype=dtype)

    def eye(self, n: int, dtype: Optional[Any] = None) -> Any:
        return np.eye(n, dtype=dtype)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return np.concatenate(arrays, axis=axis)

    def diag(self, v: Any) -> Any:
        return np.diag(v)

    # -- linalg ---------------------------------------------------------
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return np.linalg.svd(a, full_matrices=full_matrices)

    def inv(self, a: Any) -> Any:
        return np.linalg.inv(a)

    def pinv(self, a: Any) -> Any:
        return np.linalg.pinv(a)

    def solve(self, a: Any, b: Any) -> Any:
        return np.linalg.solve(a, b)

    def norm(
        self, x: Any, ord: Optional[Any] = None, axis: Optional[Any] = None
    ) -> Any:
        return np.linalg.norm(x, ord=ord, axis=axis)

    def trace(self, a: Any) -> Any:
        return np.trace(a)

    # -- elementwise ----------------------------------------------------
    def sin(self, x: Any) -> Any:
        return np.sin(x)

    def cos(self, x: Any) -> Any:
        return np.cos(x)

    def sqrt(self, x: Any) -> Any:
        return np.sqrt(x)

    def arccos(self, x: Any) -> Any:
        return np.arccos(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return np.arctan2(y, x)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        return np.clip(x, a_min, a_max)

    def maximum(self, x1: Any, x2: Any) -> Any:
        return np.maximum(x1, x2)

    def minimum(self, x1: Any, x2: Any) -> Any:
        return np.minimum(x1, x2)

    def cross(self, a: Any, b: Any) -> Any:
        return np.cross(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    # -- reductions -----------------------------------------------------
    def sum(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.sum(x, axis=axis)

    def amax(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.amax(x, axis=axis)

    def amin(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.amin(x, axis=axis)

    def mean(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.mean(x, axis=axis)

    def argmax(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.argmax(x, axis=axis)

    def all(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.all(x, axis=axis)

    def any(self, x: Any, axis: Optional[Any] = None) -> Any:
        return np.any(x, axis=axis)

    def isfinite(self, x: Any) -> Any:
        return np.isfinite(x)

    # -- device ---------------------------------------------------------
    def to_device(self, x: Any) -> Any:
        return np.asarray(x)

    def to_numpy(self, x: Any) -> Any:
        return np.asarray(x)

    def ascontiguous(self, x: Any) -> Any:
        return np.ascontiguousarray(x)

    # -- predicate ------------------------------------------------------
    def is_backend_array(self, x: Any) -> bool:
        return isinstance(x, np.ndarray)
