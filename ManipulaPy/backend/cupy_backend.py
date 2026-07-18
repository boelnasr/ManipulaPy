#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
CuPy array backend - ManipulaPy

GPU :class:`ArrayBackend` implementation backed by CuPy. This module imports
``cupy`` at the top level and therefore must never be imported eagerly: the
backend package registers it lazily behind an import probe (see
``ManipulaPy/backend/__init__.py``) so that ManipulaPy remains importable on
machines with only NumPy installed.

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

import cupy as cp

from .base import ArrayBackend


class CupyBackend(ArrayBackend):
    """GPU backend backed by CuPy."""

    float32 = cp.float32
    float64 = cp.float64
    #: CuPy arrays are concrete device values, so value-keyed caches are valid.
    is_concrete = True

    # -- construction ---------------------------------------------------
    def array(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return cp.array(obj, dtype=dtype)

    def asarray(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return cp.asarray(obj, dtype=dtype)

    def zeros(self, shape: Any, dtype: Optional[Any] = None) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def eye(self, n: int, dtype: Optional[Any] = None) -> Any:
        return cp.eye(n, dtype=dtype)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return cp.stack(arrays, axis=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return cp.concatenate(arrays, axis=axis)

    def diag(self, v: Any) -> Any:
        return cp.diag(v)

    # -- linalg ---------------------------------------------------------
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return cp.linalg.svd(a, full_matrices=full_matrices)

    def inv(self, a: Any) -> Any:
        return cp.linalg.inv(a)

    def pinv(self, a: Any) -> Any:
        return cp.linalg.pinv(a)

    def solve(self, a: Any, b: Any) -> Any:
        return cp.linalg.solve(a, b)

    def norm(
        self, x: Any, ord: Optional[Any] = None, axis: Optional[Any] = None
    ) -> Any:
        return cp.linalg.norm(x, ord=ord, axis=axis)

    def trace(self, a: Any) -> Any:
        return cp.trace(a)

    # -- elementwise ----------------------------------------------------
    def sin(self, x: Any) -> Any:
        return cp.sin(x)

    def cos(self, x: Any) -> Any:
        return cp.cos(x)

    def sqrt(self, x: Any) -> Any:
        return cp.sqrt(x)

    def arccos(self, x: Any) -> Any:
        return cp.arccos(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return cp.arctan2(y, x)

    def abs(self, x: Any) -> Any:
        return cp.abs(x)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        return cp.clip(x, a_min, a_max)

    def maximum(self, x1: Any, x2: Any) -> Any:
        return cp.maximum(x1, x2)

    def minimum(self, x1: Any, x2: Any) -> Any:
        return cp.minimum(x1, x2)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return cp.where(condition, x, y)

    def cross(self, a: Any, b: Any) -> Any:
        return cp.cross(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return cp.matmul(a, b)

    # -- reductions -----------------------------------------------------
    def sum(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.sum(x, axis=axis)

    def amax(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.amax(x, axis=axis)

    def amin(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.amin(x, axis=axis)

    def mean(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.mean(x, axis=axis)

    def argmax(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.argmax(x, axis=axis)

    def all(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.all(x, axis=axis)

    def any(self, x: Any, axis: Optional[Any] = None) -> Any:
        return cp.any(x, axis=axis)

    def isfinite(self, x: Any) -> Any:
        return cp.isfinite(x)

    # -- device ---------------------------------------------------------
    def to_device(self, x: Any) -> Any:
        return cp.asarray(x)

    def to_numpy(self, x: Any) -> Any:
        return cp.asnumpy(x)

    def ascontiguous(self, x: Any) -> Any:
        return cp.ascontiguousarray(x)

    # -- predicate ------------------------------------------------------
    def is_backend_array(self, x: Any) -> bool:
        return isinstance(x, cp.ndarray)
