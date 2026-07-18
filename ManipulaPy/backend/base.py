#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Array backend protocol - ManipulaPy

Defines :class:`ArrayBackend`, the abstract interface that every numerical
backend (NumPy, CuPy, ...) implements. The surface mirrors the numeric
primitives used across ManipulaPy's core modules so that call sites can
dispatch through a single active backend instead of importing NumPy or
CuPy directly.

The method set was fixed by a call-site audit of the ten target modules;
it is intentionally minimal. Signatures mirror NumPy semantics so an
existing ``np.<name>(...)`` call can be swapped for ``backend.<name>(...)``
without changing arguments. No method mutates its input.

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

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class ArrayBackend(ABC):
    """Abstract array backend.

    Concrete subclasses wrap a numeric library (NumPy, CuPy, ...) and expose
    a fixed surface of construction, linear-algebra, elementwise, reduction,
    and device primitives. All operations mirror NumPy semantics and never
    mutate their inputs.
    """

    #: dtype handle usable in ``array(..., dtype=...)`` / ``astype(...)``.
    float32: Any
    #: dtype handle usable in ``array(..., dtype=...)`` / ``astype(...)``.
    float64: Any
    #: True iff this backend's arrays are concrete (host/device materialized)
    #: and hashable by value, so value-keyed caches (e.g. the mass-matrix
    #: cache keyed on ``tuple(thetalist)``) are valid. Future traced backends
    #: (Torch/JAX under a tracing pass) override this to ``False`` so those
    #: caches are bypassed rather than keyed by tensor identity or a
    #: host-synced value that would break trace-safety.
    is_concrete: bool

    @abstractmethod
    def cache_token(self) -> Any:
        """Return a hashable namespace for concrete value caches.

        The token must distinguish backend instances and any device context
        that determines where cached arrays are materialized.
        """
        ...

    # -- construction ---------------------------------------------------
    @abstractmethod
    def array(self, obj: Any, dtype: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def asarray(self, obj: Any, dtype: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def zeros(self, shape: Any, dtype: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def eye(self, n: int, dtype: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def stack(self, arrays: Any, axis: int = 0) -> Any: ...

    @abstractmethod
    def concatenate(self, arrays: Any, axis: int = 0) -> Any: ...

    @abstractmethod
    def diag(self, v: Any) -> Any: ...

    # -- linalg ---------------------------------------------------------
    @abstractmethod
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]: ...

    @abstractmethod
    def svdvals(self, a: Any) -> Any: ...

    @abstractmethod
    def inv(self, a: Any) -> Any: ...

    @abstractmethod
    def pinv(self, a: Any) -> Any: ...

    @abstractmethod
    def solve(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def norm(
        self, x: Any, ord: Optional[Any] = None, axis: Optional[Any] = None
    ) -> Any: ...

    @abstractmethod
    def trace(self, a: Any) -> Any: ...

    # -- elementwise ----------------------------------------------------
    @abstractmethod
    def sin(self, x: Any) -> Any: ...

    @abstractmethod
    def cos(self, x: Any) -> Any: ...

    @abstractmethod
    def sqrt(self, x: Any) -> Any: ...

    @abstractmethod
    def arccos(self, x: Any) -> Any: ...

    @abstractmethod
    def arctan2(self, y: Any, x: Any) -> Any: ...

    @abstractmethod
    def abs(self, x: Any) -> Any: ...

    @abstractmethod
    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any: ...

    @abstractmethod
    def maximum(self, x1: Any, x2: Any) -> Any: ...

    @abstractmethod
    def minimum(self, x1: Any, x2: Any) -> Any: ...

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any: ...

    @abstractmethod
    def cross(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any: ...

    # -- reductions -----------------------------------------------------
    @abstractmethod
    def sum(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def amax(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def amin(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def mean(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def argmax(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def all(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def any(self, x: Any, axis: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def isfinite(self, x: Any) -> Any: ...

    # -- device ---------------------------------------------------------
    @abstractmethod
    def to_device(self, x: Any) -> Any:
        """Move ``x`` onto this backend's device / array type."""
        ...

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """Return ``x`` as a host NumPy array."""
        ...

    @abstractmethod
    def ascontiguous(self, x: Any) -> Any:
        """Return a C-contiguous version of ``x``."""
        ...

    # -- predicate ------------------------------------------------------
    @abstractmethod
    def is_backend_array(self, x: Any) -> bool:
        """True iff ``x`` is this backend's native array type."""
        ...
