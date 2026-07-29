#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JAX array backend - ManipulaPy

:class:`ArrayBackend` implementation backed by JAX. This module imports ``jax``
at the top level and therefore must never be imported eagerly: the backend
package registers it lazily behind an import probe (see
``ManipulaPy/backend/__init__.py``) so that ManipulaPy remains importable on
machines with only NumPy installed.

Like the Torch backend, JAX arrays can be traced values (``jax.jit`` /
``jax.grad`` tracers), so this backend reports ``is_concrete = False``: the
value-keyed caches (e.g. the mass-matrix cache) are gated off rather than keyed
on a host-synced value that a tracer cannot provide. ``gpu_capable = False``
keeps numeric work off the Numba CUDA kernel path regardless of the JAX
platform, which is a separate acceleration route. JAX arrays are immutable, so
every operation here allocates a new array; nothing is updated in place.

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

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import ArrayBackend

# JAX defaults every floating-point array to float32; ManipulaPy's numerics (and
# every float64 parity assertion against the NumPy backend) require float64, so
# x64 is enabled here, at import time, before this backend can build an array.
# This flips process-global JAX state: it affects all JAX code in the process,
# not just ManipulaPy's. It is set on import of this module -- which only
# happens when the 'jax' backend is actually requested -- and is not reverted.
jax.config.update("jax_enable_x64", True)


# --- NumPy dtype promotion bridge -------------------------------------------
# JAX has its own promotion lattice, which deliberately avoids widening to
# float64: ``int64 + float32 -> float32`` where NumPy gives ``float64``. Every
# multi-operand op therefore computes its result dtype with ``numpy.result_type``
# and casts the operands to it, so the answer matches the NumPy backend by
# construction. Casts are ``convert_element_type`` primitives, so they stay
# traceable and differentiable.
def _np_result_dtype(*operands: Any) -> "np.dtype":
    """NumPy's promoted dtype for ``operands`` (NEP 50 rules).

    Arrays (JAX or NumPy) contribute their dtype with strong semantics while
    Python scalars contribute weak-scalar semantics, exactly as NumPy sees the
    same operands. ``None`` operands (e.g. an omitted clip bound) are skipped.
    """
    np_args: List[Any] = []
    for op in operands:
        if op is None:
            continue
        if isinstance(op, (bool, int, float, complex)):
            np_args.append(op)
        elif hasattr(op, "dtype"):
            np_args.append(np.dtype(op.dtype))
        else:
            np_args.append(jnp.asarray(op).dtype)
    return np.result_type(*np_args)


def _is_complex(operand: Any) -> bool:
    """True if ``operand`` carries a complex dtype."""
    dtype = getattr(operand, "dtype", None)
    return dtype is not None and np.issubdtype(np.dtype(dtype), np.complexfloating)


def _np_inexact_dtype(*operands: Any) -> "np.dtype":
    """Dtype NumPy's transcendental ufuncs produce for ``operands``.

    NumPy promotes integer/boolean input to a *width-dependent* float
    (bool/int8/uint8 -> float16, int16/uint16 -> float32, int32/int64 ->
    float64), which ``np.result_type(..., float16)`` reproduces exactly; JAX
    would instead send every narrow integer to float32. Floating and complex
    input passes through unchanged, so a float32 input keeps its precision.
    """
    return np.result_type(_np_result_dtype(*operands), np.float16)


class JaxBackend(ArrayBackend):
    """Backend backed by JAX arrays.

    Runs on whichever platform JAX selects by default (CPU unless a GPU/TPU
    jaxlib is installed). ``gpu_capable`` stays ``False`` regardless because the
    Numba CUDA kernel routing is a separate acceleration path.

    Every method funnels its operands through :meth:`asarray` first: JAX's
    operations reject Python lists and tuples outright ("requires ndarray or
    scalar arguments") and reject byte-swapped NumPy dtypes, whereas the backend
    contract mirrors NumPy and accepts any array-like.
    """

    float32 = jnp.float32
    float64 = jnp.float64
    #: JAX arrays may be ``jax.jit``/``jax.grad`` tracers, whose values are not
    #: host-readable, so value-keyed caches (e.g. the mass-matrix cache) must be
    #: bypassed rather than keyed on a host-synced value.
    is_concrete = False
    #: Never routes to the Numba CUDA kernel path, whatever platform JAX uses.
    gpu_capable = False

    def cache_token(self) -> Any:
        """Namespace cached arrays to this backend instance and JAX platform."""
        return (self, jax.default_backend())

    # -- promotion helpers ----------------------------------------------
    @staticmethod
    def _native_byteorder(obj: Any) -> Any:
        """Return NumPy input in native byte order, passing anything else through.

        JAX rejects a byte-swapped dtype outright ("Dtype >f8 is not a valid JAX
        array type") where NumPy operates on it happily, so such an array is
        converted before it reaches JAX.
        """
        if isinstance(obj, np.ndarray) and obj.dtype.byteorder not in ("=", "|"):
            return obj.astype(obj.dtype.newbyteorder("="))
        return obj

    def _promote(self, *operands: Any) -> List[Any]:
        """Return ``operands`` as arrays of their joint ``np.result_type``."""
        target = _np_result_dtype(*operands)
        return [self.asarray(op, dtype=target) for op in operands]

    def _numpy_fallback(self, np_func: Any, *operands: Any, **kwargs: Any) -> Any:
        """Run ``np_func`` on the host and convert the result back.

        JAX declines the handful of complex operations NumPy defines by
        lexicographic ``(real, imag)`` ordering: it truth-tests a complex value
        on its real part alone, and refuses to order complex values at all.
        Delegating those to NumPy gives parity by construction rather than by
        reimplementation, which is the same strategy the Torch backend uses for
        the dtypes Torch handles differently.

        Complex values bake into a trace and are never differentiated anywhere
        in ManipulaPy, so nothing is lost by leaving the accelerator here.
        """
        host = [
            np.asarray(op) if hasattr(op, "dtype") else op for op in operands
        ]
        return self.asarray(np_func(*host, **kwargs))

    def _promote_float(self, x: Any) -> Any:
        """Promote integer/boolean input to ``float64`` to match NumPy.

        NumPy's ``mean`` and linear-algebra operations return ``float64`` for
        integer (or boolean) input, whereas JAX returns ``float32`` for every
        integer narrower than int64. Floating and complex input passes through
        unchanged, exactly as NumPy preserves it.
        """
        x = self.asarray(x)
        return x if jnp.issubdtype(x.dtype, jnp.inexact) else x.astype(jnp.float64)

    # -- construction ---------------------------------------------------
    def array(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return jnp.array(self._native_byteorder(obj), dtype=dtype)

    def asarray(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        return jnp.asarray(self._native_byteorder(obj), dtype=dtype)

    def zeros(self, shape: Any, dtype: Optional[Any] = None) -> Any:
        return jnp.zeros(shape, dtype=dtype)

    def eye(self, n: int, dtype: Optional[Any] = None) -> Any:
        return jnp.eye(n, dtype=dtype)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return jnp.stack(self._promote_operands(arrays), axis=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return jnp.concatenate(self._promote_operands(arrays), axis=axis)

    def _promote_operands(self, arrays: Any) -> Any:
        """Cast a stack/concatenate operand sequence to its joint dtype.

        np.stack/np.concatenate promote all inputs together (e.g. int64 + float32
        -> float64), whereas JAX would keep float32. An empty sequence is passed
        through so JAX raises its own "need at least one array" error.
        """
        arrays = list(arrays)
        if not arrays:
            return arrays
        return self._promote(*arrays)

    def diag(self, v: Any) -> Any:
        return jnp.diag(self.asarray(v))

    # -- linalg ---------------------------------------------------------
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return jnp.linalg.svd(self._promote_float(a), full_matrices=full_matrices)

    def svdvals(self, a: Any) -> Any:
        return jnp.linalg.svd(self._promote_float(a), compute_uv=False)

    def inv(self, a: Any) -> Any:
        return jnp.linalg.inv(self._promote_float(a))

    def pinv(self, a: Any) -> Any:
        # np.linalg.pinv zeroes singular values <= rcond * largest_sv with a
        # default rcond of 1e-15. jnp.linalg.pinv's default rtol is larger
        # (max(M, N) * eps), so a small sv NumPy keeps would be zeroed instead;
        # pass the NumPy-compatible relative cutoff as rtol to match.
        a = self._promote_float(a)
        rtol = 1e-15
        if a.dtype == jnp.float32:
            # NumPy evaluates ``s > rcond * max(s)`` in float64 even for float32
            # input, while JAX compares in the input precision. A float32
            # singular value landing exactly on the cutoff is therefore dropped
            # here but kept by NumPy; stepping the tolerance one ulp toward zero
            # restores the strict inequality NumPy effectively applies.
            rtol = float(np.nextafter(np.float32(1e-15), np.float32(0)))
        return jnp.linalg.pinv(a, rtol=rtol)

    def solve(self, a: Any, b: Any) -> Any:
        # Mixed precisions promote jointly by np.result_type (so e.g.
        # solve(float32, int64) -> float64 like NumPy), then integer/boolean
        # operands upcast to float64 as NumPy's solve does.
        a, b = self._promote(a, b)
        return jnp.linalg.solve(self._promote_float(a), self._promote_float(b))

    def norm(
        self, x: Any, ord: Optional[Any] = None, axis: Optional[Any] = None
    ) -> Any:
        return jnp.linalg.norm(self._promote_float(x), ord=ord, axis=axis)

    def trace(self, a: Any) -> Any:
        return jnp.trace(self.asarray(a))

    # -- elementwise ----------------------------------------------------
    def _transcendental(self, jnp_func: Any, x: Any) -> Any:
        """Apply a float-producing unary op, matching NumPy's dtype promotion."""
        return jnp_func(self.asarray(x, dtype=_np_inexact_dtype(x)))

    def sin(self, x: Any) -> Any:
        return self._transcendental(jnp.sin, x)

    def cos(self, x: Any) -> Any:
        return self._transcendental(jnp.cos, x)

    def sqrt(self, x: Any) -> Any:
        return self._transcendental(jnp.sqrt, x)

    def arccos(self, x: Any) -> Any:
        return self._transcendental(jnp.arccos, x)

    def arctan2(self, y: Any, x: Any) -> Any:
        # np.arctan2 promotes both operands jointly to a width-dependent float.
        target = _np_inexact_dtype(y, x)
        return jnp.arctan2(self.asarray(y, dtype=target), self.asarray(x, dtype=target))

    def abs(self, x: Any) -> Any:
        return jnp.abs(self.asarray(x))

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        # JAX refuses to order complex values; NumPy clips them lexicographically.
        if _is_complex(x) or _is_complex(a_min) or _is_complex(a_max):
            return self._numpy_fallback(np.clip, x, a_min, a_max)
        # np.clip promotes by np.result_type(x, a_min, a_max): e.g. an int array
        # with a float32 bound -> float32, int64 with a Python float -> float64,
        # all-integer bounds keep the integer dtype. Python scalar bounds stay
        # weak, so clip(float32, -1.0, 1.0) keeps float32 like NumPy.
        target = _np_result_dtype(x, a_min, a_max)
        return jnp.clip(
            self.asarray(x, dtype=target),
            self._clip_bound(a_min, target),
            self._clip_bound(a_max, target),
        )

    def _clip_bound(self, bound: Any, dtype: "np.dtype") -> Any:
        """Cast a clip bound to ``dtype``; ``None`` (an open bound) passes through."""
        return None if bound is None else self.asarray(bound, dtype=dtype)

    def maximum(self, x1: Any, x2: Any) -> Any:
        return jnp.maximum(*self._promote(x1, x2))

    def minimum(self, x1: Any, x2: Any) -> Any:
        return jnp.minimum(*self._promote(x1, x2))

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        # np.where promotes x/y but not the condition, whose dtype is left alone
        # so a numeric condition is truth-tested exactly as NumPy does.
        return jnp.where(self.asarray(condition), *self._promote(x, y))

    def cross(self, a: Any, b: Any) -> Any:
        return jnp.cross(*self._promote(a, b))

    def matmul(self, a: Any, b: Any) -> Any:
        return jnp.matmul(*self._promote(a, b))

    # -- reductions -----------------------------------------------------
    def sum(self, x: Any, axis: Optional[Any] = None) -> Any:
        return jnp.sum(self.asarray(x), axis=axis)

    def amax(self, x: Any, axis: Optional[Any] = None) -> Any:
        return jnp.amax(self.asarray(x), axis=axis)

    def amin(self, x: Any, axis: Optional[Any] = None) -> Any:
        return jnp.amin(self.asarray(x), axis=axis)

    def mean(self, x: Any, axis: Optional[Any] = None) -> Any:
        return jnp.mean(self._promote_float(x), axis=axis)

    def argmax(self, x: Any, axis: Optional[Any] = None) -> Any:
        # NumPy orders complex values lexicographically by (real, imag); JAX
        # raises rather than ordering them at all.
        if _is_complex(x):
            return self._numpy_fallback(np.argmax, x, axis=axis)
        return jnp.argmax(self.asarray(x), axis=axis)

    def all(self, x: Any, axis: Optional[Any] = None) -> Any:
        # A complex value is truthy in NumPy when EITHER part is non-zero; JAX
        # tests the real part alone, so 1j would read as False.
        if _is_complex(x):
            return self._numpy_fallback(np.all, x, axis=axis)
        return jnp.all(self.asarray(x), axis=axis)

    def any(self, x: Any, axis: Optional[Any] = None) -> Any:
        if _is_complex(x):
            return self._numpy_fallback(np.any, x, axis=axis)
        return jnp.any(self.asarray(x), axis=axis)

    def isfinite(self, x: Any) -> Any:
        return jnp.isfinite(self.asarray(x))

    # -- device ---------------------------------------------------------
    def to_device(self, x: Any) -> Any:
        # Arrays are converted first: jax.device_put treats a list as a pytree
        # and would return a list of arrays instead of one array.
        return jax.device_put(self.asarray(x))

    def to_numpy(self, x: Any) -> Any:
        return np.asarray(x)

    def ascontiguous(self, x: Any) -> Any:
        # JAX arrays own a dense, contiguous buffer, so this is just conversion.
        return self.asarray(x)

    # -- predicate ------------------------------------------------------
    def is_backend_array(self, x: Any) -> bool:
        return isinstance(x, jax.Array)
