#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PyTorch array backend - ManipulaPy

:class:`ArrayBackend` implementation backed by PyTorch. This module imports
``torch`` at the top level and therefore must never be imported eagerly: the
backend package registers it lazily behind an import probe (see
``ManipulaPy/backend/__init__.py``) so that ManipulaPy remains importable on
machines with only NumPy installed.

Unlike the NumPy and CuPy backends, PyTorch tensors can carry autograd / trace
state, so this backend reports ``is_concrete = False``. That gates the
value-keyed caches (e.g. the mass-matrix cache) off for traced inputs, keeping
gradients attached rather than detaching them through a host-synced cache key.
It defaults to the CPU device and ``gpu_capable = False`` so a CPU-default Torch
backend never routes numeric work to the Numba CUDA kernels; the numerics match
NumPy because floating-point construction is forced to ``float64`` on CPU.

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
import torch

from .base import ArrayBackend


# --- NumPy <-> Torch dtype bridge ------------------------------------------
# Bidirectional dtype map so promotion can be delegated to ``numpy.result_type``
# and the answer mapped back to a Torch dtype. Torch 2.7 exposes uint16/uint32/
# uint64/bool/complex dtypes, but many kernels (``eye``, ``matmul``, ``amax``,
# ``maximum`` on complex, ...) still reject them while NumPy succeeds. Rather
# than special-case each op, those exotic dtypes are routed through NumPy (see
# ``TorchBackend._numpy_fallback``) so the result matches NumPy by construction.
_NP_TO_TORCH_DTYPE = {
    np.bool_: torch.bool,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
_TORCH_TO_NP_DTYPE = {t: n for n, t in _NP_TO_TORCH_DTYPE.items()}
# Exotic Torch dtypes whose kernels do not (fully) match NumPy: bool, the
# unsigned integers, and complex. Operations on these dtypes are delegated to
# NumPy so the value *and* dtype equal NumPy's own result. They never carry
# autograd/trace state and never appear on ManipulaPy's float64 hot paths, so
# the ``.detach().cpu().numpy()`` round-trip is safe.
_FALLBACK_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.complex64,
        torch.complex128,
    }
)


def _torch_to_np_dtype(dtype: "torch.dtype") -> "np.dtype":
    """Map a Torch dtype to the equivalent NumPy dtype."""
    return np.dtype(_TORCH_TO_NP_DTYPE[dtype])


def _np_to_torch_dtype(dtype: Any) -> "torch.dtype":
    """Map a NumPy dtype (or dtype-like) to the equivalent Torch dtype."""
    return _NP_TO_TORCH_DTYPE[np.dtype(dtype).type]


def _np_result_dtype(*operands: Any) -> "torch.dtype":
    """Torch dtype NumPy would produce by promoting ``operands`` together.

    Promotion is delegated to ``numpy.result_type`` (NEP 50 rules): tensors and
    NumPy scalars contribute their dtype (array/strong semantics) while Python
    scalars contribute weak-scalar semantics, so the mapped Torch dtype matches
    what the NumPy backend returns for the same operands. ``None`` operands
    (e.g. an omitted clip bound) are skipped.
    """
    np_args = []
    for op in operands:
        if op is None:
            continue
        if isinstance(op, torch.Tensor):
            np_args.append(_torch_to_np_dtype(op.dtype))
        elif isinstance(op, np.generic):
            np_args.append(op.dtype)
        elif isinstance(op, (bool, int, float, complex)):
            np_args.append(op)
        else:
            np_args.append(np.asarray(op).dtype)
    return _np_to_torch_dtype(np.result_type(*np_args))


def _as_torch_dtype(dtype: Any) -> "torch.dtype":
    """Normalize a dtype argument to a ``torch.dtype``.

    Accepts a ``torch.dtype`` unchanged and maps NumPy dtype forms
    (``np.float32``, ``np.dtype("float32")``, ``"float32"``) through the bridge
    so the four constructors honour NumPy-style dtype arguments like NumPy does.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    return _np_to_torch_dtype(dtype)


def _np_reduce_dtype(t: "torch.Tensor") -> "torch.dtype":
    """Torch dtype of NumPy's additive accumulator (``sum``/``trace``) for ``t``.

    NumPy upcasts bool/int8/int16/int32 to int64 and uint8/uint16/uint32 to
    uint64 while preserving int64/uint64/float*/complex*; this delegates that
    rule to ``numpy.sum`` on a zero-size array of the same dtype.
    """
    np_dtype = np.zeros((), dtype=_torch_to_np_dtype(t.dtype)).sum().dtype
    return _np_to_torch_dtype(np_dtype)


class TorchBackend(ArrayBackend):
    """Backend backed by PyTorch tensors.

    Defaults to the CPU device with ``float64`` floating-point construction so
    numerics match the NumPy backend. Set ``device="cuda"`` to materialize
    tensors on the GPU; ``gpu_capable`` stays ``False`` regardless because the
    Numba CUDA kernel routing is a separate acceleration path.
    """

    float32 = torch.float32
    float64 = torch.float64
    #: Torch tensors can carry autograd/trace state and are not hashable by
    #: value, so value-keyed caches (e.g. the mass-matrix cache) must be
    #: bypassed rather than keyed by tensor identity or a host-synced value.
    is_concrete = False
    #: CPU-default Torch backend: never routes to the Numba CUDA kernel path.
    gpu_capable = False

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)

    def cache_token(self) -> Any:
        """Namespace cached arrays to this backend instance and device."""
        return (self, str(self._device))

    # -- construction ---------------------------------------------------
    @staticmethod
    def _contains_tensor(obj: Any) -> bool:
        """True iff ``obj`` is, or nests, a ``torch.Tensor`` leaf."""
        if isinstance(obj, torch.Tensor):
            return True
        if isinstance(obj, (list, tuple)):
            return any(TorchBackend._contains_tensor(el) for el in obj)
        return False

    @staticmethod
    def _native_byteorder(arr: "np.ndarray") -> "np.ndarray":
        """Return ``arr`` in native byte order (``torch.from_numpy`` requires it)."""
        if arr.dtype.byteorder not in ("=", "|"):
            return arr.astype(arr.dtype.newbyteorder("="))
        return arr

    def _to_tensor(self, obj: Any) -> "torch.Tensor":
        """Convert ``obj`` to a tensor, preserving tensor inputs (and autograd).

        A (possibly nested) sequence containing tensor leaves is assembled with
        ``torch.stack`` so autograd/trace state stays attached -- this is the
        build-matrix-from-computed-trig pattern in kinematics. The sequence's
        joint dtype is computed with weak-scalar semantics (Python scalars weak,
        tensors strong) so ``[int64_tensor, 2]`` stays int64 and
        ``[int64_tensor, float32_tensor]`` promotes to float64, matching NumPy.
        Pure Python/NumPy input routes through NumPy so Python floats become
        ``float64`` (matching NumPy) instead of Torch's ``float32`` default.
        """
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, np.ndarray):
            obj = self._native_byteorder(obj)
            if any(stride < 0 for stride in obj.strides):
                obj = np.ascontiguousarray(obj)
            return torch.from_numpy(obj)
        if isinstance(obj, (list, tuple)) and self._contains_tensor(obj):
            # Co-locate every leaf on this backend's device and cast it to the
            # sequence's NumPy-joint dtype before stacking: a CUDA-device
            # backend may stack a CUDA tensor leaf with a Python-scalar leaf
            # built on the CPU. ``.to(...)`` preserves autograd, and dtype/device
            # casts that are already satisfied (the all-float64 trig path) are
            # no-ops that keep the graph attached.
            target = self._seq_result_dtype(obj)
            return torch.stack(
                [self._stack_leaf(el).to(target).to(self._device) for el in obj]
            )
        return torch.as_tensor(np.asarray(obj))

    def _seq_result_dtype(self, obj: Any) -> "torch.dtype":
        """Joint Torch dtype for a (possibly nested) tensor-containing sequence.

        Leaves are gathered recursively so ``numpy.result_type`` sees each
        contributor with weak/strong semantics (Python scalars weak, tensors
        strong), matching what ``np.array`` would produce for the same nesting.
        """
        contributors: list = []

        def _walk(node: Any) -> None:
            if isinstance(node, (list, tuple)):
                for element in node:
                    _walk(element)
            else:
                contributors.append(node)

        _walk(obj)
        return _np_result_dtype(*contributors)

    def _stack_leaf(self, el: Any) -> "torch.Tensor":
        """Convert one element of a tensor-containing sequence to a tensor.

        Tensors and nested sequences defer to :meth:`_to_tensor`; Python scalars
        are materialized (as ``float64``) and cast to the sequence's joint dtype
        by the caller.
        """
        if isinstance(el, (torch.Tensor, list, tuple, np.ndarray)):
            return self._to_tensor(el)
        return torch.as_tensor(el, dtype=torch.float64)

    # -- NumPy fallback for exotic dtypes -------------------------------
    @classmethod
    def _needs_fallback(cls, *tensors: Any) -> bool:
        """True iff any tensor operand has an exotic (NumPy-delegated) dtype."""
        return any(
            isinstance(t, torch.Tensor) and t.dtype in _FALLBACK_DTYPES
            for t in tensors
        )

    @staticmethod
    def _is_python_int(value: Any) -> bool:
        """True for a Python ``int`` (weak scalar), excluding ``bool``."""
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def _as_numpy_operand(op: Any) -> Any:
        """View a tensor operand as NumPy; pass scalars/None/array-likes through.

        Detaching to a host NumPy view is safe here because the fallback only
        runs for non-float, torch-unsupported, or overflow cases, which never
        carry autograd/trace state.
        """
        if isinstance(op, torch.Tensor):
            return op.detach().cpu().numpy()
        return op

    def _from_numpy(self, result: Any) -> "torch.Tensor":
        """Materialize a NumPy op result as a tensor on this backend's device."""
        arr = self._native_byteorder(np.asarray(result))
        # ``np.ascontiguousarray`` upgrades a 0-D scalar to shape ``(1,)``; only
        # apply it to arrays so scalar reductions (e.g. ``sum(bool)``) keep the
        # 0-D shape NumPy returns.
        if arr.ndim > 0:
            arr = np.ascontiguousarray(arr)
        return torch.from_numpy(arr).to(self._device)

    def _numpy_fallback(self, np_func: Any, *operands: Any, **kwargs: Any) -> Any:
        """Compute ``np_func`` on NumPy views of ``operands`` and return a tensor.

        This yields exact NumPy parity by construction: the operands are exactly
        what the NumPy backend would receive (tensors as their host views, Python
        scalars kept weak), so the value and dtype equal NumPy's own result.
        """
        np_args = [self._as_numpy_operand(op) for op in operands]
        return self._from_numpy(np_func(*np_args, **kwargs))

    @staticmethod
    def _promote_pair(
        a: "torch.Tensor", b: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Cast ``a``/``b`` to the dtype NumPy would promote them to jointly.

        Shared by the binary ops (matmul/solve/maximum/minimum/cross) so a
        mixed-dtype pair follows ``np.result_type`` instead of Torch's own
        promotion (or a raise). Same-dtype pairs -- the all-float64 hot path --
        short-circuit with no ``result_type`` call and no copy.
        """
        if a.dtype == b.dtype:
            return a, b
        target = _np_result_dtype(a, b)
        return a.to(target), b.to(target)

    @staticmethod
    def _promote_float(t: "torch.Tensor") -> "torch.Tensor":
        """Promote integer/boolean tensors to ``float64`` to match NumPy.

        NumPy's transcendental, ``mean``, and linear-algebra operations return
        ``float64`` for integer (or boolean) input, whereas Torch either raises
        (``mean``/``norm``/linalg) or silently returns ``float32`` (``sin``,
        ``cos``, ``sqrt``, ...). Floating (and complex) tensors pass through
        unchanged so a ``float32`` input keeps its precision, exactly as NumPy
        preserves it.
        """
        if t.is_floating_point() or t.is_complex():
            return t
        return t.to(torch.float64)

    def array(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        t = self._to_tensor(obj)
        if dtype is not None:
            t = t.to(_as_torch_dtype(dtype))
        return t.clone().to(self._device)

    def asarray(self, obj: Any, dtype: Optional[Any] = None) -> Any:
        t = self._to_tensor(obj)
        if dtype is not None:
            t = t.to(_as_torch_dtype(dtype))
        return t.to(self._device)

    def zeros(self, shape: Any, dtype: Optional[Any] = None) -> Any:
        dtype = torch.float64 if dtype is None else _as_torch_dtype(dtype)
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def eye(self, n: int, dtype: Optional[Any] = None) -> Any:
        dtype = torch.float64 if dtype is None else _as_torch_dtype(dtype)
        # torch.eye rejects several exotic dtypes (uint16/32/64, ...) that NumPy
        # constructs; build via NumPy so the result matches np.eye by dtype.
        if dtype in _FALLBACK_DTYPES:
            return self._from_numpy(np.eye(n, dtype=_torch_to_np_dtype(dtype)))
        return torch.eye(n, dtype=dtype, device=self._device)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return torch.stack(self._promote_operands(arrays), dim=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return torch.cat(self._promote_operands(arrays), dim=axis)

    def _promote_operands(self, arrays: Any) -> list:
        """Tensorize ``arrays`` and cast them to their joint ``np.result_type``.

        np.stack/np.concatenate promote all inputs together (e.g. int64 + float32
        -> float64), whereas torch would keep float32; casting to the NumPy
        result dtype preserves that precision.
        """
        tensors = [self.asarray(a) for a in arrays]
        if not tensors:
            return tensors
        target = _np_result_dtype(*tensors)
        return [t if t.dtype == target else t.to(target) for t in tensors]

    def diag(self, v: Any) -> Any:
        return torch.diag(self.asarray(v))

    # -- linalg ---------------------------------------------------------
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return torch.linalg.svd(
            self._promote_float(self.asarray(a)), full_matrices=full_matrices
        )

    def svdvals(self, a: Any) -> Any:
        return torch.linalg.svdvals(self._promote_float(self.asarray(a)))

    def inv(self, a: Any) -> Any:
        return torch.linalg.inv(self._promote_float(self.asarray(a)))

    def pinv(self, a: Any) -> Any:
        # np.linalg.pinv zeroes singular values <= rcond * largest_sv with a
        # default rcond of 1e-15. torch.linalg.pinv's default rtol is far larger
        # (max(M, N) * eps), so a tiny sv would be inverted to ~1e15 instead of
        # zeroed; pass the NumPy-compatible relative cutoff as rtol to match.
        return torch.linalg.pinv(self._promote_float(self.asarray(a)), rtol=1e-15)

    def solve(self, a: Any, b: Any) -> Any:
        # Integer/bool operands upcast to float64 (torch.linalg.solve rejects
        # them) and mixed floating precisions promote jointly by np.result_type
        # so e.g. solve(float32, int64) -> float64 like NumPy.
        a = self._promote_float(self.asarray(a))
        b = self._promote_float(self.asarray(b))
        return torch.linalg.solve(*self._promote_pair(a, b))

    def norm(
        self, x: Any, ord: Optional[Any] = None, axis: Optional[Any] = None
    ) -> Any:
        return torch.linalg.norm(
            self._promote_float(self.asarray(x)), ord=ord, dim=axis
        )

    def trace(self, a: Any) -> Any:
        a = self.asarray(a)
        # torch.trace rejects bool/unsigned dtypes NumPy accepts; delegate those.
        if self._needs_fallback(a):
            return self._numpy_fallback(np.trace, a)
        target = _np_reduce_dtype(a)
        if a.ndim == 2 and a.dtype != torch.float16:
            out = torch.trace(a)
        else:
            # np.trace defaults to axis1=0, axis2=1 for stacked/batched input;
            # torch.trace only accepts 2D and rejects float16, so mirror NumPy
            # via the diagonal sum (which also handles the batched case).
            out = torch.diagonal(a, dim1=0, dim2=1).sum(dim=-1)
        return out if out.dtype == target else out.to(target)

    # -- elementwise ----------------------------------------------------
    def _transcendental(self, np_func: Any, torch_func: Any, x: Any) -> Any:
        """Apply a float-producing unary op, matching NumPy's dtype promotion.

        NumPy promotes integer/boolean input to a *width-dependent* float
        (int8/uint8/bool -> float16, int16/uint16 -> float32, ...), which Torch
        cannot reproduce, so integer/boolean input is delegated to NumPy. Float
        and complex input stay on the native Torch path (autograd-preserving).
        """
        x = self.asarray(x)
        if not (x.is_floating_point() or x.is_complex()):
            return self._numpy_fallback(np_func, x)
        return torch_func(x)

    def sin(self, x: Any) -> Any:
        return self._transcendental(np.sin, torch.sin, x)

    def cos(self, x: Any) -> Any:
        return self._transcendental(np.cos, torch.cos, x)

    def sqrt(self, x: Any) -> Any:
        return self._transcendental(np.sqrt, torch.sqrt, x)

    def arccos(self, x: Any) -> Any:
        return self._transcendental(np.arccos, torch.arccos, x)

    @staticmethod
    def _is_weak_real_scalar(value: Any) -> bool:
        """True for a Python ``int``/``float`` weak scalar (excluding ``bool``)."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def arctan2(self, y: Any, x: Any) -> Any:
        ty, tx = self.asarray(y), self.asarray(x)
        # np.arctan2 promotes both operands jointly to a width-dependent float and
        # raises on complex. Stay on the native (autograd-preserving) Torch path
        # when the computation is real-float: at least one float tensor, and every
        # other operand a float tensor or a weak Python real scalar. Otherwise
        # delegate to NumPy for exact parity.
        y_float, x_float = ty.is_floating_point(), tx.is_floating_point()
        native = (
            (y_float or x_float)
            and (y_float or self._is_weak_real_scalar(y))
            and (x_float or self._is_weak_real_scalar(x))
        )
        if not native:
            return self._numpy_fallback(np.arctan2, y, x)
        return torch.arctan2(*self._promote_weak(y, x))

    def abs(self, x: Any) -> Any:
        x = self.asarray(x)
        # torch.abs rejects bool and complex-parity differs; delegate exotics.
        if self._needs_fallback(x):
            return self._numpy_fallback(np.abs, x)
        return torch.abs(x)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        xt = self.asarray(x)
        # Delegate to NumPy for exotic dtypes and for an integer array bounded by
        # an OUT-OF-RANGE weak Python int, where np.clip range-handling differs
        # from torch.clamp (e.g. clip(int8, None, 128)). An in-range integer
        # bound (clip(int64 indices, 0, limit)) stays native and trace-safe.
        if (
            self._needs_fallback(xt)
            or self._int_scalar_overflows(a_min, xt)
            or self._int_scalar_overflows(a_max, xt)
        ):
            return self._numpy_fallback(np.clip, x, a_min, a_max)
        x = xt
        a_min = self._clip_bound(a_min)
        a_max = self._clip_bound(a_max)
        # np.clip promotes by np.result_type(x, a_min, a_max): e.g. int input
        # with a float32 bound -> float32, int64 with a python float -> float64,
        # all-integer bounds keep the integer dtype. Cast x and any tensor bound
        # to that dtype; scalar bounds are coerced by torch.clamp.
        target = _np_result_dtype(x, a_min, a_max)
        if x.dtype != target:
            x = x.to(target)
        # torch.clamp rejects a Tensor bound paired with a Number bound; when
        # either bound is array-valued, tensorize both so np.clip's one-array-
        # one-scalar form (which torch.clamp otherwise rejects) works.
        tensorize = isinstance(a_min, torch.Tensor) or isinstance(a_max, torch.Tensor)
        a_min = self._cast_bound(a_min, target, tensorize)
        a_max = self._cast_bound(a_max, target, tensorize)
        return torch.clamp(x, min=a_min, max=a_max)

    def _clip_bound(self, bound: Any) -> Any:
        """Pass scalar/``None`` bounds through; convert array-likes to tensors.

        ``np.clip`` accepts any array-like bound, but ``torch.clamp`` needs a
        Number or Tensor, so list/tuple/ndarray bounds are converted.
        """
        if bound is None or isinstance(bound, (int, float)):
            return bound
        return self.asarray(bound)

    def _cast_bound(self, bound: Any, dtype: "torch.dtype", tensorize: bool) -> Any:
        """Cast a clip bound to ``dtype`` as a Tensor/scalar for ``torch.clamp``.

        Tensor bounds are cast to ``dtype``; ``None`` passes through. Scalar
        bounds stay scalars for clamp's fast path unless ``tensorize`` (the other
        bound is a Tensor), in which case the scalar is broadcast to a Tensor so
        clamp does not reject a mixed Tensor/Number pair.
        """
        if bound is None:
            return None
        if isinstance(bound, torch.Tensor):
            return bound if bound.dtype == dtype else bound.to(dtype)
        if tensorize:
            return torch.as_tensor(bound, dtype=dtype, device=self._device)
        return bound

    def _promote_weak(
        self, x1: Any, x2: Any
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Tensorize ``x1``/``x2`` at their joint ``np.result_type``, keeping
        Python scalars weak (NEP 50).

        The dtype is computed from the *original* operands so a Python float/int
        contributes weak-scalar semantics: np.maximum(float32_array, 1.5) stays
        float32. Array operands promote strongly. Both are then materialized as
        tensors of that dtype for the binary op.
        """
        target = _np_result_dtype(x1, x2)
        return self.asarray(x1).to(target), self.asarray(x2).to(target)

    def _weak_int_overflow(
        self,
        x1: Any,
        x2: Any,
        t1: "torch.Tensor",
        t2: "torch.Tensor",
    ) -> bool:
        """True only when a weak Python int scalar is OUT OF RANGE for the
        integer array it is paired with.

        NumPy range-checks the scalar against the array's dtype (raising on
        overflow), which Torch's silent wrap does not reproduce -- but only when
        the scalar actually overflows. An in-range scalar (e.g. ``maximum(int64
        indices, 0)`` or ``clip(indices, 0, limit)``) must stay on the native
        Torch path so it remains trace-safe; the float autograd path (a float
        array + scalar) is excluded entirely.
        """
        return self._int_scalar_overflows(x1, t2) or self._int_scalar_overflows(
            x2, t1
        )

    @staticmethod
    def _int_scalar_overflows(scalar: Any, t: "torch.Tensor") -> bool:
        """True iff ``scalar`` is a Python int outside integer tensor ``t``'s
        dtype range (so NumPy would raise but Torch would silently wrap). A
        float/complex tensor or a non-int scalar never overflows here, keeping
        the native trace-safe path for in-range integer index operations."""
        if not (isinstance(scalar, int) and not isinstance(scalar, bool)):
            return False
        if t.is_floating_point() or t.is_complex():
            return False
        info = torch.iinfo(t.dtype)
        return not (info.min <= scalar <= info.max)

    def _binary_minmax(self, np_func: Any, torch_func: Any, x1: Any, x2: Any) -> Any:
        t1, t2 = self.asarray(x1), self.asarray(x2)
        if self._needs_fallback(t1, t2) or self._weak_int_overflow(x1, x2, t1, t2):
            return self._numpy_fallback(np_func, x1, x2)
        return torch_func(*self._promote_weak(x1, x2))

    def maximum(self, x1: Any, x2: Any) -> Any:
        return self._binary_minmax(np.maximum, torch.maximum, x1, x2)

    def minimum(self, x1: Any, x2: Any) -> Any:
        return self._binary_minmax(np.minimum, torch.minimum, x1, x2)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        tx, ty = self.asarray(x), self.asarray(y)
        if self._needs_fallback(tx, ty) or self._weak_int_overflow(x, y, tx, ty):
            return self._numpy_fallback(np.where, condition, x, y)
        cond = self.asarray(condition)
        if cond.dtype is not torch.bool:
            cond = cond.to(torch.bool)
        # np.where promotes x/y (not the condition) by np.result_type with weak
        # Python scalars, so mixed int64/float32 values stay float64 like NumPy.
        return torch.where(cond, *self._promote_weak(x, y))

    def cross(self, a: Any, b: Any) -> Any:
        ta, tb = self.asarray(a), self.asarray(b)
        # torch.linalg.cross rejects bool/unsigned/complex dtypes NumPy accepts.
        if self._needs_fallback(ta, tb):
            return self._numpy_fallback(np.cross, a, b)
        # np.cross promotes operands by np.result_type; cast both first so the
        # scalar-z and 3-vector results carry the NumPy dtype.
        a, b = self._promote_pair(ta, tb)
        la, lb = a.shape[-1], b.shape[-1]
        # np.cross of two 2-component vectors returns the scalar z-cross;
        # torch.linalg.cross requires length-3, so handle the 2D case directly.
        if la == 2 and lb == 2:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        # Mixed 2x3 / 3x2: np.cross treats the missing 3rd component as 0 and
        # returns a 3-vector; pad the 2-component operand with a zero z.
        if la == 2:
            a = self._pad_z(a)
        if lb == 2:
            b = self._pad_z(b)
        return torch.linalg.cross(a, b, dim=-1)

    @staticmethod
    def _pad_z(v: "torch.Tensor") -> "torch.Tensor":
        """Append a zero third component along the last axis (2-vec -> 3-vec)."""
        zeros = torch.zeros(
            (*v.shape[:-1], 1), dtype=v.dtype, device=v.device
        )
        return torch.cat((v, zeros), dim=-1)

    def matmul(self, a: Any, b: Any) -> Any:
        ta, tb = self.asarray(a), self.asarray(b)
        # torch.matmul rejects bool/unsigned/complex-mixed dtypes NumPy accepts.
        if self._needs_fallback(ta, tb):
            return self._numpy_fallback(np.matmul, a, b)
        return torch.matmul(*self._promote_pair(ta, tb))

    # -- reductions -----------------------------------------------------
    @staticmethod
    def _is_empty_axis(axis: Any) -> bool:
        """True for ``axis=()``/``[]``: NumPy reduces nothing, Torch reduces all."""
        return isinstance(axis, (tuple, list)) and len(axis) == 0

    def sum(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        # bool/unsigned/complex accumulate differently in torch; delegate them.
        if self._needs_fallback(x):
            return self._numpy_fallback(np.sum, x, axis=axis)
        # np.sum applies an additive accumulator upcast on every path (e.g.
        # bool/int8 -> int64), unlike amax/amin. torch.sum upcasts integers to
        # int64, so cast the result to NumPy's accumulator dtype.
        target = _np_reduce_dtype(x)
        if self._is_empty_axis(axis):
            # axis=() does not reduce but still applies the accumulator upcast.
            return x if x.dtype == target else x.to(target)
        out = torch.sum(x) if axis is None else torch.sum(x, dim=axis)
        return out if out.dtype == target else out.to(target)

    def amax(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        # torch.amax rejects bool/unsigned/complex dtypes NumPy accepts.
        if self._needs_fallback(x):
            return self._numpy_fallback(np.amax, x, axis=axis)
        if self._is_empty_axis(axis):
            return x  # np.amax(x, axis=()) returns x with its dtype preserved.
        return torch.amax(x) if axis is None else torch.amax(x, dim=axis)

    def amin(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        # torch.amin rejects bool/unsigned/complex dtypes NumPy accepts.
        if self._needs_fallback(x):
            return self._numpy_fallback(np.amin, x, axis=axis)
        if self._is_empty_axis(axis):
            return x  # np.amin(x, axis=()) returns x with its dtype preserved.
        return torch.amin(x) if axis is None else torch.amin(x, dim=axis)

    def mean(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self._promote_float(self.asarray(x))
        if self._is_empty_axis(axis):
            return x  # np.mean(x, axis=()) is x, float-promoted, unreduced.
        return torch.mean(x) if axis is None else torch.mean(x, dim=axis)

    def argmax(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        # torch.argmax rejects bool/unsigned dtypes NumPy accepts; delegate them.
        if self._needs_fallback(x):
            return self._numpy_fallback(np.argmax, x, axis=axis)
        return torch.argmax(x) if axis is None else torch.argmax(x, dim=axis)

    def all(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        if self._needs_fallback(x):
            return self._numpy_fallback(np.all, x, axis=axis)
        # np.all returns a bool result; torch.all on a numeric tensor would
        # return that numeric dtype, so reduce over bool.
        x = x.to(torch.bool)
        if self._is_empty_axis(axis):
            return x  # np.all(x, axis=()) is x cast to bool, unreduced.
        return torch.all(x) if axis is None else torch.all(x, dim=axis)

    def any(self, x: Any, axis: Optional[Any] = None) -> Any:
        x = self.asarray(x)
        if self._needs_fallback(x):
            return self._numpy_fallback(np.any, x, axis=axis)
        # np.any returns a bool result; reduce over bool for the same reason.
        x = x.to(torch.bool)
        if self._is_empty_axis(axis):
            return x  # np.any(x, axis=()) is x cast to bool, unreduced.
        return torch.any(x) if axis is None else torch.any(x, dim=axis)

    def isfinite(self, x: Any) -> Any:
        return torch.isfinite(self.asarray(x))

    # -- device ---------------------------------------------------------
    def to_device(self, x: Any) -> Any:
        return self.asarray(x).to(self._device)

    def to_numpy(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def ascontiguous(self, x: Any) -> Any:
        return self.asarray(x).contiguous()

    # -- predicate ------------------------------------------------------
    def is_backend_array(self, x: Any) -> bool:
        return isinstance(x, torch.Tensor)
