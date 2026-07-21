#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Backend dispatch package - ManipulaPy

Provides a single process-wide array backend that numerical modules dispatch
through instead of importing NumPy or CuPy directly. NumPy is the default and
the only backend loaded eagerly; CuPy is registered lazily behind an import
probe so ManipulaPy stays importable on machines without CuPy.

Selection API
-------------
- ``register(name, backend)``   register a backend instance (duplicate name -> error)
- ``set_backend(name)``         switch the active backend process-wide
- ``use_backend(name)``         context manager: scoped switch, always restored
- ``get_backend()``             resolve the active :class:`ArrayBackend`
- ``get_registered(name)``      look up a registered backend by name

Thread-safety
-------------
The active-backend reference and the registry are guarded by a single
re-entrant lock. There is one active backend at a time, shared across all
threads; there is no per-call backend argument and no thread-local stack.

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

import importlib.util
import threading
from contextlib import contextmanager
from typing import Dict, Iterator

from .base import ArrayBackend
from .numpy_backend import NumpyBackend

__all__ = [
    "ArrayBackend",
    "register",
    "set_backend",
    "use_backend",
    "get_backend",
    "get_registered",
]

_LOCK = threading.RLock()
_REGISTRY: Dict[str, ArrayBackend] = {}
_active: ArrayBackend


def register(name: str, backend: ArrayBackend) -> None:
    """Register ``backend`` under ``name``.

    Raises:
        ValueError: if ``name`` is already registered. Names are never
            silently overwritten.
    """
    with _LOCK:
        if name in _REGISTRY:
            raise ValueError(f"Backend {name!r} is already registered")
        _REGISTRY[name] = backend


def _ensure_cupy_registered() -> None:
    """Register the CuPy backend on first request, if CuPy is importable.

    Raises:
        ImportError: if CuPy is not installed. Kept out of the eager import
            path so NumPy-only machines can import ManipulaPy.
    """
    with _LOCK:
        if "cupy" in _REGISTRY:
            return
        if importlib.util.find_spec("cupy") is None:
            raise ImportError(
                "The 'cupy' backend was requested but CuPy is not installed. "
                "Install a matching CUDA build, e.g. `pip install ManipulaPy[cuda]`, "
                "or select another backend such as 'numpy'."
            )
        from .cupy_backend import CupyBackend

        _REGISTRY["cupy"] = CupyBackend()


def _ensure_torch_registered() -> None:
    """Register the Torch backend on first request, if Torch is importable.

    Raises:
        ImportError: if PyTorch is not installed. Kept out of the eager import
            path so NumPy-only machines can import ManipulaPy.
    """
    with _LOCK:
        if "torch" in _REGISTRY:
            return
        if importlib.util.find_spec("torch") is None:
            raise ImportError(
                "The 'torch' backend was requested but PyTorch is not installed. "
                "Install PyTorch, e.g. `pip install torch`, "
                "or select another backend such as 'numpy'."
            )
        from .torch_backend import TorchBackend

        _REGISTRY["torch"] = TorchBackend()


def get_registered(name: str) -> ArrayBackend:
    """Return the backend registered under ``name``.

    Triggers lazy CuPy registration for ``name == "cupy"`` and lazy Torch
    registration for ``name == "torch"``.

    Raises:
        ValueError: if ``name`` is not registered (message lists the
            registered names).
        ImportError: if ``name == "cupy"`` but CuPy is not installed, or
            ``name == "torch"`` but PyTorch is not installed.
    """
    if name == "cupy":
        _ensure_cupy_registered()
    elif name == "torch":
        _ensure_torch_registered()
    with _LOCK:
        if name not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown backend {name!r}. Registered backends: {known}"
            )
        return _REGISTRY[name]


def set_backend(name: str) -> None:
    """Switch the active backend process-wide.

    Selection is explicit opt-in only: there is no environment sniffing and
    no silent promotion to CuPy.

    Raises:
        ValueError: if ``name`` is not registered.
        ImportError: if ``name == "cupy"`` but CuPy is not installed.
    """
    backend = get_registered(name)
    global _active
    with _LOCK:
        _active = backend


def get_backend() -> ArrayBackend:
    """Return the currently active :class:`ArrayBackend`."""
    with _LOCK:
        return _active


@contextmanager
def use_backend(name: str) -> Iterator[ArrayBackend]:
    """Temporarily switch to backend ``name`` for the duration of the block.

    The previously active backend is restored on exit, including when the
    body raises.
    """
    previous = get_backend()
    set_backend(name)
    try:
        yield get_backend()
    finally:
        with _LOCK:
            global _active
            _active = previous


# NumPy is registered and activated eagerly; it is the process default.
register("numpy", NumpyBackend())
_active = _REGISTRY["numpy"]
