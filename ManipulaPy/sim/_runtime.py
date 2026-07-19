"""Shared patchable runtime bindings for simulation concerns."""

# ruff: noqa: F401 - these names are the runtime's compatibility surface.

import sys as _sys
from types import ModuleType as _ModuleType

from ManipulaPy.backend import get_backend

try:
    import pybullet as p  # Required for Simulation; sim cannot run without it
    import pybullet_data

    _PYBULLET_AVAILABLE = True
except ImportError:
    p = None
    pybullet_data = None
    _PYBULLET_AVAILABLE = False


def _check_pybullet_available() -> None:
    """Raise a clear ImportError if pybullet is unavailable.

    __init__ already does this, but every public method that touches p.*
    needs the same check at runtime — users can bypass __init__ via
    ``Simulation.__new__`` (tests do), or hot-swap the pybullet module after
    construction. Without this, those paths surface confusing
    ``AttributeError: 'NoneType' object has no attribute ...`` instead.
    """
    if not _PYBULLET_AVAILABLE or p is None:
        raise ImportError(
            "pybullet is required for this Simulation operation. "
            "Install with: pip install 'ManipulaPy[simulation]'"
        )


_FORWARDED_RUNTIME_NAMES = frozenset(
    {
        "p",
        "pybullet_data",
        "_PYBULLET_AVAILABLE",
        "_check_pybullet_available",
        "get_backend",
    }
)


class _SimCompatibilityModule(_ModuleType):
    def __getattribute__(self, name):
        if name in _FORWARDED_RUNTIME_NAMES:
            namespace = globals()
            if name not in namespace:
                raise AttributeError(
                    f"module {self.__name__!r} has no attribute {name!r}"
                )
            return namespace[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in _FORWARDED_RUNTIME_NAMES:
            globals()[name] = value
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _FORWARDED_RUNTIME_NAMES:
            runtime_namespace = globals()
            simulation_namespace = super().__getattribute__("__dict__")
            if name not in runtime_namespace and name not in simulation_namespace:
                raise AttributeError(
                    f"module {self.__name__!r} has no attribute {name!r}"
                )
            runtime_namespace.pop(name, None)
            simulation_namespace.pop(name, None)
            return
        super().__delattr__(name)


def _install_compatibility_proxy(module):
    module.__class__ = _SimCompatibilityModule
