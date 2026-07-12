#!/usr/bin/env python3
"""
Tests for the ManipulaPy backend dispatch package.

Covers the selection API (default backend, context-managed switching,
error handling), the NumPy backend round-trip and math surface, CuPy
lazy registration, and protocol completeness between base.py and the
concrete implementations.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import builtins
import importlib
import importlib.util

import numpy as np
import pytest

from ManipulaPy import backend as be
from ManipulaPy import utils
from ManipulaPy.backend.base import ArrayBackend
from ManipulaPy.backend.numpy_backend import NumpyBackend
from ManipulaPy.kinematics import SerialManipulator


# The full protocol surface, mirrored from the call-site audit. Kept here so
# the completeness test fails loudly if base.py and the impls drift apart.
CONSTRUCTION = ["array", "asarray", "zeros", "eye", "stack", "concatenate", "diag"]
LINALG = ["svd", "inv", "pinv", "solve", "norm", "trace"]
ELEMENTWISE = [
    "sin", "cos", "sqrt", "arccos", "arctan2", "abs", "clip",
    "maximum", "minimum", "where", "cross", "matmul",
]
REDUCTIONS = ["sum", "amax", "amin", "mean", "argmax", "all", "any", "isfinite"]
DEVICE = ["to_device", "to_numpy", "ascontiguous"]
DTYPES = ["float32", "float64"]
PREDICATE = ["is_backend_array"]
FULL_SURFACE = (
    CONSTRUCTION + LINALG + ELEMENTWISE + REDUCTIONS + DEVICE + DTYPES + PREDICATE
)


@pytest.fixture(autouse=True)
def _restore_backend():
    """Every test starts and ends on the default (NumPy) backend."""
    be.set_backend("numpy")
    yield
    be.set_backend("numpy")


def test_default_backend_is_numpy():
    """With no setup the active backend is NumPy."""
    assert isinstance(be.get_backend(), NumpyBackend)


def test_utils_default_backend_preserves_numpy_return_contract():
    """Core utils keep ndarray/shape/dtype contracts on the default backend."""
    so3mat = utils.VecToso3(np.array([0.0, 0.0, np.pi / 3]))
    transform = utils.transform_from_twist(
        np.array([0.0, 0.0, 1.0, 0.5, -0.25, 0.0]), 0.4
    )

    results = {
        "MatrixExp3": (utils.MatrixExp3(so3mat), (3, 3)),
        "MatrixLog3": (utils.MatrixLog3(utils.MatrixExp3(so3mat)), (3, 3)),
        "transform_from_twist": (transform, (4, 4)),
        "adjoint_transform": (utils.adjoint_transform(transform), (6, 6)),
        "logm": (utils.logm(transform), (6,)),
    }
    for name, (result, shape) in results.items():
        assert isinstance(result, np.ndarray), f"{name} returned {type(result)!r}"
        assert result.shape == shape
        assert result.dtype == np.float64

    _, theta = utils.rotation_logm(transform[:3, :3])
    assert type(theta) is float


def test_utils_numpy_backend_numeric_parity():
    """Dispatch through NumPy preserves representative SO(3)/SE(3) values."""
    twist = np.array([0.0, 0.0, 1.0, 0.5, -0.25, 0.0])
    theta = 0.4
    c, s = np.cos(theta), np.sin(theta)
    expected = np.array(
        [
            [c, -s, 0.0, 0.5 * s + 0.25 * (1.0 - c)],
            [s, c, 0.0, 0.5 * (1.0 - c) - 0.25 * s],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    with be.use_backend("numpy"):
        transform = utils.transform_from_twist(twist, theta)
        rotation = utils.MatrixExp3(utils.MatrixLog3(transform[:3, :3]))

    np.testing.assert_allclose(transform, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rotation, expected[:3, :3], rtol=1e-12, atol=1e-12)


def test_use_backend_restores_on_normal_exit():
    """The context manager restores the previous backend after the body."""
    original = be.get_backend()
    with be.use_backend("numpy"):
        assert be.get_backend() is not None
    assert be.get_backend() is original


def test_use_backend_restores_on_exception():
    """The previous backend is restored even when the body raises."""
    original = be.get_backend()
    with pytest.raises(ValueError):
        with be.use_backend("numpy"):
            raise ValueError("boom")
    assert be.get_backend() is original


def test_set_backend_unknown_lists_registered_names():
    """An unknown backend name raises, naming the registered backends."""
    with pytest.raises(ValueError) as exc:
        be.set_backend("nonexistent")
    assert "nonexistent" in str(exc.value)
    assert "numpy" in str(exc.value)


def test_duplicate_registration_rejected():
    """Registering an already-registered name raises rather than overwriting."""
    with pytest.raises(ValueError):
        be.register("numpy", NumpyBackend())


def test_cupy_selection():
    """CuPy: actionable error when absent (CI path), round-trip when present."""
    if importlib.util.find_spec("cupy") is None:
        with pytest.raises((ImportError, RuntimeError)) as exc:
            be.set_backend("cupy")
        assert "cupy" in str(exc.value).lower()
    else:  # pragma: no cover - device-dependent
        be.set_backend("cupy")
        backend = be.get_backend()
        result = backend.to_numpy(backend.array([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


@pytest.mark.parametrize("name", ["numpy"])
def test_round_trip_to_numpy(name):
    """to_numpy(array([...])) reproduces the source values for each backend."""
    backend = be.get_registered(name)
    result = backend.to_numpy(backend.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_math_surface_numpy():
    """One compact check per protocol group on the NumPy backend."""
    b = NumpyBackend()

    # linalg: svd shapes and norm value
    a = b.array([[3.0, 0.0], [0.0, 4.0]])
    u, s, vt = b.svd(a)
    assert u.shape == (2, 2) and s.shape == (2,) and vt.shape == (2, 2)
    assert np.isclose(b.norm(b.array([3.0, 4.0])), 5.0)
    assert np.isclose(b.trace(b.eye(3)), 3.0)

    # elementwise: arctan2 quadrant, clip bounds, matmul, cross
    assert np.isclose(b.arctan2(b.array(1.0), b.array(-1.0)), 3 * np.pi / 4)
    np.testing.assert_array_equal(b.clip(b.array([-5, 0, 5]), -1, 1), [-1, 0, 1])
    np.testing.assert_array_equal(
        b.matmul(b.eye(2), b.array([[1.0, 2.0], [3.0, 4.0]])),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.testing.assert_array_equal(
        b.cross(b.array([1.0, 0.0, 0.0]), b.array([0.0, 1.0, 0.0])),
        [0.0, 0.0, 1.0],
    )

    # reductions: sum, amax/amin, argmax, isfinite
    assert b.sum(b.array([1, 2, 3])) == 6
    assert b.amax(b.array([1, 5, 2])) == 5
    assert b.amin(b.array([1, 5, 2])) == 1
    assert b.argmax(b.array([1, 5, 2])) == 1
    assert bool(b.all(b.isfinite(b.array([1.0, 2.0]))))


def test_is_backend_array_numpy():
    """is_backend_array is True for np.ndarray, False for a list."""
    b = NumpyBackend()
    assert b.is_backend_array(np.array([1, 2, 3])) is True
    assert b.is_backend_array([1, 2, 3]) is False


@pytest.mark.parametrize("member", FULL_SURFACE)
def test_protocol_completeness(member):
    """Every declared surface member exists on the NumPy backend."""
    assert hasattr(NumpyBackend(), member), f"NumpyBackend missing {member!r}"


def test_dtype_handles_usable():
    """float32/float64 attributes are usable dtype handles."""
    b = NumpyBackend()
    x = b.array([1, 2, 3], dtype=b.float32)
    assert b.to_numpy(x).dtype == np.float32
    y = b.array([1, 2, 3], dtype=b.float64)
    assert b.to_numpy(y).dtype == np.float64


def test_import_safety_without_cupy(monkeypatch):
    """Importing the backend package must not require CuPy to be importable."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    module = importlib.reload(importlib.import_module("ManipulaPy.backend"))
    assert isinstance(module.get_backend(), NumpyBackend)


class _SpyNumpyBackend(NumpyBackend):
    """NumPy delegate that records primitives used by migrated hot paths."""

    def __init__(self):
        self.calls = []

    def eye(self, n, dtype=None):
        self.calls.append("eye")
        return super().eye(n, dtype=dtype)

    def zeros(self, shape, dtype=None):
        self.calls.append("zeros")
        return super().zeros(shape, dtype=dtype)

    def stack(self, arrays, axis=0):
        self.calls.append("stack")
        return super().stack(arrays, axis=axis)

    def matmul(self, a, b):
        self.calls.append("matmul")
        return super().matmul(a, b)

    def pinv(self, a):
        self.calls.append("pinv")
        return super().pinv(a)


def _two_joint_manipulator():
    """Return a deterministic planar arm with consistent space/body screws."""
    s_list = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    b_list = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 1.0],
            [0.0, 0.0],
        ]
    )
    home = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return SerialManipulator(
        M_list=home,
        omega_list=s_list[:3],
        S_list=s_list,
        B_list=b_list,
        joint_limits=[(None, None)] * 2,
    )


def _kinematics_results(robot):
    q = np.array([0.2, -0.3])
    dq = np.array([0.4, -0.2])
    results = {}
    for frame in ("space", "body"):
        velocity = robot.end_effector_velocity(q, dq, frame=frame)
        results[(frame, "fk")] = robot.forward_kinematics(q, frame=frame)
        results[(frame, "jacobian")] = robot.jacobian(q, frame=frame)
        results[(frame, "velocity")] = velocity
        results[(frame, "joint_velocity")] = robot.joint_velocity(
            q, velocity, frame=frame
        )
    return results


def test_kinematics_default_backend_numeric_and_return_contract():
    """Default NumPy keeps FK/Jacobian/velocity values and array contracts."""
    results = _kinematics_results(_two_joint_manipulator())

    expected_fk = np.array(
        [
            [0.9950041653, 0.0998334166, 0.0, 1.9750707431],
            [-0.0998334166, 0.9950041653, 0.0, 0.0988359141],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(results[("space", "fk")], expected_fk, atol=1e-9)
    np.testing.assert_allclose(results[("body", "fk")], expected_fk, atol=1e-9)

    # Preserve the corrected body-Jacobian seed: J_b[:, -1] is exactly B_n.
    expected_body_seed = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        results[("body", "jacobian")][:, -1], expected_body_seed
    )
    np.testing.assert_allclose(
        results[("space", "velocity")],
        [0.0, 0.0, 0.2, -0.0397338662, 0.1960133156, 0.0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        results[("body", "velocity")],
        [0.0, 0.0, 0.2, -0.1182080827, 0.5821345957, 0.0],
        atol=1e-9,
    )
    for frame in ("space", "body"):
        np.testing.assert_allclose(
            results[(frame, "joint_velocity")], [0.4, -0.2], atol=1e-12
        )
        for operation, shape in (
            ("fk", (4, 4)),
            ("jacobian", (6, 2)),
            ("velocity", (6,)),
            ("joint_velocity", (2,)),
        ):
            result = results[(frame, operation)]
            assert isinstance(result, np.ndarray)
            assert result.shape == shape
            assert result.dtype == np.float64


def test_kinematics_hot_paths_dispatch_through_active_backend(monkeypatch):
    """FK, Jacobians, and velocity helpers use the selected backend."""
    robot = _two_joint_manipulator()
    expected = _kinematics_results(robot)
    spy = _SpyNumpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    actual = _kinematics_results(robot)

    for key, expected_value in expected.items():
        np.testing.assert_allclose(actual[key], expected_value, rtol=1e-12, atol=1e-12)
    assert "eye" in spy.calls
    assert "matmul" in spy.calls
    assert "stack" in spy.calls
    assert "pinv" in spy.calls


def test_kinematics_cupy_native_parity():
    """CuPy keeps migrated kinematics native while matching NumPy values."""
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")

    import cupy as cp

    if not isinstance(getattr(cp, "ndarray", None), type):
        pytest.skip("CuPy test double does not provide native array types")
    try:
        cp.asarray([0.0])
    except Exception as exc:  # pragma: no cover - device/runtime dependent
        pytest.skip(f"CuPy runtime is unavailable: {exc}")

    robot = _two_joint_manipulator()
    expected = _kinematics_results(robot)
    with be.use_backend("cupy"):
        actual = _kinematics_results(robot)

    for key, expected_value in expected.items():
        assert isinstance(actual[key], cp.ndarray)
        np.testing.assert_allclose(
            cp.asnumpy(actual[key]), expected_value, rtol=1e-10, atol=1e-10
        )


def test_kinematics_integer_list_input_keeps_floating_point_contract():
    """Integer Python lists must not select integer arithmetic for FK/Jacobians."""
    robot = _two_joint_manipulator()

    transform = robot.forward_kinematics([0, 0], frame="space")
    jacobian = robot.jacobian([0, 0], frame="body")

    assert transform.dtype == np.float64
    assert jacobian.dtype == np.float64
    np.testing.assert_array_equal(transform, robot.M_list)
    np.testing.assert_array_equal(jacobian[:, -1], robot.B_list[:, -1])


def test_kinematics_stacked_home_transforms_use_last_pose():
    """FK preserves the established final-pose selection for stacked M_list."""
    robot = _two_joint_manipulator()
    first_home = np.eye(4)
    final_home = robot.M_list.copy()
    robot.M_list = np.stack((first_home, final_home))
    robot._m_list_is_array_of_poses = True

    for frame in ("space", "body"):
        result = robot.forward_kinematics([0.0, 0.0], frame=frame)
        np.testing.assert_array_equal(result, final_home)


def test_space_jacobian_with_no_joints_preserves_empty_shape():
    """A zero-joint space Jacobian remains a valid floating `(6, 0)` array."""
    empty = np.empty((6, 0), dtype=np.float64)
    robot = SerialManipulator(
        M_list=np.eye(4),
        omega_list=empty[:3],
        S_list=empty,
        B_list=empty,
        joint_limits=[],
    )

    result = robot.jacobian([], frame="space")

    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 0)
    assert result.dtype == np.float64
