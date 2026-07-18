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
from unittest.mock import patch

import numpy as np
import pytest

from ManipulaPy import backend as be
from ManipulaPy import utils
from ManipulaPy.backend.base import ArrayBackend
from ManipulaPy.backend.numpy_backend import NumpyBackend
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.singularity import Singularity


# The full protocol surface, mirrored from the call-site audit. Kept here so
# the completeness test fails loudly if base.py and the impls drift apart.
CONSTRUCTION = ["array", "asarray", "zeros", "eye", "stack", "concatenate", "diag"]
LINALG = ["svd", "svdvals", "inv", "pinv", "solve", "norm", "trace"]
ELEMENTWISE = [
    "sin", "cos", "sqrt", "arccos", "arctan2", "abs", "clip",
    "maximum", "minimum", "where", "cross", "matmul",
]
REDUCTIONS = ["sum", "amax", "amin", "mean", "argmax", "all", "any", "isfinite"]
DEVICE = ["to_device", "to_numpy", "ascontiguous"]
DTYPES = ["float32", "float64"]
PREDICATE = ["is_backend_array", "is_concrete", "cache_token"]
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


def test_top_level_lazy_access_exposes_backend_api():
    """The package-level lazy loader resolves the backend package, the
    selection API, and the planning package.

    __getattr__ is exercised directly because a prior submodule import binds
    the attribute on the package, which would let stale mappings pass.
    """
    import ManipulaPy

    assert ManipulaPy.__getattr__("backend") is be
    assert ManipulaPy.__getattr__("set_backend") is be.set_backend
    assert ManipulaPy.__getattr__("use_backend") is be.use_backend
    assert ManipulaPy.__getattr__("get_backend") is be.get_backend

    import ManipulaPy.planning as planning_module

    assert ManipulaPy.__getattr__("planning") is planning_module
    for name in ("backend", "planning", "set_backend", "use_backend", "get_backend"):
        assert name in dir(ManipulaPy)


def test_dtype_handles_usable():
    """float32/float64 attributes are usable dtype handles."""
    b = NumpyBackend()
    x = b.array([1, 2, 3], dtype=b.float32)
    assert b.to_numpy(x).dtype == np.float32
    y = b.array([1, 2, 3], dtype=b.float64)
    assert b.to_numpy(y).dtype == np.float64


def test_cache_tokens_are_instance_scoped():
    """Separate backend instances cannot share materialized cache entries."""
    first = NumpyBackend()
    second = NumpyBackend()
    assert first.cache_token() != second.cache_token()


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


# ---------------------------------------------------------------------------
# Dynamics dispatch
# ---------------------------------------------------------------------------


class _DynamicsSpyBackend(NumpyBackend):
    """Concrete NumPy delegate that records primitives dynamics hot paths use."""

    is_concrete = True

    def __init__(self):
        self.calls = []

    def zeros(self, shape, dtype=None):
        self.calls.append("zeros")
        return super().zeros(shape, dtype=dtype)

    def eye(self, n, dtype=None):
        self.calls.append("eye")
        return super().eye(n, dtype=dtype)

    def matmul(self, a, b):
        self.calls.append("matmul")
        return super().matmul(a, b)

    def inv(self, a):
        self.calls.append("inv")
        return super().inv(a)

    def solve(self, a, b):
        self.calls.append("solve")
        return super().solve(a, b)

    def concatenate(self, arrays, axis=0):
        self.calls.append("concatenate")
        return super().concatenate(arrays, axis=axis)


class _NonConcreteNumpyBackend(NumpyBackend):
    """NumPy numerics with the concrete flag flipped off (traced-style stand-in).

    Computes real arrays so ManipulatorDynamics runs, but reports
    ``is_concrete = False`` the way a future traced Torch/JAX backend would.
    """

    is_concrete = False


def _planar_2r_dynamics():
    """Analytical 2R planar arm with per-link CoM data (mirrors the v1.3.2 fixture)."""
    from ManipulaPy.utils import extract_screw_list

    L1 = L2 = 1.0
    omega_list = np.array([[0, 0, 1], [0, 0, 1]]).T
    r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T
    s_list = extract_screw_list(omega_list, r_list)

    home = np.eye(4)
    home[0, 3] = L1 + L2
    m_link1 = np.eye(4)
    m_link1[0, 3] = L1
    m_link2 = np.eye(4)
    m_link2[0, 3] = L1 + L2
    glist = np.array([np.diag([0.0, 0.0, 0.0, m, m, m]) for m in (1.0, 1.0)])

    return ManipulatorDynamics(
        M_list=home,
        omega_list=omega_list,
        r_list=r_list,
        b_list=r_list,
        S_list=s_list,
        B_list=None,
        Glist=glist,
        Mlist_per_link=[m_link1, m_link2],
    )


def _dynamics_results(dyn):
    """Evaluate every migrated dynamics hot path with fixed deterministic inputs."""
    theta = np.array([0.1, 0.2])
    dtheta = np.array([0.3, -0.2])
    ddtheta = np.array([0.5, 0.4])
    tau = np.array([1.0, -0.5])
    g = np.array([0.0, 0.0, -9.81])
    ftip = np.zeros(6)
    return {
        "mass_matrix": dyn.mass_matrix(theta),
        "gravity_forces": dyn.gravity_forces(theta, g),
        "velocity_quadratic_forces": dyn.velocity_quadratic_forces(theta, dtheta),
        "inverse_dynamics": dyn.inverse_dynamics(theta, dtheta, ddtheta, g, ftip),
        "forward_dynamics": dyn.forward_dynamics(theta, dtheta, tau, g, ftip),
    }


def test_dynamics_default_backend_numeric_and_return_contract():
    """Default NumPy keeps dynamics values and the float64 ndarray contract."""
    results = _dynamics_results(_planar_2r_dynamics())

    # Analytical 2R mass matrix at theta=(0.1, 0.2): m1=m2=L1=L2=1.
    c2 = np.cos(0.2)
    expected_mass = np.array(
        [
            [1.0 + (1.0 + 1.0 + 2.0 * c2), 1.0 + 1.0 * c2],
            [1.0 + 1.0 * c2, 1.0],
        ]
    )
    np.testing.assert_allclose(results["mass_matrix"], expected_mass, atol=1e-6)

    shapes = {
        "mass_matrix": (2, 2),
        "gravity_forces": (2,),
        "velocity_quadratic_forces": (2,),
        "inverse_dynamics": (2,),
        "forward_dynamics": (2,),
    }
    for name, shape in shapes.items():
        value = results[name]
        assert isinstance(value, np.ndarray), f"{name} returned {type(value)!r}"
        assert value.shape == shape
        assert value.dtype == np.float64


def test_dynamics_hot_paths_dispatch_through_active_backend(monkeypatch):
    """Mass matrix, gravity, Coriolis, and inverse/forward dynamics route through
    the active backend."""
    expected = _dynamics_results(_planar_2r_dynamics())

    spy = _DynamicsSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    actual = _dynamics_results(_planar_2r_dynamics())

    for key, expected_value in expected.items():
        np.testing.assert_allclose(
            actual[key], expected_value, rtol=1e-10, atol=1e-10
        )
    # Per-link mass/gravity build CoM Jacobians via inv + concatenate; forward
    # dynamics solves M x = rhs; every path multiplies through matmul.
    assert "matmul" in spy.calls
    assert "inv" in spy.calls
    assert "concatenate" in spy.calls
    assert "solve" in spy.calls


def test_mass_matrix_cache_used_on_concrete_backend():
    """The value-keyed cache is populated and reused under the concrete NumPy backend."""
    dyn = _planar_2r_dynamics()
    theta = np.array([0.1, 0.2])

    first = dyn.mass_matrix(theta)
    assert len(dyn._mass_matrix_cache) == 1
    second = dyn.mass_matrix(theta)
    # A concrete cache hit returns the identical stored object.
    assert second is first


def test_mass_matrix_cache_bypassed_on_nonconcrete_backend(monkeypatch):
    """A non-concrete (traced-style) backend must neither read nor write the
    tuple-keyed mass-matrix cache."""
    theta = np.array([0.1, 0.2])
    poison = np.full((2, 2), 999.0)

    # Read-bypass: a poisoned entry must be ignored, not returned.
    dyn_read = _planar_2r_dynamics()
    dyn_read._mass_matrix_cache[tuple(theta)] = poison
    monkeypatch.setattr(be, "_active", _NonConcreteNumpyBackend())
    out = dyn_read.mass_matrix(theta)
    assert out is not poison
    assert not np.array_equal(out, poison)
    # It must equal the genuinely recomputed matrix.
    c2 = np.cos(0.2)
    expected_mass = np.array(
        [[1.0 + (2.0 + 2.0 * c2), 1.0 + c2], [1.0 + c2, 1.0]]
    )
    np.testing.assert_allclose(out, expected_mass, atol=1e-6)

    # Write-bypass: a fresh cache must stay empty after a non-concrete call.
    dyn_write = _planar_2r_dynamics()
    dyn_write.mass_matrix(theta)
    assert len(dyn_write._mass_matrix_cache) == 0


def test_dynamics_caches_are_namespaced_by_backend(monkeypatch):
    """Switching concrete backend implementations must recompute cached values."""
    theta = np.array([0.1, 0.2])
    dyn = _planar_2r_dynamics()
    first_mass = dyn.mass_matrix(theta)
    first_derivatives = dyn._mass_matrix_derivatives(theta)

    spy = _DynamicsSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    second_mass = dyn.mass_matrix(theta)
    second_derivatives = dyn._mass_matrix_derivatives(theta)

    np.testing.assert_allclose(second_mass, first_mass, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        second_derivatives, first_derivatives, rtol=1e-8, atol=1e-8
    )
    assert second_mass is not first_mass
    assert second_derivatives is not first_derivatives
    assert "matmul" in spy.calls
    assert "eye" in spy.calls


def test_mass_matrix_derivative_cache_bypassed_on_nonconcrete_backend(monkeypatch):
    """Traced-style backends must neither read nor populate derivative caches."""
    theta = np.array([0.1, 0.2])
    epsilon = 1e-6
    expected = _planar_2r_dynamics()._mass_matrix_derivatives(theta, epsilon)
    poison = np.full((2, 2, 2), 999.0)

    nonconcrete = _NonConcreteNumpyBackend()
    poison_key = (nonconcrete.cache_token(), tuple(theta.tolist()), epsilon)
    dyn_read = _planar_2r_dynamics()
    dyn_read._mass_matrix_derivative_cache[poison_key] = poison
    monkeypatch.setattr(be, "_active", nonconcrete)
    out = dyn_read._mass_matrix_derivatives(theta, epsilon)

    assert out is not poison
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-8)
    assert len(dyn_read._mass_matrix_derivative_cache) == 1
    assert next(iter(dyn_read._mass_matrix_derivative_cache.values())) is poison

    dyn_write = _planar_2r_dynamics()
    dyn_write._mass_matrix_derivatives(theta, epsilon)
    assert len(dyn_write._mass_matrix_derivative_cache) == 0
    assert len(dyn_write._mass_matrix_cache) == 0


def test_dynamics_native_cupy_parity_and_cache_safety():
    """Native CuPy arrays remain device-native through dynamics and cache reuse."""
    cp = pytest.importorskip("cupy")
    if not isinstance(getattr(cp, "ndarray", None), type):
        pytest.skip("CuPy test double does not provide native array types")
    try:
        theta = cp.asarray([0.1, 0.2], dtype=cp.float64)
    except Exception as exc:
        pytest.skip(f"Native CuPy runtime unavailable: {exc}")

    dyn = _planar_2r_dynamics()
    expected = _dynamics_results(dyn)
    expected_derivatives = dyn._mass_matrix_derivatives(np.array([0.1, 0.2]))
    with be.use_backend("cupy"):
        dtheta = cp.asarray([0.3, -0.2], dtype=cp.float64)
        ddtheta = cp.asarray([0.5, 0.4], dtype=cp.float64)
        tau = cp.asarray([1.0, -0.5], dtype=cp.float64)
        g = cp.asarray([0.0, 0.0, -9.81], dtype=cp.float64)
        ftip = cp.zeros(6, dtype=cp.float64)
        actual = {
            "mass_matrix": dyn.mass_matrix(theta),
            "gravity_forces": dyn.gravity_forces(theta, g),
            "velocity_quadratic_forces": dyn.velocity_quadratic_forces(
                theta, dtheta
            ),
            "inverse_dynamics": dyn.inverse_dynamics(
                theta, dtheta, ddtheta, g, ftip
            ),
            "forward_dynamics": dyn.forward_dynamics(
                theta, dtheta, tau, g, ftip
            ),
        }
        second = dyn.mass_matrix(theta)
        derivatives = dyn._mass_matrix_derivatives(theta)

    for name, value in actual.items():
        assert isinstance(value, cp.ndarray), f"{name} returned {type(value)!r}"
        np.testing.assert_allclose(
            cp.asnumpy(value), expected[name], rtol=1e-10, atol=1e-10
        )
    assert second is actual["mass_matrix"]
    assert isinstance(derivatives, cp.ndarray)
    np.testing.assert_allclose(
        cp.asnumpy(derivatives), expected_derivatives, rtol=1e-8, atol=1e-8
    )


# ---------------------------------------------------------------------------
# Singularity dispatch
# ---------------------------------------------------------------------------


class _SingularitySpyBackend(NumpyBackend):
    """Concrete NumPy delegate that records primitives singularity paths use."""

    is_concrete = True

    def __init__(self):
        self.calls = []

    def svd(self, a, full_matrices=False):
        self.calls.append("svd")
        return super().svd(a, full_matrices=full_matrices)

    def svdvals(self, a):
        self.calls.append("svdvals")
        return super().svdvals(a)

    def amax(self, x, axis=None):
        self.calls.append("amax")
        return super().amax(x, axis=axis)

    def amin(self, x, axis=None):
        self.calls.append("amin")
        return super().amin(x, axis=axis)

    def maximum(self, x1, x2):
        self.calls.append("maximum")
        return super().maximum(x1, x2)

    def sqrt(self, x):
        self.calls.append("sqrt")
        return super().sqrt(x)


def _singularity_results(sing):
    """Evaluate the scalar SVD hot paths with fixed deterministic joint angles."""
    theta = np.array([0.2, -0.3])
    return {
        "singularity": sing.singularity_analysis(theta),
        "condition_number": sing.condition_number(theta),
        "near_singularity": sing.near_singularity_detection(theta),
    }


def test_singularity_default_backend_numeric_and_return_contract():
    """Default NumPy keeps singularity values and the exact return-type contract."""
    robot = _two_joint_manipulator()
    sing = Singularity(robot)
    results = _singularity_results(sing)

    # Return-type contract mirrors tests/data/api_contract_golden.json.
    assert type(results["singularity"]) is bool
    assert isinstance(results["condition_number"], np.float64)
    assert isinstance(results["near_singularity"], np.bool_)

    # Numeric parity against a direct NumPy computation of the same Jacobian.
    J = robot.jacobian(np.array([0.2, -0.3]), frame="space")
    s = np.linalg.svd(J, compute_uv=False)
    assert results["singularity"] == bool(s[-1] < 1e-4)
    np.testing.assert_allclose(
        results["condition_number"], np.linalg.cond(J), rtol=1e-12
    )
    assert bool(results["near_singularity"]) == bool(np.linalg.cond(J) > 1e-2)


def test_condition_number_singular_jacobian_is_infinite():
    """A rank-deficient (zero) Jacobian yields an infinite condition number."""

    class _ZeroJacobianRobot:
        def jacobian(self, thetalist, frame="space"):
            return np.zeros((6, 6))

    cond = Singularity(_ZeroJacobianRobot()).condition_number(np.zeros(6))
    assert isinstance(cond, np.float64)
    assert np.isinf(cond)


def test_singularity_hot_paths_dispatch_through_active_backend(monkeypatch):
    """singularity_analysis, condition_number, and near-singularity detection
    route their SVD math through the active backend."""
    robot = _two_joint_manipulator()
    sing = Singularity(robot)
    expected = _singularity_results(sing)

    spy = _SingularitySpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    actual = _singularity_results(sing)

    for key in expected:
        np.testing.assert_allclose(
            np.asarray(actual[key], dtype=float),
            np.asarray(expected[key], dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
    # The smallest-singular-value test and condition number both go through the
    # values-only SVD; the condition number derives from amax/amin over the spectrum.
    assert "svdvals" in spy.calls
    assert "amax" in spy.calls
    assert "amin" in spy.calls


def test_manipulability_ellipsoid_axis_math_dispatches_through_backend(monkeypatch):
    """The ellipsoid axis math (SVD + guarded radii) routes through the backend
    while plotting stays host-bound."""
    robot = _two_joint_manipulator()
    sing = Singularity(robot)

    spy = _SingularitySpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    with patch("matplotlib.pyplot.show"):
        sing.manipulability_ellipsoid(np.array([0.2, -0.3]))

    # SVD for both linear/angular parts and the guarded 1/sqrt(max(S, 1e-10)).
    assert "svd" in spy.calls
    assert "maximum" in spy.calls
    assert "sqrt" in spy.calls


# ---------------------------------------------------------------------------
# IK dispatch (TRAC-IK DLS Newton branch + IK helpers)
# ---------------------------------------------------------------------------


class _IKSpyBackend(NumpyBackend):
    """Concrete NumPy delegate that records primitives the IK paths use."""

    is_concrete = True

    def __init__(self):
        self.calls = []

    def svd(self, a, full_matrices=False):
        self.calls.append("svd")
        return super().svd(a, full_matrices=full_matrices)

    def matmul(self, a, b):
        self.calls.append("matmul")
        return super().matmul(a, b)

    def solve(self, a, b):
        self.calls.append("solve")
        return super().solve(a, b)

    def norm(self, x, ord=None, axis=None):
        self.calls.append("norm")
        return super().norm(x, ord=ord, axis=axis)

    def eye(self, n, dtype=None):
        self.calls.append("eye")
        return super().eye(n, dtype=dtype)

    def inv(self, a):
        self.calls.append("inv")
        return super().inv(a)

    def pinv(self, a):
        self.calls.append("pinv")
        return super().pinv(a)

    def arctan2(self, y, x):
        self.calls.append("arctan2")
        return super().arctan2(y, x)

    def arccos(self, x):
        self.calls.append("arccos")
        return super().arccos(x)

    def concatenate(self, arrays, axis=0):
        self.calls.append("concatenate")
        return super().concatenate(arrays, axis=axis)


def _trac_ik_fixture():
    """Return a 6-DOF TRAC-IK solver, a reachable target, and the seed angles."""
    from ManipulaPy.trac_ik import TracIKSolver

    Slist = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, -0.089, 0, 0],
            [0, -1, 0, -0.089, 0, 0.425],
            [0, -1, 0, -0.089, 0, 0.817],
            [1, 0, 0, 0, 0.109, 0],
            [0, -1, 0, -0.089, 0, 0.817],
        ]
    ).T
    M = np.array([[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]])
    robot = SerialManipulator(
        M_list=M,
        omega_list=Slist[:3, :],
        S_list=Slist,
        B_list=np.copy(Slist),
        joint_limits=[(-np.pi, np.pi)] * 6,
    )
    solver = TracIKSolver(
        fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
        jacobian_func=lambda th: robot.jacobian(th, frame="space"),
        joint_limits=robot.joint_limits,
        n_joints=6,
    )
    theta_known = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    T_desired = robot.forward_kinematics(theta_known, frame="space")
    return solver, T_desired, theta_known


def test_trac_ik_default_backend_solution_and_return_contract():
    """Default NumPy keeps the TRAC-IK solve solution and return-type contract."""
    import threading

    solver, T_desired, theta_known = _trac_ik_fixture()
    theta, success, error = solver._dls_solver(
        T_desired,
        theta_known + 0.05,
        eomg=1e-3,
        ev=1e-3,
        timeout=5.0,
        stop_event=threading.Event(),
    )
    assert isinstance(theta, np.ndarray)
    assert type(success) is bool
    assert success
    assert error < 1e-2


def test_trac_ik_dls_newton_branch_dispatches_through_active_backend(monkeypatch):
    """The DLS Newton inner loop routes its SVD/matmul/norm math through the
    active backend rather than calling numpy directly."""
    import threading

    solver, T_desired, theta_known = _trac_ik_fixture()

    spy = _IKSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    solver._dls_solver(
        T_desired,
        theta_known + 0.3,
        eomg=1e-3,
        ev=1e-3,
        timeout=0.1,
        stop_event=threading.Event(),
    )

    # The differentiable branch: SVD-robust damped least squares + step norm.
    assert "svd" in spy.calls
    assert "matmul" in spy.calls
    assert "norm" in spy.calls


def test_workspace_heuristic_guess_dispatches_through_active_backend(monkeypatch):
    """The geometric seed heuristic routes its trig math through the backend."""
    from ManipulaPy.kinematics import ik_helpers

    # A tilted target so the non-gimbal wrist branch (arccos) is exercised.
    T_desired = np.array(
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, -1.0, 0.2],
            [0.0, 1.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    limits = [(-np.pi, np.pi)] * 6

    expected = ik_helpers.workspace_heuristic_guess(T_desired, 6, limits)

    spy = _IKSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    actual = ik_helpers.workspace_heuristic_guess(T_desired, 6, limits)

    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-12)
    assert "arctan2" in spy.calls
    assert "arccos" in spy.calls


def test_extrapolate_from_current_dispatches_through_active_backend(monkeypatch):
    """Extrapolation seed math routes pseudo-inverse and transform ops through
    the backend it is driven by."""
    from ManipulaPy.kinematics import ik_helpers

    solver, T_desired, theta_known = _trac_ik_fixture()
    jac = solver.jacobian_func
    T_current = solver.fk_func(theta_known)

    expected = ik_helpers.extrapolate_from_current(
        theta_known, T_current, T_desired, jac, solver.joint_limits, alpha=0.5
    )

    spy = _IKSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    actual = ik_helpers.extrapolate_from_current(
        theta_known, T_current, T_desired, jac, solver.joint_limits, alpha=0.5
    )

    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-10)
    assert "pinv" in spy.calls
    assert "inv" in spy.calls


def test_ik_initial_guess_cache_returns_host_array():
    """IKInitialGuessCache stays host-domain: nearest-neighbor lookup returns a
    plain NumPy array so it never hashes or stores traced tensors."""
    from ManipulaPy.kinematics.ik_helpers import IKInitialGuessCache

    cache = IKInitialGuessCache(max_size=10)
    T = np.eye(4)
    T[:3, 3] = [0.3, 0.2, 0.4]
    cache.add(T, np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6]), residual=1e-4)

    guess = cache.get_nearest(T, k=1)
    assert isinstance(guess, np.ndarray)


def test_ik_initial_guess_cache_enforces_host_boundary():
    """The cache converts inputs to host NumPy at its boundary: backend-native
    (or plain array-like) entries must be stored and returned as host arrays so
    the lookup math never runs on a device array."""
    from ManipulaPy.kinematics.ik_helpers import IKInitialGuessCache

    cache = IKInitialGuessCache(max_size=10)
    # Non-ndarray array-likes (nested lists) stand in for device arrays here.
    T = [[1, 0, 0, 0.3], [0, 0, -1, 0.2], [0, 1, 0, 0.4], [0, 0, 0, 1]]
    theta = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6]
    cache.add(T, theta, residual=1e-4)

    stored_T, stored_theta, _ = cache.cache[0]
    assert isinstance(stored_T, np.ndarray)
    assert isinstance(stored_theta, np.ndarray)

    guess = cache.get_nearest(T, k=1)
    assert isinstance(guess, np.ndarray)


def test_trac_ik_solve_preserves_float32_seed_dtype():
    """A float32 seed must round-trip as float32 through solve(): the DLS branch
    preserves seed dtype (only bool/int/non-backend seeds are promoted)."""
    solver, T_desired, theta_known = _trac_ik_fixture()
    seed = theta_known.astype(np.float32)

    theta, success, _ = solver.solve(T_desired, seed, timeout=1.0)

    assert success
    assert theta.dtype == np.float32


def test_workspace_heuristic_shim_routing_stays_live(monkeypatch):
    """The solver's inline ``from .. import ik_helpers`` must resolve through the
    top-level shim, so patching the shim symbol is seen by the solver."""
    import ManipulaPy.ik_helpers as shim

    solver, T_desired, _ = _trac_ik_fixture()
    sentinel = np.full(6, 0.123)
    monkeypatch.setattr(
        shim, "workspace_heuristic_guess", lambda *a, **k: sentinel.copy()
    )

    out = solver._workspace_heuristic(T_desired)
    np.testing.assert_array_equal(out, sentinel)


class _SvdFailingIKSpyBackend(_IKSpyBackend):
    """IK spy whose SVD always raises LinAlgError to force the DLS fallback."""

    def svd(self, a, full_matrices=False):
        self.calls.append("svd")
        raise np.linalg.LinAlgError("forced SVD failure")


def test_dls_svd_failure_routes_fallback_through_backend(monkeypatch):
    """When SVD raises, the normal-equations fallback must route through the
    backend's solve/eye and still return a finite result."""
    import threading

    solver, T_desired, theta_known = _trac_ik_fixture()
    spy = _SvdFailingIKSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    theta, success, error = solver._dls_solver(
        T_desired,
        theta_known + 0.3,
        eomg=1e-3,
        ev=1e-3,
        timeout=0.1,
        stop_event=threading.Event(),
    )

    assert "svd" in spy.calls
    assert "solve" in spy.calls
    assert "eye" in spy.calls
    assert np.all(np.isfinite(np.asarray(theta, dtype=float)))


class _FixedJacobianRobot:
    """Serial-manipulator stand-in returning a fixed Jacobian matrix."""

    def __init__(self, J):
        self._J = J

    def jacobian(self, thetalist, frame="space"):
        return self._J


def test_condition_number_zero_jacobian_ignores_caller_errstate():
    """np.linalg.cond suppresses the internal 0/0 for a rank-deficient matrix,
    so condition_number must return inf even when the caller escalates
    floating-point errors to exceptions."""
    sing = Singularity(_FixedJacobianRobot(np.zeros((6, 3))))
    with np.errstate(invalid="raise", divide="raise"):
        cond = sing.condition_number(np.zeros(3))
    assert np.isinf(cond)


def test_condition_number_infinite_jacobian_matches_numpy_cond():
    """An infinite entry gives NaN singular values, which np.linalg.cond
    reports as an infinite condition number (values-only SVD — the full-USV
    LAPACK path can spin forever on non-finite input)."""
    J = np.eye(6)[:, :3].copy()
    J[0, 0] = np.inf
    cond = Singularity(_FixedJacobianRobot(J)).condition_number(np.zeros(3))
    assert np.isinf(cond)


def test_condition_number_nan_jacobian_raises_like_numpy_cond():
    """np.linalg.cond raises LinAlgError for NaN input (SVD does not
    converge); the migrated path must preserve that instead of masking it."""
    J = np.eye(6)[:, :3].copy()
    J[0, 0] = np.nan
    sing = Singularity(_FixedJacobianRobot(J))
    with pytest.raises(np.linalg.LinAlgError):
        sing.condition_number(np.zeros(3))


def test_condition_number_dtype_follows_input():
    """np.linalg.cond lets the result dtype follow the input dtype; a float32
    Jacobian must keep returning a float32 scalar."""
    J = np.eye(6)[:, :3].astype(np.float32)
    cond = Singularity(_FixedJacobianRobot(J)).condition_number(np.zeros(3))
    assert cond.dtype == np.float32
    assert cond == np.float32(1.0)


def test_each_scalar_method_dispatches_svd_in_isolation(monkeypatch):
    """Each scalar hot path must route its own SVD through the backend — an
    aggregated assertion would let one method silently fall back to np.linalg."""
    robot = _two_joint_manipulator()
    sing = Singularity(robot)
    theta = np.array([0.2, -0.3])

    for method in (
        lambda: sing.singularity_analysis(theta),
        lambda: sing.condition_number(theta),
        lambda: sing.near_singularity_detection(theta),
    ):
        spy = _SingularitySpyBackend()
        monkeypatch.setattr(be, "_active", spy)
        method()
        # Values-only SVD is required: the full-USV LAPACK path can hang on
        # non-finite input, so a regression back to it must fail here.
        assert "svdvals" in spy.calls
        assert "svd" not in spy.calls


# ---------------------------------------------------------------------------
# Trajectory-generation dispatch
# ---------------------------------------------------------------------------

from ManipulaPy.planning.trajectory_planning import OptimizedTrajectoryPlanning


class _TrajectorySpyBackend(NumpyBackend):
    """NumPy delegate that records primitives used by migrated generation paths.

    ``stack`` also records the shape of each result so a test can single out the
    Cartesian orientation assembly ((N, 3, 3)), which no other call on the path
    produces.
    """

    def __init__(self):
        self.calls = []
        self.stack_shapes = []

    def to_numpy(self, x):
        self.calls.append("to_numpy")
        return super().to_numpy(x)

    def asarray(self, obj, dtype=None):
        self.calls.append("asarray")
        return super().asarray(obj, dtype=dtype)

    def clip(self, x, a_min, a_max):
        self.calls.append("clip")
        return super().clip(x, a_min, a_max)

    def matmul(self, a, b):
        self.calls.append("matmul")
        return super().matmul(a, b)

    def stack(self, arrays, axis=0):
        self.calls.append("stack")
        out = super().stack(arrays, axis=axis)
        self.stack_shapes.append(out.shape)
        return out


def _cpu_trajectory_planner():
    """Force-CPU planner with wide joint limits and no collision checker."""
    return OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0), (-5.0, 5.0)],
        use_cuda=False,
    )


def test_joint_trajectory_cpu_default_backend_parity():
    """Default NumPy keeps quintic joint positions/velocities and float32 dtype."""
    planner = _cpu_trajectory_planner()
    traj = planner.joint_trajectory([0.0, 0.0], [1.0, -0.5], 1.0, 3, 5)

    expected_pos = np.array(
        [[0.0, 0.0], [0.5, -0.25], [1.0, -0.5]], dtype=np.float32
    )
    expected_vel = np.array(
        [[0.0, 0.0], [1.875, -0.9375], [0.0, 0.0]], dtype=np.float32
    )
    expected_acc = np.zeros((3, 2), dtype=np.float32)

    np.testing.assert_array_equal(traj["positions"], expected_pos)
    np.testing.assert_array_equal(traj["velocities"], expected_vel)
    np.testing.assert_array_equal(traj["accelerations"], expected_acc)
    for key in ("positions", "velocities", "accelerations"):
        assert traj[key].dtype == np.float32


def test_batch_joint_trajectory_cpu_default_backend_parity():
    """Default NumPy keeps batch quintic positions, shape and float32 dtype."""
    planner = _cpu_trajectory_planner()
    thetastart = np.array([[0.0, 0.0], [0.2, 0.3]], dtype=np.float32)
    thetaend = np.array([[1.0, -0.5], [0.4, 0.1]], dtype=np.float32)
    traj = planner.batch_joint_trajectory(thetastart, thetaend, 1.0, 3, 5)

    expected_pos = np.array(
        [
            [[0.0, 0.0], [0.5, -0.25], [1.0, -0.5]],
            [[0.2, 0.3], [0.3, 0.2], [0.4, 0.1]],
        ],
        dtype=np.float32,
    )
    assert traj["positions"].shape == (2, 3, 2)
    np.testing.assert_allclose(traj["positions"], expected_pos, atol=1e-6)
    for key in ("positions", "velocities", "accelerations"):
        assert traj[key].dtype == np.float32


def test_cartesian_trajectory_default_backend_parity():
    """Default NumPy keeps Cartesian positions/velocities/orientations and dtype."""
    planner = _cpu_trajectory_planner()
    Xstart = np.eye(4)
    Xend = np.eye(4)
    Xend[:3, 3] = [1.0, 2.0, 3.0]
    traj = planner.cartesian_trajectory(Xstart, Xend, 1.0, 5, 5)

    expected_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.103515625, 0.20703125, 0.310546875],
            [0.5, 1.0, 1.5],
            [0.896484375, 1.79296875, 2.689453125],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    expected_vel = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0546875, 2.109375, 3.1640625],
            [1.875, 3.75, 5.625],
            [1.0546875, 2.109375, 3.1640625],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(traj["positions"], expected_pos, atol=1e-6)
    np.testing.assert_allclose(traj["velocities"], expected_vel, atol=1e-6)
    assert traj["orientations"].shape == (5, 3, 3)
    for i in range(5):
        np.testing.assert_array_equal(traj["orientations"][i], np.eye(3, dtype=np.float32))
    for key in ("positions", "velocities", "accelerations", "orientations"):
        assert traj[key].dtype == np.float32


def test_joint_trajectory_cpu_dispatches_through_active_backend(monkeypatch):
    """The CPU joint path routes clipping and the njit boundary through the backend."""
    planner = _cpu_trajectory_planner()
    spy = _TrajectorySpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.joint_trajectory([0.0, 0.0], [1.0, -0.5], 1.0, 3, 5)

    assert "to_numpy" in spy.calls
    assert "clip" in spy.calls


def test_batch_joint_trajectory_cpu_dispatches_through_active_backend(monkeypatch):
    """The CPU batch path stacks rows and clips limits through the backend."""
    planner = _cpu_trajectory_planner()
    thetastart = np.array([[0.0, 0.0], [0.2, 0.3]], dtype=np.float32)
    thetaend = np.array([[1.0, -0.5], [0.4, 0.1]], dtype=np.float32)
    spy = _TrajectorySpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.batch_joint_trajectory(thetastart, thetaend, 1.0, 3, 5)

    assert "stack" in spy.calls
    assert "clip" in spy.calls


def test_cartesian_trajectory_dispatches_through_active_backend(monkeypatch):
    """Cartesian orientation assembly stacks (N, 3, 3) through the active backend."""
    planner = _cpu_trajectory_planner()
    Xstart = np.eye(4)
    Xend = np.eye(4)
    Xend[:3, 3] = [1.0, 2.0, 3.0]
    spy = _TrajectorySpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.cartesian_trajectory(Xstart, Xend, 1.0, 5, 5)

    # The (N, 3, 3) orientation stack is unique to the migrated assembly; the
    # SE(3) utilities on this path only stack rows/vectors (ndim <= 2).
    assert (5, 3, 3) in spy.stack_shapes
