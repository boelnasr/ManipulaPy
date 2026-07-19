#!/usr/bin/env python3
"""Control array-backend boundary regression tests."""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args, get_type_hints
from unittest.mock import patch

import numpy as np
import pytest

from ManipulaPy import backend as be
from ManipulaPy.backend.numpy_backend import NumpyBackend
import ManipulaPy.control as control
from ManipulaPy.control import ManipulatorController, _validate_i_clamp
from ManipulaPy.control import manipulator_controller
from tests import test_control as control_tests


class _BoundarySpyBackend(NumpyBackend):
    def __init__(self) -> None:
        self.asarray_calls = 0
        self.to_numpy_calls = 0

    def asarray(self, obj, dtype=None):
        self.asarray_calls += 1
        if isinstance(obj, _DeviceValue):
            return obj
        return super().asarray(obj, dtype=dtype)

    def to_numpy(self, value):
        self.to_numpy_calls += 1
        if isinstance(value, _DeviceValue):
            return value.host
        return super().to_numpy(value)

    def is_backend_array(self, value):
        return isinstance(value, _DeviceValue) or super().is_backend_array(value)


class _DeviceValue:
    def __init__(self, value) -> None:
        self.host = np.asarray(value)

    def __array__(self, dtype=None):
        raise TypeError("implicit host conversion is forbidden")


class _Dynamics:
    def mass_matrix(self, _thetalist):
        return np.eye(2)

    def inverse_dynamics(self, *args):
        return np.zeros(2)


@pytest.fixture(autouse=True)
def _restore_backend():
    original = be.get_backend()
    yield
    be._active = original


def test_control_import_does_not_attempt_to_import_cupy(monkeypatch):
    """The control package remains importable when CuPy imports are blocked."""
    real_import = builtins.__import__

    def import_without_cupy(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("CuPy blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cupy)
    module = importlib.reload(importlib.import_module("ManipulaPy.control"))

    assert module.ManipulatorController is not None


def test_control_test_fixture_restores_exact_previous_backend():
    previous = _BoundarySpyBackend()
    be._active = previous
    case = control_tests.TestManipulatorController(methodName="test_pd_control")
    processor = SimpleNamespace(dynamics=SimpleNamespace(Glist=[object(), object()]))

    with patch.object(control_tests, "URDFToSerialManipulator", return_value=processor):
        case.setUp()
        case.doCleanups()

    assert be.get_backend() is previous


def test_control_has_no_direct_cupy_or_legacy_conversion_ownership():
    """Control delegates device ownership to the active array backend."""
    control_dir = Path(__file__).parents[1] / "ManipulaPy" / "control"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in control_dir.glob("*.py")
    )

    assert "import cupy" not in source
    assert "CUPY_AVAILABLE" not in source
    assert "_to_numpy" not in source


def test_control_backend_boundary_annotations_use_backend_array_alias():
    source = Path(manipulator_controller.__file__).read_text(encoding="utf-8")
    assert "TypeAlias" not in source
    assert manipulator_controller.BackendArray is Any
    boundary_parameters = {
        "computed_torque_control": (
            "thetalistd",
            "dthetalistd",
            "ddthetalistd",
            "thetalist",
            "dthetalist",
            "g",
            "Kp",
            "Ki",
            "Kd",
        ),
        "pd_control": (
            "desired_position",
            "desired_velocity",
            "current_position",
            "current_velocity",
            "Kp",
            "Kd",
        ),
        "pid_control": (
            "thetalistd",
            "dthetalistd",
            "thetalist",
            "dthetalist",
            "Kp",
            "Ki",
            "Kd",
        ),
        "robust_control": (
            "thetalist",
            "dthetalist",
            "ddthetalist",
            "g",
            "Ftip",
            "disturbance_estimate",
            "adaptation_gain",
        ),
        "adaptive_control": (
            "thetalist",
            "dthetalist",
            "ddthetalist",
            "g",
            "Ftip",
            "measurement_error",
        ),
        "kalman_filter_predict": (
            "thetalist",
            "dthetalist",
            "taulist",
            "g",
            "Ftip",
            "Q",
        ),
        "kalman_filter_update": ("z", "R"),
        "kalman_filter_control": (
            "thetalistd",
            "dthetalistd",
            "thetalist",
            "dthetalist",
            "taulist",
            "g",
            "Ftip",
            "Q",
            "R",
        ),
        "feedforward_control": (
            "desired_position",
            "desired_velocity",
            "desired_acceleration",
            "g",
            "Ftip",
        ),
        "pd_feedforward_control": (
            "desired_position",
            "desired_velocity",
            "desired_acceleration",
            "current_position",
            "current_velocity",
            "Kp",
            "Kd",
            "g",
            "Ftip",
        ),
        "enforce_limits": (
            "thetalist",
            "dthetalist",
            "tau",
            "joint_limits",
            "torque_limits",
        ),
        "joint_space_control": (
            "desired_joint_angles",
            "current_joint_angles",
            "current_joint_velocities",
            "Kp",
            "Kd",
        ),
        "cartesian_space_control": (
            "desired_position",
            "current_joint_angles",
            "current_joint_velocities",
            "Kp",
            "Kd",
        ),
    }

    for method_name, parameter_names in boundary_parameters.items():
        hints = get_type_hints(getattr(ManipulatorController, method_name))
        for parameter_name in parameter_names:
            assert hints[parameter_name] is Any
        if method_name == "kalman_filter_control":
            assert get_args(hints["return"]) == (Any, Any)
        elif method_name == "enforce_limits":
            assert get_args(hints["return"]) == (Any, Any, Any)
        elif method_name not in ("kalman_filter_predict", "kalman_filter_update"):
            assert hints["return"] is Any


def test_pd_entry_normalizes_with_active_backend_without_host_transfer():
    backend = _BoundarySpyBackend()
    be._active = backend
    controller = ManipulatorController(_Dynamics())

    result = controller.pd_control(
        [0.5, -0.5], [0.1, -0.1], [0.2, -0.6], [0.0, 0.0], [2.0, 3.0], [0.5, 0.5]
    )

    np.testing.assert_allclose(result, [0.65, 0.25])
    assert backend.asarray_calls == 6
    assert backend.to_numpy_calls == 0


def test_computed_torque_normalizes_dynamics_results_with_active_backend():
    backend = _BoundarySpyBackend()
    be._active = backend
    controller = ManipulatorController(_Dynamics())

    result = controller.computed_torque_control(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0, -9.81],
        0.1,
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    )

    np.testing.assert_array_equal(result, np.zeros(2))
    assert backend.asarray_calls == 11
    assert backend.to_numpy_calls == 0


def test_host_helper_does_not_round_trip_numpy_host_array():
    backend = _BoundarySpyBackend()
    be._active = backend
    value = np.array([1.0, 2.0])

    result = control._to_host_array(value)

    assert result is value
    assert backend.to_numpy_calls == 0


def test_host_helper_converts_backend_native_value_once():
    backend = _BoundarySpyBackend()
    be._active = backend

    result = control._to_host_array(_DeviceValue([1.0, 2.0]))

    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))
    assert backend.to_numpy_calls == 1


def test_integral_clamp_explicitly_hosts_backend_native_scalar():
    backend = _BoundarySpyBackend()
    be._active = backend

    result = _validate_i_clamp(_DeviceValue(2.5))

    assert result == 2.5
    assert backend.to_numpy_calls == 1


def test_numpy_pd_golden_output_contract_is_unchanged():
    controller = ManipulatorController(_Dynamics())

    result = controller.pd_control(
        np.array([0.5, -0.5]),
        np.array([0.1, -0.1]),
        np.array([0.2, -0.6]),
        np.zeros(2),
        np.array([2.0, 3.0]),
        np.array([0.5, 0.5]),
    )

    np.testing.assert_allclose(result, np.array([0.65, 0.25]), rtol=0.0, atol=1e-15)
    assert type(result) is np.ndarray
    assert result.dtype == np.float64


def test_numpy_pid_state_and_invalid_clamp_exception_are_unchanged():
    controller = ManipulatorController(_Dynamics())
    args = ([1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.1)

    result = controller.pid_control(*args, [0.0, 0.0], [2.0, 2.0], [0.0, 0.0])

    np.testing.assert_array_equal(result, np.array([0.2, 0.2]))
    np.testing.assert_array_equal(controller.eint, np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="positive and finite"):
        controller.pid_control(
            *args, [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], i_clamp=np.nan
        )
