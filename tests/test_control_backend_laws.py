"""Task 8b active-backend control-law regression tests."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from ManipulaPy import backend as be
from ManipulaPy.backend.numpy_backend import NumpyBackend
from ManipulaPy.control import ManipulatorController


class _LawSpyBackend(NumpyBackend):
    """NumPy-compatible backend recording primitives used by control laws."""

    def __init__(self) -> None:
        self.calls = Counter()

    def _record(self, name, value):
        self.calls[name] += 1
        return value

    def asarray(self, obj, dtype=None):
        return self._record("asarray", super().asarray(obj, dtype=dtype))

    def zeros(self, shape, dtype=None):
        return self._record("zeros", super().zeros(shape, dtype=dtype))

    def eye(self, n, dtype=None):
        return self._record("eye", super().eye(n, dtype=dtype))

    def concatenate(self, arrays, axis=0):
        return self._record("concatenate", super().concatenate(arrays, axis=axis))

    def inv(self, a):
        return self._record("inv", super().inv(a))

    def clip(self, x, a_min, a_max):
        return self._record("clip", super().clip(x, a_min, a_max))

    def abs(self, x):
        return self._record("abs", super().abs(x))

    def amax(self, x, axis=None):
        return self._record("amax", super().amax(x, axis=axis))

    def any(self, x, axis=None):
        return self._record("any", super().any(x, axis=axis))


class _Dynamics:
    def mass_matrix(self, q):
        return np.eye(len(q)) * 2.0

    def inverse_dynamics(self, q, dq, ddq, g, ftip):
        return np.ones(len(q)) * 0.25

    def velocity_quadratic_forces(self, q, dq):
        return np.ones(len(q)) * 0.1

    def gravity_forces(self, q, g):
        return np.ones(len(q)) * 0.2

    def jacobian(self, q):
        return np.eye(3, len(q))

    def forward_dynamics(self, q, dq, tau, g, ftip):
        return np.ones(len(q)) * 0.5

    def forward_kinematics(self, q):
        transform = np.eye(4)
        transform[:3, 3] = [0.1, 0.2, 0.0]
        return transform


class _TokenArray:
    def __init__(self, value, token):
        self.value = np.asarray(value)
        self.token = token

    @property
    def shape(self):
        return self.value.shape


class _TokenBackend(NumpyBackend):
    """Device-sensitive fake whose one instance can change cache tokens."""

    def __init__(self, token):
        self.token = token
        self.to_numpy_calls = 0

    def cache_token(self):
        return (id(self), self.token)

    def asarray(self, obj, dtype=None):
        if isinstance(obj, _TokenArray):
            if obj.token != self.token:
                raise AssertionError("cross-device array was not explicitly hosted")
            return obj
        return _TokenArray(np.asarray(obj, dtype=dtype), self.token)

    def to_numpy(self, obj):
        self.to_numpy_calls += 1
        if not isinstance(obj, _TokenArray):
            raise AssertionError("stale owner attempted to host an unowned array")
        return obj.value

    def is_backend_array(self, obj):
        return isinstance(obj, _TokenArray)


@pytest.fixture(autouse=True)
def _restore_backend():
    original = be.get_backend()
    yield
    be._active = original


def _pid(controller, i_clamp=0.15):
    return controller.pid_control(
        [1, -1],
        [0, 0],
        [0, 0],
        [0, 0],
        0.1,
        [2, 2],
        [1, 1],
        [0.5, 0.5],
        i_clamp=i_clamp,
    )


def test_pid_and_computed_torque_use_backend_state_and_functional_clip():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())

    np.testing.assert_allclose(_pid(controller), [2.1, -2.1])
    np.testing.assert_allclose(
        controller.computed_torque_control(
            [1, -1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0, -9.81],
            0.1,
            [2, 2],
            [1, 1],
            [0.5, 0.5],
            i_clamp=0.15,
        ),
        [4.55, -4.05],
    )

    assert np.issubdtype(controller.eint.dtype, np.floating)
    assert spy.calls["zeros"] == 1
    assert spy.calls["clip"] == 2


def test_pid_state_is_normalized_on_numpy_alternate_numpy_transitions():
    controller = ManipulatorController(_Dynamics())
    _pid(controller, i_clamp=None)
    first = controller.eint.copy()

    alternate = _LawSpyBackend()
    be._active = alternate
    _pid(controller, i_clamp=None)
    second = controller.eint.copy()
    assert alternate.calls["asarray"] >= 1

    be.set_backend("numpy")
    _pid(controller, i_clamp=None)
    np.testing.assert_allclose(controller.eint, first * 3)
    np.testing.assert_allclose(second, first * 2)


def test_computed_torque_state_is_normalized_across_backend_transitions():
    controller = ManipulatorController(_Dynamics())
    args = (
        [1, -1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0, -9.81],
        0.1,
        [2, 2],
        [1, 1],
        [0.5, 0.5],
    )
    controller.computed_torque_control(*args)

    alternate = _LawSpyBackend()
    be._active = alternate
    controller.computed_torque_control(*args)
    assert alternate.calls["asarray"] >= 1
    np.testing.assert_allclose(controller.eint, [0.2, -0.2])

    be.set_backend("numpy")
    controller.computed_torque_control(*args)
    np.testing.assert_allclose(controller.eint, [0.3, -0.3])


@pytest.mark.parametrize("state_name", ["eint", "x_hat"])
def test_state_moves_when_same_backend_instance_changes_device_token(state_name):
    backend = _TokenBackend(token=0)
    be._active = backend
    controller = ManipulatorController(_Dynamics())
    controller._set_state(state_name, backend.asarray([1.0, 2.0]))
    old_state = getattr(controller, state_name)

    backend.token = 1
    new_state = controller._normalize_state(state_name)

    assert new_state is not old_state
    assert new_state.token == 1
    assert backend.to_numpy_calls == 1


def test_state_does_not_host_transfer_when_backend_and_token_are_unchanged():
    backend = _TokenBackend(token=0)
    be._active = backend
    controller = ManipulatorController(_Dynamics())
    state = controller._set_state("P", backend.asarray(np.eye(2)))

    assert controller._normalize_state("P") is state
    assert backend.to_numpy_calls == 0


@pytest.mark.parametrize("state_name", ["eint", "x_hat", "P"])
def test_caller_replaced_state_does_not_use_stale_owner_metadata(state_name):
    recorded_backend = _TokenBackend(token=0)
    be._active = recorded_backend
    controller = ManipulatorController(_Dynamics())
    controller._set_state(state_name, recorded_backend.asarray([1.0, 2.0]))

    active_backend = _TokenBackend(token=7)
    replacement = active_backend.asarray([3.0, 4.0])
    setattr(controller, state_name, replacement)
    be._active = active_backend

    assert controller._normalize_state(state_name) is replacement
    assert recorded_backend.to_numpy_calls == 0
    assert active_backend.to_numpy_calls == 0


def test_adaptive_normalizes_state_and_every_dynamics_result():
    controller = ManipulatorController(_Dynamics())
    args = ([0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.1, -0.2], 0.5)
    controller.adaptive_control(*args)

    spy = _LawSpyBackend()
    be._active = spy
    result = controller.adaptive_control(*args)

    np.testing.assert_allclose(result, [2.4, 2.1])
    np.testing.assert_allclose(controller.parameter_estimate, [0.1, -0.2])
    # Seven inputs, persistent state, and four dynamics results are normalized.
    assert spy.calls["asarray"] >= 12


def test_kalman_uses_backend_construction_linalg_and_normalizes_state_transitions():
    controller = ManipulatorController(_Dynamics())
    controller.kalman_filter_predict(
        [0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6, 0.1, np.eye(4) * 0.01
    )

    spy = _LawSpyBackend()
    be._active = spy
    controller.kalman_filter_predict(
        [0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6, 0.1, np.eye(4) * 0.01
    )
    controller.kalman_filter_update(np.ones(4) * 0.02, np.eye(4) * 0.1)

    assert spy.calls["concatenate"] >= 1
    assert spy.calls["eye"] >= 2
    assert spy.calls["inv"] == 1
    assert spy.calls["asarray"] >= 10
    be.set_backend("numpy")
    controller.kalman_filter_update(np.ones(4) * 0.03, np.eye(4) * 0.1)
    assert isinstance(controller.x_hat, np.ndarray)
    assert isinstance(controller.P, np.ndarray)


def test_kalman_manual_state_without_constructor_remains_supported():
    controller = ManipulatorController.__new__(ManipulatorController)
    controller.x_hat = np.zeros(4)
    controller.P = np.eye(4)

    controller.kalman_filter_update(np.ones(4), np.eye(4))

    np.testing.assert_allclose(controller.x_hat, np.full(4, 0.5))
    np.testing.assert_allclose(controller.P, np.eye(4) * 0.5)


def test_kalman_predict_and_update_preserve_analytic_equations():
    class _AnalyticDynamics:
        def forward_dynamics(self, q, dq, tau, g, ftip):
            return np.array([0.5, -1.0])

    controller = ManipulatorController(_AnalyticDynamics())
    controller.x_hat = np.array([1.0, 2.0, 3.0, 4.0])
    controller.P = np.eye(4)
    controller.kalman_filter_predict(
        [1, 2], [3, 4], [0, 0], [0, 0, -9.81], [0] * 6, 0.1, np.eye(4)
    )
    np.testing.assert_allclose(controller.x_hat, [1.3, 2.4, 3.05, 3.9])
    np.testing.assert_allclose(controller.P, np.eye(4) * 2.0)

    controller.kalman_filter_update([0, 1, 2, 3], np.eye(4) * 3.0)
    np.testing.assert_allclose(controller.x_hat, [0.78, 1.84, 2.63, 3.54])
    np.testing.assert_allclose(controller.P, np.eye(4) * 1.2)


def test_all_stateless_laws_return_backend_native_results_with_numpy_parity():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())

    results = [
        controller.pd_control([1, 1], [0, 0], [0, 0], [0, 0], [2, 2], [1, 1]),
        controller.robust_control(
            [0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.2, 0.2], [0.5, 0.5]
        ),
        controller.feedforward_control([0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6),
        controller.pd_feedforward_control(
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [2, 2],
            [1, 1],
            [0, 0, -9.81],
            [0] * 6,
        ),
        controller.joint_space_control([1, 1], [0, 0], [0, 0], [2, 2], [1, 1]),
        controller.cartesian_space_control(
            [1, 1, 0], [0, 0], [0, 0], [2, 2, 2], [1, 1, 1]
        ),
    ]

    expected = ([2, 2], [2.4, 2.4], [0.25, 0.25], [2.25, 2.25], [2, 2], [1.8, 1.6])
    for actual, wanted in zip(results, expected):
        assert spy.is_backend_array(actual)
        np.testing.assert_allclose(actual, wanted)


@pytest.mark.parametrize(
    ("kp", "kd"),
    [([2, 3, 4], [0.5, 1, 2]), (np.diag([2, 3, 4]), np.diag([0.5, 1, 2]))],
)
def test_cartesian_control_preserves_jacobian_transpose_and_gain_semantics(kp, kd):
    class _CartesianDynamics:
        def forward_kinematics(self, q):
            transform = np.eye(4)
            transform[:3, 3] = [0.5, 0.25, -0.1]
            return transform

        def jacobian(self, q):
            return np.array([[1, 0], [0, 2], [1, 1]])

    controller = ManipulatorController(_CartesianDynamics())
    tau = controller.cartesian_space_control([1, 0, 0], [0, 0], [0.2, -0.1], kp, kd)
    np.testing.assert_allclose(tau, [1.1, -0.9])


def test_limits_and_metrics_dispatch_hot_array_operations_and_return_scalars():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())

    q, dq, tau = controller.enforce_limits(
        [-2, 2], [0.1, 0.2], [-5, 5], [[-1, 1], [-1, 1]], [[-3, 3], [-3, 3]]
    )
    np.testing.assert_array_equal(q, [-1, 1])
    np.testing.assert_array_equal(dq, [0.1, 0.2])
    np.testing.assert_array_equal(tau, [-3, 3])
    assert spy.calls["clip"] == 2

    time = [0, 1, 2, 3]
    response = [0, 1, 1, 1]
    assert controller.calculate_rise_time(time, response, 1.0) == 0.0
    assert controller.calculate_percent_overshoot(response, 1.0) == 0.0
    assert controller.calculate_settling_time(time, response, 1.0) == 1.0
    assert controller.calculate_steady_state_error(response, 1.0) == 0.0
    assert spy.calls["abs"] >= 1
    assert spy.calls["amax"] >= 1
    assert spy.calls["any"] >= 1


def test_metric_edge_contracts_are_preserved():
    controller = ManipulatorController(_Dynamics())
    assert controller.calculate_percent_overshoot([0, 1], 0.0) == 0.0
    assert controller.calculate_percent_overshoot([-1, -1.2], -1.0) == pytest.approx(
        0.0
    )
    assert controller.calculate_settling_time([0, 1, 2], [0, -1, -1], -1.0) == 1.0


def test_ultimate_gain_optimizer_is_scoped_to_host_backend_and_restores_caller():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())

    controller.find_ultimate_gain_and_period([0.0, 0.0], [0.1, 0.1], 0.01, max_steps=1)

    assert be.get_backend() is spy
    assert spy.calls["asarray"] == 0
    assert controller._state_owners["eint"].backend is be.get_registered("numpy")


def test_ultimate_gain_optimizer_restores_exact_backend_on_exception():
    class _FailingDynamics(_Dynamics):
        def mass_matrix(self, q):
            raise RuntimeError("optimizer failed")

    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_FailingDynamics())

    with pytest.raises(RuntimeError, match="optimizer failed"):
        controller.find_ultimate_gain_and_period(
            [0.0, 0.0], [0.1, 0.1], 0.01, max_steps=1
        )

    assert be.get_backend() is spy


def test_plot_response_is_entirely_host_scoped_and_restores_caller_backend():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())
    backends_seen = []

    def record_plot_backend(*args, **kwargs):
        backends_seen.append(be.get_backend())

    with (
        patch("matplotlib.pyplot.plot", side_effect=record_plot_backend),
        patch("matplotlib.pyplot.show"),
    ):
        controller.plot_steady_state_response([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 1.0)

    assert backends_seen == [be.get_registered("numpy")]
    assert be.get_backend() is spy
    assert spy.calls["asarray"] == 0


def test_plot_response_restores_exact_backend_when_plotting_raises():
    spy = _LawSpyBackend()
    be._active = spy
    controller = ManipulatorController(_Dynamics())

    def fail_on_host(*args, **kwargs):
        assert be.get_backend() is be.get_registered("numpy")
        raise RuntimeError("plot failed")

    with (
        patch("matplotlib.pyplot.plot", side_effect=fail_on_host),
        pytest.raises(RuntimeError, match="plot failed"),
    ):
        controller.plot_steady_state_response([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], 1.0)

    assert be.get_backend() is spy


def test_remaining_control_laws_have_real_cupy_native_parity_when_available():
    cp = pytest.importorskip("cupy")
    cupy_array_type = getattr(cp, "ndarray", None)
    if not (
        isinstance(cupy_array_type, type)
        and cupy_array_type.__module__.startswith("cupy")
        and callable(getattr(cp, "asnumpy", None))
    ):
        pytest.skip("real CuPy is unavailable")
    try:
        cp.zeros(1)
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        pytest.skip(f"CuPy device is unavailable: {exc}")

    numpy_controller = ManipulatorController(_Dynamics())
    expected = [
        numpy_controller.robust_control(
            [0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.2, 0.2], [0.5, 0.5]
        ),
        numpy_controller.adaptive_control(
            [0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.1, -0.2], 0.5
        ),
        numpy_controller.feedforward_control(
            [0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6
        ),
        numpy_controller.pd_feedforward_control(
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [2, 2],
            [1, 1],
            [0, 0, -9.81],
            [0] * 6,
        ),
        numpy_controller.joint_space_control([1, 1], [0, 0], [0, 0], [2, 2], [1, 1]),
        numpy_controller.cartesian_space_control(
            [1, 1, 0], [0, 0], [0, 0], [2, 2, 2], [1, 1, 1]
        ),
    ]

    cupy_controller = ManipulatorController(_Dynamics())
    with be.use_backend("cupy"):
        actual = [
            cupy_controller.robust_control(
                [0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.2, 0.2], [0.5, 0.5]
            ),
            cupy_controller.adaptive_control(
                [0, 0], [0, 0], [1, 1], [0, 0, -9.81], [0, 0, 0], [0.1, -0.2], 0.5
            ),
            cupy_controller.feedforward_control(
                [0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6
            ),
            cupy_controller.pd_feedforward_control(
                [1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [2, 2],
                [1, 1],
                [0, 0, -9.81],
                [0] * 6,
            ),
            cupy_controller.joint_space_control([1, 1], [0, 0], [0, 0], [2, 2], [1, 1]),
            cupy_controller.cartesian_space_control(
                [1, 1, 0], [0, 0], [0, 0], [2, 2, 2], [1, 1, 1]
            ),
        ]
        q, dq, tau = cupy_controller.enforce_limits(
            [-2, 2], [0.1, 0.2], [-5, 5], [[-1, 1], [-1, 1]], [[-3, 3], [-3, 3]]
        )
        cupy_controller.kalman_filter_predict(
            [0, 0], [0, 0], [0, 0], [0, 0, -9.81], [0] * 6, 0.1, cp.eye(4) * 0.01
        )
        cupy_controller.kalman_filter_update(cp.ones(4) * 0.02, cp.eye(4) * 0.1)

        for value, reference in zip(actual, expected):
            assert isinstance(value, cp.ndarray)
            np.testing.assert_allclose(cp.asnumpy(value), reference)
        assert all(isinstance(value, cp.ndarray) for value in (q, dq, tau))
        assert isinstance(cupy_controller.parameter_estimate, cp.ndarray)
        assert isinstance(cupy_controller.x_hat, cp.ndarray)
        assert isinstance(cupy_controller.P, cp.ndarray)
        assert (
            cupy_controller.calculate_percent_overshoot(cp.asarray([0, 1]), 1.0) == 0.0
        )
