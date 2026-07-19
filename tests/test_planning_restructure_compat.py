"""Compatibility contracts for the planning package decomposition."""

import inspect
import pickle

import numpy as np
import pytest

import ManipulaPy.path_planning as legacy_planning
import ManipulaPy.planning as planning
import ManipulaPy.planning._kernels as runtime
import ManipulaPy.planning.trajectory_planning as implementation


class _BackendProbe:
    """Delegate to the active backend while recording runtime lookups."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)


class _NoCollision:
    def check_collision(self, _configuration):
        return False


class _DeviceArray:
    def __init__(self, shape):
        self._host = np.zeros(shape, dtype=np.float32)

    def copy_to_host(self):
        return self._host.copy()


class _KernelProbe:
    def __init__(self, calls):
        self._calls = calls

    def __getitem__(self, launch_config):
        def launch(*_args):
            self._calls.append(launch_config)

        return launch


def _bare_planner():
    planner = implementation.OptimizedTrajectoryPlanning.__new__(
        implementation.OptimizedTrajectoryPlanning
    )
    planner.joint_limits = np.array([[-1.0, 1.0]], dtype=np.float32)
    planner.torque_limits = np.array([[-10.0, 10.0]], dtype=np.float32)
    planner.collision_checker = None
    planner.potential_field = None
    planner.performance_stats = {"cpu_calls": 0, "total_cpu_time": 0.0}
    planner._last_cpu_time = 0.0
    return planner


@pytest.mark.parametrize("path", ["generation", "dynamics", "collision"])
def test_historical_get_backend_patch_reaches_moved_methods(monkeypatch, path):
    """The historical implementation-module patch point remains effective."""
    calls = []
    active_backend = runtime.get_backend()

    def patched_get_backend():
        calls.append(path)
        return _BackendProbe(active_backend)

    monkeypatch.setattr(implementation, "get_backend", patched_get_backend)
    planner = _bare_planner()

    if path == "generation":
        planner._joint_trajectory_cpu(np.array([0.0]), np.array([0.5]), 1.0, 3, 3)
    elif path == "dynamics":
        planner.calculate_derivatives(np.arange(4.0)[:, None], 1.0)
    else:
        planner.collision_checker = _NoCollision()
        planner._apply_collision_avoidance_cpu(
            np.array([[0.0]], dtype=np.float32), np.array([0.5], dtype=np.float32)
        )

    assert calls == [path]
    assert runtime.get_backend is patched_get_backend


def test_historical_cuda_patch_updates_canonical_runtime(monkeypatch):
    """CUDA objects patched at the old module flow to the shared runtime."""
    sentinel_cuda = object()
    kernel_calls = []
    sentinel_kernel = _KernelProbe(kernel_calls)

    monkeypatch.setattr(implementation, "cuda", sentinel_cuda)
    monkeypatch.setattr(
        implementation, "get_cuda_array", lambda shape, dtype: _DeviceArray(shape)
    )
    monkeypatch.setattr(implementation, "return_cuda_array", lambda _array: None)
    monkeypatch.setattr(implementation, "_h2d_pinned", lambda value: value)
    monkeypatch.setattr(
        implementation,
        "get_optimal_kernel_config",
        lambda *_args: {"grid": 2, "block": 32, "kernel_type": "probe"},
    )
    monkeypatch.setattr(implementation, "cartesian_trajectory_kernel", sentinel_kernel)

    assert runtime.cuda is sentinel_cuda
    assert runtime.cartesian_trajectory_kernel is sentinel_kernel
    planner = _bare_planner()
    planner.performance_stats.update(
        {"gpu_calls": 0, "total_gpu_time": 0.0, "kernel_launches": 0}
    )
    velocity, acceleration = planner._cartesian_trajectory_gpu(
        np.zeros(3), np.ones(3), 1.0, 4, 3
    )
    assert velocity.shape == acceleration.shape == (4, 3)
    assert kernel_calls == [(2, 32)]

    moved_globals = (
        implementation.OptimizedTrajectoryPlanning._cartesian_trajectory_gpu.__globals__
    )
    assert moved_globals["_runtime"] is runtime
    assert "cuda" not in moved_globals
    assert "cartesian_trajectory_kernel" not in moved_globals


def test_canonical_runtime_updates_facade_and_patch_restores_both(monkeypatch):
    """The runtime is authoritative and facade patches restore cleanly."""
    original = runtime.get_backend

    def direct_replacement():
        return None

    runtime.get_backend = direct_replacement
    try:
        assert implementation.get_backend is direct_replacement
    finally:
        runtime.get_backend = original

    def replacement():
        return None

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "get_backend", replacement)
        assert implementation.get_backend is replacement
        assert runtime.get_backend is replacement

    assert implementation.get_backend is original
    assert runtime.get_backend is original


def test_core_constructor_observes_historical_cuda_availability_patch(monkeypatch):
    calls = []

    def unavailable():
        calls.append(True)
        return False

    monkeypatch.setattr(implementation, "check_cuda_availability", unavailable)
    planner = implementation.OptimizedTrajectoryPlanning(
        object(), None, object(), [(-1.0, 1.0)], use_cuda=False, auto_optimize=False
    )

    assert calls == [True]
    assert planner.cuda_available is False


def test_planning_import_paths_preserve_alias_identity_and_class_contract():
    """The package split preserves historical aliases and class semantics."""
    expected_names = (
        "OptimizedTrajectoryPlanning",
        "TrajectoryPlanning",
        "create_optimized_planner",
        "compare_implementations",
        "benchmark_kernel_performance_comprehensive",
        "_trajectory_cpu_fallback",
        "_traj_cpu_njit",
    )
    for name in expected_names:
        values = [
            getattr(module, name)
            for module in (legacy_planning, planning, implementation)
        ]
        assert values[0] is values[1] is values[2]

    for name in implementation.__all__:
        assert getattr(legacy_planning, name) is getattr(implementation, name)
        assert getattr(planning, name) is getattr(implementation, name)

    restructuring_internals = {
        "_runtime",
        "_sys",
        "_ModuleType",
        "_FORWARDED_RUNTIME_NAMES",
        "_PlanningCompatibilityModule",
    }
    assert restructuring_internals.isdisjoint(vars(planning))
    assert restructuring_internals.isdisjoint(vars(legacy_planning))

    planner_class = implementation.OptimizedTrajectoryPlanning
    assert planner_class.__name__ == "OptimizedTrajectoryPlanning"
    assert planner_class.__module__ == "ManipulaPy.planning.trajectory_planning"
    assert isinstance(
        inspect.getattr_static(planner_class, "plot_trajectory"), staticmethod
    )

    planner = planner_class.__new__(planner_class)
    restored = pickle.loads(pickle.dumps(planner))
    assert isinstance(restored, planner_class)
    assert isinstance(restored, legacy_planning.OptimizedTrajectoryPlanning)
