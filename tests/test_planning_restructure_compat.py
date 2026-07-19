"""Compatibility contracts for the planning package decomposition."""

import inspect
import pickle

import numpy as np
import pytest

import ManipulaPy.path_planning as legacy_planning
import ManipulaPy.planning as planning
import ManipulaPy.planning._kernels as runtime
import ManipulaPy.planning.trajectory_planning as implementation


# Captured from release/v1.4 commit 8df3424 before the SR6 decomposition.
_BASE_IMPLEMENTATION_NAMES = frozenset(
    """
    Any CUDA_AVAILABLE CollisionChecker CubicTimeScaling Dict List MatrixExp3
    MatrixLog3 MockCuda NoReturn OptimizedTrajectoryPlanning Optional
    PotentialField QuinticTimeScaling TrajectoryPlanning TransToRp Tuple
    _best_2d_config _h2d_pinned _traj_cpu_njit _trajectory_cpu_fallback
    auto_select_optimal_kernel batch_trajectory_kernel
    benchmark_kernel_performance benchmark_kernel_performance_comprehensive
    cartesian_trajectory_kernel check_cuda_availability compare_implementations
    create_optimized_planner cuda forward_dynamics_kernel
    fused_potential_gradient_kernel get_backend get_cuda_array
    get_gpu_properties get_memory_pool_stats get_optimal_kernel_config
    inverse_dynamics_kernel logger logging make_1d_grid make_2d_grid
    make_2d_grid_optimized njit np optimized_batch_trajectory_generation
    optimized_trajectory_generation optimized_trajectory_generation_monitored
    plt prange print_performance_recommendations profile_start profile_stop
    return_cuda_array setup_cuda_environment_for_40x_speedup time
    trajectory_kernel trajectory_kernel_cache_friendly
    trajectory_kernel_memory_optimized trajectory_kernel_vectorized
    trajectory_kernel_warp_optimized warnings
    """.split()
)
_RESTRUCTURING_IMPLEMENTATION_NAMES = frozenset(
    {
        "_FORWARDED_RUNTIME_NAMES",
        "_ModuleType",
        "_PlanningCompatibilityModule",
        "_runtime",
        "_sys",
    }
)
_BASE_CLASS_METHOD_NAMES = frozenset(
    """
    _apply_collision_avoidance_cpu _apply_collision_avoidance_gpu
    _batch_joint_trajectory_cpu _cartesian_trajectory_cpu
    _cartesian_trajectory_gpu _forward_dynamics_cpu _forward_dynamics_gpu
    _get_optimal_kernel_config _get_or_resize_gpu_array _inverse_dynamics_cpu
    _inverse_dynamics_gpu _joint_trajectory_cpu _joint_trajectory_gpu
    _should_use_gpu batch_joint_trajectory benchmark_all_kernels
    benchmark_performance calculate_derivatives cartesian_trajectory
    cleanup_gpu_memory forward_dynamics_trajectory get_performance_stats
    inverse_dynamics_trajectory joint_trajectory plan_trajectory
    plot_cartesian_trajectory plot_ee_trajectory plot_tcp_trajectory
    plot_trajectory reset_performance_stats
    """.split()
)


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


def test_historical_logger_name_and_patch_reach_moved_method(monkeypatch):
    """The old logger identity and implementation-module patch point survive."""
    assert runtime.logger.name == "ManipulaPy.planning.trajectory_planning"
    original_logger = runtime.logger

    class LoggerProbe:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = LoggerProbe()
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "logger", logger)
        planner = _bare_planner()
        planner.calculate_derivatives(np.arange(4.0)[:, None], 1.0)
        planner._joint_trajectory_cpu(np.array([0.0]), np.array([0.5]), 1.0, 3, 3)
        assert implementation.logger is runtime.logger is logger

    assert any(
        "CPU trajectory generation completed" in message for message in logger.messages
    )
    assert implementation.logger is runtime.logger is original_logger


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


def test_complete_pre_sr6_namespace_manifests_and_symbol_identity():
    """All historical names remain exact; only declared internals are added."""
    implementation_names = {
        name for name in vars(implementation) if not name.startswith("__")
    }
    package_names = {name for name in vars(planning) if not name.startswith("__")}
    legacy_names = {name for name in vars(legacy_planning) if not name.startswith("__")}

    assert implementation_names == (
        _BASE_IMPLEMENTATION_NAMES | _RESTRUCTURING_IMPLEMENTATION_NAMES
    )
    assert package_names == _BASE_IMPLEMENTATION_NAMES | {"trajectory_planning"}
    assert legacy_names == package_names | {"_planning"}

    for name in _BASE_IMPLEMENTATION_NAMES:
        assert getattr(implementation, name) is getattr(planning, name)
        assert getattr(implementation, name) is getattr(legacy_planning, name)
    assert planning.trajectory_planning is implementation
    assert legacy_planning.trajectory_planning is implementation
    assert legacy_planning._planning is planning


def test_complete_pre_sr6_class_surface_and_descriptor_kinds():
    """The planned mixin MRO changes ownership, not the class method surface."""
    planner_class = implementation.OptimizedTrajectoryPlanning
    method_names = {name for name in dir(planner_class) if not name.startswith("__")}
    assert method_names == _BASE_CLASS_METHOD_NAMES

    for name in _BASE_CLASS_METHOD_NAMES:
        descriptor = inspect.getattr_static(planner_class, name)
        expected_kind = (
            staticmethod if name == "plot_trajectory" else type(lambda: None)
        )
        assert isinstance(descriptor, expected_kind), name

    assert tuple(base.__name__ for base in planner_class.__mro__) == (
        "OptimizedTrajectoryPlanning",
        "_GenerationMixin",
        "_DynamicsMixin",
        "_CollisionMixin",
        "_PlottingMixin",
        "object",
    )
