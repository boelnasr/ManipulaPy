# flake8: noqa: E501
"""Compatibility contracts for the planning package decomposition."""

import ast
import hashlib
import inspect
import pickle
import textwrap

import numpy as np
import pytest

import ManipulaPy.path_planning as legacy_planning
import ManipulaPy.planning as planning
import ManipulaPy.planning._kernels as runtime
import ManipulaPy.planning.trajectory_planning as implementation


# Captured from release/v1.4 commit 8df3424 before the SR6 decomposition.
_BASE_UNCONDITIONAL_IMPLEMENTATION_NAMES = frozenset(
    """
    Any CUDA_AVAILABLE CollisionChecker CubicTimeScaling Dict List MatrixExp3
    MatrixLog3 NoReturn OptimizedTrajectoryPlanning Optional
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
_BASE_MOVED_METHOD_HASHES = {
    "_batch_joint_trajectory_cpu": "88a5ab47b8ae9b75a8efb3c6ebe8370fb7e65aea47709b878d9c68cbeec7a976",
    "_cartesian_trajectory_cpu": "0a346cd0c12386ecd27596ddc887c18c3136e6be83ad556dffd5d1732c66373e",
    "_cartesian_trajectory_gpu": "cfe253ac0c567780e7de2ad2dc48f02c1af029c98ee15f1b19f2a16f2873ac6e",
    "_joint_trajectory_cpu": "832e4ec5ccef73fc5145ea7b4fc367e4640e3425a10dcb9c35ae2c3d7a63a7e6",
    "_joint_trajectory_gpu": "70b95df213ae58820de1ccf0ea3def7a1625f0a3e723af31c3aec2012256d364",
    "batch_joint_trajectory": "b13fa5bbd4d77ba032a2dda3d6997277121f0b56b1081c3ae2fe0ca4f0cd47f6",
    "cartesian_trajectory": "711dc4bb4c343e779bfc0259d3b06a0cf1d464c05ea24afc3e48ef87ee28dcd8",
    "joint_trajectory": "58b2685013876277df3c73f27a7e8167191e6219e9937c3e7233301c94163681",
    "_forward_dynamics_cpu": "294362385566dc0a57804bb053d91b34bea838ad9be9514541cfa5dec39cb519",
    "_forward_dynamics_gpu": "362b9aee44dd89821e036d8dd6dcbce3529d31fe8240b0af88fb4df7c069cd1e",
    "_inverse_dynamics_cpu": "78861e7bb709b2050957c19bf9a0bb4ead009a605182074d680460bd01ef6a4b",
    "_inverse_dynamics_gpu": "ec677c9880b260a6d593118e143503f4b04faf6903acdceebc2f325b5464e9e7",
    "calculate_derivatives": "5850b1c775a5675e3e1d513d2348574ff2dd6cd82790084ef9ab63599b6ac516",
    "forward_dynamics_trajectory": "11af305bd9ef9bcff92666db3f43646492eb04a75f8f3cdb42a3c946ecb437ec",
    "inverse_dynamics_trajectory": "b6aaaa0efc0eb1e8d2c3c51c2599e56bb275f454f5b6ec2182ab6cd83dcad31e",
    "_apply_collision_avoidance_cpu": "20873ee00320943622ad8c403937f104bb4db1fd4cbbe2c6660f6efafd5f739c",
    "_apply_collision_avoidance_gpu": "636cac21eeda604736bc97ee0c0cf2d2b6d53e680884ec469986f6042d50056e",
    "plan_trajectory": "6ff8739f484e73e4ae305e28d3cb883fa9b4cebd56255dfe17a831577d48ab12",
    "plot_cartesian_trajectory": "198fe815e263d618aed97a1b46840653de153177c274ca2af05b8ed22aac6783",
    "plot_ee_trajectory": "980511a94726dca6e3921f8845283e9ffc791622a60bd243837096e5634d7976",
    "plot_tcp_trajectory": "14336c8fed8947cc26edad80d3c7f73962f0d3d82dd8ca4e5ed691af8c76b43c",
    "plot_trajectory": "ccbc6e3ae35d09df8498f10663a01a8db5763f4066301a773de8363f21d36be9",
}


class _MovedMethodNormalizer(ast.NodeTransformer):
    """Normalize the declared structural changes from the SR6 pure move."""

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "_runtime":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _stable_ast(node):
    """Serialize AST structure without interpreter-specific metadata fields."""
    if isinstance(node, ast.AST):
        fields = tuple(
            (name, _stable_ast(value))
            for name, value in ast.iter_fields(node)
            if name not in {"ctx", "type_comment", "type_params"}
        )
        return type(node).__name__, fields
    if isinstance(node, list):
        return tuple(_stable_ast(value) for value in node)
    return node


def _normalized_method_hash(descriptor):
    function = (
        descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    )
    source = textwrap.dedent(inspect.getsource(function))
    method = ast.parse(source).body[0]
    normalized = _MovedMethodNormalizer().visit(method)
    payload = repr(_stable_ast(normalized)).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_base_implementation_names(cuda_available):
    """Return the pre-SR6 namespace for the selected CUDA import branch."""
    conditional_names = frozenset() if cuda_available else frozenset({"MockCuda"})
    return _BASE_UNCONDITIONAL_IMPLEMENTATION_NAMES | conditional_names


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


def test_historical_cpu_helper_patch_reaches_generation_methods(monkeypatch):
    """CPU generation still resolves its helper through the old facade."""
    expected = tuple(np.full((3, 1), value, dtype=np.float32) for value in range(3))
    calls = []

    def helper(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(implementation, "_traj_cpu_njit", helper)
    result = _bare_planner()._joint_trajectory_cpu(
        np.array([0.0]), np.array([0.5]), 1.0, 3, 3
    )

    assert calls
    assert runtime._traj_cpu_njit is helper
    np.testing.assert_array_equal(result["positions"], expected[0])


def test_historical_math_helper_patch_reaches_cartesian_generation(monkeypatch):
    """Cartesian generation dynamically resolves historical math helpers."""
    marker = RuntimeError("patched TransToRp reached")

    def patched_trans_to_rp(_transform):
        raise marker

    monkeypatch.setattr(implementation, "TransToRp", patched_trans_to_rp)

    with pytest.raises(RuntimeError, match="patched TransToRp reached") as exc_info:
        _bare_planner().cartesian_trajectory(np.eye(4), np.eye(4), 1.0, 3, 3)

    assert exc_info.value is marker
    assert runtime.TransToRp is patched_trans_to_rp


def test_historical_plotting_patch_reaches_plotting_mixin(monkeypatch):
    """Plotting methods resolve pyplot through the historical facade."""

    class PlotProbe:
        def subplots(self, *_args, **_kwargs):
            raise RuntimeError("patched pyplot reached")

    probe = PlotProbe()
    monkeypatch.setattr(implementation, "plt", probe)
    data = {
        "positions": np.zeros((2, 1)),
        "velocities": np.zeros((2, 1)),
        "accelerations": np.zeros((2, 1)),
    }

    with pytest.raises(RuntimeError, match="patched pyplot reached"):
        implementation.OptimizedTrajectoryPlanning.plot_trajectory(data, 1.0)

    assert runtime.plt is probe


def test_moved_method_bodies_match_pre_sr6_normalized_ast_hashes():
    """Every extracted method remains a pure move from base commit 8df3424."""
    planner_class = implementation.OptimizedTrajectoryPlanning
    actual = {
        name: _normalized_method_hash(inspect.getattr_static(planner_class, name))
        for name in _BASE_MOVED_METHOD_HASHES
    }

    assert actual == _BASE_MOVED_METHOD_HASHES


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
    expected_base_names = _expected_base_implementation_names(
        cuda_available=runtime.CUDA_AVAILABLE
    )
    implementation_names = {
        name for name in vars(implementation) if not name.startswith("__")
    }
    package_names = {name for name in vars(planning) if not name.startswith("__")}
    legacy_names = {name for name in vars(legacy_planning) if not name.startswith("__")}

    assert implementation_names == (
        expected_base_names | _RESTRUCTURING_IMPLEMENTATION_NAMES
    )
    assert package_names == expected_base_names | {"trajectory_planning"}
    assert legacy_names == package_names | {"_planning"}

    for name in expected_base_names:
        assert getattr(implementation, name) is getattr(planning, name)
        assert getattr(implementation, name) is getattr(legacy_planning, name)
    assert planning.trajectory_planning is implementation
    assert legacy_planning.trajectory_planning is implementation
    assert legacy_planning._planning is planning


def test_namespace_manifest_models_cuda_conditional_symbols():
    """The baseline namespace differs only by its conditional CUDA mock."""
    cpu_names = _expected_base_implementation_names(cuda_available=False)
    cuda_names = _expected_base_implementation_names(cuda_available=True)

    assert cpu_names == cuda_names | {"MockCuda"}
    assert "MockCuda" in cpu_names
    assert "MockCuda" not in cuda_names


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


def test_compatibility_subclass_preserves_surface_descriptors_and_distinct_mro():
    """TrajectoryPlanning adds only its intended compatibility subclass layer."""
    compatibility_class = implementation.TrajectoryPlanning
    method_names = {
        name for name in dir(compatibility_class) if not name.startswith("__")
    }
    assert method_names == _BASE_CLASS_METHOD_NAMES

    for name in _BASE_CLASS_METHOD_NAMES:
        descriptor = inspect.getattr_static(compatibility_class, name)
        expected_kind = (
            staticmethod if name == "plot_trajectory" else type(lambda: None)
        )
        assert isinstance(descriptor, expected_kind), name

    assert tuple(base.__name__ for base in compatibility_class.__mro__) == (
        "TrajectoryPlanning",
        "OptimizedTrajectoryPlanning",
        "_GenerationMixin",
        "_DynamicsMixin",
        "_CollisionMixin",
        "_PlottingMixin",
        "object",
    )
