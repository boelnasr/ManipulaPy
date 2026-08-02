"""Compatibility contracts for the SR9 CUDA package decomposition."""

import ast
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ManipulaPy import cuda_kernels
from ManipulaPy.cuda_kernels import (
    _runtime,
    field_kernels,
    memory,
    registry,
    trajectory_kernels,
)


EXPECTED_ALL = [
    "CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "check_cuda_availability",
    "check_cupy_availability",
    "trajectory_kernel",
    "inverse_dynamics_kernel",
    "forward_dynamics_kernel",
    "cartesian_trajectory_kernel",
    "fused_potential_gradient_kernel",
    "batch_trajectory_kernel",
    "trajectory_kernel_vectorized",
    "trajectory_kernel_memory_optimized",
    "trajectory_kernel_warp_optimized",
    "trajectory_kernel_cache_friendly",
    "optimized_trajectory_generation",
    "optimized_trajectory_generation_monitored",
    "optimized_potential_field",
    "optimized_batch_trajectory_generation",
    "get_optimal_kernel_config",
    "auto_select_optimal_kernel",
    "_best_2d_config",
    "get_cuda_array",
    "return_cuda_array",
    "get_memory_pool_stats",
    "CUDAPerformanceMonitor",
    "profile_start",
    "profile_stop",
    "benchmark_kernel_performance",
    "make_1d_grid",
    "make_2d_grid",
    "make_2d_grid_optimized",
    "get_gpu_properties",
    "trajectory_cpu_fallback",
    "print_performance_recommendations",
    "setup_cuda_environment_for_40x_speedup",
    "attractive_potential_kernel",
    "repulsive_potential_kernel",
    "gradient_kernel",
]

CPU_NAMESPACE = {
    "Any",
    "CUDAPerformanceMonitor",
    "CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "Dict",
    "FAST_MATH",
    "MockCuda",
    "NoReturn",
    "Optional",
    "Tuple",
    "_MockMemoryPool",
    "_PINNED_MEMORY_OPT_IN",
    "_best_2d_config",
    "_cuda_error",
    "_cuda_memory_pool",
    "_cuda_routing_enabled",
    "_cuda_safe_to_probe",
    "_detect_cuda_capability",
    "_get_backend",
    "_h2d_pinned",
    "_nb_cfg",
    "_optimized_trajectory_generation_monitored_cuda",
    "_perf_monitor",
    "attractive_potential_kernel",
    "auto_select_optimal_kernel",
    "batch_trajectory_kernel",
    "benchmark_kernel_performance",
    "cartesian_trajectory_kernel",
    "check_cuda_availability",
    "check_cupy_availability",
    "cp",
    "cuda",
    "float32",
    "float_t",
    "forward_dynamics_kernel",
    "fused_potential_gradient_kernel",
    "get_cuda_array",
    "get_gpu_properties",
    "get_memory_pool_stats",
    "get_optimal_kernel_config",
    "gradient_kernel",
    "int32",
    "inverse_dynamics_kernel",
    "logger",
    "logging",
    "lru_cache",
    "make_1d_grid",
    "make_2d_grid",
    "make_2d_grid_optimized",
    "math",
    "np",
    "optimized_batch_trajectory_generation",
    "optimized_potential_field",
    "optimized_trajectory_generation",
    "optimized_trajectory_generation_monitored",
    "os",
    "perf_counter",
    "print_performance_recommendations",
    "profile_start",
    "profile_stop",
    "repulsive_potential_kernel",
    "return_cuda_array",
    "setup_cuda_environment_for_40x_speedup",
    "trajectory_cpu_fallback",
    "trajectory_kernel",
    "trajectory_kernel_cache_friendly",
    "trajectory_kernel_memory_optimized",
    "trajectory_kernel_vectorized",
    "trajectory_kernel_warp_optimized",
    "warnings",
}

OWNERS = {
    _runtime: [
        "FAST_MATH",
        "logger",
        "_cuda_safe_to_probe",
        "_detect_cuda_capability",
        "CUDA_AVAILABLE",
        "cuda",
        "float32",
        "int32",
        "_cuda_error",
        "MockCuda",
        "CUPY_AVAILABLE",
        "cp",
        "float_t",
        "_PINNED_MEMORY_OPT_IN",
        "_get_backend",
    ],
    memory: [
        "_h2d_pinned",
        "_MockMemoryPool",
        "_cuda_memory_pool",
        "get_cuda_array",
        "return_cuda_array",
        "get_memory_pool_stats",
    ],
    trajectory_kernels: [
        "trajectory_cpu_fallback",
        "trajectory_kernel",
        "trajectory_kernel_vectorized",
        "trajectory_kernel_memory_optimized",
        "trajectory_kernel_warp_optimized",
        "trajectory_kernel_cache_friendly",
        "inverse_dynamics_kernel",
        "forward_dynamics_kernel",
        "cartesian_trajectory_kernel",
        "batch_trajectory_kernel",
        "_optimized_trajectory_generation_monitored_cuda",
        "optimized_trajectory_generation_monitored",
        "optimized_trajectory_generation",
        "optimized_batch_trajectory_generation",
    ],
    field_kernels: [
        "fused_potential_gradient_kernel",
        "optimized_potential_field",
        "attractive_potential_kernel",
        "repulsive_potential_kernel",
        "gradient_kernel",
    ],
    registry: [
        "check_cuda_availability",
        "check_cupy_availability",
        "make_1d_grid",
        "make_2d_grid",
        "make_2d_grid_optimized",
        "get_gpu_properties",
        "CUDAPerformanceMonitor",
        "_perf_monitor",
        "get_optimal_kernel_config",
        "_best_2d_config",
        "auto_select_optimal_kernel",
        "profile_start",
        "profile_stop",
        "benchmark_kernel_performance",
        "_cuda_routing_enabled",
        "print_performance_recommendations",
        "setup_cuda_environment_for_40x_speedup",
    ],
}

EXPECTED_HASHES = {
    "batch_trajectory_kernel": "d12d4571b9a9a1a8f25a16c1eb415b70e0f4a75e10eecb4e8d156da4267838ac",
    "cartesian_trajectory_kernel": "741a048b733cfdeeb6525971c6bc6810388049fd585b876c3cd1c05d434b3b4b",
    "forward_dynamics_kernel": "b68c1f0ba7fdf4e4f5ef6fd2d53b58d85e527b18b524ed4158c497ea77cac21b",
    "fused_potential_gradient_kernel": "afebfa8639fd3153a714f009b51fc046ddefe937d90351e77855e76c2fa4022a",
    "inverse_dynamics_kernel": "2f8a21b2ff128c3cdca9e5b4fe7dc72aadac8a4bd3ecd62162f75123420e6f3e",
    "matrix_vector_multiply_6x6": "00086faf13936989888986e40d576514fe8e3856d557892001951199d34b688e",
    "trajectory_kernel": "df1e6b8d4ca2c7490f4d67e497ba59dc41d5aac195c7c4a81066a20c2e2c164c",
    "trajectory_kernel_cache_friendly": "9302986ee8dc5596264e5a6665758ff4215cc436dcd4b328bdbb3b620cf8ca95",
    "trajectory_kernel_memory_optimized": "64f969d5bde8546939ef82ae8e87650aef34d9a219f31de5291d418eab24cd97",
    "trajectory_kernel_vectorized": "9ab7ffaf7fb7791c11a0b1319f95e31bac2b8892191f9d9b6da3c1fcae73bd52",
    "trajectory_kernel_warp_optimized": "341c79806193337a848dcd08ddf3c5d44f6ed71acb92acc501ac9566fa76f1df",
}


def test_exact_all_list_and_order():
    assert cuda_kernels.__all__ == EXPECTED_ALL


def test_complete_cpu_namespace_manifest():
    assert not cuda_kernels.CUDA_AVAILABLE
    actual = {name for name in vars(cuda_kernels) if not name.startswith("__")}
    assert actual == CPU_NAMESPACE | {
        "_runtime",
        "memory",
        "trajectory_kernels",
        "field_kernels",
        "registry",
    }


@pytest.mark.parametrize(
    "owner,name", [(owner, name) for owner, names in OWNERS.items() for name in names]
)
def test_symbol_destination_and_package_identity(owner, name):
    assert getattr(cuda_kernels, name) is getattr(owner, name)


def test_no_proxy_implementation_names_leak():
    assert {
        "_CompatibilityModule",
        "_OWNERS",
        "_SUBSCRIBERS",
        "activate_proxies",
        "set_compat_value",
    }.isdisjoint(vars(cuda_kernels))


def test_package_cuda_patch_updates_all_existing_imports(monkeypatch):
    saved_wrapper = cuda_kernels.optimized_trajectory_generation_monitored
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    for module in (_runtime, memory, trajectory_kernels, field_kernels, registry):
        assert module.CUDA_AVAILABLE is True
    monkeypatch.setattr(
        registry, "_get_backend", lambda: type("B", (), {"gpu_capable": False})()
    )
    result = saved_wrapper(np.array([0.0]), np.array([1.0]), 1.0, 2, 1)
    assert result[0].shape == (2, 1)


def test_package_cupy_patch_reaches_environment_setup(monkeypatch):
    original = cuda_kernels.CUPY_AVAILABLE
    with monkeypatch.context() as patch:
        patch.setattr(cuda_kernels, "CUPY_AVAILABLE", True)
        assert registry.CUPY_AVAILABLE is True
    assert cuda_kernels.CUPY_AVAILABLE is original
    assert registry.CUPY_AVAILABLE is original


def test_private_cuda_implementation_patch_reaches_direct_wrapper(monkeypatch):
    sentinel = object()
    fake = lambda *args, **kwargs: sentinel
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(
        registry, "_get_backend", lambda: type("B", (), {"gpu_capable": True})()
    )
    monkeypatch.setattr(
        cuda_kernels, "_optimized_trajectory_generation_monitored_cuda", fake
    )
    assert trajectory_kernels._optimized_trajectory_generation_monitored_cuda is fake
    assert cuda_kernels.optimized_trajectory_generation([0], [1], 1, 2, 1) is sentinel


def test_patch_restore_does_not_leave_split_brain(monkeypatch):
    originals = {
        name: getattr(cuda_kernels, name)
        for name in (
            "CUDA_AVAILABLE",
            "CUPY_AVAILABLE",
            "_optimized_trajectory_generation_monitored_cuda",
        )
    }
    with monkeypatch.context() as patch:
        patch.setattr(cuda_kernels, "CUDA_AVAILABLE", object())
        patch.setattr(cuda_kernels, "CUPY_AVAILABLE", object())
        patch.setattr(
            cuda_kernels,
            "_optimized_trajectory_generation_monitored_cuda",
            lambda: None,
        )
    for name, original in originals.items():
        for module in (
            cuda_kernels,
            _runtime,
            memory,
            trajectory_kernels,
            field_kernels,
            registry,
        ):
            if name in vars(module):
                assert getattr(module, name) is original


def _cuda_kernel_digests(module) -> dict:
    """Hash each cuda.jit kernel's own source lines.

    The AST is used only to *locate* the kernels; the digest is taken over
    repo-owned bytes. ``ast.dump`` cannot be hashed here because it renders
    whatever fields the running CPython declares in ``_fields``, so its text
    moves with the interpreter: 3.9 dropped ``annotation=None``/``kind=None``,
    3.12 appended PEP 695 ``type_params=[]`` to every ``FunctionDef``, and
    3.13 flips the ``show_empty`` default. The kernel source does not move.

    ``split("\\n")`` rather than ``splitlines()`` is deliberate: ``splitlines``
    also breaks on form feed and \\x0b, which the tokenizer does not count as
    line breaks, so it could desynchronise from ``lineno``.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    lines = source.split("\n")  # read_text() already normalised newlines
    digests = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_cuda_kernel = any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "cuda"
            and decorator.func.attr == "jit"
            for decorator in node.decorator_list
        )
        if not is_cuda_kernel:
            continue
        first = node.decorator_list[0].lineno  # include the decorator itself
        payload = "\n".join(
            line.rstrip() for line in lines[first - 1 : node.end_lineno]
        )
        digests[node.name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digests


def test_raw_cuda_kernel_source_hashes_across_modules():
    actual = {}
    for module in (trajectory_kernels, field_kernels):
        actual.update(_cuda_kernel_digests(module))
    assert actual == EXPECTED_HASHES


def test_cuda_decorator_definition_order():
    trajectory_source = Path(trajectory_kernels.__file__).read_text()
    field_source = Path(field_kernels.__file__).read_text()
    tree = ast.parse(trajectory_source)
    names = []
    for statement in tree.body:
        if isinstance(statement, ast.If):
            for child in statement.body:
                if isinstance(child, ast.FunctionDef) and child.decorator_list:
                    names.append(child.name)
                if (
                    isinstance(child, ast.ImportFrom)
                    and child.module == "field_kernels"
                ):
                    names.append("fused_potential_gradient_kernel")
    assert "def fused_potential_gradient_kernel" in field_source
    assert names[:11] == list(EXPECTED_HASHES)[5:6] + [
        "trajectory_kernel",
        "trajectory_kernel_vectorized",
        "trajectory_kernel_memory_optimized",
        "trajectory_kernel_warp_optimized",
        "trajectory_kernel_cache_friendly",
        "inverse_dynamics_kernel",
        "forward_dynamics_kernel",
        "cartesian_trajectory_kernel",
        "fused_potential_gradient_kernel",
        "batch_trajectory_kernel",
    ]


def test_numba_initialization_order():
    source = Path(_runtime.__file__).read_text()
    tree = ast.parse(source)

    def assignment_target(statement):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return None
        target = (
            statement.targets[0]
            if isinstance(statement, ast.Assign)
            else statement.target
        )
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            return f"{target.value.id}.{target.attr}"
        return None

    module_assignments = {
        assignment_target(statement): index
        for index, statement in enumerate(tree.body)
        if assignment_target(statement) is not None
    }
    detection_calls = [
        index
        for index, statement in enumerate(tree.body)
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_detect_cuda_capability"
    ]
    assert len(detection_calls) == 1
    detection_index = detection_calls[0]
    assert module_assignments["_nb_cfg.CUDA_CACHE_SIZE"] < detection_index
    assert module_assignments["_nb_cfg.CUDA_LOW_OCCUPANCY_WARNINGS"] < detection_index
    assert module_assignments["FAST_MATH"] < detection_index

    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "numba"
        and any(a.name == "cuda" for a in n.names)
        for n in tree.body
    )
    for module in (trajectory_kernels, field_kernels):
        assert "from numba import cuda" not in Path(module.__file__).read_text()

    facade_tree = ast.parse(Path(cuda_kernels.__file__).read_text())
    owner_modules = {
        "_runtime",
        "memory",
        "trajectory_kernels",
        "field_kernels",
        "registry",
    }
    owner_imports = []
    for index, statement in enumerate(facade_tree.body):
        if not isinstance(statement, ast.ImportFrom) or statement.level != 1:
            continue
        imported = (
            {alias.name for alias in statement.names}
            if statement.module is None
            else {statement.module.split(".", 1)[0]}
        )
        for module_name in imported & owner_modules:
            owner_imports.append((index, module_name))
    assert owner_imports
    runtime_import_index = min(
        index for index, module_name in owner_imports if module_name == "_runtime"
    )
    assert all(
        runtime_import_index < index
        for index, module_name in owner_imports
        if module_name != "_runtime"
    )


def test_cpu_mirror_surface(capsys):
    messages = {
        "trajectory_kernel": "CUDA trajectory kernel not available",
        "trajectory_kernel_vectorized": "CUDA vectorized trajectory kernel not available",
        "trajectory_kernel_memory_optimized": "CUDA memory-optimized trajectory kernel not available",
        "trajectory_kernel_warp_optimized": "CUDA warp-optimized trajectory kernel not available",
        "trajectory_kernel_cache_friendly": "CUDA cache-friendly trajectory kernel not available",
        "inverse_dynamics_kernel": "CUDA inverse dynamics kernel not available",
        "forward_dynamics_kernel": "CUDA forward dynamics kernel not available",
        "cartesian_trajectory_kernel": "CUDA Cartesian trajectory kernel not available",
        "fused_potential_gradient_kernel": "CUDA potential field kernel not available",
        "batch_trajectory_kernel": "CUDA batch trajectory kernel not available",
    }
    for name, message in messages.items():
        with pytest.raises(RuntimeError, match=message):
            getattr(cuda_kernels, name)()
    assert cuda_kernels.get_memory_pool_stats() == {}
    assert cuda_kernels.get_optimal_kernel_config() is None
    for function in (
        cuda_kernels.get_cuda_array,
        cuda_kernels.return_cuda_array,
        cuda_kernels._best_2d_config,
        cuda_kernels.auto_select_optimal_kernel,
        cuda_kernels.benchmark_kernel_performance,
    ):
        parameters = tuple(inspect.signature(function).parameters.values())
        assert tuple(parameter.kind for parameter in parameters) == (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    assert not inspect.signature(cuda_kernels.profile_start).parameters

    for function in (cuda_kernels.get_cuda_array, cuda_kernels.return_cuda_array):
        with pytest.raises(RuntimeError) as exc_info:
            function("ignored", sentinel=True)
        assert exc_info.value.args == ("CUDA memory pool not available",)
    assert cuda_kernels._best_2d_config("ignored", sentinel=True) == (
        (1, 1),
        (1, 1),
    )
    assert cuda_kernels.auto_select_optimal_kernel("ignored", sentinel=True) == "none"
    assert cuda_kernels.profile_start() is None
    assert cuda_kernels.profile_stop() == {}
    assert cuda_kernels.benchmark_kernel_performance("ignored", sentinel=True) is None
    assert capsys.readouterr() == ("CUDA benchmarking not available\n", "")


def test_logger_name_preserved():
    assert cuda_kernels.logger.name == "ManipulaPy.cuda_kernels"
    for module in (_runtime, trajectory_kernels, registry):
        assert module.logger is cuda_kernels.logger


def test_submodules_import_normally():
    for name in (
        "_runtime",
        "memory",
        "trajectory_kernels",
        "field_kernels",
        "registry",
    ):
        module = importlib.import_module(f"ManipulaPy.cuda_kernels.{name}")
        assert getattr(cuda_kernels, name) is module


def test_reload_has_no_duplicate_pool_or_monitor():
    code = """
import importlib
from ManipulaPy import cuda_kernels as c
importlib.reload(c)
assert c._cuda_memory_pool is c.memory._cuda_memory_pool
assert c._perf_monitor is c.registry._perf_monitor
"""
    env = dict(os.environ, NUMBA_DISABLE_CUDA="1", MPLBACKEND="Agg")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
