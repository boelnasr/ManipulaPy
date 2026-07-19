"""Compatibility contracts for the SR9 CUDA package decomposition."""

import ast
import hashlib
import importlib
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
    "batch_trajectory_kernel": "cbfec25a30cd927a170362f000d29eb2b7721ab8768ddc41a5760f19b6ff23c9",
    "cartesian_trajectory_kernel": "427071e518d68a0f8c5ca52d19ce328df305e44018dd80f189342750c82370b2",
    "forward_dynamics_kernel": "be68833e70b4829c1d548839a7f5c2e1b1a47ea9fe4ad89481369f0f57ee4970",
    "fused_potential_gradient_kernel": "37ca0321febd399e05b3e81746c693759c7e8264d8618b5aef2591e000cc02eb",
    "inverse_dynamics_kernel": "8a38a7f22e26c5816cba3effe17a3e3ce51716d613e1bb7dcdf3bd4f14b57d54",
    "matrix_vector_multiply_6x6": "7c2133a072cf001a55def3cfed1a2e7c41fdac5a276f03d8257bd92314ece18a",
    "trajectory_kernel": "ef1e5eab8f054fee45f65a99b1c1603be3bd2e688e96c0ddbd634c5e373d4962",
    "trajectory_kernel_cache_friendly": "bbd8792d7929260d72e0b6dab3d3621214be8828504c7bd77829264864bff975",
    "trajectory_kernel_memory_optimized": "a5b7980844c44ab20280f8ca68f5a0200beb02a90e8b11a20685a614bd705592",
    "trajectory_kernel_vectorized": "69eac7c3fe7c3b6fb99d95065c94cbaae255f84ea111383b15611218505422cf",
    "trajectory_kernel_warp_optimized": "32de4f1018d42d13dbdab99d201b54095d1e721015736f54971932455e51dac1",
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


def test_raw_cuda_kernel_ast_hashes_across_modules():
    actual = {}
    for module in (trajectory_kernels, field_kernels):
        for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and isinstance(d.func.value, ast.Name)
                and d.func.value.id == "cuda"
                and d.func.attr == "jit"
                for d in node.decorator_list
            ):
                actual[node.name] = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode()
                ).hexdigest()
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
    assert source.index("_nb_cfg.CUDA_CACHE_SIZE") < source.index(
        "_detect_cuda_capability()"
    )
    assert source.index("_nb_cfg.CUDA_LOW_OCCUPANCY_WARNINGS") < source.index(
        "_detect_cuda_capability()"
    )
    assert source.index("FAST_MATH =") < source.index("_detect_cuda_capability()")
    tree = ast.parse(source)
    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module == "numba"
        and any(a.name == "cuda" for a in n.names)
        for n in tree.body
    )
    for module in (trajectory_kernels, field_kernels):
        assert "from numba import cuda" not in Path(module.__file__).read_text()


def test_cpu_mirror_surface():
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
    assert cuda_kernels.profile_stop() == {}


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
