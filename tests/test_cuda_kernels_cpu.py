#!/usr/bin/env python3
"""
CPU-side coverage for cuda_kernels.py.

These tests exercise the NumPy fallback paths and ensure GPU-only
entry points raise clearly when CUDA is unavailable.
"""

import ast
import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from ManipulaPy import cuda_kernels
import ManipulaPy.backend as backend_dispatch
from ManipulaPy.backend.numpy_backend import NumpyBackend
from ManipulaPy.cuda_kernels import field_kernels, registry, trajectory_kernels
from ManipulaPy.cuda_kernels import (
    auto_select_optimal_kernel,
    check_cuda_availability,
    get_optimal_kernel_config,
    optimized_batch_trajectory_generation,
    optimized_potential_field,
    optimized_trajectory_generation,
    trajectory_cpu_fallback,
)


class _GpuCapableBackend(NumpyBackend):
    """NumPy test double representing an active GPU array backend."""

    gpu_capable = True


def test_cuda_routing_predicate_requires_gpu_backend_and_cuda(monkeypatch) -> None:
    """Kernel routing is enabled only when both dispatch conditions hold."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(backend_dispatch, "_active", NumpyBackend())
    assert cuda_kernels._cuda_routing_enabled() is False

    monkeypatch.setattr(backend_dispatch, "_active", _GpuCapableBackend())
    assert cuda_kernels._cuda_routing_enabled() is True

    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", False)
    assert cuda_kernels._cuda_routing_enabled() is False


def test_registry_describes_existing_trajectory_and_field_kernels() -> None:
    """Registry entries retain the raw kernel, launch policy, and CPU path."""
    trajectory = registry.get_registered_kernel("trajectory.standard")
    potential = registry.get_registered_kernel("potential_field.fused")

    assert trajectory.implementation is trajectory_kernels.trajectory_kernel
    assert trajectory.cpu_fallback is trajectory_kernels.trajectory_cpu_fallback
    assert trajectory.metadata == {
        "family": "trajectory",
        "variant": "standard",
        "dimensions": 2,
    }
    assert callable(trajectory.launch_config)

    assert potential.implementation is field_kernels.fused_potential_gradient_kernel
    assert potential.cpu_fallback is field_kernels.potential_field_cpu_fallback
    assert potential.metadata == {
        "family": "potential_field",
        "variant": "fused",
        "dimensions": 1,
    }
    assert callable(potential.launch_config)


def test_registry_rejects_unknown_kernel_name() -> None:
    """A misspelled dispatch name fails before any backend work begins."""
    with pytest.raises(KeyError, match="Unknown CUDA kernel 'trajectory.typo'"):
        registry.get_registered_kernel("trajectory.typo")


def test_registry_rejects_duplicate_kernel_name() -> None:
    """A second registration cannot silently replace a launch contract."""
    isolated = registry.KernelRegistry()
    entry = registry.get_registered_kernel("trajectory.standard")

    isolated.register(entry)
    with pytest.raises(ValueError, match="already registered"):
        isolated.register(entry)


def test_registry_executes_trajectory_cpu_fallback() -> None:
    """Registry dispatch runs the numerical CPU reference without CUDA."""
    start = np.array([0.0, 1.0], dtype=np.float32)
    end = np.array([1.0, -1.0], dtype=np.float32)

    actual = registry.execute_registered_kernel(
        "trajectory.standard", start, end, 2.0, 5, 3
    )
    expected = trajectory_cpu_fallback(start, end, 2.0, 5, 3)

    for result, wanted in zip(actual, expected):
        assert np.array_equal(result, wanted)


def test_registry_executes_fused_potential_field_cpu_fallback() -> None:
    """The fused field operation has an explicit, hand-checked CPU result."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    goal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    obstacles = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)

    potential, gradient = registry.execute_registered_kernel(
        "potential_field.fused", positions, goal, obstacles, 1.0
    )

    assert np.allclose(potential, np.array([1.0, 0.5], dtype=np.float32))
    assert np.allclose(
        gradient,
        np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_potential_field_wrapper_falls_back_for_numpy_backend(monkeypatch) -> None:
    """The existing wrapper reaches the registered CPU field implementation."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(backend_dispatch, "_active", NumpyBackend())
    positions = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

    potential, gradient = optimized_potential_field(
        positions,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        influence_distance=1.0,
        use_pinned=False,
    )

    assert np.array_equal(potential, np.array([0.5], dtype=np.float32))
    assert np.array_equal(gradient, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_direct_wrapper_falls_back_when_numpy_backend_is_active(monkeypatch) -> None:
    """A direct public wrapper call cannot bypass active-backend routing."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(backend_dispatch, "_active", NumpyBackend())

    def _unexpected_cuda(*args, **kwargs):
        raise AssertionError("Numba CUDA implementation must not run under NumPy")

    monkeypatch.setattr(
        cuda_kernels,
        "_optimized_trajectory_generation_monitored_cuda",
        _unexpected_cuda,
    )
    result = cuda_kernels.optimized_trajectory_generation(
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        1.0,
        5,
        3,
    )
    expected = trajectory_cpu_fallback(
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        1.0,
        5,
        3,
    )
    for actual, wanted in zip(result, expected):
        assert np.allclose(actual, wanted)


def test_direct_wrapper_routes_to_cuda_implementation_for_gpu_backend(
    monkeypatch,
) -> None:
    """The public wrapper reaches the existing Numba path through the seam."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(backend_dispatch, "_active", _GpuCapableBackend())
    sentinel = (object(), object(), object())
    calls = []

    def _fake_cuda(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        cuda_kernels,
        "_optimized_trajectory_generation_monitored_cuda",
        _fake_cuda,
    )
    result = cuda_kernels.optimized_trajectory_generation(
        [0.0], [1.0], 1.0, 8, 5, use_pinned=False, kernel_type="standard"
    )
    assert result is sentinel
    assert calls == [
        (
            ([0.0], [1.0], 1.0, 8, 5, False, "standard"),
            {"enable_monitoring": True},
        )
    ]


def test_raw_cuda_kernel_asts_are_unchanged() -> None:
    """Task 9 may route around raw kernels but must not modify their bodies."""
    expected = {
        "batch_trajectory_kernel": "cbfec25a30cd927a170362f000d29eb2b7721ab8768ddc41a5760f19b6ff23c9",  # noqa: E501
        "cartesian_trajectory_kernel": "427071e518d68a0f8c5ca52d19ce328df305e44018dd80f189342750c82370b2",  # noqa: E501
        "forward_dynamics_kernel": "be68833e70b4829c1d548839a7f5c2e1b1a47ea9fe4ad89481369f0f57ee4970",  # noqa: E501
        "fused_potential_gradient_kernel": "37ca0321febd399e05b3e81746c693759c7e8264d8618b5aef2591e000cc02eb",  # noqa: E501
        "inverse_dynamics_kernel": "8a38a7f22e26c5816cba3effe17a3e3ce51716d613e1bb7dcdf3bd4f14b57d54",  # noqa: E501
        "matrix_vector_multiply_6x6": "7c2133a072cf001a55def3cfed1a2e7c41fdac5a276f03d8257bd92314ece18a",  # noqa: E501
        "trajectory_kernel": "ef1e5eab8f054fee45f65a99b1c1603be3bd2e688e96c0ddbd634c5e373d4962",  # noqa: E501
        "trajectory_kernel_cache_friendly": "bbd8792d7929260d72e0b6dab3d3621214be8828504c7bd77829264864bff975",  # noqa: E501
        "trajectory_kernel_memory_optimized": "a5b7980844c44ab20280f8ca68f5a0200beb02a90e8b11a20685a614bd705592",  # noqa: E501
        "trajectory_kernel_vectorized": "69eac7c3fe7c3b6fb99d95065c94cbaae255f84ea111383b15611218505422cf",  # noqa: E501
        "trajectory_kernel_warp_optimized": "32de4f1018d42d13dbdab99d201b54095d1e721015736f54971932455e51dac1",  # noqa: E501
    }
    actual = {}
    for module in (cuda_kernels.trajectory_kernels, cuda_kernels.field_kernels):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
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
            if is_cuda_kernel:
                payload = ast.dump(node, include_attributes=False).encode()
                actual[node.name] = hashlib.sha256(payload).hexdigest()
    assert actual.keys() == expected.keys()
    assert actual == expected


def test_trajectory_cpu_fallback_linear_matches_expected() -> None:
    """Verify linear interpolation, constant velocity, and zero acceleration."""
    thetastart = np.array([0.0, 0.0], dtype=np.float32)
    thetaend = np.array([1.0, -1.0], dtype=np.float32)
    Tf, N, method = 1.0, 5, 1  # linear

    pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    # Positions should interpolate linearly, vel constant, acc zero
    t = np.linspace(0, 1.0, N, dtype=np.float32)[:, None]
    expected_pos = thetastart + t * (thetaend - thetastart)
    expected_vel = np.full_like(expected_pos, (thetaend - thetastart) / Tf)
    expected_acc = np.zeros_like(expected_pos)

    assert pos.shape == (N, thetastart.size)
    assert np.allclose(pos, expected_pos)
    assert np.allclose(vel, expected_vel)
    assert np.allclose(acc, expected_acc)


def test_trajectory_cpu_fallback_quintic_endpoints_exact() -> None:
    """Verify the quintic CPU fallback hits exact endpoints with zero boundary velocity and acceleration."""
    thetastart = np.array([0.5], dtype=np.float32)
    thetaend = np.array([1.5], dtype=np.float32)
    Tf, N, method = 2.0, 11, 5  # quintic

    pos, vel, acc = trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

    assert np.isclose(pos[0, 0], thetastart[0])
    assert np.isclose(pos[-1, 0], thetaend[0])
    # Quintic should start/end with zero velocity/acceleration
    assert np.isclose(vel[0, 0], 0.0, atol=1e-6)
    assert np.isclose(vel[-1, 0], 0.0, atol=1e-6)
    assert np.isclose(acc[0, 0], 0.0, atol=1e-6)
    assert np.isclose(acc[-1, 0], 0.0, atol=1e-6)


def test_optimized_trajectory_generation_uses_cpu_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify optimized_trajectory_generation matches the CPU fallback when CUDA is unavailable."""
    # Force CUDA unavailable
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", False)
    result_pos, result_vel, result_acc = optimized_trajectory_generation(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        1.0,
        4,
        3,
        use_pinned=False,
    )
    cpu_pos, cpu_vel, cpu_acc = trajectory_cpu_fallback(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        1.0,
        4,
        3,
    )
    assert np.allclose(result_pos, cpu_pos)
    assert np.allclose(result_vel, cpu_vel)
    assert np.allclose(result_acc, cpu_acc)


def test_import_never_crashes_on_broken_cuda_driver() -> None:
    """Importing cuda_kernels must not abort the interpreter on a bad driver.

    A mismatched NVIDIA driver can SIGSEGV inside numba's C call during CUDA
    detection. Detection runs in a sacrificial subprocess so the import always
    completes and degrades to CPU. We verify the import succeeds in a child
    process: a segfault would surface here as a non-zero (negative) return code
    rather than crashing the test runner.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ManipulaPy.cuda_kernels as ck; "
            "assert isinstance(ck.CUDA_AVAILABLE, bool); print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        proc.returncode == 0
    ), f"import crashed (rc={proc.returncode}): {proc.stderr[-2000:]}"
    assert "ok" in proc.stdout


@pytest.mark.skipif(
    cuda_kernels.CUDA_AVAILABLE,
    reason=(
        "cuda_kernels.py picks GPU vs mock function bodies at import time. "
        "Monkeypatching CUDA_AVAILABLE=False after a real-CUDA import flips "
        "the boolean but not the dispatched callables, so these tests can "
        "only exercise the no-CUDA branch when the module was imported "
        "without a working GPU."
    ),
)
def test_gpu_only_entrypoints_raise_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GPU-only entrypoints raise RuntimeError when CUDA is unavailable."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", False)
    assert check_cuda_availability() is False

    with pytest.raises(RuntimeError):
        optimized_batch_trajectory_generation(
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
            Tf=1.0,
            N=8,
            method=3,
            use_pinned=False,
        )


@pytest.mark.skipif(
    cuda_kernels.CUDA_AVAILABLE,
    reason="See test_gpu_only_entrypoints_raise_when_no_cuda — same import-time dispatch limitation.",
)
def test_kernel_selection_fallbacks_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify kernel selection helpers report no available kernel when CUDA is unavailable."""
    monkeypatch.setattr(cuda_kernels, "CUDA_AVAILABLE", False)
    assert auto_select_optimal_kernel(100, 6) == "none"
    assert get_optimal_kernel_config(100, 6) is None
