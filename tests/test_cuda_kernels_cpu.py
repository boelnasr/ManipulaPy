#!/usr/bin/env python3
"""
CPU-side coverage for cuda_kernels.py.

These tests exercise the NumPy fallback paths and ensure GPU-only
entry points raise clearly when CUDA is unavailable.
"""

import subprocess
import sys

import numpy as np
import pytest

from tests._kernel_digest import cuda_kernel_digests
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


def test_raw_cuda_kernel_sources_are_unchanged() -> None:
    """Task 9 may route around raw kernels but must not modify their bodies.

    Keys are module-qualified, so a kernel added to one module under a name
    already used in another cannot overwrite -- and thereby mask an edit to --
    the earlier entry. See tests/_kernel_digest.py.
    """
    expected = {
        "field_kernels.fused_potential_gradient_kernel": "afebfa8639fd3153a714f009b51fc046ddefe937d90351e77855e76c2fa4022a",  # noqa: E501
        "trajectory_kernels.batch_trajectory_kernel": "d12d4571b9a9a1a8f25a16c1eb415b70e0f4a75e10eecb4e8d156da4267838ac",  # noqa: E501
        "trajectory_kernels.cartesian_trajectory_kernel": "741a048b733cfdeeb6525971c6bc6810388049fd585b876c3cd1c05d434b3b4b",  # noqa: E501
        "trajectory_kernels.forward_dynamics_kernel": "b68c1f0ba7fdf4e4f5ef6fd2d53b58d85e527b18b524ed4158c497ea77cac21b",  # noqa: E501
        "trajectory_kernels.inverse_dynamics_kernel": "2f8a21b2ff128c3cdca9e5b4fe7dc72aadac8a4bd3ecd62162f75123420e6f3e",  # noqa: E501
        "trajectory_kernels.matrix_vector_multiply_6x6": "00086faf13936989888986e40d576514fe8e3856d557892001951199d34b688e",  # noqa: E501
        "trajectory_kernels.trajectory_kernel": "df1e6b8d4ca2c7490f4d67e497ba59dc41d5aac195c7c4a81066a20c2e2c164c",  # noqa: E501
        "trajectory_kernels.trajectory_kernel_cache_friendly": "9302986ee8dc5596264e5a6665758ff4215cc436dcd4b328bdbb3b620cf8ca95",  # noqa: E501
        "trajectory_kernels.trajectory_kernel_memory_optimized": "64f969d5bde8546939ef82ae8e87650aef34d9a219f31de5291d418eab24cd97",  # noqa: E501
        "trajectory_kernels.trajectory_kernel_vectorized": "9ab7ffaf7fb7791c11a0b1319f95e31bac2b8892191f9d9b6da3c1fcae73bd52",  # noqa: E501
        "trajectory_kernels.trajectory_kernel_warp_optimized": "341c79806193337a848dcd08ddf3c5d44f6ed71acb92acc501ac9566fa76f1df",  # noqa: E501
    }
    actual = {}
    for module in (cuda_kernels.trajectory_kernels, cuda_kernels.field_kernels):
        actual.update(cuda_kernel_digests(module))
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
