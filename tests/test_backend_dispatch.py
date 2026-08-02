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
import inspect
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from ManipulaPy import backend as be
from ManipulaPy import utils
from ManipulaPy.backend.numpy_backend import NumpyBackend
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.potential_field import CollisionChecker, PotentialField
from ManipulaPy.potential_field import fields as potential_field_fields
import ManipulaPy.planning.trajectory_planning as traj_impl
from ManipulaPy.planning.trajectory_planning import OptimizedTrajectoryPlanning
from ManipulaPy.singularity import Singularity


def _real_torch_available() -> bool:
    """True only when the *real* PyTorch package is importable.

    ``importlib.util.find_spec("torch")`` cannot be trusted here: this suite's
    conftest installs a lightweight ``torch`` stand-in in ``sys.modules`` when
    PyTorch is absent (so unrelated modules stay importable), and that stand-in
    makes ``find_spec`` report availability on a base install -- which would run
    the Torch-only tests against the mock and fail instead of skipping. The
    stand-in is a plain object rather than a real module, so an import plus a
    module-type check distinguishes it from genuine PyTorch, matching the
    availability signal conftest computes for its own markers.
    """
    try:
        import torch
    except Exception:
        return False
    return isinstance(torch, types.ModuleType)


_HAS_TORCH = _real_torch_available()
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed")


def _real_jax_available() -> bool:
    """True only when the *real* JAX package is importable.

    Written as an import plus a module-type check for the same reason as
    :func:`_real_torch_available`: conftest installs stand-ins for absent
    optional dependencies, and a stand-in would make ``find_spec`` report
    availability and run the JAX-only tests against the mock instead of
    skipping. JAX is not in conftest's stand-in list today, so the import alone
    decides; the module-type check keeps this honest if it ever is added.
    """
    try:
        import jax
    except Exception:
        return False
    return isinstance(jax, types.ModuleType)


_HAS_JAX = _real_jax_available()
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX is not installed")


# The full protocol surface, mirrored from the call-site audit. Kept here so
# the completeness test fails loudly if base.py and the impls drift apart.
CONSTRUCTION = ["array", "asarray", "zeros", "eye", "stack", "concatenate", "diag"]
LINALG = ["svd", "svdvals", "inv", "pinv", "solve", "norm", "trace"]
ELEMENTWISE = [
    "sin",
    "cos",
    "sqrt",
    "arccos",
    "arctan2",
    "abs",
    "clip",
    "maximum",
    "minimum",
    "where",
    "cross",
    "matmul",
]
REDUCTIONS = ["sum", "amax", "amin", "mean", "argmax", "all", "any", "isfinite"]
DEVICE = ["to_device", "to_numpy", "ascontiguous"]
DTYPES = ["float32", "float64"]
PREDICATE = ["is_backend_array", "is_concrete", "cache_token", "gpu_capable"]
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


# ---------------------------------------------------------------------------
# Torch backend (lazily registered; skipped when PyTorch is absent)
# ---------------------------------------------------------------------------


@requires_torch
def test_torch_selection():
    """Torch selects and round-trips values when PyTorch is present."""
    be.set_backend("torch")
    backend = be.get_backend()
    result = backend.to_numpy(backend.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_torch_registration_raises_when_torch_absent():
    """set_backend('torch') raises an actionable ImportError when PyTorch is not
    installed. ``find_spec`` is forced to report torch absent and the registry is
    isolated from any prior torch registration, so the error path is exercised
    deterministically regardless of whether PyTorch is installed here."""
    real_find_spec = importlib.util.find_spec

    def _torch_absent(name, *args, **kwargs):
        if name == "torch":
            return None
        return real_find_spec(name, *args, **kwargs)

    registry_without_torch = {k: v for k, v in be._REGISTRY.items() if k != "torch"}
    with patch.object(be, "_REGISTRY", registry_without_torch), patch(
        "importlib.util.find_spec", side_effect=_torch_absent
    ):
        with pytest.raises(ImportError) as exc:
            be.set_backend("torch")
    assert "torch" in str(exc.value).lower()


@requires_torch
@pytest.mark.parametrize("switch", ["set_backend", "use_backend"])
def test_torch_round_trip(switch):
    """to_numpy(array([...])) reproduces the source values under torch, via both
    set_backend and the use_backend context manager."""
    source = [[1.0, 2.0], [3.0, 4.0]]
    if switch == "set_backend":
        be.set_backend("torch")
        backend = be.get_backend()
        result = backend.to_numpy(backend.array(source))
    else:
        with be.use_backend("torch") as backend:
            result = backend.to_numpy(backend.array(source))
    np.testing.assert_allclose(result, np.array(source), rtol=1e-12, atol=1e-12)


@requires_torch
def test_torch_backend_flags_gate_cache_and_gpu_routing():
    """Torch reports non-concrete (cache bypass) and CPU-default (no GPU routing)."""
    backend = be.get_registered("torch")
    assert backend.is_concrete is False
    assert backend.gpu_capable is False


@requires_torch
def test_torch_default_float_dtype_matches_numpy():
    """Python-float construction forces float64 on CPU to match NumPy numerics,
    while explicit dtype handles are honoured."""
    backend = be.get_registered("torch")
    # Python floats default to float64 (not torch's float32 default).
    assert backend.to_numpy(backend.array([1.0, 2.0])).dtype == np.float64
    assert backend.to_numpy(backend.zeros((2, 2))).dtype == np.float64
    assert backend.to_numpy(backend.eye(3)).dtype == np.float64
    # Integer input keeps integer dtype (matching NumPy).
    assert backend.to_numpy(backend.array([1, 2, 3])).dtype == np.int64
    # Explicit dtype handles round-trip.
    assert backend.to_numpy(backend.array([1, 2, 3], dtype=backend.float32)).dtype == (
        np.float32
    )
    assert backend.to_numpy(backend.array([1, 2, 3], dtype=backend.float64)).dtype == (
        np.float64
    )


@requires_torch
def test_torch_integer_input_matches_numpy_semantics():
    """Integer input to float-producing ops promotes to float64 like NumPy.

    NumPy returns float64 for integer input to the transcendental, ``mean``, and
    linalg operations; Torch would otherwise raise (``mean``/``norm``/linalg) or
    silently return float32 (``sin``/``cos``/``sqrt``/...). Preserving ops
    (``sum``/``abs``/``trace``) must keep the integer dtype.
    """
    backend = be.get_registered("torch")
    vec = [1, 2]
    mat = [[4, 1], [1, 3]]

    # Elementwise transcendental: float64 values matching NumPy.
    for name in ("sin", "cos", "sqrt", "arccos"):
        out = getattr(backend, name)(backend.array([0, 1]))
        assert backend.to_numpy(out).dtype == np.float64, name
        np.testing.assert_allclose(
            backend.to_numpy(out), getattr(np, name)(np.array([0, 1])), rtol=1e-7
        )
    at = backend.arctan2(backend.array(vec), backend.array(vec))
    assert backend.to_numpy(at).dtype == np.float64
    np.testing.assert_allclose(
        backend.to_numpy(at), np.arctan2(np.array(vec), np.array(vec)), rtol=1e-7
    )

    # Reductions / linalg that NumPy floats: value + float64 parity.
    mean_out = backend.mean(backend.array(vec))
    assert backend.to_numpy(mean_out).dtype == np.float64
    assert np.isclose(float(mean_out), np.mean(np.array(vec)))
    norm_out = backend.norm(backend.array(vec))
    assert backend.to_numpy(norm_out).dtype == np.float64
    assert np.isclose(float(norm_out), np.linalg.norm(np.array(vec)))
    for name, npfn in (
        ("inv", np.linalg.inv),
        ("pinv", np.linalg.pinv),
        ("svdvals", lambda m: np.linalg.svd(m, compute_uv=False)),
    ):
        out = getattr(backend, name)(backend.array(mat))
        assert backend.to_numpy(out).dtype == np.float64, name
        np.testing.assert_allclose(
            backend.to_numpy(out), npfn(np.array(mat)), rtol=1e-7
        )
    solve_out = backend.solve(backend.array(mat), backend.array(vec))
    assert backend.to_numpy(solve_out).dtype == np.float64
    np.testing.assert_allclose(
        backend.to_numpy(solve_out),
        np.linalg.solve(np.array(mat), np.array(vec)),
        rtol=1e-7,
    )

    # Preserving ops keep integer dtype, matching NumPy.
    assert backend.to_numpy(backend.sum(backend.array(vec))).dtype == np.int64
    assert backend.to_numpy(backend.abs(backend.array([-1, 2]))).dtype == np.int64
    assert backend.to_numpy(backend.trace(backend.array(mat))).dtype == np.int64


@requires_torch
def test_torch_linalg_matches_numpy():
    """The linalg surface (svd/pinv/inv/solve/norm/trace) matches NumPy values."""
    backend = be.get_registered("torch")
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_vec = np.array([1.0, 2.0])

    ta = backend.array(a)
    # svd: shapes and reconstruction.
    u, s, vt = backend.svd(ta)
    assert tuple(u.shape) == (2, 2) and tuple(s.shape) == (2,)
    np.testing.assert_allclose(
        backend.to_numpy(s), np.linalg.svd(a, compute_uv=False), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.svdvals(ta)), s.detach().numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.inv(ta)), np.linalg.inv(a), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.pinv(ta)), np.linalg.pinv(a), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.solve(ta, backend.array(b_vec))),
        np.linalg.solve(a, b_vec),
        rtol=1e-10,
    )
    assert np.isclose(float(backend.norm(backend.array([3.0, 4.0]))), 5.0)
    assert np.isclose(float(backend.trace(backend.eye(3))), 3.0)


@requires_torch
def test_torch_use_backend_restores_previous_backend():
    """use_backend('torch') restores the previously active backend on exit."""
    be.set_backend("numpy")
    original = be.get_backend()
    with be.use_backend("torch") as backend:
        assert backend.is_backend_array(backend.array([1.0, 2.0]))
        assert be.get_backend() is not original
    assert be.get_backend() is original


@requires_torch
def test_torch_use_backend_restores_on_exception():
    """The previous backend is restored even when the torch-scoped body raises."""
    be.set_backend("numpy")
    original = be.get_backend()
    with pytest.raises(ValueError):
        with be.use_backend("torch"):
            raise ValueError("boom")
    assert be.get_backend() is original


@requires_torch
def test_torch_preserves_autograd_graph():
    """Backend ops on a grad-tracking tensor keep the graph attached, so the
    non-concrete cache-bypass contract can protect gradients (Task 12)."""
    import torch

    theta = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    backend = be.get_registered("torch")
    out = backend.sum(backend.sin(backend.matmul(backend.eye(3), theta)))
    assert out.requires_grad
    out.backward()
    assert theta.grad is not None
    np.testing.assert_allclose(
        theta.grad.detach().numpy(), np.cos(theta.detach().numpy()), rtol=1e-10
    )


@requires_torch
def test_torch_exotic_scalar_reduction_keeps_0d_shape():
    """A NumPy-fallback scalar reduction on an exotic dtype returns a 0-D
    result like NumPy, not shape ``(1,)`` (the fallback must not upgrade a
    0-D scalar via ``np.ascontiguousarray``)."""
    backend = be.get_registered("torch")
    npb = be.get_registered("numpy")
    for arr in (np.array([True, False, True]), np.array([[1, 2], [3, 4]], np.uint8)):
        got = backend.sum(backend.asarray(arr))
        exp = npb.sum(npb.asarray(arr))
        assert tuple(got.shape) == np.shape(exp) == ()
        assert backend.to_numpy(got).item() == np.asarray(exp).item()


@requires_torch
def test_torch_int_index_ops_stay_native_and_trace_safe():
    """In-range Python int scalars paired with int64 index arrays must stay on
    the native Torch path (not the value-baking NumPy fallback), so realistic
    index ops like ``maximum(idx, 0)``/``clip(idx, 0, limit)`` remain trace-safe;
    an actually out-of-range scalar still matches NumPy's overflow raise."""
    import torch

    backend = be.get_registered("torch")
    npb = be.get_registered("numpy")

    # Each op must trace WITHOUT baking in the sample values (the NumPy fallback
    # would freeze them), so replaying on new data recomputes correctly.
    traced_max = torch.jit.trace(
        lambda x: backend.maximum(x, 0), torch.tensor([-1, 1, 3]), check_trace=True
    )
    assert traced_max(torch.tensor([-4, 5, 1])).tolist() == [0, 5, 1]
    traced_clip = torch.jit.trace(
        lambda x: backend.clip(x, 0, 7), torch.tensor([-1, 5, 9]), check_trace=True
    )
    assert traced_clip(torch.tensor([-4, 6, 1])).tolist() == [0, 6, 1]
    traced_where = torch.jit.trace(
        lambda x: backend.where(x > 0, x, 0), torch.tensor([-1, 2, -3]), check_trace=True
    )
    assert traced_where(torch.tensor([4, -5, 6])).tolist() == [4, 0, 6]

    # Out-of-range weak int still matches NumPy's overflow behavior.
    int8 = np.array([1, 2], dtype=np.int8)
    with pytest.raises(OverflowError):
        npb.maximum(npb.asarray(int8), 128)
    with pytest.raises(OverflowError):
        backend.maximum(backend.asarray(int8), 128)


@requires_torch
def test_torch_arctan2_float_weak_scalar_stays_native_and_grads():
    """arctan2 of a grad-tracking float tensor and a weak Python scalar stays
    on the native Torch path (autograd preserved) and matches NumPy in value."""
    import torch

    backend = be.get_registered("torch")
    npb = be.get_registered("numpy")
    y = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    out = backend.arctan2(y, 1)
    assert out.requires_grad
    out.sum().backward()
    assert y.grad is not None
    np.testing.assert_allclose(
        out.detach().numpy(),
        npb.arctan2(npb.asarray([1.0, 2.0]), 1),
        rtol=1e-12,
    )


@requires_torch
def test_torch_tensor_sequence_stack_on_device_preserves_autograd():
    """Building a matrix from a tensor leaf + Python scalar stacks on the
    backend device and keeps autograd attached (Finding 1).

    On a CUDA-device backend the CUDA tensor leaf and the CPU-built Python
    scalar leaf must be co-located before ``torch.stack`` or Torch raises a
    mixed-device error. Requires a CUDA-capable install; skips otherwise.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-capable PyTorch is not available")
    from ManipulaPy.backend.torch_backend import TorchBackend

    backend = TorchBackend(device="cuda")
    theta = torch.tensor(
        0.3, dtype=torch.float64, device="cuda", requires_grad=True
    )
    mat = backend.array([[torch.cos(theta), 0.0], [0.0, torch.sin(theta)]])
    assert mat.device.type == "cuda"
    assert mat.requires_grad
    mat.sum().backward()
    assert theta.grad is not None


@requires_torch
def test_torch_negative_stride_ndarray_input():
    """A negative-stride ndarray constructs without error and preserves values."""
    backend = be.get_registered("torch")
    arr = np.arange(6.0)[::-1]
    out = backend.array(arr)
    np.testing.assert_allclose(backend.to_numpy(out), arr)


@requires_torch
def test_torch_clip_matches_numpy_dtype_and_bounds():
    """clip mirrors np.clip dtype and accepts list/tuple bounds (Finding 2)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()

    # Integer input with scalar float bounds promotes to float64 like NumPy.
    out = backend.clip(backend.array([0, 2, 4]), 1.5, 3.5)
    ref = numpy_be.clip(np.array([0, 2, 4]), 1.5, 3.5)
    assert backend.to_numpy(out).dtype == ref.dtype == np.float64
    np.testing.assert_allclose(backend.to_numpy(out), ref)

    # Integer input with integer bounds stays integer.
    out_i = backend.clip(backend.array([0, 2, 4]), 1, 3)
    assert backend.to_numpy(out_i).dtype == np.int64

    # Float input keeps its dtype.
    out_f = backend.clip(backend.array([0.0, 2.0, 4.0]), 1.5, 3.5)
    assert backend.to_numpy(out_f).dtype == np.float64

    # List/tuple bounds are accepted like np.clip.
    x = np.array([0.0, 5.0, 2.0])
    out_l = backend.clip(backend.array(x), [1.0, 1.0, 1.0], (3.0, 3.0, 3.0))
    np.testing.assert_allclose(
        backend.to_numpy(out_l), np.clip(x, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0])
    )


@requires_torch
def test_torch_where_accepts_numeric_mask():
    """where accepts a numeric (non-bool) condition like np.where."""
    backend = be.get_registered("torch")
    mask = backend.array([1, 0, 1])
    out = backend.where(
        mask, backend.array([1.0, 2.0, 3.0]), backend.array([-1.0, -2.0, -3.0])
    )
    np.testing.assert_allclose(
        backend.to_numpy(out),
        np.where(np.array([1, 0, 1]), [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]),
    )


@requires_torch
def test_torch_all_any_return_bool():
    """all/any return a NumPy-bool result even for uint8 input (Finding 3)."""
    backend = be.get_registered("torch")
    for data in ([1, 1, 0], [1, 1, 1]):
        mask = backend.array(np.array(data, dtype=np.uint8))
        a = backend.all(mask)
        assert backend.to_numpy(a).dtype == np.bool_
        assert bool(a) == bool(np.all(np.array(data)))
        o = backend.any(mask)
        assert backend.to_numpy(o).dtype == np.bool_
        assert bool(o) == bool(np.any(np.array(data)))


@requires_torch
def test_torch_to_numpy_detaches_grad_tensor():
    """to_numpy detaches and moves to CPU so a grad tensor round-trips."""
    import torch

    backend = be.get_registered("torch")
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    out = backend.to_numpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@requires_torch
def test_torch_trace_batched_matches_numpy():
    """trace handles batched (>2D) input like np.trace (Finding 4)."""
    backend = be.get_registered("torch")
    batched = np.arange(2 * 3 * 3).reshape(2, 3, 3).astype(float)
    np.testing.assert_allclose(
        backend.to_numpy(backend.trace(backend.array(batched))),
        np.trace(batched),
    )
    # The 2D fast path still returns the scalar trace, dtype preserved.
    mat = np.array([[4, 1], [1, 3]])
    assert np.isclose(float(backend.trace(backend.array(mat))), np.trace(mat))
    assert backend.to_numpy(backend.trace(backend.array(mat))).dtype == np.int64


@requires_torch
def test_torch_cross_two_component_matches_numpy():
    """cross accepts 2-component vectors (scalar z-cross) like np.cross
    while keeping the 3-vector fast path (Finding 4)."""
    backend = be.get_registered("torch")
    a2, b2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    np.testing.assert_allclose(
        backend.to_numpy(backend.cross(backend.array(a2), backend.array(b2))),
        np.cross(a2, b2),
    )
    a3, b3 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        backend.to_numpy(backend.cross(backend.array(a3), backend.array(b3))),
        np.cross(a3, b3),
    )


# ---------------------------------------------------------------------------
# Full NumPy dtype-promotion parity (compared against the NumPy backend)
# ---------------------------------------------------------------------------


@requires_torch
def test_torch_clip_dtype_matrix_matches_numpy():
    """clip promotes by np.result_type(x, *bounds) across the dtype matrix."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    cases = [
        # int input + numpy-float32 bounds -> float32 (not float64).
        (np.array([0, 2, 4], dtype=np.int8), np.float32(1.5), np.float32(3.5)),
        # int64 + python-float bounds -> float64.
        (np.array([0, 2, 4], dtype=np.int64), 1.5, 3.5),
        # float32 input + numpy-float64 bound -> float64.
        (np.array([0.0, 2.0, 4.0], dtype=np.float32), np.float64(1.5), None),
        # all-integer bounds keep the integer dtype.
        (np.array([0, 2, 4], dtype=np.int8), 1, 3),
        (np.array([0, 2, 4], dtype=np.int32), None, 3),
        # array bounds of differing precision promote jointly.
        (
            np.array([0.0, 5.0, 2.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            (3.0, 3.0, 3.0),
        ),
    ]
    for x, lo, hi in cases:
        ref = numpy_be.clip(x, lo, hi)
        out = backend.clip(backend.array(x), lo, hi)
        assert backend.to_numpy(out).dtype == ref.dtype, (x.dtype, lo, hi)
        np.testing.assert_allclose(backend.to_numpy(out), ref)


@requires_torch
def test_torch_trace_dtype_matches_numpy():
    """trace mirrors np.trace's accumulator upcast for every dtype it takes."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    for dt in (np.bool_, np.uint8, np.int32, np.int64, np.float32, np.float64):
        m = (np.eye(3) * np.arange(1, 4)).astype(dt)
        ref = numpy_be.trace(m)
        out = backend.trace(backend.array(m))
        assert backend.to_numpy(out).dtype == ref.dtype, np.dtype(dt).name
        np.testing.assert_array_equal(
            backend.to_numpy(out).astype(np.float64),
            np.asarray(ref).astype(np.float64),
        )
    # Batched (>2D) input follows np.trace (axis1=0, axis2=1) with its dtype.
    batched = np.arange(2 * 3 * 3).reshape(2, 3, 3).astype(np.int32)
    ref_b = numpy_be.trace(batched)
    out_b = backend.trace(backend.array(batched))
    assert backend.to_numpy(out_b).dtype == ref_b.dtype
    np.testing.assert_array_equal(backend.to_numpy(out_b), ref_b)


@requires_torch
def test_torch_cross_shapes_and_dtype_match_numpy():
    """cross matches np.cross value/shape/dtype for 2x2, 3x3, 2x3, 3x2 and
    heterogeneous operand dtypes."""
    import warnings

    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    cases = [
        (np.array([1.0, 2.0]), np.array([3.0, 4.0])),  # 2x2 -> scalar z
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # 3x3
        (np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])),  # 2x3
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),  # 3x2
        (
            np.array([1, 2], dtype=np.int32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        ),  # 2x3 mixed dtype
        (
            np.array([1, 0, 0], dtype=np.int64),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ),  # 3x3 mixed dtype
    ]
    for a, b in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = numpy_be.cross(a, b)
        out = backend.cross(backend.array(a), backend.array(b))
        got = backend.to_numpy(out)
        assert got.shape == np.asarray(ref).shape, (a.dtype, b.dtype)
        assert got.dtype == np.asarray(ref).dtype, (a.dtype, b.dtype)
        np.testing.assert_allclose(got, ref)


@requires_torch
def test_torch_binary_ops_promote_like_numpy():
    """solve/matmul/maximum/minimum promote both operands via np.result_type."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    a32 = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    i64 = np.array([1, 2], dtype=np.int64)
    v32 = np.array([1.0, 2.0], dtype=np.float32)

    ref_solve = numpy_be.solve(a32, i64)
    out_solve = backend.solve(backend.array(a32), backend.array(i64))
    assert backend.to_numpy(out_solve).dtype == ref_solve.dtype
    np.testing.assert_allclose(backend.to_numpy(out_solve), ref_solve, rtol=1e-6)

    ref_mm = numpy_be.matmul(a32, i64)
    out_mm = backend.matmul(backend.array(a32), backend.array(i64))
    assert backend.to_numpy(out_mm).dtype == ref_mm.dtype
    np.testing.assert_allclose(backend.to_numpy(out_mm), ref_mm, rtol=1e-6)

    for name in ("maximum", "minimum"):
        ref = getattr(numpy_be, name)(i64, v32)
        out = getattr(backend, name)(backend.array(i64), backend.array(v32))
        assert backend.to_numpy(out).dtype == ref.dtype, name
        np.testing.assert_allclose(backend.to_numpy(out), ref)


@requires_torch
def test_torch_reductions_empty_axis_match_numpy():
    """axis=() is no-reduction for all six reductions, matching np dtype rules."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    xi8 = np.array([1, 2, 3], dtype=np.int8)
    xf32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    xu8 = np.array([1, 0, 1], dtype=np.uint8)
    checks = [
        ("sum", xi8),
        ("amax", xi8),
        ("amin", xi8),
        ("mean", xi8),
        ("mean", xf32),
        ("all", xu8),
        ("any", xu8),
    ]
    for name, x in checks:
        ref = getattr(numpy_be, name)(x, axis=())
        out = getattr(backend, name)(backend.array(x), axis=())
        assert backend.to_numpy(out).dtype == ref.dtype, (name, x.dtype)
        np.testing.assert_array_equal(backend.to_numpy(out), ref)


@requires_torch
def test_torch_argmax_accepts_bool_like_numpy():
    """argmax accepts a boolean mask; torch.argmax raises on Bool (Finding 1)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    mask = np.array([False, False, True, True])
    ref = numpy_be.argmax(mask)
    out = backend.argmax(backend.array(mask))
    assert int(backend.to_numpy(out)) == int(ref)
    # 2D boolean input with an axis mirrors np.argmax(axis=...).
    m = np.array([[False, True], [True, False]])
    ref2 = numpy_be.argmax(m, axis=1)
    out2 = backend.argmax(backend.array(m), axis=1)
    np.testing.assert_array_equal(backend.to_numpy(out2), ref2)
    # Non-bool dtypes torch already accepts stay correct.
    xi = np.array([1, 5, 2], dtype=np.int32)
    assert int(backend.to_numpy(backend.argmax(backend.array(xi)))) == int(
        numpy_be.argmax(xi)
    )


@requires_torch
def test_torch_metrics_rise_time_runs_under_torch_backend():
    """The control.metrics rise-time path (argmax on a bool mask) runs on torch."""
    from ManipulaPy.control.metrics import _MetricsConcern

    time = np.linspace(0.0, 1.0, 11)
    response = np.linspace(0.0, 1.0, 11)
    set_point = 1.0
    with be.use_backend("numpy"):
        ref = _MetricsConcern.calculate_rise_time(
            _MetricsConcern(), time, response, set_point
        )
    with be.use_backend("torch"):
        out = _MetricsConcern.calculate_rise_time(
            _MetricsConcern(), time, response, set_point
        )
    assert np.isclose(out, ref)


@requires_torch
def test_torch_pinv_near_singular_matches_numpy():
    """pinv zeroes tiny singular values like np.linalg.pinv's rcond (Finding 2)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    # diag([1, 9e-16]): NumPy zeroes the tiny sv; torch's default cutoff inverts
    # it to ~1e15. The tiny entry must map to ~0, not ~1e15.
    d = np.diag([1.0, 9e-16])
    ref = numpy_be.pinv(d)
    out = backend.to_numpy(backend.pinv(backend.array(d)))
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert abs(out[1, 1]) < 1.0
    # A realistic near-singular Jacobian (one collapsing singular value).
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1e-16, 0.0]])
    ref_j = numpy_be.pinv(J)
    out_j = backend.to_numpy(backend.pinv(backend.array(J)))
    np.testing.assert_allclose(out_j, ref_j, atol=1e-6)


@requires_torch
def test_torch_clip_mixed_array_scalar_bounds_match_numpy():
    """clip accepts one array bound and one scalar bound like np.clip (Finding 3)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    x = np.array([0.0, 5.0, 2.0], dtype=np.float32)
    amin = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    amax = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    # Array min + scalar max.
    ref1 = numpy_be.clip(x, amin, 3.0)
    out1 = backend.clip(backend.array(x), backend.array(amin), 3.0)
    assert backend.to_numpy(out1).dtype == ref1.dtype
    np.testing.assert_allclose(backend.to_numpy(out1), ref1)
    # Scalar min + array max.
    ref2 = numpy_be.clip(x, 1.0, amax)
    out2 = backend.clip(backend.array(x), 1.0, backend.array(amax))
    assert backend.to_numpy(out2).dtype == ref2.dtype
    np.testing.assert_allclose(backend.to_numpy(out2), ref2)


@requires_torch
def test_torch_maximum_minimum_weak_scalar_dtype_matches_numpy():
    """maximum/minimum keep NEP-50 weak-scalar dtype: float32 array + python
    float stays float32; array+array promotes strongly (Finding 4)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    xf32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    i64 = np.array([1, 2, 3], dtype=np.int64)
    for name in ("maximum", "minimum"):
        ref = getattr(numpy_be, name)(xf32, 1.5)
        out = getattr(backend, name)(backend.array(xf32), 1.5)
        assert backend.to_numpy(out).dtype == ref.dtype == np.float32, name
        np.testing.assert_allclose(backend.to_numpy(out), ref)
        # Strong (array+array) promotion still follows np.result_type.
        ref2 = getattr(numpy_be, name)(i64, xf32)
        out2 = getattr(backend, name)(backend.array(i64), backend.array(xf32))
        assert backend.to_numpy(out2).dtype == ref2.dtype, name
        np.testing.assert_allclose(backend.to_numpy(out2), ref2)


@requires_torch
def test_torch_sum_unsigned_accumulator_matches_numpy_all_axes():
    """sum applies NumPy's accumulator upcast (uint -> uint64) on every axis
    path, not only axis=() (Finding 5)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    cases = [
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([1, 2, 3], dtype=np.uint16),
        np.array([1, 2, 3], dtype=np.uint32),
        np.array([1, 2, 3], dtype=np.int8),
        np.array([1, 0, 1], dtype=np.bool_),
    ]
    for x in cases:
        for axis in (None, 0, ()):
            ref = numpy_be.sum(x, axis=axis)
            out = backend.sum(backend.array(x), axis=axis)
            assert backend.to_numpy(out).dtype == np.asarray(ref).dtype, (
                x.dtype,
                axis,
            )
            np.testing.assert_array_equal(backend.to_numpy(out), ref)


@requires_torch
def test_torch_stack_concat_where_mixed_dtype_match_numpy():
    """stack/concatenate/where promote combined operands via np.result_type so
    mixed int64/float32 inputs stay float64, not float32 (Finding 6)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    i64 = np.array([1, 2, 3], dtype=np.int64)
    f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cond = np.array([True, False, True])

    ref_s = numpy_be.stack([i64, f32])
    out_s = backend.stack([backend.array(i64), backend.array(f32)])
    assert backend.to_numpy(out_s).dtype == ref_s.dtype
    np.testing.assert_allclose(backend.to_numpy(out_s), ref_s)

    ref_c = numpy_be.concatenate([i64, f32])
    out_c = backend.concatenate([backend.array(i64), backend.array(f32)])
    assert backend.to_numpy(out_c).dtype == ref_c.dtype
    np.testing.assert_allclose(backend.to_numpy(out_c), ref_c)

    ref_w = numpy_be.where(cond, i64, f32)
    out_w = backend.where(
        backend.array(cond), backend.array(i64), backend.array(f32)
    )
    assert backend.to_numpy(out_w).dtype == ref_w.dtype
    np.testing.assert_allclose(backend.to_numpy(out_w), ref_w)


# ---------------------------------------------------------------------------
# Exhaustive NumPy dtype-parity matrix (Torch backend vs NumPy backend)
#
# The Torch backend keeps a native path for the float / int64 autograd dtypes
# and delegates the exotic dtypes (bool, uint8/16/32/64, complex64/128) and
# value-based integer-overflow cases to NumPy, so parity holds by construction.
# ---------------------------------------------------------------------------

_MATRIX_DTYPES = [
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


def _assert_op_parity(backend, numpy_be, opname, np_args, be_args, **kwargs):
    """Assert backend.<op> matches numpy_be.<op> in dtype+value, or both raise.

    NumPy is the reference: when it raises (e.g. int8 overflow, arctan2 on
    complex) the Torch backend must raise the same exception type; otherwise
    the results must agree in dtype and value.
    """
    import warnings

    ref_exc = out_exc = None
    ref = out = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ref = getattr(numpy_be, opname)(*np_args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - parity check on the type
            ref_exc = exc
        try:
            out = getattr(backend, opname)(*be_args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - parity check on the type
            out_exc = exc
    if ref_exc is not None:
        assert out_exc is not None, f"{opname}: numpy raised {ref_exc!r}, torch did not"
        assert type(out_exc) is type(ref_exc), (
            opname,
            type(ref_exc),
            type(out_exc),
        )
        return
    assert out_exc is None, f"{opname}: torch raised {out_exc!r}, numpy did not"
    got = backend.to_numpy(out)
    ref_arr = np.asarray(ref)
    assert got.dtype == ref_arr.dtype, (opname, ref_arr.dtype, got.dtype)
    np.testing.assert_allclose(
        got.astype(np.complex128), ref_arr.astype(np.complex128), rtol=1e-5, atol=1e-6
    )


@requires_torch
@pytest.mark.parametrize("dt", _MATRIX_DTYPES)
def test_torch_construction_dtype_matrix_matches_numpy(dt):
    """array/asarray/zeros/eye match NumPy dtype+value for every dtype, including
    the exotic dtypes whose torch construction kernels (eye) would otherwise
    raise (Findings 1, 2)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    src = np.array([[1, 0], [0, 1]], dtype=dt)

    for opname in ("array", "asarray"):
        _assert_op_parity(backend, numpy_be, opname, (src,), (src,))
    _assert_op_parity(backend, numpy_be, "zeros", ((2, 2),), ((2, 2),), dtype=dt)
    _assert_op_parity(backend, numpy_be, "eye", (3,), (3,), dtype=dt)


@requires_torch
@pytest.mark.parametrize("dt", _MATRIX_DTYPES)
def test_torch_op_dtype_matrix_matches_numpy(dt):
    """The reduction / linalg / elementwise / binary surface matches the NumPy
    backend in dtype and value across the full dtype matrix (Finding 1)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    vec = np.array([0, 1, 1], dtype=dt)
    mat = np.array([[1, 1], [1, 1]], dtype=dt)
    a3 = np.array([1, 0, 0], dtype=dt)
    b3 = np.array([0, 1, 0], dtype=dt)

    def be_arr(x):
        return backend.array(x)

    # reductions
    for opname in ("sum", "amax", "amin", "mean", "argmax", "all", "any"):
        _assert_op_parity(backend, numpy_be, opname, (vec,), (be_arr(vec),))
        _assert_op_parity(
            backend, numpy_be, opname, (vec,), (be_arr(vec),), axis=()
        )
    # sum over an explicit axis exercises the accumulator upcast path.
    _assert_op_parity(backend, numpy_be, "sum", (vec,), (be_arr(vec),), axis=0)

    # linalg / construction-adjacent
    _assert_op_parity(backend, numpy_be, "trace", (mat,), (be_arr(mat),))
    _assert_op_parity(backend, numpy_be, "diag", (vec,), (be_arr(vec),))

    # unary elementwise (transcendental promotion is width-dependent in NumPy)
    for opname in ("sin", "cos", "sqrt", "arccos", "abs"):
        _assert_op_parity(backend, numpy_be, opname, (vec,), (be_arr(vec),))

    # binary elementwise
    _assert_op_parity(backend, numpy_be, "matmul", (mat, mat), (be_arr(mat), be_arr(mat)))
    _assert_op_parity(backend, numpy_be, "cross", (a3, b3), (be_arr(a3), be_arr(b3)))
    _assert_op_parity(
        backend, numpy_be, "arctan2", (vec, vec), (be_arr(vec), be_arr(vec))
    )
    for opname in ("maximum", "minimum"):
        _assert_op_parity(
            backend, numpy_be, opname, (vec, vec), (be_arr(vec), be_arr(vec))
        )
    cond = np.array([True, False, True])
    _assert_op_parity(
        backend, numpy_be, "where", (cond, vec, vec), (cond, be_arr(vec), be_arr(vec))
    )
    _assert_op_parity(backend, numpy_be, "clip", (vec, 0, 1), (be_arr(vec), 0, 1))


@requires_torch
def test_torch_dtype_arg_accepts_numpy_dtype_forms():
    """array/asarray/zeros/eye accept np.float32 / 'float32' / np.dtype forms as
    the dtype argument, matching NumpyBackend (Finding 2)."""
    backend = be.get_registered("torch")
    for dtype_arg in (np.float32, "float32", np.dtype("float32")):
        assert backend.to_numpy(backend.array([1, 2, 3], dtype=dtype_arg)).dtype == (
            np.float32
        )
        assert backend.to_numpy(backend.asarray([1, 2, 3], dtype=dtype_arg)).dtype == (
            np.float32
        )
        assert backend.to_numpy(backend.zeros((2,), dtype=dtype_arg)).dtype == (
            np.float32
        )
        assert backend.to_numpy(backend.eye(2, dtype=dtype_arg)).dtype == np.float32
    # An exotic-dtype string still round-trips through the numpy-construction path.
    assert backend.to_numpy(backend.eye(2, dtype="uint16")).dtype == np.uint16


@requires_torch
def test_torch_weak_scalar_sequence_stack_matches_numpy():
    """A tensor-containing sequence promotes with weak Python scalars: a python
    int keeps the tensor's integer dtype and a float32 tensor + int64 tensor
    promote to float64, matching np.array (Finding 3)."""
    import torch

    backend = be.get_registered("torch")
    i64 = torch.tensor(1, dtype=torch.int64)
    f32 = torch.tensor(3.0, dtype=torch.float32)

    # int64 tensor + weak python int -> int64 (not float64).
    out = backend.array([i64, 2])
    assert backend.to_numpy(out).dtype == np.int64
    np.testing.assert_array_equal(
        backend.to_numpy(out), np.array([1, 2], dtype=np.int64)
    )
    # int64 tensor + float32 tensor -> float64 (strong joint promotion).
    out2 = backend.array([i64, f32])
    assert backend.to_numpy(out2).dtype == np.float64
    np.testing.assert_allclose(backend.to_numpy(out2), np.array([1.0, 3.0]))


@requires_torch
def test_torch_weak_scalar_sequence_stack_preserves_autograd():
    """The float trig-matrix sequence path stays native and autograd-attached
    after the weak-scalar promotion change (Finding 3 must not regress autograd)."""
    import torch

    backend = be.get_registered("torch")
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    mat = backend.array([[torch.cos(theta), 0.0], [0.0, torch.sin(theta)]])
    assert mat.dtype == torch.float64
    assert mat.requires_grad
    mat.sum().backward()
    assert theta.grad is not None


@requires_torch
def test_torch_arctan2_joint_promotion_matches_numpy():
    """arctan2 promotes both operands jointly (weak/strong) like np.arctan2:
    float32 array + np.float64 scalar -> float64; np.float32 scalar + python
    float -> float32 (Finding 4)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()

    xf32 = np.array([1.0, 2.0], dtype=np.float32)
    ref1 = numpy_be.arctan2(xf32, np.float64(1.5))
    out1 = backend.arctan2(backend.array(xf32), np.float64(1.5))
    assert backend.to_numpy(out1).dtype == ref1.dtype == np.float64
    np.testing.assert_allclose(backend.to_numpy(out1), ref1)

    ref2 = numpy_be.arctan2(np.float32(1.0), 1.5)
    out2 = backend.arctan2(np.float32(1.0), 1.5)
    assert backend.to_numpy(out2).dtype == ref2.dtype == np.float32
    np.testing.assert_allclose(backend.to_numpy(out2), ref2)


@requires_torch
def test_torch_integer_scalar_overflow_matches_numpy():
    """Weak integer scalars follow NumPy's range handling: maximum(int8, 128)
    raises OverflowError like NumPy; clip(int8, None, 128) succeeds (Finding 5)."""
    backend = be.get_registered("torch")
    numpy_be = NumpyBackend()
    x = np.array([1, 2, 3], dtype=np.int8)

    with pytest.raises(OverflowError):
        numpy_be.maximum(x, 128)
    with pytest.raises(OverflowError):
        backend.maximum(backend.array(x), 128)

    ref = numpy_be.clip(x, None, 128)
    out = backend.clip(backend.array(x), None, 128)
    assert backend.to_numpy(out).dtype == ref.dtype == np.int8
    np.testing.assert_array_equal(backend.to_numpy(out), ref)


@requires_torch
def test_torch_non_native_byteorder_construction_matches_numpy():
    """array/asarray accept a non-native-byte-order ndarray, which torch's
    from_numpy rejects directly (Finding 6)."""
    backend = be.get_registered("torch")
    big = np.arange(6.0, dtype=">f8")
    little = np.arange(6.0, dtype="<f8")
    for arr in (big, little):
        out = backend.array(arr)
        np.testing.assert_allclose(backend.to_numpy(out), np.asarray(arr))
        out2 = backend.asarray(arr)
        np.testing.assert_allclose(backend.to_numpy(out2), np.asarray(arr))
    # Non-native byte order combined with a negative stride still constructs.
    rev = np.arange(6, dtype=">i4")[::-1]
    np.testing.assert_array_equal(
        backend.to_numpy(backend.array(rev)), np.asarray(rev)
    )


@requires_torch
@pytest.mark.parametrize("member", FULL_SURFACE)
def test_torch_protocol_completeness(member):
    """Every declared surface member exists on the Torch backend."""
    backend = be.get_registered("torch")
    assert hasattr(backend, member), f"TorchBackend missing {member!r}"


# ---------------------------------------------------------------------------
# JAX backend (lazily registered; skipped when JAX is absent)
# ---------------------------------------------------------------------------


def test_jax_selection():
    """JAX: actionable error when absent (base-install path), round-trip when
    present. ``_HAS_JAX`` rather than ``find_spec`` decides the branch, so a
    conftest stand-in can never route this into the present-path assertions."""
    if not _HAS_JAX:
        with pytest.raises((ImportError, RuntimeError)) as exc:
            be.set_backend("jax")
        assert "jax" in str(exc.value).lower()
    else:
        be.set_backend("jax")
        backend = be.get_backend()
        result = backend.to_numpy(backend.array([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_jax_registration_raises_when_jax_absent():
    """set_backend('jax') raises an actionable ImportError when JAX is not
    installed. ``find_spec`` is forced to report jax absent and the registry is
    isolated from any prior jax registration, so the error path is exercised
    deterministically regardless of whether JAX is installed here."""
    real_find_spec = importlib.util.find_spec

    def _jax_absent(name, *args, **kwargs):
        if name == "jax":
            return None
        return real_find_spec(name, *args, **kwargs)

    registry_without_jax = {k: v for k, v in be._REGISTRY.items() if k != "jax"}
    with patch.object(be, "_REGISTRY", registry_without_jax), patch(
        "importlib.util.find_spec", side_effect=_jax_absent
    ):
        with pytest.raises(ImportError) as exc:
            be.set_backend("jax")
    assert "jax" in str(exc.value).lower()


@requires_jax
@pytest.mark.parametrize("switch", ["set_backend", "use_backend"])
def test_jax_round_trip(switch):
    """to_numpy(array([...])) reproduces the source values under jax, via both
    set_backend and the use_backend context manager."""
    source = [[1.0, 2.0], [3.0, 4.0]]
    if switch == "set_backend":
        be.set_backend("jax")
        backend = be.get_backend()
        result = backend.to_numpy(backend.array(source))
    else:
        with be.use_backend("jax") as backend:
            result = backend.to_numpy(backend.array(source))
    np.testing.assert_allclose(result, np.array(source), rtol=1e-12, atol=1e-12)


@requires_jax
def test_jax_backend_flags_gate_cache_and_gpu_routing():
    """JAX reports non-concrete (cache bypass) and no GPU routing: values may be
    jit/grad tracers, and the Numba CUDA kernel path is a separate route."""
    backend = be.get_registered("jax")
    assert backend.is_concrete is False
    assert backend.gpu_capable is False


@requires_jax
def test_jax_default_float_dtype_matches_numpy():
    """Python-float construction yields float64 to match NumPy numerics, not
    JAX's float32 default.

    This is the guard on the ``jax_enable_x64`` update the backend module makes
    at import: without it JAX silently narrows every construction below to
    float32 (and integers to int32), which would break float64 parity with the
    NumPy backend everywhere. Dtypes are asserted against the NumPy backend
    rather than hard-coded so the reference cannot drift.
    """
    backend = be.get_registered("jax")
    numpy_be = NumpyBackend()

    # Python floats default to float64 (not jax's float32 default).
    assert (
        backend.to_numpy(backend.array([1.0, 2.0])).dtype
        == numpy_be.array([1.0, 2.0]).dtype
        == np.float64
    )
    assert (
        backend.to_numpy(backend.zeros((2, 2))).dtype
        == numpy_be.zeros((2, 2)).dtype
        == np.float64
    )
    assert (
        backend.to_numpy(backend.eye(3)).dtype == numpy_be.eye(3).dtype == np.float64
    )
    # Integer input keeps 64-bit integer dtype (matching NumPy, not jax's int32).
    assert (
        backend.to_numpy(backend.array([1, 2, 3])).dtype
        == numpy_be.array([1, 2, 3]).dtype
        == np.int64
    )
    # Explicit dtype handles round-trip.
    assert backend.to_numpy(backend.array([1, 2, 3], dtype=backend.float32)).dtype == (
        np.float32
    )
    assert backend.to_numpy(backend.array([1, 2, 3], dtype=backend.float64)).dtype == (
        np.float64
    )


@requires_jax
def test_jax_integer_input_matches_numpy_semantics():
    """Integer input to float-producing ops promotes to float64 like NumPy.

    NumPy returns float64 for integer input to the transcendental, ``mean``, and
    linalg operations; JAX's own promotion lattice deliberately avoids widening
    and would return float32 instead. Preserving ops (``sum``/``abs``/``trace``)
    must keep the integer dtype.
    """
    backend = be.get_registered("jax")
    vec = [1, 2]
    mat = [[4, 1], [1, 3]]

    # Elementwise transcendental: float64 values matching NumPy.
    for name in ("sin", "cos", "sqrt", "arccos"):
        out = getattr(backend, name)(backend.array([0, 1]))
        assert backend.to_numpy(out).dtype == np.float64, name
        np.testing.assert_allclose(
            backend.to_numpy(out), getattr(np, name)(np.array([0, 1])), rtol=1e-7
        )
    at = backend.arctan2(backend.array(vec), backend.array(vec))
    assert backend.to_numpy(at).dtype == np.float64
    np.testing.assert_allclose(
        backend.to_numpy(at), np.arctan2(np.array(vec), np.array(vec)), rtol=1e-7
    )

    # Reductions / linalg that NumPy floats: value + float64 parity.
    mean_out = backend.mean(backend.array(vec))
    assert backend.to_numpy(mean_out).dtype == np.float64
    assert np.isclose(float(mean_out), np.mean(np.array(vec)))
    norm_out = backend.norm(backend.array(vec))
    assert backend.to_numpy(norm_out).dtype == np.float64
    assert np.isclose(float(norm_out), np.linalg.norm(np.array(vec)))
    for name, npfn in (
        ("inv", np.linalg.inv),
        ("pinv", np.linalg.pinv),
        ("svdvals", lambda m: np.linalg.svd(m, compute_uv=False)),
    ):
        out = getattr(backend, name)(backend.array(mat))
        assert backend.to_numpy(out).dtype == np.float64, name
        np.testing.assert_allclose(
            backend.to_numpy(out), npfn(np.array(mat)), rtol=1e-7
        )
    solve_out = backend.solve(backend.array(mat), backend.array(vec))
    assert backend.to_numpy(solve_out).dtype == np.float64
    np.testing.assert_allclose(
        backend.to_numpy(solve_out),
        np.linalg.solve(np.array(mat), np.array(vec)),
        rtol=1e-7,
    )

    # Preserving ops keep integer dtype, matching NumPy.
    assert backend.to_numpy(backend.sum(backend.array(vec))).dtype == np.int64
    assert backend.to_numpy(backend.abs(backend.array([-1, 2]))).dtype == np.int64
    assert backend.to_numpy(backend.trace(backend.array(mat))).dtype == np.int64


@requires_jax
def test_jax_linalg_matches_numpy():
    """The linalg surface (svd/pinv/inv/solve/norm/trace) matches NumPy values.

    ``pinv`` is the load-bearing one: jnp.linalg.pinv's default rtol is far
    looser than np.linalg.pinv's 1e-15 rcond, so a small singular value NumPy
    keeps would be zeroed without the explicit rtol the backend passes.
    """
    backend = be.get_registered("jax")
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_vec = np.array([1.0, 2.0])

    ta = backend.array(a)
    # svd: shapes and reconstruction.
    u, s, vt = backend.svd(ta)
    assert tuple(u.shape) == (2, 2) and tuple(s.shape) == (2,)
    np.testing.assert_allclose(
        backend.to_numpy(s), np.linalg.svd(a, compute_uv=False), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.svdvals(ta)), backend.to_numpy(s), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.inv(ta)), np.linalg.inv(a), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.pinv(ta)), np.linalg.pinv(a), rtol=1e-10
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.solve(ta, backend.array(b_vec))),
        np.linalg.solve(a, b_vec),
        rtol=1e-10,
    )
    assert np.isclose(float(backend.norm(backend.array([3.0, 4.0]))), 5.0)
    assert np.isclose(float(backend.trace(backend.eye(3))), 3.0)
    # A tiny singular value is zeroed like np.linalg.pinv's rcond, not inverted.
    d = np.diag([1.0, 9e-16])
    out_d = backend.to_numpy(backend.pinv(backend.array(d)))
    np.testing.assert_allclose(out_d, np.linalg.pinv(d), atol=1e-6)
    assert abs(out_d[1, 1]) < 1.0


@requires_jax
def test_jax_use_backend_restores_previous_backend():
    """use_backend('jax') restores the previously active backend on exit."""
    be.set_backend("numpy")
    original = be.get_backend()
    with be.use_backend("jax") as backend:
        assert backend.is_backend_array(backend.array([1.0, 2.0]))
        assert be.get_backend() is not original
    assert be.get_backend() is original


@requires_jax
def test_jax_is_backend_array_accepts_jit_tracer():
    """is_backend_array recognises a ``jax.jit`` tracer, not just a concrete
    array.

    Under ``jax.jit`` every value reaching backend code is a tracer with no
    host-readable value, which is exactly what ``is_concrete = False`` encodes;
    a predicate that only accepted materialized arrays would send traced values
    down host-only branches. Asserting the tracer is genuinely non-concrete
    (``np.asarray`` raises on it) keeps the test from passing on a concrete
    array that merely happened to flow through jit.
    """
    import jax
    import jax.numpy as jnp

    backend = be.get_registered("jax")
    source = np.array([0.1, 0.2, 0.3])
    seen = {}

    @jax.jit
    def _traced(x):
        seen["is_backend_array"] = backend.is_backend_array(x)
        seen["is_tracer"] = isinstance(x, jax.core.Tracer)
        return backend.sum(backend.sin(x))

    out = _traced(jnp.asarray(source))
    assert seen["is_tracer"] is True
    assert seen["is_backend_array"] is True
    # The traced value carried no host value, so this was not a concrete array.
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda x: np.asarray(x))(jnp.asarray(source))
    # Concrete results still satisfy the predicate, and the trace computed.
    assert backend.is_backend_array(out) is True
    assert np.isclose(float(out), np.sin(source).sum())


@requires_jax
def test_jax_ops_do_not_mutate_their_input():
    """No backend method writes into its input.

    JAX arrays are immutable, so the backend allocates a new array per
    operation and never updates in place; this pins that contract down for the
    whole surface (an in-place write would raise on a JAX array but could still
    slip through a NumPy-valued fallback path).
    """
    backend = be.get_registered("jax")
    source = np.array([1.0, -2.0, 3.0])
    vec = backend.array(source)
    mat = backend.array(np.array([[4.0, 1.0], [1.0, 3.0]]))
    vec_before = backend.to_numpy(vec).copy()
    mat_before = backend.to_numpy(mat).copy()

    unary = [
        "sin", "cos", "sqrt", "abs", "asarray", "diag", "isfinite",
        "sum", "mean", "amax", "amin", "argmax", "all", "any", "norm",
        "to_device", "ascontiguous", "to_numpy",
    ]
    for name in unary:
        getattr(backend, name)(vec)
        np.testing.assert_array_equal(backend.to_numpy(vec), vec_before, name)
    for name in ("svd", "svdvals", "inv", "pinv", "trace"):
        getattr(backend, name)(mat)
        np.testing.assert_array_equal(backend.to_numpy(mat), mat_before, name)

    backend.arccos(backend.clip(vec, -1.0, 1.0))
    backend.arctan2(vec, vec)
    backend.cross(vec, vec)
    backend.maximum(vec, 0.0)
    backend.minimum(vec, 0.0)
    backend.where(vec > 0, vec, 0.0)
    backend.stack([vec, vec])
    backend.concatenate([vec, vec])
    backend.matmul(backend.eye(3), vec)
    backend.solve(mat, backend.array([1.0, 2.0]))
    np.testing.assert_array_equal(backend.to_numpy(vec), vec_before)
    np.testing.assert_array_equal(backend.to_numpy(mat), mat_before)
    # The host ndarray the array was built from is untouched too.
    np.testing.assert_array_equal(source, np.array([1.0, -2.0, 3.0]))
    # And the immutability the backend relies on is real, not assumed.
    with pytest.raises(TypeError):
        vec[0] = 99.0


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
    module = importlib.import_module("ManipulaPy.backend")
    # importlib.reload re-executes the module body in place, minting new
    # function objects. Modules that bound names at import time (e.g.
    # ``from ManipulaPy.backend import get_backend``) would keep the old
    # objects, splitting identity for every later test. Snapshot and restore
    # the namespace so the reload leaves no trace.
    saved_namespace = dict(module.__dict__)
    try:
        reloaded = importlib.reload(module)
        assert isinstance(reloaded.get_backend(), NumpyBackend)
    finally:
        module.__dict__.clear()
        module.__dict__.update(saved_namespace)


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
        np.testing.assert_allclose(actual[key], expected_value, rtol=1e-10, atol=1e-10)
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
    expected_mass = np.array([[1.0 + (2.0 + 2.0 * c2), 1.0 + c2], [1.0 + c2, 1.0]])
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
            "velocity_quadratic_forces": dyn.velocity_quadratic_forces(theta, dtheta),
            "inverse_dynamics": dyn.inverse_dynamics(theta, dtheta, ddtheta, g, ftip),
            "forward_dynamics": dyn.forward_dynamics(theta, dtheta, tau, g, ftip),
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


class _GpuCapableBackend(NumpyBackend):
    """NumPy-numeric double that advertises itself as GPU-capable.

    Stands in for an active CuPy backend so the kernel-routing decision can be
    exercised on a machine without a GPU: the numerics stay NumPy, only the
    ``gpu_capable`` predicate flips.
    """

    gpu_capable = True


def test_gpu_capability_attribute_reflects_active_backend(monkeypatch):
    """The active backend advertises whether it can route numeric work to GPU."""
    assert NumpyBackend().gpu_capable is False
    assert be.get_backend().gpu_capable is False

    monkeypatch.setattr(be, "_active", _GpuCapableBackend())
    assert be.get_backend().gpu_capable is True


def test_planner_routes_on_active_backend_not_raw_cuda(monkeypatch):
    """Planner initialization consumes the centralized routing decision."""
    routing = iter((False, True))
    monkeypatch.setattr(
        traj_impl._runtime,
        "_cuda_routing_enabled",
        lambda _cuda_available: next(routing),
    )

    # Default NumPy backend is not GPU-capable -> CPU routing despite CUDA probe.
    numpy_planner = OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0), (-5.0, 5.0)],
        auto_optimize=False,
    )
    assert numpy_planner.cuda_available is False

    # A GPU-capable active backend -> GPU routing.
    gpu_planner = OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0), (-5.0, 5.0)],
        auto_optimize=False,
    )
    assert gpu_planner.cuda_available is True


def test_planner_uses_cuda_dispatch_seam_for_init_and_auto_optimize(monkeypatch):
    """Planner setup and execution capability share the kernel routing seam."""
    routing_checks = []
    setup_calls = []

    def _routing_enabled(cuda_available):
        routing_checks.append(cuda_available)
        return True

    monkeypatch.setattr(traj_impl._runtime, "_cuda_routing_enabled", _routing_enabled)
    monkeypatch.setattr(
        traj_impl._runtime,
        "setup_cuda_environment_for_40x_speedup",
        lambda: setup_calls.append(True),
    )
    planner = OptimizedTrajectoryPlanning(
        None, "nonexistent.urdf", None, [(-5.0, 5.0), (-5.0, 5.0)]
    )

    assert planner.cuda_available is True
    assert setup_calls == [True]
    assert len(routing_checks) == 1


def test_gpu_capable_backend_dispatches_joint_trajectory_to_kernel(monkeypatch):
    """A GPU-capable backend sends joint_trajectory through the kernel wrapper."""
    monkeypatch.setattr(
        traj_impl._runtime, "_cuda_routing_enabled", lambda _cuda_available: True
    )
    monkeypatch.setattr(be, "_active", _GpuCapableBackend())

    launched = []

    def _fake_kernel(thetastart, thetaend, Tf, N, method, **kwargs):
        launched.append(N)
        num_joints = len(thetastart)
        zeros = np.zeros((N, num_joints), dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()

    monkeypatch.setattr(
        traj_impl._runtime,
        "optimized_trajectory_generation_monitored",
        _fake_kernel,
    )

    planner = OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0), (-5.0, 5.0)],
        cuda_threshold=0,
        auto_optimize=False,
    )
    planner.joint_trajectory([0.0, 0.0], [1.0, -0.5], 1.0, 8, 5)

    assert launched == [8]


def test_numpy_backend_never_launches_joint_trajectory_kernel(monkeypatch):
    """Under the default backend the GPU kernel wrapper is never invoked."""

    def _boom(*args, **kwargs):
        raise AssertionError("kernel wrapper must not run under the NumPy backend")

    monkeypatch.setattr(
        traj_impl._runtime,
        "optimized_trajectory_generation_monitored",
        _boom,
    )

    planner = OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0), (-5.0, 5.0)],
        cuda_threshold=0,
        auto_optimize=False,
    )
    traj = planner.joint_trajectory([0.0, 0.0], [1.0, -0.5], 1.0, 8, 5)
    assert traj["positions"].shape == (8, 2)


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

    expected_pos = np.array([[0.0, 0.0], [0.5, -0.25], [1.0, -0.5]], dtype=np.float32)
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
        np.testing.assert_array_equal(
            traj["orientations"][i], np.eye(3, dtype=np.float32)
        )
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


# ---------------------------------------------------------------------------
# Trajectory-dynamics dispatch
# ---------------------------------------------------------------------------


class _TrajectoryDynamicsSpyBackend(NumpyBackend):
    """NumPy delegate that records primitives used by migrated dynamics paths.

    ``stack`` is the tell-tale of the per-waypoint / per-step assembly (the
    pre-migration code pre-allocated and wrote in place, never stacking), and
    ``clip`` is the torque/joint-limit enforcement. The migrated
    ``ManipulatorDynamics`` layer routes through ``matmul``/``concatenate`` but
    never ``stack`` or ``clip``, so these two single out the planning-layer work.
    """

    def __init__(self):
        self.calls = []

    def asarray(self, obj, dtype=None):
        self.calls.append("asarray")
        return super().asarray(obj, dtype=dtype)

    def clip(self, x, a_min, a_max):
        self.calls.append("clip")
        return super().clip(x, a_min, a_max)

    def stack(self, arrays, axis=0):
        self.calls.append("stack")
        return super().stack(arrays, axis=axis)


class _SumInverseDynamics:
    """Deterministic dynamics: torque = theta + dtheta + ddtheta (plain NumPy)."""

    def inverse_dynamics(self, theta, dtheta, ddtheta, g, Ftip):
        return np.asarray(theta) + np.asarray(dtheta) + np.asarray(ddtheta)


class _TauForwardDynamics:
    """Deterministic dynamics: joint acceleration equals applied torque (NumPy)."""

    def forward_dynamics(self, theta, dtheta, tau, g, Ftip):
        return np.asarray(tau, dtype=np.float64)


def _dynamics_planner(dynamics, joint_limits, torque_limits):
    """Force-CPU planner wired to a deterministic dynamics stub."""
    return OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        dynamics,
        joint_limits,
        torque_limits,
        use_cuda=False,
    )


def test_inverse_dynamics_cpu_default_backend_parity():
    """Default NumPy keeps per-waypoint torques, torque-limit clip and float32."""
    planner = _dynamics_planner(
        _SumInverseDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], [[-3.0, 4.0], [-3.0, 4.0]]
    )
    theta = np.array([[0, 0], [1, 2], [3, 4]], dtype=np.float32)
    dtheta = np.array([[0, 0], [0.5, 0.5], [1, 1]], dtype=np.float32)
    ddtheta = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)

    torques = planner.inverse_dynamics_trajectory(
        theta, dtheta, ddtheta, np.array([0, 0, -9.81]), np.zeros(6)
    )

    expected = np.array([[0.0, 0.0], [1.6, 2.6], [4.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(torques, expected)
    assert torques.dtype == np.float32


def test_forward_dynamics_cpu_default_backend_parity():
    """Default NumPy keeps Euler integration order, joint clamp and float32."""
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], None)
    thetalist = np.zeros(2, dtype=np.float32)
    dthetalist = np.zeros(2, dtype=np.float32)
    # Values chosen so the Euler recurrence stays exactly representable in
    # float32, making assert_array_equal a bit-exact parity check.
    taumat = np.array([[0, 0], [2, 2], [4, 4]], dtype=np.float64)
    result = planner.forward_dynamics_trajectory(
        thetalist, dthetalist, taumat, np.zeros(3), np.zeros((3, 6)), 0.5, 1
    )

    expected_pos = np.array([[0.0, 0.0], [0.5, 0.5], [2.0, 2.0]], dtype=np.float32)
    expected_vel = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    expected_acc = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32)

    np.testing.assert_array_equal(result["positions"], expected_pos)
    np.testing.assert_array_equal(result["velocities"], expected_vel)
    np.testing.assert_array_equal(result["accelerations"], expected_acc)
    for key in ("positions", "velocities", "accelerations"):
        assert result[key].dtype == np.float32


def test_calculate_derivatives_default_backend_parity():
    """Default NumPy keeps finite-difference derivatives, shapes and dtype."""
    planner = _dynamics_planner(None, [(-5.0, 5.0), (-5.0, 5.0)], None)
    positions = np.array(
        [[0.0, 0.0], [1.0, 2.0], [4.0, 6.0], [9.0, 12.0], [16.0, 20.0]],
        dtype=np.float64,
    )
    velocity, acceleration, jerk = planner.calculate_derivatives(positions, 0.5)

    np.testing.assert_array_equal(velocity, np.diff(positions, axis=0) / 0.5)
    np.testing.assert_array_equal(acceleration, np.diff(velocity, axis=0) / 0.5)
    np.testing.assert_array_equal(jerk, np.diff(acceleration, axis=0) / 0.5)
    assert velocity.shape == (4, 2)
    assert acceleration.shape == (3, 2)
    assert jerk.shape == (2, 2)
    assert velocity.dtype == np.float64


def test_inverse_dynamics_cpu_dispatches_through_active_backend(monkeypatch):
    """The CPU inverse-dynamics path stacks per-waypoint torques and clips them."""
    planner = _dynamics_planner(
        _SumInverseDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], [[-3.0, 4.0], [-3.0, 4.0]]
    )
    theta = np.array([[0, 0], [1, 2], [3, 4]], dtype=np.float32)
    dtheta = np.array([[0, 0], [0.5, 0.5], [1, 1]], dtype=np.float32)
    ddtheta = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
    spy = _TrajectoryDynamicsSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.inverse_dynamics_trajectory(
        theta, dtheta, ddtheta, np.array([0, 0, -9.81]), np.zeros(6)
    )

    assert "stack" in spy.calls
    assert "clip" in spy.calls


def test_forward_dynamics_cpu_dispatches_through_active_backend(monkeypatch):
    """The CPU forward-dynamics path stacks per-step states and clamps joints."""
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], None)
    thetalist = np.zeros(2, dtype=np.float32)
    dthetalist = np.zeros(2, dtype=np.float32)
    taumat = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    spy = _TrajectoryDynamicsSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.forward_dynamics_trajectory(
        thetalist, dthetalist, taumat, np.zeros(3), np.zeros((3, 6)), 0.1, 1
    )

    assert "stack" in spy.calls
    assert "clip" in spy.calls


def test_calculate_derivatives_dispatches_through_active_backend(monkeypatch):
    """calculate_derivatives adopts the active backend instead of raw NumPy."""
    planner = _dynamics_planner(None, [(-5.0, 5.0), (-5.0, 5.0)], None)
    positions = np.array(
        [[0.0, 0.0], [1.0, 2.0], [4.0, 6.0], [9.0, 12.0]], dtype=np.float64
    )
    spy = _TrajectoryDynamicsSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.calculate_derivatives(positions, 0.5)

    assert "asarray" in spy.calls


# ---------------------------------------------------------------------------
# Path-planning / collision-avoidance dispatch
# ---------------------------------------------------------------------------


class _PathPlanningSpyBackend(NumpyBackend):
    """NumPy delegate recording primitives used by the migrated planning paths.

    ``stack`` is the tell-tale of the functional trajectory rebuild in
    ``_apply_collision_avoidance_cpu`` (the pre-migration code mutated rows in
    place and never stacked); ``asarray`` and ``to_numpy`` mark the backend
    and host-boundary conversions the pre-migration ``np.array``/``.tolist``
    and in-place code never made.
    """

    def __init__(self):
        self.calls = []

    def asarray(self, obj, dtype=None):
        self.calls.append("asarray")
        return super().asarray(obj, dtype=dtype)

    def to_numpy(self, x):
        self.calls.append("to_numpy")
        return super().to_numpy(x)

    def stack(self, arrays, axis=0):
        self.calls.append("stack")
        return super().stack(arrays, axis=axis)


class _CollideOnceChecker:
    """Collision checker reporting one colliding point, then clearing.

    The first ``check_collision`` returns True and every later call returns
    False, so the gradient-descent inner loop runs exactly one adjustment
    iteration deterministically.
    """

    def __init__(self):
        self.calls = 0

    def check_collision(self, q):
        self.calls += 1
        return self.calls == 1


class _AlwaysCollideChecker:
    """Collision checker that always reports a collision."""

    def check_collision(self, q):
        return True


class _ConstantGradientField:
    """Potential field returning a fixed host-NumPy (float64) gradient of ones."""

    def compute_gradient(self, q, q_goal, obstacles):
        return np.ones_like(np.asarray(q, dtype=np.float64))


class _RecordingGradientField:
    """Ones gradient that records the argument types each call receives."""

    def __init__(self):
        self.arg_types = []

    def compute_gradient(self, q, q_goal, obstacles):
        self.arg_types.append((type(q), type(q_goal), [type(o) for o in obstacles]))
        return np.ones_like(np.asarray(q, dtype=np.float64))


class _RecordingCollideOnceChecker:
    """Collide-once checker that records the type of each argument it receives."""

    def __init__(self):
        self.calls = 0
        self.arg_types = []

    def check_collision(self, q):
        self.calls += 1
        self.arg_types.append(type(q))
        return self.calls == 1


def test_collision_avoidance_cpu_default_backend_parity():
    """Default NumPy keeps the one-step gradient nudge, shape and float32 dtype."""
    planner = _cpu_trajectory_planner()
    planner.collision_checker = _CollideOnceChecker()
    planner.potential_field = _ConstantGradientField()
    traj_pos = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    result = planner._apply_collision_avoidance_cpu(
        traj_pos, np.array([0.0, 0.0], dtype=np.float32)
    )

    # Row 0 collides once: q -= 0.01 * ones. Row 1 never collides -> unchanged.
    expected = np.array([[-0.01, -0.01], [1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.shape == traj_pos.shape
    assert result.dtype == np.float32


def test_collision_avoidance_cpu_dispatches_through_active_backend(monkeypatch):
    """The CPU collision path rebuilds via stack and crosses the host boundary.

    Beyond routing, this locks the host boundary (fix): the row and the goal are
    both converted before the host potential field / collision checker see them,
    and those host modules only ever receive ``np.ndarray`` (never a
    backend-native array), so the gradient update happens entirely on the host.
    """
    planner = _cpu_trajectory_planner()
    checker = _RecordingCollideOnceChecker()
    field = _RecordingGradientField()
    planner.collision_checker = checker
    planner.potential_field = field
    traj_pos = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    spy = _PathPlanningSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner._apply_collision_avoidance_cpu(
        traj_pos, np.array([0.0, 0.0], dtype=np.float32)
    )

    assert "stack" in spy.calls
    # The goal plus each of the two rows cross the host boundary via to_numpy.
    assert spy.calls.count("to_numpy") >= 3
    # The host potential field only ever receives NumPy for q and q_goal.
    assert field.arg_types
    assert all(q is np.ndarray and g is np.ndarray for q, g, _ in field.arg_types)
    # The collision checker likewise only ever sees NumPy rows.
    assert checker.arg_types and all(t is np.ndarray for t in checker.arg_types)


def test_collision_avoidance_cpu_empty_trajectory_returns_empty():
    """An empty trajectory returns an empty (0, num_joints) float32 array.

    Base iterated an empty trajectory and returned it unchanged; the migrated
    stack-based rebuild must not raise on ``stack([])``.
    """
    planner = _cpu_trajectory_planner()
    planner.collision_checker = _CollideOnceChecker()
    planner.potential_field = _ConstantGradientField()
    traj_pos = np.zeros((0, 2), dtype=np.float32)

    result = planner._apply_collision_avoidance_cpu(
        traj_pos, np.array([0.0, 0.0], dtype=np.float32)
    )

    assert result.shape == (0, 2)
    assert result.dtype == np.float32


def test_plan_trajectory_default_backend_parity():
    """Default NumPy keeps the interpolated waypoints as a plain Python list."""
    planner = _cpu_trajectory_planner()
    result = planner.plan_trajectory([0.0, 0.0], [1.0, 2.0], [])

    assert isinstance(result, list) and isinstance(result[0], list)
    assert len(result) == 6
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[-1], [1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(result[2], [0.4, 0.8], atol=1e-6)


def test_plan_trajectory_gradient_host_boundary_parity():
    """The plan_trajectory gradient path integrates through the host boundary."""
    planner = _cpu_trajectory_planner()
    planner.potential_field = _ConstantGradientField()
    planner.collision_checker = _AlwaysCollideChecker()

    result = planner.plan_trajectory([0.0, 0.0], [1.0, 2.0], [(0.5, 0.5)])

    # The checker never clears, so all 10 gradient steps of -0.01 * ones run.
    np.testing.assert_allclose(result[0], [-0.1, -0.1], atol=1e-6)
    np.testing.assert_allclose(result[-1], [0.9, 1.9], atol=1e-6)
    assert isinstance(result[0], list)


def test_plan_trajectory_dispatches_through_active_backend(monkeypatch):
    """plan_trajectory adopts the active backend instead of raw NumPy."""
    planner = _cpu_trajectory_planner()
    spy = _PathPlanningSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    planner.plan_trajectory([0.0, 0.0], [1.0, 2.0], [])

    assert "asarray" in spy.calls
    assert "to_numpy" in spy.calls


def test_plan_trajectory_gradient_only_hosts_the_potential_field(monkeypatch):
    """The gradient path passes only NumPy (waypoint, goal, obstacles) to the field.

    Locks the host boundary: the waypoint, target and each obstacle are all
    converted to NumPy before the host potential field sees them, so the nudge
    integrates entirely on the host with no backend-native mixed arithmetic.
    """
    planner = _cpu_trajectory_planner()
    field = _RecordingGradientField()
    planner.potential_field = field
    planner.collision_checker = _AlwaysCollideChecker()
    spy = _PathPlanningSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    # A plain-tuple obstacle must be converted to NumPy at the boundary.
    planner.plan_trajectory([0.0, 0.0], [1.0, 2.0], [(0.5, 0.5)])

    assert field.arg_types
    for q, g, obs in field.arg_types:
        assert q is np.ndarray and g is np.ndarray
        assert obs and all(o is np.ndarray for o in obs)


# ---------------------------------------------------------------------------
# Potential-field dispatch and collision host boundaries
# ---------------------------------------------------------------------------


class _PotentialFieldSpyBackend(NumpyBackend):
    """NumPy delegate recording primitives used by potential-field math."""

    def __init__(self):
        self.calls = []

    def asarray(self, obj, dtype=None):
        self.calls.append("asarray")
        return super().asarray(obj, dtype=dtype)

    def zeros(self, shape, dtype=None):
        self.calls.append("zeros")
        return super().zeros(shape, dtype=dtype)

    def norm(self, x, ord=None, axis=None):
        self.calls.append("norm")
        return super().norm(x, ord=ord, axis=axis)

    def maximum(self, x1, x2):
        self.calls.append("maximum")
        return super().maximum(x1, x2)

    def where(self, condition, x, y):
        self.calls.append("where")
        return super().where(condition, x, y)

    def sum(self, x, axis=None):
        self.calls.append("sum")
        return super().sum(x, axis=axis)

    def to_numpy(self, x):
        self.calls.append("to_numpy")
        return super().to_numpy(x)


def test_attractive_potential_dispatches_with_numpy_spy_parity(monkeypatch):
    """Attractive potential independently dispatches and preserves its value."""
    field = PotentialField(attractive_gain=1.5)
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    q = np.array([0.0, 0.0])
    goal = np.array([1.0, -1.0])

    actual = field.compute_attractive_potential(q, goal)

    assert actual == pytest.approx(1.5)
    assert spy.calls.count("asarray") >= 2
    assert "sum" in spy.calls


def test_repulsive_potential_dispatches_with_numpy_spy_parity(monkeypatch):
    """Repulsive potential independently dispatches and preserves its value."""
    field = PotentialField(repulsive_gain=2.0, influence_distance=1.0)
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    actual = field.compute_repulsive_potential(
        np.array([0.0, 0.0]), [np.array([0.5, 0.0])]
    )

    assert actual == pytest.approx(40.0)
    assert spy.calls.count("asarray") >= 2
    assert "norm" in spy.calls
    assert "maximum" in spy.calls
    assert "where" in spy.calls


def test_gradient_dispatches_with_numpy_spy_parity(monkeypatch):
    """Potential gradient independently dispatches and preserves its value."""
    field = PotentialField(
        attractive_gain=1.5, repulsive_gain=2.0, influence_distance=1.0
    )
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    actual = field.compute_gradient(
        np.array([0.0, 0.0]),
        np.array([1.0, -1.0]),
        [np.array([0.5, 0.0])],
    )

    np.testing.assert_allclose(actual, [318.5, 1.5])
    assert spy.calls.count("asarray") >= 4
    assert "norm" in spy.calls
    assert "maximum" in spy.calls
    assert "where" in spy.calls
    assert "zeros" in spy.calls


def test_repulsive_potential_preserves_float32_dtype_inside_influence():
    """In-range scalar repulsion keeps the configuration's float32 dtype."""
    field = PotentialField(repulsive_gain=2.0, influence_distance=1.0)
    q = np.array([0.0, 0.0], dtype=np.float32)
    obstacle = np.array([0.5, 0.0], dtype=np.float32)

    actual = field.compute_repulsive_potential(q, [obstacle])

    assert np.asarray(actual).dtype == np.float32


def test_gradient_preserves_float32_dtype_inside_influence():
    """In-range attractive and repulsive gradient math remains float32."""
    field = PotentialField(
        attractive_gain=1.5, repulsive_gain=2.0, influence_distance=1.0
    )
    q = np.array([0.0, 0.0], dtype=np.float32)

    actual = field.compute_gradient(
        q,
        np.array([1.0, -1.0], dtype=np.float32),
        [np.array([0.5, 0.0], dtype=np.float32)],
    )

    assert actual.dtype == np.float32


def test_attractive_potential_preserves_fractional_math_for_integer_inputs():
    """Integer configurations do not truncate fractional gains or constants."""
    field = PotentialField(attractive_gain=0.5)

    actual = field.compute_attractive_potential(
        np.array([1, 2], dtype=np.int64), np.array([0, 0], dtype=np.int64)
    )

    assert actual == pytest.approx(1.25)
    assert np.asarray(actual).dtype == np.float64


def test_attractive_paths_preserve_mixed_goal_dtype_promotion():
    """A float64 goal promotes float32 attractive potential and gradient."""
    field = PotentialField(attractive_gain=1.5)
    q = np.array([0.0, 0.0], dtype=np.float32)
    goal = np.array([1.0, -1.0], dtype=np.float64)

    potential = field.compute_attractive_potential(q, goal)
    gradient = field.compute_gradient(q, goal, [])

    assert potential == pytest.approx(1.5)
    assert np.asarray(potential).dtype == np.float64
    np.testing.assert_allclose(gradient, [-1.5, 1.5])
    assert gradient.dtype == np.float64


def test_repulsive_paths_preserve_mixed_obstacle_dtype_promotion():
    """A float64 obstacle promotes float32 repulsive potential and gradient."""
    field = PotentialField(
        attractive_gain=0.0, repulsive_gain=2.0, influence_distance=1.0
    )
    q = np.array([0.0, 0.0], dtype=np.float32)
    obstacle = np.array([0.5, 0.0], dtype=np.float64)

    potential = field.compute_repulsive_potential(q, [obstacle])
    gradient = field.compute_gradient(q, np.zeros(2, dtype=np.float32), [obstacle])

    assert potential == pytest.approx(40.0)
    assert np.asarray(potential).dtype == np.float64
    np.testing.assert_allclose(gradient, [320.0, 0.0])
    assert gradient.dtype == np.float64


def test_exact_obstacle_float32_gradient_is_benign_under_strict_errstate():
    """The unselected regular branch stays finite at an exact obstacle."""
    field = PotentialField(
        attractive_gain=0.0, repulsive_gain=3.0, influence_distance=0.5
    )
    q = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        gradient = field.compute_gradient(q, np.zeros(3, dtype=np.float32), [q])

    np.testing.assert_array_equal(gradient, [3.0, 0.0, 0.0])
    assert gradient.dtype == np.float32
    assert np.all(np.isfinite(gradient))


def test_potential_field_dtype_promotion_does_not_inspect_dtype_kind():
    """Promotion stays backend-neutral for dtype objects without NumPy `.kind`."""
    source = inspect.getsource(potential_field_fields)

    assert 'getattr(dtype, "kind"' not in source
    assert ".kind" not in source


def test_repulsive_potential_empty_generator_preserves_python_zero_contract():
    """An exhausted obstacle generator returns the established Python int zero."""
    field = PotentialField()
    empty_obstacles = (obstacle for obstacle in ())

    actual = field.compute_repulsive_potential(np.zeros(2), empty_obstacles)

    assert type(actual) is int
    assert actual == 0


def test_potential_field_exact_obstacle_escape_dispatches_without_host_branch(
    monkeypatch,
):
    """Exact-obstacle handling stays finite and escapes along positive x."""
    field = PotentialField(
        attractive_gain=0.0, repulsive_gain=3.0, influence_distance=0.5
    )
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    q = np.array([1.0, 1.0, 1.0])

    potential = field.compute_repulsive_potential(q, [q.copy()])
    gradient = field.compute_gradient(q, np.zeros(3), [q.copy()])

    assert np.isfinite(potential)
    np.testing.assert_array_equal(gradient, [3.0, 0.0, 0.0])
    assert "where" in spy.calls


def test_potential_field_native_cupy_parity_when_available():
    """CuPy inputs stay device-native while matching NumPy results."""
    cp = pytest.importorskip("cupy")
    if not isinstance(getattr(cp, "ndarray", None), type):
        pytest.skip("CuPy test double does not provide native array types")
    try:
        q_device = cp.asarray([0.2, 0.1], dtype=cp.float64)
    except Exception as exc:
        pytest.skip(f"Native CuPy runtime unavailable: {exc}")

    field = PotentialField(
        attractive_gain=1.2, repulsive_gain=0.8, influence_distance=1.0
    )
    q_host = np.array([0.2, 0.1])
    goal_host = np.array([0.5, -0.1])
    obstacles_host = [np.array([0.6, 0.1])]
    expected = (
        field.compute_attractive_potential(q_host, goal_host),
        field.compute_repulsive_potential(q_host, obstacles_host),
        field.compute_gradient(q_host, goal_host, obstacles_host),
    )

    with be.use_backend("cupy"):
        actual = (
            field.compute_attractive_potential(q_device, cp.asarray(goal_host)),
            field.compute_repulsive_potential(
                q_device, [cp.asarray(obstacles_host[0])]
            ),
            field.compute_gradient(
                q_device,
                cp.asarray(goal_host),
                [cp.asarray(obstacles_host[0])],
            ),
        )

    assert all(isinstance(value, cp.ndarray) for value in actual)
    np.testing.assert_allclose(cp.asnumpy(actual[0]), expected[0])
    np.testing.assert_allclose(cp.asnumpy(actual[1]), expected[1])
    np.testing.assert_allclose(cp.asnumpy(actual[2]), expected[2])


class _RecordingFkRobot:
    def __init__(self, transforms):
        self.transforms = transforms
        self.configurations = []

    def link_fk(self, cfg, use_names):
        self.configurations.append(cfg)
        return self.transforms


def _collision_checker_without_urdf(robot, hulls):
    checker = CollisionChecker.__new__(CollisionChecker)
    checker.robot = robot
    checker.convex_hulls = hulls
    checker._acm = set()
    return checker


class _DeviceValue:
    def __init__(self, value):
        self.value = np.asarray(value)


class _HostBoundaryBackend(NumpyBackend):
    """Device-like backend that rejects attempts to host an existing NumPy value."""

    def __init__(self):
        self.to_numpy_calls = 0

    def is_backend_array(self, value):
        return isinstance(value, _DeviceValue)

    def to_numpy(self, value):
        if not isinstance(value, _DeviceValue):
            raise AssertionError("ordinary host values must not call to_numpy")
        self.to_numpy_calls += 1
        return value.value


def test_collision_checker_keeps_ordinary_numpy_configuration_on_host(monkeypatch):
    """A host configuration is not routed through an active device backend."""
    robot = _RecordingFkRobot({})
    checker = _collision_checker_without_urdf(robot, {})
    backend = _HostBoundaryBackend()
    monkeypatch.setattr(be, "_active", backend)
    configuration = np.array([0.1, -0.2])

    checker.check_collision(configuration)

    assert robot.configurations[0] is configuration
    assert backend.to_numpy_calls == 0


def test_collision_checker_hosts_backend_native_configuration(monkeypatch):
    """A backend-native configuration explicitly crosses to host NumPy."""
    robot = _RecordingFkRobot({})
    checker = _collision_checker_without_urdf(robot, {})
    backend = _HostBoundaryBackend()
    monkeypatch.setattr(be, "_active", backend)
    configuration = _DeviceValue([0.1, -0.2])

    checker.check_collision(configuration)

    np.testing.assert_array_equal(robot.configurations[0], [0.1, -0.2])
    assert backend.to_numpy_calls == 1


def test_collision_checker_hosts_array_configuration_but_preserves_dict(monkeypatch):
    """Only array configurations cross the explicit URDF host boundary."""
    robot = _RecordingFkRobot({})
    checker = _collision_checker_without_urdf(robot, {})
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    array_cfg = np.array([0.1, -0.2])
    dict_cfg = {"joint_a": 0.1}
    checker.check_collision(array_cfg)
    checker.check_collision(dict_cfg)

    assert isinstance(robot.configurations[0], np.ndarray)
    assert robot.configurations[1] is dict_cfg
    assert "to_numpy" in spy.calls


def test_collision_checker_hosts_fk_transforms_before_numpy_geometry(monkeypatch):
    """FK transforms cross to host before NumPy/SciPy collision geometry."""
    from types import SimpleNamespace

    transforms = {"a": np.eye(4), "b": np.eye(4)}
    robot = _RecordingFkRobot(transforms)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    hulls = {
        "a": SimpleNamespace(points=points),
        "b": SimpleNamespace(points=points + 10.0),
    }
    checker = _collision_checker_without_urdf(robot, hulls)
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    assert checker.check_collision(np.zeros(1)) is False
    # One configuration and two FK transforms cross the host boundary.
    assert spy.calls.count("to_numpy") >= 3


def test_collision_checker_hosts_mesh_data_before_scipy_hull(monkeypatch):
    """Mesh vertices and origin matrices cross the SciPy host boundary."""
    from types import SimpleNamespace

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    geometry = SimpleNamespace(mesh_data=SimpleNamespace(vertices=vertices))
    collision = SimpleNamespace(
        geometry=geometry, origin=SimpleNamespace(matrix=np.eye(4))
    )
    link = SimpleNamespace(name="link", collisions=[collision], visuals=[])
    checker = CollisionChecker.__new__(CollisionChecker)
    checker.robot = SimpleNamespace(links=[link])
    checker._visual_fallback_warned = set()
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)

    hulls = checker._create_convex_hulls()

    assert "link" in hulls
    assert spy.calls.count("to_numpy") >= 2


def test_collision_checker_hosts_point_clouds_before_numpy_bounds(monkeypatch):
    """Direct point-cloud callers cross the documented NumPy host boundary."""
    checker = CollisionChecker.__new__(CollisionChecker)
    spy = _PotentialFieldSpyBackend()
    monkeypatch.setattr(be, "_active", spy)
    pts_a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pts_b = np.array([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])

    assert checker._points_intersect(pts_a, pts_b) is True
    assert spy.calls.count("to_numpy") >= 2


# ---------------------------------------------------------------------------
# Degenerate-input edge semantics (restore pre-migration behavior)
# ---------------------------------------------------------------------------


def test_batch_joint_trajectory_cpu_empty_batch_returns_empty_shape():
    """A zero-length batch returns (0, N, num_joints) float32 arrays, not a raise.

    Base preallocated the empty batch; the migrated stack-based rebuild must not
    raise on ``stack([])``.
    """
    planner = _cpu_trajectory_planner()
    thetastart = np.zeros((0, 2), dtype=np.float32)
    thetaend = np.zeros((0, 2), dtype=np.float32)

    traj = planner.batch_joint_trajectory(thetastart, thetaend, 1.0, 3, 5)

    for key in ("positions", "velocities", "accelerations"):
        assert traj[key].shape == (0, 3, 2)
        assert traj[key].dtype == np.float32


def test_inverse_dynamics_cpu_empty_trajectory_returns_empty_shape():
    """A zero-point trajectory returns a (0, num_joints) float32 torque array."""
    planner = _dynamics_planner(
        _SumInverseDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], [[-3.0, 4.0], [-3.0, 4.0]]
    )
    empty = np.zeros((0, 2), dtype=np.float32)

    torques = planner.inverse_dynamics_trajectory(
        empty, empty, empty, np.array([0, 0, -9.81]), np.zeros(6)
    )

    assert torques.shape == (0, 2)
    assert torques.dtype == np.float32


def test_cartesian_trajectory_zero_points_returns_base_empty_shapes():
    """N == 0 returns base's empty shapes: (0,) positions, (0, 3, 3) orientations."""
    planner = _cpu_trajectory_planner()
    Xstart = np.eye(4)
    Xend = np.eye(4)
    Xend[:3, 3] = [1.0, 2.0, 3.0]

    traj = planner.cartesian_trajectory(Xstart, Xend, 1.0, 0, 5)

    assert traj["positions"].shape == (0,)
    assert traj["orientations"].shape == (0, 3, 3)
    assert traj["velocities"].shape == (0, 3)
    assert traj["accelerations"].shape == (0, 3)
    for key in ("positions", "velocities", "accelerations", "orientations"):
        assert traj[key].dtype == np.float32


def test_forward_dynamics_cpu_zero_steps_raises_like_base():
    """A zero-step request raises IndexError exactly as the base preallocation did."""
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], None)
    thetalist = np.zeros(2, dtype=np.float32)
    dthetalist = np.zeros(2, dtype=np.float32)
    taumat = np.zeros((0, 2), dtype=np.float64)

    with pytest.raises(IndexError):
        planner.forward_dynamics_trajectory(
            thetalist, dthetalist, taumat, np.zeros(3), np.zeros((0, 6)), 0.1, 1
        )


def test_forward_dynamics_cpu_integer_state_freezes_like_base():
    """Integer initial state refuses the float cast and freezes at the initial value.

    Base integrated with in-place ``+=``, which raised an unsafe float->int
    same-kind cast that the loop's except-handler turned into zero acceleration,
    leaving the integer state frozen. The migrated functional cast must not
    silently truncate to a diverging non-zero trajectory.
    """
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], None)
    thetalist = np.array([1, 2], dtype=np.int64)
    dthetalist = np.array([0, 0], dtype=np.int64)
    taumat = np.full((3, 2), 4.0, dtype=np.float64)

    result = planner.forward_dynamics_trajectory(
        thetalist, dthetalist, taumat, np.zeros(3), np.zeros((3, 6)), 0.5, 1
    )

    np.testing.assert_array_equal(
        result["positions"], np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result["velocities"], np.zeros((3, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result["accelerations"], np.zeros((3, 2), dtype=np.float32)
    )


def _base_forward_dynamics_inplace(
    planner, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
):
    """Faithful reimplementation of the pre-migration in-place CPU Euler loop.

    Mirrors ``np.zeros`` float32 outputs plus in-place ``+=`` accumulation and
    ``np.clip`` against the float32 joint limits, whose promotion changes the
    live state dtype exactly as the original code did.
    """
    num_steps = taumat.shape[0]
    num_joints = thetalist.shape[0]
    thetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
    dthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
    ddthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
    current_theta = thetalist.copy()
    current_dtheta = dthetalist.copy()
    thetamat[0, :] = current_theta
    dthetamat[0, :] = current_dtheta
    dt_step = dt / intRes
    for i in range(1, num_steps):
        for _ in range(intRes):
            try:
                ddtheta = planner.dynamics.forward_dynamics(
                    current_theta, current_dtheta, taumat[i], g, Ftipmat[i]
                )
                current_dtheta += ddtheta * dt_step
                current_theta += current_dtheta * dt_step
                current_theta = np.clip(
                    current_theta,
                    planner.joint_limits[:, 0],
                    planner.joint_limits[:, 1],
                )
                ddthetamat[i] = ddtheta
            except Exception:
                ddthetamat[i] = np.zeros(num_joints)
        thetamat[i, :] = current_theta
        dthetamat[i, :] = current_dtheta
    return thetamat, dthetamat, ddthetamat


def test_forward_dynamics_cpu_float16_state_matches_base_inplace_loop():
    """A float16 initial state must track base's clip-promoted live dtype.

    Base's first ``np.clip`` against the float32 joint limits promoted the state
    to float32, so later updates accumulated in float32; casting every update
    back to the initial float16 dtype rounds repeatedly and diverges.
    """
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)], None)
    thetalist = np.array([0.1, 0.2], dtype=np.float16)
    dthetalist = np.array([0.0, 0.0], dtype=np.float16)
    taumat = np.full((4, 2), 0.3, dtype=np.float64)
    g = np.zeros(3)
    Ftipmat = np.zeros((4, 6))

    result = planner.forward_dynamics_trajectory(
        thetalist, dthetalist, taumat, g, Ftipmat, 0.5, 2
    )
    exp_pos, exp_vel, exp_acc = _base_forward_dynamics_inplace(
        planner, thetalist, dthetalist, taumat, g, Ftipmat, 0.5, 2
    )

    np.testing.assert_array_equal(result["positions"], exp_pos)
    np.testing.assert_array_equal(result["velocities"], exp_vel)
    np.testing.assert_array_equal(result["accelerations"], exp_acc)


def test_cartesian_trajectory_negative_points_raises_like_base():
    """N < 0 must raise ValueError from the (N, ...) preallocation as base did.

    The empty guard keys on N == 0; a negative N must fall through to
    ``zeros((N, 3, 3))`` and raise, not silently return empty arrays.
    """
    planner = _cpu_trajectory_planner()
    Xstart = np.eye(4)
    Xend = np.eye(4)
    Xend[:3, 3] = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        planner.cartesian_trajectory(Xstart, Xend, 1.0, -1, 5)


def test_forced_cpu_planner_skips_benchmark_cpu_comparison(monkeypatch):
    """use_cuda=False must not yield a CPU-vs-CPU "speedup" on routable hardware."""
    monkeypatch.setattr(
        traj_impl._runtime, "_cuda_routing_enabled", lambda _cuda_available: True
    )

    planner = OptimizedTrajectoryPlanning(
        None,
        "nonexistent.urdf",
        None,
        [(-5.0, 5.0)],
        use_cuda=False,
        auto_optimize=False,
    )
    results = planner.benchmark_performance(
        test_cases=[{"N": 8, "joints": 1, "name": "forced-cpu"}],
        include_cpu_comparison=True,
    )

    assert "cpu_time" not in results["forced-cpu"]
    assert "actual_speedup" not in results["forced-cpu"]
    assert results["forced-cpu"]["used_gpu"] is False


# --- JAX dtype matrix and construction edge paths ---------------------------
@requires_jax
@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda b, x: b.all(x), id="all"),
        pytest.param(lambda b, x: b.any(x), id="any"),
        pytest.param(lambda b, x: b.argmax(x), id="argmax"),
        pytest.param(lambda b, x: b.clip(x, -0.5, 0.5), id="clip"),
    ],
)
def test_jax_complex_operations_match_numpy(operation):
    """Complex input follows NumPy semantics, not JAX's stricter ones.

    NumPy truth-tests a complex value on BOTH parts and orders complex values
    lexicographically by ``(real, imag)``. JAX tests the real part alone -- so
    ``1j`` reads as False -- and refuses to order complex values at all, raising
    from ``argmax`` and ``clip``. The Torch backend already delegates these to
    NumPy; the JAX backend must agree with both.
    """
    backend = be.get_registered("jax")
    numpy_be = NumpyBackend()
    values = np.array([1j, 2 + 0j, 0j])

    expected = numpy_be.to_numpy(operation(numpy_be, numpy_be.asarray(values)))
    observed = backend.to_numpy(operation(backend, backend.asarray(values)))

    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


@requires_jax
def test_jax_construction_edge_paths():
    """Construction paths that no production call site currently exercises.

    ``None`` bounds, Python-list operands, byte-swapped NumPy input (which JAX
    rejects outright) and empty sequences all reach the backend through the
    protocol, so they are pinned even though ManipulaPy does not produce them
    today.
    """
    backend = be.get_registered("jax")

    # A ``None`` clip bound means "unbounded on that side".
    np.testing.assert_allclose(
        backend.to_numpy(backend.clip(backend.asarray([-2.0, 2.0]), None, 1.0)),
        [-2.0, 1.0],
    )
    np.testing.assert_allclose(
        backend.to_numpy(backend.clip(backend.asarray([-2.0, 2.0]), -1.0, None)),
        [-1.0, 2.0],
    )

    # Python lists infer their dtype exactly as NumPy does.
    assert backend.to_numpy(backend.asarray([1, 2, 3])).dtype == np.int64
    assert backend.to_numpy(backend.asarray([1.0, 2.0])).dtype == np.float64

    # Byte-swapped input is converted rather than rejected.
    swapped = np.array([1.0, 2.0]).astype(">f8")
    np.testing.assert_allclose(
        backend.to_numpy(backend.asarray(swapped)), [1.0, 2.0]
    )

    # Empty sequences raise JAX's own error rather than an IndexError from the
    # promotion helper reaching into an empty list.
    with pytest.raises(Exception):
        backend.stack([])
    with pytest.raises(Exception):
        backend.concatenate([])


class TestBrokenOptionalBackendImport:
    """A backend that is installed but not importable must fail clearly.

    ``find_spec`` only proves a module can be *located*, not that it can be
    imported. An accelerator compiled against a different NumPy ABI resolves
    fine and then explodes on import, so the registry must distinguish
    "not installed" from "installed but broken" instead of surfacing a raw
    ImportError from deep inside a third-party dependency.
    """

    @pytest.mark.parametrize(
        "backend_name, module_name",
        [("jax", "jax_backend"), ("torch", "torch_backend"), ("cupy", "cupy_backend")],
    )
    def test_broken_backend_reports_actionable_error(
        self, backend_name, module_name, monkeypatch
    ):
        import importlib.util as _iu

        from ManipulaPy import backend as backend_pkg

        # Registry is process-global; make sure the real backend is not cached.
        monkeypatch.delitem(backend_pkg._REGISTRY, backend_name, raising=False)
        # Pretend the accelerator is installed...
        monkeypatch.setattr(
            _iu, "find_spec", lambda name, *a, **k: object(), raising=True
        )
        # ...but importing our adapter for it fails the way a NumPy ABI
        # mismatch fails: an ImportError raised from inside the dependency.
        monkeypatch.setitem(
            sys.modules, f"ManipulaPy.backend.{module_name}", None
        )

        with pytest.raises(ImportError) as excinfo:
            backend_pkg.get_registered(backend_name)

        message = str(excinfo.value)
        assert backend_name in message.lower()
        # Must say it IS installed but unusable, not "not installed".
        assert "not installed" not in message.lower(), message
        # Must give the user somewhere to go.
        assert "numpy" in message.lower(), message
