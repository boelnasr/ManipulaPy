#!/usr/bin/env python3
"""
Torch-backend call-site dispatch tests for the migrated modules.

Phase 1 routed the numeric modules through :class:`ArrayBackend`, but a number
of call sites kept NumPy-array idioms that torch tensors do not provide:

* ``.copy()`` -- torch tensors expose ``.clone()`` instead, so IK raised
  ``AttributeError`` immediately.
* ``.size`` -- a NumPy element count but a *method* on torch tensors, so it
  silently flowed on as a bound method where an ``int`` was expected.
* ``np.can_cast`` on a backend dtype -- raises ``TypeError`` for a
  ``torch.dtype``, which the CPU forward-dynamics loop's broad handler turned
  into silently zeroed acceleration.

These tests run the real migrated code paths under ``use_backend("torch")`` and
compare against the NumPy backend wherever a value comparison is meaningful.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import types

import numpy as np
import pytest

from ManipulaPy.backend import get_backend, use_backend
from ManipulaPy.control import ManipulatorController
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.kinematics.ik_helpers import _clip_to_limits
from ManipulaPy.planning.trajectory_planning import OptimizedTrajectoryPlanning
from ManipulaPy.utils import extract_screw_list


def _real_torch_available() -> bool:
    """True only when the *real* PyTorch package is importable.

    The suite's conftest installs a lightweight ``torch`` stand-in in
    ``sys.modules`` when PyTorch is absent, so ``importorskip`` would wrongly
    succeed on a base install. The stand-in is a plain object rather than a
    real module, so an import plus a module-type check distinguishes it from
    genuine PyTorch (matching ``tests/test_backend_dispatch.py``).
    """
    try:
        import torch
    except Exception:
        return False
    return isinstance(torch, types.ModuleType)


_HAS_TORCH = _real_torch_available()
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed")


# --- shared 3-DoF fixture geometry -----------------------------------------
_SLIST = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.3, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.4, -0.1],
    ],
    dtype=np.float64,
).T
_MLIST = np.array(
    [
        [1.0, 0.0, 0.0, 0.7],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_JOINT_LIMITS = [(-3.0, 3.0)] * 3
_GLIST = [np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0]) for _ in range(3)]


def _robot() -> SerialManipulator:
    return SerialManipulator(
        M_list=_MLIST,
        omega_list=_SLIST[:3, :],
        S_list=_SLIST,
        B_list=_SLIST.copy(),
        joint_limits=_JOINT_LIMITS,
    )


def _dynamics() -> ManipulatorDynamics:
    return ManipulatorDynamics(
        M_list=_MLIST,
        omega_list=_SLIST[:3, :],
        r_list=None,
        b_list=None,
        S_list=_SLIST,
        B_list=_SLIST.copy(),
        Glist=_GLIST,
    )


def _to_host(value):
    """Host NumPy view of a backend-native array."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# --- gap 1: ``.copy()`` on backend arrays ----------------------------------
@requires_torch
def test_iterative_inverse_kinematics_runs_under_torch():
    """The documented IK entry point solves under torch, matching NumPy.

    Before the fix ``best_theta = theta.copy()`` raised
    ``AttributeError: 'Tensor' object has no attribute 'copy'`` on the very
    first statement after the seed conversion, so IK was unusable under torch.
    """
    robot = _robot()
    target = np.array([0.2, -0.3, 0.4])
    T_desired = robot.forward_kinematics(target, frame="space")
    seed = np.array([0.15, -0.25, 0.35])

    theta_np, success_np, _ = robot.iterative_inverse_kinematics(T_desired, seed)
    with use_backend("torch"):
        theta_t, success_t, _ = robot.iterative_inverse_kinematics(T_desired, seed)

    assert success_np and success_t
    np.testing.assert_allclose(_to_host(theta_t), np.asarray(theta_np), atol=1e-8)


@requires_torch
def test_iterative_inverse_kinematics_backtracking_runs_under_torch():
    """The backtracking branch also survives torch.

    Exercises the ``clip_to_limits`` copy (``th.copy()``) on every line-search
    candidate plus the weighted-error copy (``V_err.copy()``).
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(np.array([0.1, 0.2, -0.2]), frame="space")
    seed = np.array([0.0, 0.1, -0.1])

    theta_np, success_np, _ = robot.iterative_inverse_kinematics(
        T_desired, seed, backtracking=True, adaptive_tuning=True
    )
    with use_backend("torch"):
        theta_t, success_t, _ = robot.iterative_inverse_kinematics(
            T_desired, seed, backtracking=True, adaptive_tuning=True
        )

    assert success_np and success_t
    np.testing.assert_allclose(_to_host(theta_t), np.asarray(theta_np), atol=1e-8)


@requires_torch
def test_clip_to_limits_accepts_a_backend_array():
    """``ik_helpers._clip_to_limits`` clips a torch tensor.

    ``trac_ik``'s ``_clip_to_limits`` forwards backend-native joint vectors
    here, so the NumPy-only ``theta.copy()`` broke that path under torch.
    """
    import torch

    limits = [(-1.0, 1.0), (-0.5, 0.5), (None, 2.0)]
    theta = torch.tensor([2.0, -3.0, 5.0], dtype=torch.float64)

    with use_backend("torch"):
        clipped = _clip_to_limits(theta, limits)

    np.testing.assert_allclose(_to_host(clipped), np.array([1.0, -0.5, 2.0]))
    # The caller's array must not be mutated by the clip.
    np.testing.assert_allclose(_to_host(theta), np.array([2.0, -3.0, 5.0]))


@requires_torch
def test_trac_ik_solver_runs_under_torch():
    """The TRAC-IK DLS path solves under torch.

    Covers the ``theta.copy()`` seed copy and best-solution tracking inside
    ``_dls_solver`` plus the ``theta0.copy()`` in ``_generate_initial_guesses``.
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(np.array([0.25, -0.1, 0.3]), frame="space")

    # trac_ik is wall-clock budgeted, and on a GPU-resident backend the first
    # call spends most of the 0.2 s default on device warm-up rather than on
    # iterations. Once warm it converges in ~40 ms, so widen the budget here
    # instead of asserting a default that measures warm-up latency; the library
    # default is deliberately unchanged.
    with use_backend("torch"):
        theta, success, _ = robot.trac_ik(
            T_desired, np.array([0.2, -0.05, 0.25]), timeout=2.0
        )

    assert success
    T_reached = robot.forward_kinematics(_to_host(theta), frame="space")
    np.testing.assert_allclose(T_reached[:3, 3], T_desired[:3, 3], atol=1e-3)


# --- gap 2: ``.size`` as a NumPy element count -----------------------------
@requires_torch
def test_extract_screw_list_accepts_documented_list_input_under_torch():
    """The documented list call works under torch and matches NumPy.

    ``b.asarray([0, 0, 1])`` yields a tensor whose ``.size`` is a *method*, so
    ``r_list.size % 3`` raised ``TypeError: unsupported operand type(s) for %:
    'builtin_function_or_method' and 'int'``.
    """
    expected = extract_screw_list([0, 0, 1], [1, 0, 0])
    with use_backend("torch"):
        actual = extract_screw_list([0, 0, 1], [1, 0, 0])

    np.testing.assert_allclose(_to_host(actual), np.asarray(expected))


@requires_torch
def test_extract_screw_list_empty_r_list_under_torch():
    """The empty-``r_list`` branch reads an element count, not a bound method."""
    omega = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    expected = extract_screw_list(omega, [])
    with use_backend("torch"):
        actual = extract_screw_list(omega, [])

    np.testing.assert_allclose(_to_host(actual), np.asarray(expected))


@requires_torch
def test_extract_screw_list_rejects_ragged_list_under_torch():
    """A non-multiple-of-three flat input still raises the documented ValueError."""
    with use_backend("torch"):
        with pytest.raises(ValueError, match="into \\(3, n\\) format"):
            extract_screw_list([0, 0, 1, 1], [1, 0, 0])


@requires_torch
def test_adaptive_control_runs_under_torch():
    """``adaptive_control`` sizes its parameter estimate from an element count.

    ``n = thetalist.size`` bound a *method* under torch, which surfaced as
    ``TypeError: zeros() received an invalid combination of arguments`` inside
    the backend once ``n`` reached ``backend.zeros((n,), ...)``.
    """
    args = (
        np.array([0.1, 0.2, 0.3]),
        np.zeros(3),
        np.zeros(3),
        np.array([0.0, 0.0, -9.81]),
        np.zeros(6),
        np.zeros(3),
        np.array([0.1]),
    )

    expected = ManipulatorController(_dynamics()).adaptive_control(*args)
    with use_backend("torch"):
        actual = ManipulatorController(_dynamics()).adaptive_control(*args)

    assert _to_host(actual).shape == (3,)
    np.testing.assert_allclose(_to_host(actual), np.asarray(expected), atol=1e-8)


# --- gap 3: dtype-capability check in the CPU forward-dynamics loop --------
class _TauForwardDynamics:
    """Deterministic dynamics: joint acceleration equals applied torque.

    Returns a backend-native array, matching ``ManipulatorDynamics``, whose
    ``forward_dynamics`` ends in ``backend.solve``. A host array here would
    mix domains with the integrator's device-resident state under a
    CUDA-bound backend and fail the add rather than exercise the dtype check.
    """

    def forward_dynamics(self, theta, dtheta, tau, g, Ftip):
        return get_backend().asarray(np.asarray(tau, dtype=np.float64))


def _dynamics_planner(dynamics, joint_limits):
    """Force-CPU planner wired to a deterministic dynamics stub."""
    return OptimizedTrajectoryPlanning(
        None, "nonexistent.urdf", dynamics, joint_limits, None, use_cuda=False
    )


@requires_torch
def test_forward_dynamics_cpu_produces_nonzero_acceleration_under_torch():
    """The CPU Euler loop integrates under torch instead of zeroing out.

    ``np.can_cast(torch.float64, ...)`` raised ``TypeError: Cannot interpret
    'torch.float64' as a data type``; the loop's broad ``except Exception``
    swallowed it and substituted zero acceleration for *every* step, silently
    producing a frozen trajectory rather than failing.
    """
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)])
    thetalist = np.zeros(2, dtype=np.float64)
    dthetalist = np.zeros(2, dtype=np.float64)
    taumat = np.tile(np.array([1.5, -0.5]), (4, 1))
    g = np.zeros(3)
    Ftipmat = np.zeros((4, 6))

    expected = planner.forward_dynamics_trajectory(
        thetalist, dthetalist, taumat, g, Ftipmat, 0.1, 2
    )
    with use_backend("torch"):
        actual = planner.forward_dynamics_trajectory(
            thetalist, dthetalist, taumat, g, Ftipmat, 0.1, 2
        )

    # The bug's signature: every acceleration row zero and the state frozen.
    assert np.any(np.abs(_to_host(actual["accelerations"])[1:]) > 0.0)
    assert np.any(np.abs(_to_host(actual["positions"])[1:]) > 0.0)

    for key in ("positions", "velocities", "accelerations"):
        np.testing.assert_allclose(
            _to_host(actual[key]), np.asarray(expected[key]), rtol=1e-6, atol=1e-6
        )


@requires_torch
def test_forward_dynamics_cpu_integer_state_freezes_under_torch():
    """An integer state still refuses the float update under torch.

    The same-kind cast refusal that NumPy's in-place ``+=`` raised must survive
    the backend-aware dtype check: the state stays frozen at its initial value
    with zero acceleration, matching the NumPy backend exactly.
    """
    planner = _dynamics_planner(_TauForwardDynamics(), [(-5.0, 5.0), (-5.0, 5.0)])
    thetalist = np.array([1, 2], dtype=np.int64)
    dthetalist = np.array([0, 0], dtype=np.int64)
    taumat = np.full((3, 2), 4.0, dtype=np.float64)

    with use_backend("torch"):
        result = planner.forward_dynamics_trajectory(
            thetalist, dthetalist, taumat, np.zeros(3), np.zeros((3, 6)), 0.5, 1
        )

    np.testing.assert_array_equal(
        _to_host(result["positions"]), np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        _to_host(result["accelerations"]), np.zeros((3, 2), dtype=np.float32)
    )


def test_forward_dynamics_cpu_does_not_mask_a_dynamics_bookkeeping_error():
    """A failure in the loop's own integration bookkeeping is not zeroed out.

    Only the dynamics *call* is tolerated. A broken dtype-capability check (or
    any other integration-side error) must propagate instead of being silently
    converted into a zero-acceleration trajectory.
    """

    class _BadArrayDynamics:
        """Returns an object whose arithmetic fails inside the integrator."""

        def forward_dynamics(self, theta, dtheta, tau, g, Ftip):
            class _Hostile(np.ndarray):
                def __rmul__(self, other):
                    raise RuntimeError("integration bookkeeping is broken")

                __mul__ = __rmul__

            return np.zeros(2).view(_Hostile)

    planner = _dynamics_planner(_BadArrayDynamics(), [(-5.0, 5.0), (-5.0, 5.0)])

    with pytest.raises(RuntimeError, match="integration bookkeeping is broken"):
        planner.forward_dynamics_trajectory(
            np.zeros(2), np.zeros(2), np.zeros((3, 2)), np.zeros(3), np.zeros((3, 6)),
            0.1, 1,
        )
