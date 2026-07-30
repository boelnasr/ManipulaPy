#!/usr/bin/env python3
"""
JAX-backend call-site dispatch tests for the migrated modules.

Phase 1 routed the numeric modules through :class:`ArrayBackend` and Phase 2
made the core math trace-safe for Torch. JAX adds one constraint neither of
those exercised: **its arrays are immutable**, so every ``arr[i] = value``
idiom raises ``TypeError`` rather than quietly doing the wrong thing. The IK
modules built their joint vectors that way:

* ``workspace_heuristic_guess`` / ``random_in_limits`` / ``midpoint_of_limits``
  filled a preallocated vector joint by joint.
* ``_clip_to_limits`` and ``iterative_inverse_kinematics``'s inner
  ``clip_to_limits`` clipped element by element.
* ``iterative_inverse_kinematics`` built a one-hot rotation axis and scaled the
  two halves of the weighted twist in place.
* ``TracIKSolver``'s near-pi branch built a corrected one-hot axis in place.

These tests run the real migrated code paths under ``use_backend("jax")`` and
compare against the NumPy backend wherever a value comparison is meaningful.
They also pin the two host-boundary contracts JAX makes load-bearing: the
NumPy-in/NumPy-out path of ``_clip_to_limits`` must not be routed through the
backend, and the SLSQP/ConvexHull boundaries must not leak a JAX array into
host code.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import threading
import types
from unittest import mock

import numpy as np
import pytest
import scipy.optimize

from ManipulaPy.backend import use_backend
from ManipulaPy.control import ManipulatorController
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.kinematics.ik_helpers import (
    _clip_to_limits,
    midpoint_of_limits,
    random_in_limits,
    workspace_heuristic_guess,
)
from ManipulaPy.kinematics.trac_ik import TracIKSolver
from ManipulaPy.singularity import Singularity
from ManipulaPy.utils import extract_screw_list


def _real_jax_available() -> bool:
    """True only when the *real* JAX package is importable.

    Mirrors ``tests/test_backend_dispatch.py``: conftest installs stand-ins for
    absent optional dependencies, so an import plus a module-type check is used
    rather than ``find_spec``.
    """
    try:
        import jax
    except Exception:
        return False
    return isinstance(jax, types.ModuleType)


_HAS_JAX = _real_jax_available()
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX is not installed")


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

# JAX dispatches eagerly op by op, which costs roughly forty times more per
# call than NumPy on problems this small. TracIKSolver.solve is time-budgeted
# (200 ms by default), so JAX needs a wider budget to reach the same answer;
# this is a dispatch-overhead characteristic, not a solver difference.
_JAX_SOLVE_TIMEOUT = 10.0


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


def _trac_ik(robot: SerialManipulator) -> TracIKSolver:
    return TracIKSolver(
        fk_func=lambda t: robot.forward_kinematics(t, frame="space"),
        jacobian_func=lambda t: robot.jacobian(t, frame="space"),
        joint_limits=_JOINT_LIMITS,
        n_joints=3,
    )


_THETA = np.array([0.2, -0.3, 0.4])


# --- immutability: vectors that were filled in place ------------------------
@requires_jax
def test_workspace_heuristic_guess_builds_a_vector_under_jax():
    """The heuristic guess assembles its joint vector without in-place writes.

    Before the fix ``theta[0] = backend.arctan2(...)`` raised ``TypeError:
    JAX arrays are immutable and do not support in-place item assignment`` on
    the first joint, so every IK entry point that seeds from the heuristic was
    unusable under JAX.
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(_THETA, frame="space")

    expected = np.asarray(workspace_heuristic_guess(T_desired, 3, _JOINT_LIMITS))
    with use_backend("jax"):
        observed = np.asarray(workspace_heuristic_guess(T_desired, 3, _JOINT_LIMITS))

    np.testing.assert_allclose(observed, expected, rtol=1e-12)
    assert observed.dtype == np.float64


@requires_jax
def test_workspace_heuristic_guess_keeps_float64_for_a_float32_pose():
    """A float32 pose still yields a float64 guess.

    The pre-fix code wrote into a float64 vector, so the result was float64
    whatever the pose dtype. Assembling by ``stack`` would otherwise narrow to
    float32, which the explicit cast prevents.
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(_THETA, frame="space")
    T32 = np.asarray(T_desired, dtype=np.float32)

    with use_backend("jax"):
        guess = workspace_heuristic_guess(T32, 3, _JOINT_LIMITS)

    assert np.asarray(guess).dtype == np.float64


@requires_jax
def test_random_and_midpoint_guesses_build_vectors_under_jax():
    """The remaining seed generators assemble rather than fill in place."""
    with use_backend("jax"):
        midpoint = midpoint_of_limits(_JOINT_LIMITS)
        rand = random_in_limits(_JOINT_LIMITS)

    midpoint = np.asarray(midpoint)
    rand = np.asarray(rand)

    np.testing.assert_allclose(midpoint, np.zeros(3), atol=1e-12)
    assert midpoint.dtype == np.float64 and rand.dtype == np.float64
    assert rand.shape == (3,)
    assert np.all(rand >= -3.0) and np.all(rand <= 3.0)


@requires_jax
def test_midpoint_of_limits_leaves_open_limits_at_zero():
    """A ``(None, None)`` limit contributes 0.0, as the in-place version did."""
    limits = [(-2.0, 4.0), (None, None), (None, 1.0)]
    with use_backend("jax"):
        observed = np.asarray(midpoint_of_limits(limits))

    np.testing.assert_allclose(observed, [1.0, 0.0, 0.0], rtol=1e-12)


# --- immutability: clipping -------------------------------------------------
@requires_jax
def test_clip_to_limits_accepts_a_backend_array_under_jax():
    """``_clip_to_limits`` clips a JAX vector without element-wise assignment.

    Before the fix this raised ``TypeError`` on the immutable array.
    """
    import jax.numpy as jnp

    with use_backend("jax"):
        clipped = _clip_to_limits(jnp.asarray([5.0, -5.0, 0.5]), _JOINT_LIMITS)

    np.testing.assert_allclose(np.asarray(clipped), [3.0, -3.0, 0.5], rtol=1e-12)


@requires_jax
def test_clip_to_limits_keeps_a_host_numpy_seed_on_the_host():
    """A NumPy input returns NumPy even while the JAX backend is active.

    ``IKInitialGuessCache`` looks up host arrays and must keep them host-side;
    routing that path through the backend would hand a JAX array back to code
    that expects NumPy. The vectorised rewrite keeps the type dispatch.
    """
    with use_backend("jax"):
        clipped = _clip_to_limits(np.array([5.0, -5.0, 0.5]), _JOINT_LIMITS)

    assert isinstance(clipped, np.ndarray)
    np.testing.assert_allclose(clipped, [3.0, -3.0, 0.5], rtol=1e-12)


@requires_jax
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_clip_to_limits_preserves_the_input_dtype(dtype):
    """Clipping does not widen the vector it was given.

    The replaced code assigned into a copy of ``theta``, so the result kept
    ``theta``'s dtype and an integer vector truncated the bound. The +/-inf
    bound arrays are float64 and would otherwise widen both, which is reachable
    through float32 TRAC-IK seeds and ``IKInitialGuessCache.get_nearest``.
    """
    import jax.numpy as jnp

    limits = [(-1.25, 1.25), (None, None)]
    expected = np.array([1.25, 3.0]).astype(dtype)

    with use_backend("jax"):
        backend_out = _clip_to_limits(jnp.asarray(np.array([2, 3], dtype=dtype)), limits)
        host_out = _clip_to_limits(np.array([2, 3], dtype=dtype), limits)

    assert np.asarray(backend_out).dtype == dtype
    assert host_out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(backend_out), expected)
    np.testing.assert_array_equal(host_out, expected)


@requires_jax
def test_pinv_float32_cutoff_matches_numpy():
    """A float32 singular value on NumPy's rcond cutoff is kept, not zeroed.

    NumPy evaluates ``s > rcond * max(s)`` in float64 even for float32 input;
    JAX compares in the input precision, so a value landing exactly on the
    cutoff was dropped where NumPy retains it.
    """
    import jax.numpy as jnp

    from ManipulaPy.backend import get_registered

    matrix = np.zeros((6, 2), dtype=np.float32)
    matrix[3, 0] = 1.0
    matrix[4, 1] = np.float32(1e-15)
    probe = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)

    expected = np.asarray(get_registered("numpy").pinv(matrix)) @ probe
    observed = np.asarray(get_registered("jax").pinv(jnp.asarray(matrix))) @ probe

    np.testing.assert_allclose(observed, expected, rtol=1e-6)


@requires_jax
def test_clip_to_limits_ignores_limits_beyond_the_vector():
    """Extra ``joint_limits`` entries and absent bounds leave values untouched.

    The replaced loop guarded on ``i < len(theta)`` and skipped ``None``
    bounds; the +/-inf padding must reproduce both.
    """
    import jax.numpy as jnp

    limits = [(None, 1.0), (-1.0, None), (None, None), (0.0, 0.0)]
    with use_backend("jax"):
        clipped = np.asarray(_clip_to_limits(jnp.asarray([5.0, -5.0, 9.0]), limits))

    np.testing.assert_allclose(clipped, [1.0, -1.0, 9.0], rtol=1e-12)


# --- immutability: the IK iteration itself ---------------------------------
@requires_jax
def test_iterative_inverse_kinematics_runs_under_jax():
    """The documented IK entry point solves under JAX, matching NumPy.

    Exercises three former in-place idioms at once: the inner
    ``clip_to_limits``, the one-hot rotation axis on the near-pi branch, and
    the two in-place slice scalings of the weighted twist.
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(_THETA, frame="space")
    seed = np.array([0.15, -0.25, 0.35])

    theta_np, success_np, _ = robot.iterative_inverse_kinematics(T_desired, seed)
    with use_backend("jax"):
        theta_jax, success_jax, _ = robot.iterative_inverse_kinematics(T_desired, seed)

    assert success_np and success_jax
    np.testing.assert_allclose(np.asarray(theta_jax), np.asarray(theta_np), rtol=1e-6)


@requires_jax
def test_iterative_inverse_kinematics_weighted_twist_scales_both_halves():
    """Orientation and position weights still scale their own half.

    The replaced code scaled ``V_weighted[:3]`` and ``V_weighted[3:]``
    separately in place. A weight vector built in the wrong order would steer
    the step differently, so an asymmetric weighting is used to make the halves
    distinguishable while staying in the regime where both backends converge.
    """
    robot = _robot()
    T_desired = robot.forward_kinematics(_THETA, frame="space")
    seed = np.array([0.15, -0.25, 0.35])

    kwargs = dict(weight_orientation=1.0, weight_position=2.0)
    theta_np, ok_np, _ = robot.iterative_inverse_kinematics(T_desired, seed, **kwargs)
    with use_backend("jax"):
        theta_jax, ok_jax, _ = robot.iterative_inverse_kinematics(
            T_desired, seed, **kwargs
        )

    assert ok_np and ok_jax
    np.testing.assert_allclose(np.asarray(theta_jax), np.asarray(theta_np), rtol=1e-6)


@requires_jax
@pytest.mark.parametrize("index", [0, 1, 2])
def test_identity_row_reproduces_the_one_hot_axis(index):
    """``eye(3)[idx]`` is the vector the replaced ``axis[idx] = 1.0`` built.

    ``iterative_inverse_kinematics`` only takes its one-hot branch when the
    rotation error is within 1e-6 of pi, which a full solve cannot be steered
    onto reliably, so the substitution itself is pinned here. The equivalent
    branch that *is* reachable through a public entry point is covered by
    :func:`test_trac_ik_half_turn_axis_matches_numpy`.
    """
    from ManipulaPy.backend import get_registered

    expected = np.zeros(3)
    expected[index] = 1.0

    with use_backend("jax"):
        backend = get_registered("jax")
        observed = backend.eye(3, dtype=backend.float64)[index]

    np.testing.assert_array_equal(np.asarray(observed), expected)


# --- TRAC-IK: immutability plus the SLSQP host boundary --------------------
@requires_jax
def test_trac_ik_dls_solver_matches_numpy_under_jax():
    """The DLS solver reaches the same joint vector under both backends.

    Called directly with an explicit budget so the comparison measures the
    solver rather than JAX's per-op dispatch overhead.
    """
    robot = _robot()
    solver = _trac_ik(robot)
    T_desired = robot.forward_kinematics(_THETA, frame="space")
    seed = np.array([0.15, -0.25, 0.35])

    theta_np, ok_np, _ = solver._dls_solver(
        T_desired, seed, 2e-3, 2e-3, _JAX_SOLVE_TIMEOUT, threading.Event()
    )
    with use_backend("jax"):
        theta_jax, ok_jax, _ = solver._dls_solver(
            T_desired, seed, 2e-3, 2e-3, _JAX_SOLVE_TIMEOUT, threading.Event()
        )

    assert ok_np and ok_jax
    np.testing.assert_allclose(np.asarray(theta_jax), np.asarray(theta_np), rtol=1e-6)


@requires_jax
def test_trac_ik_solve_runs_through_the_slsqp_boundary_under_jax():
    """``solve`` converges under JAX without leaking an array into SciPy.

    SLSQP is a host boundary: the seed is converted with ``to_numpy`` on the
    way in and ``result.x`` is normalised back onto the backend. A leak would
    surface as SciPy raising, or as the broad handler silently returning the
    untouched seed.
    """
    robot = _robot()
    solver = _trac_ik(robot)
    T_desired = robot.forward_kinematics(_THETA, frame="space")
    seed = np.array([0.15, -0.25, 0.35])

    captured = {}
    real_minimize = scipy.optimize.minimize

    def spy(fun, x0, *args, **kwargs):
        captured["x0"] = x0
        jac = kwargs.get("jac")
        if jac is not None:
            captured["grad"] = jac(x0)
        return real_minimize(fun, x0, *args, **kwargs)

    # `solve` races DLS against SQP and DLS wins on this fixture, so calling it
    # would never touch SciPy. The SQP path is invoked directly instead.
    with use_backend("jax"):
        with mock.patch.object(scipy.optimize, "minimize", spy):
            theta, success, err = solver._sqp_solver(
                T_desired, seed, 1e-4, 1e-4, _JAX_SOLVE_TIMEOUT, threading.Event()
            )

    assert "x0" in captured, "SLSQP was never reached"
    # SciPy must receive host arrays, never a JAX array.
    assert type(captured["x0"]) is np.ndarray, f"leaked {type(captured['x0'])} as x0"
    assert type(captured["grad"]) is np.ndarray, f"leaked {type(captured['grad'])}"

    # The SQP fallback does not converge on this fixture -- it stalls at the
    # seed under NumPy too, identically -- so what is pinned here is the
    # boundary contract and backend agreement, not convergence. Asserting
    # success would be asserting a pre-existing solver weakness.
    theta_np, success_np, err_np = solver._sqp_solver(
        T_desired, seed, 1e-4, 1e-4, _JAX_SOLVE_TIMEOUT, threading.Event()
    )
    assert success is success_np
    np.testing.assert_allclose(np.asarray(theta), np.asarray(theta_np), atol=1e-5)
    np.testing.assert_allclose(err, err_np, rtol=1e-5)


# Half-turn axes driving each possible ``argmax`` of the rotation-error
# diagonal. The coordinate axes alone are NOT sufficient: for them every
# off-diagonal numerator R_err[k, j] is zero, so a rewrite that simply zeroed
# the non-k components would reproduce them exactly. The oblique axes give
# non-zero numerators, so the quotient itself is under test. Each is a unit
# vector whose largest component sits at a different index.
_OBLIQUE = (0.8, 0.5, 0.331662479)
_HALF_TURN_AXES = [
    ([1.0, 0.0, 0.0], "x"),
    ([0.0, 1.0, 0.0], "y"),
    ([0.0, 0.0, 1.0], "z"),
    ([_OBLIQUE[0], _OBLIQUE[1], _OBLIQUE[2]], "oblique-k0"),
    ([_OBLIQUE[1], _OBLIQUE[0], _OBLIQUE[2]], "oblique-k1"),
    ([_OBLIQUE[1], _OBLIQUE[2], _OBLIQUE[0]], "oblique-k2"),
]


@requires_jax
@pytest.mark.parametrize(
    "axis", [pytest.param(a, id=i) for a, i in _HALF_TURN_AXES]
)
def test_trac_ik_half_turn_axis_matches_numpy(axis):
    """The rewritten near-pi axis reconstruction is numerically unchanged.

    The three former ``k == 0/1/2`` branches collapsed to one rule: component
    ``k`` is 1 and each other component ``j`` is
    ``R_err[k, j] / (1 + R_err[k, k])``. Feeding the error function an *exact*
    half turn lands on that branch deterministically (a full solve cannot be
    steered onto it, since it needs the rotation error within 1e-6 of pi), and
    the axes chosen drive every possible ``k``.

    The recovered rotation vector is checked against ``pi * n`` rather than
    against the other backend alone, so a rewrite that was wrong the same way
    everywhere still fails.
    """
    solver = _trac_ik(_robot())

    n = np.asarray(axis, dtype=float)
    n /= np.linalg.norm(n)
    R = 2.0 * np.outer(n, n) - np.eye(3)  # pi rotation about n
    T_current = np.eye(4)
    T_desired = np.eye(4)
    T_desired[:3, :3] = R

    V_np, rot_np, trans_np = solver._default_error_func(T_current, T_desired)
    with use_backend("jax"):
        V_jax, rot_jax, trans_jax = solver._default_error_func(T_current, T_desired)

    # The half turn puts the rotation error on the branch under test...
    np.testing.assert_allclose(float(rot_np), np.pi, rtol=1e-9)
    # ...and the recovered rotation vector is pi * n, up to the branch's sign
    # convention (the axis is fixed only up to sign at a half turn).
    omega_np = np.asarray(V_np)[:3]
    sign = np.sign(np.dot(omega_np, n)) or 1.0
    np.testing.assert_allclose(omega_np, sign * np.pi * n, atol=1e-7)

    np.testing.assert_allclose(np.asarray(V_jax), np.asarray(V_np), atol=1e-9)
    np.testing.assert_allclose(float(rot_jax), float(rot_np), rtol=1e-12)
    np.testing.assert_allclose(float(trans_jax), float(trans_np), atol=1e-12)


# --- core math parity -------------------------------------------------------
@requires_jax
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda r, d: r.forward_kinematics(_THETA, frame="space"), id="fk"),
        pytest.param(lambda r, d: r.jacobian(_THETA, frame="space"), id="jacobian"),
        pytest.param(lambda r, d: d.mass_matrix(_THETA), id="mass_matrix"),
        pytest.param(
            lambda r, d: d.gravity_forces(_THETA, np.array([0.0, 0.0, -9.81])),
            id="gravity_forces",
        ),
        pytest.param(
            lambda r, d: d.velocity_quadratic_forces(
                _THETA, np.array([0.1, 0.2, -0.1])
            ),
            id="velocity_quadratic_forces",
        ),
        pytest.param(
            lambda r, d: d.inverse_dynamics(
                _THETA,
                np.array([0.1, 0.2, -0.1]),
                np.array([0.05, -0.05, 0.1]),
                np.array([0.0, 0.0, -9.81]),
                np.zeros(6),
            ),
            id="inverse_dynamics",
        ),
        pytest.param(
            lambda r, d: d.forward_dynamics(
                _THETA,
                np.array([0.1, 0.2, -0.1]),
                np.array([1.0, 0.5, -0.3]),
                np.array([0.0, 0.0, -9.81]),
                np.zeros(6),
            ),
            id="forward_dynamics",
        ),
    ],
)
def test_core_math_matches_numpy_under_jax(call):
    """Core math produces NumPy-equal values under the JAX backend."""
    robot, dyn = _robot(), _dynamics()

    expected = np.asarray(call(robot, dyn))
    with use_backend("jax"):
        observed = np.asarray(call(robot, dyn))

    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-10)


# --- remaining in-scope modules --------------------------------------------
@requires_jax
def test_extract_screw_list_matches_numpy_under_jax():
    """The screw-axis builder dispatches cleanly under JAX."""
    omega = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    r_list = np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]])

    expected = np.asarray(extract_screw_list(omega, r_list))
    with use_backend("jax"):
        observed = np.asarray(extract_screw_list(omega, r_list))

    np.testing.assert_allclose(observed, expected, rtol=1e-12)


@requires_jax
def test_adaptive_control_runs_under_jax():
    """The adaptive controller runs on backend-native JAX input."""
    import jax.numpy as jnp

    controller = ManipulatorController(_dynamics())
    with use_backend("jax"):
        tau = controller.adaptive_control(
            thetalist=jnp.asarray(_THETA),
            dthetalist=jnp.asarray([0.1, 0.2, -0.1]),
            ddthetalist=jnp.zeros(3),
            g=jnp.asarray([0.0, 0.0, -9.81]),
            Ftip=jnp.zeros(6),
            measurement_error=jnp.zeros(3),
            adaptation_gain=0.1,
        )

    tau = np.asarray(tau)
    assert tau.shape == (3,)
    assert np.isfinite(tau).all()


@requires_jax
def test_condition_number_keeps_a_host_seed_on_the_host():
    """A host seed returns a host scalar even while JAX is the active backend.

    ``condition_number`` is documented as returning ``float`` and performs a
    NaN -> inf substitution that mirrors ``np.linalg.cond``. Only a caller
    passing a backend-native seed can be differentiating through it, so only
    that case stays native. Gating on ``is_concrete`` alone would hand a
    backend array back to every Torch and JAX caller passing plain NumPy, and
    would turn ``near_singularity_detection`` into a non-``bool``.
    """
    analysis = Singularity(_robot())

    with use_backend("jax"):
        value = analysis.condition_number(_THETA)
        flag = analysis.near_singularity_detection(_THETA)

    assert isinstance(value, np.floating), f"host seed returned {type(value)}"
    assert isinstance(flag, (bool, np.bool_)), f"host seed returned {type(flag)}"
    np.testing.assert_allclose(
        float(value), float(Singularity(_robot()).condition_number(_THETA)), rtol=1e-12
    )


@requires_jax
@pytest.mark.parametrize(
    "method", ["condition_number", "near_singularity_detection"]
)
def test_singularity_metrics_match_numpy_under_jax(method):
    """Singularity metrics agree across backends."""
    analysis = Singularity(_robot())

    expected = getattr(analysis, method)(_THETA)
    with use_backend("jax"):
        observed = getattr(analysis, method)(_THETA)

    np.testing.assert_allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-8,
    )


@requires_jax
@pytest.mark.parametrize(
    "axis", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
def test_iterative_inverse_kinematics_exact_half_turn_branch(axis):
    """The exact-half-turn branch of the IK error runs and agrees across backends.

    ``iterative_inverse_kinematics`` takes its one-hot axis branch only when the
    rotation error is within 1e-6 of pi, which a randomly seeded solve never
    reaches. Posing the target as exactly a half turn from the seed's own pose
    puts the very first iteration on that branch. Convergence is not asserted --
    the point is that the rewritten one-hot construction executes, stays finite,
    and produces the same step on every backend.
    """
    import jax.numpy as jnp  # noqa: F401  (ensures the backend is importable)

    robot = _robot()
    seed = np.array([0.1, -0.2, 0.15])
    pose = np.asarray(robot.forward_kinematics(seed, frame="space"))

    half_turn = np.eye(4)
    half_turn[:3, :3] = 2.0 * np.outer(axis, axis) - np.eye(3)
    T_desired = pose @ half_turn

    theta_np, _, _ = robot.iterative_inverse_kinematics(
        T_desired, seed, max_iterations=3
    )
    with use_backend("jax"):
        theta_jax, _, _ = robot.iterative_inverse_kinematics(
            T_desired, seed, max_iterations=3
        )

    theta_np = np.asarray(theta_np)
    theta_jax = np.asarray(theta_jax)
    assert np.isfinite(theta_np).all() and np.isfinite(theta_jax).all()
    np.testing.assert_allclose(theta_jax, theta_np, rtol=1e-9, atol=1e-12)
