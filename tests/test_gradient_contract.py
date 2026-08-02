#!/usr/bin/env python3
"""Feasibility checks for the planned differentiable array backends.

These tests intentionally use test-local functional Torch and JAX prototypes.
They prove that the branchless core formulas can support autodiff; they do not
claim that ManipulaPy's production backend dispatch supports either framework.
"""

import math
import types

import numpy as np
import pytest

from ManipulaPy import utils
from ManipulaPy.backend import use_backend
from ManipulaPy.dynamics import ManipulatorDynamics


def _real_torch_available() -> bool:
    """True only when the *real* PyTorch package is importable.

    The suite's conftest installs a lightweight ``torch`` stand-in in
    ``sys.modules`` when PyTorch is absent, so ``importorskip`` would wrongly
    succeed on a base install and then fail the autograd calls. The stand-in is
    a plain object rather than a real module, so an import plus a module-type
    check distinguishes it from genuine PyTorch (matching the predicate in
    ``tests/test_backend_dispatch.py``).
    """
    try:
        import torch
    except Exception:
        return False
    return isinstance(torch, types.ModuleType)


_HAS_TORCH = _real_torch_available()
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch is not installed")


def _torch_or_skip():
    """Return the real PyTorch module, or skip the calling test.

    ``pytest.importorskip("torch")`` cannot be used for this. The conftest
    stand-in described in ``_real_torch_available`` makes the import *succeed*
    on a base install, so the test would proceed against the mock and fail
    with an unrelated ``TypeError`` deep inside the prototype instead of
    skipping.
    """
    if not _HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    import torch

    return torch


SCREWS = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.3, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.4, -0.1],
    ],
    dtype=np.float64,
).T
HOME = np.array(
    [
        [1.0, 0.0, 0.0, 0.7],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CONFIGURATIONS = (
    pytest.param(np.zeros(3), id="zero"),
    pytest.param(np.full(3, np.pi / 2), id="pi-over-two"),
    pytest.param(np.full(3, np.pi), id="pi"),
    pytest.param(np.array([0.31, -1.17, 2.24]), id="fixed-random-a"),
    pytest.param(np.array([-2.71, 0.83, -0.42]), id="fixed-random-b"),
)
LOG_CASES = (
    pytest.param(np.array([1.0, 0.0, 0.0]), 0.0, id="x-zero"),
    pytest.param(np.array([0.0, 0.0, 1.0]), 0.0, id="z-zero"),
    pytest.param(np.array([1.0, 0.0, 0.0]), np.pi, id="x-pi"),
    pytest.param(np.array([0.0, 0.0, 1.0]), np.pi, id="z-pi"),
)
SMOOTH_LOG_CASES = (
    pytest.param(np.array([1.0, 0.0, 0.0]), 0.2, id="x-low"),
    pytest.param(np.array([0.0, 0.0, 1.0]), np.pi / 2, id="z-mid"),
    pytest.param(
        np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0), np.pi - 0.2,
        id="arbitrary-high",
    ),
)


def _numpy_fk(q):
    """Reference product of exponentials through the production NumPy API."""
    transform = np.eye(4)
    for index, theta in enumerate(q):
        transform = transform @ utils.transform_from_twist(SCREWS[:, index], theta)
    return transform @ HOME


def _central_jacobian(function, q, step=1e-6):
    """Return a central-difference Jacobian of a flattened array function."""
    columns = []
    for index in range(q.size):
        delta = np.zeros_like(q)
        delta[index] = step
        columns.append((function(q + delta).reshape(-1) - function(q - delta).reshape(-1)) / (2 * step))
    return np.stack(columns, axis=1)


def _expected_log(axis, angle):
    """Return production NumPy axis, angle, and matrix-log values."""
    rotation = _numpy_fk_rotation(axis, angle)
    out_axis, out_angle = utils.rotation_logm(rotation)
    return np.asarray(out_axis), out_angle, utils.MatrixLog3(rotation)


def _numpy_fk_rotation(axis, angle):
    screw = np.concatenate((axis, np.zeros(3)))
    return utils.transform_from_twist(screw, angle)[:3, :3]


def _torch_functions(torch):
    """Build trace-safe Torch prototypes without production backend dispatch."""
    dtype = torch.float64
    screws = torch.as_tensor(SCREWS, dtype=dtype)
    home = torch.as_tensor(HOME, dtype=dtype)

    def skew(vector):
        zero = vector[0] * 0
        return torch.stack(
            (
                torch.stack((zero, -vector[2], vector[1])),
                torch.stack((vector[2], zero, -vector[0])),
                torch.stack((-vector[1], vector[0], zero)),
            )
        )

    def transform(screw, theta):
        omega, velocity = screw[:3], screw[3:]
        omega_hat = skew(omega)
        omega_hat2 = omega_hat @ omega_hat
        identity = torch.eye(3, dtype=dtype, device=theta.device)
        rotation = identity + torch.sin(theta) * omega_hat + (1 - torch.cos(theta)) * omega_hat2
        g_matrix = identity * theta + (1 - torch.cos(theta)) * omega_hat + (theta - torch.sin(theta)) * omega_hat2
        position = g_matrix @ velocity
        bottom = torch.stack((theta * 0, theta * 0, theta * 0, theta * 0 + 1)).reshape(1, 4)
        return torch.cat((torch.cat((rotation, position.reshape(3, 1)), dim=1), bottom), dim=0)

    def fk(configuration):
        result = torch.eye(4, dtype=dtype, device=configuration.device)
        for index in range(3):
            result = result @ transform(screws[:, index], configuration[index])
        return result @ home

    def rotation(axis, angle):
        screw = torch.cat((axis, torch.zeros(3, dtype=dtype, device=angle.device)))
        return transform(screw, angle)[:3, :3]

    def rotation_log(rotation_matrix):
        cosine = torch.clamp((torch.trace(rotation_matrix) - 1) / 2, -1.0, 1.0)
        angle = torch.arccos(cosine)
        vee = torch.stack(
            (
                rotation_matrix[2, 1] - rotation_matrix[1, 2],
                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                rotation_matrix[1, 0] - rotation_matrix[0, 1],
            )
        )
        generic = vee / torch.clamp(2 * torch.sin(angle), min=1e-12)
        candidates = (
            torch.stack((1 + rotation_matrix[0, 0], rotation_matrix[1, 0], rotation_matrix[2, 0])),
            torch.stack((rotation_matrix[0, 1], 1 + rotation_matrix[1, 1], rotation_matrix[2, 1])),
            torch.stack((rotation_matrix[0, 2], rotation_matrix[1, 2], 1 + rotation_matrix[2, 2])),
        )
        use2 = 1 + rotation_matrix[2, 2] >= 1e-6
        use1 = (1 + rotation_matrix[1, 1] >= 1e-6) & ~use2
        candidate = torch.where(use2, candidates[2], torch.where(use1, candidates[1], candidates[0]))
        half_turn = candidate / torch.clamp(torch.linalg.norm(candidate), min=1e-12)
        axis = torch.where(angle > math.pi - 1e-6, half_turn, generic)
        axis = torch.where(angle < 1e-6, torch.zeros_like(axis), axis)
        return axis, angle

    return dtype, fk, rotation, rotation_log, skew


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_torch_fk_full_jacobian_matches_finite_difference(configuration):
    """Torch feasibility prototype matches the full finite-difference FK Jacobian."""
    torch = _torch_or_skip()
    dtype, fk, _, _, _ = _torch_functions(torch)
    q = torch.tensor(configuration, dtype=dtype, requires_grad=True)

    traced_fk = torch.jit.trace(fk, q, check_trace=False)
    actual = torch.autograd.functional.jacobian(
        lambda value: traced_fk(value).reshape(-1), q
    )
    expected = _central_jacobian(_numpy_fk, configuration)

    np.testing.assert_allclose(fk(q).detach().numpy(), _numpy_fk(configuration), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("axis, angle", LOG_CASES)
def test_torch_log_singular_endpoints_have_forward_parity(axis, angle):
    """Torch feasibility keeps exact log endpoints forward-only by contract."""
    torch = _torch_or_skip()
    dtype, _, rotation, rotation_log, skew = _torch_functions(torch)
    axis_tensor = torch.tensor(axis, dtype=dtype)
    angle_tensor = torch.tensor(angle, dtype=dtype)
    out_axis, out_angle = rotation_log(rotation(axis_tensor, angle_tensor))
    expected_axis, expected_angle, expected_matrix = _expected_log(axis, angle)

    np.testing.assert_allclose(out_axis.numpy(), expected_axis, rtol=0, atol=1e-7)
    np.testing.assert_allclose(out_angle.numpy(), expected_angle, rtol=0, atol=1e-7)
    np.testing.assert_allclose((skew(out_axis * out_angle)).numpy(), expected_matrix, rtol=0, atol=1e-7)


@pytest.mark.parametrize("axis, angle", SMOOTH_LOG_CASES)
def test_torch_log_gradient_matches_finite_difference_in_smooth_interior(axis, angle):
    """Torch log gradients agree away from the identity and half-turn singularities."""
    torch = _torch_or_skip()
    dtype, _, rotation, rotation_log, _ = _torch_functions(torch)
    axis_tensor = torch.tensor(axis, dtype=dtype)
    theta = torch.tensor(angle, dtype=dtype, requires_grad=True)

    def rotvec(value):
        out_axis, out_angle = rotation_log(rotation(axis_tensor, value))
        return out_axis * out_angle

    actual = torch.autograd.functional.jacobian(rotvec, theta).detach().numpy()
    step = 1e-6
    expected = (rotvec(theta.detach() + step) - rotvec(theta.detach() - step)).numpy() / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def _jax_functions(jax, jnp):
    """Build trace-safe JAX prototypes without production backend dispatch."""
    screws = jnp.asarray(SCREWS, dtype=jnp.float64)
    home = jnp.asarray(HOME, dtype=jnp.float64)

    def skew(vector):
        zero = vector[0] * 0
        return jnp.stack(
            (
                jnp.stack((zero, -vector[2], vector[1])),
                jnp.stack((vector[2], zero, -vector[0])),
                jnp.stack((-vector[1], vector[0], zero)),
            )
        )

    def transform(screw, theta):
        omega, velocity = screw[:3], screw[3:]
        omega_hat = skew(omega)
        omega_hat2 = omega_hat @ omega_hat
        identity = jnp.eye(3, dtype=jnp.float64)
        rotation = identity + jnp.sin(theta) * omega_hat + (1 - jnp.cos(theta)) * omega_hat2
        g_matrix = identity * theta + (1 - jnp.cos(theta)) * omega_hat + (theta - jnp.sin(theta)) * omega_hat2
        top = jnp.concatenate((rotation, (g_matrix @ velocity).reshape(3, 1)), axis=1)
        return jnp.concatenate((top, jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float64)), axis=0)

    def fk(configuration):
        result = jnp.eye(4, dtype=jnp.float64)
        for index in range(3):
            result = result @ transform(screws[:, index], configuration[index])
        return result @ home

    def rotation(axis, angle):
        return transform(jnp.concatenate((axis, jnp.zeros(3, dtype=jnp.float64))), angle)[:3, :3]

    def rotation_log(rotation_matrix):
        cosine = jnp.clip((jnp.trace(rotation_matrix) - 1) / 2, -1.0, 1.0)
        angle = jnp.arccos(cosine)
        vee = jnp.stack((rotation_matrix[2, 1] - rotation_matrix[1, 2], rotation_matrix[0, 2] - rotation_matrix[2, 0], rotation_matrix[1, 0] - rotation_matrix[0, 1]))
        generic = vee / jnp.maximum(2 * jnp.sin(angle), 1e-12)
        c0 = jnp.stack((1 + rotation_matrix[0, 0], rotation_matrix[1, 0], rotation_matrix[2, 0]))
        c1 = jnp.stack((rotation_matrix[0, 1], 1 + rotation_matrix[1, 1], rotation_matrix[2, 1]))
        c2 = jnp.stack((rotation_matrix[0, 2], rotation_matrix[1, 2], 1 + rotation_matrix[2, 2]))
        use2 = 1 + rotation_matrix[2, 2] >= 1e-6
        use1 = (1 + rotation_matrix[1, 1] >= 1e-6) & ~use2
        candidate = jnp.where(use2, c2, jnp.where(use1, c1, c0))
        half_turn = candidate / jnp.maximum(jnp.linalg.norm(candidate), 1e-12)
        axis = jnp.where(angle > jnp.pi - 1e-6, half_turn, generic)
        return jnp.where(angle < 1e-6, jnp.zeros(3), axis), angle

    return fk, rotation, rotation_log, skew


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_jax_fk_full_jacobian_matches_finite_difference(configuration):
    """JAX feasibility prototype matches the full finite-difference FK Jacobian."""
    jax = pytest.importorskip("jax", exc_type=ImportError)
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy", exc_type=ImportError)
    fk, _, _, _ = _jax_functions(jax, jnp)
    q = jnp.asarray(configuration, dtype=jnp.float64)

    compiled_jacobian = jax.jit(jax.jacrev(lambda value: fk(value).reshape(-1)))
    actual = np.asarray(compiled_jacobian(q))
    np.testing.assert_allclose(np.asarray(jax.jit(fk)(q)), _numpy_fk(configuration), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, _central_jacobian(_numpy_fk, configuration), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("axis, angle", LOG_CASES)
def test_jax_log_singular_endpoints_have_forward_parity(axis, angle):
    """JAX feasibility keeps exact log endpoints forward-only by contract."""
    jax = pytest.importorskip("jax", exc_type=ImportError)
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy", exc_type=ImportError)
    _, rotation, rotation_log, skew = _jax_functions(jax, jnp)
    out_axis, out_angle = rotation_log(rotation(jnp.asarray(axis), jnp.asarray(angle)))
    expected_axis, expected_angle, expected_matrix = _expected_log(axis, angle)

    np.testing.assert_allclose(np.asarray(out_axis), expected_axis, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(out_angle), expected_angle, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(skew(out_axis * out_angle)), expected_matrix, rtol=0, atol=1e-7)


@pytest.mark.parametrize("axis, angle", SMOOTH_LOG_CASES)
def test_jax_log_gradient_matches_finite_difference_in_smooth_interior(axis, angle):
    """JAX log gradients agree away from the identity and half-turn singularities."""
    jax = pytest.importorskip("jax", exc_type=ImportError)
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy", exc_type=ImportError)
    _, rotation, rotation_log, _ = _jax_functions(jax, jnp)
    axis_array = jnp.asarray(axis, dtype=jnp.float64)

    def rotvec(value):
        out_axis, out_angle = rotation_log(rotation(axis_array, value))
        return out_axis * out_angle

    actual = np.asarray(jax.jacrev(rotvec)(jnp.asarray(angle)))
    step = 1e-6
    expected = np.asarray((rotvec(angle + step) - rotvec(angle - step)) / (2 * step))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- #
# Production-dispatch gradient contract (Torch backend, is_concrete=False).
#
# Unlike the feasibility spikes above (which differentiate test-local
# prototypes), these tests drive the *real* SerialManipulator /
# ManipulatorDynamics methods under ``use_backend("torch")`` and check that
# torch.autograd matches central finite differences -- i.e. the core math is
# trace-safe (correct gradients), including at the zero-angle configuration.
# --------------------------------------------------------------------------- #

_RNG = np.random.default_rng(0)
FK_CONFIGS = (
    pytest.param(np.zeros(2), id="zero"),
    pytest.param(np.array([np.pi / 2, np.pi / 2]), id="pi-over-two"),
    pytest.param(_RNG.uniform(-np.pi, np.pi, size=2), id="random-a"),
    pytest.param(_RNG.uniform(-np.pi, np.pi, size=2), id="random-b"),
    pytest.param(_RNG.uniform(-np.pi, np.pi, size=2), id="random-c"),
)


def _planar_2r_dynamics():
    """2R planar arm rig (matches test_v132_regressions mass-matrix test)."""
    L1 = L2 = 1.0
    omega_list = np.array([[0, 0, 1], [0, 0, 1]]).T
    r_list = np.array([[0, 0, 0], [L1, 0, 0]]).T
    M_list = np.eye(4)
    M_list[0, 3] = L1 + L2
    M_link1 = np.eye(4)
    M_link1[0, 3] = L1
    M_link2 = np.eye(4)
    M_link2[0, 3] = L1 + L2
    Glist = np.array(
        [np.diag([0.0, 0.0, 0.0, m, m, m]) for m in (1.0, 1.0)]
    )
    return ManipulatorDynamics(
        M_list=M_list,
        omega_list=omega_list,
        r_list=r_list,
        b_list=None,
        S_list=None,
        B_list=None,
        Glist=Glist,
        Mlist_per_link=[M_link1, M_link2],
    )


@requires_torch
class TestProductionGradientContract:
    """torch.autograd through production dispatch matches finite differences."""

    @pytest.mark.parametrize("configuration", FK_CONFIGS)
    def test_forward_kinematics_jacobian_matches_finite_difference(self, configuration):
        import torch

        dyn = _planar_2r_dynamics()
        with use_backend("torch"):
            q = torch.tensor(configuration, dtype=torch.float64, requires_grad=True)
            actual = torch.autograd.functional.jacobian(
                lambda v: dyn.forward_kinematics(v).reshape(-1), q
            ).detach().numpy()

            def fk_flat(x):
                t = torch.tensor(x, dtype=torch.float64)
                return dyn.forward_kinematics(t).detach().numpy().reshape(-1)

            expected = _central_jacobian(fk_flat, configuration)
        max_err = np.max(np.abs(actual - expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
        assert np.isfinite(max_err)

    @pytest.mark.parametrize("configuration", FK_CONFIGS)
    def test_inverse_dynamics_jacobian_zero_velocity(self, configuration):
        import torch

        dyn = _planar_2r_dynamics()
        with use_backend("torch"):
            q = torch.tensor(configuration, dtype=torch.float64, requires_grad=True)
            dq = torch.zeros(2, dtype=torch.float64)
            ddq = torch.tensor([0.05, 0.1], dtype=torch.float64)
            g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
            Ftip = torch.tensor([0.0, 0.0, 0.3, 1.0, -0.5, 0.0], dtype=torch.float64)

            def idf(v):
                return dyn.inverse_dynamics(v, dq, ddq, g, Ftip)

            actual = torch.autograd.functional.jacobian(idf, q).detach().numpy()

            def idf_np(x):
                t = torch.tensor(x, dtype=torch.float64)
                return idf(t).detach().numpy()

            expected = _central_jacobian(idf_np, configuration)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_inverse_dynamics_jacobian_nonzero_velocity(self):
        import torch

        configuration = np.array([0.31, -1.17])
        dyn = _planar_2r_dynamics()
        with use_backend("torch"):
            q = torch.tensor(configuration, dtype=torch.float64, requires_grad=True)
            dq = torch.tensor([0.1, 0.2], dtype=torch.float64)
            ddq = torch.tensor([0.05, 0.1], dtype=torch.float64)
            g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
            Ftip = torch.tensor([0.0, 0.0, 0.3, 1.0, -0.5, 0.0], dtype=torch.float64)

            def idf(v):
                return dyn.inverse_dynamics(v, dq, ddq, g, Ftip)

            actual = torch.autograd.functional.jacobian(idf, q).detach().numpy()

            def idf_np(x):
                t = torch.tensor(x, dtype=torch.float64)
                return idf(t).detach().numpy()

            expected = _central_jacobian(idf_np, configuration)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_mass_matrix_cache_is_bypassed_under_torch(self):
        import torch

        dyn = _planar_2r_dynamics()
        assert dyn._mass_matrix_cache == {}
        assert dyn._mass_matrix_derivative_cache == {}

        with use_backend("torch"):
            q = torch.tensor([0.31, -1.17], dtype=torch.float64, requires_grad=True)
            M = dyn.mass_matrix(q)
            dyn._mass_matrix_derivatives(q)
            # is_concrete=False: neither cache is read nor populated.
            assert dyn._mass_matrix_cache == {}
            assert dyn._mass_matrix_derivative_cache == {}
            assert M.requires_grad
            M.sum().backward()
            assert q.grad is not None
            assert torch.isfinite(q.grad).all()

        # Pin the is_concrete gate the other way: NumPy DOES populate the cache.
        with use_backend("numpy"):
            dyn.mass_matrix(np.array([0.31, -1.17]))
            assert len(dyn._mass_matrix_cache) == 1

    def test_matrix_exp3_gradient_is_correct_at_exactly_zero_angle(self):
        import torch

        with use_backend("torch"):
            theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            zero = theta * 0
            so3 = utils.VecToso3(torch.stack((zero, zero, theta)))
            (grad,) = torch.autograd.grad(utils.MatrixExp3(so3)[1, 0], theta)
        assert torch.isfinite(grad)
        np.testing.assert_allclose(float(grad), 1.0, rtol=0, atol=1e-9)

    def test_matrix_log3_gradient_is_correct_at_exactly_zero_angle(self):
        import torch

        def rz(angle):
            zero = angle * 0
            one = angle * 0 + 1
            return torch.stack(
                (
                    torch.stack((torch.cos(angle), -torch.sin(angle), zero)),
                    torch.stack((torch.sin(angle), torch.cos(angle), zero)),
                    torch.stack((zero, zero, one)),
                )
            )

        with use_backend("torch"):
            theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (grad,) = torch.autograd.grad(utils.MatrixLog3(rz(theta))[1, 0], theta)
        assert torch.isfinite(grad)
        np.testing.assert_allclose(float(grad), 1.0, rtol=0, atol=1e-9)

    # -- Finding 1: MatrixLog3 gradient must be correct across the near-pi band -- #
    # The SO(3) log is smooth on theta in (0, pi); only theta = pi exactly is a
    # branch point. For a fixed axis n and R = exp(theta*[n]x), the rotation vector
    # is theta*n, so d(rotvec)/dtheta is exactly the (constant) axis n.
    _LOG3_PI_AXIS = np.array([0.3, -0.4, 0.8660254])
    _LOG3_PI_AXIS = _LOG3_PI_AXIS / np.linalg.norm(_LOG3_PI_AXIS)

    @staticmethod
    def _log3_rotvec(torch, axis_np, theta_tensor):
        axis = torch.as_tensor(axis_np, dtype=torch.float64)
        R = utils.MatrixExp3(utils.VecToso3(axis * theta_tensor))
        m = utils.MatrixLog3(R)
        return torch.stack((m[2, 1], m[0, 2], m[1, 0]))

    @pytest.mark.parametrize("gap", [1e-3, 1e-5, 1e-7])
    def test_matrix_log3_gradient_near_pi_matches_finite_difference(self, gap):
        import torch

        axis = self._LOG3_PI_AXIS
        theta_val = np.pi - gap
        with use_backend("torch"):
            actual = torch.autograd.functional.jacobian(
                lambda t: self._log3_rotvec(torch, axis, t),
                torch.tensor(theta_val, dtype=torch.float64),
            ).detach().numpy()

            # Central FD on a detached reference; step kept below the gap so the
            # stencil never straddles the theta = pi branch point.
            step = min(1e-6, gap / 10.0)

            def rotvec_np(t):
                return self._log3_rotvec(
                    torch, axis, torch.tensor(t, dtype=torch.float64)
                ).detach().numpy()

            expected = (rotvec_np(theta_val + step) - rotvec_np(theta_val - step)) / (2 * step)

        assert np.all(np.isfinite(actual))
        # The exact derivative is the constant axis; verify against both the true
        # value and the finite-difference reference.
        np.testing.assert_allclose(actual, axis, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("gap", [1e-3, 1e-5, 1e-7])
    def test_rotation_logm_angle_gradient_near_pi_matches_finite_difference(self, gap):
        """``rotation_logm`` recovers the angle with an exact derivative near pi.

        It shares ``MatrixLog3``'s conditioning hazard -- recovering theta from
        the trace via ``arccos`` loses half its digits near a half turn (d theta /
        d cos theta = 1 / sin theta blows up), which showed up as d theta / d theta
        = 1.061 at gap 1e-7 instead of 1.0. This is the angle IK differentiates,
        so it is pinned separately from the ``MatrixLog3`` rotation vector.
        """
        import torch

        axis = self._LOG3_PI_AXIS
        theta_val = np.pi - gap
        with use_backend("torch"):
            t = torch.tensor(theta_val, dtype=torch.float64, requires_grad=True)
            axis_t = torch.as_tensor(axis, dtype=torch.float64)
            R = utils.MatrixExp3(utils.VecToso3(axis_t * t))
            _, theta_out = utils.rotation_logm(R)
            grad = torch.autograd.grad(theta_out, t)[0].detach().numpy()

        assert np.all(np.isfinite(grad))
        # theta_out IS the input angle here, so d(theta_out)/d(theta) == 1 exactly.
        np.testing.assert_allclose(grad, 1.0, rtol=1e-5, atol=1e-7)

    # Coordinate axes matter here: a half turn about x/y/z (the joint axes of
    # virtually every URDF) makes cos_theta land on EXACTLY -1.0, whereas a
    # general axis rounds an ulp above it. Only the exact -1.0 case exposes an
    # unclamped arccos in the inactive generic branch, whose infinite derivative
    # ``where`` turns into 0 * inf = NaN in the selected branch.
    _PI_AXES = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0**-0.5, 2.0**-0.5, 0.0],
        list(_LOG3_PI_AXIS),
    )

    @pytest.mark.parametrize("axis", _PI_AXES)
    def test_matrix_log3_gradient_is_finite_at_exactly_pi(self, axis):
        import torch

        axis = np.asarray(axis, dtype=float)
        with use_backend("torch"):
            actual = torch.autograd.functional.jacobian(
                lambda t: self._log3_rotvec(torch, axis, t),
                torch.tensor(np.pi, dtype=torch.float64),
            ).detach().numpy()
        # theta = pi is a genuine branch point; only require a finite gradient.
        assert np.all(np.isfinite(actual))

    @pytest.mark.parametrize("axis", _PI_AXES)
    def test_rotation_logm_gradient_is_finite_at_exactly_pi(self, axis):
        import torch

        axis_np = np.asarray(axis, dtype=float)
        with use_backend("torch"):
            t = torch.tensor(np.pi, dtype=torch.float64, requires_grad=True)
            axis_t = torch.as_tensor(axis_np, dtype=torch.float64)
            R = utils.MatrixExp3(utils.VecToso3(axis_t * t))
            ax, theta_out = utils.rotation_logm(R)
            ax_grad = torch.autograd.grad(ax.sum(), t, retain_graph=True)[0]
            th_grad = torch.autograd.grad(theta_out, t)[0]
        assert np.isfinite(ax_grad.detach().numpy()).all()
        assert np.isfinite(th_grad.detach().numpy()).all()

    @pytest.mark.parametrize("gap", [2e-6, 1e-5, 1e-3])
    def test_rotation_logm_axis_gradient_near_pi_is_stationary(self, gap):
        """The axis is mathematically constant in theta, so its derivative is 0.

        ``vee/(2 sin theta)`` is a vanishing/vanishing cancellation as theta -> pi
        and its derivative degrades well before the singularity (~1e-5 error at
        pi - 2e-6), so the stable ``_pi_axis`` form must cover this band.
        """
        import torch

        axis_np = np.asarray(self._LOG3_PI_AXIS, dtype=float)
        with use_backend("torch"):
            t = torch.tensor(np.pi - gap, dtype=torch.float64, requires_grad=True)
            axis_t = torch.as_tensor(axis_np, dtype=torch.float64)
            R = utils.MatrixExp3(utils.VecToso3(axis_t * t))
            ax, _ = utils.rotation_logm(R)
            grad = np.stack([
                torch.autograd.grad(ax[i], t, retain_graph=True)[0].detach().numpy()
                for i in range(3)
            ])
        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(grad, 0.0, atol=1e-9)

    # -- Finding 2: exp Taylor cutoff must cover the region where the exact -- #
    # backward is unstable. Sweep spans [1e-8, 1e-2] incl. the old 1e-6 cutoff.
    _EXP_THETAS = (1e-8, 1e-6, 1.001e-6, 1e-5, 1e-4, 1e-3, 1e-2)

    @pytest.mark.parametrize("theta_val", _EXP_THETAS)
    def test_matrix_exp6_translation_gradient_sweep_matches_finite_difference(self, theta_val):
        import torch

        z = np.array([0.0, 0.0, 1.0])
        v = np.array([1.0, 2.0, 3.0])

        def translation(torch, t):
            K = utils.VecToso3(torch.as_tensor(z, dtype=torch.float64) * t)
            top = torch.cat((K, torch.as_tensor(v, dtype=torch.float64).reshape(3, 1)), dim=1)
            se3 = torch.cat((top, torch.zeros((1, 4), dtype=torch.float64)), dim=0)
            return utils.MatrixExp6(se3)[:3, 3]

        with use_backend("torch"):
            actual = torch.autograd.functional.jacobian(
                lambda t: translation(torch, t),
                torch.tensor(theta_val, dtype=torch.float64),
            ).detach().numpy()

            step = 1e-6

            def trans_np(t):
                return translation(torch, torch.tensor(t, dtype=torch.float64)).detach().numpy()

            expected = (trans_np(theta_val + step) - trans_np(theta_val - step)) / (2 * step)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("theta_val", _EXP_THETAS)
    def test_matrix_exp3_gradient_sweep_matches_finite_difference(self, theta_val):
        import torch

        axis = np.array([0.0, 0.0, 1.0])

        def rotation(torch, t):
            return utils.MatrixExp3(
                utils.VecToso3(torch.as_tensor(axis, dtype=torch.float64) * t)
            ).reshape(-1)

        with use_backend("torch"):
            actual = torch.autograd.functional.jacobian(
                lambda t: rotation(torch, t),
                torch.tensor(theta_val, dtype=torch.float64),
            ).detach().numpy()

            step = 1e-6

            def rot_np(t):
                return rotation(torch, torch.tensor(t, dtype=torch.float64)).detach().numpy()

            expected = (rot_np(theta_val + step) - rot_np(theta_val - step)) / (2 * step)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


# A transform whose translation does NOT scale with the rotation angle. The
# round-trip family below multiplies a screw by t, so its translation shrinks
# with theta and the translational error shrinks as theta^2 -- which hid a real
# defect under a 1e-12 tolerance. Holding p at O(1) while theta -> 0 keeps the
# error first order and therefore visible.
_FIXED_P = np.array([0.4, -1.2, 2.3])
_LOG6_AXIS = np.array([0.0, 0.0, 1.0])


def _fixed_translation_transform(theta_val):
    """4x4 transform: rotation of ``theta_val`` about z, translation ``_FIXED_P``."""
    cos_t, sin_t = np.cos(theta_val), np.sin(theta_val)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = _FIXED_P
    return transform


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("api", ["MatrixLog6", "logm"])
@pytest.mark.parametrize(
    "theta_val",
    [
        0.0, 5e-7, 9.99e-7, 1e-6, 1.01e-6, 1e-3, 0.01, 0.0999, 0.1, 0.1001,
        1.0, 2.0, 3.0,
    ],
)
def test_se3_log_translation_matches_scipy_with_fixed_translation(
    backend_name, api, theta_val
):
    """The translational log is correct when p does not shrink with theta.

    ``theta * G^-1`` was built from ``rotation_logm``'s axis, which is zeroed
    below theta = 1e-6, flattening the whole expression to the identity there
    and returning ``p`` unchanged instead of ``p - (theta/2) w_hat p + ...``.
    The angles bracket the old 1e-6 axis threshold and the Taylor switch at 0.1,
    and continue into the exact branch.

    Both public entry points are covered: they wire the same helpers
    independently, so testing only one leaves the other free to regress.
    ``theta`` stops short of pi because ``scipy.linalg.logm`` is itself
    inaccurate at the branch point (~1e-8 error at pi - 1e-8); near-pi accuracy
    is pinned against an exact construction in
    :func:`test_se3_log_is_exact_near_pi_with_fixed_translation`.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    if backend_name == "torch":
        _torch_or_skip()
    if backend_name == "jax":
        jax = pytest.importorskip("jax", exc_type=ImportError)
        jax.config.update("jax_enable_x64", True)

    transform = _fixed_translation_transform(theta_val)
    expected_matrix = np.real(scipy_linalg.logm(transform))
    expected = (
        expected_matrix
        if api == "MatrixLog6"
        else np.concatenate(
            (
                [expected_matrix[2, 1], expected_matrix[0, 2], expected_matrix[1, 0]],
                expected_matrix[:3, 3],
            )
        )
    )

    with use_backend(backend_name):
        observed = np.asarray(getattr(utils, api)(transform))

    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("api", ["MatrixLog6", "logm"])
@pytest.mark.parametrize("gap", [1e-8, 1e-4, 0.0])
def test_se3_log_is_exact_near_pi_with_fixed_translation(backend_name, api, gap):
    """Near and at pi the log is checked by SciPy's exponential, not its log.

    ``scipy.linalg.logm`` loses roughly seven digits at the branch point, so it
    cannot arbitrate here. Round-tripping through ManipulaPy's own ``MatrixExp6``
    would not either: a pair of mutually inverse but wrongly scaled conventions
    (exp writing ``2 rho`` while log returns ``p / 2``) satisfies it. Feeding the
    result to ``scipy.linalg.expm`` breaks that pairing, since the check no
    longer shares any code with what it is checking, and ``expm`` stays
    well conditioned exactly where ``logm`` does not.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    if backend_name == "torch":
        _torch_or_skip()
    if backend_name == "jax":
        jax = pytest.importorskip("jax", exc_type=ImportError)
        jax.config.update("jax_enable_x64", True)

    theta_val = np.pi - gap
    screw = np.concatenate((_LOG6_AXIS * theta_val, _FIXED_P))
    transform = np.asarray(utils.MatrixExp6(utils.VecTose3(screw)))

    with use_backend(backend_name):
        observed = np.asarray(getattr(utils, api)(transform))
    observed_matrix = (
        observed
        if api == "MatrixLog6"
        else np.block(
            [
                [
                    np.array(
                        [
                            [0.0, -observed[2], observed[1]],
                            [observed[2], 0.0, -observed[0]],
                            [-observed[1], observed[0], 0.0],
                        ]
                    ),
                    observed[3:].reshape(3, 1),
                ],
                [np.zeros((1, 4))],
            ]
        )
    )

    # Independent oracle: exponentiating the log must reproduce the transform.
    np.testing.assert_allclose(
        scipy_linalg.expm(observed_matrix), transform, rtol=1e-10, atol=1e-12
    )
    # ...and the result must be the PRINCIPAL branch, which pins the scale.
    rotation_norm = np.linalg.norm([observed_matrix[2, 1], observed_matrix[0, 2],
                                    observed_matrix[1, 0]])
    assert rotation_norm <= np.pi + 1e-9, f"not the principal branch: {rotation_norm}"


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("theta_val", [0.0, 5e-7, 1e-6, 1e-5, 1e-3, 1.0])
def test_matrix_log6_round_trip_is_exact_at_small_angles(backend_name, theta_val):
    """``log6(exp6(S t))`` returns ``S t`` on every backend, including tiny t.

    This is a value regression, not a gradient one, and it was never
    JAX-specific: ``MatrixLog6`` selected a pure-translation result whenever the
    angle fell below 1e-6, so a screw with a real 5e-7 rotation came back with
    its angular part zeroed on NumPy, Torch and JAX alike.
    """
    if backend_name == "torch":
        _torch_or_skip()
    if backend_name == "jax":
        jax = pytest.importorskip("jax", exc_type=ImportError)
        jax.config.update("jax_enable_x64", True)

    screw = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
    with use_backend(backend_name):
        observed = np.asarray(
            utils.se3ToVec(
                utils.MatrixLog6(utils.MatrixExp6(utils.VecTose3(screw * theta_val)))
            )
        )
        via_logm = np.asarray(
            utils.logm(utils.MatrixExp6(utils.VecTose3(screw * theta_val)))
        )

    np.testing.assert_allclose(observed, screw * theta_val, atol=1e-12)
    np.testing.assert_allclose(via_logm, screw * theta_val, atol=1e-12)


def _requires_jax():
    """Import JAX with x64 enabled, or skip.

    The backend enables x64 itself when it is registered, but these tests build
    JAX arrays directly, so the flag is set here too: without it JAX silently
    computes in float32 and the finite-difference comparisons below would be
    measuring float32 noise rather than the gradient contract.
    """
    jax = pytest.importorskip("jax", exc_type=ImportError)
    jax.config.update("jax_enable_x64", True)
    return jax, pytest.importorskip("jax.numpy", exc_type=ImportError)


class TestJaxProductionGradientContract:
    """jax.grad/jacrev through production dispatch matches finite differences.

    The mirror of :class:`TestProductionGradientContract`. The JAX tests
    earlier in this file differentiate test-local prototypes; these go through
    ``use_backend("jax")`` and the real ManipulaPy entry points, so they pin the
    differentiable contract on the shipped code rather than on a re-derivation
    of it.
    """

    @pytest.mark.parametrize("configuration", FK_CONFIGS)
    def test_forward_kinematics_jacobian_matches_finite_difference(self, configuration):
        jax, jnp = _requires_jax()

        dyn = _planar_2r_dynamics()
        with use_backend("jax"):
            q = jnp.asarray(configuration, dtype=jnp.float64)
            actual = np.asarray(
                jax.jacrev(lambda v: dyn.forward_kinematics(v).reshape(-1))(q)
            )

            def fk_flat(x):
                return np.asarray(dyn.forward_kinematics(jnp.asarray(x))).reshape(-1)

            expected = _central_jacobian(fk_flat, configuration)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("configuration", FK_CONFIGS)
    def test_inverse_dynamics_jacobian_zero_velocity(self, configuration):
        jax, jnp = _requires_jax()

        dyn = _planar_2r_dynamics()
        with use_backend("jax"):
            q = jnp.asarray(configuration, dtype=jnp.float64)
            dq = jnp.zeros(2, dtype=jnp.float64)
            ddq = jnp.asarray([0.05, 0.1], dtype=jnp.float64)
            g = jnp.asarray([0.0, -9.81, 0.0], dtype=jnp.float64)
            Ftip = jnp.asarray([0.0, 0.0, 0.3, 1.0, -0.5, 0.0], dtype=jnp.float64)

            def idf(v):
                return dyn.inverse_dynamics(v, dq, ddq, g, Ftip)

            actual = np.asarray(jax.jacrev(idf)(q))
            expected = _central_jacobian(
                lambda x: np.asarray(idf(jnp.asarray(x))), configuration
            )

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_inverse_dynamics_jacobian_nonzero_velocity(self):
        jax, jnp = _requires_jax()

        configuration = np.array([0.31, -1.17])
        dyn = _planar_2r_dynamics()
        with use_backend("jax"):
            q = jnp.asarray(configuration, dtype=jnp.float64)
            dq = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
            ddq = jnp.asarray([0.05, 0.1], dtype=jnp.float64)
            g = jnp.asarray([0.0, -9.81, 0.0], dtype=jnp.float64)
            Ftip = jnp.asarray([0.0, 0.0, 0.3, 1.0, -0.5, 0.0], dtype=jnp.float64)

            def idf(v):
                return dyn.inverse_dynamics(v, dq, ddq, g, Ftip)

            actual = np.asarray(jax.jacrev(idf)(q))
            expected = _central_jacobian(
                lambda x: np.asarray(idf(jnp.asarray(x))), configuration
            )

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_mass_matrix_cache_is_bypassed_under_jax(self):
        """A traced mass matrix must not be served from the value-keyed cache.

        ``JaxBackend.is_concrete`` is False precisely so this cache is skipped:
        a tracer has no host-readable value to key on, and a cached NumPy
        matrix would silently sever the gradient.
        """
        jax, jnp = _requires_jax()

        configuration = np.array([0.31, -1.17])
        dyn = _planar_2r_dynamics()
        with use_backend("jax"):
            q = jnp.asarray(configuration, dtype=jnp.float64)
            grad = np.asarray(jax.grad(lambda v: dyn.mass_matrix(v).sum())(q))

        assert np.all(np.isfinite(grad))
        assert np.any(grad != 0.0), "mass matrix gradient was severed by the cache"

    def test_matrix_exp3_gradient_is_correct_at_exactly_zero_angle(self):
        """d/dtheta of a z-rotation at theta=0 is 1 for the [1, 0] entry.

        The Taylor branch must carry a gradient here; a naive ``sin(t)/t``
        yields 0/0 and a ``where`` over it leaks NaN into the backward pass.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            axis = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64)

            def entry(t):
                return utils.MatrixExp3(utils.VecToso3(axis * t))[1, 0]

            grad = float(jax.grad(entry)(jnp.asarray(0.0, dtype=jnp.float64)))

        np.testing.assert_allclose(grad, 1.0, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("theta_val", [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2])
    def test_matrix_log3_gradient_is_correct_near_the_identity(self, theta_val):
        """``d/dtheta vee(log(exp(n theta)))`` is exactly ``n``, including at 0.

        The mirror of the Torch zero-angle test, widened into a sweep because
        the defect it guards was not confined to exactly zero. ``_pi_axis``
        normalised by ``maximum(norm(candidate), eps)``; near the identity
        ``candidate`` vanishes, so ``d|c|/dc = c/|c|`` was already 0/0 and the
        clamp turned it into ``0 * NaN``. Values stayed correct throughout,
        which is why only a gradient assertion catches it -- and it produced
        NaN for every theta up to ~1e-2, not just at the endpoint.
        """
        jax, jnp = _requires_jax()

        axis_np = np.array([0.0, 0.0, 1.0])
        with use_backend("jax"):
            def rotvec(t):
                R = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis_np, dtype=jnp.float64) * t)
                )
                return utils.skew_symmetric_to_vector(utils.MatrixLog3(R))

            actual = np.asarray(jax.jacrev(rotvec)(jnp.asarray(theta_val)))

        assert np.all(np.isfinite(actual)), f"non-finite gradient: {actual}"
        np.testing.assert_allclose(actual, axis_np, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("theta_val", [0.0, 5e-7, 1e-6, 1e-3, 0.5])
    def test_matrix_log6_gradient_recovers_the_screw(self, theta_val):
        """``d/dt vee6(log6(exp6(S t)))`` is the screw ``S`` at every t.

        ``MatrixLog6`` rebuilt its rotational block as ``omega * theta``, but
        ``rotation_logm`` zeroes its axis below theta = 1e-6 because the axis is
        genuinely undefined at the identity. That discarded a small-but-real
        rotation outright: the derivative of the angular part was 0 instead of
        the screw's angular component, and at t = 5e-7 even the primal rotation
        came back as 0. The block is now ``MatrixLog3`` itself, whose
        Taylor-safe coefficient is smooth in both value and gradient there.
        """
        jax, jnp = _requires_jax()

        screw = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
        with use_backend("jax"):
            def log6_vec(t):
                return utils.se3ToVec(
                    utils.MatrixLog6(
                        utils.MatrixExp6(
                            utils.VecTose3(jnp.asarray(screw, dtype=jnp.float64) * t)
                        )
                    )
                )

            value = np.asarray(log6_vec(jnp.asarray(theta_val)))
            grad = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(theta_val)))

        np.testing.assert_allclose(value, screw * theta_val, atol=1e-12)
        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(grad, screw, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize(
        "theta_val", [0.0, 5e-7, 9.99e-7, 1e-6, 1.01e-6, 1e-3, 0.01, 0.0999, 0.1]
    )
    def test_matrix_log6_translational_gradient_with_fixed_translation(self, theta_val):
        """The translational derivative is right when p does not shrink with theta.

        Two defects hid behind a translation that scaled with theta: the
        thresholded axis flattened ``theta * G^-1`` to the identity below
        theta = 1e-6 (derivative [0, 0, 0] instead of -0.5 w_hat p), and the
        Taylor switch sat where the exact branch's own gradient had already lost
        about four percent. The angles bracket both loci.

        Finite differences are compared at h = 1e-4: this quotient is roundoff
        dominated, and its error grows as 1/h (5.9e-9 at 1e-4 up to 5.6e-6 at
        1e-7), so a smaller step measures the reference, not the gradient.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            def log6_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            actual = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(theta_val)))
            step = 1e-4
            expected = (
                np.asarray(log6_vec(jnp.asarray(theta_val + step)))
                - np.asarray(log6_vec(jnp.asarray(theta_val - step)))
            ) / (2 * step)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("theta_val", [0.1001, 0.5, 1.5, 3.0])
    def test_matrix_log6_translational_gradient_in_the_exact_branch(self, theta_val):
        """The gradient is right ABOVE the Taylor switch.

        The switch at theta = 0.1 divides two formulas; the Taylor side is
        covered by the sweep above, and this covers the exact side. It stops at
        3.0 because a central difference cannot reach closer to pi than its own
        step: straddling the branch cut makes the principal log wrap, and the
        reference blows up to ~pi/step rather than the gradient being wrong. The
        half-turn region is covered by the finiteness test below instead, which
        is how this file treats the pi endpoint elsewhere.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            def log6_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            actual = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(theta_val)))
            step = 1e-5
            expected = (
                np.asarray(log6_vec(jnp.asarray(theta_val + step)))
                - np.asarray(log6_vec(jnp.asarray(theta_val - step)))
            ) / (2 * step)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "gap, step", [(1e-2, 1e-4), (1e-4, 1e-6)]
    )
    def test_matrix_log6_gradient_has_the_right_value_near_pi(self, gap, step):
        """Near pi the gradient must be RIGHT, not merely finite.

        A unique derivative exists at every gap > 0, so finiteness alone is too
        weak: an implementation returning the correct primal but a zero gradient
        in this region would satisfy it. The step is chosen well inside the gap
        so the central difference never straddles the branch cut at pi.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            def log6_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            theta_val = np.pi - gap
            actual = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(theta_val)))
            expected = (
                np.asarray(log6_vec(jnp.asarray(theta_val + step)))
                - np.asarray(log6_vec(jnp.asarray(theta_val - step)))
            ) / (2 * step)

        assert np.all(np.isfinite(actual))
        # A zero-gradient implementation must not survive this.
        assert np.abs(actual).max() > 1e-3, "gradient collapsed to zero near pi"
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("gap", [1e-8, 0.0])
    def test_matrix_log6_gradient_is_finite_at_the_half_turn(self, gap):
        """At and adjacent to pi only finiteness is required.

        The principal logarithm is genuinely non-differentiable at pi, so no
        unique correct gradient exists there, and a central difference cannot
        reach inside a 1e-8 gap without straddling the cut. What must hold is
        that the half-turn branch of MatrixLog3 and the exact branch of the
        coefficient do not leak ``0 * inf`` into the backward pass, which is the
        failure mode this whole file exists to catch. Gaps where a unique
        derivative does exist are value-checked above.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            def log6_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            actual = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(np.pi - gap)))

        assert np.all(np.isfinite(actual)), f"non-finite gradient at pi - {gap}"

    @pytest.mark.parametrize("theta_val", [0.0, 5e-7, 1e-6, 0.01, 0.1, 0.1001, 1.5])
    def test_matrix_log6_translational_gradient_matches_torch(self, theta_val):
        """Torch reaches the same translational gradient as JAX.

        The gradient coverage for this function was JAX-only, so a Torch-side
        regression in the same code path would have gone unnoticed. Both
        backends run the identical dispatch, so they must agree to round-off.
        """
        jax, jnp = _requires_jax()
        torch = _torch_or_skip()


        with use_backend("jax"):
            def jax_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            jax_grad = np.asarray(jax.jacrev(jax_vec)(jnp.asarray(theta_val)))

        with use_backend("torch"):
            def torch_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(
                        torch.as_tensor(_LOG6_AXIS, dtype=torch.float64) * t
                    )
                )
                bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
                p_col = torch.as_tensor(_FIXED_P, dtype=torch.float64).reshape(3, 1)
                transform = torch.cat(
                    (torch.cat((rot, p_col), dim=1), bottom), dim=0
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            torch_grad = (
                torch.autograd.functional.jacobian(
                    torch_vec, torch.tensor(theta_val, dtype=torch.float64)
                )
                .detach()
                .numpy()
            )

        assert np.all(np.isfinite(torch_grad))
        np.testing.assert_allclose(torch_grad, jax_grad, rtol=1e-9, atol=1e-11)

    def test_matrix_log6_translational_gradient_at_zero_is_exact(self):
        """At theta = 0 the translational derivative is exactly ``-0.5 w_hat p``.

        Pinned against the closed form rather than finite differences so the
        origin -- where the thresholded axis previously produced [0, 0, 0] -- is
        checked without any reference noise.
        """
        jax, jnp = _requires_jax()

        with use_backend("jax"):
            def log6_vec(t):
                rot = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(_LOG6_AXIS, dtype=jnp.float64) * t)
                )
                transform = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [rot, jnp.asarray(_FIXED_P).reshape(3, 1)], axis=1
                        ),
                        jnp.asarray([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                )
                return utils.se3ToVec(utils.MatrixLog6(transform))

            actual = np.asarray(jax.jacrev(log6_vec)(jnp.asarray(0.0)))

        # d/dtheta (theta G^-1 p) at 0 = -0.5 * (w_hat p); w_hat is z-skew here.
        w_hat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(actual[:3], _LOG6_AXIS, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            actual[3:], -0.5 * (w_hat @ _FIXED_P), rtol=1e-9, atol=1e-12
        )

    def test_extrapolate_from_current_gradient_keeps_the_angular_part(self):
        """The reachable IK helper inherits the recovered rotational gradient.

        ``extrapolate_from_current`` differentiates a pose error through
        ``MatrixLog6``; with the rotational block discarded its angular
        derivative was 0 rather than ``alpha`` times the screw's angular part.
        """
        jax, jnp = _requires_jax()

        from ManipulaPy.kinematics.ik_helpers import extrapolate_from_current

        screw = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
        alpha = 0.5
        with use_backend("jax"):
            def guess(t):
                target = utils.MatrixExp6(
                    utils.VecTose3(jnp.asarray(screw, dtype=jnp.float64) * t)
                )
                return extrapolate_from_current(
                    jnp.zeros(6, dtype=jnp.float64),
                    jnp.eye(4, dtype=jnp.float64),
                    target,
                    lambda q: jnp.eye(6, dtype=jnp.float64),
                    [(None, None)] * 6,
                    alpha=alpha,
                )

            grad = np.asarray(jax.jacrev(guess)(jnp.asarray(0.0)))

        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(grad, alpha * screw, rtol=1e-6, atol=1e-7)

    def test_condition_number_gradient_matches_finite_difference(self):
        """``condition_number`` stays differentiable under a traced backend.

        Singularity is inside the differentiable contract, but the metric ended
        with an unconditional host conversion for the NaN -> inf substitution,
        which raises ``TracerArrayConversionError`` under ``jax.grad``. The
        conversion is now gated on ``is_concrete``.
        """
        jax, jnp = _requires_jax()

        from ManipulaPy.kinematics import SerialManipulator
        from ManipulaPy.singularity import Singularity

        screws = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.3, 0.0, 0.2],
                [1.0, 0.0, 0.0, 0.0, 0.4, -0.1],
            ],
            dtype=np.float64,
        ).T
        home = np.eye(4)
        home[:3, 3] = [0.7, 0.2, 0.5]
        robot = SerialManipulator(
            M_list=home,
            omega_list=screws[:3, :],
            S_list=screws,
            B_list=screws.copy(),
            joint_limits=[(-3.0, 3.0)] * 3,
        )
        analysis = Singularity(robot)
        configuration = np.array([0.2, -0.3, 0.4])

        with use_backend("jax"):
            actual = np.asarray(
                jax.grad(lambda q: analysis.condition_number(q))(
                    jnp.asarray(configuration, dtype=jnp.float64)
                )
            )

        expected = _central_jacobian(
            lambda x: np.atleast_1d(float(analysis.condition_number(x))), configuration
        ).reshape(-1)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("axis", TestProductionGradientContract._PI_AXES)
    def test_matrix_log3_gradient_is_finite_at_exactly_pi(self, axis):
        """theta=pi is a branch point; the gradient must still be finite.

        Coordinate axes are kept in the parametrisation because only they make
        cos_theta land on exactly -1.0, which is what turns an unclamped
        arccos in the inactive branch into 0 * inf = NaN.
        """
        jax, jnp = _requires_jax()

        axis_np = np.asarray(axis, dtype=float)
        with use_backend("jax"):
            def rotvec(t):
                R = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis_np, dtype=jnp.float64) * t)
                )
                return utils.skew_symmetric_to_vector(utils.MatrixLog3(R))

            actual = np.asarray(jax.jacrev(rotvec)(jnp.asarray(np.pi)))

        assert np.all(np.isfinite(actual)), f"non-finite gradient at pi: {actual}"

    @pytest.mark.parametrize("axis", TestProductionGradientContract._PI_AXES)
    def test_rotation_logm_gradient_is_finite_at_exactly_pi(self, axis):
        """``rotation_logm`` is the form the angle IK differentiates."""
        jax, jnp = _requires_jax()

        axis_np = np.asarray(axis, dtype=float)
        with use_backend("jax"):
            def axis_sum(t):
                R = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis_np, dtype=jnp.float64) * t)
                )
                return utils.rotation_logm(R)[0].sum()

            def angle_out(t):
                R = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis_np, dtype=jnp.float64) * t)
                )
                return utils.rotation_logm(R)[1]

            pi = jnp.asarray(np.pi)
            ax_grad = np.asarray(jax.grad(axis_sum)(pi))
            th_grad = np.asarray(jax.grad(angle_out)(pi))

        assert np.all(np.isfinite(ax_grad))
        assert np.all(np.isfinite(th_grad))

    @pytest.mark.parametrize("gap", [1e-7, 1e-5, 1e-3])
    def test_matrix_log3_gradient_near_pi_matches_finite_difference(self, gap):
        """Near pi the recovered angle must be accurate, not merely finite.

        ``arccos(cos theta)`` is ill-conditioned here because cos theta is
        *quadratic* in the gap; the shipped ``atan2`` form is linear in it and
        so keeps full precision. A regression shows up as a gradient error
        proportional to the value error, not as a NaN.
        """
        jax, jnp = _requires_jax()

        axis_np = np.asarray(TestProductionGradientContract._LOG3_PI_AXIS, dtype=float)
        theta_val = np.pi - gap
        with use_backend("jax"):
            def rotvec(t):
                R = utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis_np, dtype=jnp.float64) * t)
                )
                return utils.skew_symmetric_to_vector(utils.MatrixLog3(R))

            actual = np.asarray(jax.jacrev(rotvec)(jnp.asarray(theta_val)))

        # d/dtheta log(exp(n * theta)) = n for theta in (0, pi).
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, axis_np, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("theta_val", TestProductionGradientContract._EXP_THETAS)
    def test_matrix_exp3_gradient_sweep_matches_finite_difference(self, theta_val):
        jax, jnp = _requires_jax()

        axis = np.array([0.0, 0.0, 1.0])
        with use_backend("jax"):
            def rotation(t):
                return utils.MatrixExp3(
                    utils.VecToso3(jnp.asarray(axis, dtype=jnp.float64) * t)
                ).reshape(-1)

            actual = np.asarray(jax.jacrev(rotation)(jnp.asarray(theta_val)))

            step = 1e-6
            expected = (
                np.asarray(rotation(jnp.asarray(theta_val + step)))
                - np.asarray(rotation(jnp.asarray(theta_val - step)))
            ) / (2 * step)

        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("configuration", FK_CONFIGS)
    def test_forward_kinematics_jacobian_survives_jit(self, configuration):
        """The FK jacobian is unchanged when the whole thing is jit-compiled.

        Eager JAX tolerates host-side control flow that ``jit`` rejects, so
        compiling is the stricter check that no tracer leaks to the host on the
        core-math path.
        """
        jax, jnp = _requires_jax()

        dyn = _planar_2r_dynamics()
        with use_backend("jax"):
            q = jnp.asarray(configuration, dtype=jnp.float64)
            jac = jax.jacrev(lambda v: dyn.forward_kinematics(v).reshape(-1))
            eager = np.asarray(jac(q))
            compiled = np.asarray(jax.jit(jac)(q))

        np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-14)
