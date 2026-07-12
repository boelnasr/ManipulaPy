#!/usr/bin/env python3
"""Feasibility checks for the planned differentiable array backends.

These tests intentionally use test-local functional Torch and JAX prototypes.
They prove that the branchless core formulas can support autodiff; they do not
claim that ManipulaPy's production backend dispatch supports either framework.
"""

import math

import numpy as np
import pytest

from ManipulaPy import utils


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
    torch = pytest.importorskip("torch", exc_type=ImportError)
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
    torch = pytest.importorskip("torch", exc_type=ImportError)
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
    torch = pytest.importorskip("torch", exc_type=ImportError)
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
