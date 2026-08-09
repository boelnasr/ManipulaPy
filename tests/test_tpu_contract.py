#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed numerical contract for a Google Cloud TPU.

This module intentionally has no skip path.  It is run only on the ephemeral
TPU VM provisioned by ``tpu-release.yml``; running it on a CPU or GPU must fail
the platform assertion instead of silently validating a fallback device.

STATUS: this contract does not pass.  Run on a real one-chip TPU v5e, 9 of its
18 checks fail.  XLA:TPU implements neither float64 ``LuDecomposition`` nor
int64 ``dot``, so ``inv``, ``solve``, ``mass_matrix`` and ``inverse_dynamics``
raise ``UNIMPLEMENTED``; and the float64 matmuls that do execute are emulated on
a bf16-native MXU at float32 accuracy, missing the tolerances asserted below by
roughly three orders of magnitude.  TPUs are therefore not a supported target
and no ``jax-tpu`` extra is published.  This file is kept as the executable
specification for the per-platform precision domain that TPU support would
need; it is excluded from CI by ``--ignore`` in ``test.yml``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.backend import use_backend
from ManipulaPy.backend.jax_backend import JaxBackend
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.urdf_processor import URDFToSerialManipulator

pytestmark = pytest.mark.tpu


_SLIST = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.3, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.4, -0.1],
    ],
    dtype=np.float64,
).T
_HOME = np.array(
    [
        [1.0, 0.0, 0.0, 0.7],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_THETA = np.array([0.2, -0.3, 0.4], dtype=np.float64)


def _sync(value):
    """Wait for one JAX value and return it for assertions."""
    value.block_until_ready()
    return value


def _robot() -> SerialManipulator:
    return SerialManipulator(
        M_list=_HOME,
        omega_list=_SLIST[:3, :],
        S_list=_SLIST,
        B_list=_SLIST.copy(),
        joint_limits=[(-3.0, 3.0)] * 3,
    )


@pytest.fixture(scope="module")
def panda_contract():
    """One stored Panda dynamics row plus the corresponding live objects."""
    golden_path = Path(__file__).parent / "data" / "dynamics_golden_panda.npz"
    data = dict(np.load(golden_path))
    processor = URDFToSerialManipulator(get_robot_urdf("panda"), load_meshes=False)
    return processor.serial_manipulator, processor.dynamics, data


def test_running_on_real_tpu():
    devices = jax.devices()
    assert devices
    assert all(device.platform == "tpu" for device in devices)


def test_float64_jit_executes_on_tpu():
    x = jax.device_put(jnp.array([1.0, 2.0], dtype=jnp.float64))
    result = _sync(jax.jit(lambda value: value @ value)(x))
    assert result.dtype == jnp.float64
    assert result.device.platform == "tpu"
    np.testing.assert_allclose(np.asarray(result), 5.0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        pytest.param(jnp.float32, jnp.float32, id="float32"),
        pytest.param(jnp.float64, jnp.float64, id="float64"),
        pytest.param(jnp.int64, jnp.int64, id="int64"),
    ],
)
def test_matmul_real_dtype_contract(dtype, expected_dtype):
    backend = JaxBackend()
    left = np.array([[1, 2], [3, 4]], dtype=np.dtype(dtype))
    right = np.array([[2, 0], [1, 2]], dtype=np.dtype(dtype))
    result = _sync(backend.matmul(left, right))
    assert result.dtype == expected_dtype
    assert result.device.platform == "tpu"
    np.testing.assert_array_equal(np.asarray(result), [[4, 4], [10, 8]])


def test_svd_float64_matches_numpy():
    matrix = np.array([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    expected = np.linalg.svd(matrix, full_matrices=False)
    u, singular, vh = JaxBackend().svd(matrix, full_matrices=False)
    u, singular, vh = _sync(u), _sync(singular), _sync(vh)
    reconstructed = _sync((u * singular) @ vh)
    assert reconstructed.dtype == jnp.float64
    np.testing.assert_allclose(
        np.asarray(reconstructed), matrix, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(singular), expected[1], rtol=1e-12, atol=1e-12
    )


def test_solve_float64_matches_numpy():
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    vector = np.array([1.0, 2.0], dtype=np.float64)
    result = _sync(JaxBackend().solve(matrix, vector))
    assert result.dtype == jnp.float64
    np.testing.assert_allclose(
        np.asarray(result), np.linalg.solve(matrix, vector), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("operation", ["inv", "pinv"])
def test_inverse_operations_float64_match_numpy(operation):
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    backend = JaxBackend()
    result = _sync(getattr(backend, operation)(matrix))
    expected = getattr(np.linalg, operation)(matrix)
    assert result.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("operation", ["fk", "jacobian"])
def test_robot_kinematics_match_numpy(operation):
    robot = _robot()
    if operation == "fk":
        expected = np.asarray(robot.forward_kinematics(_THETA, frame="space"))
        with use_backend("jax"):
            result = _sync(robot.forward_kinematics(_THETA, frame="space"))
    else:
        expected = np.asarray(robot.jacobian(_THETA, frame="space"))
        with use_backend("jax"):
            result = _sync(robot.jacobian(_THETA, frame="space"))
    assert result.dtype == jnp.float64
    assert result.device.platform == "tpu"
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-10, atol=1e-11)


def test_panda_mass_matrix_matches_stored_golden(panda_contract):
    _, dynamics, data = panda_contract
    theta = data["thetas"][0]
    with use_backend("jax"):
        result = _sync(dynamics.mass_matrix(theta))
    assert result.dtype == jnp.float64
    assert result.device.platform == "tpu"
    np.testing.assert_allclose(
        np.asarray(result), data["mass_matrix"][0], rtol=1e-7, atol=1e-9
    )


def test_panda_inverse_dynamics_matches_stored_golden(panda_contract):
    _, dynamics, data = panda_contract
    with use_backend("jax"):
        result = _sync(
            dynamics.inverse_dynamics(
                data["thetas"][0],
                data["dthetas"][0],
                data["ddthetas"][0],
                data["g"],
                data["ftips"][0],
            )
        )
    assert result.dtype == jnp.float64
    assert result.device.platform == "tpu"
    np.testing.assert_allclose(
        np.asarray(result), data["inverse_dynamics"][0], rtol=1e-7, atol=1e-8
    )


def test_jitted_forward_kinematics_matches_numpy_reference():
    robot = _robot()
    expected = np.asarray(robot.forward_kinematics(_THETA, frame="space"))
    with use_backend("jax"):
        theta = jnp.asarray(_THETA)
        compiled = _sync(
            jax.jit(lambda q: robot.forward_kinematics(q, frame="space"))(theta)
        )
    np.testing.assert_allclose(np.asarray(compiled), expected, rtol=1e-11, atol=1e-12)
    assert compiled.device.platform == "tpu"


def test_jacrev_forward_kinematics_runs_on_tpu():
    robot = _robot()
    step = 1e-6
    columns = []
    for index in range(_THETA.size):
        delta = np.zeros_like(_THETA)
        delta[index] = step
        plus = np.asarray(
            robot.forward_kinematics(_THETA + delta, frame="space")
        ).reshape(-1)
        minus = np.asarray(
            robot.forward_kinematics(_THETA - delta, frame="space")
        ).reshape(-1)
        columns.append((plus - minus) / (2 * step))
    expected = np.stack(columns, axis=1)
    with use_backend("jax"):
        theta = jnp.asarray(_THETA)
        jacobian = _sync(
            jax.jacrev(
                lambda q: robot.forward_kinematics(q, frame="space").reshape(-1)
            )(theta)
        )
    assert jacobian.dtype == jnp.float64
    assert jacobian.shape == (16, 3)
    np.testing.assert_allclose(np.asarray(jacobian), expected, rtol=1e-5, atol=1e-7)
    assert jacobian.device.platform == "tpu"


def test_complex_input_fails_before_tpu_transfer():
    backend = JaxBackend()
    with pytest.raises(TypeError, match="complex.*TPU"):
        backend.to_device(np.array([1.0 + 2.0j], dtype=np.complex128))


@pytest.mark.parametrize("constructor", ["array", "asarray"])
def test_complex_source_cannot_be_masked_by_requested_real_dtype(constructor):
    backend = JaxBackend()
    source = np.array([1.0 + 2.0j], dtype=np.complex128)
    with pytest.raises(TypeError, match="complex.*TPU"):
        getattr(backend, constructor)(source, dtype=jnp.float64)
