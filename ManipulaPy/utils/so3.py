#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SO(3) rotation-group utilities.

Copyright (c) 2025 Mohamed Aboelnasr. Licensed under AGPL-3.0-or-later.
"""

from math import pi
from typing import Any, Tuple

from numpy.typing import NDArray

from ManipulaPy.backend import get_backend
from ManipulaPy.backend.numpy_backend import NumpyBackend


def NearZero(z: float) -> bool:
    """Return whether ``z`` is within the library's zero tolerance."""
    return abs(z) < 1e-6


def skew_symmetric(v) -> NDArray:
    """Return the skew-symmetric matrix of a three-vector."""
    b = get_backend()
    zero = b.asarray(0) * v[0]
    return b.stack(
        (b.stack((zero, -v[2], v[1])),
         b.stack((v[2], zero, -v[0])),
         b.stack((-v[1], v[0], zero)))
    )


def _pi_axis(R: Any) -> Any:
    """Extract a stable half-turn axis without value-dependent Python branches."""
    b = get_backend()
    eps = b.asarray(1e-12)
    c2 = b.stack((R[0, 2], R[1, 2], 1 + R[2, 2]))
    c1 = b.stack((R[0, 1], 1 + R[1, 1], R[2, 1]))
    c0 = b.stack((1 + R[0, 0], R[1, 0], R[2, 0]))
    use2 = (1 + R[2, 2]) >= 1e-6
    use1 = ((1 + R[1, 1]) >= 1e-6) & ~use2
    candidate = b.where(use2, c2, b.where(use1, c1, c0))
    return candidate / b.maximum(b.norm(candidate), eps)


def rotation_logm(R) -> Tuple[NDArray, Any]:
    """Return the unit rotation axis and angle for a 3x3 rotation matrix.

    The masked formulation avoids scalar extraction and value-dependent Python
    branches, allowing value tracing on array backends. The principal-log
    derivative is not defined at a half turn and is not guaranteed near the
    identity or half-turn branch boundaries.
    """
    b = get_backend()
    R = b.asarray(R)
    if R.shape != (3, 3):
        raise ValueError(
            f"rotation_logm requires a 3x3 rotation matrix, got shape {R.shape}. "
            f"Matrix:\n{b.to_numpy(R)}"
        )

    cos_theta = b.clip((b.trace(R) - 1) / 2, -1.0, 1.0)
    theta = b.arccos(cos_theta)
    vee = b.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    generic = vee / b.maximum(2 * b.sin(theta), b.asarray(1e-12))
    axis = b.where(theta > pi - 1e-6, _pi_axis(R), generic)
    axis = b.where(theta < 1e-6, b.zeros(3), axis)
    if isinstance(b, NumpyBackend):
        return axis, float(theta)
    return axis, theta


def skew_symmetric_to_vector(matrix) -> NDArray:
    """Convert a skew-symmetric matrix to its three-vector."""
    b = get_backend()
    return b.stack((matrix[2, 1], matrix[0, 2], matrix[1, 0]))


def MatrixLog3(R) -> NDArray:
    """Compute the matrix logarithm of a rotation matrix."""
    b = get_backend()
    R = b.asarray(R)
    cos_theta = b.clip((b.trace(R) - 1) / 2, -1.0, 1.0)
    theta = b.arccos(cos_theta)
    skew_part = R - R.T
    safe_sine = b.maximum(b.abs(b.sin(theta)), b.asarray(1e-12))
    generic = theta / (2 * safe_sine) * skew_part
    small = 0.5 * skew_part
    half_turn = VecToso3(theta * _pi_axis(R))
    return b.where(
        theta > pi - 1e-6,
        half_turn,
        b.where(theta < 1e-6, small, generic),
    )


def VecToso3(omega) -> NDArray:
    """Convert a 3D angular velocity vector to an so(3) matrix."""
    return skew_symmetric(omega)


def MatrixExp3(so3mat) -> NDArray:
    """Compute the SO(3) exponential with the Rodrigues formula."""
    b = get_backend()
    so3mat = b.asarray(so3mat)
    omega_theta = skew_symmetric_to_vector(so3mat)
    theta = b.norm(omega_theta)
    safe_theta = b.maximum(theta, b.asarray(1e-12))
    omega_hat = so3mat / safe_theta
    rodrigues = (
        b.eye(3) + b.sin(theta) * omega_hat
        + (1 - b.cos(theta)) * b.matmul(omega_hat, omega_hat)
    )
    return b.where(theta < 1e-12, b.eye(3), rodrigues)


def rotation_matrix_to_euler_angles(R) -> NDArray:
    """Convert a rotation matrix to ZYX roll, pitch, and yaw angles."""
    b = get_backend()
    R = b.asarray(R)
    assert R.shape == (3, 3), f"Expected 3x3 rotation matrix, got shape {R.shape}"
    sy = b.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    regular = b.stack((b.arctan2(R[2, 1], R[2, 2]),
                       b.arctan2(-R[2, 0], sy),
                       b.arctan2(R[1, 0], R[0, 0])))
    singular = b.stack((b.arctan2(-R[1, 2], R[1, 1]),
                        b.arctan2(-R[2, 0], sy), sy * 0))
    return b.where(sy < 1e-6, singular, regular)


def euler_to_rotation_matrix(euler_deg) -> NDArray:
    """Convert degree-valued ZYX Euler angles to a rotation matrix."""
    b = get_backend()
    angles = b.asarray(euler_deg) * (pi / 180.0)
    roll, pitch, yaw = angles[0], angles[1], angles[2]
    zero = roll * 0
    one = zero + 1
    Rz = b.stack((b.stack((b.cos(yaw), -b.sin(yaw), zero)),
                  b.stack((b.sin(yaw), b.cos(yaw), zero)),
                  b.stack((zero, zero, one))))
    Ry = b.stack((b.stack((b.cos(pitch), zero, b.sin(pitch))),
                  b.stack((zero, one, zero)),
                  b.stack((-b.sin(pitch), zero, b.cos(pitch)))))
    Rx = b.stack((b.stack((one, zero, zero)),
                  b.stack((zero, b.cos(roll), -b.sin(roll))),
                  b.stack((zero, b.sin(roll), b.cos(roll)))))
    return b.matmul(b.matmul(Rz, Ry), Rx)
