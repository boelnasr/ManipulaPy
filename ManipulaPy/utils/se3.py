#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SE(3) homogeneous-transform utilities.

Copyright (c) 2025 Mohamed Aboelnasr. Licensed under AGPL-3.0-or-later.
"""

from typing import Any, Tuple

from numpy.typing import NDArray

from ManipulaPy.backend import get_backend

from .so3 import MatrixExp3, VecToso3, rotation_logm, skew_symmetric


def _row(*values: Any) -> Any:
    return get_backend().stack(values)


def _homogeneous(R: Any, p: Any, last: float = 1.0) -> Any:
    b = get_backend()
    top = b.concatenate((R, p.reshape(3, 1)), axis=1)
    return b.concatenate((top, b.asarray([[0, 0, 0, last]], dtype=top.dtype)), axis=0)


def transform_from_twist(S, theta: float) -> NDArray:
    """Compute an SE(3) transform from a normalized screw and displacement."""
    b = get_backend()
    S = b.asarray(S)
    omega, v = S[:3], S[3:]
    omega_hat = skew_symmetric(omega)
    omega_hat2 = b.matmul(omega_hat, omega_hat)
    R = b.eye(3) + b.sin(theta) * omega_hat + (1 - b.cos(theta)) * omega_hat2
    G = b.eye(3) * theta + (1 - b.cos(theta)) * omega_hat + (theta - b.sin(theta)) * omega_hat2
    return _homogeneous(R, b.matmul(G, v))


def adjoint_transform(T) -> NDArray:
    """Compute the 6x6 adjoint of a homogeneous transform."""
    b = get_backend()
    T = b.asarray(T)
    R, p = T[:3, :3], T[:3, 3]
    zero = b.zeros((3, 3), dtype=R.dtype)
    return b.concatenate((b.concatenate((R, zero), axis=1),
                          b.concatenate((b.matmul(skew_symmetric(p), R), R), axis=1)), axis=0)


def _g_inverse(omega: Any, theta: Any) -> Any:
    b = get_backend()
    safe_theta = b.maximum(theta, b.asarray(1e-12))
    w_hat = skew_symmetric(omega)
    half_cot = b.sin(theta) / b.maximum(2 * (1 - b.cos(theta)), b.asarray(1e-12))
    return (b.eye(3) / safe_theta - 0.5 * w_hat
            + (1 / safe_theta - half_cot) * b.matmul(w_hat, w_hat))


def logm(T) -> NDArray:
    """Return the six-vector logarithm of a homogeneous transform."""
    b = get_backend()
    T = b.asarray(T)
    p = T[:3, 3]
    omega, theta = rotation_logm(T[:3, :3])
    v = theta * b.matmul(_g_inverse(omega, theta), p)
    rotational = b.concatenate((omega * theta, v))
    translation = b.concatenate((b.zeros(3, dtype=p.dtype), p))
    return b.where(theta < 1e-6, translation, rotational)


def se3ToVec(se3_matrix) -> NDArray:
    """Convert an se(3) matrix to a six-vector."""
    b = get_backend()
    se3_matrix = b.asarray(se3_matrix)
    if se3_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")
    omega = b.stack((se3_matrix[2, 1], se3_matrix[0, 2], se3_matrix[1, 0]))
    return b.concatenate((omega, se3_matrix[:3, 3]))


def TransToRp(T) -> Tuple[NDArray, NDArray]:
    """Split a homogeneous transform into rotation and position."""
    return T[:3, :3], T[:3, 3]


def TransInv(T) -> NDArray:
    """Invert a homogeneous transform."""
    b = get_backend()
    T = b.asarray(T)
    R, p = TransToRp(T)
    Rt = R.T
    return _homogeneous(Rt, -b.matmul(Rt, p))


def MatrixLog6(T) -> NDArray:
    """Compute the se(3) matrix logarithm of a homogeneous transform."""
    b = get_backend()
    T = b.asarray(T)
    R, p = TransToRp(T)
    omega, theta = rotation_logm(R)
    w_hat = skew_symmetric(omega)
    v = theta * b.matmul(_g_inverse(omega, theta), p)
    rotational = _homogeneous(theta * w_hat, v, last=0.0)
    translation = _homogeneous(b.zeros((3, 3), dtype=T.dtype), p, last=0.0)
    return b.where(theta < 1e-6, translation, rotational)


def MatrixExp6(se3mat) -> NDArray:
    """Compute the SE(3) exponential of an se(3) matrix."""
    b = get_backend()
    se3mat = b.asarray(se3mat)
    if se3mat.shape != (4, 4):
        raise ValueError("Input matrix must be of shape (4, 4)")
    omega_theta = b.stack((se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]))
    v = se3mat[:3, 3]
    theta = b.norm(omega_theta)
    safe_theta = b.maximum(theta, b.asarray(1e-12))
    omega_hat = se3mat[:3, :3] / safe_theta
    G = (b.eye(3) * theta + (1 - b.cos(theta)) * omega_hat
         + (theta - b.sin(theta)) * b.matmul(omega_hat, omega_hat))
    general = _homogeneous(MatrixExp3(se3mat[:3, :3]), b.matmul(G, v) / safe_theta)
    translation = _homogeneous(b.eye(3), v)
    return b.where(theta < 1e-6, translation, general)


def VecTose3(V) -> NDArray:
    """Convert a spatial velocity six-vector to an se(3) matrix."""
    b = get_backend()
    V = b.asarray(V)
    top = b.concatenate((VecToso3(V[:3]), V[3:].reshape(3, 1)), axis=1)
    return b.concatenate((top, b.zeros((1, 4), dtype=top.dtype)), axis=0)
