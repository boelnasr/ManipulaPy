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


def _exp6_trans_coeffs(b: Any, theta_sq: Any) -> Tuple[Any, Any]:
    """Return the SE(3) translation coefficients ``((1-cos th)/th^2, (th-sin th)/th^3)``.

    Both are even functions of ``theta`` (smooth in ``theta^2``, so no ``sqrt``
    feeds the graph near zero) and therefore have a finite gradient through
    ``theta = 0``. Near the identity (``theta^2 < 1e-4``, i.e. ``theta < 1e-2``)
    the Taylor series is used; elsewhere the exact form with ``theta`` clamped
    away from zero so the inactive branch's gradient stays finite. The band is set
    by where the exact forms' *backward* pass stops cancelling catastrophically
    (the ``(th - sin th)/th^3`` gradient error is ~128 at ``theta = 1e-6`` and only
    falls below ~1e-10 near ``theta ~ 1e-2``), not merely where the primal is
    finite; the too-small ``1e-12`` cutoff left the exact backward corrupt just
    above it.
    """
    near_zero = theta_sq < 1e-4
    theta_safe = b.sqrt(b.maximum(theta_sq, b.asarray(1e-12)))
    c1_exact = (1 - b.cos(theta_safe)) / (theta_safe * theta_safe)
    c2_exact = (theta_safe - b.sin(theta_safe)) / (theta_safe * theta_safe * theta_safe)
    c1_taylor = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0
    c2_taylor = 1.0 / 6.0 - theta_sq / 120.0 + theta_sq * theta_sq / 5040.0
    return b.where(near_zero, c1_taylor, c1_exact), b.where(near_zero, c2_taylor, c2_exact)


def MatrixExp6(se3mat) -> NDArray:
    """Compute the SE(3) exponential of an se(3) matrix."""
    b = get_backend()
    se3mat = b.asarray(se3mat)
    if se3mat.shape != (4, 4):
        raise ValueError("Input matrix must be of shape (4, 4)")
    omega_theta = b.stack((se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]))
    v = se3mat[:3, 3]
    K = se3mat[:3, :3]
    theta_sq = (
        omega_theta[0] * omega_theta[0]
        + omega_theta[1] * omega_theta[1]
        + omega_theta[2] * omega_theta[2]
    )
    # Rotational part reuses the (locally smooth) SO(3) exponential; translation
    # is p = [I + C1*K + C2*K^2] v with C1 = (1-cos th)/th^2, C2 = (th-sin th)/th^3.
    # Folding the small-angle branch into the Taylor-safe coefficients (instead of
    # a where onto a constant) keeps dT/dtheta correct and finite at theta = 0.
    coeff_c1, coeff_c2 = _exp6_trans_coeffs(b, theta_sq)
    k_v = b.matmul(K, v)
    p = v + coeff_c1 * k_v + coeff_c2 * b.matmul(K, k_v)
    return _homogeneous(MatrixExp3(K), p)


def VecTose3(V) -> NDArray:
    """Convert a spatial velocity six-vector to an se(3) matrix."""
    b = get_backend()
    V = b.asarray(V)
    top = b.concatenate((VecToso3(V[:3]), V[3:].reshape(3, 1)), axis=1)
    return b.concatenate((top, b.zeros((1, 4), dtype=top.dtype)), axis=0)
