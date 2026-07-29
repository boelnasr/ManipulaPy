#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SE(3) homogeneous-transform utilities.

Copyright (c) 2025 Mohamed Aboelnasr. Licensed under AGPL-3.0-or-later.
"""

from typing import Any, Tuple

from numpy.typing import NDArray

from ManipulaPy.backend import get_backend

from .so3 import (
    MatrixExp3,
    MatrixLog3,
    VecToso3,
    skew_symmetric,
    skew_symmetric_to_vector,
)


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


def _log6_phi_sq_coeff(b: Any, theta_sq: Any) -> Any:
    """Return ``a = (1 - (theta/2) cot(theta/2)) / theta^2``, with ``a(0) = 1/12``.

    This is the ``Phi^2`` coefficient of ``theta * G^-1`` when the rotational
    logarithm is carried as the matrix ``Phi = theta * w_hat`` rather than as a
    separate unit axis and angle. Written as an even function of ``theta``
    (smooth in ``theta^2``, so no ``sqrt`` feeds the graph near zero); the exact
    form is a vanishing/vanishing ratio as ``theta -> 0``, so below
    ``theta^2 < 1e-2`` (``theta < 0.1``) the four-term series
    ``1/12 + theta^2/720 + theta^4/30240 + theta^6/1209600`` is used instead.

    That band is deliberately WIDER than the ``theta^2 < 1e-4`` one
    ``_exp6_trans_coeffs`` uses, because this coefficient's exact form is a
    double cancellation: ``1 - cos theta`` loses about five digits at
    ``theta = 1e-2``, and the numerator ``1 - theta sin theta / (2 (1 - cos
    theta))`` is itself ``theta^2/12`` recovered by subtracting two nearly equal
    quantities. Its *derivative* is degraded well before its value is, so
    switching at ``1e-2`` handed the coefficient's gradient to a branch that had
    already lost roughly four percent of it. Four series terms hold the Taylor
    side to ~1e-13 relative accuracy out to ``theta = 0.1``, by which point the
    exact form has recovered.

    The clamp inside the exact branch sits strictly BELOW the switch point:
    clamping at the switch value makes ``maximum`` tie exactly where the exact
    branch becomes active, and autodiff splits a tie's derivative in half.
    """
    small = (
        1 / 12
        + theta_sq / 720
        + theta_sq**2 / 30240
        + theta_sq**3 / 1209600
    )
    theta = b.sqrt(b.maximum(theta_sq, b.asarray(5e-3)))
    exact = (
        1 - theta * b.sin(theta) / b.maximum(2 * (1 - b.cos(theta)), b.asarray(1e-300))
    ) / (theta * theta)
    return b.where(theta_sq < 1e-2, small, exact)


def _theta_g_inverse(b: Any, phi: Any) -> Any:
    """Return ``theta * G^-1``, smooth and finite through zero.

    ``G^-1`` alone diverges like ``1/theta``, but the product both logarithms
    actually need does not: it is ``I - Phi/2 + a(theta^2) Phi^2`` and tends to
    the identity, which is why the pure-translation result falls out of the
    general formula rather than needing a separate branch.

    Everything is derived from ``Phi = MatrixLog3(R)``, including
    ``theta^2 = |Phi|_F^2 / 2``. Taking the axis and angle from
    ``rotation_logm`` instead would reintroduce its thresholded axis, which is
    zeroed below theta = 1e-6 and would flatten this whole expression to the
    identity there -- losing the ``-Phi/2`` term that carries the translational
    correction and its derivative.
    """
    theta_sq = b.sum(phi * phi) / 2
    coeff = _log6_phi_sq_coeff(b, theta_sq)
    return b.eye(3) - 0.5 * phi + coeff * b.matmul(phi, phi)


def logm(T) -> NDArray:
    """Return the six-vector logarithm of a homogeneous transform."""
    b = get_backend()
    T = b.asarray(T)
    p = T[:3, 3]
    # Everything derives from Phi = MatrixLog3(R), for both blocks.
    # ``rotation_logm`` zeroes its axis below theta = 1e-6 (the axis is genuinely
    # undefined at the identity), so any expression built from that axis loses a
    # small-but-real rotation and its derivative at the origin. MatrixLog3
    # evaluates 0.5 (theta/sin theta) (R - R.T) through a Taylor-safe
    # coefficient and is smooth in value and gradient there.
    #
    # No small-angle branch on the translation either: ``theta * G^-1`` -> I as
    # theta -> 0, so the general formula already yields the pure-translation
    # result that the removed ``where`` used to select.
    phi = MatrixLog3(T[:3, :3])
    rotvec = skew_symmetric_to_vector(phi)
    return b.concatenate((rotvec, b.matmul(_theta_g_inverse(b, phi), p)))


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
    # See ``logm``: both blocks derive from Phi = MatrixLog3(R), which is smooth
    # through the identity, and the general translation formula already
    # degenerates to the pure-translation result at theta = 0, so branching on a
    # small angle only served to discard the rotation and its gradient.
    phi = MatrixLog3(R)
    v = b.matmul(_theta_g_inverse(b, phi), p)
    return _homogeneous(phi, v, last=0.0)


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
