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
    """Extract a stable half-turn axis without value-dependent Python branches.

    ``B = (R + R.T)/2 - cos(theta) I`` equals ``(1 - cos theta) n n^T`` *exactly*
    for every ``theta`` (the antisymmetric ``sin theta [n]x`` part cancels), so
    every column of ``B`` is parallel to the axis ``n`` with no ``O(pi - theta)``
    contamination and a correct, finite gradient. Building the candidate from
    ``R + I`` instead only equals ``2 n n^T`` *at* ``theta = pi``; across the
    near-pi band it leaks an ``O(pi - theta)`` axis error whose derivative is
    wrong, corrupting the traced backward pass. At ``theta = pi`` both forms
    coincide (``B = R + I = 2 n n^T``), so the value is unchanged.

    A single ``B`` column only fixes the axis up to the sign of its own component
    (``sign(n_j) n``); the exact log on the open interval ``(0, pi)`` is ``+n``, so
    the overall sign is taken from the antisymmetric part
    ``vee(R - R.T) = 2 sin(theta) n`` (whose sign is ``sign(n)`` there). The
    ``>= 0`` convention reproduces the column's own sign at exactly ``theta = pi``
    (where ``vee = 0``), matching the historical half-turn value. The sign is a
    locally constant multiplier, so it does not perturb the gradient.
    """
    b = get_backend()
    eps = b.asarray(1e-12)
    cos_theta = b.clip((b.trace(R) - 1) / 2, -1.0, 1.0)
    sym = 0.5 * (R + R.T) - cos_theta * b.eye(3)
    vee = b.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    c2 = b.stack((sym[0, 2], sym[1, 2], sym[2, 2]))
    c1 = b.stack((sym[0, 1], sym[1, 1], sym[2, 1]))
    c0 = b.stack((sym[0, 0], sym[1, 0], sym[2, 0]))
    use2 = sym[2, 2] >= 1e-6
    use1 = (sym[1, 1] >= 1e-6) & ~use2
    candidate = b.where(use2, c2, b.where(use1, c1, c0))
    axis = candidate / b.maximum(b.norm(candidate), eps)
    sign_ref = b.where(use2, vee[2], b.where(use1, vee[1], vee[0]))
    sign = b.where(sign_ref >= 0, b.asarray(1.0), b.asarray(-1.0))
    return sign * axis


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
    # atan2 on the (linear-in-angle) sine keeps theta and its derivative exact at
    # both ends of the range; arccos loses half its digits near a half turn.
    theta = _log_angle(b, R, cos_theta)
    vee = b.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    generic = vee / b.maximum(2 * b.sin(theta), b.asarray(1e-12))
    # vee/(2 sin theta) is a vanishing/vanishing cancellation as theta -> pi, so
    # its derivative degrades well before the singularity (~1e-5 axis-gradient
    # error at pi - 2e-6). _pi_axis is exact for every theta and cancellation-free,
    # so switch to it across the same (pi - 1e-2, pi] band MatrixLog3 uses.
    axis = b.where(theta > pi - 1e-2, _pi_axis(R), generic)
    axis = b.where(theta < 1e-6, b.zeros(3), axis)
    if isinstance(b, NumpyBackend):
        return axis, float(theta)
    return axis, theta


def skew_symmetric_to_vector(matrix) -> NDArray:
    """Convert a skew-symmetric matrix to its three-vector."""
    b = get_backend()
    return b.stack((matrix[2, 1], matrix[0, 2], matrix[1, 0]))


def _theta_over_sin_theta(b: Any, cos_theta: Any) -> Any:
    """Return ``theta / sin(theta)`` as a locally smooth function of ``cos(theta)``.

    The rotation vector of a matrix logarithm is ``0.5 * (theta / sin theta) *
    (R - R.T)``. Written naively as ``arccos(cos_theta) / sin(arccos(cos_theta))``
    this coefficient is ``0/0`` at the identity and its gradient carries the
    ``d arccos/dx = -1/sqrt(1 - x^2)`` blow-up at ``x = 1``. Both would corrupt
    the backward pass of any traced backend even though the primal is finite.

    Near the identity (``cos_theta > 1 - 5e-5``, i.e. ``theta < ~1e-2``) the
    coefficient is evaluated with a Taylor series in ``u = 1 - cos_theta``
    (``1 + u/3 + 4u^2/45``, accurate to ~1e-13 over the band), which is smooth and
    has a finite gradient. The band is set by where the exact form's *backward*
    pass stops cancelling catastrophically (its gradient error falls below ~1e-10
    only near ``theta ~ 1e-2``), not merely where the primal is finite. Elsewhere
    the exact form is used, but its ``arccos``/``sqrt`` inputs are clamped away
    from the ``cos_theta = 1`` singularity so the inactive Taylor-region gradient
    stays finite (masking a value does not mask the gradient of a singular
    subexpression).
    """
    u = 1.0 - cos_theta
    near_zero = cos_theta > 1 - 5e-5
    # Clamp BOTH endpoints: at an exact half turn about a coordinate axis
    # cos_theta is exactly -1, so an unclamped arccos(-1) in this (inactive)
    # generic branch has an infinite derivative, and ``where`` propagates
    # 0 * inf = NaN into the selected half-turn branch.
    c_safe = b.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    exact = b.arccos(c_safe) / b.sqrt(b.maximum(1 - c_safe * c_safe, b.asarray(1e-30)))
    taylor = 1.0 + u / 3.0 + u * u * (4.0 / 45.0)
    return b.where(near_zero, taylor, exact)


def _log_angle(b: Any, R: Any, cos_theta: Any) -> Any:
    """Return the rotation angle ``theta`` in ``[0, pi]``, conditioned for autograd.

    ``arccos(cos_theta)`` is the textbook form but it is catastrophically
    ill-conditioned at BOTH ends of the range, because ``cos_theta`` is quadratic
    in the distance to the endpoint: near a half turn ``cos_theta = -1 + g^2/2``
    for ``g = pi - theta``, so at ``g = 1e-7`` it sits ~22 ulp from ``-1`` and
    ``d theta / d cos_theta = 1 / sin theta = 1e7`` amplifies that to a ~5%
    gradient error (and to a *zero* gradient by ``g = 1e-9``).

    ``sin theta = |vee(R - R.T)| / 2`` is instead LINEAR in ``g``, so
    ``atan2(sin theta, cos_theta)`` recovers ``theta`` to full precision with an
    exact derivative at every angle. ``atan2``'s partials
    (``cos/(s^2+c^2)``, ``-sin/(s^2+c^2)``) are bounded on the unit circle, so --
    unlike ``arccos`` -- no clamping is needed to keep the identity- and half-turn
    ends finite. The squared norm is floored so the ``sqrt`` gradient stays finite
    at an exact half turn, where ``vee`` vanishes and the log is a true branch
    point.
    """
    vee = b.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    sin_theta = b.sqrt(b.maximum(b.sum(vee * vee), b.asarray(1e-300))) / 2
    return b.arctan2(sin_theta, cos_theta)


def MatrixLog3(R) -> NDArray:
    """Compute the matrix logarithm of a rotation matrix."""
    b = get_backend()
    R = b.asarray(R)
    cos_theta = b.clip((b.trace(R) - 1) / 2, -1.0, 1.0)
    theta = _log_angle(b, R, cos_theta)
    skew_part = R - R.T
    # 0.5 * (theta/sin theta) * (R - R.T): the small-angle branch is folded into
    # the Taylor-safe coefficient so the rotation vector is locally smooth (finite
    # gradient) through the identity, not just finite in value.
    generic = 0.5 * _theta_over_sin_theta(b, cos_theta) * skew_part
    # The generic 0.5 * (theta/sin theta) * (R - R.T) form is smooth in value but
    # numerically unstable in the backward pass as theta -> pi: the coefficient
    # theta/sin theta diverges while vee(R - R.T) -> 0, so the gradient is a large
    # cancellation (error ~1e-7 at pi-1e-3, ~2e-2 at pi-1e-5). Near pi the
    # half-turn form theta * n with n from the symmetric part (see _pi_axis) is
    # value-identical (theta * n is the exact log there) yet cancellation-free, so
    # switch to it across the whole (pi - 1e-2, pi] band, not just at pi.
    half_turn = VecToso3(theta * _pi_axis(R))
    return b.where(theta > pi - 1e-2, half_turn, generic)


def VecToso3(omega) -> NDArray:
    """Convert a 3D angular velocity vector to an so(3) matrix."""
    return skew_symmetric(omega)


def _exp3_coeffs(b: Any, theta_sq: Any) -> Tuple[Any, Any]:
    """Return the Rodrigues coefficients ``(sin th/th, (1-cos th)/th^2)``.

    Written as smooth functions of ``theta^2`` (an even function of ``theta``, so
    no ``sqrt`` feeds the graph near zero) both coefficients have a finite
    gradient through ``theta = 0``. Near the identity (``theta^2 < 1e-4``, i.e.
    ``theta < 1e-2``) the Taylor series is used; elsewhere the exact form, with its
    ``theta`` clamped away from zero so the inactive exact branch's gradient stays
    finite (masking a value does not mask the gradient of a singular
    subexpression). The band is set by where the exact ``(1 - cos)/theta^2``
    *backward* pass stops cancelling catastrophically (its gradient error is ~89
    at ``theta = 1e-6`` and only falls below ~1e-10 near ``theta ~ 1e-2``), not
    merely where the primal is finite.
    """
    near_zero = theta_sq < 1e-4
    theta_safe = b.sqrt(b.maximum(theta_sq, b.asarray(1e-12)))
    a_exact = b.sin(theta_safe) / theta_safe
    b_exact = (1 - b.cos(theta_safe)) / (theta_safe * theta_safe)
    a_taylor = 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0
    b_taylor = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0
    return b.where(near_zero, a_taylor, a_exact), b.where(near_zero, b_taylor, b_exact)


def MatrixExp3(so3mat) -> NDArray:
    """Compute the SO(3) exponential with the Rodrigues formula."""
    b = get_backend()
    so3mat = b.asarray(so3mat)
    omega_theta = skew_symmetric_to_vector(so3mat)
    theta_sq = (
        omega_theta[0] * omega_theta[0]
        + omega_theta[1] * omega_theta[1]
        + omega_theta[2] * omega_theta[2]
    )
    # R = I + A*K + B*K^2 with K = so3mat, A = sin th/th, B = (1-cos th)/th^2.
    # Folding the small-angle branch into the Taylor-safe coefficients (instead
    # of a where onto a constant identity) keeps the rotation locally smooth so
    # dR/dtheta is correct and finite at exactly theta = 0.
    coeff_a, coeff_b = _exp3_coeffs(b, theta_sq)
    return b.eye(3) + coeff_a * so3mat + coeff_b * b.matmul(so3mat, so3mat)


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
