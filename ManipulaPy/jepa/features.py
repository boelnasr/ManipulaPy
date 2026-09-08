#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Robot state featurization for JEPA - ManipulaPy

Turns raw manipulator state ``(theta, dtheta)`` into the fixed-width vectors the
JEPA encoder consumes. The interesting part is that the geometric half of the
feature vector is *exact*: end-effector position and orientation come from
ManipulaPy's product-of-exponentials forward kinematics rather than from
something the network has to learn. The model therefore spends its capacity on
dynamics, not on rediscovering the robot's own kinematic chain.

Feature layout, for an ``n``-joint arm::

    [0    : n  ]  cos(theta)
    [n    : 2n ]  sin(theta)
    [2n   : 3n ]  dtheta * velocity_scale
    [3n   : 3n+3]  end-effector position          (include_ee_pose only)
    [3n+3 : 3n+9]  end-effector rotation, 6-D form (include_ee_pose only)

Joint angles enter as ``(cos, sin)`` so the representation is continuous across
the +/-pi wrap, and orientation uses the continuous 6-D parameterization (the
first two columns of the rotation matrix) for the same reason.

Copyright (c) 2025 Mohamed Aboelnasr

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Any, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["RobotStateFeaturizer"]


class RobotStateFeaturizer:
    """Map ``(theta, dtheta)`` to a JEPA feature vector.

    Args:
        serial_manipulator: A :class:`~ManipulaPy.kinematics.SerialManipulator`
            (or anything exposing ``forward_kinematics``). Required when
            ``include_ee_pose`` is True; its ``joint_limits`` also supplies
            ``num_joints`` when that argument is omitted.
        num_joints: Degrees of freedom. Inferred from ``serial_manipulator``
            when not given.
        include_ee_pose: Append the exact end-effector position and 6-D
            rotation computed by forward kinematics.
        velocity_scale: Multiplier applied to joint velocities. Set it to
            roughly ``1 / max_velocity`` so velocities land in the same
            numeric range as the trigonometric features.
        frame: Frame passed to ``forward_kinematics``, ``"space"`` or
            ``"body"``.

    Example:
        >>> from ManipulaPy.jepa import RobotStateFeaturizer
        >>> feat = RobotStateFeaturizer(robot)             # doctest: +SKIP
        >>> feat.featurize(theta, dtheta).shape            # doctest: +SKIP
        (27,)
    """

    def __init__(
        self,
        serial_manipulator: Optional[Any] = None,
        num_joints: Optional[int] = None,
        *,
        include_ee_pose: bool = True,
        velocity_scale: float = 1.0,
        frame: str = "space",
    ) -> None:
        if include_ee_pose and serial_manipulator is None:
            raise ValueError(
                "include_ee_pose=True requires a serial_manipulator to run "
                "forward kinematics; pass one or set include_ee_pose=False"
            )
        if frame not in ("space", "body"):
            raise ValueError(f"frame must be 'space' or 'body', got {frame!r}")

        if num_joints is None:
            num_joints = self._infer_num_joints(serial_manipulator)
        if num_joints is None or num_joints <= 0:
            raise ValueError("num_joints could not be inferred; pass it explicitly")

        self.serial_manipulator = serial_manipulator
        self.num_joints = int(num_joints)
        self.include_ee_pose = bool(include_ee_pose)
        self.velocity_scale = float(velocity_scale)
        self.frame = frame

    @staticmethod
    def _infer_num_joints(serial_manipulator: Optional[Any]) -> Optional[int]:
        """Best-effort DOF lookup from a manipulator's declared structure."""
        if serial_manipulator is None:
            return None
        limits = getattr(serial_manipulator, "joint_limits", None)
        if limits is not None:
            return len(limits)
        s_list = getattr(serial_manipulator, "S_list", None)
        if s_list is not None:
            return int(np.asarray(s_list).shape[1])
        return None

    @property
    def state_dim(self) -> int:
        """Width of one feature vector."""
        return 3 * self.num_joints + (9 if self.include_ee_pose else 0)

    def joint_feature_indices(self, joint: int) -> List[int]:
        """Return the feature indices owned by ``joint``.

        Used by the masking utilities to hide a whole joint (its angle and its
        velocity) rather than an arbitrary slice of the feature vector. The
        end-effector block belongs to no single joint and is never returned.
        """
        if not 0 <= joint < self.num_joints:
            raise IndexError(f"joint {joint} out of range for {self.num_joints} joints")
        n = self.num_joints
        return [joint, n + joint, 2 * n + joint]

    def featurize(
        self,
        theta: Sequence[float],
        dtheta: Optional[Sequence[float]] = None,
    ) -> NDArray[np.float64]:
        """Featurize a single state.

        Args:
            theta: Joint positions, shape ``(num_joints,)``.
            dtheta: Joint velocities, shape ``(num_joints,)``. Zeros if None.

        Returns:
            Feature vector of shape ``(state_dim,)``, float64.
        """
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape[0] != self.num_joints:
            raise ValueError(
                f"theta has {theta_arr.shape[0]} entries, "
                f"expected {self.num_joints}"
            )
        if dtheta is None:
            dtheta_arr = np.zeros(self.num_joints, dtype=np.float64)
        else:
            dtheta_arr = np.asarray(dtheta, dtype=np.float64).reshape(-1)
            if dtheta_arr.shape[0] != self.num_joints:
                raise ValueError(
                    f"dtheta has {dtheta_arr.shape[0]} entries, "
                    f"expected {self.num_joints}"
                )

        parts = [
            np.cos(theta_arr),
            np.sin(theta_arr),
            dtheta_arr * self.velocity_scale,
        ]
        if self.include_ee_pose:
            parts.append(self._ee_features(theta_arr))
        return np.concatenate(parts).astype(np.float64, copy=False)

    def featurize_batch(
        self,
        thetas: NDArray[np.float64],
        dthetas: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Featurize a stack of states.

        Args:
            thetas: Joint positions, shape ``(T, num_joints)``.
            dthetas: Joint velocities, shape ``(T, num_joints)``. Zeros if
                None.

        Returns:
            Features of shape ``(T, state_dim)``, float64.
        """
        thetas_arr = np.atleast_2d(np.asarray(thetas, dtype=np.float64))
        if dthetas is None:
            dthetas_arr = np.zeros_like(thetas_arr)
        else:
            dthetas_arr = np.atleast_2d(np.asarray(dthetas, dtype=np.float64))
        if dthetas_arr.shape != thetas_arr.shape:
            raise ValueError(
                f"dthetas shape {dthetas_arr.shape} does not match thetas "
                f"shape {thetas_arr.shape}"
            )

        out = np.empty((thetas_arr.shape[0], self.state_dim), dtype=np.float64)
        for i in range(thetas_arr.shape[0]):
            out[i] = self.featurize(thetas_arr[i], dthetas_arr[i])
        return out

    def _ee_features(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Exact end-effector position and 6-D rotation from FK."""
        T = np.asarray(
            self.serial_manipulator.forward_kinematics(theta, frame=self.frame),
            dtype=np.float64,
        )
        position = T[:3, 3]
        # Continuous 6-D rotation: the first two columns of R. Recovering the
        # third by cross product is exact for a proper rotation, so nothing is
        # lost relative to carrying all nine entries.
        rot6d = T[:3, :2].reshape(-1, order="F")
        return np.concatenate([position, rot6d])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_joints={self.num_joints}, "
            f"state_dim={self.state_dim}, "
            f"include_ee_pose={self.include_ee_pose}, "
            f"velocity_scale={self.velocity_scale})"
        )
