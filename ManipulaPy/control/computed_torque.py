#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Computed torque, feedforward, limit, and space-control concerns."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from . import manipulator_controller as _runtime

BackendArray = Any


class _ComputedTorqueConcern:
    """Computed torque, feedforward, limit, and space-control descriptors."""

    def computed_torque_control(
        self,
        thetalistd: BackendArray,
        dthetalistd: BackendArray,
        ddthetalistd: BackendArray,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        g: BackendArray,
        dt: float,
        Kp: BackendArray,
        Ki: BackendArray,
        Kd: BackendArray,
        i_clamp: Optional[float] = None,
    ) -> BackendArray:
        """
        Computed Torque Control.

        Inputs and guarded dynamics results follow the active backend.

        Parameters:
            thetalistd: Desired joint angles.
            dthetalistd: Desired joint velocities.
            ddthetalistd: Desired joint accelerations.
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            g: Gravity vector.
            dt: Time step.
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.

        Returns:
            NDArray: Torque command (NumPy array under the default backend).
        """
        thetalistd = _runtime._as_backend_array(thetalistd)
        dthetalistd = _runtime._as_backend_array(dthetalistd)
        ddthetalistd = _runtime._as_backend_array(ddthetalistd)
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        g = _runtime._as_backend_array(g)
        Kp = _runtime._as_backend_array(Kp)
        Ki = _runtime._as_backend_array(Ki)
        Kd = _runtime._as_backend_array(Kd)

        backend = _runtime.get_backend()
        self._normalize_state("eint")
        if self.eint is None or self.eint.shape != thetalist.shape:
            if self.eint is not None and self.eint.shape != thetalist.shape:
                _runtime.logger.debug(
                    "Controller state shape mismatch (%s vs %s); resetting integral.",
                    self.eint.shape,
                    thetalist.shape,
                )
            self._set_state(
                "eint", backend.zeros(thetalist.shape, dtype=backend.float64)
            )

        e = thetalistd - thetalist
        self._set_state("eint", self.eint + e * dt)
        i_clamp = _runtime._validate_i_clamp(i_clamp)
        if i_clamp is not None:
            self._set_state("eint", backend.clip(self.eint, -i_clamp, i_clamp))

        M = _runtime._as_backend_array(self.dynamics.mass_matrix(thetalist))
        tau = M @ (Kp * e + Ki * self.eint + Kd * (dthetalistd - dthetalist))
        tau = tau + _runtime._as_backend_array(
            self.dynamics.inverse_dynamics(
                thetalist,
                dthetalist,
                ddthetalistd,
                g,
                [0, 0, 0, 0, 0, 0],
            )
        )

        return tau

    def feedforward_control(
        self,
        desired_position: BackendArray,
        desired_velocity: BackendArray,
        desired_acceleration: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
    ) -> BackendArray:
        """
        Feedforward Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            desired_acceleration: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.

        Returns:
            NDArray: Feedforward torque (NumPy array under the default backend).
        """
        desired_position = _runtime._as_backend_array(desired_position)
        desired_velocity = _runtime._as_backend_array(desired_velocity)
        desired_acceleration = _runtime._as_backend_array(desired_acceleration)
        g = _runtime._as_backend_array(g)
        Ftip = _runtime._as_backend_array(Ftip)

        tau = _runtime._as_backend_array(
            self.dynamics.inverse_dynamics(
                desired_position,
                desired_velocity,
                desired_acceleration,
                g,
                Ftip,
            )
        )
        return tau

    def pd_feedforward_control(
        self,
        desired_position: BackendArray,
        desired_velocity: BackendArray,
        desired_acceleration: BackendArray,
        current_position: BackendArray,
        current_velocity: BackendArray,
        Kp: BackendArray,
        Kd: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
    ) -> BackendArray:
        """
        PD Feedforward Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            desired_acceleration: Desired joint accelerations.
            current_position: Current joint positions.
            current_velocity: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.

        Returns:
            NDArray: Control signal (NumPy array under the default backend).
        """
        # pd_control and feedforward_control now handle conversion internally
        pd_signal = self.pd_control(
            desired_position,
            desired_velocity,
            current_position,
            current_velocity,
            Kp,
            Kd,
        )
        ff_signal = self.feedforward_control(
            desired_position, desired_velocity, desired_acceleration, g, Ftip
        )
        control_signal = pd_signal + ff_signal
        return control_signal

    @staticmethod
    def enforce_limits(
        thetalist: BackendArray,
        dthetalist: BackendArray,
        tau: BackendArray,
        joint_limits: BackendArray,
        torque_limits: BackendArray,
    ) -> Tuple[BackendArray, BackendArray, BackendArray]:
        """
        Enforce joint and torque limits.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            tau: Torques.
            joint_limits: Joint angle limits.
            torque_limits: Torque limits.

        Returns:
            tuple: Clipped joint angles, velocities, and torques (NumPy by default).
        """
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        tau = _runtime._as_backend_array(tau)
        joint_limits = _runtime._as_backend_array(joint_limits)
        torque_limits = _runtime._as_backend_array(torque_limits)

        backend = _runtime.get_backend()
        thetalist = backend.clip(thetalist, joint_limits[:, 0], joint_limits[:, 1])
        tau = backend.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
        return thetalist, dthetalist, tau

    def joint_space_control(
        self,
        desired_joint_angles: BackendArray,
        current_joint_angles: BackendArray,
        current_joint_velocities: BackendArray,
        Kp: BackendArray,
        Kd: BackendArray,
    ) -> BackendArray:
        """
        Joint Space Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_joint_angles: Desired joint angles.
            current_joint_angles: Current joint angles.
            current_joint_velocities: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: Control torque (NumPy array under the default backend).
        """
        desired_joint_angles = _runtime._as_backend_array(desired_joint_angles)
        current_joint_angles = _runtime._as_backend_array(current_joint_angles)
        current_joint_velocities = _runtime._as_backend_array(current_joint_velocities)
        Kp = _runtime._as_backend_array(Kp)
        Kd = _runtime._as_backend_array(Kd)

        e = desired_joint_angles - current_joint_angles
        edot = 0 - current_joint_velocities
        tau = Kp * e + Kd * edot
        return tau

    def cartesian_space_control(
        self,
        desired_position: BackendArray,
        current_joint_angles: BackendArray,
        current_joint_velocities: BackendArray,
        Kp: BackendArray,
        Kd: BackendArray,
    ) -> BackendArray:
        """
        Cartesian Space Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_position: Desired end-effector position.
            current_joint_angles: Current joint angles.
            current_joint_velocities: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: Control torque (NumPy array under the default backend).
        """
        desired_position = _runtime._as_backend_array(desired_position)
        current_joint_angles = _runtime._as_backend_array(current_joint_angles)
        current_joint_velocities = _runtime._as_backend_array(current_joint_velocities)
        Kp = _runtime._as_backend_array(Kp)
        Kd = _runtime._as_backend_array(Kd)

        current_position = _runtime._as_backend_array(
            self.dynamics.forward_kinematics(current_joint_angles)
        )[:3, 3]
        e = desired_position - current_position
        # Position-only control: use linear (3xN) part of Jacobian
        J_v = _runtime._as_backend_array(self.dynamics.jacobian(current_joint_angles))[
            :3, :
        ]
        cartesian_velocity = J_v @ current_joint_velocities
        Kp_term = Kp @ e if Kp.ndim == 2 else Kp * e
        Kd_term = Kd @ cartesian_velocity if Kd.ndim == 2 else Kd * cartesian_velocity
        tau = J_v.T @ (Kp_term - Kd_term)
        return tau
