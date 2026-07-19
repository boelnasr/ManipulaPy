#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Control Module - ManipulaPy

This module provides various control algorithms for robotic manipulators including
PID, computed torque, adaptive, and robust control methods.

Control inputs, guarded dynamics results, and persistent controller state follow the
caller-selected active array backend; NumPy remains the default. Plotting and tuning
are explicit host-only boundaries, while response metrics return Python scalars after
backend-native array operations.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ManipulaPy.backend import get_backend, use_backend

from . import _as_backend_array, _to_host_array, _validate_i_clamp

BackendArray = Any


@dataclass(frozen=True)
class _StateOwner:
    """Backend placement recorded for one persistent state value."""

    backend: Any
    token: Any
    value: Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManipulatorController:
    """Manipulator controls with active-backend arrays and persistent state."""

    def __init__(self, dynamics: Any) -> None:
        """
        Initialize the ManipulatorController with the dynamics of the manipulator.

        Inputs and persistent state use the active backend selected by the caller.

        Parameters:
            dynamics (ManipulatorDynamics): An instance of ManipulatorDynamics.
        """
        self.dynamics = dynamics
        self.eint: Optional[BackendArray] = None
        self.parameter_estimate: Optional[BackendArray] = None
        self.P: Optional[BackendArray] = None
        self.x_hat: Optional[BackendArray] = None
        self._state_owners: Dict[str, _StateOwner] = {}

    def _normalize_state(self, name: str) -> Optional[BackendArray]:
        """Move persistent numeric state to the caller-selected backend."""
        value = getattr(self, name)
        if value is None:
            return None
        backend = get_backend()
        token = backend.cache_token()
        state_owners = getattr(self, "_state_owners", None)
        if state_owners is None:
            state_owners = {}
            self._state_owners = state_owners
        owner = state_owners.get(name)
        if owner is not None and owner.value is value:
            if owner.backend is not backend or owner.token != token:
                value = owner.backend.to_numpy(value)
        value = backend.asarray(value)
        setattr(self, name, value)
        state_owners[name] = _StateOwner(backend, token, value)
        return value

    def _set_state(self, name: str, value: BackendArray) -> BackendArray:
        """Store backend-native persistent numeric state."""
        if not hasattr(self, "_state_owners"):
            self._state_owners = {}
        setattr(self, name, value)
        backend = get_backend()
        self._state_owners[name] = _StateOwner(backend, backend.cache_token(), value)
        return value

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
        thetalistd = _as_backend_array(thetalistd)
        dthetalistd = _as_backend_array(dthetalistd)
        ddthetalistd = _as_backend_array(ddthetalistd)
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        g = _as_backend_array(g)
        Kp = _as_backend_array(Kp)
        Ki = _as_backend_array(Ki)
        Kd = _as_backend_array(Kd)

        backend = get_backend()
        self._normalize_state("eint")
        if self.eint is None or self.eint.shape != thetalist.shape:
            if self.eint is not None and self.eint.shape != thetalist.shape:
                logger.debug(
                    "Controller state shape mismatch (%s vs %s); resetting integral.",
                    self.eint.shape,
                    thetalist.shape,
                )
            self._set_state(
                "eint", backend.zeros(thetalist.shape, dtype=backend.float64)
            )

        e = thetalistd - thetalist
        self._set_state("eint", self.eint + e * dt)
        i_clamp = _validate_i_clamp(i_clamp)
        if i_clamp is not None:
            self._set_state("eint", backend.clip(self.eint, -i_clamp, i_clamp))

        M = _as_backend_array(self.dynamics.mass_matrix(thetalist))
        tau = M @ (Kp * e + Ki * self.eint + Kd * (dthetalistd - dthetalist))
        tau = tau + _as_backend_array(
            self.dynamics.inverse_dynamics(
                thetalist,
                dthetalist,
                ddthetalistd,
                g,
                [0, 0, 0, 0, 0, 0],
            )
        )

        return tau

    def pd_control(
        self,
        desired_position: BackendArray,
        desired_velocity: BackendArray,
        current_position: BackendArray,
        current_velocity: BackendArray,
        Kp: BackendArray,
        Kd: BackendArray,
    ) -> BackendArray:
        """
        PD Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            current_position: Current joint positions.
            current_velocity: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: PD signal (NumPy array under the default backend).
        """
        desired_position = _as_backend_array(desired_position)
        desired_velocity = _as_backend_array(desired_velocity)
        current_position = _as_backend_array(current_position)
        current_velocity = _as_backend_array(current_velocity)
        Kp = _as_backend_array(Kp)
        Kd = _as_backend_array(Kd)

        e = desired_position - current_position
        edot = desired_velocity - current_velocity
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(
        self,
        thetalistd: BackendArray,
        dthetalistd: BackendArray,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        dt: float,
        Kp: BackendArray,
        Ki: BackendArray,
        Kd: BackendArray,
        i_clamp: Optional[float] = None,
    ) -> BackendArray:
        """
        PID Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalistd: Desired joint angles.
            dthetalistd: Desired joint velocities.
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            dt: Time step.
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.

        Returns:
            NDArray: PID signal (NumPy array under the default backend).
        """
        thetalistd = _as_backend_array(thetalistd)
        dthetalistd = _as_backend_array(dthetalistd)
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        Kp = _as_backend_array(Kp)
        Ki = _as_backend_array(Ki)
        Kd = _as_backend_array(Kd)

        backend = get_backend()
        self._normalize_state("eint")
        if self.eint is None or self.eint.shape != thetalist.shape:
            if self.eint is not None and self.eint.shape != thetalist.shape:
                logger.debug(
                    "Controller state shape mismatch (%s vs %s); resetting integral.",
                    self.eint.shape,
                    thetalist.shape,
                )
            self._set_state(
                "eint", backend.zeros(thetalist.shape, dtype=backend.float64)
            )

        e = thetalistd - thetalist
        self._set_state("eint", self.eint + e * dt)
        i_clamp = _validate_i_clamp(i_clamp)
        if i_clamp is not None:
            self._set_state("eint", backend.clip(self.eint, -i_clamp, i_clamp))

        e_dot = dthetalistd - dthetalist
        tau = Kp * e + Ki * self.eint + Kd * e_dot
        return tau

    def robust_control(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        ddthetalist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        disturbance_estimate: BackendArray,
        adaptation_gain: BackendArray,
    ) -> BackendArray:
        """
        Robust Control.

        Inputs and guarded dynamics results follow the active backend.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            disturbance_estimate: Estimate of disturbances.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Robust torque (NumPy array under the default backend).
        """
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        ddthetalist = _as_backend_array(ddthetalist)
        g = _as_backend_array(g)
        Ftip = _as_backend_array(Ftip)
        disturbance_estimate = _as_backend_array(disturbance_estimate)
        adaptation_gain = _as_backend_array(adaptation_gain)

        M = _as_backend_array(self.dynamics.mass_matrix(thetalist))
        c = _as_backend_array(
            self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        )
        g_forces = _as_backend_array(self.dynamics.gravity_forces(thetalist, g))
        J_transpose = _as_backend_array(self.dynamics.jacobian(thetalist)).T
        tau = (
            M @ ddthetalist
            + c
            + g_forces
            + J_transpose @ Ftip
            + adaptation_gain * disturbance_estimate
        )
        return tau

    def adaptive_control(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        ddthetalist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        measurement_error: BackendArray,
        adaptation_gain: float,
    ) -> BackendArray:
        """
        Adaptive Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            measurement_error: Error in measurement.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Adaptive torque (NumPy array under the default backend).
        """
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        ddthetalist = _as_backend_array(ddthetalist)
        g = _as_backend_array(g)
        Ftip = _as_backend_array(Ftip)
        measurement_error = _as_backend_array(measurement_error)
        adaptation_gain = _as_backend_array(adaptation_gain)

        backend = get_backend()
        self._normalize_state("parameter_estimate")
        n = thetalist.size
        if self.parameter_estimate is None:
            self._set_state(
                "parameter_estimate", backend.zeros((n,), dtype=thetalist.dtype)
            )

        err = measurement_error.reshape(-1)
        gamma = adaptation_gain.reshape(-1)[0]

        self._set_state("parameter_estimate", self.parameter_estimate + gamma * err)

        M = _as_backend_array(self.dynamics.mass_matrix(thetalist))
        c = _as_backend_array(
            self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        )
        g_forces = _as_backend_array(self.dynamics.gravity_forces(thetalist, g))
        J_transpose = _as_backend_array(self.dynamics.jacobian(thetalist)).T

        tau = (
            M @ ddthetalist
            + c
            + g_forces
            + J_transpose @ Ftip
            + self.parameter_estimate
        )
        return tau

    def kalman_filter_predict(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        taulist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        dt: float,
        Q: BackendArray,
    ) -> None:
        """
        Kalman Filter Prediction.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            taulist: Applied torques.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            dt: Time step.
            Q: Process noise covariance.

        Returns:
            None
        """
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        taulist = _as_backend_array(taulist)
        g = _as_backend_array(g)
        Ftip = _as_backend_array(Ftip)
        Q = _as_backend_array(Q)
        backend = get_backend()
        self._normalize_state("x_hat")
        self._normalize_state("P")
        n = self.x_hat.shape[0] if self.x_hat is not None else len(thetalist) * 2
        if Q.shape != (n, n):
            raise ValueError(f"Q must have shape ({n}, {n}), got {Q.shape}")

        if self.x_hat is None:
            self._set_state("x_hat", backend.concatenate((thetalist, dthetalist)))

        thetalist_pred = (
            self.x_hat[: len(thetalist)] + self.x_hat[len(thetalist) :] * dt
        )
        dthetalist_pred = (
            _as_backend_array(
                self.dynamics.forward_dynamics(
                    self.x_hat[: len(thetalist)],
                    self.x_hat[len(thetalist) :],
                    taulist,
                    g,
                    Ftip,
                )
            )
            * dt
            + self.x_hat[len(thetalist) :]
        )
        x_hat_pred = backend.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self._set_state("P", backend.eye(len(x_hat_pred)))
        F = backend.eye(len(x_hat_pred))
        self._set_state("P", F @ self.P @ F.T + Q)

        self._set_state("x_hat", x_hat_pred)

    def kalman_filter_update(self, z: BackendArray, R: BackendArray) -> None:
        """
        Kalman Filter Update.

        Inputs follow the active backend selected by the caller.

        Parameters:
            z: Measurement vector.
            R: Measurement noise covariance.

        Returns:
            None
        """
        z = _as_backend_array(z)
        R = _as_backend_array(R)
        if self.x_hat is None:
            raise ValueError(
                "kalman_filter_update called before kalman_filter_predict; "
                "x_hat has not been initialized"
            )
        backend = get_backend()
        self._normalize_state("x_hat")
        n = self.x_hat.shape[0]
        if self.P is None:
            raise ValueError(
                f"P must be initialized with shape ({n}, {n}) before update; "
                "got None"
            )
        self._normalize_state("P")
        if getattr(self.P, "shape", None) != (n, n):
            raise ValueError(
                f"P must be initialized with shape ({n}, {n}) before update; "
                f"got {self.P.shape}"
            )
        if z.shape != (n,):
            raise ValueError(f"z must have shape ({n},) to match x_hat, got {z.shape}")
        if R.shape != (n, n):
            raise ValueError(f"R must have shape ({n}, {n}), got {R.shape}")

        H = backend.eye(len(self.x_hat))
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ backend.inv(S)
        self._set_state("x_hat", self.x_hat + K @ y)
        self._set_state("P", (backend.eye(len(self.x_hat)) - K @ H) @ self.P)

    def kalman_filter_control(
        self,
        thetalistd: BackendArray,
        dthetalistd: BackendArray,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        taulist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        dt: float,
        Q: BackendArray,
        R: BackendArray,
    ) -> Tuple[BackendArray, BackendArray]:
        """
        Kalman Filter Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalistd: Desired joint angles.
            dthetalistd: Desired joint velocities.
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            taulist: Applied torques.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            dt: Time step.
            Q: Process noise covariance.
            R: Measurement noise covariance.

        Returns:
            tuple: Estimated angles and velocities (NumPy arrays by default).
        """
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)

        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(get_backend().concatenate((thetalist, dthetalist)), R)
        return self.x_hat[: len(thetalist)], self.x_hat[len(thetalist) :]

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
        desired_position = _as_backend_array(desired_position)
        desired_velocity = _as_backend_array(desired_velocity)
        desired_acceleration = _as_backend_array(desired_acceleration)
        g = _as_backend_array(g)
        Ftip = _as_backend_array(Ftip)

        tau = _as_backend_array(
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
        thetalist = _as_backend_array(thetalist)
        dthetalist = _as_backend_array(dthetalist)
        tau = _as_backend_array(tau)
        joint_limits = _as_backend_array(joint_limits)
        torque_limits = _as_backend_array(torque_limits)

        backend = get_backend()
        thetalist = backend.clip(thetalist, joint_limits[:, 0], joint_limits[:, 1])
        tau = backend.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
        return thetalist, dthetalist, tau

    def plot_steady_state_response(
        self,
        time: Union[NDArray[np.float64], List[float]],
        response: Union[NDArray[np.float64], List[float]],
        set_point: float,
        title: str = "Steady State Response",
    ) -> None:
        """
        Plot the steady-state response of the controller.

        Parameters:
            time (np.ndarray): Array of time steps.
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.
            title (str, optional): Title of the plot.

        Returns:
            None
        """
        time = _to_host_array(time)
        response = _to_host_array(response)

        with use_backend("numpy"):
            plt.figure(figsize=(10, 5))
            plt.plot(time, response, label="Response")
            plt.axhline(y=set_point, color="r", linestyle="--", label="Set Point")

            # Calculate key metrics
            rise_time = self.calculate_rise_time(time, response, set_point)
            percent_overshoot = self.calculate_percent_overshoot(response, set_point)
            settling_time = self.calculate_settling_time(time, response, set_point)
            steady_state_error = self.calculate_steady_state_error(response, set_point)

            # Annotate metrics on the plot
            plt.axvline(
                x=rise_time,
                color="g",
                linestyle="--",
                label=f"Rise Time: {rise_time:.2f}s",
            )
            plt.axhline(
                y=set_point * (1 + percent_overshoot / 100),
                color="b",
                linestyle="--",
                label=f"Overshoot: {percent_overshoot:.2f}%",
            )
            plt.axvline(
                x=settling_time,
                color="m",
                linestyle="--",
                label=f"Settling Time: {settling_time:.2f}s",
            )
            plt.axhline(
                y=set_point + steady_state_error,
                color="c",
                linestyle="--",
                label=f"Steady State Error: {steady_state_error:.2f}",
            )

            plt.xlabel("Time (s)")
            plt.ylabel("Response")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()

    def calculate_rise_time(
        self,
        time: Union[NDArray[np.float64], List[float]],
        response: Union[NDArray[np.float64], List[float]],
        set_point: float,
    ) -> float:
        """
        Calculate the rise time.

        Parameters:
            time (np.ndarray): Array of time steps.
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Rise time.
        """
        backend = get_backend()
        time = backend.asarray(time)
        response = backend.asarray(response)

        rise_start = 0.1 * set_point
        rise_end = 0.9 * set_point
        start_mask = response >= rise_start
        end_mask = response >= rise_end
        if not bool(_to_host_array(backend.any(start_mask))) or not bool(
            _to_host_array(backend.any(end_mask))
        ):
            return float("inf")
        start_idx = int(_to_host_array(backend.argmax(start_mask)))
        end_idx = int(_to_host_array(backend.argmax(end_mask)))
        return float(_to_host_array(time[end_idx] - time[start_idx]))

    def calculate_percent_overshoot(
        self, response: Union[NDArray[np.float64], List[float]], set_point: float
    ) -> float:
        """
        Calculate the percent overshoot.

        Parameters:
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Percent overshoot.
        """
        if set_point == 0:
            return 0.0
        backend = get_backend()
        response = backend.asarray(response)
        max_response = backend.amax(response)
        return float(_to_host_array(((max_response - set_point) / set_point) * 100))

    def calculate_settling_time(
        self,
        time: Union[NDArray[np.float64], List[float]],
        response: Union[NDArray[np.float64], List[float]],
        set_point: float,
        tolerance: float = 0.02,
    ) -> float:
        """
        Calculate the settling time.

        Returns the first time at which the response enters the tolerance band
        and never leaves it again.  Returns ``float('inf')`` when the response
        never enters the band, or enters but does not remain settled through
        the end of the recorded data.

        Parameters:
            time (np.ndarray): Array of time steps.
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.
            tolerance (float): Fractional tolerance band (default 0.02 = 2 %).

        Returns:
            float: Settling time, or inf if the response never settles.
        """
        backend = get_backend()
        time = backend.asarray(time)
        response = backend.asarray(response)

        settling_threshold = abs(set_point) * tolerance
        in_band = backend.abs(response - set_point) <= settling_threshold
        if not bool(_to_host_array(backend.any(in_band))):
            return float("inf")
        in_band_host = _to_host_array(in_band)
        n = len(in_band_host)
        last_excursion = -1
        for i in range(n - 1, -1, -1):
            if not in_band_host[i]:
                last_excursion = i
                break
        if last_excursion == n - 1:
            return float("inf")  # never settles
        first_settled_idx = last_excursion + 1
        return float(_to_host_array(time[first_settled_idx]))

    def calculate_steady_state_error(
        self, response: Union[NDArray[np.float64], List[float]], set_point: float
    ) -> float:
        """
        Calculate the steady-state error.

        Parameters:
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Steady-state error.
        """
        response = get_backend().asarray(response)
        return float(_to_host_array(response[-1] - set_point))

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
        desired_joint_angles = _as_backend_array(desired_joint_angles)
        current_joint_angles = _as_backend_array(current_joint_angles)
        current_joint_velocities = _as_backend_array(current_joint_velocities)
        Kp = _as_backend_array(Kp)
        Kd = _as_backend_array(Kd)

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
        desired_position = _as_backend_array(desired_position)
        current_joint_angles = _as_backend_array(current_joint_angles)
        current_joint_velocities = _as_backend_array(current_joint_velocities)
        Kp = _as_backend_array(Kp)
        Kd = _as_backend_array(Kd)

        current_position = _as_backend_array(
            self.dynamics.forward_kinematics(current_joint_angles)
        )[:3, 3]
        e = desired_position - current_position
        # Position-only control: use linear (3xN) part of Jacobian
        J_v = _as_backend_array(self.dynamics.jacobian(current_joint_angles))[:3, :]
        cartesian_velocity = J_v @ current_joint_velocities
        Kp_term = Kp @ e if Kp.ndim == 2 else Kp * e
        Kd_term = Kd @ cartesian_velocity if Kd.ndim == 2 else Kd * cartesian_velocity
        tau = J_v.T @ (Kp_term - Kd_term)
        return tau

    # ------------------------------------------------------------------------
    def ziegler_nichols_tuning(
        self,
        Ku: Union[float, NDArray[np.float64], List[float]],
        Tu: Union[float, NDArray[np.float64], List[float]],
        kind: str = "PID",
    ) -> Tuple[
        Union[float, NDArray[np.float64]],
        Union[float, NDArray[np.float64]],
        Union[float, NDArray[np.float64]],
    ]:
        """
        Compute Ziegler-Nichols controller gains.

        Args:
            Ku: Ultimate gain as a scalar or vector.
            Tu: Ultimate period as a scalar or vector.
            kind: Controller type: ``"P"``, ``"PI"``, or ``"PID"``.

        Returns:
            Tuple of ``(Kp, Ki, Kd)`` gains.
        """
        Ku = _to_host_array(Ku).astype(float)
        kind = kind.upper()

        if kind == "P":
            Kp, Ki, Kd = 0.50 * Ku, 0.0 * Ku, 0.0 * Ku
        else:
            Tu = _to_host_array(Tu).astype(float)
            if not np.all(np.isfinite(Tu)) or np.any(Tu <= 0):
                raise ValueError(
                    f"Tu (ultimate period) must be positive and finite, got Tu={Tu!r}. "
                    "Tu == 0 typically indicates find_ultimate_gain_and_period found no "
                    "sustained oscillation; check your gain sweep."
                )

            if kind == "PI":
                Kp, Ki, Kd = 0.45 * Ku, 1.2 * Ku / Tu, 0.0 * Ku
            elif kind == "PID":
                Kp = 0.60 * Ku
                Ki = 2.0 * Kp / Tu
                Kd = 0.125 * Kp * Tu
            else:
                raise ValueError("kind must be 'P', 'PI' or 'PID'")

        # Return scalars as plain floats so assertEqual passes exactly
        if Ku.size == 1:
            return float(Kp), float(Ki), float(Kd)
        return Kp, Ki, Kd

    # ------------------------------------------------------------------------
    def tune_controller(
        self,
        Ku: Union[float, NDArray[np.float64], List[float]],
        Tu: Union[float, NDArray[np.float64], List[float]],
        kind: str = "PID",
    ) -> Tuple[
        Union[float, NDArray[np.float64]],
        Union[float, NDArray[np.float64]],
        Union[float, NDArray[np.float64]],
    ]:
        """
        Convenience wrapper that logs and returns NumPy arrays (length = DOF).
        """
        Kp, Ki, Kd = self.ziegler_nichols_tuning(Ku, Tu, kind)
        logger.info(f"Tuned Z-N ({kind}) gains\n  Kp={Kp}\n  Ki={Ki}\n  Kd={Kd}")
        return Kp, Ki, Kd

    # ------------------------------------------------------------------------
    def find_ultimate_gain_and_period(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        desired_joint_angles: Union[NDArray[np.float64], List[float]],
        dt: float,
        max_steps: int = 1000,
    ) -> Tuple[float, float, List[float], List[NDArray[np.float64]]]:
        """
        Find the ultimate gain and period using the Ziegler–Nichols method.

        This optimizer is an explicit NumPy host boundary.

        Parameters:
            thetalist: Initial joint angles (shape [6]).
            desired_joint_angles: Step target angles (shape [6]).
            dt: Simulation time step.
            max_steps: Number of integration steps to try.

        Returns:
            tuple:
              - ultimate_gain (float)
              - ultimate_period (float)
              - gain_history (list of float)
              - error_history (list of np.ndarray)
        """
        thetalist = _to_host_array(thetalist)
        desired_joint_angles = _to_host_array(desired_joint_angles)

        with use_backend("numpy"):
            Kp = 0.01
            increase = 1.1
            oscillation = False
            gain_history = []
            error_history = []

            while not oscillation and Kp < 1000:
                theta = thetalist.copy()
                omega = np.zeros_like(theta)
                self._set_state("eint", np.zeros_like(theta))
                errors = []

                for step in range(max_steps):
                    # pure-PD poke
                    tau = self.pd_control(
                        desired_joint_angles,
                        np.zeros_like(theta),
                        theta,
                        omega,
                        Kp,
                        0.0,
                    )
                    # alpha = M⁻¹ (tau – C – G)
                    M = self.dynamics.mass_matrix(theta)
                    C = self.dynamics.velocity_quadratic_forces(theta, omega)
                    Gf = self.dynamics.gravity_forces(theta, np.array([0, 0, -9.81]))
                    alpha = np.linalg.solve(M, tau - C - Gf)

                    omega += alpha * dt
                    theta += omega * dt

                    err = np.linalg.norm(theta - desired_joint_angles)
                    errors.append(err)
                    # blow-up guard
                    if step > 10 and err > 1e10:
                        break

                gain_history.append(Kp)
                error_history.append(np.array(errors))

                # look for the first upward slope after initial increase
                if len(errors) >= 2 and errors[-2] < errors[-1] < errors[-2] * 1.2:
                    oscillation = True
                else:
                    Kp *= increase

            ultimate_gain = float(Kp)
            ultimate_period = (max_steps * dt) / max(
                1, np.count_nonzero(np.diff(np.sign(error_history[-1]))) // 2
            )

            return ultimate_gain, ultimate_period, gain_history, error_history
