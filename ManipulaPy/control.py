#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Control Module - ManipulaPy

This module provides various control algorithms for robotic manipulators including
PID, computed torque, adaptive, and robust control methods.

Note: All control methods use CPU-based NumPy computation to avoid GPU-CPU transfer
overhead. Since the dynamics module operates on NumPy arrays, keeping everything on
the CPU is significantly more efficient than repeated PCIe transfers between GPU and
CPU memory spaces.

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
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import logging

# Optional CuPy import for defensive array handling
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_numpy(arr):
    """
    Safely convert array to NumPy, handling both NumPy and CuPy arrays.

    Args:
        arr: Input array (can be NumPy array, CuPy array, or list)

    Returns:
        NumPy array

    Note:
        This is necessary because np.asarray() does not work with CuPy arrays.
        CuPy raises "Implicit conversion to a NumPy array is not allowed"
        to prevent accidental performance issues. We must explicitly call .get()
        to transfer CuPy arrays from GPU to CPU.
    """
    if CUPY_AVAILABLE and cp is not None:
        try:
            if isinstance(arr, cp.ndarray):
                # CuPy array: explicitly transfer from GPU to CPU
                return arr.get()
        except (TypeError, AttributeError):
            # cp.ndarray may not be a real type when CuPy is mocked; treat as non-CuPy
            pass

    # NumPy array, list, or other: convert to NumPy
    return np.asarray(arr)


class ManipulatorController:
    def __init__(self, dynamics: Any) -> None:
        """
        Initialize the ManipulatorController with the dynamics of the manipulator.

        Note: Control algorithms now use CPU (NumPy) to avoid GPU-CPU transfer
        overhead, since the dynamics module operates on NumPy arrays.

        Parameters:
            dynamics (ManipulatorDynamics): An instance of ManipulatorDynamics.
        """
        self.dynamics = dynamics
        self.eint: Optional[NDArray[np.float64]] = None
        self.parameter_estimate: Optional[NDArray[np.float64]] = None
        self.P: Optional[NDArray[np.float64]] = None
        self.x_hat: Optional[NDArray[np.float64]] = None

    def computed_torque_control(
        self,
        thetalistd: Union[NDArray[np.float64], List[float]],
        dthetalistd: Union[NDArray[np.float64], List[float]],
        ddthetalistd: Union[NDArray[np.float64], List[float]],
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        dt: float,
        Kp: Union[NDArray[np.float64], List[float]],
        Ki: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        Computed Torque Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.
        The dynamics module operates on NumPy arrays, so keeping everything
        on CPU is more efficient than repeated GPU↔CPU transfers.

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
            NDArray: Torque command (CPU-based NumPy array).
        """
        # Convert to NumPy arrays (CPU) - avoid GPU↔CPU transfer bottleneck
        # Use _to_numpy() to safely handle both NumPy and CuPy arrays
        thetalistd = _to_numpy(thetalistd)
        dthetalistd = _to_numpy(dthetalistd)
        ddthetalistd = _to_numpy(ddthetalistd)
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        g = _to_numpy(g)
        Kp = _to_numpy(Kp)
        Ki = _to_numpy(Ki)
        Kd = _to_numpy(Kd)

        if self.eint is None:
            self.eint = np.zeros_like(thetalist)

        e = thetalistd - thetalist
        self.eint += e * dt

        # Dynamics computations (no GPU↔CPU transfers)
        M = self.dynamics.mass_matrix(thetalist)
        tau = M @ (Kp * e + Ki * self.eint + Kd * (dthetalistd - dthetalist))
        tau += self.dynamics.inverse_dynamics(
            thetalist,
            dthetalist,
            ddthetalistd,
            g,
            [0, 0, 0, 0, 0, 0],
        )

        return tau

    def pd_control(
        self,
        desired_position: Union[NDArray[np.float64], List[float]],
        desired_velocity: Union[NDArray[np.float64], List[float]],
        current_position: Union[NDArray[np.float64], List[float]],
        current_velocity: Union[NDArray[np.float64], List[float]],
        Kp: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        PD Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            current_position: Current joint positions.
            current_velocity: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: PD control signal (CPU-based NumPy array).
        """
        desired_position = _to_numpy(desired_position)
        desired_velocity = _to_numpy(desired_velocity)
        current_position = _to_numpy(current_position)
        current_velocity = _to_numpy(current_velocity)
        Kp = _to_numpy(Kp)
        Kd = _to_numpy(Kd)

        e = desired_position - current_position
        edot = desired_velocity - current_velocity
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(
        self,
        thetalistd: Union[NDArray[np.float64], List[float]],
        dthetalistd: Union[NDArray[np.float64], List[float]],
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        dt: float,
        Kp: Union[NDArray[np.float64], List[float]],
        Ki: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        PID Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

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
            NDArray: PID control signal (CPU-based NumPy array).
        """
        thetalistd = _to_numpy(thetalistd)
        dthetalistd = _to_numpy(dthetalistd)
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        Kp = _to_numpy(Kp)
        Ki = _to_numpy(Ki)
        Kd = _to_numpy(Kd)

        if self.eint is None:
            self.eint = np.zeros_like(thetalist)

        e = thetalistd - thetalist
        self.eint += e * dt

        e_dot = dthetalistd - dthetalist
        tau = Kp * e + Ki * self.eint + Kd * e_dot
        return tau

    def robust_control(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        ddthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
        disturbance_estimate: Union[NDArray[np.float64], List[float]],
        adaptation_gain: float,
    ) -> NDArray[np.float64]:
        """
        Robust Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            disturbance_estimate: Estimate of disturbances.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Robust control torque (CPU-based NumPy array).
        """
        # Convert to NumPy arrays (CPU) - avoid GPU↔CPU transfer bottleneck
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        ddthetalist = _to_numpy(ddthetalist)
        g = _to_numpy(g)
        Ftip = _to_numpy(Ftip)
        disturbance_estimate = _to_numpy(disturbance_estimate)

        # Dynamics computations (no GPU↔CPU transfers)
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T
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
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        ddthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
        measurement_error: Union[NDArray[np.float64], List[float]],
        adaptation_gain: float,
    ) -> NDArray[np.float64]:
        """
        Adaptive Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            measurement_error: Error in measurement.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Adaptive control torque (CPU-based NumPy array).
        """
        # Convert to NumPy arrays (CPU) - avoid GPU↔CPU transfer bottleneck
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        ddthetalist = _to_numpy(ddthetalist)
        g = _to_numpy(g)
        Ftip = _to_numpy(Ftip)
        measurement_error = _to_numpy(measurement_error)

        # ---- parameter update (make it 1-D, same length as joints) ----
        n = thetalist.size
        if getattr(self, "parameter_estimate", None) is None:
            self.parameter_estimate = np.zeros((n,), dtype=thetalist.dtype)

        err = measurement_error.reshape(-1)        # (n,) - already NumPy from _to_numpy() above
        # Handle both scalar and array adaptation_gain
        gamma = float(np.atleast_1d(adaptation_gain).ravel()[0])

        # simple gradient-like update
        self.parameter_estimate = self.parameter_estimate + gamma * err

        # ---- standard torque computation (no GPU↔CPU transfers) ----
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T

        tau = M @ ddthetalist + c + g_forces + J_transpose @ Ftip + self.parameter_estimate
        return tau


    def kalman_filter_predict(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        taulist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
        dt: float,
        Q: NDArray[np.float64]
    ) -> None:
        """
        Kalman Filter Prediction.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

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
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        taulist = _to_numpy(taulist)
        g = _to_numpy(g)
        Ftip = _to_numpy(Ftip)
        Q = _to_numpy(Q)

        if self.x_hat is None:
            self.x_hat = np.concatenate((thetalist, dthetalist))

        thetalist_pred = (
            self.x_hat[: len(thetalist)] + self.x_hat[len(thetalist):] * dt
        )
        dthetalist_pred = (
            self.dynamics.forward_dynamics(
                self.x_hat[: len(thetalist)],
                self.x_hat[len(thetalist):],
                taulist,
                g,
                Ftip,
            )
            * dt
            + self.x_hat[len(thetalist):]
        )
        x_hat_pred = np.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self.P = np.eye(len(x_hat_pred))
        F = np.eye(len(x_hat_pred))
        self.P = F @ self.P @ F.T + Q

        self.x_hat = x_hat_pred

    def kalman_filter_update(
        self,
        z: Union[NDArray[np.float64], List[float]],
        R: NDArray[np.float64]
    ) -> None:
        """
        Kalman Filter Update.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            z: Measurement vector.
            R: Measurement noise covariance.

        Returns:
            None
        """
        z = _to_numpy(z)
        R = _to_numpy(R)

        H = np.eye(len(self.x_hat))
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x_hat += K @ y
        self.P = (np.eye(len(self.x_hat)) - K @ H) @ self.P

    def kalman_filter_control(
        self,
        thetalistd: Union[NDArray[np.float64], List[float]],
        dthetalistd: Union[NDArray[np.float64], List[float]],
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        taulist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
        dt: float,
        Q: NDArray[np.float64],
        R: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Kalman Filter Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

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
            tuple: Estimated joint angles and velocities (CPU-based NumPy arrays).
        """
        # Convert to NumPy (predictions and updates already handle this, but for consistency)
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)

        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(np.concatenate((thetalist, dthetalist)), R)
        return self.x_hat[: len(thetalist)], self.x_hat[len(thetalist):]

    def feedforward_control(
        self,
        desired_position: Union[NDArray[np.float64], List[float]],
        desired_velocity: Union[NDArray[np.float64], List[float]],
        desired_acceleration: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        Feedforward Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            desired_acceleration: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.

        Returns:
            NDArray: Feedforward torque (CPU-based NumPy array).
        """
        desired_position = _to_numpy(desired_position)
        desired_velocity = _to_numpy(desired_velocity)
        desired_acceleration = _to_numpy(desired_acceleration)
        g = _to_numpy(g)
        Ftip = _to_numpy(Ftip)

        tau = self.dynamics.inverse_dynamics(
            desired_position,
            desired_velocity,
            desired_acceleration,
            g,
            Ftip,
        )
        return tau

    def pd_feedforward_control(
        self,
        desired_position: Union[NDArray[np.float64], List[float]],
        desired_velocity: Union[NDArray[np.float64], List[float]],
        desired_acceleration: Union[NDArray[np.float64], List[float]],
        current_position: Union[NDArray[np.float64], List[float]],
        current_velocity: Union[NDArray[np.float64], List[float]],
        Kp: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        PD Feedforward Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

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
            NDArray: Control signal (CPU-based NumPy array).
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
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        tau: Union[NDArray[np.float64], List[float]],
        joint_limits: Union[cp.ndarray, NDArray[np.float64], List[Tuple[float, float]]],
        torque_limits: Union[cp.ndarray, NDArray[np.float64], List[Tuple[float, float]]]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Enforce joint and torque limits.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            tau: Torques.
            joint_limits: Joint angle limits.
            torque_limits: Torque limits.

        Returns:
            tuple: Clipped joint angles, velocities, and torques (CPU-based NumPy arrays).
        """
        thetalist = _to_numpy(thetalist)
        dthetalist = _to_numpy(dthetalist)
        tau = _to_numpy(tau)
        joint_limits = _to_numpy(joint_limits)
        torque_limits = _to_numpy(torque_limits)

        thetalist = np.clip(thetalist, joint_limits[:, 0], joint_limits[:, 1])
        tau = np.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
        return thetalist, dthetalist, tau

    def plot_steady_state_response(
        self, time, response, set_point, title="Steady State Response"
    ):
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
        time = _to_numpy(time)
        response = _to_numpy(response)

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
            x=rise_time, color="g", linestyle="--", label=f"Rise Time: {rise_time:.2f}s"
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

    def calculate_rise_time(self, time, response, set_point):
        """
        Calculate the rise time.

        Parameters:
            time (np.ndarray): Array of time steps.
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Rise time.
        """
        time = _to_numpy(time)
        response = _to_numpy(response)

        rise_start = 0.1 * set_point
        rise_end = 0.9 * set_point
        start_idx = np.where(response >= rise_start)[0][0]
        end_idx = np.where(response >= rise_end)[0][0]
        rise_time = time[end_idx] - time[start_idx]
        return rise_time

    def calculate_percent_overshoot(self, response, set_point):
        """
        Calculate the percent overshoot.

        Parameters:
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Percent overshoot.
        """
        response = _to_numpy(response)

        max_response = np.max(response)
        percent_overshoot = ((max_response - set_point) / set_point) * 100
        return percent_overshoot

    def calculate_settling_time(self, time, response, set_point, tolerance=0.02):
        """
        Calculate the settling time.

        Parameters:
            time (np.ndarray): Array of time steps.
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.
            tolerance (float): Tolerance for settling time calculation.

        Returns:
            float: Settling time.
        """
        time = _to_numpy(time)
        response = _to_numpy(response)

        settling_threshold = set_point * tolerance
        settling_idx = np.where(np.abs(response - set_point) <= settling_threshold)[0]
        settling_time = time[settling_idx[-1]] if len(settling_idx) > 0 else time[-1]
        return settling_time

    def calculate_steady_state_error(self, response, set_point):
        """
        Calculate the steady-state error.

        Parameters:
            response (np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Steady-state error.
        """
        response = _to_numpy(response)

        steady_state_error = response[-1] - set_point
        return steady_state_error

    def joint_space_control(
        self,
        desired_joint_angles: Union[NDArray[np.float64], List[float]],
        current_joint_angles: Union[NDArray[np.float64], List[float]],
        current_joint_velocities: Union[NDArray[np.float64], List[float]],
        Kp: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        Joint Space Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            desired_joint_angles: Desired joint angles.
            current_joint_angles: Current joint angles.
            current_joint_velocities: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: Control torque (CPU-based NumPy array).
        """
        desired_joint_angles = _to_numpy(desired_joint_angles)
        current_joint_angles = _to_numpy(current_joint_angles)
        current_joint_velocities = _to_numpy(current_joint_velocities)
        Kp = _to_numpy(Kp)
        Kd = _to_numpy(Kd)

        e = desired_joint_angles - current_joint_angles
        edot = 0 - current_joint_velocities
        tau = Kp * e + Kd * edot
        return tau

    def cartesian_space_control(
        self,
        desired_position: Union[NDArray[np.float64], List[float]],
        current_joint_angles: Union[NDArray[np.float64], List[float]],
        current_joint_velocities: Union[NDArray[np.float64], List[float]],
        Kp: Union[NDArray[np.float64], List[float]],
        Kd: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
        """
        Cartesian Space Control.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

        Parameters:
            desired_position: Desired end-effector position.
            current_joint_angles: Current joint angles.
            current_joint_velocities: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: Control torque (CPU-based NumPy array).
        """
        desired_position = _to_numpy(desired_position)
        current_joint_angles = _to_numpy(current_joint_angles)
        current_joint_velocities = _to_numpy(current_joint_velocities)
        Kp = _to_numpy(Kp)
        Kd = _to_numpy(Kd)

        current_position = self.dynamics.forward_kinematics(current_joint_angles)[:3, 3]
        e = desired_position - current_position
        dthetalist = current_joint_velocities
        J = self.dynamics.jacobian(current_joint_angles)
        tau = J.T @ (Kp * e - Kd @ J @ dthetalist)
        return tau
# ------------------------------------------------------------------------
    def ziegler_nichols_tuning(self, Ku, Tu, kind="PID"):
        Ku = _to_numpy(Ku).astype(float)
        Tu = _to_numpy(Tu).astype(float)

        kind = kind.upper()
        if kind == "P":
            Kp, Ki, Kd = 0.50 * Ku, 0.0 * Ku, 0.0 * Ku
        elif kind == "PI":
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
    def tune_controller(self, Ku, Tu, kind="PID"):
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
        max_steps: int = 1000
    ) -> Tuple[float, float, List[float], List[NDArray[np.float64]]]:
        """
        Find the ultimate gain and period using the Ziegler–Nichols method.

        Uses CPU-based computation to avoid GPU-CPU transfer overhead.

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
        thetalist = _to_numpy(thetalist)
        desired_joint_angles = _to_numpy(desired_joint_angles)

        Kp = 0.01
        increase = 1.1
        oscillation = False
        gain_history = []
        error_history = []

        while not oscillation and Kp < 1000:
            theta = thetalist.copy()
            omega = np.zeros_like(theta)
            self.eint = np.zeros_like(theta)
            errors = []

            for step in range(max_steps):
                # pure-PD poke
                tau = self.pd_control(
                    desired_joint_angles,
                    np.zeros_like(theta),
                    theta,
                    omega,
                    Kp,
                    0.0
                )
                # alpha = M⁻¹ (tau – C – G)
                M  = self.dynamics.mass_matrix(theta)
                C  = self.dynamics.velocity_quadratic_forces(theta, omega)
                Gf = self.dynamics.gravity_forces(theta, np.array([0, 0, -9.81]))
                alpha  = np.linalg.solve(M, tau - C - Gf)

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

        ultimate_gain   = float(Kp)
        ultimate_period = (max_steps * dt) / max(1,
            np.count_nonzero(np.diff(np.sign(error_history[-1])) ) // 2
        )

        return ultimate_gain, ultimate_period, gain_history, error_history
