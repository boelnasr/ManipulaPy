#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from .dynamics import ManipulatorDynamics


class ManipulatorController:
    def __init__(self, dynamics):
        """
        Initialize the ManipulatorController with the dynamics of the manipulator.

        Parameters:
            dynamics (ManipulatorDynamics): An instance of ManipulatorDynamics.
        """
        self.dynamics = dynamics
        self.eint = None
        self.parameter_estimate = None
        self.P = None
        self.x_hat = None

    def computed_torque_control(
        self,
        thetalistd,
        dthetalistd,
        ddthetalistd,
        thetalist,
        dthetalist,
        g,
        dt,
        Kp,
        Ki,
        Kd,
    ):
        """
        Computed Torque Control.

        Parameters:
            thetalistd (np.ndarray): Desired joint angles.
            dthetalistd (np.ndarray): Desired joint velocities.
            ddthetalistd (np.ndarray): Desired joint accelerations.
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            g (np.ndarray): Gravity vector.
            dt (float): Time step.
            Kp (np.ndarray): Proportional gain.
            Ki (np.ndarray): Integral gain.
            Kd (np.ndarray): Derivative gain.

        Returns:
            np.ndarray: Torque command.
        """
        if self.eint is None:
            self.eint = np.zeros_like(thetalist)

        e = np.subtract(thetalistd, thetalist)
        self.eint += e * dt

        M = self.dynamics.mass_matrix(thetalist)
        tau = np.dot(
            M, Kp * e + Ki * self.eint + Kd * np.subtract(dthetalistd, dthetalist)
        )
        tau += self.dynamics.inverse_dynamics(
            thetalist, dthetalist, ddthetalistd, g, [0, 0, 0, 0, 0, 0]
        )

        return tau

    def pd_control(
        self,
        desired_position,
        desired_velocity,
        current_position,
        current_velocity,
        Kp,
        Kd,
    ):
        """
        PD Control.

        Parameters:
            desired_position (np.ndarray): Desired joint positions.
            desired_velocity (np.ndarray): Desired joint velocities.
            current_position (np.ndarray): Current joint positions.
            current_velocity (np.ndarray): Current joint velocities.
            Kp (np.ndarray): Proportional gain.
            Kd (np.ndarray): Derivative gain.

        Returns:
            np.ndarray: PD control signal.
        """
        e = np.subtract(desired_position, current_position)
        edot = np.subtract(desired_velocity, current_velocity)
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(
        self, thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd
    ):
        """
        PID Control.

        Parameters:
            thetalistd (np.ndarray): Desired joint angles.
            dthetalistd (np.ndarray): Desired joint velocities.
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            dt (float): Time step.
            Kp (np.ndarray): Proportional gain.
            Ki (np.ndarray): Integral gain.
            Kd (np.ndarray): Derivative gain.

        Returns:
            np.ndarray: PID control signal.
        """
        if self.eint is None:
            self.eint = np.zeros_like(thetalist)

        e = np.subtract(thetalistd, thetalist)
        self.eint += e * dt

        e_dot = np.subtract(dthetalistd, dthetalist)
        tau = Kp * e + Ki * self.eint + Kd * e_dot
        return tau

    def robust_control(
        self,
        thetalist,
        dthetalist,
        ddthetalist,
        g,
        Ftip,
        disturbance_estimate,
        adaptation_gain,
    ):
        """
        Robust Control.

        Parameters:
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            ddthetalist (np.ndarray): Desired joint accelerations.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.
            disturbance_estimate (np.ndarray): Estimate of disturbances.
            adaptation_gain (float): Gain for the adaptation term.

        Returns:
            np.ndarray: Robust control torque.
        """
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T
        tau = (
            np.dot(M, ddthetalist)
            + c
            + g_forces
            + np.dot(J_transpose, Ftip)
            + adaptation_gain * disturbance_estimate
        )
        return tau

    def adaptive_control(
        self,
        thetalist,
        dthetalist,
        ddthetalist,
        g,
        Ftip,
        measurement_error,
        adaptation_gain,
    ):
        """
        Adaptive Control.

        Parameters:
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            ddthetalist (np.ndarray): Desired joint accelerations.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.
            measurement_error (np.ndarray): Error in measurement.
            adaptation_gain (float): Gain for the adaptation term.

        Returns:
            np.ndarray: Adaptive control torque.
        """
        if self.parameter_estimate is None:
            self.parameter_estimate = np.zeros_like(self.dynamics.Glist)

        self.parameter_estimate += adaptation_gain * measurement_error
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T
        tau = (
            np.dot(M, ddthetalist)
            + c
            + g_forces
            + np.dot(J_transpose, Ftip)
            + self.parameter_estimate
        )
        return tau

    def kalman_filter_predict(self, thetalist, dthetalist, taulist, g, Ftip, dt, Q):
        """
        Kalman Filter Prediction.

        Parameters:
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            taulist (np.ndarray): Applied torques.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.
            dt (float): Time step.
            Q (np.ndarray): Process noise covariance.

        Returns:
            None
        """
        if self.x_hat is None:
            self.x_hat = np.concatenate((thetalist, dthetalist))

        thetalist_pred = (
            self.x_hat[: len(thetalist)] + self.x_hat[len(thetalist) :] * dt
        )
        dthetalist_pred = (
            self.dynamics.forward_dynamics(
                self.x_hat[: len(thetalist)],
                self.x_hat[len(thetalist) :],
                taulist,
                g,
                Ftip,
            )
            * dt
            + self.x_hat[len(thetalist) :]
        )
        x_hat_pred = np.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self.P = np.eye(len(x_hat_pred))
        F = np.eye(len(x_hat_pred))
        self.P = np.dot(F, np.dot(self.P, F.T)) + Q

        self.x_hat = x_hat_pred

    def kalman_filter_update(self, z, R):
        """
        Kalman Filter Update.

        Parameters:
            z (np.ndarray): Measurement vector.
            R (np.ndarray): Measurement noise covariance.

        Returns:
            None
        """
        H = np.eye(len(self.x_hat))
        y = z - np.dot(H, self.x_hat)
        S = np.dot(H, np.dot(self.P, H.T)) + R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        self.x_hat += np.dot(K, y)
        self.P = np.dot(np.eye(len(self.x_hat)) - np.dot(K, H), self.P)

    def kalman_filter_control(
        self, thetalistd, dthetalistd, thetalist, dthetalist, taulist, g, Ftip, dt, Q, R
    ):
        """
        Kalman Filter Control.

        Parameters:
            thetalistd (np.ndarray): Desired joint angles.
            dthetalistd (np.ndarray): Desired joint velocities.
            thetalist (np.ndarray): Current joint angles.
            dthetalist (np.ndarray): Current joint velocities.
            taulist (np.ndarray): Applied torques.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.
            dt (float): Time step.
            Q (np.ndarray): Process noise covariance.
            R (np.ndarray): Measurement noise covariance.

        Returns:
            tuple: Estimated joint angles and velocities.
        """
        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(np.concatenate((thetalist, dthetalist)), R)
        return self.x_hat[: len(thetalist)], self.x_hat[len(thetalist) :]

    def feedforward_control(
        self, desired_position, desired_velocity, desired_acceleration, g, Ftip
    ):
        """
        Feedforward Control.

        Parameters:
            desired_position (np.ndarray): Desired joint positions.
            desired_velocity (np.ndarray): Desired joint velocities.
            desired_acceleration (np.ndarray): Desired joint accelerations.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.

        Returns:
            np.ndarray: Feedforward torque.
        """
        tau = self.dynamics.inverse_dynamics(
            desired_position, desired_velocity, desired_acceleration, g, Ftip
        )
        return tau

    def pd_feedforward_control(
        self,
        desired_position,
        desired_velocity,
        desired_acceleration,
        current_position,
        current_velocity,
        Kp,
        Kd,
        g,
        Ftip,
    ):
        """
        PD Feedforward Control.

        Parameters:
            desired_position (np.ndarray): Desired joint positions.
            desired_velocity (np.ndarray): Desired joint velocities.
            desired_acceleration (np.ndarray): Desired joint accelerations.
            current_position (np.ndarray): Current joint positions.
            current_velocity (np.ndarray): Current joint velocities.
            Kp (np.ndarray): Proportional gain.
            Kd (np.ndarray): Derivative gain.
            g (np.ndarray): Gravity vector.
            Ftip (np.ndarray): External forces applied at the end effector.

        Returns:
            np.ndarray: Control signal.
        """
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
    def enforce_limits(thetalist, dthetalist, tau, joint_limits, torque_limits):
        """
        Enforce joint and torque limits.

        Parameters:
            thetalist (np.ndarray): Joint angles.
            dthetalist (np.ndarray): Joint velocities.
            tau (np.ndarray): Torques.
            joint_limits (np.ndarray): Joint angle limits.
            torque_limits (np.ndarray): Torque limits.

        Returns:
            tuple: Clipped joint angles, velocities, and torques.
        """
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
        steady_state_error = response[-1] - set_point
        return steady_state_error

    def joint_space_control(
        self,
        desired_joint_angles,
        current_joint_angles,
        current_joint_velocities,
        Kp,
        Kd,
    ):
        """
        Joint Space Control.

        Parameters:
            desired_joint_angles (np.ndarray): Desired joint angles.
            current_joint_angles (np.ndarray): Current joint angles.
            current_joint_velocities (np.ndarray): Current joint velocities.
            Kp (np.ndarray): Proportional gain.
            Kd (np.ndarray): Derivative gain.

        Returns:
            np.ndarray: Control torque.
        """
        e = np.subtract(desired_joint_angles, current_joint_angles)
        edot = np.subtract(0, current_joint_velocities)
        tau = Kp * e + Kd * edot
        return tau

    def cartesian_space_control(
        self,
        desired_position,
        current_joint_angles,
        current_joint_velocities,
        Kp,
        Kd,
    ):
        """
        Cartesian Space Control.

        Parameters:
            desired_position (np.ndarray): Desired end-effector position.
            current_joint_angles (np.ndarray): Current joint angles.
            current_joint_velocities (np.ndarray): Current joint velocities.
            Kp (np.ndarray): Proportional gain.
            Kd (np.ndarray): Derivative gain.

        Returns:
            np.ndarray: Control torque.
        """
        current_position = self.dynamics.forward_kinematics(current_joint_angles)[:3, 3]
        e = np.subtract(desired_position, current_position)
        dthetalist = current_joint_velocities
        J = self.dynamics.jacobian(current_joint_angles)
        tau = np.dot(J.T, Kp * e - Kd * np.dot(J, dthetalist))
        return tau
