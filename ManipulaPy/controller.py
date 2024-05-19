#!/usr/bin/env python3

import numpy as np
from .dynamics import ManipulatorDynamics
import matplotlib.pyplot as plt

class ManipulatorController:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.eint = None
        self.parameter_estimate = None
        self.P = None
        self.x_hat = None

    def computed_torque_control(self, thetalistd, dthetalistd, ddthetalistd, thetalist, dthetalist, g, dt, Kp, Ki, Kd):
        """
        Computed Torque Control
        """
        if self.eint is None:
            self.eint = np.zeros_like(thetalist)
        
        e = np.subtract(thetalistd, thetalist)
        self.eint += e * dt
        
        M = self.dynamics.mass_matrix(thetalist)
        tau = np.dot(M, Kp * e + Ki * self.eint + Kd * np.subtract(dthetalistd, dthetalist))
        tau += self.dynamics.inverse_dynamics(thetalist, dthetalist, ddthetalistd, g, [0, 0, 0, 0, 0, 0])
        
        return tau

    def pd_control(self, desired_position, desired_velocity, current_position, current_velocity, Kp, Kd):
        """
        PD Control
        """
        e = np.subtract(desired_position, current_position)
        edot = np.subtract(desired_velocity, current_velocity)
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(self, thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd):
        """
        PID Control
        """
        if self.eint is None:
            self.eint = np.zeros_like(thetalist)
        
        e = np.subtract(thetalistd, thetalist)
        self.eint += e * dt
        
        e_dot = np.subtract(dthetalistd, dthetalist)
        tau = Kp * e + Ki * self.eint + Kd * e_dot
        return tau

    def robust_control(self, thetalist, dthetalist, ddthetalist, g, Ftip, disturbance_estimate, adaptation_gain):
        """
        Robust Control
        """
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T
        tau = np.dot(M, ddthetalist) + c + g_forces + np.dot(J_transpose, Ftip) + adaptation_gain * disturbance_estimate
        return tau

    def adaptive_control(self, thetalist, dthetalist, ddthetalist, g, Ftip, measurement_error, adaptation_gain):
        """
        Adaptive Control
        """
        if self.parameter_estimate is None:
            self.parameter_estimate = np.zeros_like(self.dynamics.Glist)
            
        self.parameter_estimate += adaptation_gain * measurement_error
        M = self.dynamics.mass_matrix(thetalist)
        c = self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.dynamics.gravity_forces(thetalist, g)
        J_transpose = self.dynamics.jacobian(thetalist).T
        tau = np.dot(M, ddthetalist) + c + g_forces + np.dot(J_transpose, Ftip) + self.parameter_estimate
        return tau

    def kalman_filter_predict(self, thetalist, dthetalist, taulist, g, Ftip, dt, Q):
        """
        Kalman Filter Prediction
        """
        if self.x_hat is None:
            self.x_hat = np.concatenate((thetalist, dthetalist))

        thetalist_pred = self.x_hat[:len(thetalist)] + self.x_hat[len(thetalist):] * dt
        dthetalist_pred = self.dynamics.forward_dynamics(self.x_hat[:len(thetalist)], self.x_hat[len(thetalist):], taulist, g, Ftip) * dt + self.x_hat[len(thetalist):]
        x_hat_pred = np.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self.P = np.eye(len(x_hat_pred))
        F = np.eye(len(x_hat_pred))
        self.P = np.dot(F, np.dot(self.P, F.T)) + Q

        self.x_hat = x_hat_pred

    def kalman_filter_update(self, z, R):
        """
        Kalman Filter Update
        """
        H = np.eye(len(self.x_hat))
        y = z - np.dot(H, self.x_hat)
        S = np.dot(H, np.dot(self.P, H.T)) + R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        self.x_hat += np.dot(K, y)
        self.P = np.dot(np.eye(len(self.x_hat)) - np.dot(K, H), self.P)

    def kalman_filter_control(self, thetalistd, dthetalistd, thetalist, dthetalist, taulist, g, Ftip, dt, Q, R):
        """
        Kalman Filter Control
        """
        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(np.concatenate((thetalist, dthetalist)), R)
        return self.x_hat[:len(thetalist)], self.x_hat[len(thetalist):]

    def feedforward_control(self, desired_position, desired_velocity, desired_acceleration, g, Ftip):
        """
        Computes the feedforward torque required to achieve the desired position, velocity, and acceleration, given the robot dynamics and external forces.
        
        Args:
            desired_position (numpy.ndarray): The desired joint positions.
            desired_velocity (numpy.ndarray): The desired joint velocities.
            desired_acceleration (numpy.ndarray): The desired joint accelerations.
            g (numpy.ndarray): The gravitational acceleration vector.
            Ftip (numpy.ndarray): The external force/torque vector applied at the end-effector.
        
        Returns:
            numpy.ndarray: The feedforward torque required to achieve the desired motion.
        """
        
        tau = self.dynamics.inverse_dynamics(desired_position, desired_velocity, desired_acceleration, g, Ftip)
        return tau

    def pd_feedforward_control(self, desired_position, desired_velocity, desired_acceleration, current_position, current_velocity, Kp, Kd, g, Ftip):
        """
        Computes the control signal for a PD (Proportional-Derivative) feedback controller with feedforward compensation.
        
        Args:
            desired_position (float): The desired position of the system.
            desired_velocity (float): The desired velocity of the system.
            desired_acceleration (float): The desired acceleration of the system.
            current_position (float): The current position of the system.
            current_velocity (float): The current velocity of the system.
            Kp (float): The proportional gain of the PD controller.
            Kd (float): The derivative gain of the PD controller.
            g (float): The gravitational acceleration.
            Ftip (float): The force applied at the tip of the system.
        
        Returns:
            float: The computed control signal.
        """
        
        pd_signal = self.pd_control(desired_position, desired_velocity, current_position, current_velocity, Kp, Kd)
        ff_signal = self.feedforward_control(desired_position, desired_velocity, desired_acceleration, g, Ftip)
        control_signal = pd_signal + ff_signal
        return control_signal

    @staticmethod
    def enforce_limits(thetalist, dthetalist, tau, joint_limits, torque_limits):
        """
        Enforce joint and torque limits.

        Args:
            thetalist (numpy.ndarray): The joint angles.
            dthetalist (numpy.ndarray): The joint velocities.
            tau (numpy.ndarray): The torques.
            joint_limits (numpy.ndarray): The limits of the joint angles.
            torque_limits (numpy.ndarray): The limits of the torques.

        Returns:
            tuple: A tuple containing the clipped joint angles, joint velocities, and torques.
        """
        
        thetalist = np.clip(thetalist, joint_limits[:, 0], joint_limits[:, 1])
        tau = np.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
        return thetalist, dthetalist, tau

    def plot_steady_state_response(self, time, response, set_point, title='Steady State Response'):
        """
        Plot the steady-state response of the controller.

        :param time: Array of time steps.
        :param response: Array of response values.
        :param set_point: Desired set point value.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(time, response, label='Response')
        plt.axhline(y=set_point, color='r', linestyle='--', label='Set Point')

        # Calculate key metrics
        rise_time = self.calculate_rise_time(time, response, set_point)
        percent_overshoot = self.calculate_percent_overshoot(response, set_point)
        settling_time = self.calculate_settling_time(time, response, set_point)
        steady_state_error = self.calculate_steady_state_error(response, set_point)

        # Annotate metrics on the plot
        plt.axvline(x=rise_time, color='g', linestyle='--', label=f'Rise Time: {rise_time:.2f}s')
        plt.axhline(y=set_point * (1 + percent_overshoot / 100), color='b', linestyle='--', label=f'Overshoot: {percent_overshoot:.2f}%')
        plt.axvline(x=settling_time, color='m', linestyle='--', label=f'Settling Time: {settling_time:.2f}s')
        plt.axhline(y=set_point + steady_state_error, color='c', linestyle='--', label=f'Steady State Error: {steady_state_error:.2f}')

        plt.xlabel('Time (s)')
        plt.ylabel('Response')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_rise_time(self, time, response, set_point):
        """
        Calculate the rise time.

        :param time: Array of time steps.
        :param response: Array of response values.
        :param set_point: Desired set point value.
        :return: Rise time.
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

        :param response: Array of response values.
        :param set_point: Desired set point value.
        :return: Percent overshoot.
        """
        max_response = np.max(response)
        percent_overshoot = ((max_response - set_point) / set_point) * 100
        return percent_overshoot

    def calculate_settling_time(self, time, response, set_point, tolerance=0.02):
        """
        Calculate the settling time.

        :param time: Array of time steps.
        :param response: Array of response values.
        :param set_point: Desired set point value.
        :param tolerance: Tolerance for settling time calculation.
        :return: Settling time.
        """
        settling_threshold = set_point * tolerance
        settling_idx = np.where(np.abs(response - set_point) <= settling_threshold)[0]
        settling_time = time[settling_idx[-1]] if len(settling_idx) > 0 else time[-1]
        return settling_time

    def calculate_steady_state_error(self, response, set_point):
        """
        Calculate the steady-state error.

        :param response: Array of response values.
        :param set_point: Desired set point value.
        :return: Steady-state error.
        """
        steady_state_error = response[-1] - set_point
        return steady_state_error