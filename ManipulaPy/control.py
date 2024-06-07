#!/usr/bin/env python3

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from .dynamics import ManipulatorDynamics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            thetalistd (cp.ndarray): Desired joint angles.
            dthetalistd (cp.ndarray): Desired joint velocities.
            ddthetalistd (cp.ndarray): Desired joint accelerations.
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            g (cp.ndarray): Gravity vector.
            dt (float): Time step.
            Kp (cp.ndarray): Proportional gain.
            Ki (cp.ndarray): Integral gain.
            Kd (cp.ndarray): Derivative gain.

        Returns:
            cp.ndarray: Torque command.
        """
        thetalistd = cp.asarray(thetalistd)
        dthetalistd = cp.asarray(dthetalistd)
        ddthetalistd = cp.asarray(ddthetalistd)
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        g = cp.asarray(g)
        Kp = cp.asarray(Kp)
        Ki = cp.asarray(Ki)
        Kd = cp.asarray(Kd)

        if self.eint is None:
            self.eint = cp.zeros_like(thetalist)

        e = thetalistd - thetalist
        self.eint += e * dt

        M = cp.asarray(self.dynamics.mass_matrix(thetalist.get()))
        tau = M @ (Kp * e + Ki * self.eint + Kd * (dthetalistd - dthetalist))
        tau += cp.asarray(self.dynamics.inverse_dynamics(
            thetalist.get(), dthetalist.get(), ddthetalistd.get(), g.get(), [0, 0, 0, 0, 0, 0]
        ))

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
            desired_position (cp.ndarray): Desired joint positions.
            desired_velocity (cp.ndarray): Desired joint velocities.
            current_position (cp.ndarray): Current joint positions.
            current_velocity (cp.ndarray): Current joint velocities.
            Kp (cp.ndarray): Proportional gain.
            Kd (cp.ndarray): Derivative gain.

        Returns:
            cp.ndarray: PD control signal.
        """
        desired_position = cp.asarray(desired_position)
        desired_velocity = cp.asarray(desired_velocity)
        current_position = cp.asarray(current_position)
        current_velocity = cp.asarray(current_velocity)
        Kp = cp.asarray(Kp)
        Kd = cp.asarray(Kd)

        e = desired_position - current_position
        edot = desired_velocity - current_velocity
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(
        self, thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd
    ):
        """
        PID Control.

        Parameters:
            thetalistd (cp.ndarray): Desired joint angles.
            dthetalistd (cp.ndarray): Desired joint velocities.
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            dt (float): Time step.
            Kp (cp.ndarray): Proportional gain.
            Ki (cp.ndarray): Integral gain.
            Kd (cp.ndarray): Derivative gain.

        Returns:
            cp.ndarray: PID control signal.
        """
        thetalistd = cp.asarray(thetalistd)
        dthetalistd = cp.asarray(dthetalistd)
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        Kp = cp.asarray(Kp)
        Ki = cp.asarray(Ki)
        Kd = cp.asarray(Kd)

        if self.eint is None:
            self.eint = cp.zeros_like(thetalist)

        e = thetalistd - thetalist
        self.eint += e * dt

        e_dot = dthetalistd - dthetalist
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
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            ddthetalist (cp.ndarray): Desired joint accelerations.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.
            disturbance_estimate (cp.ndarray): Estimate of disturbances.
            adaptation_gain (float): Gain for the adaptation term.

        Returns:
            cp.ndarray: Robust control torque.
        """
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        ddthetalist = cp.asarray(ddthetalist)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)
        disturbance_estimate = cp.asarray(disturbance_estimate)

        M = cp.asarray(self.dynamics.mass_matrix(thetalist.get()))
        c = cp.asarray(self.dynamics.velocity_quadratic_forces(thetalist.get(), dthetalist.get()))
        g_forces = cp.asarray(self.dynamics.gravity_forces(thetalist.get(), g.get()))
        J_transpose = cp.asarray(self.dynamics.jacobian(thetalist.get()).T)
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
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            ddthetalist (cp.ndarray): Desired joint accelerations.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.
            measurement_error (cp.ndarray): Error in measurement.
            adaptation_gain (float): Gain for the adaptation term.

        Returns:
            cp.ndarray: Adaptive control torque.
        """
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        ddthetalist = cp.asarray(ddthetalist)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)
        measurement_error = cp.asarray(measurement_error)

        if self.parameter_estimate is None:
            self.parameter_estimate = cp.zeros_like(self.dynamics.Glist)

        self.parameter_estimate += adaptation_gain * measurement_error
        M = cp.asarray(self.dynamics.mass_matrix(thetalist.get()))
        c = cp.asarray(self.dynamics.velocity_quadratic_forces(thetalist.get(), dthetalist.get()))
        g_forces = cp.asarray(self.dynamics.gravity_forces(thetalist.get(), g.get()))
        J_transpose = cp.asarray(self.dynamics.jacobian(thetalist.get()).T)
        tau = (
            M @ ddthetalist
            + c
            + g_forces
            + J_transpose @ Ftip
            + self.parameter_estimate
        )
        return tau

    def kalman_filter_predict(self, thetalist, dthetalist, taulist, g, Ftip, dt, Q):
        """
        Kalman Filter Prediction.

        Parameters:
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            taulist (cp.ndarray): Applied torques.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.
            dt (float): Time step.
            Q (cp.ndarray): Process noise covariance.

        Returns:
            None
        """
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        taulist = cp.asarray(taulist)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)
        Q = cp.asarray(Q)

        if self.x_hat is None:
            self.x_hat = cp.concatenate((thetalist, dthetalist))

        thetalist_pred = (
            self.x_hat[: len(thetalist)] + self.x_hat[len(thetalist) :] * dt
        )
        dthetalist_pred = (
            cp.asarray(self.dynamics.forward_dynamics(
                self.x_hat[: len(thetalist)].get(),
                self.x_hat[len(thetalist) :].get(),
                taulist.get(),
                g.get(),
                Ftip.get(),
            )) * dt
            + self.x_hat[len(thetalist) :]
        )
        x_hat_pred = cp.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self.P = cp.eye(len(x_hat_pred))
        F = cp.eye(len(x_hat_pred))
        self.P = F @ self.P @ F.T + Q

        self.x_hat = x_hat_pred

    def kalman_filter_update(self, z, R):
        """
        Kalman Filter Update.

        Parameters:
            z (cp.ndarray): Measurement vector.
            R (cp.ndarray): Measurement noise covariance.

        Returns:
            None
        """
        z = cp.asarray(z)
        R = cp.asarray(R)

        H = cp.eye(len(self.x_hat))
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ cp.linalg.inv(S)
        self.x_hat += K @ y
        self.P = (cp.eye(len(self.x_hat)) - K @ H) @ self.P

    def kalman_filter_control(
        self, thetalistd, dthetalistd, thetalist, dthetalist, taulist, g, Ftip, dt, Q, R
    ):
        """
        Kalman Filter Control.

        Parameters:
            thetalistd (cp.ndarray): Desired joint angles.
            dthetalistd (cp.ndarray): Desired joint velocities.
            thetalist (cp.ndarray): Current joint angles.
            dthetalist (cp.ndarray): Current joint velocities.
            taulist (cp.ndarray): Applied torques.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.
            dt (float): Time step.
            Q (cp.ndarray): Process noise covariance.
            R (cp.ndarray): Measurement noise covariance.

        Returns:
            tuple: Estimated joint angles and velocities.
        """
        thetalistd = cp.asarray(thetalistd)
        dthetalistd = cp.asarray(dthetalistd)
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        taulist = cp.asarray(taulist)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)
        Q = cp.asarray(Q)
        R = cp.asarray(R)

        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(cp.concatenate((thetalist, dthetalist)), R)
        return self.x_hat[: len(thetalist)], self.x_hat[len(thetalist) :]

    def feedforward_control(
        self, desired_position, desired_velocity, desired_acceleration, g, Ftip
    ):
        """
        Feedforward Control.

        Parameters:
            desired_position (cp.ndarray): Desired joint positions.
            desired_velocity (cp.ndarray): Desired joint velocities.
            desired_acceleration (cp.ndarray): Desired joint accelerations.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.

        Returns:
            cp.ndarray: Feedforward torque.
        """
        desired_position = cp.asarray(desired_position)
        desired_velocity = cp.asarray(desired_velocity)
        desired_acceleration = cp.asarray(desired_acceleration)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)

        tau = cp.asarray(self.dynamics.inverse_dynamics(
            desired_position.get(), desired_velocity.get(), desired_acceleration.get(), g.get(), Ftip.get()
        ))
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
            desired_position (cp.ndarray): Desired joint positions.
            desired_velocity (cp.ndarray): Desired joint velocities.
            desired_acceleration (cp.ndarray): Desired joint accelerations.
            current_position (cp.ndarray): Current joint positions.
            current_velocity (cp.ndarray): Current joint velocities.
            Kp (cp.ndarray): Proportional gain.
            Kd (cp.ndarray): Derivative gain.
            g (cp.ndarray): Gravity vector.
            Ftip (cp.ndarray): External forces applied at the end effector.

        Returns:
            cp.ndarray: Control signal.
        """
        desired_position = cp.asarray(desired_position)
        desired_velocity = cp.asarray(desired_velocity)
        desired_acceleration = cp.asarray(desired_acceleration)
        current_position = cp.asarray(current_position)
        current_velocity = cp.asarray(current_velocity)
        Kp = cp.asarray(Kp)
        Kd = cp.asarray(Kd)
        g = cp.asarray(g)
        Ftip = cp.asarray(Ftip)

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
            thetalist (cp.ndarray): Joint angles.
            dthetalist (cp.ndarray): Joint velocities.
            tau (cp.ndarray): Torques.
            joint_limits (cp.ndarray): Joint angle limits.
            torque_limits (cp.ndarray): Torque limits.

        Returns:
            tuple: Clipped joint angles, velocities, and torques.
        """
        thetalist = cp.asarray(thetalist)
        dthetalist = cp.asarray(dthetalist)
        tau = cp.asarray(tau)
        joint_limits = cp.asarray(joint_limits)
        torque_limits = cp.asarray(torque_limits)

        thetalist = cp.clip(thetalist, joint_limits[:, 0], joint_limits[:, 1])
        tau = cp.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
        return thetalist, dthetalist, tau

    def plot_steady_state_response(
        self, time, response, set_point, title="Steady State Response"
    ):
        """
        Plot the steady-state response of the controller.

        Parameters:
            time (cp.ndarray): Array of time steps.
            response (cp.ndarray): Array of response values.
            set_point (float): Desired set point value.
            title (str, optional): Title of the plot.

        Returns:
            None
        """
        time = cp.asnumpy(time)
        response = cp.asnumpy(response)
        
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
            label=f"Settling Time: {settling_time:.2f}s"
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
            time (cp.ndarray): Array of time steps.
            response (cp.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Rise time.
        """
        time = cp.asnumpy(time)
        response = cp.asnumpy(response)

        rise_start = 0.1 * set_point
        rise_end = 0.9 * set_point
        start_idx = cp.where(response >= rise_start)[0][0]
        end_idx = cp.where(response >= rise_end)[0][0]
        rise_time = time[end_idx] - time[start_idx]
        return rise_time

    def calculate_percent_overshoot(self, response, set_point):
        """
        Calculate the percent overshoot.

        Parameters:
            response (cp.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Percent overshoot.
        """
        response = cp.asnumpy(response)

        max_response = cp.max(response)
        percent_overshoot = ((max_response - set_point) / set_point) * 100
        return percent_overshoot

    def calculate_settling_time(self, time, response, set_point, tolerance=0.02):
        """
        Calculate the settling time.

        Parameters:
            time (cp.ndarray): Array of time steps.
            response (cp.ndarray): Array of response values.
            set_point (float): Desired set point value.
            tolerance (float): Tolerance for settling time calculation.

        Returns:
            float: Settling time.
        """
        time = cp.asnumpy(time)
        response = cp.asnumpy(response)

        settling_threshold = set_point * tolerance
        settling_idx = cp.where(cp.abs(response - set_point) <= settling_threshold)[0]
        settling_time = time[settling_idx[-1]] if len(settling_idx) > 0 else time[-1]
        return settling_time

    def calculate_steady_state_error(self, response, set_point):
        """
        Calculate the steady-state error.

        Parameters:
            response (cp.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Steady-state error.
        """
        response = cp.asnumpy(response)

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
            desired_joint_angles (cp.ndarray): Desired joint angles.
            current_joint_angles (cp.ndarray): Current joint angles.
            current_joint_velocities (cp.ndarray): Current joint velocities.
            Kp (cp.ndarray): Proportional gain.
            Kd (cp.ndarray): Derivative gain.

        Returns:
            cp.ndarray: Control torque.
        """
        desired_joint_angles = cp.asarray(desired_joint_angles)
        current_joint_angles = cp.asarray(current_joint_angles)
        current_joint_velocities = cp.asarray(current_joint_velocities)
        Kp = cp.asarray(Kp)
        Kd = cp.asarray(Kd)

        e = desired_joint_angles - current_joint_angles
        edot = 0 - current_joint_velocities
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
            desired_position (cp.ndarray): Desired end-effector position.
            current_joint_angles (cp.ndarray): Current joint angles.
            current_joint_velocities (cp.ndarray): Current joint velocities.
            Kp (cp.ndarray): Proportional gain.
            Kd (cp.ndarray): Derivative gain.

        Returns:
            cp.ndarray: Control torque.
        """
        desired_position = cp.asarray(desired_position)
        current_joint_angles = cp.asarray(current_joint_angles)
        current_joint_velocities = cp.asarray(current_joint_velocities)
        Kp = cp.asarray(Kp)
        Kd = cp.asarray(Kd)

        current_position = cp.asarray(self.dynamics.forward_kinematics(current_joint_angles.get())[:3, 3])
        e = desired_position - current_position
        dthetalist = current_joint_velocities
        J = cp.asarray(self.dynamics.jacobian(current_joint_angles.get()))
        tau = J.T @ (Kp * e - Kd @ J @ dthetalist)
        return tau

    def ziegler_nichols_tuning(self, ultimate_gain, ultimate_period, controller_type="PID"):
        """
        Perform Ziegler-Nichols tuning.

        Parameters:
            ultimate_gain (float): Ultimate gain (K_u).
            ultimate_period (float): Ultimate period (T_u).
            controller_type (str): Type of controller. Options are "P", "PI", "PID".

        Returns:
            tuple: Tuned Kp, Ki, Kd.
        """
        if controller_type == "P":
            Kp = 0.5 * ultimate_gain
            Ki = 0.0
            Kd = 0.0
        elif controller_type == "PI":
            Kp = 0.45 * ultimate_gain
            Ki = 1.2 * Kp / ultimate_period
            Kd = 0.0
        elif controller_type == "PID":
            Kp = 0.6 * ultimate_gain
            Ki = 2 * Kp / ultimate_period
            Kd = Kp * ultimate_period / 8
        else:
            raise ValueError("Invalid controller type. Choose 'P', 'PI', or 'PID'.")
        
        return Kp, Ki, Kd

    def tune_controller(self, ultimate_gain, ultimate_period, controller_type="PID"):
        """
        Tune the controller using Ziegler-Nichols method.

        Parameters:
            ultimate_gain (float): Ultimate gain (K_u).
            ultimate_period (float): Ultimate period (T_u).
            controller_type (str): Type of controller. Options are "P", "PI", "PID".

        Returns:
            tuple: Tuned Kp, Ki, Kd.
        """
        Kp, Ki, Kd = self.ziegler_nichols_tuning(ultimate_gain, ultimate_period, controller_type)
        logger.info(f"Tuned Parameters: Kp = {Kp}, Ki = {Ki}, Kd = {Kd}")
        return Kp, Ki, Kd
    
    def find_ultimate_gain_and_period(self, thetalist, desired_joint_angles, dt, max_steps=1000):
        """
        Find the ultimate gain and period using the Ziegler-Nichols method.

        Parameters:
            thetalist (cp.ndarray): Initial joint angles.
            desired_joint_angles (cp.ndarray): Desired joint angles.
            dt (float): Time step for simulation.
            max_steps (int, optional): Maximum number of steps for the simulation.

        Returns:
            tuple: Ultimate gain, ultimate period, gain history, error history.
        """
        thetalist = cp.asarray(thetalist)
        desired_joint_angles = cp.asarray(desired_joint_angles)

        Kp = 0.01  # Start with a small proportional gain
        increase_factor = 1.1  # Factor to increase Kp each iteration
        oscillation_detected = False
        error_history = []
        gain_history = []
        
        while not oscillation_detected and Kp < 1000:  # Upper limit to prevent infinite loop
            thetalist_current = thetalist.copy()
            dthetalist = cp.zeros_like(thetalist)
            self.eint = cp.zeros_like(thetalist)  # Reset integral term
            
            error = []
            for step in range(max_steps):
                tau = self.pd_control(desired_joint_angles, cp.zeros_like(thetalist), thetalist_current, dthetalist, Kp, 0)
                ddthetalist = cp.dot(cp.linalg.inv(cp.asarray(self.dynamics.mass_matrix(thetalist_current.get()))), 
                                     (tau - cp.asarray(self.dynamics.velocity_quadratic_forces(thetalist_current.get(), dthetalist.get())) - 
                                      cp.asarray(self.dynamics.gravity_forces(thetalist_current.get(), np.array([0, 0, -9.81])))))
                dthetalist += ddthetalist * dt
                thetalist_current += dthetalist * dt
                
                error.append(cp.linalg.norm(thetalist_current - desired_joint_angles))
                
                if step > 10 and error[-1] > 1e10:  # If error explodes, stop
                    break

            error_history.append(cp.array(error))
            gain_history.append(Kp)
            
            if len(error) >= 2 and error[-2] < error[-1] and error[-1] < error[-2] * 1.2:
                oscillation_detected = True
            else:
                Kp *= increase_factor

        # Determine the ultimate gain and period
        ultimate_gain = Kp
        oscillation_period = (max_steps * dt) / (cp.count_nonzero(cp.diff(cp.sign(cp.array(error_history[-1])))) // 2)
        
        return ultimate_gain, oscillation_period, gain_history, error_history
