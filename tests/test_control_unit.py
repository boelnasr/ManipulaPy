#!/usr/bin/env python3
"""
Focused unit tests for ManipulaPy.control using a lightweight mock dynamics.

These tests cover:
- PID math (including integral accumulation and zero gains)
- PD computation
- Computed torque control formula with mocked dynamics terms
- Robust and adaptive control torque paths
- Feedforward variants
"""

import numpy as np
import pytest

from ManipulaPy.control import ManipulatorController


class MockDynamics:
    """Minimal dynamics model with deterministic outputs for testing."""
    def __init__(self, n=2):
        self.n = n
        self.M_const = np.diag([2.0] * n)
        self.C_const = np.ones(n) * 0.1
        self.G_const = np.ones(n) * 0.05
        self.J_const = np.eye(6, n)  # shape (6,n) so J^T is (n,6)

    def mass_matrix(self, thetalist):
        return self.M_const

    def velocity_quadratic_forces(self, thetalist, dthetalist):
        return self.C_const

    def gravity_forces(self, thetalist, g):
        return self.G_const

    def jacobian(self, thetalist):
        return self.J_const

    def inverse_dynamics(self, thetalist, dthetalist, ddthetalist, g, Ftip):
        # simple linear model: M*qdd + C*dq + G
        return self.M_const @ ddthetalist + self.C_const + self.G_const

    def forward_dynamics(self, thetalist, dthetalist, taulist, g, Ftip):
        # naive model: qdd = (tau - C - G)
        return taulist - self.C_const - self.G_const


@pytest.fixture
def controller():
    return ManipulatorController(MockDynamics(n=2))


def test_pid_control_zero_gains(controller: ManipulatorController):
    thetalistd = np.array([1.0, -1.0])
    dthetalistd = np.zeros(2)
    thetalist = np.array([0.5, -0.5])
    dthetalist = np.zeros(2)
    dt = 0.1
    Kp = np.zeros(2)
    Ki = np.zeros(2)
    Kd = np.zeros(2)

    tau = controller.pid_control(thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd)
    assert np.allclose(tau, 0.0)


def test_pid_control_integral_accumulates(controller: ManipulatorController):
    thetalistd = np.array([1.0, 1.0])
    dthetalistd = np.zeros(2)
    thetalist = np.array([0.0, 0.0])
    dthetalist = np.zeros(2)
    dt = 0.1
    Kp = np.zeros(2)
    Ki = np.ones(2) * 2.0
    Kd = np.zeros(2)

    tau_1 = controller.pid_control(thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd)
    tau_2 = controller.pid_control(thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd)
    # Integral term should double after second call
    assert np.allclose(tau_2, 2 * tau_1)


def test_pd_control_matches_formula(controller: ManipulatorController):
    desired_pos = np.array([0.5, -0.5])
    desired_vel = np.array([0.1, -0.1])
    current_pos = np.array([0.2, -0.6])
    current_vel = np.array([0.0, 0.0])
    Kp = np.array([2.0, 3.0])
    Kd = np.array([0.5, 0.5])

    expected = Kp * (desired_pos - current_pos) + Kd * (desired_vel - current_vel)
    tau = controller.pd_control(desired_pos, desired_vel, current_pos, current_vel, Kp, Kd)
    assert np.allclose(tau, expected)


def test_computed_torque_control_combines_terms(controller: ManipulatorController):
    n = 2
    thetalistd = np.zeros(n)
    dthetalistd = np.zeros(n)
    ddthetalistd = np.ones(n) * 0.2
    thetalist = np.ones(n) * 0.1
    dthetalist = np.ones(n) * -0.05
    g = np.array([0, 0, -9.81])
    dt = 0.1
    Kp = np.ones(n) * 1.5
    Ki = np.zeros(n)
    Kd = np.ones(n) * 0.7

    tau = controller.computed_torque_control(
        thetalistd, dthetalistd, ddthetalistd,
        thetalist, dthetalist, g, dt, Kp, Ki, Kd
    )

    # Expected: M@(Kp*e + Ki*eint + Kd*edot) + inverse_dynamics(...)
    e = thetalistd - thetalist
    edot = dthetalistd - dthetalist
    M = controller.dynamics.M_const
    inv_dyn = controller.dynamics.inverse_dynamics(thetalist, dthetalist, ddthetalistd, g, [0]*6)
    expected = M @ (Kp * e + Kd * edot) + inv_dyn
    assert np.allclose(tau, expected)


def test_robust_control_includes_adaptation_term(controller: ManipulatorController):
    n = 2
    thetalist = np.zeros(n)
    dthetalist = np.zeros(n)
    ddthetalist = np.ones(n) * 0.3
    g = np.array([0, 0, -9.81])
    Ftip = np.zeros(6)
    disturbance = np.ones(n) * 0.2
    adaptation_gain = 0.5

    tau = controller.robust_control(thetalist, dthetalist, ddthetalist, g, Ftip, disturbance, adaptation_gain)
    # base torque from dynamics + adaptation term
    M = controller.dynamics.M_const
    c = controller.dynamics.C_const
    grav = controller.dynamics.G_const
    JtF = controller.dynamics.J_const.T @ Ftip
    expected = M @ ddthetalist + c + grav + JtF + adaptation_gain * disturbance
    assert np.allclose(tau, expected)


def test_adaptive_control_updates_parameters(controller: ManipulatorController):
    n = 2
    thetalist = np.zeros(n)
    dthetalist = np.zeros(n)
    ddthetalist = np.ones(n) * 0.1
    g = np.array([0, 0, -9.81])
    Ftip = np.zeros(6)
    measurement_error = np.array([0.05, -0.02])
    adaptation_gain = 0.5

    tau = controller.adaptive_control(
        thetalist, dthetalist, ddthetalist, g, Ftip, measurement_error, adaptation_gain
    )
    # parameter_estimate should be updated by gamma * err
    expected_params = adaptation_gain * measurement_error
    assert np.allclose(controller.parameter_estimate, expected_params)
    # torque includes parameter_estimate
    base = controller.dynamics.M_const @ ddthetalist
    base += controller.dynamics.C_const + controller.dynamics.G_const
    base += controller.dynamics.J_const.T @ Ftip
    assert np.allclose(tau, base + expected_params)


def test_feedforward_controls_use_inverse_dynamics(controller: ManipulatorController):
    n = 2
    desired_position = np.array([0.1, -0.2])
    desired_velocity = np.zeros(n)
    desired_accel = np.ones(n) * 0.05
    g = np.array([0, 0, -9.81])
    Ftip = np.zeros(6)
    tau_ff = controller.feedforward_control(desired_position, desired_velocity, desired_accel, g, Ftip)
    expected = controller.dynamics.inverse_dynamics(desired_position, desired_velocity, desired_accel, g, Ftip)
    assert np.allclose(tau_ff, expected)

    current_position = np.array([0.0, 0.0])
    current_velocity = np.zeros(n)
    Kp = np.ones(n) * 2.0
    Kd = np.ones(n) * 0.1
    tau_pd_ff = controller.pd_feedforward_control(
        desired_position, desired_velocity, desired_accel,
        current_position, current_velocity, Kp, Kd, g, Ftip
    )
    # PD term
    pd_term = Kp * (desired_position - current_position) + Kd * (desired_velocity - current_velocity)
    assert np.allclose(tau_pd_ff, expected + pd_term)
