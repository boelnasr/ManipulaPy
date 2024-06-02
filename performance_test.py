#!/usr/bin/env python3



import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.control import ManipulatorController
from math import pi

def main():
    # Path to your URDF file
    urdf_file_path = "ur5/ur5/ur5.urdf"

    # Initialize the URDFToSerialManipulator with the URDF file
    urdf_processor = URDFToSerialManipulator(urdf_file_path)

    # Extract the SerialManipulator object
    ur5 = urdf_processor.serial_manipulator

    # Example joint angles (thetalist) for the manipulator
    thetalist = np.array([pi, -pi/6, -pi/4, -pi/3, -pi/2, -2*pi/3])
    dthetalist = np.array([0.0] * 6)

    # Define joint limits
    Joint_limits = [
        (-np.pi, np.pi),  # Joint 1
        (-np.pi/2, np.pi/2),  # Joint 2
        (-np.pi/2, np.pi/2),  # Joint 3
        (-np.pi, np.pi/3),  # Joint 4
        (-np.pi/2, np.pi),  # Joint 5
        (-np.pi, np.pi)   # Joint 6
    ]

    # Define torque limits (optional)
    torque_limits = [
        (-100, 100),  # Joint 1
        (-100, 100),  # Joint 2
        (-100, 100),  # Joint 3
        (-100, 100),  # Joint 4
        (-100, 100),  # Joint 5
        (-100, 100)   # Joint 6
    ]

    # Initialize ManipulatorDynamics with the URDF processor
    dynamics = urdf_processor.dynamics

    # Initialize the Controller
    controller = ManipulatorController(dynamics)

    # Initialize the Trajectory Planner
    trajectory_planner = tp(ur5, urdf_file_path, dynamics, Joint_limits, torque_limits)

    # Perform forward kinematics to get the desired end-effector position
    T_desired = ur5.forward_kinematics(thetalist, frame='space')
    initial_guess = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, (-2*pi/3)]) + np.random.normal(0, 0.2, 6)
    desired_joint_angles, success, _ = ur5.iterative_inverse_kinematics(T_desired, initial_guess)

    if not success:
        print("Inverse kinematics did not converge.")
        return

    # Generate a joint trajectory
    Tf = 2  # Total time for trajectory
    N = 1000  # Number of steps
    method = 5  # Quintic time scaling
    trajectory = trajectory_planner.joint_trajectory(thetalist, desired_joint_angles, Tf, N, method)

    # Time step for simulation
    dt = Tf / N

    # Define ultimate gain and ultimate period
    ultimate_gain = 0.05  # Replace with your actual ultimate gain
    ultimate_period = 10  # Replace with your actual ultimate period

    # Tune control gains using Ziegler-Nichols method
    Kp, Ki, Kd = controller.tune_controller(ultimate_gain, ultimate_period, controller_type="PID")

    controllers = {
        "Computed Torque": controller.computed_torque_control,
        "PD Control": controller.pd_control,
        "PID Control": controller.pid_control,
        "Robust Control": controller.robust_control,
        "Adaptive Control": controller.adaptive_control,
        "Feedforward Control": controller.feedforward_control,
        "PD Feedforward Control": controller.pd_feedforward_control
    }

    results = {name: [] for name in controllers.keys()}
    errors = {name: [] for name in controllers.keys()}

    # Simulate the control for each controller
    for name, control_method in controllers.items():
        thetalist = cp.asarray([pi, -pi/6, -pi/4, -pi/3, -pi/2, -2*pi/3])
        dthetalist = cp.zeros(6)
        controller.eint = cp.zeros(6)
        controller.parameter_estimate = cp.zeros(6)

        for i in range(N):
            desired_position = cp.asarray(trajectory["positions"][i])
            desired_velocity = cp.asarray(trajectory["velocities"][i])
            desired_acceleration = cp.asarray(trajectory["accelerations"][i])

            if name == "PD Control":
                tau = control_method(desired_position, desired_velocity, thetalist, dthetalist, Kp, Kd)
            elif name == "PID Control":
                tau = control_method(desired_position, desired_velocity, thetalist, dthetalist, dt, Kp, Ki, Kd)
            elif name == "Robust Control":
                disturbance_estimate = cp.zeros(6)  # Example disturbance estimate
                adaptation_gain = 1.0  # Example adaptation gain
                tau = control_method(thetalist, dthetalist, desired_acceleration, cp.array([0, 0, -9.81]), cp.zeros(6), disturbance_estimate, adaptation_gain)
            elif name == "Adaptive Control":
                measurement_error = cp.zeros(6)  # Example measurement error
                adaptation_gain = 1.0  # Example adaptation gain
                tau = control_method(thetalist, dthetalist, desired_acceleration, cp.array([0, 0, -9.81]), cp.zeros(6), measurement_error, adaptation_gain)
            elif name == "Feedforward Control":
                tau = control_method(desired_position, desired_velocity, desired_acceleration, cp.array([0, 0, -9.81]), cp.zeros(6))
            elif name == "PD Feedforward Control":
                tau = control_method(desired_position, desired_velocity, desired_acceleration, thetalist, dthetalist, Kp, Kd, cp.array([0, 0, -9.81]), cp.zeros(6))
            else:
                tau = control_method(desired_position, desired_velocity, desired_acceleration, thetalist, dthetalist, cp.array([0, 0, -9.81]), dt, Kp, Ki, Kd)

            ddthetalist = cp.dot(cp.linalg.inv(cp.asarray(dynamics.mass_matrix(thetalist.get()))), (tau - cp.asarray(dynamics.velocity_quadratic_forces(thetalist.get(), dthetalist.get())) - cp.asarray(dynamics.gravity_forces(thetalist.get(), np.array([0, 0, -9.81])))))
            dthetalist += ddthetalist * dt
            thetalist += dthetalist * dt

            results[name].append((thetalist.copy(), dthetalist.copy()))
            errors[name].append(cp.linalg.norm(thetalist - cp.asarray(desired_joint_angles)))

    # # Plot the results
    # time_steps = np.linspace(0, Tf, N)
    # fig, axs = plt.subplots(len(controllers), 1, figsize=(12, 3 * len(controllers)), sharex=True)

    # for ax, (name, error) in zip(axs, errors.items()):
    #     ax.plot(time_steps, cp.asnumpy(cp.array(error)), label=f'{name} Error')
    #     ax.set_title(f'{name} Error')
    #     ax.set_ylabel("Error Norm")
    #     ax.grid(True)
    #     ax.legend()

    # axs[-1].set_xlabel("Time (s)")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()