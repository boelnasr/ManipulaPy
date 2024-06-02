#!/usr/bin/env python3

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.control import ManipulatorController
from math import pi
from ManipulaPy.ManipulaPy_data.ur5 import urdf_file as xarm_urdf_file

def main():
    # Path to your URDF file
    urdf_file_path = xarm_urdf_file

    # Initialize the URDFToSerialManipulator with the URDF file
    urdf_processor = URDFToSerialManipulator(urdf_file_path)

    # Extract the SerialManipulator object
    ur5 = urdf_processor.serial_manipulator

    # Example joint angles (thetalist) for the manipulator
    thetalist = np.array([pi, -pi/6, -pi/4, -pi/3, -pi/2, -2*pi/3])

    # Initialize ManipulatorDynamics with the URDF processor
    dynamics = urdf_processor.dynamics

    # Initialize the Controller
    controller = ManipulatorController(dynamics)

    # Perform forward kinematics to get the desired end-effector position
    T_desired = ur5.forward_kinematics(thetalist, frame='space')
    initial_guess = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, (-2*pi/3)]) + np.random.normal(0, 0.2, 6)
    desired_joint_angles, success, _ = ur5.iterative_inverse_kinematics(T_desired, initial_guess)

    if not success:
        print("Inverse kinematics did not converge.")
        return

    # Find the ultimate gain and period
    dt = 0.01  # Time step for simulation
    ultimate_gain, ultimate_period, gain_history, error_history = controller.find_ultimate_gain_and_period(thetalist, desired_joint_angles, dt)

    print(f"Ultimate Gain (K_u): {ultimate_gain}")
    print(f"Ultimate Period (T_u): {ultimate_period}")

    # Plot the error history for the last gain value
    plt.plot(cp.asnumpy(cp.array(error_history[-1])))
    plt.xlabel("Time Steps")
    plt.ylabel("Error")
    plt.title("Error History with Ultimate Gain")
    plt.show()

    # Kalman Filter example
    Q = cp.eye(12) * 0.01  # Process noise covariance
    R = cp.eye(12) * 0.01  # Measurement noise covariance
    thetalist = cp.array([pi, -pi/6, -pi/4, -pi/3, -pi/2, -2*pi/3])
    dthetalist = cp.zeros(6)
    tau = cp.zeros(6)  # Applied torque
    Ftip = cp.zeros(6)  # External forces at the end effector

    estimated_states = []
    true_states = []

    for _ in range(100):
        # Apply some control input (here just an example torque)
        tau = cp.random.normal(0, 1, 6)
        
        # True state evolution (for example purposes, in practice this would come from the system)
        ddthetalist = cp.dot(cp.linalg.inv(cp.asarray(dynamics.mass_matrix(thetalist.get()))), 
                            (tau - cp.asarray(dynamics.velocity_quadratic_forces(thetalist.get(), dthetalist.get())) - 
                            cp.asarray(dynamics.gravity_forces(thetalist.get(), np.array([0, 0, -9.81])))))
        dthetalist += ddthetalist * dt
        thetalist += dthetalist * dt

        # Kalman filter prediction
        controller.kalman_filter_predict(thetalist, dthetalist, tau, cp.array([0, 0, -9.81]), Ftip, dt, Q)

        # Simulate measurement
        measurement = cp.concatenate((thetalist, dthetalist)) + cp.random.normal(0, 0.1, 12)

        # Kalman filter update
        controller.kalman_filter_update(measurement, R)

        # Store the states
        estimated_states.append(controller.x_hat.copy())
        true_states.append(cp.concatenate((thetalist, dthetalist)))

    estimated_states = cp.array(estimated_states)
    true_states = cp.array(true_states)

    # Plot the Kalman filter results
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(cp.asnumpy(estimated_states[:, i]), label='Estimated')
        plt.plot(cp.asnumpy(true_states[:, i]), label='True')
        plt.title(f'Joint {i + 1} Angle')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
