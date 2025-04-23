#!/usr/bin/env python3

import os
import pybullet as p
import pybullet_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.singularity import Singularity
from ManipulaPy.control import ManipulatorController
from ManipulaPy.sim import Simulation
from math import pi
import numpy as np
import time
import matplotlib.pyplot as plt


def main():
    # Set environment variable to use software rendering
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

    # Path to your URDF file
    urdf_file_path = "ur5/ur5/ur5.urdf"

    # Initialize the URDFToSerialManipulator with the URDF file
    urdf_processor = URDFToSerialManipulator(urdf_file_path)

    # Extract the SerialManipulator object
    ur5 = urdf_processor.serial_manipulator

    # Initialize Singularity Analysis
    singularity_analyzer = Singularity(ur5)

    # Example joint angles (thetalist) for the manipulator
    thetalist = np.array([pi, -pi / 6, -pi / 4, -pi / 3, -pi / 2, -2 * pi / 3])
    dthetalist = np.array([0.1] * 6)
    ddthetalist = np.array([0.1] * 6)

    # Perform forward kinematics using the space frame
    T_space = ur5.forward_kinematics(thetalist, frame="space")
    print("\nForward Kinematics (Space Frame):")
    print(T_space)

    # Define joint limits
    Joint_limits = [
        (-np.pi, np.pi),  # Joint 1
        (-np.pi / 2, np.pi / 2),  # Joint 2
        (-np.pi / 2, np.pi / 2),  # Joint 3
        (-np.pi, np.pi / 3),  # Joint 4
        (-np.pi / 2, np.pi),  # Joint 5
        (-np.pi, np.pi),  # Joint 6
    ]

    # Define torque limits (optional)
    torque_limits = [
        (-100, 100),  # Joint 1
        (-100, 100),  # Joint 2
        (-100, 100),  # Joint 3
        (-100, 100),  # Joint 4
        (-100, 100),  # Joint 5
        (-100, 100),  # Joint 6
    ]

    # Initialize ManipulatorDynamics with the URDF processor
    dynamics = urdf_processor.dynamics

    # Initialize the Controller
    controller = ManipulatorController(dynamics)

    # Initialize the Trajectory Planner
    trajectory_planner = tp(ur5, urdf_file_path, dynamics, Joint_limits, torque_limits)

    # Create an instance of the Simulation class
    simulation = Simulation(urdf_file_path, Joint_limits, torque_limits)
    # Generate a joint trajectory (example)
    thetastart = np.array([0, 0, 0, 0, 0, 0])
    thetaend = np.array([pi / 2, -pi / 4, pi / 6, -pi / 3, pi / 4, pi / 2])
    Tf = 5  # Total time for the trajectory
    N = 100  # Number of trajectory points
    method = 5  # Quintic time scaling

    trajectory = trajectory_planner.joint_trajectory(
        thetastart, thetaend, Tf, N, method
    )
    joint_positions = trajectory["positions"]
    joint_velocities = trajectory["velocities"]
    joint_accelerations = trajectory["accelerations"]

    # Plot the generated trajectory
    # trajectory_planner.plot_trajectory(trajectory, Tf, title="Generated Joint Trajectory")

    # Simulate the robot motion with the generated joint trajectory
    simulation.run_trajectory(joint_positions)

    # Example usage of the controller with the simulation
    g = np.array([0, 0, -9.81])
    Ftip = np.zeros(6)
    Kp = np.array([100, 100, 100, 100, 100, 100])
    Ki = np.array([0, 0, 0, 0, 0, 0])
    Kd = np.array([20, 20, 20, 20, 20, 20])

    # Run the controller on the trajectory
    simulation.run_controller(
        controller,
        joint_positions,
        joint_velocities,
        joint_accelerations,
        g,
        Ftip,
        Kp,
        Ki,
        Kd,
    )

    # Plot the trajectory in the simulation scene
    # Forward kinematics to get end-effector positions
    ee_trajectory = [
        ur5.forward_kinematics(joint_positions[i])[:3, 3] for i in range(N)
    ]
    ee_trajectory = np.array(ee_trajectory)


if __name__ == "__main__":
    main()
