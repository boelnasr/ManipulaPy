#!/usr/bin/env python3

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from math import pi
import numpy as np
from ManipulaPy.singularity import Singularity

# Path to your URDF file
urdf_file_path = "ur5/ur5/ur5.urdf"

# Initialize the URDFToSerialManipulator with the URDF file
urdf_processor = URDFToSerialManipulator(urdf_file_path)

# Extract the SerialManipulator object
ur5 = urdf_processor.serial_manipulator

# Initialize Singularity Analysis
singularity_analyzer = Singularity(ur5)

# Example joint angles (thetalist) for the manipulator
thetalist = np.array([pi, pi / 6, pi / 4, -pi / 3, -pi / 2, -2 * pi / 3])
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

# Initialize the Trajectory Planning object
trr = tp(
    ur5, dynamics=urdf_processor.dynamics, joint_limits=Joint_limits, torque_limits=None
)

# Generate the joint trajectory
traj = trr.JointTrajectory(
    thetastart=[0] * 6, thetaend=thetalist, Tf=5, N=10000, method=5
)
positions = traj["positions"]
velocities = traj["velocities"]
accelerations = traj["accelerations"]

# Compute inverse dynamics trajectory
traj_tau = trr.InverseDynamicsTrajectory(positions, velocities, accelerations)

# Use only the initial joint positions and velocities
initial_thetalist = positions[0]
initial_dthetalist = velocities[0]

# Perform forward dynamics simulation
forward_dynamics_result = trr.forward_dynamics_trajectory(
    thetalist=initial_thetalist,
    dthetalist=initial_dthetalist,
    taumat=traj_tau,
    g=np.array([0, 0, -9.81]),  # Assuming a gravity vector
    Ftipmat=np.zeros((positions.shape[0], 6)),  # Assuming no external forces
    dt=0.001,
    intRes=1000,
)

# Extract the results from the forward dynamics simulation
thetamat = forward_dynamics_result["positions"]
dthetamat = forward_dynamics_result["velocities"]
ddthetamat = forward_dynamics_result["accelerations"]

# Plot the joint trajectory
trr.plot_trajectory(
    {"positions": thetamat, "velocities": dthetamat, "accelerations": ddthetamat}, Tf=5
)

# Define start and end configurations for Cartesian trajectory
Xstart = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]])

Xend = T_space

# Generate the Cartesian trajectory
Tf = 5  # Total time of the trajectory in seconds
N = 100  # Number of points in the trajectory
method = 5  # Quintic time scaling

cartesian_traj = trr.CartesianTrajectory(Xstart, Xend, Tf, N, method)

# Plot the Cartesian trajectory
trr.plot_cartesian_trajectory(cartesian_traj, Tf, title="Cartesian Trajectory")

# Plot the end-effector trajectory
trr.plot_ee_trajectory(forward_dynamics_result, Tf, title="End-Effector Trajectory")
