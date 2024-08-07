#!/usr/bin/env python3

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from math import pi
import numpy as np
from ManipulaPy.singularity import Singularity
import ManipulaPy.utils 
# Path to your URDF file
urdf_file_path = "xarm/xarm6_robot.urdf"
# Initialize the URDFToSerialManipulator with the URDF file
urdf_processor = URDFToSerialManipulator(urdf_file_path)
# Extract the SerialManipulator object
ur5 = urdf_processor.serial_manipulator
# Initialize Singularity Analysis
singularity_analyzer = Singularity(ur5)
# Example joint angles (thetalist) for the manipulator
thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, (-2*pi/3)])
dthetalist = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ddthetalist = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# Perform forward kinematics using the space frame
T_space = ur5.forward_kinematics(thetalist, frame='space')
print("\nForward Kinematics (Space Frame):")
print(T_space)
# Initialize Singularity Analysis
singularity_analyzer = Singularity(ur5)
# Example to plot manipulability ellipsoid
#singularity_analyzer.manipulability_ellipsoid(thetalist)
joint_limits = [
    (-np.pi, np.pi),  # Joint 1
    (-np.pi/2, np.pi/2),  # Joint 2
    (-np.pi/2, np.pi/2),  # Joint 3
    (-np.pi, np.pi/3),  # Joint 4
    (-np.pi/2, np.pi),  # Joint 5
    (-np.pi, np.pi)   # Joint 6
]
#singularity_analyzer.plot_workspace_monte_carlo(joint_limits,num_samples=200000)

# Perform forward kinematics using the body frame
T_body = ur5.forward_kinematics(thetalist, frame='body')
print("\nForward Kinematics (Body Frame):")
print(T_body)
g = [0, 0, -9.81]  # Gravity vector
Ftip = np.array([1, 1, 1, 1, 1, 1])  # External forces on the end-effector
# Example end-effector twist
V_ee = [0.1, 3, 3, -0.1, -0.2, 0.1]

# Compute joint velocities to achieve the desired end-effector twist
joint_velocities = ur5.joint_velocity(thetalist, V_ee, frame='space')
print("\nJoint Velocities (Space Frame):")
print(joint_velocities)
# Adjusted initial guess
new_initial_thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, (-2*pi/3)]) + np.random.normal(0, 0.2, 6)

# Perform inverse kinematics with the new initial guess
thetalistd = ur5.iterative_inverse_kinematics(T_space, new_initial_thetalist,plot_residuals=True)

print("\nInverse Kinematics with Adjusted Initial Guess (Space Frame):")
print(thetalistd)
T_d= ur5.forward_kinematics(np.array([ 3.14579196,  0.56079453,  0.71427021, -1.00188078, -1.56211583,-2.1069051 ]), frame='space')
print("\nForward Kinematics (Space Frame):")
print(T_d)
#define the dynamics 
ur5_dynamics = urdf_processor.initialize_manipulator_dynamics()
# Calculate the mass matrix
mass_matrix = ur5_dynamics.mass_matrix(thetalist)
print("\nMass Matrix:")
print(np.array(mass_matrix))

# Test inverse dynamics
taulist = ur5_dynamics.inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip)
print("Inverse Dynamics (Joint Torques):\n", taulist)

#urdf_processor.simulate_robot_with_desired_angles(thetalist)



