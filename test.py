from urdf_processor import URDFToSerialManipulator
from kinematics import SerialManipulator
from dynamics import ManipulatorDynamics
from math import pi
import numpy as np
from singularity import Singularity
# Path to your URDF file
urdf_file_path = "ur5/ur5/ur5.urdf"

# Initialize the URDFToSerialManipulator with the URDF file
urdf_processor = URDFToSerialManipulator(urdf_file_path)

# Extract the SerialManipulator object
ur5 = urdf_processor.serial_manipulator
print("\n rlist:\n", np.array(ur5.r_list))
print("\n vlist:\n", np.array(ur5.S_list[3:]).T)

# Initialize ManipulatorDynamics
ur5_dynamics = ManipulatorDynamics(ur5.M_list, ur5.omega_list, ur5.r_list, ur5.b_list, ur5.S_list, ur5.B_list, urdf_processor.robot_data["Glist"])

# Example joint angles (thetalist) for the manipulator
thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, (-2*pi/3)])

# Perform forward kinematics using the space frame
T_space = ur5.forward_kinematics(thetalist, frame='space')
print("\nForward Kinematics (Space Frame):")
print(T_space)

# Perform forward kinematics using the body frame
T_body = ur5.forward_kinematics(thetalist, frame='body')
print("\nForward Kinematics (Body Frame):")
print(T_body)

# Example end-effector twist
V_ee = [0.1, 3, 3, -0.1, -0.2, 0.1]

# Compute joint velocities to achieve the desired end-effector twist
joint_velocities = ur5.joint_velocity(thetalist, V_ee, frame='space')
print("\nJoint Velocities (Space Frame):")
print(joint_velocities)

# Calculate the mass matrix
mass_matrix = ur5_dynamics.mass_matrix(thetalist)
print("\nMass Matrix:")
print(np.array(mass_matrix))

#plot the manipuality elipsoide:
elips = Singularity(ur5)
elips = elips.manipulability_ellipsoid(thetalist, ax=None)
# Simulate the robot using PyBullet (this will open a PyBullet GUI window)
urdf_processor.simulate_robot_with_desired_angles([pi/2, pi/2, -pi/2, 0, 0, 0])


#plot the manipuality elipsoide:
elips = Singularity(ur5)
elips = elips.manipulability_ellipsoid(thetalist, ax=None)
# Note: The simulation part will run indefinitely until you manually close the PyBullet window.
