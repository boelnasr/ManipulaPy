from urdf_processor import URDFToSerialManipulator
from kinematics import SerialManipulator

# Path to your URDF file
urdf_file_path = "ur5/ur5/ur5.urdf"

# Initialize the URDFToSerialManipulator with the URDF file
urdf_processor = URDFToSerialManipulator(urdf_file_path)

# Extract the SerialManipulator object
serial_manipulator = urdf_processor.serial_manipulator

# Example joint angles (thetalist) for the manipulator
thetalist = [0.5, -0.1, 0.5, 0.75, -0.2, 0.3]
print(serial_manipulator.M_list)

# Perform forward kinematics using the space frame
T_space = serial_manipulator.forward_kinematics(thetalist, frame='space')
print("Forward Kinematics (Space Frame):")
print(T_space)

# Perform forward kinematics using the body frame
T_body = serial_manipulator.forward_kinematics(thetalist, frame='body')
print("\nForward Kinematics (Body Frame):")
print(T_body)

# Example end-effector twist
V_ee = [0.1, 0.2, 0.3, -0.1, -0.2, 0.1]

# Compute joint velocities to achieve the desired end-effector twist
joint_velocities = serial_manipulator.joint_velocity(thetalist, V_ee, frame='space')
print("\nJoint Velocities (Space Frame):")
print(joint_velocities)

# Simulate the robot using PyBullet (this will open a PyBullet GUI window)
urdf_processor.simulate_robot()

# Note: The simulation part will run indefinitely until you manually close the PyBullet window.