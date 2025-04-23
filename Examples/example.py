#!/usr/bin/env python3

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp

# import the data for the xarm
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

#intialize the manipulator
xarm = URDFToSerialManipulator(xarm_urdf_file)

#extract the serial manipulator
xarm_manipulator = xarm.serial_manipulator
#initialize the dynamics
xarm_dynamics = xarm.dynamics
joint_limits = xarm_manipulator.joint_limits
planner = tp(xarm_manipulator,
            xarm_urdf_file,
            xarm_dynamics,
            joint_limits)

# Generate a trajectory between start and end joint configurations
trajectory = planner.joint_trajectory(  # Use planner instance instead of tp class
    thetastart=[0, 0, 0, 0, 0, 0],  # Starting joint angles
    thetaend=[1.0, 0.5, 0.7, 0.3, 0.2, 0.1],  # Target joint angles
    Tf=5.0,  # Total time in seconds
    N=100,  # Number of waypoints
    method=3  # 3 for cubic time scaling, 5 for quintic
)

# The trajectory contains positions, velocities, and accelerations
positions = trajectory["positions"]
velocities = trajectory["velocities"]
accelerations = trajectory["accelerations"]

