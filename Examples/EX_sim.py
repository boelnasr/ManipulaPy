#!/usr/bin/env python3
import numpy as np
import time

from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.sim import Simulation

# 1) Load the robot model (for joint limits)
urdf_proc = URDFToSerialManipulator(xarm_urdf_file)
joint_limits = urdf_proc.serial_manipulator.joint_limits

# (Optional) set torque limits per joint
torque_limits = [(-10, 10)] * len(joint_limits)

# 2) Initialize Simulation
sim = Simulation(
    urdf_file_path=xarm_urdf_file,
    joint_limits=joint_limits,
    torque_limits=torque_limits,
    time_step=0.02,
    real_time_factor=0.5
)

# 3) Define a simple joint-space trajectory:
#    here we ramp all joints from 0 to 45° over 100 steps
n_steps = 100
ramp = np.deg2rad(45) * np.linspace(0, 1, n_steps)
trajectory = [ramp.copy() for _ in joint_limits]  # shape (n_steps, 6)
trajectory = np.stack(trajectory, axis=1)          # (n_steps, 6)

# 4) Run it in the simulation (blocks until finished)
print("Running trajectory in PyBullet…")
end_effector_pos = sim.run_trajectory(trajectory)
print("Final end‐effector position:", end_effector_pos)

# 5) Now switch to manual‐control mode (use the GUI sliders to move joints)
print("Entering manual control—use sliders in the PyBullet window. Ctrl+C to exit.")
try:
    sim.manual_control()
except KeyboardInterrupt:
    pass

# 6) Clean up
sim.close_simulation()
print("Simulation closed.")
