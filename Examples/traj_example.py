#!/usr/bin/env python3

import numpy as np
from ManipulaPy.sim import Simulation
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as ur5_urdf_file
import time
import pybullet as p
def main():
    # Define the joint limits for the UR5 robot
    joint_limits = [
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi)
    ]

    # Define the torque limits for the UR5 robot (optional)
    torque_limits = [
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100)
    ]

    # Initialize the simulation
    sim = Simulation(
        urdf_file_path=ur5_urdf_file,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        time_step=0.01,
        real_time_factor=1.0
    )

    # Define a simple joint trajectory for demonstration
    num_steps = 1000
    joint_trajectory = np.linspace(
        np.zeros(len(joint_limits)),
        np.array([np.pi/2, -np.pi/4, np.pi/4, -np.pi/2, np.pi/2, -np.pi/4]),
        num=num_steps
    )

    try:
        while True:
            # Simulate robot motion using the defined trajectory
            
            sim.simulate_robot_motion(joint_trajectory)

            # Allow manual control
            sim.manual_control()

            # Check for reset button press
            if p.readUserDebugParameter(sim.reset_button) == 1:
                sim.set_joint_positions(sim.home_position)

            # Step the simulation
            p.stepSimulation()
            time.sleep(sim.time_step / sim.real_time_factor)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        sim.close_simulation()

if __name__ == "__main__":
    main()
