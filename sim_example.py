#!/usr/bin/env python3

import numpy as np
from ManipulaPy.sim import Simulation
from ManipulaPy.ManipulaPy_data.ur5 import urdf_file as ur5_urdf_file
import time
import pybullet as p
import imageio

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
    num_steps = 100
    joint_trajectory = np.linspace(
        np.zeros(len(joint_limits)),
        np.array([np.pi/2, -np.pi/4, np.pi/4, -np.pi/2, np.pi/2, -np.pi/4]),
        num=num_steps
    )

    # List to store frames
    frames = []

    # Set camera parameters
    width, height = 640, 480
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.2, 1.2, 1.2],
        cameraTargetPosition=[0, 0, 0.5],
        cameraUpVector=[0, 0, 1]
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=float(width) / height,
        nearVal=0.1,
        farVal=3.1
    )

    try:
        while True:
            # Simulate robot motion using the defined trajectory
            sim.simulate_robot_motion(joint_trajectory)

            # Capture frames
            for _ in range(num_steps):
                # Capture frame
                _, _, rgbPixels, _, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
                frame = np.reshape(rgbPixels, (height, width, 4))
                frames.append(frame)
                p.stepSimulation()
                time.sleep(sim.time_step / sim.real_time_factor)

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

        # Save frames as GIF
    imageio.mimsave('simulation1001.gif', frames, fps=90)
    print("Simulation saved as simulation.gif")

if __name__ == "__main__":
    main()

