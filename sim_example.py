#!/usr/bin/env python3

import numpy as np
from ManipulaPy.sim import Simulation
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as ur5_urdf_file
import time
import pybullet as p
import imageio

def main():
    # Define the joint limits for the UR5 robot.
    # Each joint is allowed to move between -2π and 2π.
    joint_limits = [
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi)
    ]

    # Define the torque limits for the UR5 robot (optional).
    torque_limits = [
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100),
        (-100, 100)
    ]

    # Initialize the simulation with the URDF file, joint/torque limits,
    # a simulation time step, and a real time factor.
    sim = Simulation(
        urdf_file_path=ur5_urdf_file,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        time_step=0.01,
        real_time_factor=1.0
    )

    # Define a simple joint trajectory for demonstration.
    # The robot will transition from its initial (all zeros) position to a target pose.
    num_steps = 100
    joint_trajectory = np.linspace(
        np.zeros(len(joint_limits)),
        np.array([-np.pi/2, np.pi/2, np.pi/4, -np.pi/2, np.pi/2, -np.pi/4]),
        num=num_steps
    )

    # List to store frames captured from the simulation.
    frames = []

    # Set camera parameters for capturing simulation images.
    width, height = 640, 480
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.2, 1.2, 1.2],      # Position of the virtual camera.
        cameraTargetPosition=[0, 0, 0.5],         # Where the camera is looking.
        cameraUpVector=[0, 0, 1]                  # "Up" direction for the camera.
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=float(width) / height,
        nearVal=0.1,
        farVal=3.1
    )

    try:
        while True:
            # Simulate robot motion using the defined joint trajectory.
            sim.simulate_robot_motion(joint_trajectory)

            # Capture frames for each step in the trajectory.
            for _ in range(num_steps):
                # Capture a frame from the simulation.
                _, _, rgbPixels, _, _ = p.getCameraImage(
                    width, height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix
                )
                # Reshape the captured pixels into an image frame (RGBA format).
                frame = np.reshape(rgbPixels, (height, width, 4))
                frames.append(frame)
                # Advance the simulation.
                p.stepSimulation()
                time.sleep(sim.time_step / sim.real_time_factor)

            # Allow manual control of the simulation (e.g., using PyBullet GUI controls).
            sim.manual_control()

            # Check for a reset condition via a debug slider/button.
            if p.readUserDebugParameter(sim.reset_button) == 1:
                sim.set_joint_positions(sim.home_position)

            # Step the simulation and wait for the appropriate time.
            p.stepSimulation()
            time.sleep(sim.time_step / sim.real_time_factor)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        sim.close_simulation()

        # Save the collected frames as an animated GIF.
    imageio.mimsave('simulation1001.gif', frames, fps=90)
    print("Simulation saved as simulation1001.gif")

if __name__ == "__main__":
    main()
