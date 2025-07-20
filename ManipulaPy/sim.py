#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Simulation Module - ManipulaPy

This module provides PyBullet-based simulation capabilities for robotic manipulators
including real-time visualization, physics simulation, and interactive control.

UPDATED VERSION with VISIBLE TRAJECTORY SPLINE:
- Replaced addUserDebugLine() with real capsule geometry
- Trajectory splines now appear in getCameraImage() screenshots
- Added proper cleanup for trajectory visualization

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""
import pybullet as p
import pybullet_data
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.control import ManipulatorController
import numpy as np
import cupy as cp  # Import cupy for CUDA acceleration
import time
import logging
import matplotlib.pyplot as plt


class Simulation:
    def __init__(
        self,
        urdf_file_path,
        joint_limits,
        torque_limits=None,
        time_step=0.01,
        real_time_factor=1.0,
        physics_client=None,
    ):
        self.urdf_file_path = urdf_file_path
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.real_time_factor = real_time_factor
        self.logger = self.setup_logger()
        self.physics_client = physics_client
        self.joint_params = []
        self.reset_button = None
        self.home_position = None
        
        # NEW: Track trajectory visualization bodies for cleanup
        self.trajectory_body_ids = []
        
        self.setup_simulation()

    def setup_logger(self):
        """
        Sets up the logger for the simulation.
        """
        logger = logging.getLogger("SimulationLogger")
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def connect_simulation(self):
        """
        Connects to the PyBullet simulation.
        """
        self.logger.info("Connecting to PyBullet simulation...")
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI)
        p.resetSimulation()  # Clear the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def disconnect_simulation(self):
        """
        Disconnects from the PyBullet simulation.
        """
        self.logger.info("Disconnecting from PyBullet simulation...")
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            self.logger.info("Disconnected successfully.")

    def setup_simulation(self):
        """
        Sets up the simulation environment.
        """
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI)
            p.resetSimulation()
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.time_step)

            # Load the ground plane
            self.plane_id = p.loadURDF("plane.urdf")

            # Load the robot
            self.robot_id = p.loadURDF(self.urdf_file_path, useFixedBase=True)

            # Identify non-fixed joints
            self.non_fixed_joints = [
                i
                for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
            ]
            self.home_position = np.zeros(len(self.non_fixed_joints))
        else:
            print("Simulation already initialized.")

    def initialize_robot(self):
        """
        Initializes the robot using the URDF processor.
        """
        # Only skip URDF processing if self.robot is already set.
        if hasattr(self, "robot") and self.robot is not None:
            self.logger.warning("Robot already initialized. Skipping URDF processing.")
        else:
            # Even if self.robot_id is already set from setup_simulation(),
            # we need to process the URDF to set self.robot and self.dynamics.
            if not (hasattr(self, "robot_id") and self.robot_id is not None):
                self.robot_id = p.loadURDF(
                    self.urdf_file_path, [0, 0, 0.1], useFixedBase=True
                )
            # Process the URDF to generate the robot model and dynamics.
            from ManipulaPy.urdf_processor import URDFToSerialManipulator

            urdf_processor = URDFToSerialManipulator(self.urdf_file_path)
            self.robot = urdf_processor.serial_manipulator
            self.dynamics = urdf_processor.dynamics
            # Identify non-fixed joints
            self.non_fixed_joints = [
                i
                for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
            ]
            self.home_position = np.zeros(len(self.non_fixed_joints))

    def set_robot_models(self, robot, dynamics):
        """
        Set pre-existing robot models to avoid reprocessing.
        
        Args:
            robot: SerialManipulator instance
            dynamics: ManipulatorDynamics instance
        """
        self.robot = robot
        self.dynamics = dynamics
        self.logger.info("Pre-existing robot models set successfully.")

    def initialize_planner_and_controller(self):
        """
        Initializes the trajectory planner and the manipulator controller.
        """
        self.trajectory_planner = tp(
            self.robot,
            self.urdf_file_path,
            self.dynamics,
            self.joint_limits,
            self.torque_limits,
        )
        self.controller = ManipulatorController(self.dynamics)

    def add_joint_parameters(self):
        """
        Adds GUI sliders for each joint.
        """
        if not self.joint_params:
            for i, joint_index in enumerate(self.non_fixed_joints):
                param_id = p.addUserDebugParameter(
                    f"Joint {joint_index}",
                    self.joint_limits[i][0],
                    self.joint_limits[i][1],
                    0,
                )
                self.joint_params.append(param_id)

    def add_reset_button(self):
        """
        Adds a reset button to the simulation.
        """
        if self.reset_button is None:
            try:
                self.reset_button = p.addUserDebugParameter("Reset", 1, 0, 1)
            except Exception as e:
                self.logger.error(f"Failed to add reset button: {e}")

    def set_joint_positions(self, joint_positions):
        """
        Sets the joint positions of the robot.
        """
        p.setJointMotorControlArray(
            self.robot_id,
            self.non_fixed_joints,
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
        )

    def get_joint_positions(self):
        """
        Gets the current joint positions of the robot.
        """
        joint_positions = [
            p.getJointState(self.robot_id, i)[0] for i in self.non_fixed_joints
        ]
        return np.array(joint_positions)

    def _capsule_line(self, a, b, radius=0.006, rgba=(1, 0.5, 0, 1)):
        """
        Create a thin capsule between point a and b; returns body-id.
        This creates REAL GEOMETRY that appears in getCameraImage() screenshots.
        
        Args:
            a: Start point [x, y, z]
            b: End point [x, y, z]
            radius: Capsule radius in world units
            rgba: Color as [r, g, b, a] where values are 0-1
        
        Returns:
            int: PyBullet body ID, or -1 if failed
        """
        a, b = np.array(a), np.array(b)
        v = b - a
        L = np.linalg.norm(v)
        
        if L < 1e-6:
            return -1
        
        # Calculate orientation to align capsule with the line direction
        z = v / L  # Direction vector
        
        # Find perpendicular vectors
        x = np.cross([0, 0, 1], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([0, 1, 0], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        # Calculate proper orientation for capsule
        # PyBullet capsules are aligned with Z-axis by default
        if abs(z[2]) > 0.99:  # Nearly vertical
            orn = p.getQuaternionFromEuler([0, 0, 0])
        else:
            # Calculate rotation to align Z-axis with direction vector
            angle = np.arccos(np.clip(z[2], -1, 1))
            if angle > 1e-6:
                axis = np.cross([0, 0, 1], z)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    orn = p.getQuaternionFromAxisAngle(axis, angle)
                else:
                    orn = p.getQuaternionFromEuler([0, 0, 0])
            else:
                orn = p.getQuaternionFromEuler([0, 0, 0])
        
        # Midpoint of the line segment
        mid = (a + b) / 2
        
        try:
            # Create collision and visual shapes
            col = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=L)
            vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=L, rgbaColor=rgba)
            
            # Create static body (mass=0)
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=mid,
                baseOrientation=orn
            )
            
            return body_id
            
        except Exception as e:
            self.logger.error(f"Failed to create capsule line: {e}")
            return -1

    def plot_trajectory(self, ee_positions, line_width=3, color=[1, 0, 0]):
        """
        Plots the end-effector trajectory in PyBullet using REAL GEOMETRY.
        
        This method now creates actual 3D capsules that will appear in screenshots
        taken with getCameraImage(), unlike the previous addUserDebugLine() approach.
        
        Args:
            ee_positions: List of end-effector positions [[x,y,z], ...]
            line_width: Width factor for trajectory visualization
            color: RGB color as [r, g, b] where values are 0-1
        
        Returns:
            list: Body IDs of created trajectory geometry (for cleanup)
        """
        # Clear any existing trajectory bodies
        self.clear_trajectory_visualization()
        
        if len(ee_positions) < 2:
            self.logger.warning("Not enough positions to plot trajectory")
            return []
        
        # Convert color to RGBA
        if len(color) == 3:
            rgba_color = color + [1.0]  # Add alpha
        else:
            rgba_color = color
        
        # Calculate radius based on line_width (convert to world scale)
        base_radius = 0.003  # Base radius in world units
        radius = base_radius * (line_width / 3.0)  # Scale with line_width
        
        trajectory_bodies = []
        
        self.logger.info(f"Creating trajectory visualization with {len(ee_positions)} points")
        
        # Create capsule segments between consecutive points
        for i in range(1, len(ee_positions)):
            try:
                # Get consecutive points
                start_pos = ee_positions[i - 1]
                end_pos = ee_positions[i]
                
                # Create multiple parallel capsules for thickness effect
                for j in range(max(1, line_width // 2)):
                    # Slight offset for thickness
                    offset = j * 0.002  # Small offset in world units
                    
                    start_offset = [
                        start_pos[0] + offset,
                        start_pos[1],
                        start_pos[2]
                    ]
                    end_offset = [
                        end_pos[0] + offset,
                        end_pos[1],
                        end_pos[2]
                    ]
                    
                    # Create capsule segment
                    body_id = self._capsule_line(
                        start_offset,
                        end_offset,
                        radius=radius,
                        rgba=rgba_color
                    )
                    
                    if body_id != -1:
                        trajectory_bodies.append(body_id)
                        
            except Exception as e:
                self.logger.error(f"Failed to create trajectory segment {i}: {e}")
        
        # Store body IDs for cleanup
        self.trajectory_body_ids.extend(trajectory_bodies)
        
        # Add trajectory markers
        marker_bodies = self._add_trajectory_markers(ee_positions, rgba_color)
        self.trajectory_body_ids.extend(marker_bodies)
        
        self.logger.info(f"✅ Created trajectory visualization: {len(trajectory_bodies)} segments + {len(marker_bodies)} markers")
        self.logger.info("🎯 Trajectory will now appear in screenshots as 3D geometry!")
        
        return trajectory_bodies

    def _add_trajectory_markers(self, ee_positions, color):
        """
        Add START/END markers using real geometry.
        
        Args:
            ee_positions: List of end-effector positions
            color: RGBA color for markers
        
        Returns:
            list: Body IDs of created markers
        """
        marker_bodies = []
        
        try:
            # START marker (green sphere)
            start_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0.0, 1.0, 0.0, 1.0]  # Green
            )
            start_collision = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02
            )
            start_marker = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=start_collision,
                baseVisualShapeIndex=start_visual,
                basePosition=[
                    ee_positions[0][0],
                    ee_positions[0][1],
                    ee_positions[0][2] + 0.05
                ]
            )
            marker_bodies.append(start_marker)
            
            # END marker (red sphere)
            end_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Red
            )
            end_collision = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.02
            )
            end_marker = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=end_collision,
                baseVisualShapeIndex=end_visual,
                basePosition=[
                    ee_positions[-1][0],
                    ee_positions[-1][1],
                    ee_positions[-1][2] + 0.05
                ]
            )
            marker_bodies.append(end_marker)
            
            # Add intermediate waypoints if trajectory is long enough
            if len(ee_positions) > 10:
                waypoint_indices = [
                    len(ee_positions) // 4,
                    len(ee_positions) // 2,
                    3 * len(ee_positions) // 4
                ]
                
                for idx in waypoint_indices:
                    if 0 <= idx < len(ee_positions):
                        waypoint_visual = p.createVisualShape(
                            shapeType=p.GEOM_SPHERE,
                            radius=0.015,
                            rgbaColor=[0.0, 0.0, 1.0, 1.0]  # Blue
                        )
                        waypoint_collision = p.createCollisionShape(
                            shapeType=p.GEOM_SPHERE,
                            radius=0.015
                        )
                        waypoint_marker = p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=waypoint_collision,
                            baseVisualShapeIndex=waypoint_visual,
                            basePosition=[
                                ee_positions[idx][0],
                                ee_positions[idx][1],
                                ee_positions[idx][2] + 0.03
                            ]
                        )
                        marker_bodies.append(waypoint_marker)
            
        except Exception as e:
            self.logger.error(f"Failed to create trajectory markers: {e}")
        
        return marker_bodies

    def clear_trajectory_visualization(self):
        """
        Clear all trajectory visualization bodies from the simulation.
        """
        if hasattr(self, 'trajectory_body_ids'):
            removed_count = 0
            for body_id in self.trajectory_body_ids:
                try:
                    p.removeBody(body_id)
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not remove trajectory body {body_id}: {e}")
            
            if removed_count > 0:
                self.logger.info(f"🧹 Removed {removed_count} trajectory visualization bodies")
            
            self.trajectory_body_ids = []

    def run_trajectory(self, joint_trajectory):
        """
        Runs a joint trajectory in the simulation.
        """
        self.logger.info("Running trajectory...")
        ee_positions = []

        for joint_positions in joint_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Trajectory completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def run_controller(
        self,
        controller,
        desired_positions,
        desired_velocities,
        desired_accelerations,
        g,
        Ftip,
        Kp,
        Ki,
        Kd,
    ):
        """
        Runs the controller with the specified parameters.
        """
        self.logger.info("Running controller...")
        current_positions = self.get_joint_positions()
        current_velocities = np.zeros_like(current_positions)
        ee_positions = []

        for i in range(len(desired_positions)):
            control_signal = controller.computed_torque_control(
                thetalistd=cp.array(desired_positions[i]),
                dthetalistd=cp.array(desired_velocities[i]),
                ddthetalistd=cp.array(desired_accelerations[i]),
                thetalist=cp.array(current_positions),
                dthetalist=cp.array(current_velocities),
                g=cp.array(g),
                dt=self.time_step,
                Kp=cp.array(Kp),
                Ki=cp.array(Ki),
                Kd=cp.array(Kd),
            )

            self.set_joint_positions(
                cp.asnumpy(current_positions)
                + cp.asnumpy(control_signal) * self.time_step
            )
            current_positions = self.get_joint_positions()
            current_velocities = cp.asnumpy(control_signal) / self.time_step

            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Controller run completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def get_joint_parameters(self):
        """
        Gets the current values of the GUI sliders.
        """
        return [p.readUserDebugParameter(param_id) for param_id in self.joint_params]

    def simulate_robot_motion(self, desired_angles_trajectory):
        """
        Simulates the robot's motion using a given trajectory of desired joint angles.
        """
        self.logger.info("Simulating robot motion...")
        ee_positions = []

        for joint_positions in desired_angles_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Robot motion simulation completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def simulate_robot_with_desired_angles(self, desired_angles):
        """
        Simulates the robot using PyBullet with desired joint angles.

        Args:
            desired_angles (np.ndarray): Desired joint angles.
        """
        self.logger.info("Simulating robot with desired joint angles...")

        p.setJointMotorControlArray(
            self.robot_id,
            self.non_fixed_joints,
            p.POSITION_CONTROL,
            targetPositions=desired_angles,
            forces=[1000] * len(desired_angles),
        )

        time_step = 0.00001
        p.setTimeStep(time_step)
        try:
            while True:
                p.stepSimulation()
                time.sleep(time_step / self.real_time_factor)
        except KeyboardInterrupt:
            print("Simulation stopped by user.")
            self.logger.info("Robot simulation with desired angles completed.")

    def close_simulation(self):
        """
        Closes the simulation.
        """
        self.logger.info("Closing simulation...")
        
        # Clear trajectory visualization
        self.clear_trajectory_visualization()
        
        self.disconnect_simulation()
        self.logger.info("Simulation closed.")

    def check_collisions(self):
        """
        Checks for collisions in the simulation and logs them.
        """
        if self.robot_id is None:
            self.logger.warning(
                "Cannot check for collisions before simulation is started."
            )
            return
        for i in self.non_fixed_joints:
            contact_points = p.getContactPoints(self.robot_id, self.robot_id, i)
            if contact_points:
                self.logger.warning(f"Collision detected at joint {i}.")
                for point in contact_points:
                    self.logger.warning(f"Contact point: {point}")

    def step_simulation(self):
        """
        Steps the simulation forward by one time step.
        """
        self.logger.info("Setting up the simulation environment...")
        self.connect_simulation()
        self.add_additional_parameters()

    def add_additional_parameters(self):
        """
        Adds additional GUI parameters for controlling physics properties like gravity and time step.
        """
        if not hasattr(self, "gravity_param"):
            self.gravity_param = p.addUserDebugParameter("Gravity", -20, 20, -9.81)
        if not hasattr(self, "time_step_param"):
            self.time_step_param = p.addUserDebugParameter(
                "Time Step", 0.001, 0.1, self.time_step
            )

    def update_simulation_parameters(self):
        """
        Updates simulation parameters from GUI controls.
        """
        gravity = p.readUserDebugParameter(self.gravity_param)
        time_step = p.readUserDebugParameter(self.time_step_param)
        p.setGravity(0, 0, gravity)
        p.setTimeStep(time_step)
        self.time_step = time_step

    def manual_control(self):
        """
        Allows manual control of the robot through the PyBullet UI sliders.
        """
        self.logger.info("Starting manual control...")
        if not self.joint_params:
            self.add_joint_parameters()  # Ensure sliders are created
        self.add_additional_parameters()  # Additional controls like gravity and time step

        # Add reset button if it doesn't exist
        if self.reset_button is None:
            self.add_reset_button()

        try:
            while True:
                joint_positions = self.get_joint_parameters()
                if len(joint_positions) != len(self.non_fixed_joints):
                    raise ValueError(
                        f"Number of joint positions ({len(joint_positions)}) does not match number of non-fixed joints ({len(self.non_fixed_joints)})."
                    )
                self.set_joint_positions(joint_positions)
                self.check_collisions()  # Check for collisions in each step
                self.update_simulation_parameters()  # Update simulation parameters

                p.stepSimulation()
                time.sleep(self.time_step / self.real_time_factor)

                # Check if reset button exists before reading it
                if (
                    self.reset_button is not None
                    and p.readUserDebugParameter(self.reset_button) == 1
                ):
                    self.logger.info("Resetting simulation state...")
                    self.set_joint_positions(self.home_position)
                    break  # Exit manual control to restart trajectory loop
        except KeyboardInterrupt:
            print("Manual control stopped by user.")
            self.logger.info("Manual control stopped.")

    def save_joint_states(self, filename="joint_states.csv"):
        """
        Saves the joint states to a CSV file.

        Args:
            filename (str): The filename for the CSV file.
        """
        joint_states = [
            p.getJointState(self.robot_id, i) for i in self.non_fixed_joints
        ]
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]

        data = np.column_stack((positions, velocities))
        np.savetxt(
            filename, data, delimiter=",", header="Position,Velocity", comments=""
        )
        self.logger.info(f"Joint states saved to {filename}.")

    def plot_trajectory_in_scene(self, joint_trajectory, end_effector_trajectory):
        """
        Plots the trajectory in the simulation scene.
        """
        self.logger.info("Plotting trajectory in simulation scene...")
        ee_positions = np.array(end_effector_trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            label="End-Effector Trajectory",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()

        self.run_trajectory(joint_trajectory)
        self.logger.info("Trajectory plotted and simulation completed.")

    def run(self, joint_trajectory):
        """
        Main loop for running the simulation.
        """
        try:
            reset_pressed = False
            mode = "trajectory"  # Mode can be 'trajectory' or 'manual'

            while True:
                if mode == "trajectory":
                    end_pos = self.run_trajectory(joint_trajectory)
                    self.logger.info("Trajectory completed. Waiting for reset...")
                    mode = "wait_reset"

                while mode == "wait_reset" and not reset_pressed:
                    p.stepSimulation()
                    time.sleep(0.01)

                    if p.readUserDebugParameter(self.reset_button) > 0:
                        self.logger.info(
                            "Reset button pressed. Returning to home position and entering manual control..."
                        )
                        self.set_joint_positions(self.home_position)
                        mode = "manual"
                        break

                if mode == "manual":
                    self.manual_control()
                    reset_pressed = False  # Reset the flag to restart the trajectory
                    mode = "trajectory"  # Go back to trajectory mode

        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
            self.close_simulation()

    def __del__(self):
        """
        Destructor to clean up trajectory visualization when simulation is destroyed.
        """
        try:
            if hasattr(self, 'trajectory_body_ids'):
                self.clear_trajectory_visualization()
        except Exception:
            pass  # Ignore errors during cleanup