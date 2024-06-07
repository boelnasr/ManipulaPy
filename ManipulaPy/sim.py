#!/usr/bin/env python3

import pybullet as p
import pybullet_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.control import ManipulatorController
import numpy as np
import cupy as cp  # Import cupy for CUDA acceleration
import time
import logging
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, urdf_file_path, joint_limits, torque_limits=None, time_step=0.01, real_time_factor=1.0):
        self.urdf_file_path = urdf_file_path
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.real_time_factor = real_time_factor
        self.logger = self.setup_logger()
        self.physics_client = None
        self.joint_params = []
        self.reset_button = None
        self.home_position = None
        self.setup_simulation()

    def setup_logger(self):
        """
        Sets up the logger for the simulation.
        """
        logger = logging.getLogger('SimulationLogger')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def connect_simulation(self):
        """
        Connects to the PyBullet simulation.
        """
        self.logger.info("Connecting to PyBullet simulation...")
        self.physics_client = p.connect(p.GUI)
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
        self.connect_simulation()
        self.logger.info("Initializing simulation environment...")

        self.load_urdf_files()
        self.initialize_robot()
        self.initialize_planner_and_controller()
        self.add_joint_parameters()
        self.add_reset_button()
        self.logger.info("Simulation environment initialized.")

    def load_urdf_files(self):
        """
        Loads the necessary URDF files into the simulation.
        """
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_file_path, [0, 0, 0.1], useFixedBase=True)

    def initialize_robot(self):
        """
        Initializes the robot using the URDF processor.
        """
        urdf_processor = URDFToSerialManipulator(self.urdf_file_path)
        self.robot = urdf_processor.serial_manipulator
        self.dynamics = urdf_processor.dynamics
        self.non_fixed_joints = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.home_position = np.zeros(len(self.non_fixed_joints))

    def initialize_planner_and_controller(self):
        """
        Initializes the trajectory planner and the manipulator controller.
        """
        self.trajectory_planner = tp(self.robot, self.urdf_file_path, self.dynamics, self.joint_limits, self.torque_limits)
        self.controller = ManipulatorController(self.dynamics)

    def add_joint_parameters(self):
        """
        Adds GUI sliders for each joint.
        """
        if not self.joint_params:
            for i, joint_index in enumerate(self.non_fixed_joints):
                param_id = p.addUserDebugParameter(f'Joint {joint_index}', self.joint_limits[i][0], self.joint_limits[i][1], 0)
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
            targetPositions=joint_positions
        )

    def get_joint_positions(self):
        """
        Gets the current joint positions of the robot.
        """
        joint_positions = [p.getJointState(self.robot_id, i)[0] for i in self.non_fixed_joints]
        return np.array(joint_positions)

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

        self.plot_trajectory(ee_positions)
        self.logger.info("Trajectory completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def plot_trajectory(self, ee_positions, line_width=3, color=[1, 0, 0]):
        """
        Plots the end-effector trajectory in PyBullet using lines.
        """
        for i in range(1, len(ee_positions)):
            for j in range(line_width):
                try:
                    p.addUserDebugLine(
                        lineFromXYZ=[ee_positions[i-1][0] + j * 0.005, ee_positions[i-1][1], ee_positions[i-1][2]],
                        lineToXYZ=[ee_positions[i][0] + j * 0.005, ee_positions[i][1], ee_positions[i][2]],
                        lineColorRGB=color,
                        lifeTime=0  # Set to 0 for the line to remain indefinitely
                    )
                except Exception as e:
                    self.logger.error(f"Failed to add user debug line: {e}")

    def run_controller(self, controller, desired_positions, desired_velocities, desired_accelerations, g, Ftip, Kp, Ki, Kd):
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
                Kd=cp.array(Kd)
            )

            self.set_joint_positions(cp.asnumpy(current_positions) + cp.asnumpy(control_signal) * self.time_step)
            current_positions = self.get_joint_positions()
            current_velocities = cp.asnumpy(control_signal) / self.time_step

            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

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
            forces=[1000]*len(desired_angles)
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
        self.disconnect_simulation()
        self.logger.info("Simulation closed.")

    def check_collisions(self):
        """
        Checks for collisions in the simulation and logs them.
        """
        if self.robot_id is None:
            self.logger.warning("Cannot check for collisions before simulation is started.")
            return
        for i in self.non_fixed_joints:
            contact_points = p.getContactPoints(self.robot_id, self.robot_id, i)
            if contact_points:
                self.logger.warning(f"Collision detected at joint {i}.")
                for point in contact_points:
                    self.logger.warning(f"Contact point: {point}")

    def add_additional_parameters(self):
        """
        Adds additional GUI parameters for controlling physics properties.
        """
        if not hasattr(self, 'gravity_param') and not hasattr(self, 'time_step_param'):
            self.gravity_param = p.addUserDebugParameter("Gravity", -20, 20, -9.81)
            self.time_step_param = p.addUserDebugParameter("Time Step", 0.001, 0.1, self.time_step)

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
        self.add_additional_parameters()
        try:
            while True:
                joint_positions = self.get_joint_parameters()
                if len(joint_positions) != len(self.non_fixed_joints):
                    raise ValueError(f"Number of joint positions ({len(joint_positions)}) does not match number of non-fixed joints ({len(self.non_fixed_joints)}).")
                self.set_joint_positions(joint_positions)
                self.check_collisions()  # Check for collisions in each step
                self.update_simulation_parameters()  # Update simulation parameters

                p.stepSimulation()
                time.sleep(self.time_step / self.real_time_factor)

                # Check for reset button press
                if p.readUserDebugParameter(self.reset_button) == 1:
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
        joint_states = [p.getJointState(self.robot_id, i) for i in self.non_fixed_joints]
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]

        data = np.column_stack((positions, velocities))
        np.savetxt(filename, data, delimiter=",", header="Position,Velocity", comments="")
        self.logger.info(f"Joint states saved to {filename}.")

    def plot_trajectory_in_scene(self, joint_trajectory, end_effector_trajectory):
        """
        Plots the trajectory in the simulation scene.
        """
        self.logger.info("Plotting trajectory in simulation scene...")
        ee_positions = np.array(end_effector_trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], label='End-Effector Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
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
        mode = 'trajectory'  # Mode can be 'trajectory' or 'manual'

        while True:
            if mode == 'trajectory':
                end_pos = self.run_trajectory(joint_trajectory)
                self.logger.info("Trajectory completed. Waiting for reset...")
                mode = 'wait_reset'

            while mode == 'wait_reset' and not reset_pressed:
                p.stepSimulation()
                time.sleep(0.01)

                if p.readUserDebugParameter(self.reset_button) > 0:
                    self.logger.info("Reset button pressed. Returning to home position and entering manual control...")
                    self.set_joint_positions(self.home_position)
                    mode = 'manual'
                    break

            if mode == 'manual':
                self.manual_control()
                reset_pressed = False  # Reset the flag to restart the trajectory
                mode = 'trajectory'  # Go back to trajectory mode

    except KeyboardInterrupt:
        self.logger.info("Simulation stopped by user.")
        self.close_simulation()





