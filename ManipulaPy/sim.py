#!/usr/bin/env python3

import pybullet as p
import pybullet_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.control import ManipulatorController
import numpy as np
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
        self.setup_simulation()

    def setup_logger(self):
        logger = logging.getLogger('SimulationLogger')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def connect_simulation(self):
        self.logger.info("Connecting to PyBullet simulation...")
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def disconnect_simulation(self):
        self.logger.info("Disconnecting from PyBullet simulation...")
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            self.logger.info("Disconnected successfully.")

    def setup_simulation(self):
        self.connect_simulation()
        self.logger.info("Initializing simulation environment...")
        
        # Load the plane and robot URDF
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_file_path, useFixedBase=True)

        # Initialize the URDF processor
        urdf_processor = URDFToSerialManipulator(self.urdf_file_path)
        self.robot = urdf_processor.serial_manipulator
        self.dynamics = urdf_processor.dynamics
        self.non_fixed_joints = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        # Initialize the Trajectory Planner
        self.trajectory_planner = tp(
            self.robot, self.urdf_file_path, self.dynamics, 
            self.joint_limits, self.torque_limits
        )

        # Initialize the Controller
        self.controller = ManipulatorController(self.dynamics)
        self.logger.info("Simulation environment initialized.")
        self.add_joint_parameters()

    def reset_simulation(self):
        self.logger.info("Resetting simulation...")
        self.disconnect_simulation()
        self.setup_simulation()
        self.logger.info("Simulation reset.")

    def add_joint_parameters(self):
        """
        Adds GUI sliders for each joint.
        """
        self.joint_params = []
        for i, joint_index in enumerate(self.non_fixed_joints):
            param_id = p.addUserDebugParameter(f'Joint {joint_index}', self.joint_limits[i][0], self.joint_limits[i][1], 0)
            self.joint_params.append(param_id)

    def set_joint_positions(self, joint_positions):
        for i, joint_index in enumerate(self.non_fixed_joints):
            p.resetJointState(self.robot_id, joint_index, joint_positions[i])

    def get_joint_positions(self):
        joint_positions = [p.getJointState(self.robot_id, i)[0] for i in self.non_fixed_joints]
        return np.array(joint_positions)

    def run_trajectory(self, joint_trajectory):
        self.logger.info("Running trajectory...")
        for joint_positions in joint_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()
            time.sleep(self.time_step / self.real_time_factor)
        self.logger.info("Trajectory completed.")

    def run_controller(self, controller, desired_positions, desired_velocities, desired_accelerations, g, Ftip, Kp, Ki, Kd):
        self.logger.info("Running controller...")
        current_positions = self.get_joint_positions()
        current_velocities = np.zeros_like(current_positions)

        for i in range(len(desired_positions)):
            control_signal = controller.computed_torque_control(
                thetalistd=np.array(desired_positions[i]), 
                dthetalistd=np.array(desired_velocities[i]), 
                ddthetalistd=np.array(desired_accelerations[i]), 
                thetalist=current_positions, 
                dthetalist=current_velocities, 
                g=g, 
                dt=self.time_step, 
                Kp=Kp, 
                Ki=Ki, 
                Kd=Kd
            )

            self.set_joint_positions(current_positions + control_signal * self.time_step)
            current_positions = self.get_joint_positions()
            current_velocities = control_signal / self.time_step

            p.stepSimulation()
            time.sleep(self.time_step / self.real_time_factor)
        self.logger.info("Controller run completed.")

    def get_joint_parameters(self):
        """
        Gets the current values of the GUI sliders.
        """
        return [p.readUserDebugParameter(param_id) for param_id in self.joint_params]

    def simulate_robot_motion(self, desired_angles_trajectory):
        """
        Simulates the robot's motion using a given trajectory of desired joint angles.

        Args:
            desired_angles_trajectory (np.ndarray): A trajectory of desired joint angles.
        """
        self.logger.info("Simulating robot motion...")
        for joint_positions in desired_angles_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()
            time.sleep(self.time_step / self.real_time_factor)
        self.logger.info("Robot motion simulation completed.")

    def simulate_robot_with_desired_angles(self, desired_angles):
        """
        Simulates the robot using PyBullet with desired joint angles.

        Args:
            desired_angles (np.ndarray): Desired joint angles.
        """
        self.logger.info("Simulating robot with desired joint angles...")

        for i, joint_index in enumerate(self.non_fixed_joints):
            if i < len(desired_angles):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=desired_angles[i],
                    force=1000,
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
        self.logger.info("Closing simulation...")
        self.disconnect_simulation()
        self.logger.info("Simulation closed.")
    
    def manual_control(self):
        """
        Allows manual control of the robot through the PyBullet UI sliders.
        """
        self.logger.info("Starting manual control...")
        try:
            while True:
                joint_positions = self.get_joint_parameters()
                if len(joint_positions) != len(self.non_fixed_joints):
                    raise ValueError(f"Number of joint positions ({len(joint_positions)}) does not match number of non-fixed joints ({len(self.non_fixed_joints)}).")
                self.set_joint_positions(joint_positions)
                p.stepSimulation()
                time.sleep(self.time_step / self.real_time_factor)
        except KeyboardInterrupt:
            print("Manual control stopped by user.")
            self.logger.info("Manual control stopped.")

    def plot_trajectory_in_scene(self, joint_trajectory, end_effector_trajectory):
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