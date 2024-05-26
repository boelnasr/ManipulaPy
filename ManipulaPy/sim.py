#!/usr/bin/env python3
import pybullet as p  # Add this import statement
import pybullet_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.singularity import Singularity
from ManipulaPy.control import ManipulatorController 
from math import pi
import numpy as np
import time
import logging

class Simulation:
    def __init__(self, urdf_file_path, joint_limits, torque_limits=None, time_step=0.01, real_time_factor=1.0):
        self.urdf_file_path = urdf_file_path
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.real_time_factor = real_time_factor
        self.logger = self.setup_logger()
        self.physics_client = None
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

        # Initialize the Trajectory Planner
        self.trajectory_planner = tp(
            self.robot, self.urdf_file_path, self.dynamics, 
            self.joint_limits, self.torque_limits
        )

        # Initialize the Controller
        self.controller = ManipulatorController(self.dynamics)
        self.logger.info("Simulation environment initialized.")

    def reset_simulation(self):
        self.logger.info("Resetting simulation...")
        self.disconnect_simulation()
        self.setup_simulation()
        self.logger.info("Simulation reset.")

    def set_joint_positions(self, joint_positions):
        for i in range(len(joint_positions)):
            p.resetJointState(self.robot_id, i, joint_positions[i])

    def get_joint_positions(self):
        joint_positions = [p.getJointState(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id))]
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

        numJoints = p.getNumJoints(self.robot_id)
        for i in range(numJoints):
            if i < len(desired_angles):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=desired_angles[i],
                    force=1000,
                )
            else:
                p.setJointMotorControl2(
                    self.robot_id, i, p.POSITION_CONTROL, targetPosition=0, force=1000
                )

        time_step = 1 
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