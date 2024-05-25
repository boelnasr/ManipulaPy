#!/usr/bin/env python3

import numpy as np
import pybullet as p
import pybullet_data
import time
import logging
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning
from ManipulaPy.controller import ManipulatorController

class Simulation:
    def __init__(self, urdf_file_path, joint_limits, torque_limits=None, time_step=0.01):
        self.urdf_file_path = urdf_file_path
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.logger = self.setup_logger()
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

    def setup_simulation(self):
        self.logger.info("Initializing simulation...")
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load the plane and robot URDF
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_file_path, useFixedBase=True)

        # Initialize the URDF processor
        urdf_processor = URDFToSerialManipulator(self.urdf_file_path)
        self.robot = urdf_processor.serial_manipulator
        self.dynamics = urdf_processor.dynamics

        # Initialize the Trajectory Planner
        self.trajectory_planner = TrajectoryPlanning(urdf_processor)

        # Initialize the Controller
        self.controller = ManipulatorController(self.dynamics)
        self.logger.info("Simulation initialized.")

    def reset_simulation(self):
        self.logger.info("Resetting simulation...")
        p.resetSimulation()
        self.setup_simulation()
        self.logger.info("Simulation reset.")

    def set_joint_positions(self, joint_positions):
        for i in range(len(joint_positions)):
            p.resetJointState(self.robot_id, i, joint_positions[i])

    def get_joint_positions(self):
        joint_positions = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_positions.append(p.getJointState(self.robot_id, i)[0])
        return np.array(joint_positions)

    def run_trajectory(self, joint_trajectory):
        self.logger.info("Running trajectory...")
        for joint_positions in joint_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()
            time.sleep(self.time_step)
        self.logger.info("Trajectory completed.")

    def run_controller(self, controller, desired_positions, desired_velocities, desired_accelerations, g, Ftip, Kp, Ki, Kd):
        self.logger.info("Running controller...")
        current_positions = self.get_joint_positions()
        current_velocities = np.zeros_like(current_positions)

        for i in range(len(desired_positions)):
            control_signal = controller.control(
                desired_positions[i], 
                desired_velocities[i], 
                desired_accelerations[i], 
                current_positions, 
                current_velocities, 
                g, 
                Ftip, 
                Kp, 
                Ki, 
                Kd
            )

            self.set_joint_positions(current_positions + control_signal * self.time_step)
            current_positions = self.get_joint_positions()
            current_velocities = control_signal / self.time_step

            p.stepSimulation()
            time.sleep(self.time_step)
        self.logger.info("Controller run completed.")

    def close_simulation(self):
        self.logger.info("Closing simulation...")
        p.disconnect()
        self.logger.info("Simulation closed.")

if __name__ == "__main__":
    # Example usage:
    urdf_file_path = "path_to_urdf_file.urdf"
    joint_limits = [(-np.pi, np.pi)] * 6
    torque_limits = [(-100, 100)] * 6
    sim = Simulation(urdf_file_path, joint_limits, torque_limits)

    # Example to run trajectory
    thetastart = [0] * 6
    thetaend = [np.pi / 4] * 6
    Tf = 5
    N = 100
    traj = sim.trajectory_planner.JointTrajectory(thetastart, thetaend, Tf, N, method=5)
    sim.run_trajectory(traj['positions'])

    # Example to run controller
    desired_positions = traj['positions']
    desired_velocities = traj['velocities']
    desired_accelerations = traj['accelerations']
    g = [0, 0, -9.81]
    Ftip = [0, 0, 0, 0, 0, 0]
    Kp, Ki, Kd = 100, 0, 20
    sim.run_controller(sim.controller, desired_positions, desired_velocities, desired_accelerations, g, Ftip, Kp, Ki, Kd)

    sim.close_simulation()
