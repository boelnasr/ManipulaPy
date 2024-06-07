#!/usr/bin/env python3

from numba import cuda, float32
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from .utils import (
    TransToRp,
    MatrixLog3,
    MatrixExp3,
    CubicTimeScaling,
    QuinticTimeScaling,
)
from urchin.urdf import URDF
from .cuda_kernels import trajectory_kernel, inverse_dynamics_kernel, forward_dynamics_kernel, cartesian_trajectory_kernel
from .potential_field import CollisionChecker, PotentialField
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
class TrajectoryPlanning:
    def __init__(self, serial_manipulator, urdf_path, dynamics, joint_limits, torque_limits=None):
        """
        Initializes a TrajectoryPlanning object.

        Args:
            serial_manipulator (SerialManipulator): An instance of SerialManipulator.
            urdf_path (str): The path to the URDF file.
            dynamics (ManipulatorDynamics): An instance of ManipulatorDynamics.
            joint_limits (list): A list of tuples representing the joint limits.
            torque_limits (list, optional): A list of tuples representing the torque limits. Defaults to None.
        """
        self.serial_manipulator = serial_manipulator
        self.dynamics = dynamics
        self.joint_limits = np.array(joint_limits)
        self.torque_limits = (
            np.array(torque_limits)
            if torque_limits
            else np.array([[-np.inf, np.inf]] * len(joint_limits))
        )
        self.collision_checker = CollisionChecker(urdf_path)
        self.potential_field = PotentialField()

    def joint_trajectory(self, thetastart, thetaend, Tf, N, method):
        """
        Generates a joint trajectory for a robot based on the given start and end joint angles, final time, and number of steps.

        Args:
            thetastart (numpy.ndarray): The starting joint angles.
            thetaend (numpy.ndarray): The ending joint angles.
            Tf (float): The final time for the trajectory.
            N (int): The number of steps in the trajectory.
            method (str): The method to use for generating the trajectory.

        Returns:
            dict: A dictionary containing the positions, velocities, and accelerations of the joint trajectory.
        """
        logger.info("Generating joint trajectory.")

        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        traj_pos = np.zeros((N, num_joints), dtype=np.float32)
        traj_vel = np.zeros((N, num_joints), dtype=np.float32)
        traj_acc = np.zeros((N, num_joints), dtype=np.float32)

        threads_per_block = 256  # Adjusted threads per block for better GPU utilization
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)

        d_thetastart = cuda.to_device(thetastart)
        d_thetaend = cuda.to_device(thetaend)
        d_traj_pos = cuda.device_array_like(traj_pos)
        d_traj_vel = cuda.device_array_like(traj_vel)
        d_traj_acc = cuda.device_array_like(traj_acc)

        try:
            trajectory_kernel[blocks_per_grid, threads_per_block](
                d_thetastart, d_thetaend, d_traj_pos, d_traj_vel, d_traj_acc, Tf, N, method
            )

            d_traj_pos.copy_to_host(traj_pos)
            d_traj_vel.copy_to_host(traj_vel)
            d_traj_acc.copy_to_host(traj_acc)
        finally:
            # Explicitly free GPU memory using del statement
            del d_thetastart
            del d_thetaend
            del d_traj_pos
            del d_traj_vel
            del d_traj_acc

        for i in range(num_joints):
            traj_pos[:, i] = np.clip(
                traj_pos[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1]
            )

        # Initialize potential field parameters
        q_goal = thetaend
        obstacles = []  # Define obstacles here as needed

        # Apply potential field for collision avoidance
        for idx, step in enumerate(traj_pos):
            if self.collision_checker.check_collision(step):
                for _ in range(100):  # Max iterations to adjust trajectory
                    gradient = self.potential_field.compute_gradient(step, q_goal, obstacles)
                    step -= 0.01 * gradient  # Adjust step size as needed
                    if not self.collision_checker.check_collision(step):
                        break
                traj_pos[idx] = step

        return {
            "positions": traj_pos,
            "velocities": traj_vel,
            "accelerations": traj_acc,
        }


    def inverse_dynamics_trajectory(
        self, thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory, gravity_vector=None, Ftip=None):
        """
        Compute joint torques with enforced torque limits based on a trajectory using CUDA acceleration.

        Args:
            thetalist_trajectory (np.ndarray): Array of joint angles over the trajectory.
            dthetalist_trajectory (np.ndarray): Array of joint velocities over the trajectory.
            ddthetalist_trajectory (np.ndarray): Array of joint accelerations over the trajectory.
            gravity_vector (np.ndarray, optional): Gravity vector affecting the system, defaulting to [0, 0, -9.81].
            Ftip (list, optional): External forces applied at the end effector, defaulting to [0, 0, 0, 0, 0, 0].

        Returns:
            np.ndarray: Array of joint torques required to follow the trajectory.
        """
        if gravity_vector is None:
            gravity_vector = np.array([0, 0, -9.81])
        if Ftip is None:
            Ftip = [0, 0, 0, 0, 0, 0]

        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]
        torques_trajectory = np.zeros((num_points, num_joints), dtype=np.float32)

        threads_per_block = 1024
        blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)

        d_thetalist_trajectory = cuda.to_device(thetalist_trajectory)
        d_dthetalist_trajectory = cuda.to_device(dthetalist_trajectory)
        d_ddthetalist_trajectory = cuda.to_device(ddthetalist_trajectory)
        d_gravity_vector = cuda.to_device(gravity_vector)
        d_Ftip = cuda.to_device(np.array(Ftip, dtype=np.float32))
        d_Glist = cuda.to_device(np.array(self.dynamics.Glist, dtype=np.float32))
        d_Slist = cuda.to_device(np.array(self.dynamics.S_list, dtype=np.float32))
        d_M = cuda.to_device(np.array(self.dynamics.M_list, dtype=np.float32))
        d_torques_trajectory = cuda.device_array_like(torques_trajectory)
        d_torque_limits = cuda.to_device(self.torque_limits)

        inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
            d_thetalist_trajectory, d_dthetalist_trajectory, d_ddthetalist_trajectory,
            d_gravity_vector, d_Ftip, d_Glist, d_Slist, d_M, d_torques_trajectory, d_torque_limits)

        d_torques_trajectory.copy_to_host(torques_trajectory)

        if self.torque_limits is not None:
            torques_trajectory = np.clip(
                torques_trajectory, self.torque_limits[:, 0], self.torque_limits[:, 1])

        return torques_trajectory

    def forward_dynamics_trajectory(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """
        Calculates the forward dynamics trajectory of a robotic system given the joint angles, joint velocities, joint torques, gravity vector, and external forces.

        Args:
            thetalist (np.ndarray): Array of joint angles over the trajectory.
            dthetalist (np.ndarray): Array of joint velocities over the trajectory.
            taumat (np.ndarray): Array of joint torques over the trajectory.
            g (np.ndarray): Gravity vector affecting the system.
            Ftipmat (np.ndarray): Array of external forces applied at the end effector.
            dt (float): Time step for the trajectory.
            intRes (int): Number of integration steps per time step.

        Returns:
            dict: Dictionary containing the joint positions, joint velocities, and joint accelerations over the trajectory.
                - thetamat (np.ndarray): Array of joint positions over the trajectory.
                - dthetamat (np.ndarray): Array of joint velocities over the trajectory.
                - ddthetamat (np.ndarray): Array of joint accelerations over the trajectory.
        """
        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]
        thetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        dthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        ddthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        thetamat[0, :] = thetalist
        dthetamat[0, :] = dthetalist
        threads_per_block = 1024
        blocks_per_grid = (num_steps + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)
        d_thetalist = cuda.to_device(thetalist)
        d_dthetalist = cuda.to_device(dthetalist)
        d_taumat = cuda.to_device(taumat)
        d_g = cuda.to_device(g)
        d_Ftipmat = cuda.to_device(Ftipmat)
        d_Glist = cuda.to_device(np.array(self.dynamics.Glist, dtype=np.float32))
        d_Slist = cuda.to_device(np.array(self.dynamics.S_list, dtype=np.float32))
        d_M = cuda.to_device(np.array(self.dynamics.M_list, dtype=np.float32))
        d_thetamat = cuda.device_array_like(thetamat)
        d_dthetamat = cuda.device_array_like(dthetamat)
        d_ddthetamat = cuda.device_array_like(ddthetamat)
        d_joint_limits = cuda.to_device(self.joint_limits)
        forward_dynamics_kernel[blocks_per_grid, threads_per_block](
            d_thetalist, d_dthetalist, d_taumat, d_g, d_Ftipmat, dt, intRes,
            d_Glist, d_Slist, d_M, d_thetamat, d_dthetamat, d_ddthetamat, d_joint_limits)
        d_thetamat.copy_to_host(thetamat)
        d_dthetamat.copy_to_host(dthetamat)
        d_ddthetamat.copy_to_host(ddthetamat)
        return {
            "positions": thetamat,
            "velocities": dthetamat,
            "accelerations": ddthetamat,
        }

    def cartesian_trajectory(self, Xstart, Xend, Tf, N, method):
        """
        Generates a Cartesian trajectory between two end-effector configurations in SE(3).

        Args:
            Xstart (np.ndarray): The initial end-effector configuration (SE(3) matrix).
            Xend (np.ndarray): The final end-effector configuration (SE(3) matrix).
            Tf (float): The total time of the motion in seconds from rest to rest.
            N (int): The number of points N > 1 (Start and stop) in the discrete representation of the trajectory.
            method (int): The time-scaling method, where 3 indicates cubic (third-order polynomial) time scaling and 5 indicates quintic (fifth-order polynomial) time scaling.

        Returns:
            dict: A dictionary containing the following keys:
                - "positions" (np.ndarray): The trajectory positions as an array of shape (N, 3).
                - "velocities" (np.ndarray): The trajectory velocities as an array of shape (N, 3).
                - "accelerations" (np.ndarray): The trajectory accelerations as an array of shape (N, 3).
                - "orientations" (np.ndarray): The trajectory orientations as an array of shape (N, 3, 3).
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [None] * N
        Rstart, pstart = TransToRp(Xstart)
        Rend, pend = TransToRp(Xend)

        orientations = np.zeros((N, 3, 3), dtype=np.float32)

        for i in range(N):
            if method == 3:
                s = CubicTimeScaling(Tf, timegap * i)
            else:
                s = QuinticTimeScaling(Tf, timegap * i)
            traj[i] = np.r_[
                np.c_[
                    np.dot(Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)),
                    s * pend + (1 - s) * pstart,
                ],
                [[0, 0, 0, 1]],
            ]
            orientations[i] = np.dot(
                Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)
            )

        traj_pos = np.array([TransToRp(T)[1] for T in traj], dtype=np.float32)

        pstart = np.ascontiguousarray(pstart)
        pend = np.ascontiguousarray(pend)
        traj_pos = np.ascontiguousarray(traj_pos)

        traj_vel = np.zeros((N, 3), dtype=np.float32)
        traj_acc = np.zeros((N, 3), dtype=np.float32)

        threads_per_block = 256
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)

        d_pstart = cuda.to_device(pstart)
        d_pend = cuda.to_device(pend)
        d_traj_pos = cuda.to_device(traj_pos)
        d_traj_vel = cuda.device_array_like(traj_vel)
        d_traj_acc = cuda.device_array_like(traj_acc)

        cartesian_trajectory_kernel[blocks_per_grid, threads_per_block](
            d_pstart, d_pend, d_traj_pos, d_traj_vel, d_traj_acc, Tf, N, method
        )

        d_traj_pos.copy_to_host(traj_pos)
        d_traj_vel.copy_to_host(traj_vel)
        d_traj_acc.copy_to_host(traj_acc)

        return {
            "positions": traj_pos,
            "velocities": traj_vel,
            "accelerations": traj_acc,
            "orientations": orientations,
        }

    @staticmethod
    def plot_trajectory(trajectory_data, Tf, title="Joint Trajectory", labels=None):
        """
        Plot the joint trajectory.

        Parameters:
            trajectory_data (dict): A dictionary containing the joint trajectory data.
                It should have the following keys:
                - "positions" (ndarray): The array of joint positions.
                - "velocities" (ndarray): The array of joint velocities.
                - "accelerations" (ndarray): The array of joint accelerations.
            Tf (float): The total duration of the trajectory.
            title (str, optional): The title of the plot. Defaults to "Joint Trajectory".
            labels (list, optional): The labels for each joint. If provided, it should have the same length as the number of joints.
                Defaults to None.

        Returns:
            None
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        num_joints = positions.shape[1]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, num_joints, figsize=(15, 10), sharex="col")
        fig.suptitle(title)

        for i in range(num_joints):
            if labels and len(labels) == num_joints:
                label = labels[i]
            else:
                label = f"Joint {i+1}"

            axs[0, i].plot(time_steps, positions[:, i], label=f"{label} Position")
            axs[0, i].set_ylabel("Position")
            axs[0, i].legend()

            axs[1, i].plot(time_steps, velocities[:, i], label=f"{label} Velocity")
            axs[1, i].set_ylabel("Velocity")
            axs[1, i].legend()

            axs[2, i].plot(time_steps, accelerations[:, i], label=f"{label} Acceleration")
            axs[2, i].set_ylabel("Acceleration")
            axs[2, i].legend()

        for ax in axs[-1]:
            ax.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def plot_tcp_trajectory(self, trajectory, dt):
        """
        Plots the trajectory of the TCP (Tool Center Point) of a serial manipulator.
        
        Args:
            trajectory (list): A list of joint angle configurations representing the trajectory.
            dt (float): The time step between consecutive points in the trajectory.
        
        Returns:
            None
        """
        tcp_trajectory = [
            self.serial_manipulator.forward_kinematics(joint_angles)
            for joint_angles in trajectory
        ]
        tcp_positions = [pose[:3, 3] for pose in tcp_trajectory]

        velocity, acceleration, jerk = self.calculate_derivatives(tcp_positions, dt)
        time = np.arange(0, len(tcp_positions) * dt, dt)

        plt.figure(figsize=(12, 8))
        for i, label in enumerate(["X", "Y", "Z"]):
            plt.subplot(4, 1, 1)
            plt.plot(time, np.array(tcp_positions)[:, i], label=f"TCP {label} Position")
            plt.ylabel("Position")
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(time[:-1], velocity[:, i], label=f"TCP {label} Velocity")
            plt.ylabel("Velocity")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(time[:-2], acceleration[:, i], label=f"TCP {label} Acceleration")
            plt.ylabel("Acceleration")
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(time[:-3], jerk[:, i], label=f"TCP {label} Jerk")
            plt.xlabel("Time")
            plt.ylabel("Jerk")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_cartesian_trajectory(self, trajectory_data, Tf, title="Cartesian Trajectory"):
        """
        Plots the Cartesian trajectory of a robot's motion, including position, velocity, and acceleration.
        
        Args:
            trajectory_data (dict): A dictionary containing the position, velocity, and acceleration data for the Cartesian trajectory.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot. Defaults to "Cartesian Trajectory".
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex="col")
        fig.suptitle(title)

        axs[0].plot(time_steps, positions[:, 0], label="X Position")
        axs[0].plot(time_steps, positions[:, 1], label="Y Position")
        axs[0].plot(time_steps, positions[:, 2], label="Z Position")
        axs[0].set_ylabel("Position")
        axs[0].legend()

        axs[1].plot(time_steps, velocities[:, 0], label="X Velocity")
        axs[1].plot(time_steps, velocities[:, 1], label="Y Velocity")
        axs[1].plot(time_steps, velocities[:, 2], label="Z Velocity")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()

        axs[2].plot(time_steps, accelerations[:, 0], label="X Acceleration")
        axs[2].plot(time_steps, accelerations[:, 1], label="Y Acceleration")
        axs[2].plot(time_steps, accelerations[:, 2], label="Z Acceleration")
        axs[2].set_ylabel("Acceleration")
        axs[2].legend()

        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def calculate_derivatives(self, positions, dt):
        """
        Calculate the velocity, acceleration, and jerk of a trajectory.

        Parameters:
            positions (list or numpy.ndarray): A list or array of positions.
            dt (float): The time step between each position.

        Returns:
            velocity (numpy.ndarray): An array of velocities.
            acceleration (numpy.ndarray): An array of accelerations.
            jerk (numpy.ndarray): An array of jerks.
        """
        positions = np.array(positions)
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        return velocity, acceleration, jerk

    def plot_ee_trajectory(self, trajectory_data, Tf, title="End-Effector Trajectory"):
        """
        Plots the end-effector trajectory of a serial manipulator.
        
        Args:
            trajectory_data (dict): A dictionary containing the position and orientation data of the end-effector trajectory.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot. Defaults to "End-Effector Trajectory".
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        if "orientations" in trajectory_data:
            orientations = trajectory_data["orientations"]
        else:
            orientations = np.array(
                [
                    self.serial_manipulator.forward_kinematics(pos)[:3, :3]
                    for pos in positions
                ]
            )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.suptitle(title)

        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2], label="EE Position", color="b"
        )

        for i in range(0, num_steps, max(1, num_steps // 20)):
            R = orientations[i]
            pos = positions[i]
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 0], R[1, 0], R[2, 0], length=0.01, color="r"
            )
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 1], R[1, 1], R[2, 1], length=0.01, color="g"
            )
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 2], R[1, 2], R[2, 2], length=0.01, color="b"
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.legend()
        plt.show()
