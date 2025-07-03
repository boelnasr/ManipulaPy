#!/usr/bin/env python3

from numba import cuda, float32
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from ManipulaPy.utils import (
    TransToRp,
    MatrixLog3,
    MatrixExp3,
    CubicTimeScaling,
    QuinticTimeScaling,
)
from urchin.urdf import URDF
from ManipulaPy.cuda_kernels import trajectory_kernel, inverse_dynamics_kernel, forward_dynamics_kernel, cartesian_trajectory_kernel
from ManipulaPy.potential_field import CollisionChecker, PotentialField
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
        
        # Initialize collision checker and potential field with error handling
        try:
            self.collision_checker = CollisionChecker(urdf_path)
        except Exception as e:
            logger.warning(f"Failed to initialize collision checker: {e}")
            self.collision_checker = None
        
        try:
            self.potential_field = PotentialField()
        except Exception as e:
            logger.warning(f"Failed to initialize potential field: {e}")
            self.potential_field = None

        # Check if CUDA is available
        try:
            cuda.current_context()
            self.cuda_available = True
            logger.info("CUDA is available for trajectory planning")
        except Exception:
            self.cuda_available = False
            logger.info("CUDA not available, will use NumPy fallback for trajectory planning")

    def _fallback_joint_trajectory(self, thetastart, thetaend, Tf, N, method):
        """
        Pure NumPy implementation of joint trajectory generation as fallback.
        """
        logger.info("Using NumPy fallback for joint trajectory generation")
        
        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        traj_pos = np.zeros((N, num_joints), dtype=np.float32)
        traj_vel = np.zeros((N, num_joints), dtype=np.float32)
        traj_acc = np.zeros((N, num_joints), dtype=np.float32)

        for i in range(N):
            t = i * (Tf / (N - 1))
            if method == 3:  # Cubic time scaling
                s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
                s_dot = 6 * (t / Tf) * (1 - t / Tf) / Tf
                s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
            elif method == 5:  # Quintic time scaling
                s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
                s_dot = 30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + (t / Tf) ** 2) / Tf
                s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
            else:
                s = t / Tf  # Linear
                s_dot = 1 / Tf
                s_ddot = 0

            for j in range(num_joints):
                traj_pos[i, j] = s * (thetaend[j] - thetastart[j]) + thetastart[j]
                traj_vel[i, j] = s_dot * (thetaend[j] - thetastart[j])
                traj_acc[i, j] = s_ddot * (thetaend[j] - thetastart[j])

        # Apply joint limits
        for i in range(num_joints):
            traj_pos[:, i] = np.clip(
                traj_pos[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1]
            )

        # Apply potential field for collision avoidance if available
        if self.collision_checker and self.potential_field:
            q_goal = thetaend
            obstacles = []  # Define obstacles here as needed

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

        # Try CUDA implementation first if available
        if self.cuda_available:
            try:
                return self._cuda_joint_trajectory(thetastart, thetaend, Tf, N, method)
            except Exception as e:
                logger.warning(f"CUDA trajectory generation failed: {e}. Falling back to NumPy.")
                return self._fallback_joint_trajectory(thetastart, thetaend, Tf, N, method)
        else:
            return self._fallback_joint_trajectory(thetastart, thetaend, Tf, N, method)

    def _cuda_joint_trajectory(self, thetastart, thetaend, Tf, N, method):
        """
        CUDA implementation of joint trajectory generation.
        """
        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        traj_pos = np.zeros((N, num_joints), dtype=np.float32)
        traj_vel = np.zeros((N, num_joints), dtype=np.float32)
        traj_acc = np.zeros((N, num_joints), dtype=np.float32)

        threads_per_block = 256
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
            
            # Check if CUDA computation actually worked
            if not np.any(traj_pos):
                raise RuntimeError("CUDA kernel returned all zeros")
                
        finally:
            # Explicitly free GPU memory using del statement
            del d_thetastart
            del d_thetaend
            del d_traj_pos
            del d_traj_vel
            del d_traj_acc

        # Apply joint limits
        for i in range(num_joints):
            traj_pos[:, i] = np.clip(
                traj_pos[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1]
            )

        # Apply potential field for collision avoidance if available
        if self.collision_checker and self.potential_field:
            q_goal = thetaend
            obstacles = []

            for idx, step in enumerate(traj_pos):
                if self.collision_checker.check_collision(step):
                    for _ in range(100):
                        gradient = self.potential_field.compute_gradient(step, q_goal, obstacles)
                        step -= 0.01 * gradient
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
        Compute joint torques with enforced torque limits based on a trajectory using CUDA acceleration or NumPy fallback.

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

        # Try CUDA implementation first if available
        if self.cuda_available:
            try:
                return self._cuda_inverse_dynamics_trajectory(
                    thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                    gravity_vector, Ftip
                )
            except Exception as e:
                logger.warning(f"CUDA inverse dynamics failed: {e}. Using fallback.")
        
        # Fallback to NumPy implementation
        return self._fallback_inverse_dynamics_trajectory(
            thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
            gravity_vector, Ftip
        )

    def _cuda_inverse_dynamics_trajectory(self, thetalist_trajectory, dthetalist_trajectory, 
                                        ddthetalist_trajectory, gravity_vector, Ftip):
        """CUDA implementation of inverse dynamics trajectory."""
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

    def _fallback_inverse_dynamics_trajectory(self, thetalist_trajectory, dthetalist_trajectory, 
                                            ddthetalist_trajectory, gravity_vector, Ftip):
        """NumPy fallback implementation of inverse dynamics trajectory."""
        logger.info("Using NumPy fallback for inverse dynamics trajectory")
        
        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]
        torques_trajectory = np.zeros((num_points, num_joints), dtype=np.float32)

        for i in range(num_points):
            try:
                torques = self.dynamics.inverse_dynamics(
                    thetalist_trajectory[i],
                    dthetalist_trajectory[i],
                    ddthetalist_trajectory[i],
                    gravity_vector,
                    Ftip
                )
                torques_trajectory[i] = torques
            except Exception as e:
                logger.warning(f"Inverse dynamics failed at point {i}: {e}")
                # Use zeros as fallback
                torques_trajectory[i] = np.zeros(num_joints)

        # Apply torque limits
        if self.torque_limits is not None:
            torques_trajectory = np.clip(
                torques_trajectory, self.torque_limits[:, 0], self.torque_limits[:, 1])

        return torques_trajectory

    def forward_dynamics_trajectory(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """
        Calculates the forward dynamics trajectory of a robotic system.

        Args:
            thetalist (np.ndarray): Array of joint angles over the trajectory.
            dthetalist (np.ndarray): Array of joint velocities over the trajectory.
            taumat (np.ndarray): Array of joint torques over the trajectory.
            g (np.ndarray): Gravity vector affecting the system.
            Ftipmat (np.ndarray): Array of external forces applied at the end effector.
            dt (float): Time step for the trajectory.
            intRes (int): Number of integration steps per time step.

        Returns:
            dict: Dictionary containing the joint positions, velocities, and accelerations.
        """
        # Try CUDA implementation first if available
        if self.cuda_available:
            try:
                return self._cuda_forward_dynamics_trajectory(
                    thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
                )
            except Exception as e:
                logger.warning(f"CUDA forward dynamics failed: {e}. Using fallback.")
        
        # Fallback to NumPy implementation
        return self._fallback_forward_dynamics_trajectory(
            thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
        )

    def _cuda_forward_dynamics_trajectory(self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """CUDA implementation of forward dynamics trajectory."""
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

    def _fallback_forward_dynamics_trajectory(self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """NumPy fallback implementation of forward dynamics trajectory."""
        logger.info("Using NumPy fallback for forward dynamics trajectory")
        
        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]
        thetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        dthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        ddthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        
        # Initialize with starting conditions
        current_theta = np.array(thetalist, dtype=np.float32)
        current_dtheta = np.array(dthetalist, dtype=np.float32)
        
        thetamat[0, :] = current_theta
        dthetamat[0, :] = current_dtheta
        
        for i in range(1, num_steps):
            try:
                # Compute forward dynamics for current state
                ddtheta = self.dynamics.forward_dynamics(
                    current_theta, current_dtheta, taumat[i], g, Ftipmat[i]
                )
                
                # Integrate using simple Euler method
                for _ in range(intRes):
                    current_dtheta += ddtheta * (dt / intRes)
                    current_theta += current_dtheta * (dt / intRes)
                    
                    # Apply joint limits
                    for j in range(num_joints):
                        current_theta[j] = np.clip(
                            current_theta[j], 
                            self.joint_limits[j, 0], 
                            self.joint_limits[j, 1]
                        )
                
                thetamat[i, :] = current_theta
                dthetamat[i, :] = current_dtheta
                ddthetamat[i, :] = ddtheta
                
            except Exception as e:
                logger.warning(f"Forward dynamics failed at step {i}: {e}")
                # Use previous values as fallback
                thetamat[i, :] = thetamat[i-1, :]
                dthetamat[i, :] = dthetamat[i-1, :]
                ddthetamat[i, :] = np.zeros(num_joints)
        
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
        # Try CUDA implementation first if available
        if self.cuda_available:
            try:
                return self._cuda_cartesian_trajectory(Xstart, Xend, Tf, N, method)
            except Exception as e:
                logger.warning(f"CUDA Cartesian trajectory failed: {e}. Using fallback.")
        
        # Fallback to NumPy implementation
        return self._fallback_cartesian_trajectory(Xstart, Xend, Tf, N, method)

    def _cuda_cartesian_trajectory(self, Xstart, Xend, Tf, N, method):
        """CUDA implementation of Cartesian trajectory generation."""
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

    def _fallback_cartesian_trajectory(self, Xstart, Xend, Tf, N, method):
        """NumPy fallback implementation of Cartesian trajectory generation."""
        logger.info("Using NumPy fallback for Cartesian trajectory generation")
        
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [None] * N
        Rstart, pstart = TransToRp(Xstart)
        Rend, pend = TransToRp(Xend)

        orientations = np.zeros((N, 3, 3), dtype=np.float32)
        traj_pos = np.zeros((N, 3), dtype=np.float32)
        traj_vel = np.zeros((N, 3), dtype=np.float32)
        traj_acc = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            t = timegap * i
            if method == 3:
                s = CubicTimeScaling(Tf, t)
                s_dot = 6 * (t / Tf) * (1 - t / Tf) / Tf
                s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
            else:
                s = QuinticTimeScaling(Tf, t)
                s_dot = 30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + (t / Tf) ** 2) / Tf
                s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
            
            # Position interpolation
            traj_pos[i] = s * pend + (1 - s) * pstart
            traj_vel[i] = s_dot * (pend - pstart)
            traj_acc[i] = s_ddot * (pend - pstart)
            
            # Orientation interpolation
            traj[i] = np.r_[
                np.c_[
                    np.dot(Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)),
                    traj_pos[i],
                ],
                [[0, 0, 0, 1]],
            ]
            orientations[i] = np.dot(
                Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)
            )

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

    def plan_trajectory(self, start_position, target_position, obstacle_points):
        """
        Plans a collision-free trajectory from start to target position.

        Args:
            start_position (list): Initial joint configuration.
            target_position (list): Desired joint configuration.
            obstacle_points (list): List of obstacle points in the environment.

        Returns:
            list: Joint trajectory as a list of joint configurations.
        """
        # Perform trajectory planning (e.g., RRT or interpolation)
        joint_trajectory = [start_position, target_position]  # Example
        logger.info(f"Planned trajectory with {len(joint_trajectory)} waypoints.")
        return joint_trajectory