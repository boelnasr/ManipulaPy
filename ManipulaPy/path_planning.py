#!/usr/bin/env python3

from numba import cuda, float32
import numpy as np
import matplotlib.pyplot as plt
from .utils import TransToRp, MatrixLog3, MatrixExp3, CubicTimeScaling, QuinticTimeScaling

@cuda.jit
def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
    """
    CUDA kernel to compute positions, velocities, and accelerations using cubic or quintic time scaling.
    """
    idx = cuda.grid(1)
    if idx < N:
        t = idx * (Tf / (N - 1))
        if method == 3:  # Cubic time scaling
            s = 3 * (t / Tf)**2 - 2 * (t / Tf)**3
            s_dot = 6 * (t / Tf) * (1 - t / Tf)
            s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
        elif method == 5:  # Quintic time scaling
            s = 10 * (t / Tf)**3 - 15 * (t / Tf)**4 + 6 * (t / Tf)**5
            s_dot = 30 * (t / Tf)**2 * (1 - 2 * (t / Tf) + t / Tf**2)
            s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
        elif method == 6:  # Bezier curve
            s = (1 - t / Tf)**3
            s_dot = -3 * (1 - t / Tf)**2 / Tf
            s_ddot = 6 * (1 - t / Tf) / Tf**2
        else:
            s = s_dot = s_ddot = 0

        for j in range(thetastart.shape[0]):
            traj_pos[idx, j] = s * (thetaend[j] - thetastart[j]) + thetastart[j]
            traj_vel[idx, j] = s_dot * (thetaend[j] - thetastart[j])
            traj_acc[idx, j] = s_ddot * (thetaend[j] - thetastart[j])

@cuda.jit
def inverse_dynamics_kernel(thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory, gravity_vector, Ftip, Glist, Slist, M, torques_trajectory, torque_limits):
    # Same as before, but add torque limits enforcement
    idx = cuda.grid(1)
    if idx < thetalist_trajectory.shape[0]:
        thetalist = thetalist_trajectory[idx]
        dthetalist = dthetalist_trajectory[idx]
        ddthetalist = ddthetalist_trajectory[idx]

        # Mass matrix computation
        M_temp = cuda.local.array((6, 6), dtype=float32)
        for i in range(len(thetalist)):
            for row in range(6):
                for col in range(6):
                    M_temp[row, col] += Glist[i, row, col]  # Simplified for demonstration, replace with full computation

        # Velocity quadratic forces computation
        c_temp = cuda.local.array(6, dtype=float32)
        for i in range(len(thetalist)):
            for j in range(6):
                c_temp[j] += Slist[i, j] * dthetalist[i]  # Simplified for demonstration, replace with full computation

        # Gravity forces computation
        g_temp = cuda.local.array(6, dtype=float32)
        for i in range(len(thetalist)):
            g_temp[2] += gravity_vector[i]  # Simplified for demonstration, replace with full computation

        # External forces (Ftip) are considered in the dynamics
        F_ext = cuda.local.array(6, dtype=float32)
        for i in range(len(Ftip)):
            F_ext[i] += Ftip[i]

        # Torque computation
        tau_temp = cuda.local.array(6, dtype=float32)
        for row in range(6):
            for col in range(6):
                tau_temp[row] += M_temp[row, col] * ddthetalist[col]
            tau_temp[row] += c_temp[row] + g_temp[row] + F_ext[row]
        for j in range(len(tau_temp)):
            # Enforce torque limits
            tau_temp[j] = max(torque_limits[j, 0], min(tau_temp[j], torque_limits[j, 1]))
            torques_trajectory[idx, j] = tau_temp[j]

@cuda.jit
def forward_dynamics_kernel(thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes, Glist, Slist, M, thetamat, dthetamat, ddthetamat, joint_limits):
    # Same as before, but add joint limits enforcement
    idx = cuda.grid(1)
    if idx < taumat.shape[0]:
        # Initialize local variables
        current_thetalist = thetamat[idx - 1, :] if idx > 0 else thetalist
        current_dthetalist = dthetamat[idx - 1, :] if idx > 0 else dthetalist
        current_tau = taumat[idx, :]
        current_Ftip = Ftipmat[idx, :]

        # Placeholder for the mass matrix and other dynamics quantities
        M_temp = cuda.local.array((6, 6), dtype=float32)
        c_temp = cuda.local.array((6,), dtype=float32)
        g_temp = cuda.local.array((6,), dtype=float32)
        ddthetalist_local = cuda.local.array((6,), dtype=float32)

        for _ in range(intRes):
            # Compute forward dynamics (simplified for demonstration)
            for i in range(len(thetalist)):
                for row in range(6):
                    for col in range(6):
                        M_temp[row, col] = Glist[i, row, col]  # Replace with full computation
                    c_temp[row] = Slist[i, row] * current_dthetalist[i]  # Replace with full computation
                    g_temp[row] = g[row]  # Replace with full computation

            # Compute joint accelerations
            for i in range(len(thetalist)):
                ddthetalist_local[i] = (current_tau[i] - c_temp[i] - g_temp[i]) / M_temp[i, i]  # Simplified

            # Integrate to get velocities and positions
            for i in range(len(thetalist)):
                current_dthetalist[i] += ddthetalist_local[i] * (dt / intRes)
                current_thetalist[i] += current_dthetalist[i] * (dt / intRes)

            # Enforce joint limits
            for i in range(len(thetalist)):
                current_thetalist[i] = max(joint_limits[i, 0], min(current_thetalist[i], joint_limits[i, 1]))

        # Store results
        for i in range(len(thetalist)):
            thetamat[idx, i] = current_thetalist[i]
            dthetamat[idx, i] = current_dthetalist[i]
            ddthetamat[idx, i] = ddthetalist_local[i]

@cuda.jit
def cartesian_trajectory_kernel(pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method):
    idx = cuda.grid(1)
    if idx < N:
        t = idx * (Tf / (N - 1))
        if method == 3:
            s = 3 * (t / Tf)**2 - 2 * (t / Tf)**3
            s_dot = 6 * (t / Tf) * (1 - t / Tf)
            s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
        elif method == 5:
            s = 10 * (t / Tf)**3 - 15 * (t / Tf)**4 + 6 * (t / Tf)**5
            s_dot = 30 * (t / Tf)**2 * (1 - 2 * (t / Tf) + t / Tf**2)
            s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
        else:
            s = s_dot = s_ddot = 0

        for j in range(3):  # For x, y, z positions
            traj_pos[idx, j] = s * (pend[j] - pstart[j]) + pstart[j]
            traj_vel[idx, j] = s_dot * (pend[j] - pstart[j])
            traj_acc[idx, j] = s_ddot * (pend[j] - pstart[j])

class TrajectoryPlanning:
    def __init__(self, serial_manipulator, dynamics, joint_limits, torque_limits=None):
        """
        Initialize TrajectoryPlanning with manipulator, dynamics, and limit constraints.
        
        Args:
            serial_manipulator (SerialManipulator): Kinematics module.
            dynamics (ManipulatorDynamics): Dynamics module.
            joint_limits (list of tuples): List where each tuple is (min, max) for each joint.
            torque_limits (list of tuples, optional): List where each tuple is (min, max) for each joint.
        """
        self.serial_manipulator = serial_manipulator
        self.dynamics = dynamics
        self.joint_limits = np.array(joint_limits)
        self.torque_limits = np.array(torque_limits) if torque_limits else np.array([[-np.inf, np.inf]] * len(joint_limits))

    def JointTrajectory(self, thetastart, thetaend, Tf, N, method):
        """
        Generates joint trajectories with positions, velocities, and accelerations using CUDA acceleration.

        Args:
            thetastart (list or np.ndarray): Starting joint angles.
            thetaend (list or np.ndarray): Ending joint angles.
            Tf (float): Total duration of the trajectory.
            N (int): Number of points in the trajectory.
            method (int): Time-scaling method (3 for cubic, 5 for quintic, or 6 for Bezier curve).

        Returns:
            dict: Dictionary containing 'positions', 'velocities', and 'accelerations'.
        """
        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        # Initialize arrays to hold position, velocity, and acceleration trajectories
        traj_pos = np.zeros((N, num_joints), dtype=np.float32)
        traj_vel = np.zeros((N, num_joints), dtype=np.float32)
        traj_acc = np.zeros((N, num_joints), dtype=np.float32)

        # Set up GPU memory and kernel execution
        threads_per_block = 1024
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)  # Ensure at least one block
        d_thetastart = cuda.to_device(thetastart)
        d_thetaend = cuda.to_device(thetaend)
        d_traj_pos = cuda.device_array_like(traj_pos)
        d_traj_vel = cuda.device_array_like(traj_vel)
        d_traj_acc = cuda.device_array_like(traj_acc)

        # Execute CUDA kernel
        trajectory_kernel[blocks_per_grid, threads_per_block](d_thetastart, d_thetaend, d_traj_pos, d_traj_vel, d_traj_acc, Tf, N, method)

        # Copy computed results back to host memory
        d_traj_pos.copy_to_host(traj_pos)
        d_traj_vel.copy_to_host(traj_vel)
        d_traj_acc.copy_to_host(traj_acc)

        # Enforce joint limits
        for i in range(num_joints):
            traj_pos[:, i] = np.clip(traj_pos[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1])

        return {'positions': traj_pos, 'velocities': traj_vel, 'accelerations': traj_acc}

    def InverseDynamicsTrajectory(self, thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory, gravity_vector=None, Ftip=None):
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

        # Set up GPU memory and kernel execution
        threads_per_block = 1024
        blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
        blocks_per_grid = max(blocks_per_grid, 1)  # Ensure at least one block

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

        # Execute CUDA kernel
        inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
            d_thetalist_trajectory, d_dthetalist_trajectory, d_ddthetalist_trajectory,
            d_gravity_vector, d_Ftip, d_Glist, d_Slist, d_M, d_torques_trajectory, d_torque_limits)

        # Copy computed results back to host memory
        d_torques_trajectory.copy_to_host(torques_trajectory)

        # Enforce torque limits if they exist
        if self.torque_limits is not None:
            torques_trajectory = np.clip(torques_trajectory, self.torque_limits[:, 0], self.torque_limits[:, 1])

        return torques_trajectory

    def forward_dynamics_trajectory(self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
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
        blocks_per_grid = max(blocks_per_grid, 1)  # Ensure at least one block
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
            d_thetalist, d_dthetalist, d_taumat, d_g, d_Ftipmat, dt, intRes, d_Glist, d_Slist, d_M, d_thetamat, d_dthetamat, d_ddthetamat, d_joint_limits)
        d_thetamat.copy_to_host(thetamat)
        d_dthetamat.copy_to_host(dthetamat)
        d_ddthetamat.copy_to_host(ddthetamat)
        return {'positions': thetamat, 'velocities': dthetamat, 'accelerations': ddthetamat}

    def CartesianTrajectory(self, Xstart, Xend, Tf, N, method):
        """
        Calculates a Cartesian trajectory between two given end-effector poses.

        Parameters:
            Xstart (numpy.ndarray): The starting end-effector pose.
            Xend (numpy.ndarray): The ending end-effector pose.
            Tf (float): The total time of the trajectory.
            N (int): The number of waypoints in the trajectory.
            method (int): The method to use for trajectory generation (3 for cubic, otherwise quintic).

        Returns:
            dict: A dictionary containing the trajectory information:
                - 'positions' (numpy.ndarray): The Cartesian positions of the end-effector along the trajectory.
                - 'velocities' (numpy.ndarray): The Cartesian velocities of the end-effector along the trajectory.
                - 'accelerations' (numpy.ndarray): The Cartesian accelerations of the end-effector along the trajectory.
                - 'orientations' (numpy.ndarray): The orientations of the end-effector along the trajectory.
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [None] * N
        Rstart, pstart = TransToRp(Xstart)
        Rend, pend = TransToRp(Xend)
        
        orientations = np.zeros((N, 3, 3), dtype=np.float32)  # To store orientations

        for i in range(N):
            if method == 3:
                s = CubicTimeScaling(Tf, timegap * i)
            else:
                s = QuinticTimeScaling(Tf, timegap * i)
            traj[i] = np.r_[
                np.c_[np.dot(Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)),
                    s * pend + (1 - s) * pstart],
                [[0, 0, 0, 1]]
            ]
            orientations[i] = np.dot(Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s))

        traj_pos = np.array([TransToRp(T)[1] for T in traj], dtype=np.float32)

        # Ensure the arrays are contiguous
        pstart = np.ascontiguousarray(pstart)
        pend = np.ascontiguousarray(pend)
        traj_pos = np.ascontiguousarray(traj_pos)

        # Use CUDA to compute velocities and accelerations
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

        cartesian_trajectory_kernel[blocks_per_grid, threads_per_block](d_pstart, d_pend, d_traj_pos, d_traj_vel, d_traj_acc, Tf, N, method)

        d_traj_pos.copy_to_host(traj_pos)
        d_traj_vel.copy_to_host(traj_vel)
        d_traj_acc.copy_to_host(traj_acc)

        return {'positions': traj_pos, 'velocities': traj_vel, 'accelerations': traj_acc, 'orientations': orientations}

    @staticmethod
    def plot_trajectory(trajectory_data, Tf, title='Joint Trajectory', labels=None):
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']

        num_steps = positions.shape[0]
        num_joints = positions.shape[1]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, num_joints, figsize=(15, 10), sharex='col')
        fig.suptitle(title)

        for i in range(num_joints):
            if labels and len(labels) == num_joints:
                label = labels[i]
            else:
                label = f'Joint {i+1}'

            axs[0, i].plot(time_steps, positions[:, i], label=f'{label} Position')
            axs[0, i].set_ylabel('Position')
            axs[0, i].legend()

            axs[1, i].plot(time_steps, velocities[:, i], label=f'{label} Velocity')
            axs[1, i].set_ylabel('Velocity')
            axs[1, i].legend()

            axs[2, i].plot(time_steps, accelerations[:, i], label=f'{label} Acceleration')
            axs[2, i].set_ylabel('Acceleration')
            axs[2, i].legend()

        for ax in axs[-1]:
            ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def plot_tcp_trajectory(self, trajectory, dt):
        tcp_trajectory = [self.serial_manipulator.forward_kinematics(joint_angles) for joint_angles in trajectory]
        tcp_positions = [pose[:3, 3] for pose in tcp_trajectory]

        velocity, acceleration, jerk = self.calculate_derivatives(tcp_positions, dt)
        time = np.arange(0, len(tcp_positions) * dt, dt)

        plt.figure(figsize=(12, 8))
        for i, label in enumerate(['X', 'Y', 'Z']):
            plt.subplot(4, 1, 1)
            plt.plot(time, np.array(tcp_positions)[:, i], label=f'TCP {label} Position')
            plt.ylabel('Position')
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(time[:-1], velocity[:, i], label=f'TCP {label} Velocity')
            plt.ylabel('Velocity')
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(time[:-2], acceleration[:, i], label=f'TCP {label} Acceleration')
            plt.ylabel('Acceleration')
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(time[:-3], jerk[:, i], label=f'TCP {label} Jerk')
            plt.xlabel('Time')
            plt.ylabel('Jerk')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_cartesian_trajectory(self, trajectory_data, Tf, title='Cartesian Trajectory'):
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']

        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex='col')
        fig.suptitle(title)

        axs[0].plot(time_steps, positions[:, 0], label='X Position')
        axs[0].plot(time_steps, positions[:, 1], label='Y Position')
        axs[0].plot(time_steps, positions[:, 2], label='Z Position')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        axs[1].plot(time_steps, velocities[:, 0], label='X Velocity')
        axs[1].plot(time_steps, velocities[:, 1], label='Y Velocity')
        axs[1].plot(time_steps, velocities[:, 2], label='Z Velocity')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()

        axs[2].plot(time_steps, accelerations[:, 0], label='X Acceleration')
        axs[2].plot(time_steps, accelerations[:, 1], label='Y Acceleration')
        axs[2].plot(time_steps, accelerations[:, 2], label='Z Acceleration')
        axs[2].set_ylabel('Acceleration')
        axs[2].legend()

        axs[2].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def calculate_derivatives(self, positions, dt):
        positions = np.array(positions)
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        return velocity, acceleration, jerk
    
    def plot_ee_trajectory(self, trajectory_data, Tf, title='End-Effector Trajectory'):
        """
        Plot the end-effector trajectory in 3D space for positions and orientations.

        Args:
            trajectory_data (dict): Dictionary containing 'positions', 'velocities', 'accelerations', and 'orientations'.
            Tf (float): Total duration of the trajectory in seconds.
            title (str, optional): Title for the plot. Defaults to 'End-Effector Trajectory'.

        Returns:
            None
        """
        positions = trajectory_data['positions']
        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        # If orientations are provided, use them; otherwise, compute them
        if 'orientations' in trajectory_data:
            orientations = trajectory_data['orientations']
        else:
            # Compute orientations using forward kinematics
            orientations = np.array([self.serial_manipulator.forward_kinematics(pos)[:3, :3] for pos in positions])

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title)

        # Plot the EE position as a 3D spline
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='EE Position', color='b')

        # Plot orientation as quivers
        for i in range(0, num_steps, max(1, num_steps // 20)):  # Plot a few orientations along the trajectory
            R = orientations[i]
            pos = positions[i]
            ax.quiver(pos[0], pos[1], pos[2], R[0, 0], R[1, 0], R[2, 0], length=1, color='r')  # X direction
            ax.quiver(pos[0], pos[1], pos[2], R[0, 1], R[1, 1], R[2, 1], length=1, color='g')  # Y direction
            ax.quiver(pos[0], pos[1], pos[2], R[0, 2], R[1, 2], R[2, 2], length=1, color='b')  # Z direction

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        plt.show()

