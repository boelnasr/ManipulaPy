import numpy as np
import matplotlib.pyplot as plt
from utils import MatrixLog6, TransInv, MatrixExp6
class TrajectoryPlanning:
    def __init__(self, serial_manipulator, dynamics):
        self.serial_manipulator = serial_manipulator
        self.dynamics = dynamics

    def cubic_time_scaling(self, Tf, t):
        return 3 * (t / Tf)**2 - 2 * (t / Tf)**3

    def quintic_time_scaling(self, Tf, t):
        return 10 * (t / Tf)**3 - 15 * (t / Tf)**4 + 6 * (t / Tf)**5

    def _select_time_scaling(self, method, Tf, t):
        if method == 'linear':
            return t / Tf
        elif method == 'cubic':
            return self.cubic_time_scaling(Tf, t)
        elif method == 'quintic':
            return self.quintic_time_scaling(Tf, t)
        else:
            raise ValueError("Invalid method specified. Choose 'linear', 'cubic', or 'quintic'.")

    def ScrewTrajectory(self,Xstart, Xend, Tf, N, method):
        """
        Generate a screw trajectory from a start point to an end point over a specified time.

        Parameters:
            Xstart (array-like): The start point of the trajectory.
            Xend (array-like): The end point of the trajectory.
            Tf (float): The total time duration of the trajectory.
            N (int): The number of points to generate along the trajectory.
            method (str): The method to use for time scaling.

        Returns:
            trajectory (ndarray): The generated screw trajectory as a 2D array.
        """
        time_stamps = np.linspace(0, Tf, N)
        trajectory = np.zeros((N, len(Xstart)))

        for i, t in enumerate(time_stamps):
            s = self._select_time_scaling(method, Tf, t)
            trajectory[i] = np.dot(Xstart, MatrixExp6(MatrixLog6(np.dot(TransInv(Xstart), Xend)) * s))

        return trajectory

    def plot_trajectory(self, trajectory):
        """
        Plots the trajectory of a robot arm based on the given trajectory data.

        Parameters:
            trajectory (dict): A dictionary containing the trajectory data.
                - time_stamps (list): A list of time stamps for each data point.
                - positions (ndarray): An array of joint positions over time.
                - velocities (ndarray): An array of joint velocities over time.
                - accelerations (ndarray): An array of joint accelerations over time.
                - torques (ndarray): An array of joint torques over time.

        Returns:
            None
        """
        time_stamps = trajectory["time_stamps"]
        plt.figure(figsize=(12, 8))

        # Plotting joint angles
        plt.subplot(4, 1, 1)
        for i in range(trajectory["positions"].shape[1]):
            plt.plot(time_stamps, trajectory["positions"][:, i], label=f'Joint {i+1} Position')
        plt.ylabel('Position')
        plt.legend()

        # Plotting joint velocities
        plt.subplot(4, 1, 2)
        for i in range(trajectory["velocities"].shape[1]):
            plt.plot(time_stamps, trajectory["velocities"][:, i], label=f'Joint {i+1} Velocity')
        plt.ylabel('Velocity')
        plt.legend()

        # Plotting joint accelerations
        plt.subplot(4, 1, 3)
        for i in range(trajectory["accelerations"].shape[1]):
            plt.plot(time_stamps, trajectory["accelerations"][:, i], label=f'Joint {i+1} Acceleration')
        plt.ylabel('Acceleration')
        plt.legend()

        # Plotting joint torques
        plt.subplot(4, 1, 4)
        for i in range(trajectory["torques"].shape[1]):
            plt.plot(time_stamps, trajectory["torques"][:, i], label=f'Joint {i+1} Torque')
        plt.xlabel('Time')
        plt.ylabel('Torque')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def check_singularity(self, trajectory, near_singularity_threshold=0.1):
        """
        Check for singularity and near-singularity in the trajectory.

        Parameters:
            trajectory (list): A list of joint angles representing the trajectory.
            near_singularity_threshold (float): Threshold for near-singularity detection.

        Returns:
            dict: A dictionary with keys 'singular_points' and 'near_singular_points',
                each containing lists of points in the trajectory that are singular
                or near-singular, respectively.
        """
        singular_points = []
        near_singular_points = []

        for i, point in enumerate(trajectory):
            if self.singularity.singularity_analysis(point):
                singular_points.append((i, point))
            elif self.singularity.near_singularity_detection(point, near_singularity_threshold):
                near_singular_points.append((i, point))

        return {
            "singular_points": singular_points,
            "near_singular_points": near_singular_points
        }

    def optimize_trajectory(self, trajectory, g, Ftipmat, Mlist, Glist, Slist, dt, intRes):
        """
        Optimize a trajectory using forward dynamics and Euler integration, including jerk calculation.

        Parameters:
            trajectory (dict): A dictionary containing joint positions, velocities, and accelerations.
            g (np.ndarray): Gravity vector.
            Ftipmat (np.ndarray): An N x 6 matrix of spatial forces applied by the end-effector.
            Mlist (list): List of link frames {i} relative to {i-1} at the home position.
            Glist (list): Spatial inertia matrices Gi of the links.
            Slist (list): Screw axes Si of the joints in a space frame.
            dt (float): The time step for the Euler integration.
            intRes (int): The number of integration steps per time step.

        Returns:
            dict: Optimized trajectory with updated positions, velocities, accelerations, and jerks.
        """
        N = len(trajectory["time_stamps"])
        optimized_positions = np.zeros_like(trajectory["positions"])
        optimized_velocities = np.zeros_like(trajectory["velocities"])
        optimized_accelerations = np.zeros_like(trajectory["accelerations"])
        optimized_jerks = np.zeros_like(trajectory["jerks"])

        for i in range(N):
            thetalist = trajectory["positions"][i]
            dthetalist = trajectory["velocities"][i]
            ddthetalist = trajectory["accelerations"][i] if i < N - 1 else np.zeros_like(dthetalist)

            for _ in range(intRes):
                # Compute forward dynamics
                ddthetalist_new = self.dynamics.forward_dynamics(thetalist, dthetalist, g, Ftipmat[i], Mlist, Glist, Slist)
                # Euler integration to update thetalist and dthetalist
                thetalist, dthetalist = self.euler_step(thetalist, dthetalist, ddthetalist_new, dt / intRes)

            optimized_positions[i] = thetalist
            optimized_velocities[i] = dthetalist
            optimized_accelerations[i] = ddthetalist

            if i > 0:
                optimized_jerks[i] = (optimized_accelerations[i] - optimized_accelerations[i - 1]) / dt

        return {
            "positions": optimized_positions,
            "velocities": optimized_velocities,
            "accelerations": optimized_accelerations,
            "jerks": optimized_jerks,
            "time_stamps": trajectory["time_stamps"]
            }

    def euler_step(self, thetalist, dthetalist, ddthetalist, dt):
        """
        Perform an Euler step to update the state of the system.
        
        This method takes in the current state of the system represented by thetalist,
        dthetalist, and ddthetalist, and calculates the updated state after a time
        interval of dt using the Euler method.
        
        Parameters:
            thetalist (list): The current joint angles of the system.
            dthetalist (list): The current joint velocities of the system.
            ddthetalist (list): The current joint accelerations of the system.
            dt (float): The time interval for the Euler step.
        
        Returns:
            tuple: A tuple containing the updated joint angles and velocities after
                    the Euler step.
        """

        new_thetalist = thetalist + dt * dthetalist
        new_dthetalist = dthetalist + dt * ddthetalist
        return new_thetalist, new_dthetalist

    def interpolate_trajectory(self, trajectory, interpolation_factor):
        """
        Interpolate a trajectory to increase the number of points.

        Parameters:
            trajectory (list): A list of joint angles representing the trajectory.
            interpolation_factor (int): Factor by which to increase the number of points.

        Returns:
            list: The interpolated trajectory.
        """
        interpolated_trajectory = []
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]
            step = 1.0 / interpolation_factor
            for s in np.arange(0, 1, step):
                interpolated_point = (1 - s) * np.array(start) + s * np.array(end)
                interpolated_trajectory.append(interpolated_point.tolist())
        interpolated_trajectory.append(trajectory[-1])
        return interpolated_trajectory

    def smooth_trajectory(self, trajectory, smoothing_factor):
        """
        Apply a simple smoothing filter to the trajectory.

        Parameters:
            trajectory (list): A list of joint angles representing the trajectory.
            smoothing_factor (float): Smoothing factor for the filter.

        Returns:
            list: The smoothed trajectory.
        """
        smoothed_trajectory = np.copy(trajectory)
        for i in range(1, len(trajectory) - 1):
            smoothed_trajectory[i] = smoothing_factor * trajectory[i] + \
                                     (1 - smoothing_factor) * 0.5 * (trajectory[i - 1] + trajectory[i + 1])
        return smoothed_trajectory.tolist()

    def plot_tcp_trajectory(self, trajectory, dt):
        """
        Generate a plot of the TCP trajectory.

        Parameters:
            trajectory (list): A list of joint angles.
            dt (float): The time step between each joint angle.

        Returns:
            None
        """
        
        # Convert joint angles to TCP poses using forward kinematics
        tcp_trajectory = [self.serial_manipulator.forward_kinematics(joint_angles) for joint_angles in trajectory]
        # Extract TCP positions from poses
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