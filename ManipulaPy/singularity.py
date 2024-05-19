#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numpy import linalg as la

class Singularity:
    def __init__(self, serial_manipulator):
        """
        Initialize the Singularity class with a SerialManipulator object.

        Parameters:
            serial_manipulator (SerialManipulator): An instance of SerialManipulator.
        """
        self.serial_manipulator = serial_manipulator

    def singularity_analysis(self, thetalist):
        """
        Analyze if the manipulator is at a singularity based on the determinant of the Jacobian matrix.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.

        Returns:
            bool: True if the manipulator is at a singularity, False otherwise.
        """
        J = self.serial_manipulator.jacobian(thetalist, frame='space')
        det_J = np.linalg.det(J)
        return abs(det_J) < 1e-4

    def manipulability_ellipsoid(self, thetalist, ax=None):
        """
        Plot the manipulability ellipsoid for a given set of joint angles.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Matplotlib 3D axis to plot on. Defaults to None.
        """
        J = self.serial_manipulator.jacobian(thetalist, frame='space')
        J_v = J[:3, :]  # Linear velocity part of the Jacobian
        J_w = J[3:, :]  # Angular velocity part of the Jacobian

        # Singular Value Decomposition (SVD) for both parts
        U_v, S_v, _ = np.linalg.svd(J_v)
        radii_v = 1.0 / np.sqrt(S_v)

        U_w, S_w, _ = np.linalg.svd(J_w)
        radii_w = 1.0 / np.sqrt(S_w)

        # Generate points on a unit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        points = np.array([x.flatten(), y.flatten(), z.flatten()])

        # Transform points to ellipsoids
        ellipsoid_points_v = np.dot(U_v, np.diag(radii_v)).dot(points)
        ellipsoid_points_w = np.dot(U_w, np.diag(radii_w)).dot(points)

        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            ax1 = ax2 = ax

        # Plot the linear velocity ellipsoid
        ax1.plot_surface(ellipsoid_points_v[0].reshape(x.shape), ellipsoid_points_v[1].reshape(y.shape), ellipsoid_points_v[2].reshape(z.shape), color='b', alpha=0.5)
        ax1.set_title('Linear Velocity Ellipsoid')

        # Plot the angular velocity ellipsoid
        ax2.plot_surface(ellipsoid_points_w[0].reshape(x.shape), ellipsoid_points_w[1].reshape(y.shape), ellipsoid_points_w[2].reshape(z.shape), color='r', alpha=0.5)
        ax2.set_title('Angular Velocity Ellipsoid')

        if ax is None:
            plt.show()

    def plot_workspace_monte_carlo(self, joint_limits, num_samples=10000):
        """
        Estimate the robot workspace using Monte Carlo sampling.

        Parameters:
            joint_limits (list): A list of tuples representing the joint limits.
            num_samples (int, optional): Number of samples for Monte Carlo simulation. Defaults to 10000.
        """
        # Initialize device arrays
        joint_samples = cuda.device_array((num_samples, len(joint_limits)), dtype=np.float32)

        # Define the CUDA kernel for generating joint angles
        @cuda.jit
        def generate_joint_samples(rng_states, joint_limits, joint_samples):
            pos = cuda.grid(1)
            if pos < joint_samples.shape[0]:
                for i in range(joint_samples.shape[1]):
                    low, high = joint_limits[i]
                    joint_samples[pos, i] = xoroshiro128p_uniform_float32(rng_states, pos) * (high - low) + low

        # Setup random states
        rng_states = create_xoroshiro128p_states(num_samples, seed=1234)
        device_joint_limits = cuda.to_device(np.array(joint_limits, dtype=np.float32))

        # Launch kernel
        threadsperblock = 256
        blockspergrid = (num_samples + threadsperblock - 1) // threadsperblock
        generate_joint_samples[blockspergrid, threadsperblock](rng_states, device_joint_limits, joint_samples)

        # Copy joint samples to host and calculate workspace points
        host_joint_samples = joint_samples.copy_to_host()
        workspace_points = np.array([self.serial_manipulator.forward_kinematics(thetas)[:3, 3] for thetas in host_joint_samples])

        # Compute convex hull
        hull = ConvexHull(workspace_points)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], triangles=hull.simplices, cmap='viridis', edgecolor='none', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot Workspace (Smooth Convex Hull)')
        plt.show()

    def condition_number(self, thetalist):
        """
        Calculate the condition number of the Jacobian for a given set of joint angles.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.

        Returns:
            float: The condition number of the Jacobian matrix.
        """
        J = self.serial_manipulator.jacobian(thetalist, frame='space')
        return np.linalg.cond(J)

    def near_singularity_detection(self, thetalist, threshold=1e-2):
        """
        Detect if the manipulator is near a singularity by comparing the condition number with a threshold.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.
            threshold (float, optional): Threshold value for the condition number. Defaults to 1e-2.

        Returns:
            bool: True if the manipulator is near a singularity, False otherwise.
        """
        cond_number = self.condition_number(thetalist)
        return cond_number > threshold
