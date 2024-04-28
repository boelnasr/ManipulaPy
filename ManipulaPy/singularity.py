import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from numpy import linalg as la
import time
class Singularity:
    def __init__(self, serial_manipulator):
        self.serial_manipulator = serial_manipulator

    def singularity_analysis(self, thetalist):
        """
        Calculate the singularity analysis for a given set of joint angles.

        Parameters:
        - thetalist: a list of joint angles (numpy array)

        Returns:
        - boolean: True if the robot is in a singularity, False otherwise
        """
        J_s = mr.JacobianSpace(self.serial_manipulator.S_list, thetalist)
        A = np.matmul(J_s, J_s.T)
        ev, _ = np.linalg.eig(A)
        return min(ev) <= 1e-4  # True if in singularity

    def manipulability_ellipsoid(self, thetalist, ax=None):
        """
        Compute and plot the manipulability ellipsoids for linear and angular velocities.

        Parameters:
            thetalist (list): A list of joint angles.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot the ellipsoid on.
        """
        J = mr.JacobianSpace(self.serial_manipulator.S_list, thetalist)
        J_v = J[:3, :]  # Linear velocity part of the Jacobian
        J_w = J[3:, :]  # Angular velocity part of the Jacobian

        # Compute SVD for linear velocity Jacobian
        U_v, S_v, _ = np.linalg.svd(J_v)
        radii_v = 1.0 / np.sqrt(S_v)

        # Compute SVD for angular velocity Jacobian
        U_w, S_w, _ = np.linalg.svd(J_w)
        radii_w = 1.0 / np.sqrt(S_w)

        # Generate points on a unit sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        # Transform points to ellipsoids
        ellipsoid_points_v = np.dot(np.diag(radii_v), np.dot(U_v, np.array([x.flatten(), y.flatten(), z.flatten()])))
        ellipsoid_points_w = np.dot(np.diag(radii_w), np.dot(U_w, np.array([x.flatten(), y.flatten(), z.flatten()])))

        if ax is None:
            fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        else:
            axs = [ax, ax]

        # Plot the linear velocity ellipsoid
        axs[0].plot_surface(ellipsoid_points_v[0].reshape(x.shape), 
                            ellipsoid_points_v[1].reshape(y.shape), 
                            ellipsoid_points_v[2].reshape(z.shape), 
                            color='b', alpha=0.5)
        axs[0].set_title('Linear Velocity Ellipsoid')

        # Plot the angular velocity ellipsoid
        axs[1].plot_surface(ellipsoid_points_w[0].reshape(x.shape), 
                            ellipsoid_points_w[1].reshape(y.shape), 
                            ellipsoid_points_w[2].reshape(z.shape), 
                            color='r', alpha=0.5)
        axs[1].set_title('Angular Velocity Ellipsoid')

        # Plot and label the main axes for both ellipsoids
        for ax, U, radii in zip(axs, [U_v, U_w], [radii_v, radii_w]):
            for i in range(3):
                ax.quiver(0, 0, 0, U[0, i]*radii[i], U[1, i]*radii[i], U[2, i]*radii[i], color='k')
                ax.text(U[0, i]*radii[i], U[1, i]*radii[i], U[2, i]*radii[i], f'Axis {i+1}', color='k')

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

        plt.show()


    def workspace_analysis(self, joint_limits, resolution=0.1):
        """
        Generate a list of workspace points for a given set of joint limits and resolution.

        Parameters:
            joint_limits (list): A list of tuples representing the lower and upper limits of each joint.
            resolution (float, optional): The step size used to generate the joint angles. Defaults to 0.1.

        Returns:
            list: A list of 3D workspace points representing the end effector positions.
        """
        workspace_points = []
        num_joints = len(joint_limits)
        joint_angles = [np.arange(limits[0], limits[1], resolution) for limits in joint_limits]
        for joint_combination in np.array(np.meshgrid(*joint_angles)).T.reshape(-1, num_joints):
            T = mr.FKinSpace(self.serial_manipulator.M_list, self.serial_manipulator.S_list, joint_combination)
            end_effector_pos = T[0:3, 3]
            workspace_points.append(end_effector_pos)
        return workspace_points

    def plot_workspace(self, workspace_points):
        """
        Plot the given workspace points in a 3D scatter plot.

        Parameters:
            workspace_points (list): A list of 3D points representing the workspace.

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_coords = [point[0] for point in workspace_points]
        y_coords = [point[1] for point in workspace_points]
        z_coords = [point[2] for point in workspace_points]
        ax.scatter(x_coords, y_coords, z_coords)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Workspace of the Manipulator')
        plt.show()

    def plot_workspace_convex_hull(self, workspace_points):
        """
        Plots the convex hull of the workspace using the given workspace points.

        Parameters:
            workspace_points (list of arrays): List of workspace points in 3D space.

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        hull = ConvexHull(workspace_points)
        for simplex in hull.simplices:
            plt.plot([workspace_points[simplex, 0], workspace_points[simplex, 1], workspace_points[simplex, 2]], 'k-')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Workspace Convex Hull of the Manipulator')
        plt.show()

    def dexterity_analysis(self, Thetalist):
        """
        Calculates the condition number of the Jacobian matrix for dexterity analysis.

        Args:
            Thetalist (list): List of joint angles [theta1, theta2, ..., thetan].

        Returns:
            float: The condition number of the Jacobian matrix.
        """
        J_s = mr.JacobianSpace(self.serial_manipulator.S_list, Thetalist)
        condition_number = np.linalg.cond(J_s)
        return condition_number
    
    def near_singularity_detection(self, Thetalist, threshold):
        """
        Generate a function comment for the given function body.

        Args:
            Thetalist (list): A list of joint angles.
            threshold (float): The threshold for determining if the condition number is near singularity.

        Returns:
            bool: True if the condition number is greater than the threshold, False otherwise.
        """
        J_s = mr.JacobianSpace(self.serial_manipulator.S_list, Thetalist)
        condition_number = np.linalg.cond(J_s)
        return condition_number > threshold

    def jacobian_null_space_analysis(self, Thetalist):
        """
        Calculate the null space of the Jacobian matrix for a given set of joint angles.

        Args:
            Thetalist (list): A list of joint angles.

        Returns:
            numpy.ndarray: The null space of the Jacobian matrix.
        """
        J_s = mr.JacobianSpace(self.serial_manipulator.S_list, Thetalist)
        null_space = la.null_space(J_s)
        return null_space
    
    def inverse_kinematics_robustness(self, desired_end_effector_pose, initial_thetalist, tolerance=0.001, max_iterations=100):
        """
        Analyze the robustness of inverse kinematics solutions for a desired end-effector pose.

        :param desired_end_effector_pose: The desired pose of the end-effector (4x4 homogeneous transformation matrix).
        :param initial_thetalist: Initial guess for the joint angles (list or numpy array).
        :param tolerance: Tolerance for the end-effector position and orientation error.
        :param max_iterations: Maximum number of iterations for the inverse kinematics algorithm.
        :return: Robustness measure of the inverse kinematics solution.
        """
        # Calculate the inverse kinematics solution using iterative method
        ik_solution, success = self.iterative_inverse_kinematics(desired_end_effector_pose, initial_thetalist, tolerance, tolerance, max_iterations)

        if not success:
            return "Inverse kinematics solution not found within given tolerance and iterations."

        # Perturb the desired pose slightly and solve inverse kinematics again
        perturbations = [1e-4, -1e-4]
        robustness_measures = []

        for perturbation in perturbations:
            for i in range(3):  # Perturb x, y, z positions
                perturbed_pose = np.copy(desired_end_effector_pose)
                perturbed_pose[i, 3] += perturbation

                perturbed_ik_solution, _ = self.iterative_inverse_kinematics(perturbed_pose, initial_thetalist, tolerance, tolerance, max_iterations)

                # Calculate the norm of the difference between the original and perturbed solutions
                robustness_measure = np.linalg.norm(np.array(ik_solution) - np.array(perturbed_ik_solution))
                robustness_measures.append(robustness_measure)

        # Average robustness measure
        average_robustness = np.mean(robustness_measures)
        return average_robustness