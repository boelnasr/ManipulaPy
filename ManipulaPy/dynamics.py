#!/usr/bin/env python3

import numpy as np
from .kinematics import SerialManipulator
from .utils import adjoint_transform as ad
class ManipulatorDynamics(SerialManipulator):
    def __init__(self, M_list, omega_list, r_list, b_list, S_list, B_list, Glist):
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist
        self._mass_matrix_cache = {}

    def mass_matrix(self, thetalist):
        """
        Compute the mass matrix of the robot given the joint angles.

        Parameters:
            thetalist (list): A list of joint angles.
        
        Returns:
            The mass matrix of the robot.
        """
        thetalist_key = tuple(thetalist)
        if thetalist_key in self._mass_matrix_cache:
            return self._mass_matrix_cache[thetalist_key]

        n = len(thetalist)
        M = np.zeros((n, n))
        AdT = np.zeros((6, 6, n + 1))
        AdT[:, :, 0] = np.eye(6)

        for i in range(n):
            T = self.forward_kinematics(thetalist[:i + 1], 'space')
            AdT[:, :, i + 1] = ad(T)

        for i in range(n):
            F = np.zeros(6)
            for j in range(i, n):
                AdTi = AdT[:, :, j + 1].T
                I = self.Glist[j]
                Ia = np.dot(AdTi, np.dot(I, AdT[:, :, j + 1]))
                dV = np.zeros(6)
                dV[5] = 1 if i == j else 0
                F += np.dot(Ia, dV)
            M[i, i:] = np.dot(self.jacobian( thetalist).T, F)[i:]

        M = M + M.T - np.diag(np.diag(M))
        self._mass_matrix_cache[thetalist_key] = M
        return M

    def partial_derivative(self, i, j, k, thetalist):
        """
        Calculates the partial derivative of the mass matrix with respect to a specific joint angle.

        Parameters:
            i (int): The row index of the desired element in the mass matrix.
            j (int): The column index of the desired element in the mass matrix.
            k (int): The index of the joint angle with respect to which the partial derivative is calculated.
            thetalist (List[float]): A list of joint angles.

        Returns:
            float: The value of the partial derivative.
        """
        epsilon = 1e-6
        thetalist_plus = np.array(thetalist)
        thetalist_plus[k] += epsilon
        M_plus = self.mass_matrix(thetalist_plus)

        thetalist_minus = np.array(thetalist)
        thetalist_minus[k] -= epsilon
        M_minus = self.mass_matrix(thetalist_minus)

        return (M_plus[i, j] - M_minus[i, j]) / (2 * epsilon)

    def velocity_quadratic_forces(self, thetalist, dthetalist):
        """
        Calculate the velocities of the quadratic forces acting on the robot's links.

        Parameters:
            thetalist (list): A list of joint angles (in radians) of the robot's links.
            dthetalist (list): A list of joint velocities (in radians/second) of the robot's links.

        Returns:
            c (ndarray): An array of the velocities of the quadratic forces acting on the robot's links.

        """
        n = len(thetalist)
        c = np.zeros(n)
        J = self.jacobian(thetalist)
        for i in range(n):
            c[i] = sum([self.partial_derivative(i, j, k, thetalist) * dthetalist[j] * dthetalist[k] for j in range(n) for k in range(n)])
        return c

    def gravity_forces(self, thetalist, g=[0, 0, -9.81]):
        """
        Calculates the gravity forces for a given joint configuration.

        Parameters:
            thetalist (list): A list of joint angles.
            g (list, optional): The gravity vector in the base frame. Defaults to [0, 0, -9.81].

        Returns:
            list: A list of gravity forces acting on each joint.
        """
        n = len(thetalist)
        grav = np.zeros(n)
        M = self.mass_matrix(thetalist)
        G = np.zeros((6, 1))
        G[5] = -1
        for i in range(n):
            for j in range(i, n):
                AdT =ad(self.forward_kinematics(thetalist[:j + 1], 'space'))
                Gj = np.dot(AdT.T, G)
                grav[i] += np.dot(M[i, :], Gj.flatten())
        return grav

    def inverse_dynamics(self, thetalist, dthetalist, ddthetalist, g, Ftip):
        """
        Compute the inverse dynamics of a robotic arm.

        Parameters:
            thetalist (list): The joint angles of the arm in radians.
            dthetalist (list): The joint velocities of the arm in radians per second.
            ddthetalist (list): The joint accelerations of the arm in radians per second squared.
            g (float): The acceleration due to gravity.
            Ftip (list): The external forces applied to the end effector of the arm.

        Returns:
            taulist (list): The torques required at each joint to achieve the desired motion.

        This function computes the inverse dynamics of a robotic arm using the given joint angles, velocities,
        accelerations, gravity, and external forces. It first calculates the mass matrix, velocity quadratic forces,
        and gravity forces. Then it computes the transpose of the Jacobian matrix and uses it to compute the
        torques required at each joint. The computed torques are returned as a list.
        """
        n = len(thetalist)
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        taulist = np.dot(M, ddthetalist) + c + g_forces + np.dot(J_transpose, Ftip)
        return taulist

    def forward_dynamics(self, thetalist, dthetalist, taulist, g, Ftip):
        """
        Compute the forward dynamics of the robot given the current joint angles, joint velocities, applied torques, gravity vector, and external forces.
        
        Parameters:
            thetalist (list): List of joint angles (radians) [theta_1, theta_2, ..., theta_n].
            dthetalist (list): List of joint velocities (radians/s) [dtheta_1, dtheta_2, ..., dtheta_n].
            taulist (list): List of applied torques (Nm) [tau_1, tau_2, ..., tau_n].
            g (list): Gravity vector (m/s^2) [g_x, g_y, g_z].
            Ftip (list): External force applied at the end effector (N) [Fx, Fy, Fz].
        
        Returns:
            ddthetalist (list): List of joint accelerations (rad/s^2) [ddtheta_1, ddtheta_2, ..., ddtheta_n].
        """
        
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = taulist - c - g_forces - np.dot(J_transpose, Ftip)
        ddthetalist = np.linalg.solve(M, rhs)
        return ddthetalist

