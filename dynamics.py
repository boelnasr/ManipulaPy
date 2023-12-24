import numpy as np
import modern_robotics as mr
from kinematics import SerialManipulator

class ManipulatorDynamics(SerialManipulator):
    def __init__(self, M_list, omega_list, r_list, b_list, S_list, B_list, Glist):
        super().__init__(M_list, omega_list, r_list, b_list, S_list, B_list)
        self.Glist = Glist  # Spatial inertia matrices of the links

    def mass_matrix(self, thetalist):
        """
        Computes the mass matrix of the manipulator in the given configuration.

        :param thetalist: A list of joint variables.
        :return: The mass matrix of the manipulator.
        """
        n = len(thetalist)
        M = np.zeros((n, n))
        AdT = np.zeros((6, 6, n + 1))
        AdT[:, :, 0] = np.eye(6)  # Initialize with identity for the base frame

        # Forward iteration to compute AdT
        for i in range(n):
            T = self.forward_kinematics(thetalist[:i + 1], 'space')  # Assuming 'space' frame
            AdT[:, :, i + 1] = mr.Adjoint(T)

        # Backward iteration to compute the mass matrix
        for i in range(n):
            F = np.zeros(6)
            for j in range(i, n):
                AdTi = AdT[:, :, j + 1].T
                I = self.Glist[j]  # Spatial inertia matrix of link j
                Ia = np.dot(AdTi, np.dot(I, AdT[:, :, j + 1]))

                # Compute force due to unit acceleration at joint i
                dV = np.zeros(6)
                dV[5] = 1 if i == j else 0
                F += np.dot(Ia, dV)

            # Compute the i-th column of the mass matrix
            M[i, :] = np.dot(mr.JacobianSpace(self.S_list,thetalist).T, F)

        return M
