import numpy as np
import modern_robotics as mr 


class SerialManipulator:
    def __init__(self, M_list: np.ndarray, omega_list: np.ndarray, r_list: np.ndarray = None, b_list: np.ndarray = None):
        """
        Initializes the serial manipulator with given parameters.

        Args:
            M_list (np.ndarray): Home position matrix.
            omega_list (np.ndarray): List of rotation axes.
            r_list (np.ndarray, optional): Screw positions in the base frame.
            b_list (np.ndarray, optional): Screw positions in the body frame.
        """
        self.M_list = M_list
        self.omega_list = omega_list
        self.r_list = r_list if r_list is not None else []
        self.b_list = b_list if b_list is not None else []
        if self.r_list:
            self.V_s_list = [np.cross(omega, -r) for omega, r in zip(self.omega_list, self.r_list.T)]
            self.S_list = np.vstack((self.omega_list.T, self.V_s_list)).T
            
        if self.b_list:
            self.V_b_list = [np.cross(omega, -b) for omega, b in zip(self.omega_list, self.b_list.T)]
            self.B_list = np.vstack((self.omega_list.T, self.V_b_list)).T

    def forward_kinematics(self, thetalist, frame='space'):
        """
        Computes forward kinematics in the specified frame.

        Args:
            thetalist (np.ndarray): Joint angles.
            frame (str): The frame of reference ('space' or 'body').

        Returns:
            np.ndarray: Transformation matrix of the end-effector in the specified frame.
        """
        if frame == 'space':
            return np.around(mr.FKinSpace(self.M_list, self.S_list, thetalist), 2)
        elif frame == 'body':
            return np.around(mr.FKinBody(self.M_list, self.B_list, thetalist), 2)
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")

    
