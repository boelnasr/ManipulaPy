import numpy as np
import modern_robotics as mr

class SerialManipulator:
    def __init__(self, M_list, omega_list, r_list=None, b_list=None, S_list=None, B_list=None):
        """
        Initialize the serial manipulator with given parameters.

        Args:
            M_list (np.ndarray): Home position matrix of the end-effector.
            omega_list (np.ndarray): List of rotation axes for each joint.
            r_list (np.ndarray, optional): List of positions of each screw axis in the base frame.
            b_list (np.ndarray, optional): List of positions of each screw axis in the body frame.
            S_list (np.ndarray, optional): List of screw axes in the space frame.
            B_list (np.ndarray, optional): List of screw axes in the body frame.
        """
        self.M_list = M_list
        self.omega_list = omega_list
        self.r_list = r_list if r_list is not None else np.zeros_like(omega_list)
        self.b_list = b_list if b_list is not None else np.zeros_like(omega_list)
        self.S_list = S_list if S_list is not None else self.calculate_screw_list(self.omega_list, self.r_list, space=True)
        self.B_list = B_list if B_list is not None else self.calculate_screw_list(self.omega_list, self.b_list, space=False)

    @staticmethod
    def calculate_screw_list(omega_list, position_list, space=True):
        """
        Calculate the screw axes in either the space frame or the body frame.

        Args:
            omega_list (np.ndarray): List of rotation axes for each joint.
            position_list (np.ndarray): List of positions of each screw axis.
            space (bool): Flag indicating if the screw list is for the space frame (True) or body frame (False).

        Returns:
            np.ndarray: List of screw axes.
        """
        V_list = [np.cross(-omega, position) if space else np.cross(omega, position) for omega, position in zip(omega_list, position_list)]
        return np.vstack((omega_list.T, np.array(V_list).T)).T

    def forward_kinematics(self, thetalist, frame='space'):
        """
        Compute forward kinematics in the specified frame.

        Args:
            thetalist (np.ndarray): Array of joint angles.
            frame (str): The frame of reference ('space' or 'body').

        Returns:
            np.ndarray: Transformation matrix of the end-effector in the specified frame.
        """
        if frame == 'space':
            T = mr.FKinSpace(self.M_list, self.S_list, thetalist)
        elif frame == 'body':
            T = mr.FKinBody(self.M_list, self.B_list, thetalist)
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return np.around(T, decimals=2)

    def end_effector_velocity(self, thetalist, theta_dot, frame='space'):
        """
        Compute the end effector velocity in the specified frame.

        Args:
            thetalist (np.ndarray): Array of joint angles.
            theta_dot (np.ndarray): Array of joint velocities.
            frame (str): The frame of reference ('space' or 'body').

        Returns:
            np.ndarray: The twist of the end effector velocities.
        """
        try:
            J = mr.JacobianSpace(self.S_list, thetalist) if frame == 'space' else mr.JacobianBody(self.B_list, thetalist)
            V = np.dot(J, theta_dot)
            return np.around(V, 2)
        except Exception as e:
            return f"Error in end_effector_velocity: {e}"

    def iterative_inverse_kinematics(self, T, thetalist0, eomg, ev, max_iterations=50):
        """
        Iteratively compute inverse kinematics in the body frame.

        Args:
            T (np.ndarray): The desired end-effector configuration.
            thetalist0 (np.ndarray): An initial guess of joint angles.
            eomg (float): Tolerance on the end-effector orientation error.
            ev (float): Tolerance on the end-effector linear position error.
            max_iterations (int): Maximum number of iterations for the algorithm.

        Returns:
            (np.ndarray, bool): Tuple containing the final joint angles and a boolean indicating success.
        """
        thetalist = np.array(thetalist0).copy()
        for i in range(max_iterations):
            T_current = mr.FKinBody(self.M_list, self.B_list, thetalist)
            Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(T_current), T)))
            if np.linalg.norm(Vb[0:3]) < eomg and np.linalg.norm(Vb[3:6]) < ev:
                return thetalist, True

            Jb = mr.JacobianBody(self.B_list, thetalist)
            thetalist += np.dot(np.linalg.pinv(Jb), Vb)

        return thetalist, False

    def joint_velocity(self, thetalist, V_ee, frame='space'):
        """
        Compute the joint velocities to achieve a desired end effector twist.

        Args:
            thetalist (np.ndarray): Array of joint angles.
            V_ee (np.ndarray): Desired twist of the end effector.
            frame (str): The frame of reference ('space' or 'body').

        Returns:
            np.ndarray: Array of joint velocities.
        """
        try:
            J = mr.JacobianSpace(self.S_list, thetalist) if frame == 'space' else mr.JacobianBody(self.B_list, thetalist)
            theta_dot = np.linalg.pinv(J) @ V_ee
            return np.around(theta_dot, 2)
        except Exception as e:
            return f"Error in compute_joint_velocity: {e}"