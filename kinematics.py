import numpy as np
import utils

class SerialManipulator:
    def __init__(self, M_list, omega_list, r_list=None, b_list=None, S_list=None, B_list=None, G_list=None):
        """
    	    Initialize the class with the given parameters.
    	
    	    Parameters:
    	        M_list (list): A list of M values.
    	        omega_list (list): A list of omega values.
    	        r_list (list, optional): A list of r values. Defaults to None.
    	        b_list (list, optional): A list of b values. Defaults to None.
    	        S_list (list, optional): A list of S values. Defaults to None.
    	        B_list (list, optional): A list of B values. Defaults to None.
    	        G_list (list, optional): A list of G values. Defaults to None.
        """
    
        self.M_list = M_list
        self.G_list = G_list
        self.omega_list = omega_list
        self.r_list = r_list if r_list is not None else utils.extract_r_list(S_list)
        self.b_list = b_list if b_list is not None else utils.extract_r_list(B_list)
        self.S_list = S_list if S_list is not None else utils.extract_screw_list(omega_list, self.r_list)
        self.B_list = B_list if B_list is not None else utils.extract_screw_list(omega_list, self.b_list)

    def forward_kinematics(self, thetalist, frame='space'):
        """
        Compute the forward kinematics of a robotic arm.

        Args:
            thetalist (numpy.ndarray): A 1D array of joint angles in radians.
            frame (str, optional): The frame in which to compute the forward kinematics. 
                                    Either 'space' (default) or 'body'.

        Returns:
            numpy.ndarray: The transformation matrix representing the end-effector's pose.
        """
        
        T = np.eye(4)
        if frame == 'space':
            for i, theta in enumerate(thetalist):
                T = np.dot(T, utils.transform_from_twist(self.S_list[:, i], theta))
        elif frame == 'body':
            for i, theta in reversed(list(enumerate(thetalist))):
                T = np.dot(utils.transform_from_twist(self.B_list[:, i], theta), T)
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return T

    def end_effector_velocity(self, thetalist, dthetalist, frame='space'):
        """
        Calculate the end effector velocity given the joint angles and joint velocities.

        Parameters:
            thetalist (list): A list of joint angles.
            dthetalist (list): A list of joint velocities.
            frame (str): The frame in which the Jacobian is calculated. Valid values are 'space' and 'body'.

        Returns:
            numpy.ndarray: The end effector velocity.

        Raises:
            ValueError: If an invalid frame is specified.
        """
        
        if frame == 'space':
            J = self.jacobian_space(thetalist)
        elif frame == 'body':
            J = self.jacobian_body(thetalist)
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return np.dot(J, dthetalist)

    def jacobian(self, thetalist, frame='space'):
        """
        Calculate the Jacobian matrix for the given joint angles.

        Parameters:
            thetalist (list): A list of joint angles.
            frame (str): The reference frame for the Jacobian calculation. 
                        Valid values are 'space' or 'body'. Defaults to 'space'.

        Returns:
            numpy.ndarray: The Jacobian matrix of shape (6, len(thetalist)).

        Raises:
            ValueError: If an invalid frame is specified.

        """
        
        J = np.zeros((6, len(thetalist)))
        T = np.eye(4)
        if frame == 'space':
            for i in range(len(thetalist)):
                J[:, i] = np.dot(utils.adjoint_transform(T), self.S_list[:, i])
                T = np.dot(T, utils.transform_from_twist(self.S_list[:, i], thetalist[i]))
        elif frame == 'body':
            T = self.forward_kinematics(thetalist, frame='body')
            for i in reversed(range(len(thetalist))):
                J[:, i] = np.dot(utils.adjoint_transform(np.linalg.inv(T)), self.B_list[:, i])
                T = np.dot(T, np.linalg.inv(utils.transform_from_twist(self.B_list[:, i], thetalist[i])))
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return J

    def iterative_inverse_kinematics(self, T_desired, thetalist0, eomg =1*10^-4, ev=1*10^-4, max_iterations=50):
        """
        Perform iterative inverse kinematics to find the joint angles that achieve a desired end-effector pose.

        Parameters:
            T_desired (numpy.ndarray): The desired end-effector pose as a 4x4 transformation matrix.
            thetalist0 (list): The initial guess for the joint angles.
            eomg (float): The maximum error allowed for the orientation of the end-effector.
            ev (float): The maximum error allowed for the position of the end-effector.
            max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 50.

        Returns:
            tuple: A tuple containing the joint angles (thetalist) and a boolean value indicating whether the solution was found.
        """
        
        thetalist = np.array(thetalist0)
        for _ in range(max_iterations):
            T_current = self.forward_kinematics(thetalist, frame='body')
            Vb = utils.logm(np.dot(utils.adjoint_transform(np.linalg.inv(T_current)), T_desired))
            if np.linalg.norm(Vb[:3]) < eomg and np.linalg.norm(Vb[3:]) < ev:
                return thetalist, True
            Jb = self.jacobian_body(thetalist)
            thetalist += np.dot(np.linalg.pinv(Jb), Vb)
        return thetalist, False

    def joint_velocity(self, thetalist, V_ee, frame='space'):
        """
        Calculates the joint velocity given the joint positions, end-effector velocity, and frame type.

        Parameters:
            thetalist (list): A list of joint positions.
            V_ee (array-like): The end-effector velocity.
            frame (str, optional): The frame type. Defaults to 'space'.

        Returns:
            array-like: The joint velocity.

        Raises:
            ValueError: If an invalid frame type is specified.
        """
        
        if frame == 'space':
            J = self.jacobian(thetalist)
        elif frame == 'body':
            J = self.jacobian(thetalist,frame='body')
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return np.linalg.pinv(J) @ V_ee
