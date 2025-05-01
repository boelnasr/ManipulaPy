"""
ManipulaPy Kinematics Module

This module provides classes and functions for performing kinematic analysis and computations
for serial manipulators, including forward and inverse kinematics, Jacobian calculations,
and end-effector velocity calculations.

The main class in this module is `SerialManipulator`, which represents a serial robotic
manipulator with an arbitrary number of joints. This class provides methods for:

- Forward kinematics: Compute the end-effector pose given joint angles
- Inverse kinematics: Compute joint angles to achieve a desired end-effector pose
- Jacobian calculation: Compute the Jacobian matrix in the space or body frame
- End-effector velocity: Compute the end-effector velocity given joint angles and velocities
- Joint velocity: Compute the joint velocities required to achieve a desired end-effector velocity

The module also includes utility functions for working with transformation matrices, twists,
and other kinematic quantities, which are imported from the `utils` module.

To use this module, you need to provide the kinematic parameters of the manipulator, such as
the  screw axes, or the M, G, and B matrices. The `SerialManipulator`
class can then be instantiated with these parameters, and its methods can be used for various
kinematic computations.

Note: This module assumes familiarity with concepts from robotics, kinematics, and Lie theory.
For more information, refer to the documentation or relevant literature.

"""

#!/usr/bin/env python3

import numpy as np
from . import utils
import matplotlib.pyplot as plt
import torch


class SerialManipulator:
    def __init__(
        self,
        M_list,
        omega_list,
        r_list=None,
        b_list=None,
        S_list=None,
        B_list=None,
        G_list=None,
        joint_limits=None,
    ):
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
            joint_limits (list, optional): A list of joint limits. Defaults to None.
        """
        self.M_list = M_list
        self.G_list = G_list
        self.omega_list = omega_list
        self.r_list = r_list if r_list is not None else utils.extract_r_list(S_list)
        self.b_list = b_list if b_list is not None else utils.extract_r_list(B_list)
        self.S_list = (
            S_list
            if S_list is not None
            else utils.extract_screw_list(-omega_list, self.r_list)
        )
        self.B_list = (
            B_list
            if B_list is not None
            else utils.extract_screw_list(omega_list, self.b_list)
        )
        self.joint_limits = (
            joint_limits if joint_limits is not None else [(None, None)] * len(M_list)
        )

    def update_state(self, joint_positions, joint_velocities=None):
        """
        Updates the internal state of the manipulator.

        Args:
            joint_positions (np.ndarray): Current joint positions.
            joint_velocities (np.ndarray, optional): Current joint velocities. Default is None.
        """
        self.joint_positions = np.array(joint_positions)
        if joint_velocities is not None:
            self.joint_velocities = np.array(joint_velocities)
        else:
            self.joint_velocities = np.zeros_like(self.joint_positions)

    def forward_kinematics(self, thetalist, frame="space"):
        """
        Compute the forward kinematics of a robotic arm using the product of exponentials method.

        Args:
            thetalist (numpy.ndarray): A 1D array of joint angles in radians.
            frame (str, optional): The frame in which to compute the forward kinematics.
                Either 'space' or 'body'.

        Returns:
            numpy.ndarray: The 4x4 transformation matrix representing the end-effector's pose.
        """
        if frame == "space":
            # T(θ) = e^[S1θ1] e^[S2θ2] ... e^[Snθn] * M
            T = np.eye(4)
            for i, theta in enumerate(thetalist):
                T = T @ utils.transform_from_twist(self.S_list[:, i], theta)
            # Multiply by home pose
            T = T @ self.M_list

        elif frame == "body":
            # T(θ) = M * e^[B1θ1] e^[B2θ2] ... e^[Bnθn]
            T = np.eye(4)
            # Build the product of exponentials from left to right
            for i, theta in enumerate(thetalist):
                T = T @ utils.transform_from_twist(self.B_list[:, i], theta)
            # Then multiply from the left by M
            T = self.M_list @ T

        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")

        return T

    def end_effector_velocity(self, thetalist, dthetalist, frame="space"):
        """
        Calculate the end effector velocity given the joint angles and joint velocities.

        Parameters:
            thetalist (list): A list of joint angles.
            dthetalist (list): A list of joint velocities.
            frame (str): The frame in which the Jacobian is calculated. Valid values are 'space' and 'body'.

        Returns:
            numpy.ndarray: The end effector velocity.
        """
        if frame == "space":
            J = self.jacobian(thetalist,frame="space")
        elif frame == "body":
            J = self.jacobian(thetalist,frame="body")
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return np.dot(J, dthetalist)

    def jacobian(self, thetalist, frame="space"):
        """
        Calculate the Jacobian matrix for the given joint angles.

        Parameters:
            thetalist (list): A list of joint angles.
            frame (str): The reference frame for the Jacobian calculation.
                        Valid values are 'space' or 'body'. Defaults to 'space'.

        Returns:
            numpy.ndarray: The Jacobian matrix of shape (6, len(thetalist)).
        """
        J = np.zeros((6, len(thetalist)))
        T = np.eye(4)
        if frame == "space":
            for i in range(len(thetalist)):
                J[:, i] = np.dot(utils.adjoint_transform(T), self.S_list[:, i])
                T = np.dot(
                    T, utils.transform_from_twist(self.S_list[:, i], thetalist[i])
                )
        elif frame == "body":
            T = self.forward_kinematics(thetalist, frame="body")
            for i in reversed(range(len(thetalist))):
                J[:, i] = np.dot(
                    utils.adjoint_transform(np.linalg.inv(T)), self.B_list[:, i]
                )
                T = np.dot(
                    T,
                    np.linalg.inv(
                        utils.transform_from_twist(self.B_list[:, i], thetalist[i])
                    ),
                )
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return J
    def iterative_inverse_kinematics(
        self,
        T_desired,
        thetalist0,
        eomg=1e-6,
        ev=1e-6,
        max_iterations=5000,
        plot_residuals=False,
        damping=5e-2,            # λ for damped least-squares
        step_cap=0.5,            # max ‖Δθ‖ per iteration (rad)
        png_name="ik_residuals.png",
    ):
        """
        Damped-least-squares iterative IK with joint-limit projection and
        residual plot saved to file (no interactive window).
        """
        θ = np.array(thetalist0, dtype=float)
        V_desired = utils.se3ToVec(utils.MatrixLog6(T_desired))
        residuals = []

        for k in range(max_iterations):
            # Current pose & twist error
            T_curr = self.forward_kinematics(θ, frame="space")
            V_curr = utils.se3ToVec(utils.MatrixLog6(T_curr))
            V_err  = V_desired - V_curr
            rot_err, trans_err = np.linalg.norm(V_err[:3]), np.linalg.norm(V_err[3:])
            residuals.append((trans_err, rot_err))

            if rot_err < eomg and trans_err < ev:
                success = True
                break

            # Damped least-squares Δθ
            J = self.jacobian(θ, frame="space")
            JJt = J @ J.T
            Δθ  = J.T @ np.linalg.inv(JJt + (damping ** 2) * np.eye(6)) @ V_err

            # Cap step size
            normΔ = np.linalg.norm(Δθ)
            if normΔ > step_cap:
                Δθ *= step_cap / normΔ

            θ += Δθ

            # Project into joint limits
            for i, (mn, mx) in enumerate(self.joint_limits):
                if mn is not None: θ[i] = max(θ[i], mn)
                if mx is not None: θ[i] = min(θ[i], mx)
        else:
            success = False
            k += 1   # max_iterations reached

        # Optional residual plot (non-interactive)
        if plot_residuals:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            it = np.arange(len(residuals))
            tr, rt = zip(*residuals)
            plt.plot(it, tr, label="Translation error")
            plt.plot(it, rt, label="Rotation error")
            plt.xlabel("Iteration"); plt.ylabel("Norm")
            plt.title("IK convergence")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(png_name, dpi=400)
            plt.close()
            print(f"Residual plot saved to {png_name}")

        return θ, success, k + 1

    def joint_velocity(self, thetalist, V_ee, frame="space"):
        """
        Calculates the joint velocity given the joint positions, end-effector velocity, and frame type.

        Parameters:
            thetalist (list): A list of joint positions.
            V_ee (array-like): The end-effector velocity.
            frame (str, optional): The frame type. Defaults to 'space'.

        Returns:
            array-like: The joint velocity.
        """
        if frame == "space":
            J = self.jacobian(thetalist)
        elif frame == "body":
            J = self.jacobian(thetalist, frame="body")
        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")
        return np.linalg.pinv(J) @ V_ee

    def end_effector_pose(self, thetalist):
        """
        Computes the end-effector's position and orientation given joint angles.

        Parameters:
            thetalist (numpy.ndarray): A 1D array of joint angles in radians.

        Returns:
            numpy.ndarray: A 6x1 vector representing the position and orientation (Euler angles) of the end-effector.
        """
        T = self.forward_kinematics(thetalist)
        R, p = utils.TransToRp(T)
        orientation = utils.rotation_matrix_to_euler_angles(R)
        return np.concatenate((p, orientation))

    def hybrid_inverse_kinematics(
        self,
        T_desired,
        neural_network,
        scaler_X,
        scaler_y,
        device,
        thetalist0=None,
        eomg=1e-6,
        ev=1e-6,
        max_iterations=500,
    ):
        """
        Perform hybrid inverse kinematics using a neural network for initial guess and iterative refinement.

        Parameters:
            T_desired (np.ndarray): The desired end-effector pose as a 4x4 transformation matrix.
            neural_network (nn.Module): The trained neural network model.
            scaler_X (StandardScaler): Scaler for the input features.
            scaler_y (StandardScaler): Scaler for the output features (joint angles).
            device (torch.device): The device on which to run the neural network.
            thetalist0 (np.ndarray, optional): Initial guess for joint angles. If None, the neural network is used to generate an initial guess.
            eomg (float, optional): The tolerance for rotational error convergence. Defaults to 1e-6.
            ev (float, optional): The tolerance for translational error convergence. Defaults to 1e-6.
            max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 500.

        Returns:
            np.ndarray: The resulting joint angles.
            bool: Indicates whether the algorithm converged.
            int: The number of iterations performed.
        """
        if thetalist0 is None:
            # Use neural network to get initial guess
            end_effector_pose = np.concatenate(
                (
                    T_desired[:3, 3],
                    utils.rotation_matrix_to_euler_angles(T_desired[:3, :3]),
                )
            )
            end_effector_pose = scaler_X.transform([end_effector_pose])
            end_effector_pose = torch.tensor(end_effector_pose, dtype=torch.float32).to(
                device
            )
            thetalist0 = (
                neural_network(end_effector_pose).detach().cpu().numpy().flatten()
            )
            thetalist0 = scaler_y.inverse_transform([thetalist0])[0]

        # Refine using iterative method
        thetalist, success, num_iterations = self.iterative_inverse_kinematics(
            T_desired, thetalist0, eomg, ev, max_iterations
        )

        return thetalist, success, num_iterations
