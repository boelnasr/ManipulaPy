#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Kinematics Module - ManipulaPy

This module provides classes and functions for performing kinematic analysis and computations
for serial manipulators, including forward and inverse kinematics, Jacobian calculations,
and end-effector velocity calculations.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Any
from numpy.typing import NDArray
from . import utils
import matplotlib.pyplot as plt
import torch


class SerialManipulator:
    def __init__(
        self,
        M_list: NDArray[np.float64],
        omega_list: Union[NDArray[np.float64], List[float]],
        r_list: Optional[Union[NDArray[np.float64], List[float]]] = None,
        b_list: Optional[Union[NDArray[np.float64], List[float]]] = None,
        S_list: Optional[NDArray[np.float64]] = None,
        B_list: Optional[NDArray[np.float64]] = None,
        G_list: Optional[Union[NDArray[np.float64], List[NDArray[np.float64]]]] = None,
        joint_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    ) -> None:
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
        
        # Extract r_list from S_list if not provided
        self.r_list = r_list if r_list is not None else utils.extract_r_list(S_list)
        # Extract b_list from B_list if not provided  
        self.b_list = b_list if b_list is not None else utils.extract_r_list(B_list)
        
        # Generate S_list if not provided
        self.S_list = (
            S_list
            if S_list is not None
            else utils.extract_screw_list(-omega_list, self.r_list)
        )
        
        # Generate B_list if not provided
        self.B_list = (
            B_list
            if B_list is not None
            else utils.extract_screw_list(omega_list, self.b_list)
        )
        
        # Determine number of joints for joint limits
        if joint_limits is not None:
            self.joint_limits = joint_limits
        else:
            # Try to infer number of joints from available data
            if hasattr(omega_list, 'shape'):
                if omega_list.ndim == 2:
                    n_joints = omega_list.shape[1]
                else:
                    n_joints = len(omega_list) // 3 if len(omega_list) % 3 == 0 else len(omega_list)
            elif hasattr(M_list, 'shape'):
                n_joints = 6  # Default assumption for 6-DOF robot
            else:
                n_joints = 6  # Default fallback
            
            self.joint_limits = [(None, None)] * n_joints

    def update_state(
        self,
        joint_positions: Union[NDArray[np.float64], List[float]],
        joint_velocities: Optional[Union[NDArray[np.float64], List[float]]] = None
    ) -> None:
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

    def forward_kinematics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        frame: str = "space"
    ) -> NDArray[np.float64]:
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
            # Multiply by home pose (use end-effector pose if M_list is an array of poses)
            M = self.M_list[-1] if isinstance(self.M_list, (list, np.ndarray)) and hasattr(self.M_list, '__len__') and len(np.asarray(self.M_list).shape) > 2 else self.M_list
            T = T @ M

        elif frame == "body":
            # T(θ) = M * e^[B1θ1] e^[B2θ2] ... e^[Bnθn]
            T = np.eye(4)
            # Build the product of exponentials from left to right
            for i, theta in enumerate(thetalist):
                T = T @ utils.transform_from_twist(self.B_list[:, i], theta)
            # Then multiply from the left by M (use end-effector pose if M_list is an array of poses)
            M = self.M_list[-1] if isinstance(self.M_list, (list, np.ndarray)) and hasattr(self.M_list, '__len__') and len(np.asarray(self.M_list).shape) > 2 else self.M_list
            T = M @ T

        else:
            raise ValueError("Invalid frame specified. Choose 'space' or 'body'.")

        return T

    def end_effector_velocity(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        frame: str = "space"
    ) -> NDArray[np.float64]:
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

    def jacobian(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        frame: str = "space"
    ) -> NDArray[np.float64]:
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
        T_desired: NDArray[np.float64],
        thetalist0: Union[NDArray[np.float64], List[float]],
        eomg: float = 1e-6,
        ev: float = 1e-6,
        max_iterations: int = 5000,
        plot_residuals: bool = False,
        damping: float = 5e-2,            # lambda for damped least-squares
        step_cap: float = 0.1,            # max norm(delta_theta) per iteration (rad) - reduced from 0.5 for better convergence
        png_name: str = "ik_residuals.png",
    ) -> Tuple[NDArray[np.float64], bool, int]:
        """
        Damped-least-squares iterative IK with joint-limit projection and
        residual plot saved to file (no interactive window).

        Uses the correct error computation: T_err = T_desired @ inv(T_curr)
        to ensure proper convergence in SE(3).
        """
        theta = np.array(thetalist0, dtype=float)
        residuals = []

        for k in range(max_iterations):
            # Current pose
            T_curr = self.forward_kinematics(theta, frame="space")

            # Compute error transformation (correct formulation for SE(3))
            T_err = T_desired @ np.linalg.inv(T_curr)

            # Extract error twist
            V_err = utils.se3ToVec(utils.MatrixLog6(T_err))

            # Compute error magnitudes
            rot_err = np.linalg.norm(V_err[:3])
            trans_err = np.linalg.norm(V_err[3:])
            residuals.append((trans_err, rot_err))

            if rot_err < eomg and trans_err < ev:
                success = True
                break

            # Damped least-squares update using pseudo-inverse (more robust)
            J = self.jacobian(theta, frame="space")
            J_pinv = np.linalg.pinv(J, rcond=damping)
            delta_theta = J_pinv @ V_err

            # Cap step size
            norm_delta = np.linalg.norm(delta_theta)
            if norm_delta > step_cap:
                delta_theta *= step_cap / norm_delta

            theta += delta_theta

            # Project into joint limits
            for i, (mn, mx) in enumerate(self.joint_limits):
                if mn is not None: theta[i] = max(theta[i], mn)
                if mx is not None: theta[i] = min(theta[i], mx)
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

        return theta, success, k + 1

    def joint_velocity(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        V_ee: Union[NDArray[np.float64], List[float]],
        frame: str = "space"
    ) -> NDArray[np.float64]:
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

    def end_effector_pose(
        self,
        thetalist: Union[NDArray[np.float64], List[float]]
    ) -> NDArray[np.float64]:
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