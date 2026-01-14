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
        max_iterations: int = 10000,
        plot_residuals: bool = False,
        damping: float = 2e-2,            # lambda for damped least-squares (optimized: 2e-2 for 6-DOF, 1e-2 for 2-DOF)
        step_cap: float = 0.3,            # max norm(delta_theta) per iteration (rad). Optimized: 0.3 for 6-DOF stability, 0.1 for 2-DOF
        png_name: str = "ik_residuals.png",
        weight_orientation: float = 1.0,  # scale for rotational error in solve step
        weight_position: float = 1.0,     # scale for translational error in solve step
        adaptive_tuning: bool = False,
        backtracking: bool = False,
    ) -> Tuple[NDArray[np.float64], bool, int]:
        """
        Damped-least-squares iterative IK with joint-limit projection and
        residual plot saved to file (no interactive window).

        Features:
        - Levenberg-Marquardt style adaptive damping
        - SVD-robust Jacobian solve for near-singular configs
        - Stagnation detection with perturbation recovery
        - Improved line search with multiple scales
        - Best solution tracking
        """
        theta = np.array(thetalist0, dtype=float)
        residuals = []
        damping_local = damping
        step_cap_local = step_cap
        prev_error = float('inf')
        min_damping, max_damping = 1e-6, 5e-1
        min_step_cap = 0.01
        nu = 2.0  # LM damping adjustment factor

        # Best solution tracking
        best_theta = theta.copy()
        best_error = float('inf')

        # Stagnation detection
        stall_count = 0
        max_stall = 20

        def compute_geometric_error(T_curr, T_target):
            """Compute geometric error without adjoint amplification."""
            # Position error
            pos_err = T_target[:3, 3] - T_curr[:3, 3]
            trans_err = np.linalg.norm(pos_err)

            # Rotation error using axis-angle
            R_curr = T_curr[:3, :3]
            R_target = T_target[:3, :3]
            R_err = R_curr.T @ R_target

            trace_val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            angle = np.arccos(trace_val)
            rot_err = abs(angle)

            # Extract rotation axis
            if angle < 1e-6:
                omega_err = np.array([R_err[2, 1] - R_err[1, 2],
                                      R_err[0, 2] - R_err[2, 0],
                                      R_err[1, 0] - R_err[0, 1]]) / 2
            elif abs(angle - np.pi) < 1e-6:
                diag = np.diag(R_err)
                idx = np.argmax(diag)
                axis = np.zeros(3)
                axis[idx] = 1.0
                omega_err = angle * axis
            else:
                axis = np.array([R_err[2, 1] - R_err[1, 2],
                                 R_err[0, 2] - R_err[2, 0],
                                 R_err[1, 0] - R_err[0, 1]]) / (2 * np.sin(angle) + 1e-10)
                omega_err = angle * axis

            # Transform to space frame
            omega_err_space = R_curr @ omega_err

            # 6D error [angular, linear]
            V_err = np.concatenate([omega_err_space, pos_err])
            return V_err, rot_err, trans_err

        def svd_robust_solve(J, V_err, damping_val):
            """SVD-based damped least squares for near-singular Jacobians."""
            try:
                U, s, Vt = np.linalg.svd(J, full_matrices=False)
                # Damped pseudo-inverse: σ / (σ² + λ²)
                s_damped = s / (s ** 2 + damping_val ** 2 + 1e-12)
                return Vt.T @ (s_damped * (U.T @ V_err))
            except np.linalg.LinAlgError:
                # Fallback to standard solve
                JTJ = J.T @ J
                lambda_I = (damping_val ** 2) * np.eye(JTJ.shape[0])
                return np.linalg.solve(JTJ + lambda_I, J.T @ V_err)

        def clip_to_limits(th):
            """Clip joint angles to limits."""
            th_clipped = th.copy()
            for i, (mn, mx) in enumerate(self.joint_limits):
                if mn is not None:
                    th_clipped[i] = max(th_clipped[i], mn)
                if mx is not None:
                    th_clipped[i] = min(th_clipped[i], mx)
            return th_clipped

        for k in range(max_iterations):
            # Current pose & geometric error
            T_curr = self.forward_kinematics(theta, frame="space")
            V_err, rot_err, trans_err = compute_geometric_error(T_curr, T_desired)
            current_error = rot_err + trans_err
            residuals.append((trans_err, rot_err))

            # Check convergence
            if rot_err < eomg and trans_err < ev:
                success = True
                break

            # Track best solution
            if current_error < best_error:
                best_error = current_error
                best_theta = theta.copy()
                stall_count = 0
            else:
                stall_count += 1

            # Stagnation recovery: perturb if stuck
            if stall_count > max_stall:
                # Add small random perturbation to escape local minimum
                perturbation = 0.1 * np.random.randn(len(theta))
                theta = clip_to_limits(best_theta + perturbation)
                damping_local = damping  # Reset damping
                stall_count = 0
                nu = 2.0
                continue

            # Levenberg-Marquardt adaptive damping
            if adaptive_tuning and k > 0:
                if current_error < prev_error * 0.75:
                    # Good progress - reduce damping (more Newton-like)
                    damping_local = max(min_damping, damping_local / 3)
                    step_cap_local = min(step_cap * 1.5, step_cap_local * 1.2)
                    nu = 2.0
                elif current_error < prev_error * 0.95:
                    # Modest progress - slightly reduce damping
                    damping_local = max(min_damping, damping_local / 1.5)
                elif current_error > prev_error:
                    # Got worse - increase damping (more gradient-like)
                    damping_local = min(max_damping, damping_local * nu)
                    nu = min(nu * 1.5, 8)
                    step_cap_local = max(min_step_cap, step_cap_local * 0.7)

            prev_error = current_error

            # Compute Jacobian and weighted error
            J_space = self.jacobian(theta, frame="space")
            V_weighted = V_err.copy()
            V_weighted[:3] *= weight_orientation
            V_weighted[3:] *= weight_position

            # SVD-robust solve
            delta_theta = svd_robust_solve(J_space, V_weighted, damping_local)

            # Cap step size
            norm_delta = np.linalg.norm(delta_theta)
            if norm_delta > step_cap_local:
                delta_theta *= step_cap_local / norm_delta

            # Line search with multiple scales
            if backtracking:
                best_scale_theta = theta
                best_scale_error = current_error
                scales = [1.0, 0.5, 0.25, 0.125, 0.75]  # More scales for better search

                for scale in scales:
                    candidate = clip_to_limits(theta + scale * delta_theta)
                    T_try = self.forward_kinematics(candidate, frame="space")
                    _, rot_try, trans_try = compute_geometric_error(T_try, T_desired)
                    error_try = rot_try + trans_try

                    if error_try < best_scale_error:
                        best_scale_error = error_try
                        best_scale_theta = candidate

                # Accept best step (even if worse, to avoid getting stuck)
                if best_scale_error < current_error * 1.1:  # Allow slight increase
                    theta = best_scale_theta
                else:
                    # All scales failed - take small step anyway
                    theta = clip_to_limits(theta + 0.1 * delta_theta)
            else:
                theta = clip_to_limits(theta + delta_theta)

        else:
            success = False
            k += 1   # max_iterations reached

        # Return best solution found if current isn't converged
        if not success and best_error < current_error:
            theta = best_theta
            T_curr = self.forward_kinematics(theta, frame="space")
            _, rot_err, trans_err = compute_geometric_error(T_curr, T_desired)
            # Check if best solution meets tolerance
            if rot_err < eomg and trans_err < ev:
                success = True

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

    def smart_inverse_kinematics(
        self,
        T_desired: NDArray[np.float64],
        strategy: str = "workspace_heuristic",
        theta_current: Optional[Union[NDArray[np.float64], List[float]]] = None,
        T_current: Optional[NDArray[np.float64]] = None,
        cache: Optional[Any] = None,  # IKInitialGuessCache instance
        eomg: float = 1e-6,
        ev: float = 1e-6,
        max_iterations: int = 10000,
        plot_residuals: bool = False,
        damping: float = 2e-2,
        step_cap: float = 0.3,
        png_name: str = "ik_residuals.png",
        weight_orientation: float = 1.0,
        weight_position: float = 1.0,
        adaptive_tuning: bool = True,
        backtracking: bool = True,
        auto_fallback: bool = True,
    ) -> Tuple[NDArray[np.float64], bool, int]:
        """
        Smart inverse kinematics with intelligent initial guess strategies.

        Automatically selects initial guess using various strategies for improved
        convergence. With auto_fallback=True, tries multiple strategies if first fails.

        Args:
            T_desired: Target 4x4 transformation matrix
            strategy: Initial guess strategy to use:
                - 'workspace_heuristic': Geometric approximation (default, recommended)
                - 'extrapolate': Extrapolate from current config (for trajectories)
                - 'cached': Use nearest cached solution (requires cache parameter)
                - 'random': Random within joint limits
                - 'midpoint': Midpoint of joint limits
            theta_current: Current joint angles (required for 'extrapolate')
            T_current: Current end-effector pose (required for 'extrapolate')
            cache: IKInitialGuessCache instance (required for 'cached')
            auto_fallback: If True, try other strategies if primary fails (default: True)
            Other args same as iterative_inverse_kinematics()

        Returns:
            Tuple of (theta, success, iterations)
        """
        from . import ik_helpers

        n_joints = len(self.joint_limits)

        valid_strategies = ["workspace_heuristic", "extrapolate", "cached", "random", "midpoint"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {valid_strategies}"
            )

        def get_initial_guess(strat):
            """Generate initial guess for given strategy."""
            if strat == "workspace_heuristic":
                return ik_helpers.workspace_heuristic_guess(
                    T_desired, n_joints, self.joint_limits
                )
            elif strat == "extrapolate":
                if theta_current is None or T_current is None:
                    return None
                return ik_helpers.extrapolate_from_current(
                    theta_current, T_current, T_desired,
                    lambda th: self.jacobian(th, frame="space"),
                    self.joint_limits, alpha=0.5
                )
            elif strat == "cached":
                if cache is None:
                    return None
                return cache.get_nearest(T_desired, k=3, joint_limits=self.joint_limits)
            elif strat == "random":
                return ik_helpers.random_in_limits(self.joint_limits)
            elif strat == "midpoint":
                return ik_helpers.midpoint_of_limits(self.joint_limits)
            else:
                return None

        def try_ik(theta0):
            """Try IK with given initial guess."""
            return self.iterative_inverse_kinematics(
                T_desired, theta0, eomg, ev, max_iterations,
                plot_residuals, damping, step_cap, png_name,
                weight_orientation, weight_position, adaptive_tuning, backtracking
            )

        # Primary strategy
        theta0 = get_initial_guess(strategy)
        if theta0 is None:
            theta0 = ik_helpers.workspace_heuristic_guess(T_desired, n_joints, self.joint_limits)

        theta, success, iters = try_ik(theta0)

        if success or not auto_fallback:
            return theta, success, iters

        # Fallback strategies if primary failed
        fallback_strategies = ["midpoint", "random", "random", "random"]
        total_iters = iters
        best_theta = theta
        best_error = float('inf')

        # Evaluate initial result
        T_curr = self.forward_kinematics(theta, frame="space")
        pos_err = np.linalg.norm(T_curr[:3, 3] - T_desired[:3, 3])
        R_err = T_curr[:3, :3].T @ T_desired[:3, :3]
        rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        best_error = pos_err + rot_err

        for fallback in fallback_strategies:
            theta0 = get_initial_guess(fallback)
            if theta0 is None:
                continue

            theta_try, success_try, iters_try = try_ik(theta0)
            total_iters += iters_try

            if success_try:
                return theta_try, True, total_iters

            # Track best solution
            T_curr = self.forward_kinematics(theta_try, frame="space")
            pos_err = np.linalg.norm(T_curr[:3, 3] - T_desired[:3, 3])
            R_err = T_curr[:3, :3].T @ T_desired[:3, :3]
            rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            error = pos_err + rot_err

            if error < best_error:
                best_error = error
                best_theta = theta_try

        return best_theta, False, total_iters

    def robust_inverse_kinematics(
        self,
        T_desired: NDArray[np.float64],
        max_attempts: int = 10,
        eomg: float = 2e-3,
        ev: float = 2e-3,
        max_iterations: int = 5000,
        verbose: bool = False
    ) -> Tuple[NDArray[np.float64], bool, int, str]:
        """
        Robust inverse kinematics with adaptive multi-start strategy.

        Tries multiple initial guesses and parameter combinations to maximize
        success rate. Tracks best solution across all attempts.

        Args:
            T_desired: Target 4x4 transformation matrix
            max_attempts: Maximum IK attempts (default: 10)
            eomg: Orientation tolerance in radians (default: 2e-3 = 2mrad)
            ev: Position tolerance in meters (default: 2e-3 = 2mm)
            max_iterations: Max iterations per attempt (default: 5000)
            verbose: Print detailed progress (default: False)

        Returns:
            Tuple of (theta, success, total_iterations, winning_strategy)
        """
        from . import ik_helpers

        n_joints = len(self.joint_limits)

        # Strategy configurations: (name, damping, step_cap)
        strategies = [
            ("workspace_heuristic", 0.02, 0.3),
            ("midpoint", 0.02, 0.3),
            ("workspace_heuristic", 0.01, 0.4),
            ("random", 0.02, 0.3),
            ("random", 0.03, 0.25),
            ("midpoint", 0.01, 0.4),
            ("random", 0.015, 0.35),
            ("random", 0.025, 0.3),
            ("workspace_heuristic", 0.03, 0.25),
            ("random", 0.02, 0.35),
        ]

        best_theta = None
        best_error = float('inf')
        total_iterations = 0
        winning_strategy = "none"

        for attempt in range(min(max_attempts, len(strategies))):
            strategy_name, damping, step_cap = strategies[attempt]

            if verbose:
                print(f"Attempt {attempt + 1}/{max_attempts}: {strategy_name}, "
                      f"damping={damping}, step_cap={step_cap}")

            # Generate initial guess
            if strategy_name == "workspace_heuristic":
                theta0 = ik_helpers.workspace_heuristic_guess(
                    T_desired, n_joints, self.joint_limits
                )
            elif strategy_name == "midpoint":
                theta0 = ik_helpers.midpoint_of_limits(self.joint_limits)
            else:  # random
                theta0 = ik_helpers.random_in_limits(self.joint_limits)

            try:
                theta, success, iters = self.iterative_inverse_kinematics(
                    T_desired, theta0, eomg, ev, max_iterations,
                    damping=damping, step_cap=step_cap,
                    adaptive_tuning=True, backtracking=True
                )
                total_iterations += iters

                if success:
                    if verbose:
                        print(f"  ✓ SUCCESS in {iters} iterations")
                    return theta, True, total_iterations, strategy_name

                # Evaluate error for tracking best
                T_curr = self.forward_kinematics(theta, frame="space")
                pos_err = np.linalg.norm(T_curr[:3, 3] - T_desired[:3, 3])
                R_err = T_curr[:3, :3].T @ T_desired[:3, :3]
                rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                error = pos_err + rot_err

                if verbose:
                    print(f"  ✗ Failed (pos_err={pos_err*1000:.2f}mm, rot_err={np.degrees(rot_err):.2f}°)")

                if error < best_error:
                    best_error = error
                    best_theta = theta.copy()
                    winning_strategy = strategy_name

            except Exception as e:
                if verbose:
                    print(f"  ✗ Exception: {e}")
                continue

        # Return best solution found
        if best_theta is None:
            best_theta = ik_helpers.midpoint_of_limits(self.joint_limits)

        return best_theta, False, total_iterations, winning_strategy

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

    def trac_ik(
        self,
        T_desired: NDArray[np.float64],
        theta0: Optional[Union[NDArray[np.float64], List[float]]] = None,
        timeout: float = 0.05,
        eomg: float = 1e-4,
        ev: float = 1e-4,
        num_restarts: int = 3
    ) -> Tuple[NDArray[np.float64], bool, float]:
        """
        TRAC-IK style inverse kinematics solver.

        Runs DLS and SQP solvers in parallel with random restarts.
        Returns the first successful solution. Achieves 95-99% success
        rate with 1-10ms typical solve times.

        This is the FASTEST and MOST RELIABLE IK method available.

        Args:
            T_desired: Target 4x4 transformation matrix
            theta0: Initial guess (optional, uses heuristic if None)
            timeout: Maximum solve time in seconds (default: 50ms)
            eomg: Orientation tolerance in radians (default: 1e-4)
            ev: Position tolerance in meters (default: 1e-4)
            num_restarts: Number of random restarts per solver (default: 3)

        Returns:
            Tuple of (theta, success, solve_time)
            - theta: Joint configuration (best found if not successful)
            - success: True if solution within tolerances
            - solve_time: Actual solve time in seconds

        Performance Comparison:
            | Method      | Success | Time    | Use Case           |
            |-------------|---------|---------|-------------------|
            | trac_ik     | 95-99%  | 1-10ms  | Real-time, best   |
            | robust_ik   | 50-80%  | 150ms   | Reliable, slower  |
            | iterative_ik| 35%     | 25ms    | Basic             |

        Example:
            >>> # Basic usage
            >>> theta, success, time = robot.trac_ik(T_target)
            >>> print(f"Solved: {success} in {time*1000:.1f}ms")
            >>>
            >>> # With custom parameters
            >>> theta, success, time = robot.trac_ik(
            ...     T_target,
            ...     timeout=0.1,      # 100ms timeout
            ...     num_restarts=5    # More exploration
            ... )
            >>>
            >>> # For real-time control (tight timeout)
            >>> theta, success, time = robot.trac_ik(
            ...     T_target,
            ...     theta0=current_angles,  # Warm start
            ...     timeout=0.005,          # 5ms for 200Hz control
            ...     num_restarts=1
            ... )
        """
        from .trac_ik import TracIKSolver

        # Create solver (could cache this for repeated calls)
        solver = TracIKSolver(
            fk_func=lambda th: self.forward_kinematics(th, frame="space"),
            jacobian_func=lambda th: self.jacobian(th, frame="space"),
            joint_limits=self.joint_limits,
            n_joints=len(self.joint_limits)
        )

        # Convert theta0 if provided
        if theta0 is not None:
            theta0 = np.array(theta0, dtype=float)

        return solver.solve(T_desired, theta0, timeout, eomg, ev, num_restarts)
