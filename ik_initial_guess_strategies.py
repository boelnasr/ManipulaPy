#!/usr/bin/env python3
"""
IK Initial Guess Improvement Strategies

This module provides various strategies to generate better initial guesses
for inverse kinematics, improving convergence rate and success probability.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Callable
from numpy.typing import NDArray


class IKInitialGuessGenerator:
    """
    Generate smart initial guesses for inverse kinematics.

    Strategies:
    1. Workspace-based heuristic (geometric approximation)
    2. Current configuration (if robot is moving continuously)
    3. Multiple random restarts
    4. Cached solutions lookup
    5. Analytical approximation (robot-specific)
    """

    def __init__(self, manipulator):
        """
        Initialize the generator with a manipulator instance.

        Args:
            manipulator: SerialManipulator instance
        """
        self.manipulator = manipulator
        self.solution_cache = []  # Store recent successful solutions
        self.cache_size = 100

    def workspace_heuristic(
        self,
        T_desired: NDArray[np.float64],
        use_position_only: bool = True
    ) -> NDArray[np.float64]:
        """
        Generate initial guess based on geometric workspace heuristic.

        For most manipulators, the first 3 joints control position and
        the last 3 control orientation. This provides a rough estimate.

        Args:
            T_desired: Desired 4x4 transformation matrix
            use_position_only: If True, focus on position, set orientation joints to 0

        Returns:
            Initial guess for joint angles
        """
        n_joints = len(self.manipulator.joint_limits)
        theta_guess = np.zeros(n_joints)

        # Extract desired position
        p_desired = T_desired[:3, 3]

        # Simple heuristic: estimate first joint from XY plane angle
        if n_joints >= 1:
            theta_guess[0] = np.arctan2(p_desired[1], p_desired[0])

        # Estimate second joint from elevation angle (rough approximation)
        if n_joints >= 2:
            r_xy = np.sqrt(p_desired[0]**2 + p_desired[1]**2)
            theta_guess[1] = np.arctan2(p_desired[2], r_xy)

        # Third joint: rough elbow angle based on distance
        if n_joints >= 3:
            distance = np.linalg.norm(p_desired)
            # Assume a typical elbow configuration
            theta_guess[2] = np.pi / 4  # 45 degrees as neutral position

        # Orientation joints (wrist): start at neutral if position-only mode
        if not use_position_only and n_joints > 3:
            # Extract rotation matrix
            R_desired = T_desired[:3, :3]
            # Estimate wrist angles from rotation (simplified)
            theta_guess[3:] = self._estimate_wrist_angles(R_desired)

        # Ensure within joint limits
        theta_guess = self._clip_to_limits(theta_guess)

        return theta_guess

    def _estimate_wrist_angles(self, R: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Estimate wrist joint angles from rotation matrix.
        Uses ZYZ Euler angle decomposition as approximation.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Array of wrist angles (typically 3 values)
        """
        # ZYZ Euler angles
        if np.abs(R[2, 2]) < 0.9999:
            theta_y = np.arccos(np.clip(R[2, 2], -1, 1))
            theta_z1 = np.arctan2(R[1, 2], R[0, 2])
            theta_z2 = np.arctan2(R[2, 1], -R[2, 0])
        else:
            # Gimbal lock case
            theta_y = 0
            theta_z1 = 0
            theta_z2 = np.arctan2(R[1, 0], R[0, 0])

        return np.array([theta_z1, theta_y, theta_z2])

    def multiple_random_restarts(
        self,
        T_desired: NDArray[np.float64],
        n_attempts: int = 5,
        ik_func: Optional[Callable] = None,
        **ik_kwargs
    ) -> Tuple[NDArray[np.float64], bool, List[Tuple]]:
        """
        Try multiple random initial guesses and return the best result.

        Args:
            T_desired: Desired transformation
            n_attempts: Number of random restarts to try
            ik_func: IK function to call (default: manipulator.iterative_inverse_kinematics)
            **ik_kwargs: Additional arguments to pass to IK function

        Returns:
            (best_solution, success, all_results)
        """
        if ik_func is None:
            ik_func = self.manipulator.iterative_inverse_kinematics

        results = []
        best_solution = None
        best_error = float('inf')

        for i in range(n_attempts):
            if i == 0:
                # First attempt: use workspace heuristic
                theta0 = self.workspace_heuristic(T_desired)
            else:
                # Subsequent attempts: random within joint limits
                theta0 = self._random_configuration()

            # Run IK
            theta, success, iterations = ik_func(T_desired, theta0, **ik_kwargs)

            # Evaluate quality of solution
            T_result = self.manipulator.forward_kinematics(theta)
            error = self._pose_error(T_desired, T_result)

            results.append((theta, success, iterations, error))

            if error < best_error:
                best_error = error
                best_solution = theta

        # Return best solution
        final_success = best_error < (ik_kwargs.get('ev', 1e-6) + ik_kwargs.get('eomg', 1e-6))
        return best_solution, final_success, results

    def cached_nearest_neighbor(
        self,
        T_desired: NDArray[np.float64],
        k: int = 3
    ) -> NDArray[np.float64]:
        """
        Use nearest neighbor from solution cache as initial guess.

        Maintains a cache of recent successful IK solutions. When a new
        IK problem arrives, find the k nearest cached solutions (by pose
        distance) and use their average as initial guess.

        Args:
            T_desired: Desired transformation
            k: Number of nearest neighbors to consider

        Returns:
            Initial guess based on cached solutions
        """
        if len(self.solution_cache) == 0:
            # No cache yet, use workspace heuristic
            return self.workspace_heuristic(T_desired)

        # Find k nearest neighbors
        distances = []
        for cached_T, cached_theta in self.solution_cache:
            dist = self._pose_error(T_desired, cached_T)
            distances.append((dist, cached_theta))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Take k nearest
        k_nearest = distances[:min(k, len(distances))]

        # Average the joint angles
        theta_avg = np.mean([theta for _, theta in k_nearest], axis=0)

        # Clip to limits
        theta_avg = self._clip_to_limits(theta_avg)

        return theta_avg

    def add_to_cache(self, T: NDArray[np.float64], theta: NDArray[np.float64]):
        """
        Add a successful IK solution to the cache.

        Args:
            T: Transformation matrix
            theta: Corresponding joint angles
        """
        self.solution_cache.append((T.copy(), theta.copy()))

        # Limit cache size (FIFO)
        if len(self.solution_cache) > self.cache_size:
            self.solution_cache.pop(0)

    def current_configuration_extrapolation(
        self,
        theta_current: NDArray[np.float64],
        T_current: NDArray[np.float64],
        T_desired: NDArray[np.float64],
        alpha: float = 0.5
    ) -> NDArray[np.float64]:
        """
        Extrapolate from current configuration for smooth trajectories.

        When the robot is moving continuously, the best initial guess
        is often close to the current configuration. This method
        extrapolates based on the motion trend.

        Args:
            theta_current: Current joint angles
            T_current: Current end-effector pose
            T_desired: Desired end-effector pose
            alpha: Extrapolation factor (0 = current, 1 = full extrapolation)

        Returns:
            Extrapolated initial guess
        """
        # Compute pose error
        T_err = T_desired @ np.linalg.inv(T_current)

        # Extract twist
        from ManipulaPy import utils
        V_err = utils.se3ToVec(utils.MatrixLog6(T_err))

        # Compute joint velocity estimate using Jacobian
        J = self.manipulator.jacobian(theta_current)
        dtheta = np.linalg.pinv(J) @ V_err

        # Extrapolate
        theta_guess = theta_current + alpha * dtheta

        # Clip to limits
        theta_guess = self._clip_to_limits(theta_guess)

        return theta_guess

    def analytical_guess_6dof_spherical_wrist(
        self,
        T_desired: NDArray[np.float64],
        d1: float,
        a2: float,
        a3: float
    ) -> NDArray[np.float64]:
        """
        Analytical initial guess for 6-DOF robots with spherical wrist.

        Many industrial robots (like PUMA, UR5) have spherical wrists.
        This uses geometric decoupling: position (joints 1-3) and
        orientation (joints 4-6).

        Args:
            T_desired: Desired 4x4 transformation
            d1: Base height (link 1 offset in Z)
            a2: Length of link 2
            a3: Length of link 3

        Returns:
            Analytical initial guess for 6 joints
        """
        theta = np.zeros(6)

        # Extract desired position and orientation
        p_desired = T_desired[:3, 3]
        R_desired = T_desired[:3, :3]

        # --- Position IK (joints 1-3) using geometric approach ---

        # Joint 1: rotation about Z axis
        theta[0] = np.arctan2(p_desired[1], p_desired[0])

        # Wrist center position (subtract wrist offset)
        # Assuming wrist is d6 units along the approach vector
        # For simplicity, we'll work with end-effector position directly
        r = np.sqrt(p_desired[0]**2 + p_desired[1]**2)
        s = p_desired[2] - d1

        # Distance to wrist center in XY plane
        D = np.sqrt(r**2 + s**2)

        # Law of cosines for elbow (joint 3)
        cos_theta3 = (D**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cos_theta3 = np.clip(cos_theta3, -1, 1)  # Ensure valid range
        theta[2] = np.arccos(cos_theta3)  # Elbow-up solution

        # Joint 2: shoulder angle
        alpha = np.arctan2(s, r)
        beta = np.arctan2(a3 * np.sin(theta[2]), a2 + a3 * np.cos(theta[2]))
        theta[1] = alpha - beta

        # --- Orientation IK (joints 4-6) using wrist decoupling ---

        # Compute R_0_3 (rotation from base to wrist)
        # This requires forward kinematics of first 3 joints
        # For initial guess, we'll use simplified estimation

        # ZYZ Euler angles for wrist orientation
        wrist_angles = self._estimate_wrist_angles(R_desired)
        theta[3:6] = wrist_angles

        # Clip to limits
        theta = self._clip_to_limits(theta)

        return theta

    # ========== Helper Methods ==========

    def _random_configuration(self) -> NDArray[np.float64]:
        """Generate random joint configuration within limits."""
        n_joints = len(self.manipulator.joint_limits)
        theta = np.zeros(n_joints)

        for i, (mn, mx) in enumerate(self.manipulator.joint_limits):
            if mn is not None and mx is not None:
                theta[i] = np.random.uniform(mn, mx)
            elif mn is not None:
                theta[i] = mn + np.random.uniform(0, np.pi)
            elif mx is not None:
                theta[i] = mx - np.random.uniform(0, np.pi)
            else:
                theta[i] = np.random.uniform(-np.pi, np.pi)

        return theta

    def _clip_to_limits(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip joint angles to their limits."""
        theta_clipped = theta.copy()
        for i, (mn, mx) in enumerate(self.manipulator.joint_limits):
            if mn is not None:
                theta_clipped[i] = max(theta_clipped[i], mn)
            if mx is not None:
                theta_clipped[i] = min(theta_clipped[i], mx)
        return theta_clipped

    def _pose_error(
        self,
        T1: NDArray[np.float64],
        T2: NDArray[np.float64]
    ) -> float:
        """
        Compute pose error between two transformations.

        Args:
            T1, T2: 4x4 transformation matrices

        Returns:
            Scalar error (combined position and orientation error)
        """
        # Position error
        p_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])

        # Orientation error (Frobenius norm of rotation difference)
        R_err = np.linalg.norm(T1[:3, :3] - T2[:3, :3], 'fro')

        # Combined error (weighted)
        return p_err + 0.1 * R_err


# ========== Usage Examples ==========

def example_usage():
    """
    Example demonstrating how to use the IK initial guess strategies.
    """
    from ManipulaPy.kinematics import SerialManipulator
    import numpy as np

    # Create a manipulator (example with 6 DOF)
    # You'll need to provide actual DH parameters or screw parameters
    M = np.eye(4)
    M[:3, 3] = [0.5, 0, 0.5]  # Example end-effector position at home

    omega_list = np.array([
        [0, 0, 1],  # Joint 1: rotation about Z
        [0, 1, 0],  # Joint 2: rotation about Y
        [0, 1, 0],  # Joint 3: rotation about Y
        [1, 0, 0],  # Joint 4: rotation about X
        [0, 1, 0],  # Joint 5: rotation about Y
        [1, 0, 0],  # Joint 6: rotation about X
    ]).T

    r_list = np.array([
        [0, 0, 0],
        [0, 0, 0.1],
        [0, 0, 0.3],
        [0, 0, 0.5],
        [0, 0, 0.5],
        [0, 0, 0.5],
    ]).T

    joint_limits = [
        (-np.pi, np.pi),
        (-np.pi/2, np.pi/2),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi/2, np.pi/2),
        (-np.pi, np.pi),
    ]

    manipulator = SerialManipulator(
        M_list=M,
        omega_list=omega_list,
        r_list=r_list,
        joint_limits=joint_limits
    )

    # Create initial guess generator
    ik_guesser = IKInitialGuessGenerator(manipulator)

    # Desired pose
    T_desired = np.eye(4)
    T_desired[:3, 3] = [0.3, 0.2, 0.4]  # Target position

    print("=== IK Initial Guess Strategies ===\n")

    # Strategy 1: Workspace heuristic
    print("1. Workspace Heuristic:")
    theta_ws = ik_guesser.workspace_heuristic(T_desired)
    print(f"   Initial guess: {np.round(theta_ws, 3)}")

    # Strategy 2: Multiple random restarts
    print("\n2. Multiple Random Restarts:")
    theta_best, success, results = ik_guesser.multiple_random_restarts(
        T_desired,
        n_attempts=3,
        max_iterations=100
    )
    print(f"   Best solution: {np.round(theta_best, 3)}")
    print(f"   Success: {success}")
    print(f"   Errors from all attempts: {[r[3] for r in results]}")

    # Strategy 3: Cache-based (after adding solutions)
    print("\n3. Cached Nearest Neighbor:")
    # Add some solutions to cache
    ik_guesser.add_to_cache(T_desired, theta_best)
    theta_cached = ik_guesser.cached_nearest_neighbor(T_desired)
    print(f"   Cache-based guess: {np.round(theta_cached, 3)}")

    # Strategy 4: Extrapolation (for trajectory tracking)
    print("\n4. Current Configuration Extrapolation:")
    theta_current = np.array([0, 0, 0, 0, 0, 0])
    T_current = manipulator.forward_kinematics(theta_current)
    theta_extrap = ik_guesser.current_configuration_extrapolation(
        theta_current, T_current, T_desired
    )
    print(f"   Extrapolated guess: {np.round(theta_extrap, 3)}")


if __name__ == "__main__":
    example_usage()
