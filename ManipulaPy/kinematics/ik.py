#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Kinematics Module - ManipulaPy

This module provides classes and functions for performing kinematic analysis
and computations for serial manipulators, including forward and inverse
kinematics, Jacobian calculations, and end-effector velocity calculations.

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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import serial_manipulator as _runtime


class _InverseKinematicsConcern:
    def iterative_inverse_kinematics(
        self,
        T_desired: NDArray[np.float64],
        thetalist0: Union[NDArray[np.float64], List[float]],
        eomg: float = 1e-6,
        ev: float = 1e-6,
        max_iterations: int = 10000,
        plot_residuals: bool = False,
        # DLS lambda: 2e-2 for 6-DOF, 1e-2 for 2-DOF.
        damping: float = 2e-2,
        # Max step norm (rad): 0.3 for 6-DOF, 0.1 for 2-DOF.
        step_cap: float = 0.3,
        png_name: str = "ik_residuals.png",
        weight_orientation: float = 1.0,  # scale for rotational error in solve step
        weight_position: float = 1.0,  # scale for translational error in solve step
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
        backend = _runtime.get_backend()
        theta = backend.asarray(thetalist0, dtype=backend.float64)
        T_desired = backend.asarray(T_desired, dtype=backend.float64)
        residuals = []
        damping_local = damping
        step_cap_local = step_cap
        prev_error = float("inf")
        min_damping, max_damping = 1e-6, 5e-1
        min_step_cap = 0.01
        nu = 2.0  # LM damping adjustment factor

        # Best solution tracking. ``backend.array`` copies on every backend;
        # ``.copy()`` is NumPy-only (Torch tensors expose ``.clone()``).
        best_theta = backend.array(theta)
        best_error = float("inf")

        # Stagnation detection
        stall_count = 0
        max_stall = 20

        def compute_geometric_error(
            T_curr: NDArray[np.float64], T_target: NDArray[np.float64]
        ) -> Tuple[NDArray[np.float64], float, float]:
            """Compute geometric error without adjoint amplification."""
            # Position error
            pos_err = T_target[:3, 3] - T_curr[:3, 3]
            trans_err = backend.norm(pos_err)

            # Rotation error using axis-angle
            R_curr = T_curr[:3, :3]
            R_target = T_target[:3, :3]
            R_err = backend.matmul(R_curr.T, R_target)

            trace_val = backend.clip((backend.trace(R_err) - 1) / 2, -1, 1)
            angle = backend.arccos(trace_val)
            rot_err = backend.abs(angle)

            # Extract rotation axis
            if angle < 1e-6:
                omega_err = (
                    backend.stack(
                        (
                            R_err[2, 1] - R_err[1, 2],
                            R_err[0, 2] - R_err[2, 0],
                            R_err[1, 0] - R_err[0, 1],
                        )
                    )
                    / 2
                )
            elif abs(angle - _runtime.np.pi) < 1e-6:
                diag = backend.diag(R_err)
                idx = backend.argmax(diag)
                axis = backend.zeros(3, dtype=R_err.dtype)
                axis[idx] = 1.0
                omega_err = angle * axis
            else:
                axis = backend.stack(
                    (
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    )
                ) / (2 * backend.sin(angle) + 1e-10)
                omega_err = angle * axis

            # Transform to space frame
            omega_err_space = backend.matmul(R_curr, omega_err)

            # 6D error [angular, linear]
            V_err = backend.concatenate((omega_err_space, pos_err))
            return V_err, rot_err, trans_err

        def svd_robust_solve(
            J: NDArray[np.float64],
            V_err: NDArray[np.float64],
            damping_val: float,
        ) -> NDArray[np.float64]:
            """SVD-based damped least squares for near-singular Jacobians."""
            try:
                U, s, Vt = backend.svd(J, full_matrices=False)
                # Damped pseudo-inverse: σ / (σ² + λ²)
                s_damped = s / (s**2 + damping_val**2 + 1e-12)
                return backend.matmul(Vt.T, s_damped * backend.matmul(U.T, V_err))
            except Exception as exc:
                # CuPy exposes its own LinAlgError class rather than NumPy's.
                if not isinstance(exc, _runtime.np.linalg.LinAlgError) and (
                    exc.__class__.__name__ != "LinAlgError"
                ):
                    raise
                # Fallback to standard solve
                JTJ = backend.matmul(J.T, J)
                lambda_I = (damping_val**2) * backend.eye(JTJ.shape[0])
                return backend.solve(JTJ + lambda_I, backend.matmul(J.T, V_err))

        def clip_to_limits(th: NDArray[np.float64]) -> NDArray[np.float64]:
            """Clip joint angles to limits."""
            th_clipped = backend.array(th)
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
                best_theta = backend.array(theta)
                stall_count = 0
            else:
                stall_count += 1

            # Stagnation recovery: perturb if stuck
            if stall_count > max_stall:
                # Add small random perturbation to escape local minimum
                perturbation = 0.1 * _runtime.np.random.randn(len(theta))
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
            V_weighted = backend.array(V_err)
            V_weighted[:3] *= weight_orientation
            V_weighted[3:] *= weight_position

            # SVD-robust solve
            delta_theta = svd_robust_solve(J_space, V_weighted, damping_local)

            # Cap step size
            norm_delta = backend.norm(delta_theta)
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
            k += 1  # max_iterations reached

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

            it = _runtime.np.arange(len(residuals))
            tr, rt = zip(
                *(
                    (float(backend.to_numpy(t)), float(backend.to_numpy(r)))
                    for t, r in residuals
                )
            )
            plt.plot(it, tr, label="Translation error")
            plt.plot(it, rt, label="Rotation error")
            plt.xlabel("Iteration")
            plt.ylabel("Norm")
            plt.title("IK convergence")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(png_name, dpi=400)
            plt.close()
            print(f"Residual plot saved to {png_name}")

        return theta, success, k + 1

    @staticmethod
    def _pose_error(
        T_curr: NDArray[np.float64], T_desired: NDArray[np.float64]
    ) -> float:
        """Compute combined position + orientation error between two poses."""
        backend = _runtime.get_backend()
        T_curr = backend.to_numpy(T_curr)
        T_desired = backend.to_numpy(T_desired)
        pos_err = _runtime.np.linalg.norm(T_curr[:3, 3] - T_desired[:3, 3])
        R_err = T_curr[:3, :3].T @ T_desired[:3, :3]
        rot_err = _runtime.np.arccos(
            _runtime.np.clip((_runtime.np.trace(R_err) - 1) / 2, -1, 1)
        )
        return pos_err + rot_err

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
            auto_fallback: Try other strategies if the primary strategy fails.
            Other args same as iterative_inverse_kinematics()

        Returns:
            Tuple of (theta, success, iterations)
        """
        from .. import ik_helpers

        n_joints = len(self.joint_limits)

        valid_strategies = [
            "workspace_heuristic",
            "extrapolate",
            "cached",
            "random",
            "midpoint",
        ]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {valid_strategies}"
            )

        def get_initial_guess(strat: str) -> Optional[NDArray[np.float64]]:
            """Generate initial guess for given strategy."""
            if strat == "workspace_heuristic":
                return ik_helpers.workspace_heuristic_guess(
                    T_desired, n_joints, self.joint_limits
                )
            elif strat == "extrapolate":
                if theta_current is None or T_current is None:
                    return None
                return ik_helpers.extrapolate_from_current(
                    theta_current,
                    T_current,
                    T_desired,
                    lambda th: self.jacobian(th, frame="space"),
                    self.joint_limits,
                    alpha=0.5,
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

        def try_ik(
            theta0: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], bool, int]:
            """Try IK with given initial guess."""
            return self.iterative_inverse_kinematics(
                T_desired,
                theta0,
                eomg,
                ev,
                max_iterations,
                plot_residuals,
                damping,
                step_cap,
                png_name,
                weight_orientation,
                weight_position,
                adaptive_tuning,
                backtracking,
            )

        # Primary strategy
        theta0 = get_initial_guess(strategy)
        if theta0 is None:
            theta0 = ik_helpers.workspace_heuristic_guess(
                T_desired, n_joints, self.joint_limits
            )

        theta, success, iters = try_ik(theta0)

        if success or not auto_fallback:
            return theta, success, iters

        # Fallback strategies if primary failed
        fallback_strategies = ["midpoint", "random", "random", "random"]
        total_iters = iters
        best_theta = theta
        best_error = float("inf")

        # Evaluate initial result
        T_curr = self.forward_kinematics(theta, frame="space")
        best_error = self._pose_error(T_curr, T_desired)

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
            error = self._pose_error(T_curr, T_desired)

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
        verbose: bool = False,
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
        from .. import ik_helpers

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
        best_error = float("inf")
        total_iterations = 0
        winning_strategy = "none"

        for attempt in range(min(max_attempts, len(strategies))):
            strategy_name, damping, step_cap = strategies[attempt]

            if verbose:
                print(
                    f"Attempt {attempt + 1}/{max_attempts}: {strategy_name}, "
                    f"damping={damping}, step_cap={step_cap}"
                )

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
                    T_desired,
                    theta0,
                    eomg,
                    ev,
                    max_iterations,
                    damping=damping,
                    step_cap=step_cap,
                    adaptive_tuning=True,
                    backtracking=True,
                )
                total_iterations += iters

                if success:
                    if verbose:
                        print(f"  ✓ SUCCESS in {iters} iterations")
                    return theta, True, total_iterations, strategy_name

                # Evaluate error for tracking best
                T_curr = self.forward_kinematics(theta, frame="space")
                backend = _runtime.get_backend()
                T_curr_host = backend.to_numpy(T_curr)
                T_desired_host = backend.to_numpy(T_desired)
                pos_err = _runtime.np.linalg.norm(
                    T_curr_host[:3, 3] - T_desired_host[:3, 3]
                )
                R_err = T_curr_host[:3, :3].T @ T_desired_host[:3, :3]
                rot_err = _runtime.np.arccos(
                    _runtime.np.clip((_runtime.np.trace(R_err) - 1) / 2, -1, 1)
                )
                error = pos_err + rot_err

                if verbose:
                    print(
                        f"  ✗ Failed (pos_err={pos_err*1000:.2f}mm, "
                        f"rot_err={_runtime.np.degrees(rot_err):.2f}°)"
                    )

                if error < best_error:
                    best_error = error
                    best_theta = backend.array(theta)
                    winning_strategy = strategy_name

            except Exception as e:
                if verbose:
                    print(f"  ✗ Exception: {e}")
                continue

        # Return best solution found
        if best_theta is None:
            best_theta = ik_helpers.midpoint_of_limits(self.joint_limits)

        return best_theta, False, total_iterations, winning_strategy

    def trac_ik(
        self,
        T_desired: NDArray[np.float64],
        theta0: Optional[Union[NDArray[np.float64], List[float]]] = None,
        timeout: float = 0.2,
        eomg: float = 1e-4,
        ev: float = 1e-4,
        num_restarts: int = 5,
        use_parallel: bool = False,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        """
        TRAC-IK style inverse kinematics solver.

        Uses a DLS-first strategy with SQP fallback and diverse initial guesses.
        Sequential mode (default) avoids Python GIL contention for best results.

        Args:
            T_desired: Target 4x4 transformation matrix
            theta0: Initial guess (optional, uses heuristic if None)
            timeout: Maximum total solve time in seconds (default: 200ms)
            eomg: Orientation tolerance in radians (default: 1e-4)
            ev: Position tolerance in meters (default: 1e-4)
            num_restarts: Number of initial guesses (default: 5)
            use_parallel: Run DLS+SQP in parallel per guess (default: False)

        Returns:
            Tuple of (theta, success, solve_time)
            - theta: Joint configuration (best found if not successful)
            - success: True if solution within tolerances
            - solve_time: Actual solve time in seconds

        Example:
            >>> # Basic usage
            >>> theta, success, time = robot.trac_ik(T_target)
            >>> print(f"Solved: {success} in {time*1000:.1f}ms")
            >>>
            >>> # For real-time control (tight timeout)
            >>> theta, success, time = robot.trac_ik(
            ...     T_target,
            ...     theta0=current_angles,  # Warm start
            ...     timeout=0.01,           # 10ms for 100Hz control
            ...     num_restarts=2
            ... )
        """
        from ..trac_ik import trac_ik_solve

        if theta0 is not None:
            theta0 = _runtime.np.array(theta0, dtype=float)

        return trac_ik_solve(
            self, T_desired, theta0, timeout, eomg, ev, num_restarts, use_parallel
        )
