#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
TRAC-IK Style Solver - ManipulaPy

A high-performance IK solver inspired by TRAC-IK that runs multiple algorithms
in parallel and returns the first successful result.

Key Features:
- Dual solver: DLS (Damped Least-Squares) + SQP (Sequential Quadratic Programming)
- Levenberg-Marquardt style adaptive damping with trust region
- SVD-robust Jacobian solve (primary path, not fallback)
- Stagnation detection with perturbation recovery
- Backtracking line search for step acceptance
- Parallel execution using threading
- Timeout-based termination (not iteration count)
- Diverse initial guesses with random restarts

Algorithm:
    ┌─────────────────────────────────────────────┐
    │                 TRAC-IK                      │
    │  ┌─────────────┐     ┌─────────────┐        │
    │  │    DLS      │     │    SQP      │        │
    │  │ (Levenberg- │     │ (Constrained│        │
    │  │  Marquardt) │     │  Optimizer) │        │
    │  └──────┬──────┘     └──────┬──────┘        │
    │         │                   │               │
    │         └───── First ───────┘               │
    │               Success                       │
    └─────────────────────────────────────────────┘

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..backend import get_backend

ErrorFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], float, float],
]


class TracIKSolver:
    """
    TRAC-IK style solver combining DLS and SQP algorithms.

    Runs multiple IK algorithms in parallel and returns the first successful
    solution. Achieves high success rates through diverse initial guesses,
    SVD-robust Jacobian solving, and perturbation-based stagnation recovery.

    Example:
        >>> from ManipulaPy.trac_ik import TracIKSolver
        >>>
        >>> # Create solver
        >>> trac_ik = TracIKSolver(
        ...     fk_func=robot.forward_kinematics,
        ...     jacobian_func=robot.jacobian,
        ...     joint_limits=robot.joint_limits,
        ...     n_joints=6
        ... )
        >>>
        >>> # Solve IK
        >>> theta, success, solve_time = trac_ik.solve(T_desired)
        >>> if success:
        ...     print(f"Solution found in {solve_time*1000:.1f}ms")
    """

    def __init__(
        self,
        fk_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        jacobian_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        joint_limits: List[Tuple[Optional[float], Optional[float]]],
        n_joints: int,
        error_func: Optional[ErrorFunction] = None,
    ) -> None:
        """
        Initialize TRAC-IK solver.

        Args:
            fk_func: Forward kinematics function (theta -> T)
            jacobian_func: Jacobian function (theta -> J)
            joint_limits: List of (min, max) tuples for each joint
            n_joints: Number of joints
            error_func: Optional custom error function (T_current, T_desired) -> error_vector
        """
        self.fk_func = fk_func
        self.jacobian_func = jacobian_func
        self.joint_limits = joint_limits
        self.n_joints = n_joints
        self.error_func = error_func or self._default_error_func

        # Build bounds for SQP optimizer
        self.bounds = []
        for mn, mx in joint_limits:
            lb = mn if mn is not None else -2 * np.pi
            ub = mx if mx is not None else 2 * np.pi
            self.bounds.append((lb, ub))

    def solve(
        self,
        T_desired: NDArray[np.float64],
        theta0: Optional[NDArray[np.float64]] = None,
        timeout: float = 0.2,  # 200ms default
        eomg: float = 1e-4,
        ev: float = 1e-4,
        num_restarts: int = 5,
        use_parallel: bool = False,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        """
        Solve IK using TRAC-IK algorithm.

        Uses a DLS-first strategy with SQP fallback. Supports both sequential
        (default) and parallel execution modes.

        Sequential mode (default, recommended): Tries DLS on each initial guess
        with time budgeting, then falls back to SQP if needed. Avoids Python GIL
        contention that limits threaded performance for CPU-bound NumPy work.

        Parallel mode: Runs DLS + SQP simultaneously on each guess using 2
        worker threads. Useful when one solver may find a solution the other
        cannot, but subject to GIL overhead.

        Args:
            T_desired: Target 4x4 transformation matrix
            theta0: Initial guess (optional, uses heuristic if None)
            timeout: Maximum total solve time in seconds (default: 200ms)
            eomg: Orientation tolerance in radians
            ev: Position tolerance in meters
            num_restarts: Number of initial guesses (default: 5)
            use_parallel: Run DLS+SQP in parallel per guess (default: False)

        Returns:
            Tuple of (theta, success, solve_time)
            - theta: Joint configuration (best found if not successful)
            - success: True if solution within tolerances
            - solve_time: Actual solve time in seconds
        """
        start_time = time.perf_counter()

        # Generate diverse initial guesses
        initial_guesses = self._generate_initial_guesses(
            T_desired, theta0, num_restarts
        )

        # Result storage (thread-safe)
        result_lock = threading.Lock()
        best_result = {"theta": None, "success": False, "error": float("inf")}
        stop_event = threading.Event()

        def update_result(
            theta: NDArray[np.float64], success: bool, error: float
        ) -> None:
            """Thread-safe result update with error-based selection."""
            with result_lock:
                if success:
                    if not best_result["success"] or error < best_result["error"]:
                        best_result["theta"] = theta
                        best_result["success"] = True
                        best_result["error"] = error
                        stop_event.set()
                elif error < best_result["error"] and not best_result["success"]:
                    best_result["theta"] = theta
                    best_result["error"] = error

        def _remaining() -> float:
            """Return remaining solve budget in seconds."""
            return max(0, timeout - (time.perf_counter() - start_time))

        if use_parallel:
            # Parallel: submit all DLS tasks with limited workers to reduce
            # GIL contention. DLS tasks self-terminate via perturbation limit,
            # so they won't burn the full timeout on hopeless guesses.
            # SQP runs as fallback if DLS fails on all guesses.
            dls_timeout = timeout * 0.8

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Phase 1: Submit all DLS tasks (more perturbations since
                # all guesses run concurrently — no time-sharing penalty)
                dls_futures = []
                for guess in initial_guesses:
                    dls_futures.append(
                        executor.submit(
                            self._dls_solver,
                            T_desired,
                            guess,
                            eomg,
                            ev,
                            dls_timeout,
                            stop_event,
                            5,  # max_perturbations (higher for parallel)
                        )
                    )

                # Collect DLS results as they complete
                try:
                    for future in as_completed(dls_futures, timeout=dls_timeout + 0.05):
                        try:
                            theta, success, error = future.result(timeout=0.05)
                            update_result(theta, success, error)
                            if best_result["success"]:
                                stop_event.set()
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

                # Phase 2: SQP fallback if DLS didn't solve it
                if not best_result["success"] and _remaining() > 0.01:
                    sqp_start = (
                        best_result["theta"]
                        if best_result["theta"] is not None
                        else initial_guesses[0]
                    )
                    try:
                        theta, success, error = self._sqp_solver(
                            T_desired,
                            sqp_start,
                            eomg,
                            ev,
                            _remaining(),
                            stop_event,
                        )
                        update_result(theta, success, error)
                    except Exception:
                        pass

                stop_event.set()
        else:
            # Sequential: DLS-first with time budgeting, SQP fallback.
            # Per-guess cap prevents one bad guess from burning all the time.
            # DLS also self-terminates after max_perturbations stall cycles.
            n_guesses = len(initial_guesses)
            max_per_guess = timeout * 0.8 / max(n_guesses - 1, 1)

            # Phase 1: Try DLS on each guess
            for guess in initial_guesses:
                if stop_event.is_set() or _remaining() < 0.005:
                    break

                dls_budget = min(max_per_guess, _remaining() * 0.9)
                theta, success, error = self._dls_solver(
                    T_desired,
                    guess,
                    eomg,
                    ev,
                    dls_budget,
                    stop_event,
                )
                update_result(theta, success, error)

            # Phase 2: SQP fallback using best theta found so far
            if not best_result["success"] and _remaining() > 0.01:
                sqp_start = (
                    best_result["theta"]
                    if best_result["theta"] is not None
                    else initial_guesses[0]
                )
                theta, success, error = self._sqp_solver(
                    T_desired, sqp_start, eomg, ev, _remaining(), stop_event
                )
                update_result(theta, success, error)

        solve_time = time.perf_counter() - start_time

        if best_result["theta"] is None:
            best_result["theta"] = initial_guesses[0]

        return best_result["theta"], best_result["success"], solve_time

    def _generate_initial_guesses(
        self,
        T_desired: NDArray[np.float64],
        theta0: Optional[NDArray[np.float64]],
        num_restarts: int,
    ) -> List[NDArray[np.float64]]:
        """Generate diverse initial guesses for broad workspace coverage."""
        backend = get_backend()
        guesses = []

        # 1. User-provided or workspace heuristic
        if theta0 is not None:
            guesses.append(backend.array(theta0))
        else:
            guesses.append(self._workspace_heuristic(T_desired))

        # 2. Midpoint of limits
        guesses.append(self._midpoint_guess())

        # 3. Zero configuration
        guesses.append(backend.zeros(self.n_joints, dtype=backend.float64))

        # 4. Flipped midpoint (negated, clipped)
        mid = self._midpoint_guess()
        guesses.append(self._clip_to_limits(-mid))

        # 5+. Random guesses to fill remaining slots
        for _ in range(max(0, num_restarts - 4)):
            guesses.append(self._random_guess())

        return guesses

    def _dls_solver(
        self,
        T_desired: NDArray[np.float64],
        theta0: NDArray[np.float64],
        eomg: float,
        ev: float,
        timeout: float,
        stop_event: threading.Event,
        max_perturbations: int = 3,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        """
        Damped Least-Squares solver with Levenberg-Marquardt adaptive damping.

        Uses SVD-robust Jacobian solve, perturbation-based stagnation recovery,
        and backtracking line search.

        Args:
            max_perturbations: Max perturbation recovery attempts before giving
                up on this initial guess. Lower = faster fail-over to next guess
                (good for sequential). Higher = more thorough search per guess
                (good for parallel where all guesses run concurrently).
        """
        backend = get_backend()
        # Preserve the seed dtype (float32 seeds stay float32); only promote
        # bool/int or non-backend array-likes to float64, matching kinematics.
        # Copy so the caller's seed array is never aliased.
        theta = backend.asarray(theta0)
        dtype_kind = getattr(theta.dtype, "kind", None)
        if not backend.is_backend_array(theta0) or dtype_kind in ("b", "i", "u"):
            theta = backend.asarray(theta0, dtype=backend.float64)
        theta = backend.array(theta)

        # Adaptive damping parameters (LM-style) — match kinematics.py
        damping = 0.02  # Initial damping
        min_damping = 1e-6
        max_damping = 0.5
        nu = 2.0

        # Step control
        step_cap = 0.3
        min_step_cap = 0.01

        # Algorithm parameters
        max_iters = 3000
        start_time = time.perf_counter()

        # State tracking
        best_theta = backend.array(theta)
        best_error = float("inf")
        error_history = deque(maxlen=10)
        prev_error = float("inf")

        # Stagnation detection
        stall_count = 0
        max_stall = 20
        perturbation_count = 0

        for iteration in range(max_iters):
            # Check termination conditions
            if stop_event.is_set():
                break
            if time.perf_counter() - start_time > timeout:
                break

            # Compute current error
            T_curr = self.fk_func(theta)
            V_err, rot_err, trans_err = self.error_func(T_curr, T_desired)
            # Normalize the (possibly injected) error callable's vector onto the
            # active backend so a NumPy-returning error_func keeps working.
            V_err = backend.asarray(V_err)
            # Convergence and stagnation are host decisions: pull the scalar
            # error norms to the host explicitly at this boundary.
            rot_err = float(rot_err)
            trans_err = float(trans_err)
            current_error = rot_err + trans_err

            # Check convergence
            if rot_err < eomg and trans_err < ev:
                return theta, True, current_error

            # Track best solution
            if current_error < best_error:
                best_error = current_error
                best_theta = backend.array(theta)
                stall_count = 0
            else:
                stall_count += 1

            # Perturbation recovery — escape local minima
            if stall_count > max_stall:
                perturbation_count += 1
                if perturbation_count > max_perturbations:
                    break  # This guess is stuck, let caller try next guess
                # np.random is the host RNG boundary; move the perturbation into
                # the backend domain before adding it to the backend-native seed
                # so a non-NumPy backend never mixes host and device arrays.
                perturbation = backend.asarray(0.1 * np.random.randn(self.n_joints))
                theta = self._clip_to_limits(best_theta + perturbation)
                damping = 0.02  # Reset damping
                stall_count = 0
                nu = 2.0
                continue

            error_history.append(current_error)

            # Oscillation detection — increase damping if oscillating
            is_stagnating, is_oscillating = self._detect_oscillation(error_history)
            if is_oscillating:
                damping = min(max_damping, damping * 4)
                step_cap = max(min_step_cap, step_cap * 0.5)

            # Get Jacobian (normalize the injected callable's output onto the
            # active backend so a NumPy-returning jacobian_func keeps working).
            J = backend.asarray(self.jacobian_func(theta))

            # Levenberg-Marquardt adaptive damping
            if iteration > 0:
                if current_error < prev_error * 0.75:
                    damping = max(min_damping, damping / 3)
                    step_cap = min(0.5, step_cap * 1.2)
                    nu = 2.0
                elif current_error < prev_error * 0.95:
                    damping = max(min_damping, damping / 1.5)
                elif current_error > prev_error:
                    damping = min(max_damping, damping * nu)
                    nu = min(nu * 1.5, 8)
                    step_cap = max(min_step_cap, step_cap * 0.7)

            prev_error = current_error

            # SVD-robust Jacobian solve (primary path)
            try:
                U, s, Vt = backend.svd(J, full_matrices=False)
                s_damped = s / (s**2 + damping**2 + 1e-12)
                delta_theta = backend.matmul(
                    Vt.T, s_damped * backend.matmul(U.T, V_err)
                )
            except Exception as exc:
                # CuPy raises its own LinAlgError class rather than NumPy's.
                if not isinstance(exc, np.linalg.LinAlgError) and (
                    exc.__class__.__name__ != "LinAlgError"
                ):
                    raise
                # Fallback to normal equations
                JTJ = backend.matmul(J.T, J)
                lambda_I = (damping**2) * backend.eye(JTJ.shape[0])
                try:
                    delta_theta = backend.solve(
                        JTJ + lambda_I, backend.matmul(J.T, V_err)
                    )
                except Exception as exc2:
                    if not isinstance(exc2, np.linalg.LinAlgError) and (
                        exc2.__class__.__name__ != "LinAlgError"
                    ):
                        raise
                    delta_theta = backend.zeros(self.n_joints)

            # Step size limiting
            step_norm = backend.norm(delta_theta)
            if step_norm > step_cap:
                delta_theta = delta_theta * (step_cap / step_norm)

            # Backtracking line search — accept best scale
            best_candidate = self._clip_to_limits(theta + delta_theta)
            best_cand_err = current_error

            for scale in [0.5, 0.25]:
                candidate = self._clip_to_limits(theta + scale * delta_theta)
                T_try = self.fk_func(candidate)
                _, rot_try, trans_try = self.error_func(T_try, T_desired)
                err_try = float(rot_try) + float(trans_try)
                if err_try < best_cand_err:
                    best_cand_err = err_try
                    best_candidate = candidate

            # Check full step too
            T_full = self.fk_func(self._clip_to_limits(theta + delta_theta))
            _, rot_full, trans_full = self.error_func(T_full, T_desired)
            err_full = float(rot_full) + float(trans_full)
            if err_full < best_cand_err:
                best_candidate = self._clip_to_limits(theta + delta_theta)

            theta = best_candidate

        # Return best found
        T_curr = self.fk_func(best_theta)
        _, rot_err, trans_err = self.error_func(T_curr, T_desired)
        rot_err = float(rot_err)
        trans_err = float(trans_err)
        success = rot_err < eomg and trans_err < ev
        return best_theta, success, rot_err + trans_err

    def _detect_oscillation(
        self, error_history: deque, window_size: int = 5
    ) -> Tuple[bool, bool]:
        """
        Detect stagnation and oscillation in the optimization trajectory.

        Returns:
            Tuple of (is_stagnating, is_oscillating)
        """
        if len(error_history) < window_size:
            return False, False

        recent_errors = list(error_history)[-window_size:]

        # Stagnation detection
        error_std = np.std(recent_errors)
        error_mean = np.mean(recent_errors)
        is_stagnating = (
            error_std < 1e-6 * error_mean if error_mean > 1e-10 else error_std < 1e-10
        )

        # Oscillation detection
        diffs = np.diff(recent_errors)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        is_oscillating = sign_changes >= window_size - 2

        return is_stagnating, is_oscillating

    def _sqp_solver(
        self,
        T_desired: NDArray[np.float64],
        theta0: NDArray[np.float64],
        eomg: float,
        ev: float,
        timeout: float,
        stop_event: threading.Event,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        """
        Sequential Quadratic Programming solver.

        Optimization-based solver, good for constrained problems and
        escaping local minima.
        """
        start_time = time.perf_counter()
        backend = get_backend()

        # SLSQP is host-bound: SciPy is not backend-dispatchable, so the
        # optimizer and everything it hands back (objective scalars, gradient,
        # and the x0 seed) live on the NumPy host. The backend-native math
        # inside the error/Jacobian callables is converted to the host exactly
        # here, at the optimizer boundary.
        def objective(theta: NDArray[np.float64]) -> float:
            """Objective function: minimize pose error."""
            if stop_event.is_set():
                return 1e10
            if time.perf_counter() - start_time > timeout:
                return 1e10

            T_curr = self.fk_func(theta)
            V_err, rot_err, trans_err = self.error_func(T_curr, T_desired)
            return float(rot_err) ** 2 + float(trans_err) ** 2

        def jacobian_objective(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            """Gradient of objective function."""
            if stop_event.is_set():
                return np.zeros(self.n_joints)

            T_curr = self.fk_func(theta)
            V_err, _, _ = self.error_func(T_curr, T_desired)
            V_err = backend.asarray(V_err)
            J = backend.asarray(self.jacobian_func(theta))
            return backend.to_numpy(2 * backend.matmul(J.T, V_err))

        try:
            from scipy.optimize import minimize

            result = minimize(
                objective,
                backend.to_numpy(backend.asarray(theta0)),
                method="SLSQP",
                jac=jacobian_objective,
                bounds=self.bounds,
                options={"ftol": 1e-8, "maxiter": 500, "disp": False},
            )

            # SciPy hands back a NumPy x; normalize onto the active backend so
            # solve() has one consistent (backend-native) return domain.
            theta = backend.asarray(result.x)

        except Exception:
            theta = backend.asarray(theta0)

        # Check final error
        T_curr = self.fk_func(theta)
        _, rot_err, trans_err = self.error_func(T_curr, T_desired)
        rot_err = float(rot_err)
        trans_err = float(trans_err)
        success = rot_err < eomg and trans_err < ev

        return theta, success, rot_err + trans_err

    def _workspace_heuristic(
        self, T_desired: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Generate initial guess using geometric heuristic."""
        from .. import ik_helpers

        return ik_helpers.workspace_heuristic_guess(
            T_desired, self.n_joints, self.joint_limits
        )

    def _midpoint_guess(self) -> NDArray[np.float64]:
        """Generate midpoint of joint limits."""
        from .. import ik_helpers

        return ik_helpers.midpoint_of_limits(self.joint_limits)

    def _random_guess(self) -> NDArray[np.float64]:
        """Generate random configuration within limits."""
        from .. import ik_helpers

        return ik_helpers.random_in_limits(self.joint_limits)

    def _clip_to_limits(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip joint angles to limits."""
        from .. import ik_helpers

        return ik_helpers._clip_to_limits(theta, self.joint_limits)

    def _default_error_func(
        self, T_current: NDArray[np.float64], T_desired: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float, float]:
        """
        Default error function using geometric error for stable convergence.

        Returns:
            Tuple of (error_vector, rotation_error, translation_error)
            - error_vector: 6D error for Jacobian-based updates [omega, v]
            - rotation_error: Actual rotation error in radians
            - translation_error: Actual Cartesian position error in meters
        """
        backend = get_backend()
        T_current = backend.asarray(T_current)
        T_desired = backend.asarray(T_desired)

        # Position error
        p_current = T_current[:3, 3]
        p_desired = T_desired[:3, 3]
        pos_error = p_desired - p_current
        trans_err = backend.norm(pos_error)

        # Rotation error using axis-angle
        R_current = T_current[:3, :3]
        R_desired = T_desired[:3, :3]
        R_err = backend.matmul(R_current.T, R_desired)

        trace_val = backend.clip((backend.trace(R_err) - 1) / 2, -1, 1)
        angle = backend.arccos(trace_val)
        rot_err = backend.abs(angle)

        # Compute rotation axis
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
        elif abs(angle - np.pi) < 1e-6:  # np.pi: host constant
            diag = backend.diag(R_err)
            k = int(backend.argmax(diag))
            # All three former branches follow one rule: component ``k`` is 1
            # and every other component ``j`` is R_err[k, j] / (1 + R_err[k, k]),
            # falling back to 0 when that denominator is degenerate. Assembled
            # as a sequence because JAX's immutable arrays reject the
            # element-wise assignment this replaces.
            denom = 1 + R_err[k, k]
            usable = bool(backend.abs(denom) > 1e-6)
            axis = backend.asarray(
                backend.stack(
                    tuple(
                        1.0
                        if j == k
                        else (R_err[k, j] / denom if usable else 0.0)
                        for j in range(3)
                    )
                ),
                dtype=R_err.dtype,
            )
            axis = axis / (backend.norm(axis) + 1e-10)
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

        omega_err_space = backend.matmul(R_current, omega_err)
        V_err = backend.concatenate((omega_err_space, pos_error))

        return V_err, rot_err, trans_err


def trac_ik_solve(
    robot: Any,
    T_desired: NDArray[np.float64],
    theta0: Optional[NDArray[np.float64]] = None,
    timeout: float = 0.2,
    eomg: float = 1e-4,
    ev: float = 1e-4,
    num_restarts: int = 5,
    use_parallel: bool = False,
) -> Tuple[NDArray[np.float64], bool, float]:
    """
    Convenience function to solve IK using TRAC-IK for a SerialManipulator.

    Args:
        robot: SerialManipulator instance
        T_desired: Target 4x4 transformation matrix
        theta0: Initial guess (optional)
        timeout: Maximum total solve time in seconds (default: 200ms)
        eomg: Orientation tolerance
        ev: Position tolerance
        num_restarts: Number of initial guesses (default: 5)
        use_parallel: Run DLS+SQP in parallel per guess (default: False)

    Returns:
        Tuple of (theta, success, solve_time)

    Example:
        >>> from ManipulaPy.trac_ik import trac_ik_solve
        >>> theta, success, time = trac_ik_solve(robot, T_target)
    """
    solver = TracIKSolver(
        fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
        jacobian_func=lambda th: robot.jacobian(th, frame="space"),
        joint_limits=robot.joint_limits,
        n_joints=len(robot.joint_limits),
    )

    return solver.solve(
        T_desired, theta0, timeout, eomg, ev, num_restarts, use_parallel
    )


__all__ = ["TracIKSolver", "trac_ik_solve"]
