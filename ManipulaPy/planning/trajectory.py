#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Joint- and Cartesian-space trajectory generation mixin - ManipulaPy"""

from ._kernels import *  # noqa: F401,F403


@njit(parallel=True, fastmath=True)
def _trajectory_cpu_fallback(
    thetastart, thetaend, Tf, N, method
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimised CPU trajectory generation (parallel).

    Computes joint-space position, velocity and acceleration trajectories by
    point-to-point time scaling between ``thetastart`` and ``thetaend``. The
    nested (timestep, joint) iteration is flattened into a single ``prange``
    loop so Numba can parallelise it.

    Args:
        thetastart: (num_joints,) ndarray of starting joint angles, radians.
        thetaend: (num_joints,) ndarray of ending joint angles, radians.
        Tf (float): Total trajectory duration, seconds.
        N (int): Number of trajectory points (timesteps).
        method (int): Time-scaling method; 3 for cubic, 5 for quintic. Any
            other value yields zero scaling.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
        traj_acc)``, each an ``(N, num_joints)`` float32 array of positions
        (rad), velocities (rad/s) and accelerations (rad/s^2).
    """
    num_joints = len(thetastart)

    traj_pos = np.zeros((N, num_joints), dtype=np.float32)
    traj_vel = np.zeros((N, num_joints), dtype=np.float32)
    traj_acc = np.zeros((N, num_joints), dtype=np.float32)

    # Flatten (idx, j) → k  to avoid nested loops that block parallelisation
    total_elems = N * num_joints
    for k in prange(total_elems):
        idx = k // num_joints  # timestep
        j = k % num_joints  # joint index

        t = idx * (Tf / (N - 1))
        tau = t / Tf

        # Time-scaling
        if method == 3:  # cubic
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
        elif method == 5:  # quintic
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
            s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
        else:  # unsupported method
            s = s_dot = s_ddot = 0.0

        dtheta = thetaend[j] - thetastart[j]
        traj_pos[idx, j] = s * dtheta + thetastart[j]
        traj_vel[idx, j] = s_dot * dtheta
        traj_acc[idx, j] = s_ddot * dtheta

    return traj_pos, traj_vel, traj_acc

@njit(fastmath=True)
def _traj_cpu_njit(
    thetastart, thetaend, Tf, N, method
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the optimized CPU fallback kernel.

    Thin ``@njit`` wrapper that forwards its arguments to
    :func:`_trajectory_cpu_fallback`, keeping a stable public entry point for
    the CPU trajectory path.

    Args:
        thetastart: (num_joints,) ndarray of starting joint angles, radians.
        thetaend: (num_joints,) ndarray of ending joint angles, radians.
        Tf (float): Total trajectory duration, seconds.
        N (int): Number of trajectory points (timesteps).
        method (int): Time-scaling method; 3 for cubic, 5 for quintic.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
        traj_acc)``, each an ``(N, num_joints)`` float32 array.
    """
    return _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)


class _GenerationMixin:
    def joint_trajectory(
        self,
        thetastart,
        thetaend,
        Tf,
        N,
        method,
        kernel_type=None,
        enable_monitoring=None,
    ) -> Dict[str, np.ndarray]:
        """
        Enhanced joint trajectory generation with advanced CUDA optimizations.

        Args:
            thetastart (numpy.ndarray): The starting joint angles.
            thetaend (numpy.ndarray): The ending joint angles.
            Tf (float): The final time for the trajectory.
            N (int): The number of steps in the trajectory.
            method (int): The method to use (3=cubic, 5=quintic).
            kernel_type (str, optional): Override default kernel selection.
            enable_monitoring (bool, optional): Override default monitoring.

        Returns:
            dict: A dictionary containing positions, velocities, and accelerations.
        """
        # Use instance defaults if not specified, with safety checks
        if kernel_type is None:
            kernel_type = getattr(self, "kernel_type", "auto")
        if enable_monitoring is None:
            enable_monitoring = getattr(self, "enable_profiling", False)

        logger.info(
            f"Generating joint trajectory: N={N}, joints={len(thetastart)}, "
            f"method={method}, kernel={kernel_type}"
        )

        # Keep host NumPy inputs here: the unchanged CUDA path
        # (_joint_trajectory_gpu -> optimized_trajectory_generation_monitored,
        # np.ascontiguousarray, pinned H2D) requires real NumPy. Callers may pass
        # backend-native arrays, so force them to host through the backend before
        # the float32 cast (a no-op for NumPy inputs; np.array() alone would
        # raise on a CuPy array). The backend domain is re-entered inside
        # _joint_trajectory_cpu; entering it before _should_use_gpu would feed
        # device arrays to the GPU path and force a silent CPU fallback.
        backend = get_backend()
        thetastart = np.array(
            backend.to_numpy(backend.asarray(thetastart)), dtype=np.float32
        )
        thetaend = np.array(
            backend.to_numpy(backend.asarray(thetaend)), dtype=np.float32
        )
        num_joints = len(thetastart)

        # Print performance recommendations if beneficial
        total_work = N * num_joints
        if self.cuda_available and total_work >= 10000:
            print_performance_recommendations(N, num_joints)

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(N, num_joints)

        if use_gpu:
            return self._joint_trajectory_gpu(
                thetastart, thetaend, Tf, N, method, kernel_type, enable_monitoring
            )
        else:
            return self._joint_trajectory_cpu(thetastart, thetaend, Tf, N, method)

    def _joint_trajectory_gpu(
        self, thetastart, thetaend, Tf, N, method, kernel_type, enable_monitoring
    ) -> Dict[str, np.ndarray]:
        """GPU joint trajectory generation with optimal kernel selection.

        Runs the monitored GPU trajectory generator, clips positions to the
        configured joint limits, optionally applies GPU collision avoidance,
        updates performance stats, and falls back to the CPU path on any error.

        Args:
            thetastart: (num_joints,) ndarray of starting joint angles (rad).
            thetaend: (num_joints,) ndarray of ending joint angles (rad).
            Tf (float): Total trajectory duration, seconds.
            N (int): Number of trajectory points.
            method (int): Time-scaling method; 3 for cubic, 5 for quintic.
            kernel_type (str): Kernel selection strategy passed to the GPU
                generator (e.g. "auto", "vectorized").
            enable_monitoring (bool): If True, enable CUDA profiling/logging.

        Returns:
            Dict[str, np.ndarray]: Dict with keys ``"positions"``,
            ``"velocities"`` and ``"accelerations"``, each an
            ``(N, num_joints)`` array.
        """
        start_time = time.time()

        try:
            # Use the monitored high-level wrapper for maximum performance
            traj_pos_host, traj_vel_host, traj_acc_host = (
                optimized_trajectory_generation_monitored(
                    thetastart,
                    thetaend,
                    Tf,
                    N,
                    method,
                    use_pinned=True,
                    kernel_type=kernel_type,
                    enable_monitoring=enable_monitoring,
                )
            )

            # Apply joint limits
            num_joints = len(thetastart)
            for i in range(num_joints):
                traj_pos_host[:, i] = np.clip(
                    traj_pos_host[:, i],
                    self.joint_limits[i, 0],
                    self.joint_limits[i, 1],
                )

            # Apply collision avoidance if available
            if self.collision_checker and self.potential_field:
                traj_pos_host = self._apply_collision_avoidance_gpu(
                    traj_pos_host, thetaend
                )

            # Calculate achieved speedup
            elapsed = time.time() - start_time
            if hasattr(self, "_last_cpu_time") and self._last_cpu_time > 0:
                speedup = self._last_cpu_time / elapsed
                self.performance_stats["speedup_achieved"] = speedup
                if enable_monitoring:
                    logger.info(f"🎯 Achieved {speedup:.1f}x speedup over CPU!")
            else:
                # No previous CPU time to compare against
                if enable_monitoring:
                    logger.info(
                        "🚀 GPU execution completed (no CPU baseline for comparison)"
                    )

            # Update performance stats
            self.performance_stats["gpu_calls"] += 1
            self.performance_stats["total_gpu_time"] += elapsed
            self.performance_stats["kernel_launches"] += 1

            # Update best kernel used
            config = self._get_optimal_kernel_config(N, len(thetastart))
            if config:
                self.performance_stats["best_kernel_used"] = config.get(
                    "kernel_type", "unknown"
                )

            logger.info(f"GPU trajectory generation completed in {elapsed:.4f}s")

            # Normalize into one domain: collision avoidance re-enters the
            # backend for positions while the GPU velocities/accelerations stay
            # host NumPy, so wrap every entry with backend.asarray (a no-op under
            # the NumPy backend).
            backend = get_backend()
            return {
                "positions": backend.asarray(traj_pos_host),
                "velocities": backend.asarray(traj_vel_host),
                "accelerations": backend.asarray(traj_acc_host),
            }

        except Exception as e:
            logger.warning(
                f"GPU trajectory generation failed: {e}, falling back to CPU"
            )
            return self._joint_trajectory_cpu(thetastart, thetaend, Tf, N, method)

    def _joint_trajectory_cpu(
        self, thetastart, thetaend, Tf, N, method
    ) -> Dict[str, np.ndarray]:
        """CPU joint trajectory generation with performance tracking.

        Generates the trajectory via the Numba CPU kernel, clips positions to
        the configured joint limits, optionally applies CPU collision
        avoidance, and records the elapsed time as the CPU baseline used for
        speedup reporting.

        Args:
            thetastart: (num_joints,) ndarray of starting joint angles (rad).
            thetaend: (num_joints,) ndarray of ending joint angles (rad).
            Tf (float): Total trajectory duration, seconds.
            N (int): Number of trajectory points.
            method (int): Time-scaling method; 3 for cubic, 5 for quintic.

        Returns:
            Dict[str, np.ndarray]: Dict with keys ``"positions"``,
            ``"velocities"`` and ``"accelerations"``, each an
            ``(N, num_joints)`` array.
        """
        backend = get_backend()
        start_time = time.time()

        # Optimized CPU fallback: the Numba kernel is a host (real-NumPy)
        # boundary, so convert at the call site and re-enter the backend.
        traj_pos, traj_vel, traj_acc = _traj_cpu_njit(
            backend.to_numpy(thetastart), backend.to_numpy(thetaend), Tf, N, method
        )
        traj_pos = backend.asarray(traj_pos)
        traj_vel = backend.asarray(traj_vel)
        traj_acc = backend.asarray(traj_acc)

        # Apply joint limits column-wise (broadcast, no in-place writes)
        lower = backend.asarray(self.joint_limits[:, 0])
        upper = backend.asarray(self.joint_limits[:, 1])
        traj_pos = backend.clip(traj_pos, lower, upper)

        # Apply collision avoidance if available
        if self.collision_checker and self.potential_field:
            traj_pos = self._apply_collision_avoidance_cpu(traj_pos, thetaend)

        # Store CPU time for speedup calculations
        elapsed = time.time() - start_time
        self._last_cpu_time = elapsed

        # Update performance stats
        self.performance_stats["cpu_calls"] += 1
        self.performance_stats["total_cpu_time"] += elapsed

        logger.info(f"CPU trajectory generation completed in {elapsed:.4f}s")

        return {
            "positions": traj_pos,
            "velocities": traj_vel,
            "accelerations": traj_acc,
        }

    def batch_joint_trajectory(
        self, thetastart_batch, thetaend_batch, Tf, N, method, kernel_type=None
    ) -> Dict[str, np.ndarray]:
        """
        Enhanced batch trajectory generation with optimal kernel selection.

        Args:
            thetastart_batch (numpy.ndarray): Starting angles (batch_size, num_joints)
            thetaend_batch (numpy.ndarray): Ending angles (batch_size, num_joints)
            Tf (float): Final time for all trajectories
            N (int): Number of trajectory points
            method (int): Time scaling method
            kernel_type (str, optional): Override kernel selection

        Returns:
            dict: Batch trajectory data with shape (batch_size, N, num_joints)
        """
        if kernel_type is None:
            kernel_type = getattr(self, "kernel_type", "auto")

        batch_size, num_joints = thetastart_batch.shape
        logger.info(
            f"Generating batch trajectories: batch_size={batch_size}, "
            f"N={N}, joints={num_joints}, kernel={kernel_type}"
        )

        if not self.cuda_available:
            logger.warning(
                "Batch processing requires CUDA. Falling back to sequential processing."
            )
            return self._batch_joint_trajectory_cpu(
                thetastart_batch, thetaend_batch, Tf, N, method
            )

        # Print performance recommendations for batch processing
        total_work = batch_size * N * num_joints
        if total_work >= 50000:
            print_performance_recommendations(N * batch_size, num_joints)

        start_time = time.time()

        try:
            # Use optimized batch trajectory generation
            traj_pos_host, traj_vel_host, traj_acc_host = (
                optimized_batch_trajectory_generation(
                    thetastart_batch, thetaend_batch, Tf, N, method, use_pinned=True
                )
            )

            # Apply joint limits for all trajectories
            for batch_idx in range(batch_size):
                for i in range(num_joints):
                    traj_pos_host[batch_idx, :, i] = np.clip(
                        traj_pos_host[batch_idx, :, i],
                        self.joint_limits[i, 0],
                        self.joint_limits[i, 1],
                    )

            elapsed = time.time() - start_time
            throughput = total_work / elapsed / 1e6  # Million elements per second

            self.performance_stats["gpu_calls"] += 1
            self.performance_stats["total_gpu_time"] += elapsed
            self.performance_stats["kernel_launches"] += 1

            logger.info(f"Batch GPU trajectory generation completed in {elapsed:.4f}s")
            logger.info(f"📊 Throughput: {throughput:.1f} M elements/sec")

            return {
                "positions": traj_pos_host,
                "velocities": traj_vel_host,
                "accelerations": traj_acc_host,
            }

        except Exception as e:
            logger.warning(
                f"Batch GPU trajectory generation failed: {e}, falling back to CPU"
            )
            return self._batch_joint_trajectory_cpu(
                thetastart_batch, thetaend_batch, Tf, N, method
            )

    def _batch_joint_trajectory_cpu(
        self, thetastart_batch, thetaend_batch, Tf, N, method
    ) -> Dict[str, np.ndarray]:
        """CPU fallback for batch trajectory generation.

        Generates each trajectory in the batch sequentially with the Numba CPU
        kernel and clips positions to the configured joint limits (matching the
        GPU batch path).

        Args:
            thetastart_batch: (batch_size, num_joints) ndarray of starting
                joint angles, radians.
            thetaend_batch: (batch_size, num_joints) ndarray of ending joint
                angles, radians.
            Tf (float): Total trajectory duration, seconds.
            N (int): Number of trajectory points per trajectory.
            method (int): Time-scaling method; 3 for cubic, 5 for quintic.

        Returns:
            Dict[str, np.ndarray]: Dict with keys ``"positions"``,
            ``"velocities"`` and ``"accelerations"``, each a
            ``(batch_size, N, num_joints)`` array.
        """
        backend = get_backend()
        start_time = time.time()

        batch_size, num_joints = thetastart_batch.shape

        # Numba kernel is a host (real-NumPy) boundary; convert at the call site.
        start_host = backend.to_numpy(thetastart_batch)
        end_host = backend.to_numpy(thetaend_batch)

        # Process each trajectory in the batch, then stack into one array
        pos_rows, vel_rows, acc_rows = [], [], []
        for i in range(batch_size):
            traj_pos, traj_vel, traj_acc = _traj_cpu_njit(
                start_host[i], end_host[i], Tf, N, method
            )
            pos_rows.append(backend.asarray(traj_pos))
            vel_rows.append(backend.asarray(traj_vel))
            acc_rows.append(backend.asarray(traj_acc))

        if pos_rows:
            traj_pos_batch = backend.stack(pos_rows)
            traj_vel_batch = backend.stack(vel_rows)
            traj_acc_batch = backend.stack(acc_rows)

            # Enforce joint limits column-wise (broadcast, parity with GPU path)
            lower = backend.asarray(self.joint_limits[:, 0])
            upper = backend.asarray(self.joint_limits[:, 1])
            traj_pos_batch = backend.clip(traj_pos_batch, lower, upper)
        else:
            # Base preallocated (0, N, num_joints) zeros and skipped the per-row
            # and per-limit loops for an empty batch.
            shape = (0, N, num_joints)
            traj_pos_batch = backend.zeros(shape, dtype=backend.float32)
            traj_vel_batch = backend.zeros(shape, dtype=backend.float32)
            traj_acc_batch = backend.zeros(shape, dtype=backend.float32)

        elapsed = time.time() - start_time
        self.performance_stats["cpu_calls"] += 1
        self.performance_stats["total_cpu_time"] += elapsed

        logger.info(f"Batch CPU trajectory generation completed in {elapsed:.4f}s")

        return {
            "positions": traj_pos_batch,
            "velocities": traj_vel_batch,
            "accelerations": traj_acc_batch,
        }

    def cartesian_trajectory(
        self, Xstart, Xend, Tf, N, method
    ) -> Dict[str, np.ndarray]:
        """
        Enhanced Cartesian trajectory generation with optimal kernel selection.

        Args:
            Xstart (np.ndarray): Initial end-effector configuration (SE(3) matrix).
            Xend (np.ndarray): Final end-effector configuration (SE(3) matrix).
            Tf (float): Total time of motion.
            N (int): Number of trajectory points.
            method (int): Time-scaling method (3=cubic, 5=quintic).

        Returns:
            dict: Dictionary with positions, velocities, accelerations, and orientations.
        """
        logger.info(f"Generating Cartesian trajectory: N={N}, method={method}")

        backend = get_backend()
        N = int(N)
        timegap = Tf / (N - 1.0)
        # Callers may pass backend-native transforms, and TransToRp only slices,
        # so force pstart/pend to host NumPy for the unchanged GPU velocity path
        # (see _cartesian_trajectory_gpu). This is a no-op for NumPy inputs. The
        # assembly below runs on the backend copies (``*_b``).
        Rstart, pstart = TransToRp(Xstart)
        Rend, pend = TransToRp(Xend)
        pstart = backend.to_numpy(backend.asarray(pstart))
        pend = backend.to_numpy(backend.asarray(pend))
        Rstart_b = backend.asarray(Rstart)
        Rend_b = backend.asarray(Rend)
        pstart_b = backend.asarray(pstart)
        pend_b = backend.asarray(pend)

        # Compute orientation interpolation on the host (complex matrix ops).
        # The straight-line position at step i is the SE(3) translation column,
        # so it is assembled directly instead of building the full 4x4 pose.
        orientation_rows = []
        position_rows = []
        for i in range(N):
            if method == 3:
                s = CubicTimeScaling(Tf, timegap * i)
            else:
                s = QuinticTimeScaling(Tf, timegap * i)

            orientation_rows.append(
                backend.matmul(
                    Rstart_b,
                    MatrixExp3(MatrixLog3(backend.matmul(Rstart_b.T, Rend_b)) * s),
                )
            )
            position_rows.append(s * pend_b + (1 - s) * pstart_b)

        if position_rows:
            orientations = backend.asarray(
                backend.stack(orientation_rows), dtype=backend.float32
            )
            traj_pos = backend.asarray(
                backend.stack(position_rows), dtype=backend.float32
            )
        else:
            # N <= 0: base built np.zeros((N, 3, 3)) and np.array([]) (shape
            # (0,)). Route N through the orientation preallocation so N == 0
            # yields (0, 3, 3) while N < 0 raises ValueError exactly as base did.
            orientations = backend.zeros((N, 3, 3), dtype=backend.float32)
            traj_pos = backend.zeros((0,), dtype=backend.float32)

        # Use GPU for position/velocity/acceleration computation if beneficial
        use_gpu = self._should_use_gpu(N, 3)  # 3 coordinates (x,y,z)

        if use_gpu:
            traj_vel, traj_acc = self._cartesian_trajectory_gpu(
                pstart, pend, Tf, N, method
            )
        else:
            traj_vel, traj_acc = self._cartesian_trajectory_cpu(
                pstart, pend, Tf, N, method
            )

        # Normalize into one domain: the assembled positions/orientations are
        # backend-native, but the GPU velocity path returns host NumPy, so wrap
        # every entry with backend.asarray (a no-op under the NumPy backend).
        return {
            "positions": backend.asarray(traj_pos),
            "velocities": backend.asarray(traj_vel),
            "accelerations": backend.asarray(traj_acc),
            "orientations": backend.asarray(orientations),
        }

    def _cartesian_trajectory_gpu(
        self, pstart, pend, Tf, N, method
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GPU computation of Cartesian linear velocity and acceleration.

        Launches the Cartesian trajectory kernel to compute the time-scaled
        translational velocity and acceleration of the straight-line path from
        ``pstart`` to ``pend``. Falls back to the CPU implementation on error.

        Args:
            pstart: (3,) ndarray of the start position, metres.
            pend: (3,) ndarray of the end position, metres.
            Tf (float): Total trajectory duration, seconds.
            N (int): Number of trajectory points.
            method (int): Time-scaling method; 3 for cubic, 5 for quintic.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(traj_vel, traj_acc)``, each an
            ``(N, 3)`` array of linear velocity (m/s) and acceleration
            (m/s^2).
        """
        start_time = time.time()

        try:
            pstart = np.ascontiguousarray(pstart.astype(np.float32))
            pend = np.ascontiguousarray(pend.astype(np.float32))

            traj_vel = get_cuda_array((N, 3), dtype=np.float32)
            traj_acc = get_cuda_array((N, 3), dtype=np.float32)
            traj_pos_dummy = get_cuda_array((N, 3), dtype=np.float32)

            # Transfer data using pinned memory
            d_pstart = _h2d_pinned(pstart)
            d_pend = _h2d_pinned(pend)

            # Get optimal launch configuration
            grid_config = get_optimal_kernel_config(N, 3, "warp_optimized")
            if grid_config:
                blocks_per_grid = grid_config["grid"]
                threads_per_block = grid_config["block"]
                logger.info(
                    f"Using {grid_config['kernel_type']} for Cartesian trajectory"
                )
            else:
                blocks_per_grid, threads_per_block = _best_2d_config(N, 3)

            # Launch Cartesian trajectory kernel
            cartesian_trajectory_kernel[blocks_per_grid, threads_per_block](
                d_pstart, d_pend, traj_pos_dummy, traj_vel, traj_acc, Tf, N, method
            )

            # Copy results back
            traj_vel_host = traj_vel.copy_to_host()
            traj_acc_host = traj_acc.copy_to_host()

            elapsed = time.time() - start_time
            self.performance_stats["gpu_calls"] += 1
            self.performance_stats["total_gpu_time"] += elapsed
            self.performance_stats["kernel_launches"] += 1

            logger.info(f"GPU Cartesian trajectory completed in {elapsed:.4f}s")

            return traj_vel_host, traj_acc_host

        except Exception as e:
            logger.warning(f"GPU Cartesian trajectory failed: {e}, falling back to CPU")
            return self._cartesian_trajectory_cpu(pstart, pend, Tf, N, method)
        finally:
            # Return memory to pool
            if "traj_vel" in locals():
                return_cuda_array(traj_vel)
            if "traj_acc" in locals():
                return_cuda_array(traj_acc)
            if "traj_pos_dummy" in locals():
                return_cuda_array(traj_pos_dummy)

    def _cartesian_trajectory_cpu(
        self, pstart, pend, Tf, N, method
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU computation of Cartesian linear velocity and acceleration.

        Evaluates the time-scaling derivatives at each point and applies them to
        the straight-line displacement ``pend - pstart`` to obtain the
        translational velocity and acceleration profiles.

        Args:
            pstart: (3,) ndarray of the start position, metres.
            pend: (3,) ndarray of the end position, metres.
            Tf (float): Total trajectory duration, seconds.
            N (int): Number of trajectory points.
            method (int): Time-scaling method; 3 for cubic, 5 for quintic.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``(traj_vel, traj_acc)``, each an
            ``(N, 3)`` array of linear velocity (m/s) and acceleration
            (m/s^2).
        """
        backend = get_backend()
        start_time = time.time()

        dp = backend.asarray(pend) - backend.asarray(pstart)
        vel_rows = []
        acc_rows = []
        for i in range(N):
            t = i * (Tf / (N - 1))
            tau = t / Tf

            if method == 3:
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:
                tau2 = tau * tau
                tau3 = tau2 * tau
                tau4 = tau2 * tau2
                s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
                s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
            else:
                s_dot = s_ddot = 0.0

            vel_rows.append(s_dot * dp)
            acc_rows.append(s_ddot * dp)

        if vel_rows:
            traj_vel = backend.asarray(backend.stack(vel_rows), dtype=backend.float32)
            traj_acc = backend.asarray(backend.stack(acc_rows), dtype=backend.float32)
        else:
            # N <= 0: base preallocated np.zeros((N, 3)); route N through the
            # shape so N == 0 yields (0, 3) and N < 0 raises ValueError.
            traj_vel = backend.zeros((N, 3), dtype=backend.float32)
            traj_acc = backend.zeros((N, 3), dtype=backend.float32)

        elapsed = time.time() - start_time
        self.performance_stats["cpu_calls"] += 1
        self.performance_stats["total_cpu_time"] += elapsed

        logger.info(f"CPU Cartesian trajectory completed in {elapsed:.4f}s")

        return traj_vel, traj_acc
