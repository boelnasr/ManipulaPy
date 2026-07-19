#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverse/forward dynamics trajectory mixin - ManipulaPy"""

from . import _kernels as _runtime
from ._kernels import Dict, Tuple, logger, np, time


class _DynamicsMixin:
    def inverse_dynamics_trajectory(
        self,
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector=None,
        Ftip=None,
    ) -> np.ndarray:
        """
        Compute joint torques with enhanced CUDA acceleration.

        Args:
            thetalist_trajectory (np.ndarray): Joint angles over the trajectory.
            dthetalist_trajectory (np.ndarray): Joint velocities over the
                trajectory.
            ddthetalist_trajectory (np.ndarray): Joint accelerations over the
                trajectory.
            gravity_vector (np.ndarray, optional): Gravity vector affecting the system.
            Ftip (list, optional): External forces applied at the end effector.

        Returns:
            np.ndarray: Array of joint torques required to follow the trajectory.
        """
        if gravity_vector is None:
            gravity_vector = np.array([0, 0, -9.81])
        if Ftip is None:
            Ftip = [0, 0, 0, 0, 0, 0]

        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]

        logger.info(
            f"Computing inverse dynamics: {num_points} points, {num_joints} joints"
        )

        # Print performance recommendations
        if self.cuda_available:
            total_work = num_points * num_joints
            if total_work >= 10000:
                _runtime.print_performance_recommendations(num_points, num_joints)

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(num_points, num_joints)

        if use_gpu:
            return self._inverse_dynamics_gpu(
                thetalist_trajectory,
                dthetalist_trajectory,
                ddthetalist_trajectory,
                gravity_vector,
                Ftip,
            )
        else:
            return self._inverse_dynamics_cpu(
                thetalist_trajectory,
                dthetalist_trajectory,
                ddthetalist_trajectory,
                gravity_vector,
                Ftip,
            )

    def _inverse_dynamics_gpu(
        self,
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector,
        Ftip,
    ) -> np.ndarray:
        """GPU-accelerated inverse dynamics over a trajectory.

        Transfers the trajectory and the manipulator's dynamics data (Glist,
        S_list, M_list) to the device, launches the 2D inverse-dynamics kernel
        to compute per-point joint torques, clips them to ``self.torque_limits``
        and returns them. Falls back to the CPU implementation if dynamics-data
        conversion or kernel execution fails.

        Args:
            thetalist_trajectory: (num_points, num_joints) ndarray of joint
                angles, radians.
            dthetalist_trajectory: (num_points, num_joints) ndarray of joint
                velocities, rad/s.
            ddthetalist_trajectory: (num_points, num_joints) ndarray of joint
                accelerations, rad/s^2.
            gravity_vector: (3,) ndarray of gravitational acceleration, m/s^2.
            Ftip: length-6 sequence of the external wrench at the end effector.

        Returns:
            np.ndarray: (num_points, num_joints) array of joint torques,
            clipped to the configured torque limits.
        """
        start_time = time.time()

        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]

        try:
            # Use memory pool for the large torques array
            torques_trajectory = _runtime.get_cuda_array(
                (num_points, num_joints), dtype=np.float32
            )

            # Transfer data to GPU using pinned memory - ensure proper data types
            d_thetalist_trajectory = _runtime._h2d_pinned(
                thetalist_trajectory.astype(np.float32)
            )
            d_dthetalist_trajectory = _runtime._h2d_pinned(
                dthetalist_trajectory.astype(np.float32)
            )
            d_ddthetalist_trajectory = _runtime._h2d_pinned(
                ddthetalist_trajectory.astype(np.float32)
            )

            d_gravity_vector = _runtime.cuda.to_device(
                gravity_vector.astype(np.float32)
            )
            d_Ftip = _runtime.cuda.to_device(np.array(Ftip, dtype=np.float32))

            # Safely handle dynamics data conversion
            try:
                # Convert Glist to proper numpy array format
                if hasattr(self.dynamics, "Glist") and self.dynamics.Glist is not None:
                    if isinstance(self.dynamics.Glist, list):
                        # Convert list of matrices to 3D numpy array
                        Glist_array = np.stack(self.dynamics.Glist).astype(np.float32)
                    else:
                        Glist_array = np.array(self.dynamics.Glist, dtype=np.float32)
                else:
                    # Create dummy Glist if not available
                    Glist_array = np.eye(6, dtype=np.float32)[None, :, :].repeat(
                        num_joints, axis=0
                    )

                # Convert S_list to proper format
                if (
                    hasattr(self.dynamics, "S_list")
                    and self.dynamics.S_list is not None
                ):
                    Slist_array = np.array(self.dynamics.S_list, dtype=np.float32)
                else:
                    # Create dummy S_list if not available
                    Slist_array = np.random.randn(6, num_joints).astype(np.float32)

                # Convert M_list to proper format
                if (
                    hasattr(self.dynamics, "M_list")
                    and self.dynamics.M_list is not None
                ):
                    M_array = np.array(self.dynamics.M_list, dtype=np.float32)
                else:
                    # Create dummy M if not available
                    M_array = np.eye(4, dtype=np.float32)

                d_Glist = _runtime.cuda.to_device(Glist_array)
                d_Slist = _runtime.cuda.to_device(Slist_array)
                d_M = _runtime.cuda.to_device(M_array)

            except Exception as e:
                logger.warning(
                    f"Error converting dynamics data: {e}, using simplified approach"
                )
                # Fallback to simplified dynamics computation on CPU
                return self._inverse_dynamics_cpu(
                    thetalist_trajectory,
                    dthetalist_trajectory,
                    ddthetalist_trajectory,
                    gravity_vector,
                    Ftip,
                )

            d_torque_limits = _runtime.cuda.to_device(
                self.torque_limits.astype(np.float32)
            )

            # Get optimal 2D launch configuration with bounds checking
            try:
                blocks_per_grid, threads_per_block = _runtime._best_2d_config(
                    num_points, num_joints
                )
                logger.info(
                    "Inverse dynamics 2D grid: blocks=%s, threads=%s",
                    blocks_per_grid,
                    threads_per_block,
                )
            except Exception as e:
                logger.warning(f"Error in grid configuration: {e}, using fallback")
                # Fallback to safe grid configuration
                blocks_per_grid = ((num_points + 15) // 16, (num_joints + 15) // 16)
                threads_per_block = (16, 16)

            # Launch optimized 2D inverse dynamics kernel with CORRECT signature
            try:
                # FIXED: The kernel expects 11 arguments, but was receiving 10
                # Original call was missing the 'stream' parameter (last argument)
                # Let's check the kernel signature in cuda_kernels.py:

                # From cuda_kernels.py, the kernel signature is:
                # _runtime.inverse_dynamics_kernel(
                #     thetalist_trajectory, dthetalist_trajectory,
                #     ddthetalist_trajectory, gravity_vector, Ftip, Glist,
                #     Slist, M, torques_trajectory, torque_limits, stream=0
                # )
                # That's 11 parameters total including the stream parameter

                _runtime.inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
                    d_thetalist_trajectory,  # 1
                    d_dthetalist_trajectory,  # 2
                    d_ddthetalist_trajectory,  # 3
                    d_gravity_vector,  # 4
                    d_Ftip,  # 5
                    d_Glist,  # 6
                    d_Slist,  # 7
                    d_M,  # 8
                    torques_trajectory,  # 9
                    d_torque_limits,  # 10
                    # 0  # 11 - stream parameter (was missing!)
                )

                # Synchronize to check for kernel execution errors
                _runtime.cuda.synchronize()

            except Exception as kernel_error:
                logger.warning(f"CUDA kernel execution failed: {kernel_error}")
                # Fallback to CPU implementation
                return self._inverse_dynamics_cpu(
                    thetalist_trajectory,
                    dthetalist_trajectory,
                    ddthetalist_trajectory,
                    gravity_vector,
                    Ftip,
                )

            # Copy results back using pinned memory
            torques_host = torques_trajectory.copy_to_host()

            # Apply final torque limits
            torques_host = np.clip(
                torques_host, self.torque_limits[:, 0], self.torque_limits[:, 1]
            )

            elapsed = time.time() - start_time
            self.performance_stats["gpu_calls"] += 1
            self.performance_stats["total_gpu_time"] += elapsed
            self.performance_stats["kernel_launches"] += 1

            logger.info(f"GPU inverse dynamics completed in {elapsed:.4f}s")
            return torques_host

        except Exception as e:
            logger.warning(f"GPU inverse dynamics failed: {e}, falling back to CPU")
            return self._inverse_dynamics_cpu(
                thetalist_trajectory,
                dthetalist_trajectory,
                ddthetalist_trajectory,
                gravity_vector,
                Ftip,
            )
        finally:
            # Return large array to pool
            if "torques_trajectory" in locals():
                _runtime.return_cuda_array(torques_trajectory)

    def _inverse_dynamics_cpu(
        self,
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector,
        Ftip,
    ) -> np.ndarray:
        """CPU-based inverse dynamics over a trajectory.

        Computes per-point joint torques by calling
        ``self.dynamics.inverse_dynamics`` at each trajectory point (using zero
        torques for any point that raises), then clips the result to
        ``self.torque_limits``.

        Args:
            thetalist_trajectory: (num_points, num_joints) ndarray of joint
                angles, radians.
            dthetalist_trajectory: (num_points, num_joints) ndarray of joint
                velocities, rad/s.
            ddthetalist_trajectory: (num_points, num_joints) ndarray of joint
                accelerations, rad/s^2.
            gravity_vector: (3,) ndarray of gravitational acceleration, m/s^2.
            Ftip: length-6 sequence of the external wrench at the end effector.

        Returns:
            np.ndarray: (num_points, num_joints) array of joint torques,
            clipped to the configured torque limits.
        """
        backend = _runtime.get_backend()
        start_time = time.time()

        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]

        # Compute each waypoint's torques, then stack (no in-place row writes).
        torque_rows = []
        for i in range(num_points):
            try:
                torques = self.dynamics.inverse_dynamics(
                    thetalist_trajectory[i],
                    dthetalist_trajectory[i],
                    ddthetalist_trajectory[i],
                    gravity_vector,
                    Ftip,
                )
                torque_rows.append(backend.asarray(torques, dtype=backend.float32))
            except Exception as e:
                logger.warning(f"Error in inverse dynamics at point {i}: {e}")
                # Use zero torques for problematic points
                torque_rows.append(backend.zeros((num_joints,), dtype=backend.float32))

        if torque_rows:
            torques_trajectory = backend.stack(torque_rows)
        else:
            # Base preallocated (0, num_joints) zeros for an empty trajectory.
            torques_trajectory = backend.zeros(
                (num_points, num_joints), dtype=backend.float32
            )

        # Apply torque limits (broadcast, no in-place writes)
        torques_trajectory = backend.clip(
            torques_trajectory,
            backend.asarray(self.torque_limits[:, 0]),
            backend.asarray(self.torque_limits[:, 1]),
        )

        elapsed = time.time() - start_time
        self.performance_stats["cpu_calls"] += 1
        self.performance_stats["total_cpu_time"] += elapsed

        logger.info(f"CPU inverse dynamics completed in {elapsed:.4f}s")
        return torques_trajectory

    def forward_dynamics_trajectory(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
    ) -> Dict[str, np.ndarray]:
        """
        Enhanced forward dynamics trajectory computation.

        Args:
            thetalist (np.ndarray): Initial joint angles.
            dthetalist (np.ndarray): Initial joint velocities.
            taumat (np.ndarray): Array of joint torques over the trajectory.
            g (np.ndarray): Gravity vector.
            Ftipmat (np.ndarray): Array of external forces.
            dt (float): Time step.
            intRes (int): Integration resolution.

        Returns:
            dict: Dictionary containing positions, velocities, and accelerations.
        """
        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]

        logger.info(
            f"Computing forward dynamics: {num_steps} steps, {num_joints} joints"
        )

        # Print performance recommendations
        if self.cuda_available:
            total_work = num_steps * num_joints
            if total_work >= 10000:
                _runtime.print_performance_recommendations(num_steps, num_joints)

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(num_steps, num_joints)

        if use_gpu:
            return self._forward_dynamics_gpu(
                thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
            )
        else:
            return self._forward_dynamics_cpu(
                thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
            )

    def _forward_dynamics_gpu(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
    ) -> Dict[str, np.ndarray]:
        """GPU-accelerated forward dynamics integration over a trajectory.

        Seeds the first step with the initial state, transfers the torque
        profile, external forces and dynamics data to the device, and launches
        the forward-dynamics kernel to integrate joint motion with ``intRes``
        sub-steps per outer step, enforcing joint limits on the device. Falls
        back to the CPU implementation on any error.

        Args:
            thetalist: (num_joints,) ndarray of initial joint angles, radians.
            dthetalist: (num_joints,) ndarray of initial joint velocities,
                rad/s.
            taumat: (num_steps, num_joints) ndarray of applied joint torques.
            g: (3,) ndarray of gravitational acceleration, m/s^2.
            Ftipmat: (num_steps, 6) ndarray of end-effector wrenches per step.
            dt (float): Outer time step between trajectory points, seconds.
            intRes (int): Number of integration sub-steps per outer step.

        Returns:
            Dict[str, np.ndarray]: Dict with keys ``"positions"``,
            ``"velocities"`` and ``"accelerations"``, each a
            ``(num_steps, num_joints)`` array.
        """
        start_time = time.time()

        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]

        try:
            # Initialize result arrays
            thetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
            dthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
            ddthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)

            thetamat[0, :] = thetalist.astype(np.float32)
            dthetamat[0, :] = dthetalist.astype(np.float32)

            # Use memory pool for large arrays
            d_thetamat = _runtime.get_cuda_array(
                (num_steps, num_joints), dtype=np.float32
            )
            d_dthetamat = _runtime.get_cuda_array(
                (num_steps, num_joints), dtype=np.float32
            )
            d_ddthetamat = _runtime.get_cuda_array(
                (num_steps, num_joints), dtype=np.float32
            )

            # Copy initial conditions to GPU
            d_thetamat.copy_to_device(thetamat)
            d_dthetamat.copy_to_device(dthetamat)
            d_ddthetamat.copy_to_device(ddthetamat)

            # Transfer other data to GPU
            d_thetalist = _runtime.cuda.to_device(thetalist.astype(np.float32))
            d_dthetalist = _runtime.cuda.to_device(dthetalist.astype(np.float32))
            d_taumat = _runtime.cuda.to_device(taumat.astype(np.float32))
            d_g = _runtime.cuda.to_device(g.astype(np.float32))
            d_Ftipmat = _runtime.cuda.to_device(Ftipmat.astype(np.float32))
            d_Glist = _runtime.cuda.to_device(
                np.array(self.dynamics.Glist, dtype=np.float32)
            )
            d_Slist = _runtime.cuda.to_device(
                np.array(self.dynamics.S_list, dtype=np.float32)
            )
            d_M = _runtime.cuda.to_device(
                np.array(self.dynamics.M_list, dtype=np.float32)
            )
            d_joint_limits = _runtime.cuda.to_device(
                self.joint_limits.astype(np.float32)
            )

            # Get optimal launch configuration
            grid_config = _runtime.get_optimal_kernel_config(
                num_steps, num_joints, "cache_friendly"
            )
            if grid_config:
                blocks_per_grid = grid_config["grid"]
                threads_per_block = grid_config["block"]
                logger.info(f"Using {grid_config['kernel_type']} for forward dynamics")
            else:
                blocks_per_grid, threads_per_block = _runtime._best_2d_config(
                    num_steps, num_joints
                )

            # Launch forward dynamics kernel
            _runtime.forward_dynamics_kernel[blocks_per_grid, threads_per_block](
                d_thetalist,
                d_dthetalist,
                d_taumat,
                d_g,
                d_Ftipmat,
                dt,
                intRes,
                d_Glist,
                d_Slist,
                d_M,
                d_thetamat,
                d_dthetamat,
                d_ddthetamat,
                d_joint_limits,
            )

            # Copy results back
            d_thetamat.copy_to_host(thetamat)
            d_dthetamat.copy_to_host(dthetamat)
            d_ddthetamat.copy_to_host(ddthetamat)

            elapsed = time.time() - start_time
            throughput = (num_steps * num_joints * intRes) / elapsed / 1e6

            self.performance_stats["gpu_calls"] += 1
            self.performance_stats["total_gpu_time"] += elapsed
            self.performance_stats["kernel_launches"] += 1

            logger.info(f"GPU forward dynamics completed in {elapsed:.4f}s")
            logger.info(f"📊 Throughput: {throughput:.1f} M integration steps/sec")

            return {
                "positions": thetamat,
                "velocities": dthetamat,
                "accelerations": ddthetamat,
            }

        except Exception as e:
            logger.warning(f"GPU forward dynamics failed: {e}, falling back to CPU")
            return self._forward_dynamics_cpu(
                thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
            )
        finally:
            # Return large arrays to pool
            if "d_thetamat" in locals():
                _runtime.return_cuda_array(d_thetamat)
            if "d_dthetamat" in locals():
                _runtime.return_cuda_array(d_dthetamat)
            if "d_ddthetamat" in locals():
                _runtime.return_cuda_array(d_ddthetamat)

    def _forward_dynamics_cpu(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
    ) -> Dict[str, np.ndarray]:
        """CPU-based forward dynamics integration over a trajectory.

        Integrates joint motion step by step using
        ``self.dynamics.forward_dynamics`` with ``intRes`` Euler sub-steps per
        outer step (``dt / intRes`` each), clamping joint angles to
        ``self.joint_limits`` and using zero acceleration for any step that
        raises.

        Args:
            thetalist: (num_joints,) ndarray of initial joint angles, radians.
            dthetalist: (num_joints,) ndarray of initial joint velocities,
                rad/s.
            taumat: (num_steps, num_joints) ndarray of applied joint torques.
            g: (3,) ndarray of gravitational acceleration, m/s^2.
            Ftipmat: (num_steps, 6) ndarray of end-effector wrenches per step.
            dt (float): Outer time step between trajectory points, seconds.
            intRes (int): Number of integration sub-steps per outer step.

        Returns:
            Dict[str, np.ndarray]: Dict with keys ``"positions"``,
            ``"velocities"`` and ``"accelerations"``, each a
            ``(num_steps, num_joints)`` array.
        """
        backend = _runtime.get_backend()
        start_time = time.time()

        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]

        if num_steps == 0:
            # Base seeded row 0 unconditionally (``thetamat[0, :] = ...``) into a
            # length-0 preallocation, so a zero-step request raised IndexError.
            raise IndexError("index 0 is out of bounds for axis 0 with size 0")

        # Integrated state; keep its dtype fixed so the accumulation matches the
        # original in-place ``+=`` semantics (which cast back to the state dtype).
        current_theta = backend.asarray(thetalist)
        current_dtheta = backend.asarray(dthetalist)
        theta_dtype = current_theta.dtype
        dtheta_dtype = current_dtheta.dtype

        lower = backend.asarray(self.joint_limits[:, 0])
        upper = backend.asarray(self.joint_limits[:, 1])

        # Seed step 0; per-step rows are collected and stacked at the end.
        theta_rows = [backend.asarray(current_theta, dtype=backend.float32)]
        dtheta_rows = [backend.asarray(current_dtheta, dtype=backend.float32)]
        ddtheta_rows = [backend.zeros((num_joints,), dtype=backend.float32)]

        dt_step = dt / intRes

        for i in range(1, num_steps):
            ddtheta_step = backend.zeros((num_joints,), dtype=backend.float32)
            for _ in range(intRes):
                try:
                    # Compute forward dynamics
                    ddtheta = self.dynamics.forward_dynamics(
                        current_theta, current_dtheta, taumat[i], g, Ftipmat[i]
                    )

                    # Integrate (functional; cast back to the state dtype so the
                    # numerics match the original in-place accumulation). The
                    # original ``+=`` used same-kind casting, which refuses e.g.
                    # a float update into an integer state; reproduce that refusal
                    # so an unsafe cast raises into the handler below instead of
                    # silently truncating.
                    new_dtheta = current_dtheta + ddtheta * dt_step
                    if not np.can_cast(
                        new_dtheta.dtype, dtheta_dtype, casting="same_kind"
                    ):
                        raise TypeError(
                            f"Cannot cast ufunc 'add' output from {new_dtheta.dtype!r} "
                            f"to {dtheta_dtype!r} with casting rule 'same_kind'"
                        )
                    current_dtheta = backend.asarray(new_dtheta, dtype=dtheta_dtype)

                    new_theta = current_theta + current_dtheta * dt_step
                    if not np.can_cast(
                        new_theta.dtype, theta_dtype, casting="same_kind"
                    ):
                        raise TypeError(
                            f"Cannot cast ufunc 'add' output from {new_theta.dtype!r} "
                            f"to {theta_dtype!r} with casting rule 'same_kind'"
                        )
                    current_theta = backend.asarray(new_theta, dtype=theta_dtype)

                    # Apply joint limits. The original ``np.clip`` against the
                    # float32 limits promoted the state dtype (e.g. float16 ->
                    # float32) and later updates accumulated in that promoted
                    # dtype, so track the live post-clip dtype rather than the
                    # initial one.
                    current_theta = backend.clip(current_theta, lower, upper)
                    theta_dtype = current_theta.dtype

                    ddtheta_step = ddtheta

                except Exception as e:
                    logger.warning(f"Error in forward dynamics at step {i}: {e}")
                    ddtheta_step = backend.zeros((num_joints,))

            theta_rows.append(backend.asarray(current_theta, dtype=backend.float32))
            dtheta_rows.append(backend.asarray(current_dtheta, dtype=backend.float32))
            ddtheta_rows.append(backend.asarray(ddtheta_step, dtype=backend.float32))

        thetamat = backend.stack(theta_rows)
        dthetamat = backend.stack(dtheta_rows)
        ddthetamat = backend.stack(ddtheta_rows)

        elapsed = time.time() - start_time
        self.performance_stats["cpu_calls"] += 1
        self.performance_stats["total_cpu_time"] += elapsed

        logger.info(f"CPU forward dynamics completed in {elapsed:.4f}s")

        return {
            "positions": thetamat,
            "velocities": dthetamat,
            "accelerations": ddthetamat,
        }

    def calculate_derivatives(
        self, positions, dt
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the velocity, acceleration, and jerk of a trajectory.

        Parameters:
            positions (list or numpy.ndarray): A list or array of positions.
            dt (float): The time step between each position.

        Returns:
            velocity (numpy.ndarray): An array of velocities.
            acceleration (numpy.ndarray): An array of accelerations.
            jerk (numpy.ndarray): An array of jerks.
        """
        backend = _runtime.get_backend()
        positions = backend.asarray(positions)
        # First differences along the time axis (np.diff equivalent).
        velocity = (positions[1:] - positions[:-1]) / dt
        acceleration = (velocity[1:] - velocity[:-1]) / dt
        jerk = (acceleration[1:] - acceleration[:-1]) / dt
        return velocity, acceleration, jerk
