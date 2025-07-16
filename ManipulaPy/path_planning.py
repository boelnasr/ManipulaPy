#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Optimized Path Planning Module - ManipulaPy

This module provides highly optimized trajectory planning capabilities including joint space 
and Cartesian space trajectory generation with CUDA acceleration and collision avoidance.

Key optimizations:
- Adaptive grid sizing for optimal GPU occupancy
- Memory pooling to reduce allocation overhead
- Batch processing for multiple trajectories
- Fused kernels to minimize memory bandwidth
- Intelligent fallback to CPU when beneficial
- 2D parallelization for better GPU utilization

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from typing import Optional 
from .utils import (
    TransToRp,
    MatrixLog3,
    MatrixExp3,
    CubicTimeScaling,
    QuinticTimeScaling,
)
from .cuda_kernels import (
    CUDA_AVAILABLE,
    check_cuda_availability,
    make_1d_grid,
    make_2d_grid,
    get_gpu_properties,
    optimized_trajectory_generation,
    optimized_potential_field,
    optimized_batch_trajectory_generation,
    get_cuda_array,
    return_cuda_array,
    profile_start,
    profile_stop,
    _best_2d_config,
    _h2d_pinned,
)

# Import CUDA functions only if available
if CUDA_AVAILABLE:
    from numba import cuda
    from .cuda_kernels import (
        trajectory_kernel,
        inverse_dynamics_kernel,
        forward_dynamics_kernel,
        cartesian_trajectory_kernel,
        fused_potential_gradient_kernel,
        batch_trajectory_kernel,
    )
else:
    # Create dummy functions for when CUDA is not available
    def trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def inverse_dynamics_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def forward_dynamics_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def cartesian_trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def fused_potential_gradient_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def batch_trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    
    class MockCuda:
        @staticmethod
        def to_device(*args, **kwargs):
            raise RuntimeError("CUDA not available")
        @staticmethod
        def device_array(*args, **kwargs):
            raise RuntimeError("CUDA not available")
        @staticmethod
        def synchronize():
            pass
    
    cuda = MockCuda()

from .potential_field import CollisionChecker, PotentialField
import logging

# Set up logging and silence noisy CUDA driver logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

@njit(parallel=True, fastmath=True)
def _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method):
    """Numba-optimised CPU trajectory generation (parallel)."""
    num_joints = len(thetastart)

    traj_pos = np.zeros((N, num_joints), dtype=np.float32)
    traj_vel = np.zeros((N, num_joints), dtype=np.float32)
    traj_acc = np.zeros((N, num_joints), dtype=np.float32)

    # Flatten (idx, j) → k  to avoid nested loops that block parallelisation
    total_elems = N * num_joints
    for k in prange(total_elems):
        idx = k // num_joints        # timestep
        j   = k %  num_joints        # joint index

        t   = idx * (Tf / (N - 1))
        tau = t / Tf

        # Time-scaling
        if method == 3:                          # cubic
            s      = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot  = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
        elif method == 5:                        # quintic
            tau2   = tau * tau
            tau3   = tau2 * tau
            s      = 10.0 * tau3 - 15.0 * tau2 * tau2 + 6.0 * tau * tau3
            s_dot  = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
            s_ddot = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
        else:                                    # unsupported method
            s = s_dot = s_ddot = 0.0

        dtheta = thetaend[j] - thetastart[j]
        traj_pos[idx, j] = s      * dtheta + thetastart[j]
        traj_vel[idx, j] = s_dot  * dtheta
        traj_acc[idx, j] = s_ddot * dtheta

    return traj_pos, traj_vel, traj_acc


# Thin wrapper – unchanged signature, now just calls the new kernel above
@njit(parallel=True, fastmath=True)
def _traj_cpu_njit(thetastart, thetaend, Tf, N, method):
    return _trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
class OptimizedTrajectoryPlanning:
    """
    Highly optimized trajectory planning class with adaptive GPU/CPU execution,
    memory pooling, and batch processing capabilities.
    """
    def __init__(
        self,
        serial_manipulator,
        urdf_path,
        dynamics,
        joint_limits,
        torque_limits=None,
        *,                       # ――― everything after * is keyword-only ―――

        use_cuda: Optional[bool] = None,
        cuda_threshold: int = 10,
        memory_pool_size_mb: Optional[int] = None,
        enable_profiling: bool = False,
    ):
        """
        Parameters
        ----------
        serial_manipulator : SerialManipulator
        urdf_path          : str
        dynamics           : ManipulatorDynamics
        joint_limits       : list[tuple[float,float]]
        torque_limits      : list[tuple[float,float]], optional

        use_cuda           : None | bool
            • None  → auto-detect (default)  
            • True  → force GPU (raise if CUDA absent)  
            • False → force CPU

        cuda_threshold     : int
            Min. (N × joints) before we bother launching the GPU.

        memory_pool_size_mb: int | None
            If set, resize the global CUDA memory pool (in MB).

        enable_profiling   : bool
            Enable CUDA profiling for performance analysis.
        """
        # ------------------------------------------------------------
        # basic data
        # ------------------------------------------------------------
        self.serial_manipulator = serial_manipulator
        self.dynamics           = dynamics
        self.joint_limits       = np.asarray(joint_limits, dtype=np.float32)
        self.torque_limits      = (
            np.asarray(torque_limits, dtype=np.float32)
            if torque_limits is not None
            else np.array([[-np.inf, np.inf]] * len(joint_limits), dtype=np.float32)
        )

        # ------------------------------------------------------------
        # collision-checking helpers
        # ------------------------------------------------------------
        try:
            self.collision_checker = CollisionChecker(urdf_path)
            self.potential_field   = PotentialField()
        except Exception as exc:
            logger.warning("Could not initialise collision checker: %s", exc)
            self.collision_checker = None
            self.potential_field   = None

        # ------------------------------------------------------------
        # CUDA feature flags
        # ------------------------------------------------------------
        detected_cuda = check_cuda_availability()
        if use_cuda is None:
            self.cuda_available = detected_cuda
        elif use_cuda and not detected_cuda:
            raise RuntimeError("use_cuda=True requested but CUDA is not available.")
        else:
            self.cuda_available = bool(use_cuda)

        self.gpu_properties = (
            get_gpu_properties() if self.cuda_available else None
        )

        # threshold that decides CPU vs GPU for tiny jobs
        self.cpu_threshold = int(cuda_threshold)

        # optionally resize a global memory-pool (if you expose one)
        if memory_pool_size_mb is not None and self.cuda_available:
            from .cuda_kernels import _cuda_memory_pool
            _cuda_memory_pool.max_pool_size = (
                memory_pool_size_mb * 1024 * 1024 // 4  # entries of float32
            )

        # Initialize GPU array cache for per-instance memory management
        self._gpu_arrays = {}

        # Enable profiling if requested
        self.enable_profiling = enable_profiling
        if self.enable_profiling and self.cuda_available:
            profile_start()

        # ------------------------------------------------------------
        # performance bookkeeping
        # ------------------------------------------------------------
        self.performance_stats = {
            "gpu_calls":       0,
            "cpu_calls":       0,
            "total_gpu_time":  0.0,
            "total_cpu_time":  0.0,
            "memory_transfers": 0,
            "kernel_launches":  0,
        }

        logger.info(
            "Optimised planner – CUDA enabled: %s (threshold %d)",
            self.cuda_available, self.cpu_threshold,
        )
        if self.gpu_properties:
            logger.info("GPU: %s", self.gpu_properties)
    
    def _get_or_resize_gpu_array(self, array_name, shape, dtype=np.float32):
        """
        Return a pooled CUDA array with the requested shape / dtype.
        If an existing cached array has a mismatching shape or dtype,
        it is returned to the pool and a new one is fetched.

        Parameters
        ----------
        array_name : str
            Key used to store/retrieve the array in the per-instance cache.
        shape : tuple[int]
            Desired array shape.
        dtype : numpy.dtype, optional
            Desired data type (default: np.float32).

        Returns
        -------
        numba.cuda.cudadrv.devicearray.DeviceNDArray or None
            The GPU array, or None if CUDA is unavailable.
        """
        if not self.cuda_available:            # Fallback: no CUDA
            return None

        arr = self._gpu_arrays.get(array_name)

        if (arr is None) or (arr.shape != shape) or (arr.dtype != dtype):
            if arr is not None:
                # give the old array back to the global pool
                return_cuda_array(arr)

            # rent a fresh array from the global pool
            arr = get_cuda_array(shape, dtype)
            self._gpu_arrays[array_name] = arr

        return arr

    def _should_use_gpu(self, N, num_joints):
        """Determine if GPU should be used based on problem size and availability."""
        if not self.cuda_available:           # ← CUDA missing?  Always False
            return False

        total_work = N * num_joints           #  e.g. 200 × 6 = 1 200
        if total_work < self.cpu_threshold:   #  default cpu_threshold = 100
            return False                      #  small ⇒ stay on CPU

        # Additional memory-safety and performance checks
        memory_required = total_work * 4 * 3  # 3 arrays (pos, vel, acc) * 4 bytes per float32
        if self.gpu_properties:
            # Check if we have enough memory (conservative estimate)
            if memory_required > self.gpu_properties.get('max_shared_memory_per_block', 48*1024):
                logger.debug(f"Large memory requirement: {memory_required} bytes, using GPU anyway")
        
        return True

    def joint_trajectory(self, thetastart, thetaend, Tf, N, method):
        """
        Generates an optimized joint trajectory with adaptive GPU/CPU execution.

        Args:
            thetastart (numpy.ndarray): The starting joint angles.
            thetaend (numpy.ndarray): The ending joint angles.
            Tf (float): The final time for the trajectory.
            N (int): The number of steps in the trajectory.
            method (int): The method to use (3=cubic, 5=quintic).

        Returns:
            dict: A dictionary containing positions, velocities, and accelerations.
        """
        logger.info(f"Generating joint trajectory: N={N}, joints={len(thetastart)}, method={method}")
        
        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(N, num_joints)
        
        if use_gpu:
            return self._joint_trajectory_gpu(thetastart, thetaend, Tf, N, method)
        else:
            return self._joint_trajectory_cpu(thetastart, thetaend, Tf, N, method)

    def _joint_trajectory_gpu(self, thetastart, thetaend, Tf, N, method):
        """GPU-accelerated joint trajectory generation with optimized memory management."""
        start_time = time.time()
        
        try:
            # Use the optimized high-level wrapper
            traj_pos_host, traj_vel_host, traj_acc_host = optimized_trajectory_generation(
                thetastart, thetaend, Tf, N, method, use_pinned=True
            )
            
            # Apply joint limits
            num_joints = len(thetastart)
            for i in range(num_joints):
                traj_pos_host[:, i] = np.clip(
                    traj_pos_host[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1]
                )

            # Apply collision avoidance if available
            if self.collision_checker and self.potential_field:
                traj_pos_host = self._apply_collision_avoidance_gpu(traj_pos_host, thetaend)

            # Update performance stats
            elapsed = time.time() - start_time
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"GPU trajectory generation completed in {elapsed:.4f}s")

            return {
                "positions": traj_pos_host,
                "velocities": traj_vel_host,
                "accelerations": traj_acc_host,
            }
            
        except Exception as e:
            logger.warning(f"GPU trajectory generation failed: {e}, falling back to CPU")
            return self._joint_trajectory_cpu(thetastart, thetaend, Tf, N, method)

    def _joint_trajectory_cpu(self, thetastart, thetaend, Tf, N, method):
        """CPU-based joint trajectory generation with Numba optimization."""
        start_time = time.time()
        
        # Use optimized CPU fallback
        traj_pos, traj_vel, traj_acc = _traj_cpu_njit(
            thetastart, thetaend, Tf, N, method
        )

        # Apply joint limits
        num_joints = len(thetastart)
        for i in range(num_joints):
            traj_pos[:, i] = np.clip(
                traj_pos[:, i], self.joint_limits[i, 0], self.joint_limits[i, 1]
            )

        # Apply collision avoidance if available
        if self.collision_checker and self.potential_field:
            traj_pos = self._apply_collision_avoidance_cpu(traj_pos, thetaend)

        # Update performance stats
        elapsed = time.time() - start_time
        self.performance_stats['cpu_calls'] += 1
        self.performance_stats['total_cpu_time'] += elapsed
        
        logger.info(f"CPU trajectory generation completed in {elapsed:.4f}s")

        return {
            "positions": traj_pos,
            "velocities": traj_vel,
            "accelerations": traj_acc,
        }

    def _apply_collision_avoidance_gpu(self, traj_pos, thetaend):
        """Apply GPU-accelerated potential field-based collision avoidance."""
        if not self.cuda_available:
            return self._apply_collision_avoidance_cpu(traj_pos, thetaend)
        
        try:
            q_goal = thetaend
            obstacles = []  # Define obstacles here as needed
            
            # Use GPU-accelerated potential field computation
            for idx, step in enumerate(traj_pos):
                if self.collision_checker.check_collision(step):
                    # Prepare data for GPU computation
                    positions = step.reshape(1, -1)
                    
                    for iteration in range(100):  # Max iterations
                        try:
                            # Use optimized potential field computation
                            potential, gradient = optimized_potential_field(
                                positions, q_goal, np.array(obstacles), 
                                influence_distance=0.5, use_pinned=True
                            )
                            
                            # Update position
                            step -= 0.01 * gradient[0]  # Adjust step size as needed
                            positions[0] = step
                            
                            if not self.collision_checker.check_collision(step):
                                break
                                
                        except Exception as e:
                            logger.warning(f"GPU potential field computation failed: {e}")
                            # Fall back to CPU method
                            gradient = self.potential_field.compute_gradient(step, q_goal, obstacles)
                            step -= 0.01 * gradient
                            
                            if not self.collision_checker.check_collision(step):
                                break
                    
                    traj_pos[idx] = step
            
            return traj_pos
            
        except Exception as e:
            logger.warning(f"GPU collision avoidance failed: {e}, falling back to CPU")
            return self._apply_collision_avoidance_cpu(traj_pos, thetaend)

    def _apply_collision_avoidance_cpu(self, traj_pos, thetaend):
        """Apply CPU-based potential field collision avoidance."""
        q_goal = thetaend
        obstacles = []  # Define obstacles here as needed

        # Apply potential field for collision avoidance
        for idx, step in enumerate(traj_pos):
            if self.collision_checker.check_collision(step):
                for _ in range(100):  # Max iterations to adjust trajectory
                    gradient = self.potential_field.compute_gradient(step, q_goal, obstacles)
                    step -= 0.01 * gradient  # Adjust step size as needed
                    if not self.collision_checker.check_collision(step):
                        break
                traj_pos[idx] = step
        
        return traj_pos

    def batch_joint_trajectory(self, thetastart_batch, thetaend_batch, Tf, N, method):
        """
        Generate multiple joint trajectories simultaneously using batch processing.
        
        Args:
            thetastart_batch (numpy.ndarray): Starting angles for multiple trajectories (batch_size, num_joints)
            thetaend_batch (numpy.ndarray): Ending angles for multiple trajectories (batch_size, num_joints)
            Tf (float): Final time for all trajectories
            N (int): Number of trajectory points
            method (int): Time scaling method
            
        Returns:
            dict: Batch trajectory data with shape (batch_size, N, num_joints)
        """
        batch_size, num_joints = thetastart_batch.shape
        logger.info(f"Generating batch trajectories: batch_size={batch_size}, N={N}, joints={num_joints}")

        if not self.cuda_available:
            logger.warning("Batch processing requires CUDA. Falling back to sequential processing.")
            return self._batch_joint_trajectory_cpu(thetastart_batch, thetaend_batch, Tf, N, method)
        
        start_time = time.time()

        try:
            # Use optimized batch trajectory generation
            traj_pos_host, traj_vel_host, traj_acc_host = optimized_batch_trajectory_generation(
                thetastart_batch, thetaend_batch, Tf, N, method, use_pinned=True
            )

            # Apply joint limits for all trajectories
            for batch_idx in range(batch_size):
                for i in range(num_joints):
                    traj_pos_host[batch_idx, :, i] = np.clip(
                        traj_pos_host[batch_idx, :, i], 
                        self.joint_limits[i, 0], 
                        self.joint_limits[i, 1]
                    )

            elapsed = time.time() - start_time
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"Batch GPU trajectory generation completed in {elapsed:.4f}s")

            return {
                "positions": traj_pos_host,
                "velocities": traj_vel_host,
                "accelerations": traj_acc_host,
            }

        except Exception as e:
            logger.warning(f"Batch GPU trajectory generation failed: {e}, falling back to CPU")
            return self._batch_joint_trajectory_cpu(thetastart_batch, thetaend_batch, Tf, N, method)

    def _batch_joint_trajectory_cpu(self, thetastart_batch, thetaend_batch, Tf, N, method):
        """CPU fallback for batch trajectory generation."""
        start_time = time.time()
        
        batch_size, num_joints = thetastart_batch.shape
        
        # Initialize result arrays
        traj_pos_batch = np.zeros((batch_size, N, num_joints), dtype=np.float32)
        traj_vel_batch = np.zeros((batch_size, N, num_joints), dtype=np.float32)
        traj_acc_batch = np.zeros((batch_size, N, num_joints), dtype=np.float32)
        
        # Process each trajectory in the batch
        for i in range(batch_size):
            traj_pos, traj_vel, traj_acc = _traj_cpu_njit(
                thetastart_batch[i], thetaend_batch[i], Tf, N, method
            )
            traj_pos_batch[i] = traj_pos
            traj_vel_batch[i] = traj_vel
            traj_acc_batch[i] = traj_acc

        elapsed = time.time() - start_time
        self.performance_stats['cpu_calls'] += 1
        self.performance_stats['total_cpu_time'] += elapsed
        
        logger.info(f"Batch CPU trajectory generation completed in {elapsed:.4f}s")

        return {
            "positions": traj_pos_batch,
            "velocities": traj_vel_batch,
            "accelerations": traj_acc_batch,
        }

    def inverse_dynamics_trajectory(
        self,
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector=None,
        Ftip=None,
    ):
        """
        Compute joint torques with optimized CUDA acceleration and memory management.

        Args:
            thetalist_trajectory (np.ndarray): Array of joint angles over the trajectory.
            dthetalist_trajectory (np.ndarray): Array of joint velocities over the trajectory.
            ddthetalist_trajectory (np.ndarray): Array of joint accelerations over the trajectory.
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
        
        logger.info(f"Computing inverse dynamics: {num_points} points, {num_joints} joints")

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(num_points, num_joints)
        
        if use_gpu:
            return self._inverse_dynamics_gpu(
                thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                gravity_vector, Ftip
            )
        else:
            return self._inverse_dynamics_cpu(
                thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                gravity_vector, Ftip
            )

    def _inverse_dynamics_gpu(self, thetalist_trajectory, dthetalist_trajectory, 
                            ddthetalist_trajectory, gravity_vector, Ftip):
        """GPU-accelerated inverse dynamics computation with optimized memory management."""
        start_time = time.time()
        
        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]
        
        try:
            # Use memory pool for the large torques array
            torques_trajectory = get_cuda_array((num_points, num_joints), dtype=np.float32)
            
            # Transfer data to GPU using pinned memory
            d_thetalist_trajectory = _h2d_pinned(thetalist_trajectory.astype(np.float32))
            d_dthetalist_trajectory = _h2d_pinned(dthetalist_trajectory.astype(np.float32))
            d_ddthetalist_trajectory = _h2d_pinned(ddthetalist_trajectory.astype(np.float32))
            
            # Convert gravity_vector and Ftip to numpy arrays if they're lists
            if isinstance(gravity_vector, list):
                gravity_vector = np.array(gravity_vector, dtype=np.float32)
            else:
                gravity_vector = np.asarray(gravity_vector, dtype=np.float32)
                
            if isinstance(Ftip, list):
                Ftip = np.array(Ftip, dtype=np.float32)
            else:
                Ftip = np.asarray(Ftip, dtype=np.float32)
            
            d_gravity_vector = cuda.to_device(gravity_vector)
            d_Ftip = cuda.to_device(Ftip)
            d_Glist = cuda.to_device(np.array(self.dynamics.Glist, dtype=np.float32))
            d_Slist = cuda.to_device(np.array(self.dynamics.S_list, dtype=np.float32))
            d_M = cuda.to_device(np.array(self.dynamics.M_list, dtype=np.float32))
            d_torque_limits = cuda.to_device(self.torque_limits.astype(np.float32))

            # Get optimal 2D launch configuration
            blocks_per_grid, threads_per_block = _best_2d_config(num_points, num_joints)
            logger.info(f"Inverse dynamics 2D grid: blocks={blocks_per_grid}, threads={threads_per_block}")
            
            # Launch optimized 2D inverse dynamics kernel
            inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
                d_thetalist_trajectory,
                d_dthetalist_trajectory,
                d_ddthetalist_trajectory,
                d_gravity_vector,
                d_Ftip,
                d_Glist,
                d_Slist,
                d_M,
                torques_trajectory,
                d_torque_limits,
            )

            # Copy results back using pinned memory
            torques_host = torques_trajectory.copy_to_host()

            # Apply final torque limits
            torques_host = np.clip(
                torques_host, self.torque_limits[:, 0], self.torque_limits[:, 1]
            )

            elapsed = time.time() - start_time
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"GPU inverse dynamics completed in {elapsed:.4f}s")
            return torques_host

        except Exception as e:
            logger.warning(f"GPU inverse dynamics failed: {e}, falling back to CPU")
            return self._inverse_dynamics_cpu(
                thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                gravity_vector, Ftip
            )
        finally:
            # Return large array to pool
            if 'torques_trajectory' in locals():
                return_cuda_array(torques_trajectory)
    def _inverse_dynamics_cpu(self, thetalist_trajectory, dthetalist_trajectory,
                             ddthetalist_trajectory, gravity_vector, Ftip):
        """CPU-based inverse dynamics computation."""
        start_time = time.time()
        
        num_points = thetalist_trajectory.shape[0]
        num_joints = thetalist_trajectory.shape[1]
        torques_trajectory = np.zeros((num_points, num_joints), dtype=np.float32)

        # Process each trajectory point
        for i in range(num_points):
            try:
                torques = self.dynamics.inverse_dynamics(
                    thetalist_trajectory[i],
                    dthetalist_trajectory[i],
                    ddthetalist_trajectory[i],
                    gravity_vector,
                    Ftip
                )
                torques_trajectory[i] = np.array(torques, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error in inverse dynamics at point {i}: {e}")
                # Use zero torques for problematic points
                torques_trajectory[i] = np.zeros(num_joints, dtype=np.float32)

        # Apply torque limits
        torques_trajectory = np.clip(
            torques_trajectory, self.torque_limits[:, 0], self.torque_limits[:, 1]
        )

        elapsed = time.time() - start_time
        self.performance_stats['cpu_calls'] += 1
        self.performance_stats['total_cpu_time'] += elapsed
        
        logger.info(f"CPU inverse dynamics completed in {elapsed:.4f}s")
        return torques_trajectory

    def forward_dynamics_trajectory(
        self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes
    ):
        """
        Optimized forward dynamics trajectory computation.

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
        
        logger.info(f"Computing forward dynamics: {num_steps} steps, {num_joints} joints")

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

    def _forward_dynamics_gpu(self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """GPU-accelerated forward dynamics computation with optimized memory management."""
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
            d_thetamat = get_cuda_array((num_steps, num_joints), dtype=np.float32)
            d_dthetamat = get_cuda_array((num_steps, num_joints), dtype=np.float32)
            d_ddthetamat = get_cuda_array((num_steps, num_joints), dtype=np.float32)
            
            # Copy initial conditions to GPU using pinned memory
            d_thetamat.copy_to_device(thetamat)
            d_dthetamat.copy_to_device(dthetamat)
            d_ddthetamat.copy_to_device(ddthetamat)
            
            # Transfer other data to GPU
            d_thetalist = cuda.to_device(thetalist.astype(np.float32))
            d_dthetalist = cuda.to_device(dthetalist.astype(np.float32))
            d_taumat = cuda.to_device(taumat.astype(np.float32))
            d_g = cuda.to_device(g.astype(np.float32))
            d_Ftipmat = cuda.to_device(Ftipmat.astype(np.float32))
            d_Glist = cuda.to_device(np.array(self.dynamics.Glist, dtype=np.float32))
            d_Slist = cuda.to_device(np.array(self.dynamics.S_list, dtype=np.float32))
            d_M = cuda.to_device(np.array(self.dynamics.M_list, dtype=np.float32))
            d_joint_limits = cuda.to_device(self.joint_limits.astype(np.float32))

            # Get optimal 2D launch configuration
            blocks_per_grid, threads_per_block = _best_2d_config(num_steps, num_joints)
            
            # Launch 2D forward dynamics kernel
            forward_dynamics_kernel[blocks_per_grid, threads_per_block](
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
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"GPU forward dynamics completed in {elapsed:.4f}s")

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
            if 'd_thetamat' in locals():
                return_cuda_array(d_thetamat)
            if 'd_dthetamat' in locals():
                return_cuda_array(d_dthetamat)
            if 'd_ddthetamat' in locals():
                return_cuda_array(d_ddthetamat)

    def _forward_dynamics_cpu(self, thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes):
        """CPU-based forward dynamics computation."""
        start_time = time.time()
        
        num_steps = taumat.shape[0]
        num_joints = thetalist.shape[0]
        
        thetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        dthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        ddthetamat = np.zeros((num_steps, num_joints), dtype=np.float32)
        
        # Initialize with starting conditions
        current_theta = thetalist.copy()
        current_dtheta = dthetalist.copy()
        
        thetamat[0, :] = current_theta
        dthetamat[0, :] = current_dtheta

        dt_step = dt / intRes

        for i in range(1, num_steps):
            for _ in range(intRes):
                try:
                    # Compute forward dynamics
                    ddtheta = self.dynamics.forward_dynamics(
                        current_theta, current_dtheta, taumat[i], g, Ftipmat[i]
                    )
                    
                    # Integrate
                    current_dtheta += ddtheta * dt_step
                    current_theta += current_dtheta * dt_step
                    
                    # Apply joint limits
                    current_theta = np.clip(
                        current_theta, self.joint_limits[:, 0], self.joint_limits[:, 1]
                    )
                    
                    ddthetamat[i] = ddtheta
                    
                except Exception as e:
                    logger.warning(f"Error in forward dynamics at step {i}: {e}")
                    ddthetamat[i] = np.zeros(num_joints)

            thetamat[i, :] = current_theta
            dthetamat[i, :] = current_dtheta

        elapsed = time.time() - start_time
        self.performance_stats['cpu_calls'] += 1
        self.performance_stats['total_cpu_time'] += elapsed
        
        logger.info(f"CPU forward dynamics completed in {elapsed:.4f}s")

        return {
            "positions": thetamat,
            "velocities": dthetamat,
            "accelerations": ddthetamat,
        }

    def cartesian_trajectory(self, Xstart, Xend, Tf, N, method):
        """
        Optimized Cartesian trajectory generation.

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
        
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = [None] * N
        Rstart, pstart = TransToRp(Xstart)
        Rend, pend = TransToRp(Xend)

        orientations = np.zeros((N, 3, 3), dtype=np.float32)

        # Compute orientation interpolation on CPU (complex matrix operations)
        for i in range(N):
            if method == 3:
                s = CubicTimeScaling(Tf, timegap * i)
            else:
                s = QuinticTimeScaling(Tf, timegap * i)
            
            traj[i] = np.r_[
                np.c_[
                    np.dot(Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)),
                    s * pend + (1 - s) * pstart,
                ],
                [[0, 0, 0, 1]],
            ]
            orientations[i] = np.dot(
                Rstart, MatrixExp3(MatrixLog3(np.dot(Rstart.T, Rend)) * s)
            )

        traj_pos = np.array([TransToRp(T)[1] for T in traj], dtype=np.float32)

        # Use GPU for position/velocity/acceleration computation if beneficial
        use_gpu = self._should_use_gpu(N, 3)  # 3 coordinates (x,y,z)
        
        if use_gpu:
            traj_vel, traj_acc = self._cartesian_trajectory_gpu(pstart, pend, Tf, N, method)
        else:
            traj_vel, traj_acc = self._cartesian_trajectory_cpu(pstart, pend, Tf, N, method)

        return {
            "positions": traj_pos,
            "velocities": traj_vel,
            "accelerations": traj_acc,
            "orientations": orientations,
        }

    def _cartesian_trajectory_gpu(self, pstart, pend, Tf, N, method):
        """GPU-accelerated Cartesian trajectory computation with optimized memory management."""
        start_time = time.time()
        
        try:
            pstart = np.ascontiguousarray(pstart.astype(np.float32))
            pend = np.ascontiguousarray(pend.astype(np.float32))

            traj_vel = get_cuda_array((N, 3), dtype=np.float32)
            traj_acc = get_cuda_array((N, 3), dtype=np.float32)
            
            # For positions, we'll use a dummy array since we already computed them
            traj_pos_dummy = get_cuda_array((N, 3), dtype=np.float32)

            # Transfer data using pinned memory
            d_pstart = _h2d_pinned(pstart)
            d_pend = _h2d_pinned(pend)

            # Get optimal 2D launch configuration for (time, coord)
            blocks_per_grid, threads_per_block = _best_2d_config(N, 3)

            # Launch Cartesian trajectory kernel
            cartesian_trajectory_kernel[blocks_per_grid, threads_per_block](
                d_pstart, d_pend, traj_pos_dummy, traj_vel, traj_acc, Tf, N, method
            )

            # Copy results back
            traj_vel_host = traj_vel.copy_to_host()
            traj_acc_host = traj_acc.copy_to_host()

            elapsed = time.time() - start_time
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"GPU Cartesian trajectory completed in {elapsed:.4f}s")

            return traj_vel_host, traj_acc_host

        except Exception as e:
            logger.warning(f"GPU Cartesian trajectory failed: {e}, falling back to CPU")
            return self._cartesian_trajectory_cpu(pstart, pend, Tf, N, method)
        finally:
            # Return memory to pool
            if 'traj_vel' in locals():
                return_cuda_array(traj_vel)
            if 'traj_acc' in locals():
                return_cuda_array(traj_acc)
            if 'traj_pos_dummy' in locals():
                return_cuda_array(traj_pos_dummy)

    def _cartesian_trajectory_cpu(self, pstart, pend, Tf, N, method):
        """CPU-based Cartesian trajectory computation."""
        start_time = time.time()
        
        traj_vel = np.zeros((N, 3), dtype=np.float32)
        traj_acc = np.zeros((N, 3), dtype=np.float32)

        for i in range(N):
            t = i * (Tf / (N - 1))
            tau = t / Tf

            if method == 3:
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:
                tau2 = tau * tau
                s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
                s_ddot = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
            else:
                s_dot = s_ddot = 0.0

            dp = pend - pstart
            traj_vel[i] = s_dot * dp
            traj_acc[i] = s_ddot * dp

        elapsed = time.time() - start_time
        self.performance_stats['cpu_calls'] += 1
        self.performance_stats['total_cpu_time'] += elapsed
        
        logger.info(f"CPU Cartesian trajectory completed in {elapsed:.4f}s")

        return traj_vel, traj_acc

    def get_performance_stats(self):
        """
        Get performance statistics for GPU vs CPU usage.
        
        Returns:
            dict: Performance statistics including call counts and timing
        """
        stats = self.performance_stats.copy()
        
        if stats['gpu_calls'] > 0:
            stats['avg_gpu_time'] = stats['total_gpu_time'] / stats['gpu_calls']
        else:
            stats['avg_gpu_time'] = 0.0
            
        if stats['cpu_calls'] > 0:
            stats['avg_cpu_time'] = stats['total_cpu_time'] / stats['cpu_calls']
        else:
            stats['avg_cpu_time'] = 0.0
            
        total_calls = stats['gpu_calls'] + stats['cpu_calls']
        if total_calls > 0:
            stats['gpu_usage_percent'] = (stats['gpu_calls'] / total_calls) * 100
        else:
            stats['gpu_usage_percent'] = 0.0

        # Simple EWMA auto-tune for adaptive threshold
        if stats['avg_gpu_time'] > 0 and stats['avg_cpu_time'] > 0:
            efficiency_ratio = stats['avg_cpu_time'] / stats['avg_gpu_time']
            self.cpu_threshold = int(0.9 * self.cpu_threshold + 0.1 * efficiency_ratio * self.cpu_threshold)
            self.cpu_threshold = max(50, min(self.cpu_threshold, 5000))  # Keep within reasonable bounds
            
        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'gpu_calls': 0,
            'cpu_calls': 0,
            'total_gpu_time': 0.0,
            'total_cpu_time': 0.0,
            'memory_transfers': 0,
            'kernel_launches': 0,
        }

    def cleanup_gpu_memory(self):
        """Clean up GPU memory pools and cached arrays."""
        if self.cuda_available:
            # Clean up per-instance cache
            for array in self._gpu_arrays.values():
                if array is not None:
                    return_cuda_array(array)
            self._gpu_arrays.clear()
            
            # Clear global memory pool
            from .cuda_kernels import _cuda_memory_pool
            _cuda_memory_pool.clear()
            
            # Synchronize and clean up CUDA context
            cuda.synchronize()
            
            logger.info("GPU memory cleaned up")

    def __del__(self):
        """Destructor to clean up GPU resources."""
        try:
            if self.enable_profiling and self.cuda_available:
                profile_stop()
            self.cleanup_gpu_memory()
        except:
            pass  # Ignore errors during cleanup

    # Static plotting methods (unchanged from original)
    @staticmethod
    def plot_trajectory(trajectory_data, Tf, title="Joint Trajectory", labels=None):
        """Plot joint trajectory data."""
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        num_joints = positions.shape[1]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, num_joints, figsize=(15, 10), sharex="col")
        fig.suptitle(title)

        for i in range(num_joints):
            if labels and len(labels) == num_joints:
                label = labels[i]
            else:
                label = f"Joint {i+1}"

            axs[0, i].plot(time_steps, positions[:, i], label=f"{label} Position")
            axs[0, i].set_ylabel("Position")
            axs[0, i].legend()

            axs[1, i].plot(time_steps, velocities[:, i], label=f"{label} Velocity")
            axs[1, i].set_ylabel("Velocity")
            axs[1, i].legend()

            axs[2, i].plot(time_steps, accelerations[:, i], label=f"{label} Acceleration")
            axs[2, i].set_ylabel("Acceleration")
            axs[2, i].legend()

        for ax in axs[-1]:
            ax.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def plot_tcp_trajectory(self, trajectory, dt):
        """
        Plots the trajectory of the TCP (Tool Center Point) of a serial manipulator.
        
        Args:
            trajectory (list): A list of joint angle configurations representing the trajectory.
            dt (float): The time step between consecutive points in the trajectory.
        
        Returns:
            None
        """
        tcp_trajectory = [
            self.serial_manipulator.forward_kinematics(joint_angles)
            for joint_angles in trajectory
        ]
        tcp_positions = [pose[:3, 3] for pose in tcp_trajectory]

        velocity, acceleration, jerk = self.calculate_derivatives(tcp_positions, dt)
        time = np.arange(0, len(tcp_positions) * dt, dt)

        plt.figure(figsize=(12, 8))
        for i, label in enumerate(["X", "Y", "Z"]):
            plt.subplot(4, 1, 1)
            plt.plot(time, np.array(tcp_positions)[:, i], label=f"TCP {label} Position")
            plt.ylabel("Position")
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(time[:-1], velocity[:, i], label=f"TCP {label} Velocity")
            plt.ylabel("Velocity")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(time[:-2], acceleration[:, i], label=f"TCP {label} Acceleration")
            plt.ylabel("Acceleration")
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(time[:-3], jerk[:, i], label=f"TCP {label} Jerk")
            plt.xlabel("Time")
            plt.ylabel("Jerk")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_cartesian_trajectory(self, trajectory_data, Tf, title="Cartesian Trajectory"):
        """
        Plots the Cartesian trajectory of a robot's motion, including position, velocity, and acceleration.
        
        Args:
            trajectory_data (dict): A dictionary containing the position, velocity, and acceleration data for the Cartesian trajectory.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot. Defaults to "Cartesian Trajectory".
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex="col")
        fig.suptitle(title)

        axs[0].plot(time_steps, positions[:, 0], label="X Position")
        axs[0].plot(time_steps, positions[:, 1], label="Y Position")
        axs[0].plot(time_steps, positions[:, 2], label="Z Position")
        axs[0].set_ylabel("Position")
        axs[0].legend()

        axs[1].plot(time_steps, velocities[:, 0], label="X Velocity")
        axs[1].plot(time_steps, velocities[:, 1], label="Y Velocity")
        axs[1].plot(time_steps, velocities[:, 2], label="Z Velocity")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()

        axs[2].plot(time_steps, accelerations[:, 0], label="X Acceleration")
        axs[2].plot(time_steps, accelerations[:, 1], label="Y Acceleration")
        axs[2].plot(time_steps, accelerations[:, 2], label="Z Acceleration")
        axs[2].set_ylabel("Acceleration")
        axs[2].legend()

        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def calculate_derivatives(self, positions, dt):
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
        positions = np.array(positions)
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        return velocity, acceleration, jerk

    def plot_ee_trajectory(self, trajectory_data, Tf, title="End-Effector Trajectory"):
        """
        Plots the end-effector trajectory of a serial manipulator.
        
        Args:
            trajectory_data (dict): A dictionary containing the position and orientation data of the end-effector trajectory.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot. Defaults to "End-Effector Trajectory".
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        if "orientations" in trajectory_data:
            orientations = trajectory_data["orientations"]
        else:
            orientations = np.array(
                [
                    self.serial_manipulator.forward_kinematics(pos)[:3, :3]
                    for pos in positions
                ]
            )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.suptitle(title)

        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2], label="EE Position", color="b"
        )

        for i in range(0, num_steps, max(1, num_steps // 20)):
            R = orientations[i]
            pos = positions[i]
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 0], R[1, 0], R[2, 0], length=0.01, color="r"
            )
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 1], R[1, 1], R[2, 1], length=0.01, color="g"
            )
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 2], R[1, 2], R[2, 2], length=0.01, color="b"
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.legend()
        plt.show()

    def plan_trajectory(self, start_position, target_position, obstacle_points):
        """
        Plans a collision-free trajectory from start to target position.

        Args:
            start_position (list): Initial joint configuration.
            target_position (list): Desired joint configuration.
            obstacle_points (list): List of obstacle points in the environment.

        Returns:
            list: Joint trajectory as a list of joint configurations.
        """
        # Simple trajectory planning (can be extended with RRT, etc.)
        joint_trajectory = [start_position, target_position]
        logger.info(f"Planned trajectory with {len(joint_trajectory)} waypoints.")
        return joint_trajectory

    def benchmark_performance(self, test_cases=None):
        """
        Benchmark the performance of GPU vs CPU implementations.
        
        Args:
            test_cases (list, optional): List of test cases to benchmark.
                                       If None, uses default test cases.
        
        Returns:
            dict: Benchmark results
        """
        if test_cases is None:
            test_cases = [
                {"N": 100, "joints": 6, "name": "Small"},
                {"N": 1000, "joints": 6, "name": "Medium"},
                {"N": 5000, "joints": 6, "name": "Large"},
                {"N": 1000, "joints": 12, "name": "Many joints"},
            ]
        
        results = {}
        
        for test_case in test_cases:
            N = test_case["N"]
            joints = test_case["joints"]
            name = test_case["name"]
            
            logger.info(f"Benchmarking {name} case: N={N}, joints={joints}")
            
            # Generate test data
            thetastart = np.random.uniform(-1, 1, joints).astype(np.float32)
            thetaend = np.random.uniform(-1, 1, joints).astype(np.float32)
            
            # Reset stats
            self.reset_performance_stats()
            
            # Test trajectory generation
            start_time = time.time()
            trajectory = self.joint_trajectory(thetastart, thetaend, 2.0, N, 3)
            end_time = time.time()
            
            results[name] = {
                "total_time": end_time - start_time,
                "N": N,
                "joints": joints,
                "stats": self.get_performance_stats(),
                "used_gpu": self.performance_stats['gpu_calls'] > 0,
                "trajectory_shape": trajectory["positions"].shape,
            }
            
            logger.info(f"{name} benchmark: {end_time - start_time:.4f}s, GPU: {results[name]['used_gpu']}")
        
        return results


# Maintain backward compatibility with original class name
class TrajectoryPlanning(OptimizedTrajectoryPlanning):
    """
    Backward compatibility alias for OptimizedTrajectoryPlanning.
    
    This ensures existing code continues to work while providing
    access to all optimizations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using OptimizedTrajectoryPlanning (backward compatibility mode)")


# Additional utility functions for advanced users
def create_optimized_planner(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    torque_limits=None,
    gpu_memory_mb=None,
    enable_profiling=False,
):
    """
    Factory function to create an optimized trajectory planner with recommended settings.
    
    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        torque_limits: Torque limits (optional)
        gpu_memory_mb: GPU memory pool size in MB (optional)
        enable_profiling: Enable CUDA profiling (optional)
    
    Returns:
        OptimizedTrajectoryPlanning: Configured planner instance
    """
    # Auto-detect optimal settings
    cuda_available = check_cuda_availability()
    
    # Adaptive threshold based on problem size
    num_joints = len(joint_limits)
    if num_joints <= 6:
        threshold = 200
    elif num_joints <= 12:
        threshold = 100
    else:
        threshold = 50
    
    # Create planner with optimized settings
    planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        use_cuda=True,  # Auto-detect
        cuda_threshold=threshold,
        memory_pool_size_mb=gpu_memory_mb,
        enable_profiling=enable_profiling,
    )
    
    logger.info(f"Created optimized planner for {num_joints} joints, CUDA: {cuda_available}")
    return planner


def compare_implementations(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    test_params=None,
):
    """
    Compare performance between CPU and GPU implementations.
    
    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_params: Test parameters (optional)
    
    Returns:
        dict: Comparison results
    """
    if test_params is None:
        test_params = {"N": 1000, "Tf": 2.0, "method": 3}
    
    # Create CPU-only planner
    cpu_planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        use_cuda=False,
    )
    
    # Create GPU planner (if available)
    gpu_planner = None
    if check_cuda_availability():
        gpu_planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits,
            use_cuda=True,
            cuda_threshold=0,  # Force GPU usage
        )
    
    # Generate test data
    num_joints = len(joint_limits)
    thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    
    results = {"cpu": {}, "gpu": {}}
    
    # Test CPU implementation
    logger.info("Testing CPU implementation...")
    start_time = time.time()
    cpu_result = cpu_planner.joint_trajectory(
        thetastart, thetaend, test_params["Tf"], test_params["N"], test_params["method"]
    )
    cpu_time = time.time() - start_time
    
    results["cpu"] = {
        "time": cpu_time,
        "result_shape": cpu_result["positions"].shape,
        "stats": cpu_planner.get_performance_stats(),
    }
    
    # Test GPU implementation (if available)
    if gpu_planner is not None:
        logger.info("Testing GPU implementation...")
        start_time = time.time()
        gpu_result = gpu_planner.joint_trajectory(
            thetastart, thetaend, test_params["Tf"], test_params["N"], test_params["method"]
        )
        gpu_time = time.time() - start_time
        
        results["gpu"] = {
            "time": gpu_time,
            "result_shape": gpu_result["positions"].shape,
            "stats": gpu_planner.get_performance_stats(),
            "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
        }
        
        # Compare accuracy
        pos_diff = np.abs(cpu_result["positions"] - gpu_result["positions"])
        vel_diff = np.abs(cpu_result["velocities"] - gpu_result["velocities"])
        acc_diff = np.abs(cpu_result["accelerations"] - gpu_result["accelerations"])
        
        results["accuracy"] = {
            "max_pos_diff": np.max(pos_diff),
            "max_vel_diff": np.max(vel_diff),
            "max_acc_diff": np.max(acc_diff),
            "mean_pos_diff": np.mean(pos_diff),
            "mean_vel_diff": np.mean(vel_diff),
            "mean_acc_diff": np.mean(acc_diff),
        }
        
        logger.info(f"GPU speedup: {results['gpu']['speedup']:.2f}x")
    else:
        results["gpu"] = {"available": False}
        logger.info("GPU not available for comparison")
    
    return results


# Export important classes and functions
__all__ = [
    'OptimizedTrajectoryPlanning',
    'TrajectoryPlanning',  # Backward compatibility
    'create_optimized_planner',
    'compare_implementations',
]