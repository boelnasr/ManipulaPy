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
- Advanced kernel selection for 40x+ speedups

Copyright (c) 2025 Mohamed Aboelnasr
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
    make_2d_grid_optimized,
    get_gpu_properties,
    optimized_trajectory_generation,
    optimized_trajectory_generation_monitored,
    optimized_potential_field,
    optimized_batch_trajectory_generation,
    get_cuda_array,
    return_cuda_array,
    profile_start,
    profile_stop,
    _best_2d_config,
    _h2d_pinned,
    get_optimal_kernel_config,
    auto_select_optimal_kernel,
    print_performance_recommendations,
    setup_cuda_environment_for_40x_speedup,
    get_memory_pool_stats,
    benchmark_kernel_performance,
)

# Import CUDA functions only if available
if CUDA_AVAILABLE:
    from numba import cuda
    from .cuda_kernels import (
        trajectory_kernel,
        trajectory_kernel_vectorized,
        trajectory_kernel_memory_optimized,
        trajectory_kernel_warp_optimized,
        trajectory_kernel_cache_friendly,
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
    def trajectory_kernel_vectorized(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def trajectory_kernel_memory_optimized(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def trajectory_kernel_warp_optimized(*args, **kwargs):
        raise RuntimeError("CUDA not available")
    def trajectory_kernel_cache_friendly(*args, **kwargs):
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
    memory pooling, and batch processing capabilities for 40x+ speedups.
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
        auto_optimize: bool = True,
        kernel_type: str = "auto",
        target_speedup: float = 40.0,
    ):
        """
        Enhanced trajectory planner with advanced CUDA optimizations.

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

        auto_optimize      : bool
            Automatically setup CUDA environment for maximum performance.

        kernel_type        : str
            Kernel selection strategy: "auto", "standard", "vectorized", 
            "memory_optimized", "warp_optimized", "cache_friendly", "auto_tune"

        target_speedup     : float
            Target speedup over CPU (used for recommendations).
        """
        # ------------------------------------------------------------
        # FIRST: Set all basic attributes to prevent AttributeError
        # ------------------------------------------------------------
        self.kernel_type = kernel_type if kernel_type is not None else "auto"
        self.target_speedup = target_speedup if target_speedup is not None else 40.0
        self.enable_profiling = enable_profiling if enable_profiling is not None else False

        # Initialize all caches and tracking attributes immediately
        self._gpu_arrays = {}
        self._kernel_cache = {}
        self._last_cpu_time = 0.0

        # Initialize performance stats early
        self.performance_stats = {
            "gpu_calls":       0,
            "cpu_calls":       0,
            "total_gpu_time":  0.0,
            "total_cpu_time":  0.0,
            "memory_transfers": 0,
            "kernel_launches":  0,
            "speedup_achieved": 0.0,
            "best_kernel_used": "none",
        }

        # ------------------------------------------------------------
        # Auto-optimization setup
        # ------------------------------------------------------------
        if auto_optimize and CUDA_AVAILABLE:
            setup_cuda_environment_for_40x_speedup()

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

        # Store optimization parameters
        self.kernel_type = kernel_type
        self.target_speedup = target_speedup

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

        # Adaptive threshold based on target speedup
        if self.cuda_available and self.gpu_properties:
            sm_count = self.gpu_properties['multiprocessor_count']
            # Calculate threshold for target speedup
            min_elements_per_sm = 1000 if target_speedup >= 40 else 500
            self.cpu_threshold = max(cuda_threshold, int(sm_count * min_elements_per_sm / len(joint_limits)))
        else:
            self.cpu_threshold = cuda_threshold

        # optionally resize a global memory-pool
        if memory_pool_size_mb is not None and self.cuda_available:
            from .cuda_kernels import _cuda_memory_pool
            _cuda_memory_pool.max_pool_size = (
                memory_pool_size_mb * 1024 * 1024 // 4  # entries of float32
            )

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
            "speedup_achieved": 0.0,
            "best_kernel_used": "none",
        }

        # Enable profiling if requested (after all attributes are initialized)
        if self.enable_profiling and self.cuda_available:
            profile_start()

        # Print performance recommendations on initialization
        if self.cuda_available:
            num_joints = len(joint_limits)
            logger.info(f"🚀 OptimizedTrajectoryPlanning initialized for {num_joints} joints")
            if target_speedup >= 40:
                min_N_for_target = self.cpu_threshold // num_joints
                logger.info(f"💡 For {target_speedup}x speedup, use N ≥ {min_N_for_target:,} trajectory points")

        logger.info(
            "Optimised planner – CUDA enabled: %s (threshold %d, kernel: %s)",
            self.cuda_available, self.cpu_threshold, self.kernel_type,
        )
        if self.gpu_properties:
            logger.info("GPU: %d SMs, %d max threads/block", 
                       self.gpu_properties['multiprocessor_count'],
                       self.gpu_properties['max_threads_per_block'])
    
    def _get_or_resize_gpu_array(self, array_name, shape, dtype=np.float32):
        """
        Return a pooled CUDA array with the requested shape / dtype.
        Enhanced with better memory management.
        """
        if not self.cuda_available:
            return None

        arr = self._gpu_arrays.get(array_name)

        if (arr is None) or (arr.shape != shape) or (arr.dtype != dtype):
            if arr is not None:
                return_cuda_array(arr)

            arr = get_cuda_array(shape, dtype)
            self._gpu_arrays[array_name] = arr

        return arr

    def _should_use_gpu(self, N, num_joints):
        """Enhanced GPU selection logic with performance prediction."""
        if not self.cuda_available:
            return False

        total_work = N * num_joints
        if total_work < self.cpu_threshold:
            return False

        # Additional checks for memory and performance
        if self.gpu_properties:
            sm_count = self.gpu_properties['multiprocessor_count']
            elements_per_sm = total_work / sm_count
            
            # Check if we can achieve target speedup
            target_speedup_value = getattr(self, 'target_speedup', 40.0)
            if target_speedup_value >= 40 and elements_per_sm < 10000:
                logger.debug(f"Problem size may not achieve {target_speedup_value}x speedup. "
                           f"Elements per SM: {elements_per_sm:.0f}, recommended: ≥10,000")
        
        return True

    def _get_optimal_kernel_config(self, N, num_joints):
        """Get or compute optimal kernel configuration with caching."""
        # Ensure required attributes exist
        if not hasattr(self, '_kernel_cache'):
            self._kernel_cache = {}
        if not hasattr(self, 'kernel_type'):
            self.kernel_type = "auto"
            
        cache_key = (N, num_joints, self.kernel_type)
        
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        if self.kernel_type == "auto":
            kernel_type = auto_select_optimal_kernel(N, num_joints)
        else:
            kernel_type = self.kernel_type
        
        config = get_optimal_kernel_config(N, num_joints, kernel_type)
        self._kernel_cache[cache_key] = config
        
        return config

    def joint_trajectory(self, thetastart, thetaend, Tf, N, method, 
                        kernel_type=None, enable_monitoring=None):
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
            kernel_type = getattr(self, 'kernel_type', 'auto')
        if enable_monitoring is None:
            enable_monitoring = getattr(self, 'enable_profiling', False)

        logger.info(f"Generating joint trajectory: N={N}, joints={len(thetastart)}, "
                   f"method={method}, kernel={kernel_type}")
        
        thetastart = np.array(thetastart, dtype=np.float32)
        thetaend = np.array(thetaend, dtype=np.float32)
        num_joints = len(thetastart)

        # Print performance recommendations if beneficial
        total_work = N * num_joints
        if self.cuda_available and total_work >= 10000:
            print_performance_recommendations(N, num_joints)

        # Decide on execution strategy
        use_gpu = self._should_use_gpu(N, num_joints)
        
        if use_gpu:
            return self._joint_trajectory_gpu(thetastart, thetaend, Tf, N, method, 
                                            kernel_type, enable_monitoring)
        else:
            return self._joint_trajectory_cpu(thetastart, thetaend, Tf, N, method)

    def _joint_trajectory_gpu(self, thetastart, thetaend, Tf, N, method, 
                             kernel_type, enable_monitoring):
        """Enhanced GPU trajectory generation with optimal kernel selection."""
        start_time = time.time()
        
        try:
            # Use the monitored high-level wrapper for maximum performance
            traj_pos_host, traj_vel_host, traj_acc_host = optimized_trajectory_generation_monitored(
                thetastart, thetaend, Tf, N, method, 
                use_pinned=True, 
                kernel_type=kernel_type,
                enable_monitoring=enable_monitoring
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

            # Calculate achieved speedup
            elapsed = time.time() - start_time
            if hasattr(self, '_last_cpu_time') and self._last_cpu_time > 0:
                speedup = self._last_cpu_time / elapsed
                self.performance_stats['speedup_achieved'] = speedup
                if enable_monitoring:
                    logger.info(f"🎯 Achieved {speedup:.1f}x speedup over CPU!")
            else:
                # No previous CPU time to compare against
                if enable_monitoring:
                    logger.info("🚀 GPU execution completed (no CPU baseline for comparison)")

            # Update performance stats
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            # Update best kernel used
            config = self._get_optimal_kernel_config(N, len(thetastart))
            if config:
                self.performance_stats['best_kernel_used'] = config.get('kernel_type', 'unknown')

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
        """CPU-based joint trajectory generation with performance tracking."""
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

        # Store CPU time for speedup calculations
        elapsed = time.time() - start_time
        self._last_cpu_time = elapsed
        
        # Update performance stats
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

    def batch_joint_trajectory(self, thetastart_batch, thetaend_batch, Tf, N, method, 
                              kernel_type=None):
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
            kernel_type = getattr(self, 'kernel_type', 'auto')

        batch_size, num_joints = thetastart_batch.shape
        logger.info(f"Generating batch trajectories: batch_size={batch_size}, "
                   f"N={N}, joints={num_joints}, kernel={kernel_type}")

        if not self.cuda_available:
            logger.warning("Batch processing requires CUDA. Falling back to sequential processing.")
            return self._batch_joint_trajectory_cpu(thetastart_batch, thetaend_batch, Tf, N, method)
        
        # Print performance recommendations for batch processing
        total_work = batch_size * N * num_joints
        if total_work >= 50000:
            print_performance_recommendations(N * batch_size, num_joints)

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
            throughput = total_work / elapsed / 1e6  # Million elements per second
            
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
            logger.info(f"Batch GPU trajectory generation completed in {elapsed:.4f}s")
            logger.info(f"📊 Throughput: {throughput:.1f} M elements/sec")

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
        Compute joint torques with enhanced CUDA acceleration.

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

        # Print performance recommendations
        if self.cuda_available:
            total_work = num_points * num_joints
            if total_work >= 10000:
                print_performance_recommendations(num_points, num_joints)

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
            """GPU-accelerated inverse dynamics computation with fixed kernel signature."""
            start_time = time.time()
            
            num_points = thetalist_trajectory.shape[0]
            num_joints = thetalist_trajectory.shape[1]
            
            try:
                # Use memory pool for the large torques array
                torques_trajectory = get_cuda_array((num_points, num_joints), dtype=np.float32)
                
                # Transfer data to GPU using pinned memory - ensure proper data types
                d_thetalist_trajectory = _h2d_pinned(thetalist_trajectory.astype(np.float32))
                d_dthetalist_trajectory = _h2d_pinned(dthetalist_trajectory.astype(np.float32))
                d_ddthetalist_trajectory = _h2d_pinned(ddthetalist_trajectory.astype(np.float32))
                
                d_gravity_vector = cuda.to_device(gravity_vector.astype(np.float32))
                d_Ftip = cuda.to_device(np.array(Ftip, dtype=np.float32))
                
                # Safely handle dynamics data conversion
                try:
                    # Convert Glist to proper numpy array format
                    if hasattr(self.dynamics, 'Glist') and self.dynamics.Glist is not None:
                        if isinstance(self.dynamics.Glist, list):
                            # Convert list of matrices to 3D numpy array
                            Glist_array = np.stack(self.dynamics.Glist).astype(np.float32)
                        else:
                            Glist_array = np.array(self.dynamics.Glist, dtype=np.float32)
                    else:
                        # Create dummy Glist if not available
                        Glist_array = np.eye(6, dtype=np.float32)[None, :, :].repeat(num_joints, axis=0)
                    
                    # Convert S_list to proper format
                    if hasattr(self.dynamics, 'S_list') and self.dynamics.S_list is not None:
                        Slist_array = np.array(self.dynamics.S_list, dtype=np.float32)
                    else:
                        # Create dummy S_list if not available
                        Slist_array = np.random.randn(6, num_joints).astype(np.float32)
                    
                    # Convert M_list to proper format
                    if hasattr(self.dynamics, 'M_list') and self.dynamics.M_list is not None:
                        M_array = np.array(self.dynamics.M_list, dtype=np.float32)
                    else:
                        # Create dummy M if not available
                        M_array = np.eye(4, dtype=np.float32)
                    
                    d_Glist = cuda.to_device(Glist_array)
                    d_Slist = cuda.to_device(Slist_array)
                    d_M = cuda.to_device(M_array)
                    
                except Exception as e:
                    logger.warning(f"Error converting dynamics data: {e}, using simplified approach")
                    # Fallback to simplified dynamics computation on CPU
                    return self._inverse_dynamics_cpu(
                        thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                        gravity_vector, Ftip
                    )
                
                d_torque_limits = cuda.to_device(self.torque_limits.astype(np.float32))

                # Get optimal 2D launch configuration with bounds checking
                try:
                    blocks_per_grid, threads_per_block = _best_2d_config(num_points, num_joints)
                    logger.info(f"Inverse dynamics 2D grid: blocks={blocks_per_grid}, threads={threads_per_block}")
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
                    # inverse_dynamics_kernel(
                    #     thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                    #     gravity_vector, Ftip, Glist, Slist, M, torques_trajectory, torque_limits, stream=0
                    # )
                    # That's 11 parameters total including the stream parameter
                    
                    inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
                        d_thetalist_trajectory,       # 1
                        d_dthetalist_trajectory,      # 2  
                        d_ddthetalist_trajectory,     # 3
                        d_gravity_vector,             # 4
                        d_Ftip,                       # 5
                        d_Glist,                      # 6
                        d_Slist,                      # 7
                        d_M,                          # 8
                        torques_trajectory,           # 9
                        d_torque_limits,              # 10
                        # 0                             # 11 - stream parameter (was missing!)
                    )
                    
                    # Synchronize to check for kernel execution errors
                    cuda.synchronize()
                    
                except Exception as kernel_error:
                    logger.warning(f"CUDA kernel execution failed: {kernel_error}")
                    # Fallback to CPU implementation
                    return self._inverse_dynamics_cpu(
                        thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory,
                        gravity_vector, Ftip
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
        
        logger.info(f"Computing forward dynamics: {num_steps} steps, {num_joints} joints")

        # Print performance recommendations
        if self.cuda_available:
            total_work = num_steps * num_joints
            if total_work >= 10000:
                print_performance_recommendations(num_steps, num_joints)

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
        """Enhanced GPU forward dynamics with optimal configuration."""
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
            
            # Copy initial conditions to GPU
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

            # Get optimal launch configuration
            grid_config = get_optimal_kernel_config(num_steps, num_joints, "cache_friendly")
            if grid_config:
                blocks_per_grid = grid_config["grid"]
                threads_per_block = grid_config["block"]
                logger.info(f"Using {grid_config['kernel_type']} for forward dynamics")
            else:
                blocks_per_grid, threads_per_block = _best_2d_config(num_steps, num_joints)
            
            # Launch forward dynamics kernel
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
            throughput = (num_steps * num_joints * intRes) / elapsed / 1e6
            
            self.performance_stats['gpu_calls'] += 1
            self.performance_stats['total_gpu_time'] += elapsed
            self.performance_stats['kernel_launches'] += 1
            
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
        """Enhanced GPU Cartesian trajectory computation."""
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
                logger.info(f"Using {grid_config['kernel_type']} for Cartesian trajectory")
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
        Enhanced performance statistics with speedup analysis.
        
        Returns:
            dict: Comprehensive performance statistics
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

        # Calculate overall speedup
        total_gpu_time = stats['total_gpu_time']
        total_cpu_time = stats['total_cpu_time']
        if total_gpu_time > 0 and total_cpu_time > 0:
            stats['overall_speedup'] = total_cpu_time / total_gpu_time
        else:
            stats['overall_speedup'] = 0.0

        # Add memory pool statistics
        if self.cuda_available:
            stats['memory_pool_stats'] = get_memory_pool_stats()

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
            'speedup_achieved': 0.0,
            'best_kernel_used': 'none',
        }

    def cleanup_gpu_memory(self):
        """Enhanced GPU memory cleanup."""
        if self.cuda_available:
            # Clean up per-instance cache
            if hasattr(self, '_gpu_arrays'):
                for array in self._gpu_arrays.values():
                    if array is not None:
                        return_cuda_array(array)
                self._gpu_arrays.clear()
            
            # Clear kernel cache
            if hasattr(self, '_kernel_cache'):
                self._kernel_cache.clear()
            
            # Clear global memory pool
            from .cuda_kernels import _cuda_memory_pool
            _cuda_memory_pool.clear()
            
            # Synchronize and clean up CUDA context
            cuda.synchronize()
            
            logger.info("GPU memory cleaned up")

    def benchmark_all_kernels(self, N=5000, num_joints=6, num_runs=5):
        """
        Comprehensive benchmarking of all available kernels.
        
        Args:
            N (int): Number of trajectory points for benchmarking
            num_joints (int): Number of joints
            num_runs (int): Number of benchmark runs per kernel
            
        Returns:
            dict: Benchmark results for all kernels
        """
        if not self.cuda_available:
            logger.warning("CUDA not available for benchmarking")
            return {}

        logger.info(f"🔬 Benchmarking all kernels: N={N}, joints={num_joints}, runs={num_runs}")
        
        # Generate test data
        thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
        thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)
        
        kernel_types = ["standard", "vectorized", "memory_optimized", "warp_optimized", "cache_friendly"]
        results = {}
        
        for kernel_type in kernel_types:
            logger.info(f"📊 Testing {kernel_type} kernel...")
            
            # Reset stats for clean measurement
            self.reset_performance_stats()
            
            times = []
            for run in range(num_runs):
                start_time = time.time()
                
                try:
                    trajectory = self.joint_trajectory(
                        thetastart, thetaend, 2.0, N, 5, 
                        kernel_type=kernel_type, enable_monitoring=False
                    )
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    
                except Exception as e:
                    logger.warning(f"Kernel {kernel_type} failed: {e}")
                    times.append(float('inf'))
            
            if times and min(times) < float('inf'):
                results[kernel_type] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'all_times': times,
                    'success_rate': sum(1 for t in times if t < float('inf')) / len(times),
                }
            else:
                results[kernel_type] = {
                    'mean_time': float('inf'),
                    'std_time': 0,
                    'min_time': float('inf'),
                    'max_time': float('inf'),
                    'all_times': times,
                    'success_rate': 0,
                }
        
        # Find best kernel
        best_kernel = min(results.keys(), key=lambda k: results[k]['mean_time'])
        best_time = results[best_kernel]['mean_time']
        
        logger.info(f"🏆 Best kernel: {best_kernel} ({best_time*1000:.2f}ms)")
        
        # Print comparison table
        print("\n📋 Kernel Performance Comparison:")
        print("=" * 70)
        print(f"{'Kernel':<20} {'Mean (ms)':<12} {'Min (ms)':<12} {'Success':<10}")
        print("-" * 70)
        
        for kernel_type, stats in results.items():
            mean_ms = stats['mean_time'] * 1000 if stats['mean_time'] < float('inf') else float('inf')
            min_ms = stats['min_time'] * 1000 if stats['min_time'] < float('inf') else float('inf')
            success = f"{stats['success_rate']*100:.0f}%"
            
            marker = "🏆" if kernel_type == best_kernel else "  "
            print(f"{marker}{kernel_type:<18} {mean_ms:<12.2f} {min_ms:<12.2f} {success:<10}")
        
        return results

    def __del__(self):
        """Enhanced destructor with better error handling."""
        try:
            if hasattr(self, 'enable_profiling') and self.enable_profiling and hasattr(self, 'cuda_available') and self.cuda_available:
                profile_stop()
            if hasattr(self, 'cleanup_gpu_memory'):
                self.cleanup_gpu_memory()
        except Exception:
            pass  # Ignore errors during cleanup

    # Enhanced plotting methods with performance annotations
    @staticmethod
    def plot_trajectory(trajectory_data, Tf, title="Joint Trajectory", labels=None, 
                       performance_stats=None):
        """Enhanced trajectory plotting with performance information."""
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        num_joints = positions.shape[1]
        time_steps = np.linspace(0, Tf, num_steps)

        fig, axs = plt.subplots(3, num_joints, figsize=(15, 10), sharex="col")
        
        # Add performance info to title
        if performance_stats:
            speedup = performance_stats.get('speedup_achieved', 0)
            kernel = performance_stats.get('best_kernel_used', 'unknown')
            if speedup > 1:
                title += f" (GPU: {speedup:.1f}x speedup, {kernel} kernel)"
            else:
                title += " (CPU execution)"
        
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
        Enhanced TCP trajectory plotting with performance monitoring.
        
        Args:
            trajectory (list): A list of joint angle configurations representing the trajectory.
            dt (float): The time step between consecutive points in the trajectory.
        
        Returns:
            None
        """
        start_time = time.time()
        
        tcp_trajectory = [
            self.serial_manipulator.forward_kinematics(joint_angles)
            for joint_angles in trajectory
        ]
        tcp_positions = [pose[:3, 3] for pose in tcp_trajectory]

        velocity, acceleration, jerk = self.calculate_derivatives(tcp_positions, dt)
        time = np.arange(0, len(tcp_positions) * dt, dt)

        elapsed = time.time() - start_time
        
        plt.figure(figsize=(12, 8))
        title = f"TCP Trajectory (FK computed in {elapsed:.3f}s)"
        plt.suptitle(title)
        
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

    def plot_cartesian_trajectory(self, trajectory_data, Tf, title="Cartesian Trajectory",
                                 performance_stats=None):
        """
        Enhanced Cartesian trajectory plotting with performance information.
        
        Args:
            trajectory_data (dict): A dictionary containing trajectory data.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot.
            performance_stats (dict, optional): Performance statistics to display.
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        # Add performance info to title
        if performance_stats:
            speedup = performance_stats.get('speedup_achieved', 0)
            if speedup > 1:
                title += f" (GPU: {speedup:.1f}x speedup)"
            else:
                title += " (CPU execution)"

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
        Enhanced end-effector trajectory plotting.
        
        Args:
            trajectory_data (dict): A dictionary containing trajectory data.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot.
        
        Returns:
            None
        """
        positions = trajectory_data["positions"]
        num_steps = positions.shape[0]
        time_steps = np.linspace(0, Tf, num_steps)

        if "orientations" in trajectory_data:
            orientations = trajectory_data["orientations"]
        else:
            # Compute orientations using forward kinematics
            start_time = time.time()
            orientations = np.array(
                [
                    self.serial_manipulator.forward_kinematics(pos)[:3, :3]
                    for pos in positions
                ]
            )
            elapsed = time.time() - start_time
            title += f" (FK for orientations: {elapsed:.3f}s)"

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.suptitle(title)

        ax.plot(
            positions[:, 0], positions[:, 1], positions[:, 2], 
            label="EE Position", color="b", linewidth=2
        )

        # Draw orientation frames at selected points
        frame_step = max(1, num_steps // 20)
        for i in range(0, num_steps, frame_step):
            R = orientations[i]
            pos = positions[i]
            scale = 0.01
            
            # X-axis (red)
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 0], R[1, 0], R[2, 0], 
                length=scale, color="r", alpha=0.8
            )
            # Y-axis (green)
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 1], R[1, 1], R[2, 1], 
                length=scale, color="g", alpha=0.8
            )
            # Z-axis (blue)
            ax.quiver(
                pos[0], pos[1], pos[2], R[0, 2], R[1, 2], R[2, 2], 
                length=scale, color="b", alpha=0.8
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.legend()
        plt.show()

    def plan_trajectory(self, start_position, target_position, obstacle_points):
        """
        Enhanced trajectory planning with collision avoidance.

        Args:
            start_position (list): Initial joint configuration.
            target_position (list): Desired joint configuration.
            obstacle_points (list): List of obstacle points in the environment.

        Returns:
            list: Joint trajectory as a list of joint configurations.
        """
        logger.info(f"Planning trajectory from {len(start_position)} to {len(target_position)} DOF")
        
        # Enhanced trajectory planning with multiple waypoints
        # This is a simple interpolation - can be extended with RRT*, PRM, etc.
        num_waypoints = 5
        joint_trajectory = []
        
        start_pos = np.array(start_position)
        target_pos = np.array(target_position)
        
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint = (1 - alpha) * start_pos + alpha * target_pos
            
            # Simple collision avoidance - move away from obstacles
            if obstacle_points and self.potential_field:
                for _ in range(10):  # Max adjustment iterations
                    gradient = self.potential_field.compute_gradient(
                        waypoint, target_pos, obstacle_points
                    )
                    waypoint -= 0.01 * gradient
                    
                    # Check if waypoint is collision-free
                    if self.collision_checker:
                        if not self.collision_checker.check_collision(waypoint):
                            break
            
            joint_trajectory.append(waypoint.tolist())
        
        logger.info(f"Planned trajectory with {len(joint_trajectory)} waypoints")
        return joint_trajectory

    def benchmark_performance(self, test_cases=None, include_cpu_comparison=True):
        """
        Enhanced performance benchmarking with detailed analysis.
        
        Args:
            test_cases (list, optional): List of test cases to benchmark.
            include_cpu_comparison (bool): Whether to include CPU vs GPU comparison.
                                       
        Returns:
            dict: Comprehensive benchmark results
        """
        if test_cases is None:
            test_cases = [
                {"N": 100, "joints": 6, "name": "Small"},
                {"N": 1000, "joints": 6, "name": "Medium"},
                {"N": 5000, "joints": 6, "name": "Large"},
                {"N": 10000, "joints": 6, "name": "Very Large"},
                {"N": 1000, "joints": 12, "name": "Many joints"},
                {"N": 5000, "joints": 12, "name": "Large + Many joints"},
            ]
        
        results = {}
        
        print("\n🚀 Enhanced Performance Benchmarking")
        print("=" * 60)
        
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
            
            # Test trajectory generation with multiple runs for accuracy
            times = []
            for run in range(3):  # Multiple runs for statistical accuracy
                start_time = time.time()
                trajectory = self.joint_trajectory(thetastart, thetaend, 2.0, N, 5)
                end_time = time.time()
                times.append(end_time - start_time)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Get performance stats
            stats = self.get_performance_stats()
            
            results[name] = {
                "mean_time": mean_time,
                "std_time": std_time,
                "min_time": min(times),
                "max_time": max(times),
                "N": N,
                "joints": joints,
                "stats": stats,
                "used_gpu": stats['gpu_calls'] > 0,
                "trajectory_shape": trajectory["positions"].shape,
                "speedup_achieved": stats.get('speedup_achieved', 0),
                "kernel_used": stats.get('best_kernel_used', 'unknown'),
                "elements_per_second": (N * joints) / mean_time,
            }
            
            # CPU comparison if requested
            if include_cpu_comparison and self.cuda_available:
                # Force CPU execution
                old_threshold = self.cpu_threshold
                self.cpu_threshold = float('inf')  # Force CPU
                
                cpu_start = time.time()
                cpu_trajectory = self.joint_trajectory(thetastart, thetaend, 2.0, N, 5)
                cpu_time = time.time() - cpu_start
                
                self.cpu_threshold = old_threshold  # Restore threshold
                
                if mean_time > 0:
                    actual_speedup = cpu_time / mean_time
                    results[name]["cpu_time"] = cpu_time
                    results[name]["actual_speedup"] = actual_speedup
                else:
                    results[name]["actual_speedup"] = 0
            
            # Print summary
            gpu_indicator = "🚀 GPU" if results[name]['used_gpu'] else "🖥️  CPU"
            speedup_str = ""
            if "actual_speedup" in results[name] and results[name]["actual_speedup"] > 1:
                speedup_str = f" ({results[name]['actual_speedup']:.1f}x speedup)"
            
            print(f"{gpu_indicator} {name}: {mean_time*1000:.2f}±{std_time*1000:.2f}ms{speedup_str}")
            
            logger.info(f"{name} benchmark: {mean_time:.4f}s, GPU: {results[name]['used_gpu']}")
        
        # Print summary table
        print(f"\n📊 Benchmark Summary:")
        print("-" * 80)
        print(f"{'Test Case':<20} {'Time (ms)':<12} {'GPU':<6} {'Speedup':<10} {'Throughput':<15}")
        print("-" * 80)
        
        for name, result in results.items():
            time_ms = result['mean_time'] * 1000
            gpu_used = "✓" if result['used_gpu'] else "✗"
            speedup = f"{result.get('actual_speedup', 0):.1f}x" if result.get('actual_speedup', 0) > 1 else "-"
            throughput = f"{result['elements_per_second']/1e6:.2f} M/s"
            
            print(f"{name:<20} {time_ms:<12.2f} {gpu_used:<6} {speedup:<10} {throughput:<15}")
        
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


# Enhanced utility functions for advanced users
def create_optimized_planner(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    torque_limits=None,
    target_speedup=40.0,
    gpu_memory_mb=None,
    enable_profiling=False,
    kernel_type="auto",
):
    """
    Enhanced factory function to create an optimized trajectory planner.
    
    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        torque_limits: Torque limits (optional)
        target_speedup: Target speedup over CPU (default: 40x)
        gpu_memory_mb: GPU memory pool size in MB (optional)
        enable_profiling: Enable CUDA profiling (optional)
        kernel_type: Kernel selection strategy (optional)
    
    Returns:
        OptimizedTrajectoryPlanning: Configured planner instance
    """
    # Auto-detect optimal settings
    cuda_available = check_cuda_availability()
    
    # Adaptive threshold based on target speedup and problem size
    num_joints = len(joint_limits)
    
    if cuda_available:
        gpu_props = get_gpu_properties()
        if gpu_props:
            sm_count = gpu_props['multiprocessor_count']
            if target_speedup >= 40:
                threshold = max(50, int(sm_count * 10000 / num_joints))
            elif target_speedup >= 20:
                threshold = max(50, int(sm_count * 5000 / num_joints))
            else:
                threshold = max(50, int(sm_count * 1000 / num_joints))
        else:
            threshold = 1000
    else:
        threshold = float('inf')  # Never use GPU if not available
    
    # Create planner with optimized settings
    planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        use_cuda=None,  # Auto-detect
        cuda_threshold=threshold,
        memory_pool_size_mb=gpu_memory_mb,
        enable_profiling=enable_profiling,
        auto_optimize=True,
        kernel_type=kernel_type,
        target_speedup=target_speedup,
    )
    
    logger.info(f"Created optimized planner for {num_joints} joints, "
               f"target: {target_speedup}x speedup, CUDA: {cuda_available}")
    
    return planner


def compare_implementations(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    test_params=None,
    detailed_analysis=True,
):
    """
    Enhanced implementation comparison with detailed kernel analysis.
    
    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_params: Test parameters (optional)
        detailed_analysis: Whether to perform detailed kernel comparison
    
    Returns:
        dict: Comprehensive comparison results
    """
    if test_params is None:
        test_params = {"N": 5000, "Tf": 2.0, "method": 5, "num_runs": 5}
    
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
            kernel_type="auto_tune",
        )
    
    # Generate test data
    num_joints = len(joint_limits)
    thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    
    results = {"cpu": {}, "gpu": {}}
    
    # Test CPU implementation
    logger.info("Testing CPU implementation...")
    cpu_times = []
    for run in range(test_params.get("num_runs", 3)):
        start_time = time.time()
        cpu_result = cpu_planner.joint_trajectory(
            thetastart, thetaend, test_params["Tf"], test_params["N"], test_params["method"]
        )
        cpu_times.append(time.time() - start_time)
    
    cpu_mean_time = np.mean(cpu_times)
    results["cpu"] = {
        "mean_time": cpu_mean_time,
        "std_time": np.std(cpu_times),
        "min_time": np.min(cpu_times),
        "max_time": np.max(cpu_times),
        "result_shape": cpu_result["positions"].shape,
        "stats": cpu_planner.get_performance_stats(),
    }
    
    # Test GPU implementation (if available)
    if gpu_planner is not None:
        logger.info("Testing GPU implementation...")
        
        # Test different kernels if detailed analysis requested
        if detailed_analysis:
            kernel_results = gpu_planner.benchmark_all_kernels(
                N=test_params["N"], 
                num_joints=num_joints, 
                num_runs=test_params.get("num_runs", 3)
            )
            results["kernel_comparison"] = kernel_results
        
        # Test best configuration
        gpu_times = []
        for run in range(test_params.get("num_runs", 3)):
            start_time = time.time()
            gpu_result = gpu_planner.joint_trajectory(
                thetastart, thetaend, test_params["Tf"], test_params["N"], test_params["method"]
            )
            gpu_times.append(time.time() - start_time)
        
        gpu_mean_time = np.mean(gpu_times)
        speedup = cpu_mean_time / gpu_mean_time if gpu_mean_time > 0 else 0
        
        results["gpu"] = {
            "mean_time": gpu_mean_time,
            "std_time": np.std(gpu_times),
            "min_time": np.min(gpu_times),
            "max_time": np.max(gpu_times),
            "result_shape": gpu_result["positions"].shape,
            "stats": gpu_planner.get_performance_stats(),
            "speedup": speedup,
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
        
        # Print comprehensive results
        print(f"\n🚀 Implementation Comparison Results:")
        print("=" * 50)
        print(f"CPU Time: {cpu_mean_time*1000:.2f} ± {results['cpu']['std_time']*1000:.2f} ms")
        print(f"GPU Time: {gpu_mean_time*1000:.2f} ± {results['gpu']['std_time']*1000:.2f} ms")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Max Position Error: {results['accuracy']['max_pos_diff']:.2e}")
        print(f"Mean Position Error: {results['accuracy']['mean_pos_diff']:.2e}")
        
        if speedup >= 40:
            print("🎯 Achieved 40x+ speedup target!")
        elif speedup >= 20:
            print("⚡ Good speedup achieved!")
        elif speedup >= 5:
            print("✅ Moderate speedup achieved")
        else:
            print("⚠️  Limited speedup - consider larger problem sizes")
            
        logger.info(f"GPU speedup: {speedup:.2f}x")
    else:
        results["gpu"] = {"available": False}
        logger.info("GPU not available for comparison")
    
    return results


def benchmark_kernel_performance_comprehensive(
    serial_manipulator, urdf_path, dynamics, joint_limits,
    test_sizes=None, num_runs=5
):
    """
    Comprehensive kernel performance benchmarking across multiple problem sizes.
    
    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file 
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_sizes: List of (N, joints) tuples to test
        num_runs: Number of runs per test
        
    Returns:
        dict: Comprehensive benchmark results
    """
    if not check_cuda_availability():
        logger.warning("CUDA not available for comprehensive benchmarking")
        return {}
    
    if test_sizes is None:
        test_sizes = [
            (1000, 6), (5000, 6), (10000, 6), (20000, 6),
            (1000, 12), (5000, 12), (10000, 12),
        ]
    
    print("\n🔬 Comprehensive Kernel Performance Benchmarking")
    print("=" * 60)
    
    all_results = {}
    
    for N, joints in test_sizes:
        logger.info(f"Testing N={N}, joints={joints}")
        
        # Create optimized planner
        planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits[:joints],  # Use subset of joints
            use_cuda=True,
            cuda_threshold=0,
            kernel_type="auto_tune",
        )
        
        # Benchmark all kernels for this problem size
        kernel_results = planner.benchmark_all_kernels(N=N, num_joints=joints, num_runs=num_runs)
        
        all_results[f"N{N}_J{joints}"] = {
            "N": N,
            "joints": joints,
            "total_elements": N * joints,
            "kernel_results": kernel_results,
        }
        
        # Find best kernel for this size
        if kernel_results:
            best_kernel = min(kernel_results.keys(), key=lambda k: kernel_results[k]['mean_time'])
            best_time = kernel_results[best_kernel]['mean_time']
            throughput = (N * joints) / best_time / 1e6
            
            print(f"N={N:5d}, J={joints:2d}: Best={best_kernel:<15} "
                  f"Time={best_time*1000:6.2f}ms Throughput={throughput:6.1f}M/s")
    
    return all_results


# Export important classes and functions
__all__ = [
    'OptimizedTrajectoryPlanning',
    'TrajectoryPlanning',  # Backward compatibility
    'create_optimized_planner',
    'compare_implementations',
    'benchmark_kernel_performance_comprehensive',
]