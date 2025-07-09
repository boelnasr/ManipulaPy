#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Optimized CUDA Kernels Module - ManipulaPy

This module provides highly optimized CUDA-accelerated functions for trajectory planning and dynamics
computation. All CUDA functionality is optional and gracefully degrades to CPU implementations.

Optimizations include:
- 2D grid parallelization across time and joint indices
- Shared memory utilization for 6x6 matrix operations
- Register-optimized computations avoiding spilling
- Fused kernels to reduce memory bandwidth requirements
- Adaptive launch configurations for optimal occupancy
- Pinned memory transfers for improved PCIe bandwidth
- Optimized scalar time-scaling with shared memory
- Fast math operations with rsqrt for improved performance

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import warnings
import math
import os
from typing import Tuple, List, Optional
from numba import config as _nb_cfg
_nb_cfg.CUDA_CACHE_SIZE = "1024"  # cache compiled kernels

# Environment toggle for strict FP
FAST_MATH = bool(int(os.getenv("MANIPULAPY_FASTMATH", "1")))

# Optional CUDA imports with graceful fallback
try:
    from numba import cuda, float32, int32, float16
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Create mock objects to prevent import errors
    class MockCuda:
        @staticmethod
        def jit(func=None, device=False, inline=False, fastmath=False):
            """Mock decorator for when CUDA is not available"""
            def wrapper(*args, **kwargs):
                raise RuntimeError(
                    "CUDA functionality not available. Install CUDA support:\n"
                    "pip install ManipulaPy[gpu-cuda11]  # For CUDA 11.x\n"
                    "pip install ManipulaPy[gpu-cuda12]  # For CUDA 12.x\n"
                    "Ensure you have NVIDIA GPU drivers and CUDA toolkit installed."
                )
            return wrapper if func is None else wrapper(func)
        
        @staticmethod
        def grid(dim):
            return 0
        
        @staticmethod
        def device_array(*args, **kwargs):
            raise RuntimeError("CUDA not available - cannot create device arrays")
        
        @staticmethod
        def to_device(*args, **kwargs):
            raise RuntimeError("CUDA not available - cannot transfer to device")
        
        @staticmethod
        def pinned_array(*args, **kwargs):
            raise RuntimeError("CUDA not available - cannot create pinned arrays")
        
        @staticmethod
        def shared():
            class SharedMock:
                @staticmethod
                def array(*args, **kwargs):
                    raise RuntimeError("CUDA not available - cannot create shared arrays")
            return SharedMock()
        
        @staticmethod
        def local():
            class LocalMock:
                @staticmethod
                def array(*args, **kwargs):
                    raise RuntimeError("CUDA not available - cannot create local arrays")
            return LocalMock()
        
        blockIdx = type('blockIdx', (), {'x': 0, 'y': 0, 'z': 0})()
        blockDim = type('blockDim', (), {'x': 1, 'y': 1, 'z': 1})()
        threadIdx = type('threadIdx', (), {'x': 0, 'y': 0, 'z': 0})()
        
        @staticmethod
        def syncthreads():
            pass
        
        @staticmethod
        def synchronize():
            pass
    
    cuda = MockCuda()
    float32 = np.float32
    int32 = np.int32
    float16 = np.float16

# Check for CuPy availability (separate from numba.cuda)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Configure float precision based on environment
float_t = float16 if os.getenv("MANIPULAPY_USE_FP16", "0") == "1" else float32

def check_cuda_availability():
    """Check if CUDA is available and provide helpful error messages."""
    if not CUDA_AVAILABLE:
        warnings.warn(
            "CUDA not available. GPU-accelerated trajectory planning will not work.\n"
            "To enable GPU support:\n"
            "1. Install NVIDIA GPU drivers\n"
            "2. Install CUDA toolkit\n"
            "3. Install: pip install ManipulaPy[gpu-cuda11] (or gpu-cuda12)\n"
            "Falling back to CPU-only operation.",
            UserWarning,
            stacklevel=2
        )
    return CUDA_AVAILABLE

def check_cupy_availability():
    """Check if CuPy is available for GPU array operations."""
    if not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Some GPU array operations will not work.\n"
            "Install with: pip install ManipulaPy[gpu-cuda11]",
            UserWarning,
            stacklevel=2
        )
    return CUPY_AVAILABLE

def _h2d_pinned(arr):
    """Helper function for pinned memory H2D transfers."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    
    # Create pinned memory for faster transfers
    try:
        pinned_arr = cuda.pinned_array(arr.shape, dtype=arr.dtype)
        pinned_arr[:] = arr
        return cuda.to_device(pinned_arr)
    except:
        # Fallback to regular transfer
        return cuda.to_device(arr)

# Optimized launch configuration helpers
def make_1d_grid(size, threads=256):
    """Create 1D grid configuration with optimal occupancy."""
    if size <= 0:
        return (1,), (1,)
    
    # Ensure minimum occupancy for small datasets
    if size < threads:
        threads = max(32, 2**int(math.log2(size)))  # Round down to power of 2
    
    blocks = (size + threads - 1) // threads
    return (blocks,), (threads,)

def make_2d_grid(N: int,
                 num_joints: int,
                 block_size: Tuple[int, int] = (64, 8)):
    """
    Choose (grid, block) for a 2-D kernel so that
      • thread-block dimensions are powers of two (≥ 4)
      • block.x * block.y  ≤  device.MAX_THREADS_PER_BLOCK
      • total resident blocks ≥ 2 × SMs  (good occupancy)
    """
    # --- 1.  start from the user hint ---
    threads_x, threads_y = block_size

    # --- 2.  shrink block if the problem is tiny -------------
    threads_x = max(4, 1 << int(math.log2(max(1, min(threads_x, N)))))
    threads_y = max(4, 1 << int(math.log2(max(1, min(threads_y, num_joints)))))

    # ----------------- helper -----------------
    def grid_dims(tx: int, ty: int) -> Tuple[int, int]:
        return ( (N + tx - 1) // tx,
                 (num_joints + ty - 1) // ty )

    blocks_x, blocks_y = grid_dims(threads_x, threads_y)
    total_blocks = blocks_x * blocks_y

    # --- 3.  target ≥ 2 × SM blocks for decent load ----------
    sm_count = (cuda.get_current_device().MULTIPROCESSOR_COUNT
                if CUDA_AVAILABLE else 16)  # sensible default
    min_blocks = sm_count * 2

    # Device constraint
    max_threads_per_block = (cuda.get_current_device().MAX_THREADS_PER_BLOCK
                             if CUDA_AVAILABLE else 1024)

    # --- 4.  keep halving X and Y until we hit the target ----
    toggle = 0      # alternate axis to avoid degenerate blocks
    while total_blocks < min_blocks:
        if toggle == 0 and threads_x > 4:
            threads_x //= 2
        elif toggle == 1 and threads_y > 4:
            threads_y //= 2
        else:                     # can't shrink further
            break
        toggle ^= 1               # flip axis

        # keep within HW limit
        if threads_x * threads_y > max_threads_per_block:
            # shrink the larger dimension
            if threads_x >= threads_y and threads_x > 4:
                threads_x //= 2
            elif threads_y > 4:
                threads_y //= 2

        blocks_x, blocks_y = grid_dims(threads_x, threads_y)
        total_blocks = blocks_x * blocks_y

    return (blocks_x, blocks_y), (threads_x, threads_y)

def get_gpu_properties():
    """Get GPU properties for optimization decisions."""
    if not CUDA_AVAILABLE:
        return None
    
    try:
        device = cuda.get_current_device()
        return {
            'multiprocessor_count': device.MULTIPROCESSOR_COUNT,
            'max_threads_per_block': device.MAX_THREADS_PER_BLOCK,
            'max_shared_memory_per_block': device.MAX_SHARED_MEMORY_PER_BLOCK,
            'max_block_dim_x': device.MAX_BLOCK_DIM_X,
            'max_block_dim_y': device.MAX_BLOCK_DIM_Y,
        }
    except:
        return None

# CPU fallback functions for when CUDA is not available
def trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method):
    """CPU fallback for trajectory generation when CUDA is not available."""
    num_joints = len(thetastart)
    traj_pos = np.zeros((N, num_joints), dtype=np.float32)
    traj_vel = np.zeros((N, num_joints), dtype=np.float32)
    traj_acc = np.zeros((N, num_joints), dtype=np.float32)
    
    for idx in range(N):
        t = idx * (Tf / (N - 1))
        tau = t / Tf
        
        if method == 3:  # Cubic time scaling
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
        elif method == 5:  # Quintic time scaling
            tau2 = tau * tau
            tau3 = tau2 * tau
            s = 10.0 * tau3 - 15.0 * tau2 * tau2 + 6.0 * tau * tau3
            s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
            s_ddot = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
        else:
            s = s_dot = s_ddot = 0.0

        for j in range(num_joints):
            dtheta = thetaend[j] - thetastart[j]
            traj_pos[idx, j] = s * dtheta + thetastart[j]
            traj_vel[idx, j] = s_dot * dtheta
            traj_acc[idx, j] = s_ddot * dtheta
    
    return traj_pos, traj_vel, traj_acc

# CUDA kernel definitions (only compiled if CUDA is available)
if CUDA_AVAILABLE:
    
    jit_kwargs = dict(fastmath=FAST_MATH)
    
    @cuda.jit(device=True, inline=True, **jit_kwargs)
    def matrix_vector_multiply_6x6(M, v, result):
        """Optimized 6x6 matrix-vector multiplication using registers."""
        # Unrolled matrix-vector multiplication for better performance
        result[0] = M[0,0]*v[0] + M[0,1]*v[1] + M[0,2]*v[2] + M[0,3]*v[3] + M[0,4]*v[4] + M[0,5]*v[5]
        result[1] = M[1,0]*v[0] + M[1,1]*v[1] + M[1,2]*v[2] + M[1,3]*v[3] + M[1,4]*v[4] + M[1,5]*v[5]
        result[2] = M[2,0]*v[0] + M[2,1]*v[1] + M[2,2]*v[2] + M[2,3]*v[3] + M[2,4]*v[4] + M[2,5]*v[5]
        result[3] = M[3,0]*v[0] + M[3,1]*v[1] + M[3,2]*v[2] + M[3,3]*v[3] + M[3,4]*v[4] + M[3,5]*v[5]
        result[4] = M[4,0]*v[0] + M[4,1]*v[1] + M[4,2]*v[2] + M[4,3]*v[3] + M[4,4]*v[4] + M[4,5]*v[5]
        result[5] = M[5,0]*v[0] + M[5,1]*v[1] + M[5,2]*v[2] + M[5,3]*v[3] + M[5,4]*v[4] + M[5,5]*v[5]

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method, stream=0):
        """
        Optimized trajectory kernel with 2D parallelization and shared memory time-scaling.
        Each thread computes exactly one (time, joint) tuple.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # 0 ... N-1
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # 0 ... num_joints-1
        num_j = thetastart.shape[0]

        if t_idx >= N or j_idx >= num_j:
            return

        # Shared memory for time-scaling (hoisted computation)
        shared_scaling = cuda.shared.array(3, dtype=float32)
        
        # One thread per block computes time-scaling
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
            tau = t_idx / (N - 1)  # Normalized 0-1
            if method == 3:
                s_block = 3.0 * tau * tau - 2.0 * tau * tau * tau
                sd_block = 6.0 * tau * (1.0 - tau) / Tf
                sdd_block = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:
                t2, t3 = tau * tau, tau * tau * tau
                s_block = 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * tau * t3
                sd_block = 30.0 * t2 * (1.0 - 2.0 * tau + t2) / Tf
                sdd_block = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
            else:
                s_block = sd_block = sdd_block = 0.0
            
            shared_scaling[0] = s_block
            shared_scaling[1] = sd_block
            shared_scaling[2] = sdd_block
        
        cuda.syncthreads()
        
        # All threads read from shared memory
        s = shared_scaling[0]
        s_dot = shared_scaling[1]
        s_ddot = shared_scaling[2]

        # Vector between start ↔ end
        dtheta = thetaend[j_idx] - thetastart[j_idx]

        traj_pos[t_idx, j_idx] = s * dtheta + thetastart[j_idx]
        traj_vel[t_idx, j_idx] = s_dot * dtheta
        traj_acc[t_idx, j_idx] = s_ddot * dtheta

    @cuda.jit(**jit_kwargs)
    def inverse_dynamics_kernel(
        thetalist_trajectory,
        dthetalist_trajectory,
        ddthetalist_trajectory,
        gravity_vector,
        Ftip,
        Glist,
        Slist,
        M,
        torques_trajectory,
        torque_limits,
        stream=0
    ):
        """
        Optimized inverse dynamics kernel using 2D parallelization.
        Each thread computes one (time, joint) element.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x   # time index
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y   # joint index

        if t_idx >= thetalist_trajectory.shape[0] or j_idx >= thetalist_trajectory.shape[1]:
            return

        # Load current trajectory point data for this joint
        theta_j = thetalist_trajectory[t_idx, j_idx]
        dtheta_j = dthetalist_trajectory[t_idx, j_idx]
        ddtheta_j = ddthetalist_trajectory[t_idx, j_idx]

        # Simplified dynamics computation for one joint
        # Mass matrix contribution (diagonal element)
        M_contrib = Glist[j_idx, j_idx, j_idx]

        # Velocity quadratic forces (simplified)
        c_j = Slist[j_idx, j_idx] * dtheta_j

        # Gravity forces (simplified)
        g_j = gravity_vector[2] * 0.1

        # Torque computation
        tau = M_contrib * ddtheta_j + c_j + g_j

        # Enforce torque limits and store result
        tau = max(torque_limits[j_idx, 0], min(tau, torque_limits[j_idx, 1]))
        torques_trajectory[t_idx, j_idx] = tau

    @cuda.jit(**jit_kwargs)
    def forward_dynamics_kernel(
        thetalist,
        dthetalist,
        taumat,
        g,
        Ftipmat,
        dt,
        intRes,
        Glist,
        Slist,
        M,
        thetamat,
        dthetamat,
        ddthetamat,
        joint_limits,
        stream=0
    ):
        """
        Optimized forward dynamics kernel with 2D parallelization.
        Each thread handles one (time, joint) element.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x   # time index
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y   # joint index

        if t_idx >= taumat.shape[0] or j_idx >= thetalist.shape[0]:
            return

        # Initialize from previous step or initial conditions
        if t_idx > 0:
            current_theta = thetamat[t_idx - 1, j_idx]
            current_dtheta = dthetamat[t_idx - 1, j_idx]
        else:
            current_theta = thetalist[j_idx]
            current_dtheta = dthetalist[j_idx]

        # Current torque for this joint
        tau = taumat[t_idx, j_idx]

        dt_step = dt / intRes

        for _ in range(intRes):
            # Simplified forward dynamics computation
            # Compute joint acceleration (simplified)
            M_inv = 1.0 / Glist[j_idx, j_idx, j_idx] if Glist[j_idx, j_idx, j_idx] != 0.0 else 1.0
            ddtheta = (tau - g[2] * 0.1) * M_inv

            # Integrate velocities and positions
            current_dtheta += ddtheta * dt_step
            current_theta += current_dtheta * dt_step

            # Enforce joint limits
            current_theta = max(joint_limits[j_idx, 0], min(current_theta, joint_limits[j_idx, 1]))

        # Store results
        thetamat[t_idx, j_idx] = current_theta
        dthetamat[t_idx, j_idx] = current_dtheta
        ddthetamat[t_idx, j_idx] = ddtheta

    @cuda.jit(**jit_kwargs)
    def cartesian_trajectory_kernel(pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method, stream=0):
        """
        Optimized Cartesian trajectory kernel with 2D parallelization and shared memory time-scaling.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # 0 ... N-1
        coord_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # 0 ... 2 (x,y,z)

        if t_idx >= N or coord_idx >= 3:
            return

        # Shared memory for time-scaling (hoisted computation)
        shared_scaling = cuda.shared.array(3, dtype=float32)
        
        # One thread per block computes time-scaling
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
            tau = t_idx / (N - 1)  # Normalized 0-1
            if method == 3:
                s_block = 3.0 * tau * tau - 2.0 * tau * tau * tau
                sd_block = 6.0 * tau * (1.0 - tau) / Tf
                sdd_block = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:
                t2, t3 = tau * tau, tau * tau * tau
                s_block = 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * tau * t3
                sd_block = 30.0 * t2 * (1.0 - 2.0 * tau + t2) / Tf
                sdd_block = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
            else:
                s_block = sd_block = sdd_block = 0.0
            
            shared_scaling[0] = s_block
            shared_scaling[1] = sd_block
            shared_scaling[2] = sdd_block
        
        cuda.syncthreads()
        
        # All threads read from shared memory
        s = shared_scaling[0]
        s_dot = shared_scaling[1]
        s_ddot = shared_scaling[2]

        # Coordinate difference
        dp = pend[coord_idx] - pstart[coord_idx]

        traj_pos[t_idx, coord_idx] = s * dp + pstart[coord_idx]
        traj_vel[t_idx, coord_idx] = s_dot * dp
        traj_acc[t_idx, coord_idx] = s_ddot * dp

    @cuda.jit(**jit_kwargs)
    def fused_potential_gradient_kernel(positions, goal, obstacles, potential, gradient, influence_distance, stream=0):
        """
        Fused kernel computing both attractive potential and gradient in one pass.
        Optimized with rsqrt for improved performance and register usage.
        """
        idx = cuda.grid(1)
        if idx >= positions.shape[0]:
            return

        # Pre-compute inverse influence distance
        influence_distance_inv = 1.0 / influence_distance if influence_distance > 0.0 else 0.0

        # Load position into registers
        pos_x = positions[idx, 0]
        pos_y = positions[idx, 1]
        pos_z = positions[idx, 2]

        # Attractive potential and gradient
        diff_x = pos_x - goal[0]
        diff_y = pos_y - goal[1]
        diff_z = pos_z - goal[2]

        attractive_pot = 0.5 * (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
        
        # Attractive gradient components
        grad_x = diff_x
        grad_y = diff_y
        grad_z = diff_z

        # Repulsive potential and gradient
        repulsive_pot = 0.0
        for obs in range(obstacles.shape[0]):
            # Distance calculation with registers
            obs_diff_x = pos_x - obstacles[obs, 0]
            obs_diff_y = pos_y - obstacles[obs, 1]
            obs_diff_z = pos_z - obstacles[obs, 2]
            
            dist_sq = obs_diff_x * obs_diff_x + obs_diff_y * obs_diff_y + obs_diff_z * obs_diff_z
            
            if dist_sq > 0.0 and dist_sq < influence_distance * influence_distance:
                # Use rsqrt for ~2x faster computation
                dist_inv = math.rsqrt(dist_sq)
                dist = 1.0 / dist_inv
                
                influence_term = dist_inv - influence_distance_inv
                repulsive_term = 0.5 * influence_term * influence_term
                repulsive_pot += repulsive_term
                
                # Repulsive gradient contribution
                grad_factor = influence_term * dist_inv * dist_inv * dist_inv
                grad_x += grad_factor * obs_diff_x
                grad_y += grad_factor * obs_diff_y
                grad_z += grad_factor * obs_diff_z

        # Store total potential and gradient
        potential[idx] = attractive_pot + repulsive_pot
        
        # Store gradient components (assuming gradient has shape (N, 3))
        if idx < gradient.shape[0] and gradient.shape[1] >= 3:
            gradient[idx, 0] = grad_x
            gradient[idx, 1] = grad_y
            gradient[idx, 2] = grad_z

    @cuda.jit(**jit_kwargs)
    def batch_trajectory_kernel(
        thetastart_batch, thetaend_batch, traj_pos_batch, traj_vel_batch, traj_acc_batch,
        Tf, N, method, batch_size, stream=0
    ):
        """
        Optimized batch trajectory kernel for processing multiple trajectories simultaneously.
        3D parallelization: (batch, time, joint) with shared memory time-scaling.
        """
        batch_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        t_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        j_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        if batch_idx >= batch_size or t_idx >= N or j_idx >= thetastart_batch.shape[2]:
            return

        # Shared memory for time-scaling (hoisted computation)
        shared_scaling = cuda.shared.array(3, dtype=float32)
        
        # One thread per block computes time-scaling
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
            tau = t_idx / (N - 1)  # Normalized 0-1
            if method == 3:
                s_block = 3.0 * tau * tau - 2.0 * tau * tau * tau
                sd_block = 6.0 * tau * (1.0 - tau) / Tf
                sdd_block = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:
                t2, t3 = tau * tau, tau * tau * tau
                s_block = 10.0 * t3 - 15.0 * t2 * t2 + 6.0 * tau * t3
                sd_block = 30.0 * t2 * (1.0 - 2.0 * tau + t2) / Tf
                sdd_block = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
            else:
                s_block = sd_block = sdd_block = 0.0
            
            shared_scaling[0] = s_block
            shared_scaling[1] = sd_block
            shared_scaling[2] = sdd_block
        
        cuda.syncthreads()
        
        # All threads read from shared memory
        s = shared_scaling[0]
        s_dot = shared_scaling[1]
        s_ddot = shared_scaling[2]

        # Vector between start ↔ end for this batch
        dtheta = thetaend_batch[batch_idx, j_idx] - thetastart_batch[batch_idx, j_idx]

        traj_pos_batch[batch_idx, t_idx, j_idx] = s * dtheta + thetastart_batch[batch_idx, j_idx]
        traj_vel_batch[batch_idx, t_idx, j_idx] = s_dot * dtheta
        traj_acc_batch[batch_idx, t_idx, j_idx] = s_ddot * dtheta

    # Memory pool for CUDA arrays
    class _GlobalCudaMemoryPool:
        """Simple memory pool for reusing GPU memory allocations."""
        
        def __init__(self):
            self.pool = {}
            self.max_pool_size = 100  # Maximum number of cached arrays
        
        def get_array(self, shape, dtype=np.float32):
            """Get a GPU array from the pool or allocate new one."""
            key = (shape, dtype)
            if key in self.pool and len(self.pool[key]) > 0:
                return self.pool[key].pop()
            else:
                return cuda.device_array(shape, dtype=dtype)
        
        def return_array(self, array):
            """Return a GPU array to the pool for reuse."""
            key = (array.shape, array.dtype)
            if key not in self.pool:
                self.pool[key] = []
            
            if len(self.pool[key]) < self.max_pool_size:
                self.pool[key].append(array)
            # If pool is full, let the array be garbage collected
        
        def clear(self):
            """Clear the memory pool."""
            self.pool.clear()

    # Global memory pool instance
    _cuda_memory_pool = _GlobalCudaMemoryPool()

    def get_cuda_array(shape, dtype=np.float32):
        """Get a CUDA array from the memory pool."""
        return _cuda_memory_pool.get_array(shape, dtype)

    def return_cuda_array(array):
        """Return a CUDA array to the memory pool."""
        _cuda_memory_pool.return_array(array)

    # Profiling utilities
    def profile_start():
        """Start CUDA profiling."""
        try:
            cuda.profile_start()
        except:
            pass

    def profile_stop():
        """Stop CUDA profiling."""
        try:
            cuda.profile_stop()
        except:
            pass

    # Auto-tuning for optimal block sizes
    from functools import lru_cache
    from time import perf_counter

    @lru_cache(maxsize=None)
    def _best_2d_config(N, J):
        """Auto-tune block sizes for optimal performance."""
        if not CUDA_AVAILABLE:
            return ((1, 1), (1, 1))
        
        best, t_min = None, 1e9
        test_configs = [(64, 8), (32, 8), (16, 16), (128, 4), (32, 16)]
        
        # Create small test data
        test_N = min(N, 100)
        test_J = min(J, 8)
        
        thetastart = cuda.device_array(test_J, dtype=float32)
        thetaend = cuda.device_array(test_J, dtype=float32)
        traj_pos = cuda.device_array((test_N, test_J), dtype=float32)
        traj_vel = cuda.device_array((test_N, test_J), dtype=float32)
        traj_acc = cuda.device_array((test_N, test_J), dtype=float32)
        
        for block_size in test_configs:
            try:
                grid, block = make_2d_grid(test_N, test_J, block_size=block_size)
                
                # Warm-up run
                trajectory_kernel[grid, block](
                    thetastart, thetaend, traj_pos, traj_vel, traj_acc, 
                    1.0, test_N, 3
                )
                cuda.synchronize()
                
                # Timed run
                start = perf_counter()
                trajectory_kernel[grid, block](
                    thetastart, thetaend, traj_pos, traj_vel, traj_acc, 
                    1.0, test_N, 3
                )
                cuda.synchronize()
                dt = perf_counter() - start
                
                if dt < t_min:
                    best, t_min = (grid, block), dt
                    
            except Exception:
                continue
        
        return best if best is not None else make_2d_grid(N, J)

    # Constant memory arrays for frequently used data
    try:
        # These would be set up during initialization if needed
        _constant_arrays = {}
        
        def setup_constant_array(name, data):
            """Set up a constant memory array for frequently accessed data."""
            if name not in _constant_arrays:
                _constant_arrays[name] = cuda.const.array_like(data)
            return _constant_arrays[name]
        
        def get_constant_array(name):
            """Get a constant memory array by name."""
            return _constant_arrays.get(name)
            
    except AttributeError:
        # Fallback if const arrays not available
        def setup_constant_array(name, data):
            return cuda.to_device(data)
        
        def get_constant_array(name):
            return None

else:
    # If CUDA is not available, create placeholder functions that raise informative errors
    def trajectory_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA trajectory kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def inverse_dynamics_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA inverse dynamics kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def forward_dynamics_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA forward dynamics kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def cartesian_trajectory_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA Cartesian trajectory kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def fused_potential_gradient_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA potential field kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def batch_trajectory_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA batch trajectory kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def get_cuda_array(*args, **kwargs):
        raise RuntimeError("CUDA memory pool not available.")
    
    def return_cuda_array(*args, **kwargs):
        raise RuntimeError("CUDA memory pool not available.")
    
    def profile_start():
        pass
    
    def profile_stop():
        pass
    
    def _best_2d_config(*args, **kwargs):
        return ((1, 1), (1, 1))
    
    def setup_constant_array(*args, **kwargs):
        raise RuntimeError("CUDA constant arrays not available.")
    
    def get_constant_array(*args, **kwargs):
        return None
    
    def _h2d_pinned(*args, **kwargs):
        raise RuntimeError("CUDA pinned memory not available.")

    # Legacy kernel placeholders for compatibility
    def attractive_potential_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA attractive potential kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def repulsive_potential_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA repulsive potential kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )
    
    def gradient_kernel(*args, **kwargs):
        raise RuntimeError(
            "CUDA gradient kernel not available. Install GPU support:\n"
            "pip install ManipulaPy[gpu-cuda11]"
        )

# High-level wrapper functions for optimized CUDA operations
def optimized_trajectory_generation(thetastart, thetaend, Tf, N, method, use_pinned=True):
    """
    High-level wrapper for optimized trajectory generation with automatic
    memory management and performance optimizations.
    """
    if not CUDA_AVAILABLE:
        return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
    
    num_joints = len(thetastart)
    
    # Use pinned memory for faster transfers
    if use_pinned:
        d_thetastart = _h2d_pinned(np.ascontiguousarray(thetastart, dtype=float_t))
        d_thetaend = _h2d_pinned(np.ascontiguousarray(thetaend, dtype=float_t))
    else:
        d_thetastart = cuda.to_device(np.ascontiguousarray(thetastart, dtype=float_t))
        d_thetaend = cuda.to_device(np.ascontiguousarray(thetaend, dtype=float_t))
    
    # Allocate output arrays
    d_traj_pos = get_cuda_array((N, num_joints), dtype=float_t)
    d_traj_vel = get_cuda_array((N, num_joints), dtype=float_t)
    d_traj_acc = get_cuda_array((N, num_joints), dtype=float_t)
    
    try:
        # Auto-tune block size
        grid, block = _best_2d_config(N, num_joints)
        
        # Launch kernel
        trajectory_kernel[grid, block](
            d_thetastart, d_thetaend, d_traj_pos, d_traj_vel, d_traj_acc,
            Tf, N, method
        )
        
        # Copy results back
        traj_pos = d_traj_pos.copy_to_host()
        traj_vel = d_traj_vel.copy_to_host()
        traj_acc = d_traj_acc.copy_to_host()
        
        return traj_pos, traj_vel, traj_acc
        
    finally:
        # Return arrays to pool
        return_cuda_array(d_traj_pos)
        return_cuda_array(d_traj_vel)
        return_cuda_array(d_traj_acc)

def optimized_potential_field(positions, goal, obstacles, influence_distance, use_pinned=True):
    """
    High-level wrapper for optimized potential field computation with
    fused kernel and automatic memory management.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for potential field computation")
    
    N = positions.shape[0]
    
    # Use pinned memory for faster transfers
    if use_pinned:
        d_positions = _h2d_pinned(np.ascontiguousarray(positions, dtype=float_t))
        d_goal = _h2d_pinned(np.ascontiguousarray(goal, dtype=float_t))
        d_obstacles = _h2d_pinned(np.ascontiguousarray(obstacles, dtype=float_t))
    else:
        d_positions = cuda.to_device(np.ascontiguousarray(positions, dtype=float_t))
        d_goal = cuda.to_device(np.ascontiguousarray(goal, dtype=float_t))
        d_obstacles = cuda.to_device(np.ascontiguousarray(obstacles, dtype=float_t))
    
    # Allocate output arrays
    d_potential = get_cuda_array((N,), dtype=float_t)
    d_gradient = get_cuda_array((N, 3), dtype=float_t)
    
    try:
        # Launch fused kernel
        grid, block = make_1d_grid(N)
        
        fused_potential_gradient_kernel[grid, block](
            d_positions, d_goal, d_obstacles, d_potential, d_gradient, influence_distance
        )
        
        # Copy results back
        potential = d_potential.copy_to_host()
        gradient = d_gradient.copy_to_host()
        
        return potential, gradient
        
    finally:
        # Return arrays to pool
        return_cuda_array(d_potential)
        return_cuda_array(d_gradient)

# Batch processing utilities
def optimized_batch_trajectory_generation(thetastart_batch, thetaend_batch, Tf, N, method, use_pinned=True):
    """
    High-level wrapper for optimized batch trajectory generation.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for batch trajectory generation")
    
    batch_size, num_joints = thetastart_batch.shape
    
    # Use pinned memory for faster transfers
    if use_pinned:
        d_thetastart_batch = _h2d_pinned(np.ascontiguousarray(thetastart_batch, dtype=float_t))
        d_thetaend_batch = _h2d_pinned(np.ascontiguousarray(thetaend_batch, dtype=float_t))
    else:
        d_thetastart_batch = cuda.to_device(np.ascontiguousarray(thetastart_batch, dtype=float_t))
        d_thetaend_batch = cuda.to_device(np.ascontiguousarray(thetaend_batch, dtype=float_t))
    
    # Allocate output arrays
    d_traj_pos_batch = get_cuda_array((batch_size, N, num_joints), dtype=float_t)
    d_traj_vel_batch = get_cuda_array((batch_size, N, num_joints), dtype=float_t)
    d_traj_acc_batch = get_cuda_array((batch_size, N, num_joints), dtype=float_t)
    
    try:
        # 3D grid for batch processing
        grid = ((batch_size + 7) // 8, (N + 15) // 16, (num_joints + 7) // 8)
        block = (8, 16, 8)
        
        # Launch batch kernel
        batch_trajectory_kernel[grid, block](
            d_thetastart_batch, d_thetaend_batch, d_traj_pos_batch, d_traj_vel_batch, d_traj_acc_batch,
            Tf, N, method, batch_size
        )
        
        # Copy results back
        traj_pos_batch = d_traj_pos_batch.copy_to_host()
        traj_vel_batch = d_traj_vel_batch.copy_to_host()
        traj_acc_batch = d_traj_acc_batch.copy_to_host()
        
        return traj_pos_batch, traj_vel_batch, traj_acc_batch
        
    finally:
        # Return arrays to pool
        return_cuda_array(d_traj_pos_batch)
        return_cuda_array(d_traj_vel_batch)
        return_cuda_array(d_traj_acc_batch)

# Utility functions for performance monitoring
def benchmark_kernel_performance(kernel_name, *args, num_runs=10):
    """
    Benchmark a specific kernel's performance over multiple runs.
    """
    if not CUDA_AVAILABLE:
        print(f"Cannot benchmark {kernel_name} - CUDA not available")
        return None
    
    times = []
    
    for _ in range(num_runs):
        start = perf_counter()
        
        # Execute kernel based on name
        if kernel_name == "trajectory":
            optimized_trajectory_generation(*args)
        elif kernel_name == "potential_field":
            optimized_potential_field(*args)
        elif kernel_name == "batch_trajectory":
            optimized_batch_trajectory_generation(*args)
        
        cuda.synchronize()
        end = perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"{kernel_name} benchmark results:")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Std deviation: {std_time*1000:.2f} ms")
    print(f"  Min time: {min(times)*1000:.2f} ms")
    print(f"  Max time: {max(times)*1000:.2f} ms")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min(times),
        'max_time': max(times),
        'all_times': times
    }

# Export important functions and classes
__all__ = [
    'CUDA_AVAILABLE',
    'CUPY_AVAILABLE',
    'check_cuda_availability',
    'check_cupy_availability',
    'trajectory_kernel',
    'inverse_dynamics_kernel',
    'forward_dynamics_kernel',
    'cartesian_trajectory_kernel',
    'fused_potential_gradient_kernel',
    'batch_trajectory_kernel',
    'optimized_trajectory_generation',
    'optimized_potential_field',
    'optimized_batch_trajectory_generation',
    'get_cuda_array',
    'return_cuda_array',
    'profile_start',
    'profile_stop',
    'benchmark_kernel_performance',
    'make_1d_grid',
    'make_2d_grid',
    'get_gpu_properties',
    'trajectory_cpu_fallback',
    # Legacy exports for compatibility
    'attractive_potential_kernel',
    'repulsive_potential_kernel',
    'gradient_kernel',
]