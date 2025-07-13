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

# IMPROVED CUDA DETECTION LOGIC WITH BETTER ERROR HANDLING
def _detect_cuda_capability():
    """
    Comprehensive CUDA detection that handles various failure modes gracefully.
    
    Returns:
        tuple: (cuda_available, cuda_module, float32, int32, error_message)
    """
    try:
        # Step 1: Try importing numba.cuda - avoid float16 completely
        from numba import cuda, float32, int32
        
        # Step 2: Check if CUDA is available in numba (this might fail with CUDA_ERROR_NO_DEVICE)
        try:
            cuda_available = cuda.is_available()
            if not cuda_available:
                return False, None, None, None, "numba.cuda.is_available() returned False - likely driver/GPU issue"
        except Exception as e:
            # Common issues: CUDA_ERROR_NO_DEVICE, driver problems, etc.
            return False, None, None, None, f"numba.cuda.is_available() failed: {e}"
        
        # Step 3: Try to detect actual GPU devices
        try:
            devices = cuda.list_devices()
            if not devices:
                return False, None, None, None, "No CUDA devices detected by numba"
        except Exception as e:
            return False, None, None, None, f"Failed to list CUDA devices: {e}"
        
        # Step 4: Try a simple GPU operation to verify functionality
        try:
            # Test basic GPU memory allocation
            test_array = cuda.device_array(10, dtype=np.float32)
            
            # Test basic kernel compilation (but don't execute)
            @cuda.jit
            def test_kernel(arr):
                idx = cuda.grid(1)
                if idx < arr.shape[0]:
                    arr[idx] = idx
            
            # Try to get device properties
            current_device = cuda.get_current_device()
            _ = current_device.MULTIPROCESSOR_COUNT
            
            # Clean up test array
            del test_array
            
            return True, cuda, float32, int32, None
            
        except Exception as e:
            return False, None, None, None, f"CUDA functionality test failed: {e}"
            
    except ImportError as e:
        return False, None, None, None, f"numba.cuda import failed: {e}"
    except Exception as e:
        return False, None, None, None, f"Unexpected error in CUDA detection: {e}"

# Perform the detection - no float16!
CUDA_AVAILABLE, cuda, float32, int32, _cuda_error = _detect_cuda_capability()

# Set up the rest based on detection results
if not CUDA_AVAILABLE:
    # CUDA not available - create mock objects
    _cuda_error_msg = _cuda_error or "CUDA not available"
    
    class MockCuda:
        @staticmethod
        def jit(func=None, device=False, inline=False, fastmath=False):
            """Mock decorator for when CUDA is not available"""
            def wrapper(*args, **kwargs):
                raise RuntimeError(
                    f"CUDA functionality not available: {_cuda_error_msg}\n"
                    "To fix CUDA issues:\n"
                    "1. Check GPU is properly connected: nvidia-smi\n"
                    "2. Reinstall NVIDIA drivers: sudo apt purge nvidia-* && sudo apt install nvidia-driver-545\n"
                    "3. Reboot system\n"
                    "4. Verify with: python -c 'from numba import cuda; print(cuda.is_available())'"
                )
            return wrapper if func is None else wrapper(func)
        
        @staticmethod
        def grid(dim):
            return 0
        
        @staticmethod
        def device_array(*args, **kwargs):
            raise RuntimeError(f"CUDA not available: {_cuda_error_msg}")
        
        @staticmethod
        def to_device(*args, **kwargs):
            raise RuntimeError(f"CUDA not available: {_cuda_error_msg}")
        
        @staticmethod
        def pinned_array(*args, **kwargs):
            raise RuntimeError(f"CUDA not available: {_cuda_error_msg}")
        
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def list_devices():
            return []
        
        @staticmethod
        def get_current_device():
            class MockDevice:
                MULTIPROCESSOR_COUNT = 1
                MAX_THREADS_PER_BLOCK = 1024
                MAX_SHARED_MEMORY_PER_BLOCK = 48*1024
                MAX_BLOCK_DIM_X = 1024
                MAX_BLOCK_DIM_Y = 1024
            return MockDevice()
        
        @staticmethod
        def shared():
            class SharedMock:
                @staticmethod
                def array(*args, **kwargs):
                    raise RuntimeError(f"CUDA not available: {_cuda_error_msg}")
            return SharedMock()
        
        @staticmethod
        def local():
            class LocalMock:
                @staticmethod
                def array(*args, **kwargs):
                    raise RuntimeError(f"CUDA not available: {_cuda_error_msg}")
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
    # Ensure we have float types even when CUDA is not available
    if float32 is None:
        float32 = np.float32
    if int32 is None:
        int32 = np.int32

# Check for CuPy availability (separate from numba.cuda)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Configure float precision - always use float32 to avoid numba float16 issues
float_t = float32

def check_cuda_availability():
    """
    Check if CUDA is available and provide helpful diagnostic information.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    if CUDA_AVAILABLE:
        try:
            # Additional verification - try to get device info
            devices = cuda.list_devices()
            device_info = []
            for device in devices:
                device_info.append(f"  - {device}")
            
            print(f"✅ CUDA is available and functional!")
            print(f"✅ Detected {len(devices)} CUDA device(s):")
            for info in device_info:
                print(info)
                
            # Get current device properties
            current_device = cuda.get_current_device()
            print(f"✅ Current device: SM count={current_device.MULTIPROCESSOR_COUNT}, "
                  f"Max threads/block={current_device.MAX_THREADS_PER_BLOCK}")
                  
        except Exception as e:
            print(f"⚠️ CUDA available but device info failed: {e}")
        
        return True
    else:
        print(f"❌ CUDA not available: {_cuda_error}")
        
        # Provide specific diagnostic help based on the error
        if "CUDA_ERROR_NO_DEVICE" in str(_cuda_error):
            print("\n🔧 CUDA_ERROR_NO_DEVICE detected - GPU not found by driver!")
            print("Solutions:")
            print("1. Check GPU is detected: nvidia-smi")
            print("2. Reinstall NVIDIA drivers:")
            print("   sudo apt purge nvidia-*")
            print("   sudo apt update") 
            print("   sudo apt install nvidia-driver-545")  # or latest stable
            print("3. Reboot system")
            print("4. Check PCIe connection (reseat GPU if needed)")
            
        elif "import" in str(_cuda_error).lower():
            print("\n🔧 Import/numba version issue")
            print("Solutions:")
            print("1. Update numba: pip install --upgrade numba")
            print("2. Check CUDA toolkit compatibility with numba version")
            
        else:
            print("\n🔧 General CUDA troubleshooting:")
            print("1. Ensure NVIDIA GPU drivers are installed")
            print("2. Install CUDA toolkit")
            print("3. Verify driver with: nvidia-smi")
            print("4. Test with: python -c 'from numba import cuda; print(cuda.is_available())'")
        
        # Additional diagnostics
        print(f"\n📊 Diagnostic info:")
        try:
            from numba import cuda as test_cuda
            print(f"📊 numba.cuda import: ✅")
            try:
                is_avail = test_cuda.is_available()
                print(f"📊 numba.cuda.is_available(): {is_avail}")
            except Exception as e:
                print(f"📊 numba.cuda.is_available() failed: {e}")
            
            try:
                devices = test_cuda.list_devices()
                print(f"📊 Device list attempt: {len(devices)} devices")
                for i, device in enumerate(devices):
                    print(f"📊   Device {i}: {device}")
            except Exception as e:
                print(f"📊 Device detection failed: {e}")
        except ImportError:
            print(f"📊 numba.cuda import: ❌")
            
        return False

def check_cupy_availability():
    """Check if CuPy is available for GPU array operations."""
    if not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Some GPU array operations will not work.\n"
            "Install with: pip install cupy-cuda12x",  # Updated for newer CUDA
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
    """Create a 1D grid configuration for CUDA kernel launch."""
    if size <= 0:
        return (1,), (1,)
    
    # Ensure minimum occupancy for small datasets
    if size < threads:
        threads = max(32, 2**int(math.log2(size)))  # Round down to power of 2
    
    blocks = (size + threads - 1) // threads
    return (blocks,), (threads,)

def make_2d_grid(N: int, num_joints: int, block_size: Tuple[int, int] = (128, 8)):
    """Compute optimal 2D grid configuration for CUDA kernel launch."""
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
    """Retrieve current CUDA device properties."""
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
    """Compute trajectory on CPU when CUDA is unavailable."""
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
        """Optimized 6x6 matrix-vector multiplication."""
        # Unrolled matrix-vector multiplication for better performance
        result[0] = M[0,0]*v[0] + M[0,1]*v[1] + M[0,2]*v[2] + M[0,3]*v[3] + M[0,4]*v[4] + M[0,5]*v[5]
        result[1] = M[1,0]*v[0] + M[1,1]*v[1] + M[1,2]*v[2] + M[1,3]*v[3] + M[1,4]*v[4] + M[1,5]*v[5]
        result[2] = M[2,0]*v[0] + M[2,1]*v[1] + M[2,2]*v[2] + M[2,3]*v[3] + M[2,4]*v[4] + M[2,5]*v[5]
        result[3] = M[3,0]*v[0] + M[3,1]*v[1] + M[3,2]*v[2] + M[3,3]*v[3] + M[3,4]*v[4] + M[3,5]*v[5]
        result[4] = M[4,0]*v[0] + M[4,1]*v[1] + M[4,2]*v[2] + M[4,3]*v[3] + M[4,4]*v[4] + M[4,5]*v[5]
        result[5] = M[5,0]*v[0] + M[5,1]*v[1] + M[5,2]*v[2] + M[5,3]*v[3] + M[5,4]*v[4] + M[5,5]*v[5]

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
        """
        Highly optimized CUDA kernel for trajectory generation with advanced optimizations.
        
        Optimizations applied:
        1. Coalesced memory access patterns
        2. Shared memory for time-scaling computation reuse
        3. Register-based computations to minimize memory access
        4. Optimized thread block configuration
        5. Eliminated thread divergence
        6. Vectorized operations within threads
        """
        # 2D thread indexing for optimal memory coalescing
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        # Global thread indices
        t_idx = bx * cuda.blockDim.x + tx  # Time index
        j_idx = by * cuda.blockDim.y + ty  # Joint index
        
        num_joints = thetastart.shape[0]
        
        if t_idx >= N or j_idx >= num_joints:
            return
        
        # Shared memory for time-scaling coefficients (one per block)
        # Using more shared memory to reduce recomputation
        shared_time_data = cuda.shared.array(6, dtype=float32)  # [s, s_dot, s_ddot, tau, tau2, tau3]
        
        # Only one thread per block computes time scaling
        if tx == 0 and ty == 0:
            tau = t_idx / (N - 1.0)  # Normalized time [0,1]
            tau2 = tau * tau
            tau3 = tau2 * tau
            
            # Pre-compute powers for reuse
            shared_time_data[3] = tau    # Store tau
            shared_time_data[4] = tau2   # Store tau^2  
            shared_time_data[5] = tau3   # Store tau^3
            
            if method == 3:  # Cubic polynomial
                s = 3.0 * tau2 - 2.0 * tau3
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            elif method == 5:  # Quintic polynomial
                tau4 = tau2 * tau2
                tau5 = tau4 * tau
                s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
                s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
                s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
            else:  # Linear fallback
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0
            
            shared_time_data[0] = s
            shared_time_data[1] = s_dot
            shared_time_data[2] = s_ddot
        
        # Synchronize to ensure shared data is ready
        cuda.syncthreads()
        
        # All threads read from shared memory (coalesced access)
        s = shared_time_data[0]
        s_dot = shared_time_data[1] 
        s_ddot = shared_time_data[2]
        
        # Register-based computation to minimize memory access
        start_angle = thetastart[j_idx]  # Single memory read
        end_angle = thetaend[j_idx]      # Single memory read
        delta_angle = end_angle - start_angle  # Register computation
        
        # Compute trajectory values in registers
        pos_val = start_angle + s * delta_angle
        vel_val = s_dot * delta_angle
        acc_val = s_ddot * delta_angle
        
        # Coalesced memory writes (single write per thread)
        traj_pos[t_idx, j_idx] = pos_val
        traj_vel[t_idx, j_idx] = vel_val
        traj_acc[t_idx, j_idx] = acc_val

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_vectorized(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
        """
        Advanced vectorized CUDA kernel that processes multiple time steps per thread.
        
        Each thread processes VECTOR_SIZE consecutive time steps to:
        1. Improve memory bandwidth utilization
        2. Reduce kernel launch overhead
        3. Better utilize GPU compute resources
        4. Optimize for larger trajectory sizes
        """
        VECTOR_SIZE = 4  # Each thread processes 4 time steps
        
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        # Base time index for this thread
        base_t_idx = (bx * cuda.blockDim.x + tx) * VECTOR_SIZE
        j_idx = by * cuda.blockDim.y + ty
        
        num_joints = thetastart.shape[0]
        
        if j_idx >= num_joints:
            return
        
        # Shared memory for joint data (reused across time steps)
        shared_joint_data = cuda.shared.array((32, 2), dtype=float32)  # [start, delta] for up to 32 joints
        
        # Load joint data to shared memory (coalesced)
        if tx == 0 and j_idx < 32:
            start_val = thetastart[j_idx]
            shared_joint_data[j_idx, 0] = start_val
            shared_joint_data[j_idx, 1] = thetaend[j_idx] - start_val  # delta
        
        cuda.syncthreads()
        
        # Read joint data from shared memory if available, otherwise from global
        if j_idx < 32:
            start_angle = shared_joint_data[j_idx, 0]
            delta_angle = shared_joint_data[j_idx, 1]
        else:
            start_angle = thetastart[j_idx]
            delta_angle = thetaend[j_idx] - start_angle
        
        # Process VECTOR_SIZE time steps in this thread
        for vec_i in range(VECTOR_SIZE):
            t_idx = base_t_idx + vec_i
            
            if t_idx >= N:
                break
            
            # Compute time scaling for this time step
            tau = t_idx / (N - 1.0)
            
            if method == 3:  # Cubic - optimized with fewer operations
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau2 * (3.0 - 2.0 * tau)  # Factored form
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            elif method == 5:  # Quintic - optimized computation
                tau2 = tau * tau
                tau3 = tau2 * tau
                tau4 = tau2 * tau2
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)  # Factored form
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0
            
            # Compute and store results (coalesced writes)
            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_memory_optimized(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
        """
        Memory-bandwidth optimized kernel for very large trajectories.
        
        Optimizations:
        1. Minimizes global memory access
        2. Uses shared memory extensively
        3. Optimized for high memory bandwidth utilization
        4. Reduces register pressure
        """
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        # Thread block indices
        t_block = cuda.blockIdx.x
        j_block = cuda.blockIdx.y
        
        # Global indices with stride
        stride_t = cuda.gridDim.x * cuda.blockDim.x
        stride_j = cuda.gridDim.y * cuda.blockDim.y
        
        # Starting indices for this thread
        t_start = t_block * cuda.blockDim.x + tx
        j_start = j_block * cuda.blockDim.y + ty
        
        # Shared memory for time coefficients and joint data
        shared_coeffs = cuda.shared.array(3, dtype=float32)
        shared_joints = cuda.shared.array((8, 3), dtype=float32)  # [start, end, delta] for 8 joints
        
        # Process multiple elements with stride (grid-stride loop)
        for j_idx in range(j_start, thetastart.shape[0], stride_j):
            # Load joint data to shared memory
            local_j = ty
            if local_j < 8 and j_idx < thetastart.shape[0]:
                start_val = thetastart[j_idx]
                end_val = thetaend[j_idx]
                shared_joints[local_j, 0] = start_val
                shared_joints[local_j, 1] = end_val
                shared_joints[local_j, 2] = end_val - start_val  # delta
            
            cuda.syncthreads()
            
            for t_idx in range(t_start, N, stride_t):
                # Compute time scaling (shared across joint dimension)
                if ty == 0:
                    tau = t_idx / (N - 1.0)
                    
                    if method == 5:  # Focus on quintic for performance
                        tau_sq = tau * tau
                        tau_cb = tau_sq * tau
                        s = tau_cb * (10.0 + tau * (-15.0 + 6.0 * tau))
                        s_dot = tau_sq * (30.0 + tau * (-60.0 + 30.0 * tau)) / Tf
                        s_ddot = tau * (60.0 + tau * (-180.0 + 120.0 * tau)) / (Tf * Tf)
                    else:  # Simplified cubic
                        tau_sq = tau * tau
                        s = tau_sq * (3.0 - 2.0 * tau)
                        s_dot = 6.0 * tau * (1.0 - tau) / Tf
                        s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
                    
                    shared_coeffs[0] = s
                    shared_coeffs[1] = s_dot
                    shared_coeffs[2] = s_ddot
                
                cuda.syncthreads()
                
                # Read coefficients from shared memory
                s = shared_coeffs[0]
                s_dot = shared_coeffs[1]
                s_ddot = shared_coeffs[2]
                
                # Use joint data from shared memory if available
                if ty < 8 and j_idx < thetastart.shape[0]:
                    start_angle = shared_joints[ty, 0]
                    delta_angle = shared_joints[ty, 2]
                else:
                    start_angle = thetastart[j_idx] if j_idx < thetastart.shape[0] else 0.0
                    delta_angle = (thetaend[j_idx] - start_angle) if j_idx < thetastart.shape[0] else 0.0
                
                # Compute and store results
                if t_idx < N and j_idx < thetastart.shape[0]:
                    traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
                    traj_vel[t_idx, j_idx] = s_dot * delta_angle
                    traj_acc[t_idx, j_idx] = s_ddot * delta_angle
                
                cuda.syncthreads()

    def get_optimal_kernel_config(self, N, num_joints, kernel_type="standard"):
        """
        Automatically select optimal kernel and configuration based on problem size.
        
        Args:
            N: Number of trajectory points
            num_joints: Number of robot joints
            kernel_type: "standard", "vectorized", or "memory_optimized"
        
        Returns:
            dict: Optimal configuration including kernel function and launch parameters
        """
        total_elements = N * num_joints
        
        if not CUDA_AVAILABLE:
            return None
            
        device = cuda.get_current_device()
        max_threads = device.MAX_THREADS_PER_BLOCK
        sm_count = device.MULTIPROCESSOR_COUNT
        
        # Automatically select kernel based on problem characteristics
        if kernel_type == "auto":
            if total_elements < 10000:
                kernel_type = "standard"
            elif total_elements < 100000:
                kernel_type = "vectorized" 
            else:
                kernel_type = "memory_optimized"
        
        if kernel_type == "vectorized":
            # For vectorized kernel, adjust N for vector processing
            vector_size = 4
            effective_N = (N + vector_size - 1) // vector_size
            
            # Optimize for vectorized access patterns
            threads_x = min(256, max(32, effective_N))
            threads_y = min(max_threads // threads_x, max(1, num_joints))
            
            blocks_x = (effective_N + threads_x - 1) // threads_x
            blocks_y = (num_joints + threads_y - 1) // threads_y
            
            kernel_func = trajectory_kernel_vectorized
            
        elif kernel_type == "memory_optimized":
            # For memory-optimized kernel, use larger blocks
            threads_x = min(128, max(32, N // sm_count))
            threads_y = min(max_threads // threads_x, max(1, min(8, num_joints)))
            
            blocks_x = min(sm_count * 2, (N + threads_x - 1) // threads_x)
            blocks_y = min(sm_count * 2, (num_joints + threads_y - 1) // threads_y)
            
            kernel_func = trajectory_kernel_memory_optimized
            
        else:  # standard
            # Standard optimized configuration
            if num_joints <= 8:
                threads_x, threads_y = 128, min(8, num_joints)
            elif num_joints <= 16:
                threads_x, threads_y = 64, min(16, num_joints)
            else:
                threads_x, threads_y = 32, min(32, num_joints)
            
            # Ensure we don't exceed max threads per block
            while threads_x * threads_y > max_threads:
                if threads_x > threads_y:
                    threads_x //= 2
                else:
                    threads_y //= 2
                    
                if threads_x < 32:
                    threads_x = 32
                    threads_y = max_threads // threads_x
                    break
            
            blocks_x = (N + threads_x - 1) // threads_x
            blocks_y = (num_joints + threads_y - 1) // threads_y
            
            kernel_func = trajectory_kernel
        
        # Calculate occupancy metrics
        total_blocks = blocks_x * blocks_y
        theoretical_occupancy = max(100, (total_blocks / sm_count) * 100)
        
        return {
            "kernel_func": kernel_func,
            "kernel_type": kernel_type,
            "grid": (blocks_x, blocks_y),
            "block": (threads_x, threads_y),
            "total_blocks": total_blocks,
            "threads_per_block": threads_x * threads_y,
            "theoretical_occupancy": theoretical_occupancy,
            "estimated_registers": 32,  # Estimated register usage
            "shared_memory_bytes": 256,  # Estimated shared memory usage
        }

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
        torque_limits
    ):
        """Optimized CUDA kernel for computing inverse dynamics using 2D parallelization."""
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
        M_contrib = Glist[j_idx, j_idx, j_idx] if j_idx < Glist.shape[0] and j_idx < Glist.shape[1] and j_idx < Glist.shape[2] else 1.0

        # Velocity quadratic forces (simplified)
        c_j = Slist[j_idx, j_idx] * dtheta_j if j_idx < Slist.shape[0] and j_idx < Slist.shape[1] else 0.0

        # Gravity forces (simplified)
        g_j = gravity_vector[2] * 0.1 if gravity_vector.shape[0] > 2 else 0.0

        # Torque computation
        tau = M_contrib * ddtheta_j + c_j + g_j

        # Enforce torque limits and store result
        if j_idx < torque_limits.shape[0]:
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
        joint_limits
    ):
        """Compute forward dynamics for a robotic system using a CUDA kernel."""
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
            M_inv = 1.0 / Glist[j_idx, j_idx, j_idx] if (j_idx < Glist.shape[0] and 
                                                         j_idx < Glist.shape[1] and 
                                                         j_idx < Glist.shape[2] and 
                                                         Glist[j_idx, j_idx, j_idx] != 0.0) else 1.0
            g_force = g[2] * 0.1 if g.shape[0] > 2 else 0.0
            ddtheta = (tau - g_force) * M_inv

            # Integrate velocities and positions
            current_dtheta += ddtheta * dt_step
            current_theta += current_dtheta * dt_step

            # Enforce joint limits
            if j_idx < joint_limits.shape[0]:
                current_theta = max(joint_limits[j_idx, 0], min(current_theta, joint_limits[j_idx, 1]))

        # Store results
        thetamat[t_idx, j_idx] = current_theta
        dthetamat[t_idx, j_idx] = current_dtheta
        ddthetamat[t_idx, j_idx] = ddtheta

    @cuda.jit(**jit_kwargs)
    def cartesian_trajectory_kernel(pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method):
        """CUDA kernel for generating Cartesian trajectory with time-scaling."""
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
    def fused_potential_gradient_kernel(positions, goal, obstacles, potential, gradient, influence_distance):
        """CUDA kernel for computing potential and gradient for path planning."""
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
        Tf, N, method, batch_size
    ):
        """Optimized CUDA kernel for batch trajectory generation with time-scaling."""
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
        """A memory pool for managing CUDA device arrays to improve memory allocation efficiency."""
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
            """Return a GPU array to the memory pool for potential future reuse."""
            key = (array.shape, array.dtype)
            if key not in self.pool:
                self.pool[key] = []
            
            if len(self.pool[key]) < self.max_pool_size:
                self.pool[key].append(array)
        
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
        """Auto-tune 2D CUDA kernel launch configuration for optimal performance."""
        if not CUDA_AVAILABLE:
            return ((1, 1), (1, 1))
        
        best, t_min = None, 1e9
        test_configs = [(64, 8), (32, 8), (16, 16), (128, 4), (32, 16)]
        
        # Create small test data
        test_N = min(N, 100)
        test_J = min(J, 8)
        
        try:
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
        except Exception:
            return make_2d_grid(N, J)

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
            """Retrieve a constant memory array by its name."""
            return _constant_arrays.get(name)
            
    except AttributeError:
        # Fallback if const arrays not available
        def setup_constant_array(name, data):
            return cuda.to_device(data)
        
        def get_constant_array(name):
            return None

else:
    # Mock functions for when CUDA is not available
    class _MockMemoryPool:
        def get_array(self, *args, **kwargs):
            raise RuntimeError("CUDA memory pool not available.")
        def return_array(self, *args, **kwargs):
            raise RuntimeError("CUDA memory pool not available.")
        def clear(self):
            pass

    _cuda_memory_pool = _MockMemoryPool()
    
    def trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA trajectory kernel not available.")
    
    def inverse_dynamics_kernel(*args, **kwargs):
        raise RuntimeError("CUDA inverse dynamics kernel not available.")
    
    def forward_dynamics_kernel(*args, **kwargs):
        raise RuntimeError("CUDA forward dynamics kernel not available.")
    
    def cartesian_trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA Cartesian trajectory kernel not available.")
    
    def fused_potential_gradient_kernel(*args, **kwargs):
        raise RuntimeError("CUDA potential field kernel not available.")
    
    def batch_trajectory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA batch trajectory kernel not available.")
    
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

    # Legacy kernel placeholders for compatibility
    def attractive_potential_kernel(*args, **kwargs):
        raise RuntimeError("CUDA attractive potential kernel not available.")
    
    def repulsive_potential_kernel(*args, **kwargs):
        raise RuntimeError("CUDA repulsive potential kernel not available.")
    
    def gradient_kernel(*args, **kwargs):
        raise RuntimeError("CUDA gradient kernel not available.")

# Legacy kernel functions that need to be available at module level
def attractive_potential_kernel(*args, **kwargs):
    if CUDA_AVAILABLE:
        raise RuntimeError("Legacy attractive_potential_kernel - use fused_potential_gradient_kernel instead")
    else:
        raise RuntimeError("CUDA attractive potential kernel not available.")

def repulsive_potential_kernel(*args, **kwargs):
    if CUDA_AVAILABLE:
        raise RuntimeError("Legacy repulsive_potential_kernel - use fused_potential_gradient_kernel instead")
    else:
        raise RuntimeError("CUDA repulsive potential kernel not available.")

def gradient_kernel(*args, **kwargs):
    if CUDA_AVAILABLE:
        raise RuntimeError("Legacy gradient_kernel - use fused_potential_gradient_kernel instead")
    else:
        raise RuntimeError("CUDA gradient kernel not available.")

# High-level wrapper functions for optimized CUDA operations
def optimized_trajectory_generation(thetastart, thetaend, Tf, N, method, use_pinned=True, kernel_type="auto"):
    """
    Generate optimized trajectory using advanced CUDA acceleration with multiple kernel options.
    
    Args:
        thetastart, thetaend: Start and end joint angles
        Tf: Final time
        N: Number of trajectory points  
        method: Time scaling method (3=cubic, 5=quintic)
        use_pinned: Use pinned memory for faster transfers
        kernel_type: "auto", "standard", "vectorized", or "memory_optimized"
    """
    if not CUDA_AVAILABLE:
        return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
    
    num_joints = len(thetastart)
    
    try:
        # Convert to proper numpy arrays first
        thetastart_arr = np.ascontiguousarray(thetastart, dtype=np.float32)
        thetaend_arr = np.ascontiguousarray(thetaend, dtype=np.float32)
        
        # Get optimal kernel configuration
        kernel_config = _get_optimal_kernel_config_func(N, num_joints, kernel_type)
        if not kernel_config:
            raise RuntimeError("Failed to get optimal kernel configuration")
        
        print(f"🚀 Using {kernel_config['kernel_type']} kernel with {kernel_config['theoretical_occupancy']:.1f}% occupancy")
        
        # Use pinned memory for faster transfers
        if use_pinned:
            d_thetastart = _h2d_pinned(thetastart_arr)
            d_thetaend = _h2d_pinned(thetaend_arr)
        else:
            d_thetastart = cuda.to_device(thetastart_arr)
            d_thetaend = cuda.to_device(thetaend_arr)
        
        # Allocate output arrays with proper dtype
        d_traj_pos = get_cuda_array((N, num_joints), dtype=np.float32)
        d_traj_vel = get_cuda_array((N, num_joints), dtype=np.float32)
        d_traj_acc = get_cuda_array((N, num_joints), dtype=np.float32)
        
        try:
            # Launch optimized kernel
            kernel_func = kernel_config["kernel_func"]
            grid = kernel_config["grid"]
            block = kernel_config["block"]
            
            # Warm up kernel with smaller problem first (improves performance measurement)
            if N > 1000:
                warm_N = min(100, N)
                warm_pos = get_cuda_array((warm_N, num_joints), dtype=np.float32)
                warm_vel = get_cuda_array((warm_N, num_joints), dtype=np.float32) 
                warm_acc = get_cuda_array((warm_N, num_joints), dtype=np.float32)
                
                warm_grid = ((warm_N + block[0] - 1) // block[0], (num_joints + block[1] - 1) // block[1])
                kernel_func[warm_grid, block](
                    d_thetastart, d_thetaend, warm_pos, warm_vel, warm_acc, Tf, warm_N, method
                )
                cuda.synchronize()
                
                # Return warmup arrays to pool
                return_cuda_array(warm_pos)
                return_cuda_array(warm_vel)
                return_cuda_array(warm_acc)
            
            # Launch main kernel
            kernel_func[grid, block](
                d_thetastart, d_thetaend, d_traj_pos, d_traj_vel, d_traj_acc,
                Tf, N, method
            )
            
            # Synchronize to ensure completion
            cuda.synchronize()
            
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
            
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"GPU trajectory generation failed: {e}, falling back to CPU")
        return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

# Helper function to make the kernel config selection available
def _get_optimal_kernel_config_func(N, num_joints, kernel_type="auto"):
    """Helper function for kernel configuration selection"""
    total_elements = N * num_joints
    
    if not CUDA_AVAILABLE:
        return None
    
    device = cuda.get_current_device()
    max_threads = device.MAX_THREADS_PER_BLOCK
    sm_count = device.MULTIPROCESSOR_COUNT
    
    # Automatically select kernel based on problem characteristics
    if kernel_type == "auto":
        if total_elements < 10000:
            kernel_type = "standard"
        elif total_elements < 100000:
            kernel_type = "vectorized"
        else:
            kernel_type = "memory_optimized"
    
    if kernel_type == "vectorized":
        # For vectorized kernel, adjust N for vector processing
        vector_size = 4
        effective_N = (N + vector_size - 1) // vector_size
        
        # Optimize for vectorized access patterns
        threads_x = min(256, max(32, effective_N))
        threads_y = min(max_threads // threads_x, max(1, num_joints))
        
        blocks_x = (effective_N + threads_x - 1) // threads_x
        blocks_y = (num_joints + threads_y - 1) // threads_y
        
        kernel_func = trajectory_kernel_vectorized
        
    elif kernel_type == "memory_optimized":
        # For memory-optimized kernel, use larger blocks
        threads_x = min(128, max(32, N // sm_count))
        threads_y = min(max_threads // threads_x, max(1, min(8, num_joints)))
        
        blocks_x = min(sm_count * 2, (N + threads_x - 1) // threads_x)
        blocks_y = min(sm_count * 2, (num_joints + threads_y - 1) // threads_y)
        
        kernel_func = trajectory_kernel_memory_optimized
        
    else:  # standard
        # Standard optimized configuration
        if num_joints <= 8:
            threads_x, threads_y = 128, min(8, num_joints)
        elif num_joints <= 16:
            threads_x, threads_y = 64, min(16, num_joints)
        else:
            threads_x, threads_y = 32, min(32, num_joints)
        
        # Ensure we don't exceed max threads per block
        while threads_x * threads_y > max_threads:
            if threads_x > threads_y:
                threads_x //= 2
            else:
                threads_y //= 2
            
            if threads_x < 32:
                threads_x = 32
                threads_y = max_threads // threads_x
                break
        
        blocks_x = (N + threads_x - 1) // threads_x
        blocks_y = (num_joints + threads_y - 1) // threads_y
        
        kernel_func = trajectory_kernel
    
    # Calculate occupancy metrics
    total_blocks = blocks_x * blocks_y
    theoretical_occupancy = min(100, (total_blocks / sm_count) * 100)
    
    return {
        "kernel_func": kernel_func,
        "kernel_type": kernel_type,
        "grid": (blocks_x, blocks_y),
        "block": (threads_x, threads_y),
        "total_blocks": total_blocks,
        "threads_per_block": threads_x * threads_y,
        "theoretical_occupancy": theoretical_occupancy,
        "estimated_registers": 32,  # Estimated register usage
        "shared_memory_bytes": 256,  # Estimated shared memory usage
    }
def optimized_potential_field(positions, goal, obstacles, influence_distance, use_pinned=True):
    """Compute potential field and gradient for a set of positions using a CUDA-accelerated kernel."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for potential field computation")
    
    N = positions.shape[0]
    
    # Use pinned memory for faster transfers - always use float32
    if use_pinned:
        d_positions = _h2d_pinned(np.ascontiguousarray(positions, dtype=np.float32))
        d_goal = _h2d_pinned(np.ascontiguousarray(goal, dtype=np.float32))
        d_obstacles = _h2d_pinned(np.ascontiguousarray(obstacles, dtype=np.float32))
    else:
        d_positions = cuda.to_device(np.ascontiguousarray(positions, dtype=np.float32))
        d_goal = cuda.to_device(np.ascontiguousarray(goal, dtype=np.float32))
        d_obstacles = cuda.to_device(np.ascontiguousarray(obstacles, dtype=np.float32))
    
    # Allocate output arrays
    d_potential = get_cuda_array((N,), dtype=np.float32)
    d_gradient = get_cuda_array((N, 3), dtype=np.float32)
    
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
    """Efficiently generate batch trajectories using CUDA acceleration."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for batch trajectory generation")
    
    batch_size, num_joints = thetastart_batch.shape
    
    # Use pinned memory for faster transfers - always use float32
    if use_pinned:
        d_thetastart_batch = _h2d_pinned(np.ascontiguousarray(thetastart_batch, dtype=np.float32))
        d_thetaend_batch = _h2d_pinned(np.ascontiguousarray(thetaend_batch, dtype=np.float32))
    else:
        d_thetastart_batch = cuda.to_device(np.ascontiguousarray(thetastart_batch, dtype=np.float32))
        d_thetaend_batch = cuda.to_device(np.ascontiguousarray(thetaend_batch, dtype=np.float32))
    
    # Allocate output arrays
    d_traj_pos_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_vel_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_acc_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    
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

# Legacy kernel compatibility functions - these need to be at module level
# These functions are deprecated and should use the new fused kernels instead
def attractive_potential_kernel(*args, **kwargs):
    """Legacy function - use fused_potential_gradient_kernel instead."""
    if CUDA_AVAILABLE:
        raise RuntimeError(
            "Legacy attractive_potential_kernel is deprecated.\n"
            "Use fused_potential_gradient_kernel for better performance."
        )
    else:
        raise RuntimeError("CUDA not available for potential kernel operations.")

def repulsive_potential_kernel(*args, **kwargs):
    """Legacy function - use fused_potential_gradient_kernel instead."""
    if CUDA_AVAILABLE:
        raise RuntimeError(
            "Legacy repulsive_potential_kernel is deprecated.\n"
            "Use fused_potential_gradient_kernel for better performance."
        )
    else:
        raise RuntimeError("CUDA not available for potential kernel operations.")

def gradient_kernel(*args, **kwargs):
    """Legacy function - use fused_potential_gradient_kernel instead."""
    if CUDA_AVAILABLE:
        raise RuntimeError(
            "Legacy gradient_kernel is deprecated.\n"
            "Use fused_potential_gradient_kernel for better performance."
        )
    else:
        raise RuntimeError("CUDA not available for gradient kernel operations.")

# Utility functions for performance monitoring
def benchmark_kernel_performance(kernel_name, *args, num_runs=10):
    """Benchmark the performance of a specific CUDA kernel by executing it multiple times."""
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
    'trajectory_kernel_vectorized',
    'trajectory_kernel_memory_optimized', 
    'trajectory_kernel_warp_optimized',
    'trajectory_kernel_cache_friendly',
    'inverse_dynamics_kernel',
    'forward_dynamics_kernel',
    'cartesian_trajectory_kernel',
    'fused_potential_gradient_kernel',
    'batch_trajectory_kernel',
    'optimized_trajectory_generation',
    'optimized_trajectory_generation_monitored',
    'auto_select_optimal_kernel',
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
    'CUDAPerformanceMonitor',
    # Legacy exports for compatibility
    'attractive_potential_kernel',
    'repulsive_potential_kernel', 
    'gradient_kernel',
]
# Export important functions and classes
__all__ = [
    'CUDA_AVAILABLE',
    'CUPY_AVAILABLE', 
    'check_cuda_availability',
    'check_cupy_availability',
    'trajectory_kernel',
    'trajectory_kernel_vectorized',
    'trajectory_kernel_memory_optimized', 
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
    '_get_optimal_kernel_config_func',
    # Legacy exports for compatibility
    'attractive_potential_kernel',
    'repulsive_potential_kernel', 
    'gradient_kernel',
]