#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
CUDA Kernels Module - ManipulaPy

This module provides CUDA-accelerated functions for trajectory planning and dynamics
computation. All CUDA functionality is optional and gracefully degrades to CPU implementations.

Copyright (c) 2025 Mohamed Aboelnar
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
import warnings

# Optional CUDA imports with graceful fallback
try:
    from numba import cuda, float32
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Create mock objects to prevent import errors
    class MockCuda:
        @staticmethod
        def jit(func):
            """Mock decorator for when CUDA is not available"""
            def wrapper(*args, **kwargs):
                raise RuntimeError(
                    "CUDA functionality not available. Install CUDA support:\n"
                    "pip install ManipulaPy[gpu-cuda11]  # For CUDA 11.x\n"
                    "pip install ManipulaPy[gpu-cuda12]  # For CUDA 12.x\n"
                    "Ensure you have NVIDIA GPU drivers and CUDA toolkit installed."
                )
            return wrapper
        
        @staticmethod
        def grid(dim):
            """Mock grid function"""
            return 0
        
        @staticmethod
        def device_array(*args, **kwargs):
            """Mock device array function"""
            raise RuntimeError("CUDA not available - cannot create device arrays")
        
        @staticmethod
        def to_device(*args, **kwargs):
            """Mock to_device function"""
            raise RuntimeError("CUDA not available - cannot transfer to device")
        
        @staticmethod
        def local():
            """Mock local memory"""
            class LocalMock:
                @staticmethod
                def array(*args, **kwargs):
                    raise RuntimeError("CUDA not available - cannot create local arrays")
            return LocalMock()
    
    cuda = MockCuda()
    float32 = np.float32

# Check for CuPy availability (separate from numba.cuda)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def check_cuda_availability():
    """
    Check if CUDA is available and provide helpful error messages.
    
    Returns
    -------
    bool
        True if CUDA is available, False otherwise
    """
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
    """
    Check if CuPy is available for GPU array operations.
    
    Returns
    -------
    bool
        True if CuPy is available, False otherwise
    """
    if not CUPY_AVAILABLE:
        warnings.warn(
            "CuPy not available. Some GPU array operations will not work.\n"
            "Install with: pip install ManipulaPy[gpu-cuda11]",
            UserWarning,
            stacklevel=2
        )
    return CUPY_AVAILABLE


# CPU fallback functions for when CUDA is not available
def trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method):
    """
    CPU fallback for trajectory generation when CUDA is not available.
    
    Parameters
    ----------
    thetastart : np.ndarray
        Starting joint angles
    thetaend : np.ndarray
        Ending joint angles  
    Tf : float
        Final time
    N : int
        Number of trajectory points
    method : int
        Time scaling method (3=cubic, 5=quintic)
        
    Returns
    -------
    tuple
        (positions, velocities, accelerations) arrays
    """
    num_joints = len(thetastart)
    traj_pos = np.zeros((N, num_joints), dtype=np.float32)
    traj_vel = np.zeros((N, num_joints), dtype=np.float32)
    traj_acc = np.zeros((N, num_joints), dtype=np.float32)
    
    for idx in range(N):
        t = idx * (Tf / (N - 1))
        
        if method == 3:  # Cubic time scaling
            s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
            s_dot = 6 * (t / Tf) * (1 - t / Tf) / Tf
            s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
        elif method == 5:  # Quintic time scaling
            s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
            s_dot = (30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + (t / Tf) ** 2)) / Tf
            s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
        else:
            s = s_dot = s_ddot = 0

        for j in range(num_joints):
            traj_pos[idx, j] = s * (thetaend[j] - thetastart[j]) + thetastart[j]
            traj_vel[idx, j] = s_dot * (thetaend[j] - thetastart[j])
            traj_acc[idx, j] = s_ddot * (thetaend[j] - thetastart[j])
    
    return traj_pos, traj_vel, traj_acc


# CUDA kernel definitions (only compiled if CUDA is available)
if CUDA_AVAILABLE:
    @cuda.jit
    def trajectory_kernel(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ):
        """
        CUDA kernel to compute positions, velocities, and accelerations using cubic or quintic time scaling.
        """
        idx = cuda.grid(1)
        if idx < N:
            t = idx * (Tf / (N - 1))
            if method == 3:  # Cubic time scaling
                s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
                s_dot = 6 * (t / Tf) * (1 - t / Tf) / Tf
                s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
            elif method == 5:  # Quintic time scaling
                s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
                s_dot = (30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + (t / Tf) ** 2)) / Tf
                s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
            else:
                s = s_dot = s_ddot = 0

            for j in range(thetastart.shape[0]):
                traj_pos[idx, j] = s * (thetaend[j] - thetastart[j]) + thetastart[j]
                traj_vel[idx, j] = s_dot * (thetaend[j] - thetastart[j])
                traj_acc[idx, j] = s_ddot * (thetaend[j] - thetastart[j])

    @cuda.jit
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
    ):
        """
        Computes the inverse dynamics of a robot manipulator given the joint angle, velocity, and acceleration trajectories,
        as well as the external forces acting on the end-effector.
        """
        idx = cuda.grid(1)
        if idx < thetalist_trajectory.shape[0]:
            thetalist = thetalist_trajectory[idx]
            dthetalist = dthetalist_trajectory[idx]
            ddthetalist = ddthetalist_trajectory[idx]

            # Mass matrix computation
            M_temp = cuda.local.array((6, 6), dtype=float32)
            for i in range(len(thetalist)):
                for row in range(6):
                    for col in range(6):
                        M_temp[row, col] += Glist[
                            i, row, col
                        ]  # Simplified for demonstration

            # Velocity quadratic forces computation
            c_temp = cuda.local.array(6, dtype=float32)
            for i in range(len(thetalist)):
                for j in range(6):
                    c_temp[j] += Slist[i, j] * dthetalist[i]  # Simplified for demonstration

            # Gravity forces computation
            g_temp = cuda.local.array(6, dtype=float32)
            for i in range(len(thetalist)):
                g_temp[2] += gravity_vector[i]  # Simplified for demonstration

            # External forces (Ftip)
            F_ext = cuda.local.array(6, dtype=float32)
            for i in range(len(Ftip)):
                F_ext[i] += Ftip[i]

            # Torque computation
            tau_temp = cuda.local.array(6, dtype=float32)
            for row in range(6):
                for col in range(6):
                    tau_temp[row] += M_temp[row, col] * ddthetalist[col]
                tau_temp[row] += c_temp[row] + g_temp[row] + F_ext[row]
            for j in range(len(tau_temp)):
                # Enforce torque limits
                tau_temp[j] = max(
                    torque_limits[j, 0], min(tau_temp[j], torque_limits[j, 1])
                )
                torques_trajectory[idx, j] = tau_temp[j]

    @cuda.jit
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
    ):
        """
        CUDA kernel to compute forward dynamics for a robotic system.
        """
        idx = cuda.grid(1)
        if idx < taumat.shape[0]:
            # Initialize local variables
            current_thetalist = thetamat[idx - 1, :] if idx > 0 else thetalist
            current_dthetalist = dthetamat[idx - 1, :] if idx > 0 else dthetalist
            current_tau = taumat[idx, :]
            current_Ftip = Ftipmat[idx, :]

            # Placeholder for the mass matrix and other dynamics quantities
            M_temp = cuda.local.array((6, 6), dtype=float32)
            c_temp = cuda.local.array((6,), dtype=float32)
            g_temp = cuda.local.array((6,), dtype=float32)
            ddthetalist_local = cuda.local.array((6,), dtype=float32)

            for _ in range(intRes):
                # Compute forward dynamics (simplified for demonstration)
                for i in range(len(thetalist)):
                    for row in range(6):
                        for col in range(6):
                            M_temp[row, col] = Glist[i, row, col]  # Simplified
                        c_temp[row] = Slist[i, row] * current_dthetalist[i]  # Simplified
                        g_temp[row] = g[row]  # Simplified

                # Compute joint accelerations
                for i in range(len(thetalist)):
                    ddthetalist_local[i] = (
                        current_tau[i] - c_temp[i] - g_temp[i]
                    ) / M_temp[
                        i, i
                    ]  # Simplified

                # Integrate to get velocities and positions
                for i in range(len(thetalist)):
                    current_dthetalist[i] += ddthetalist_local[i] * (dt / intRes)
                    current_thetalist[i] += current_dthetalist[i] * (dt / intRes)

                # Enforce joint limits
                for i in range(len(thetalist)):
                    current_thetalist[i] = max(
                        joint_limits[i, 0], min(current_thetalist[i], joint_limits[i, 1])
                    )

            # Store results
            for i in range(len(thetalist)):
                thetamat[idx, i] = current_thetalist[i]
                dthetamat[idx, i] = current_dthetalist[i]
                ddthetamat[idx, i] = ddthetalist_local[i]

    @cuda.jit
    def cartesian_trajectory_kernel(
        pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ):
        """
        CUDA kernel to compute Cartesian trajectory positions, velocities, and accelerations.
        """
        idx = cuda.grid(1)
        if idx < N:
            t = idx * (Tf / (N - 1))
            if method == 3:
                s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
                s_dot = 6 * (t / Tf) * (1 - t / Tf) / Tf
                s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
            elif method == 5:
                s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
                s_dot = (30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + (t / Tf) ** 2)) / Tf
                s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
            else:
                s = s_dot = s_ddot = 0

            for j in range(3):  # For x, y, z positions
                traj_pos[idx, j] = s * (pend[j] - pstart[j]) + pstart[j]
                traj_vel[idx, j] = s_dot * (pend[j] - pstart[j])
                traj_acc[idx, j] = s_ddot * (pend[j] - pstart[j])

    @cuda.jit
    def attractive_potential_kernel(positions, goal, potential):
        """
        CUDA kernel to compute attractive potential field.
        """
        idx = cuda.grid(1)
        if idx < positions.shape[0]:
            for i in range(3):
                potential[idx] += 0.5 * (positions[idx, i] - goal[i]) ** 2

    @cuda.jit
    def repulsive_potential_kernel(positions, obstacles, potential, influence_distance):
        """
        CUDA kernel to compute repulsive potential field.
        """
        idx = cuda.grid(1)
        if idx < positions.shape[0]:
            for obs in range(obstacles.shape[0]):
                dist_sq = 0.0
                for i in range(3):
                    diff = positions[idx, i] - obstacles[obs, i]
                    dist_sq += diff * diff
                dist = dist_sq ** 0.5  # Use power instead of np.sqrt for CUDA compatibility
                if dist < influence_distance and dist > 1e-10:  # Avoid division by zero
                    repulsive_term = (1.0 / dist - 1.0 / influence_distance) ** 2
                    potential[idx] += 0.5 * repulsive_term

    @cuda.jit
    def gradient_kernel(potential, gradient):
        """
        CUDA kernel to compute the gradient of the potential field.
        """
        idx = cuda.grid(1)
        if idx < potential.shape[0] - 1:
            gradient[idx] = potential[idx + 1] - potential[idx]

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

