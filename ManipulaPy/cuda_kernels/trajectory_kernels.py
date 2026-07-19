#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA kernel implementation split by runtime concern."""

from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, NoReturn, Optional, Tuple

import numpy as np

from . import _runtime
from ._runtime import CUDA_AVAILABLE, FAST_MATH, cuda, float32, logger
from .memory import _h2d_pinned, get_cuda_array, return_cuda_array

get_optimal_kernel_config: Any = None
_perf_monitor: Any = None
_cuda_routing_enabled: Any = None


def trajectory_cpu_fallback(
    thetastart: np.ndarray,
    thetaend: np.ndarray,
    Tf: float,
    N: int,
    method: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized CPU fallback using NumPy vectorization.

    Args:
        thetastart: (num_joints,) ndarray of starting joint angles, radians.
        thetaend: (num_joints,) ndarray of ending joint angles, radians.
        Tf: Total trajectory duration, seconds. Values <= 0 collapse to the
            start configuration with zero velocity and acceleration.
        N: Number of trajectory time steps. Values <= 1 collapse to the start
            configuration.
        method: Time-scaling polynomial order: 3 for cubic, 5 for quintic, any
            other value (e.g. 1) for linear.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
        traj_acc)``, each an ``(N, num_joints)`` float32 ndarray of joint
        positions (radians), velocities (radians/s), and accelerations
        (radians/s^2).
    """
    num_joints = len(thetastart)

    # Degenerate inputs collapse to "sit at start": s = s_dot = s_ddot = 0.
    # Matches the GPU kernels' N<=1 guard and avoids the divide-by-zero
    # RuntimeWarning when callers pass Tf=0 (regression coverage in
    # test_zero_time_trajectory accepts NaN but the warning is noisy).
    if N <= 1 or Tf <= 0.0:
        s = np.zeros(N, dtype=np.float32)
        s_dot = np.zeros(N, dtype=np.float32)
        s_ddot = np.zeros(N, dtype=np.float32)
    else:
        # Vectorized time computation
        t = np.linspace(0, Tf, N, dtype=np.float32)
        tau = t / Tf

        # Vectorized time scaling
        if method == 3:  # Cubic
            s = 3.0 * tau**2 - 2.0 * tau**3
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:  # Quintic
            tau2 = tau**2
            tau3 = tau**3
            tau4 = tau**4
            tau5 = tau**5
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
            s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = np.ones_like(tau) / Tf
            s_ddot = np.zeros_like(tau)

    # Vectorized trajectory computation
    delta = thetaend - thetastart
    traj_pos = thetastart[np.newaxis, :] + s[:, np.newaxis] * delta[np.newaxis, :]
    traj_vel = s_dot[:, np.newaxis] * delta[np.newaxis, :]
    traj_acc = s_ddot[:, np.newaxis] * delta[np.newaxis, :]

    return (
        traj_pos.astype(np.float32),
        traj_vel.astype(np.float32),
        traj_acc.astype(np.float32),
    )


if CUDA_AVAILABLE:
    jit_kwargs = {"fastmath": FAST_MATH}
    _runtime.jit_kwargs = jit_kwargs

    @cuda.jit(device=True, inline=True, **jit_kwargs)
    def matrix_vector_multiply_6x6(M, v, result) -> None:
        """Optimized 6x6 matrix-vector multiplication using registers.

        Args:
            M: (6, 6) device array, the matrix operand.
            v: (6,) device array, the vector operand.
            result: (6,) device array, in-place output buffer set to ``M @ v``.
        """
        # Unrolled for maximum performance
        result[0] = (
            M[0, 0] * v[0]
            + M[0, 1] * v[1]
            + M[0, 2] * v[2]
            + M[0, 3] * v[3]
            + M[0, 4] * v[4]
            + M[0, 5] * v[5]
        )
        result[1] = (
            M[1, 0] * v[0]
            + M[1, 1] * v[1]
            + M[1, 2] * v[2]
            + M[1, 3] * v[3]
            + M[1, 4] * v[4]
            + M[1, 5] * v[5]
        )
        result[2] = (
            M[2, 0] * v[0]
            + M[2, 1] * v[1]
            + M[2, 2] * v[2]
            + M[2, 3] * v[3]
            + M[2, 4] * v[4]
            + M[2, 5] * v[5]
        )
        result[3] = (
            M[3, 0] * v[0]
            + M[3, 1] * v[1]
            + M[3, 2] * v[2]
            + M[3, 3] * v[3]
            + M[3, 4] * v[4]
            + M[3, 5] * v[5]
        )
        result[4] = (
            M[4, 0] * v[0]
            + M[4, 1] * v[1]
            + M[4, 2] * v[2]
            + M[4, 3] * v[3]
            + M[4, 4] * v[4]
            + M[4, 5] * v[5]
        )
        result[5] = (
            M[5, 0] * v[0]
            + M[5, 1] * v[1]
            + M[5, 2] * v[2]
            + M[5, 3] * v[3]
            + M[5, 4] * v[4]
            + M[5, 5] * v[5]
        )

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """Each thread computes its own time scaling — no shared memory race.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= N or j_idx >= thetastart.shape[0]:
            return

        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

        if method == 3:  # Cubic
            tau2 = tau * tau
            tau3 = tau2 * tau
            s = 3.0 * tau2 - 2.0 * tau3
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:  # Quintic
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / Tf
            s_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (Tf * Tf)
        else:  # Linear
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        start_angle = thetastart[j_idx]
        delta_angle = thetaend[j_idx] - start_angle

        traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
        traj_vel[t_idx, j_idx] = s_dot * delta_angle
        traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_vectorized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Vectorized trajectory kernel with correct 8-parameter signature.
        Each thread processes multiple time steps for better throughput.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        VECTOR_SIZE = 8  # Each thread processes 8 time steps

        t_base = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) * VECTOR_SIZE
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if j_idx >= thetastart.shape[0]:
            return

        # Shared memory for joint data
        shared_joints = cuda.shared.array((32, 2), dtype=float32)

        # Load joint data to shared memory
        if cuda.threadIdx.x == 0 and j_idx < 32:
            start_val = thetastart[j_idx]
            shared_joints[j_idx, 0] = start_val
            shared_joints[j_idx, 1] = thetaend[j_idx] - start_val

        cuda.syncthreads()

        # Get joint data
        if j_idx < 32:
            start_angle = shared_joints[j_idx, 0]
            delta_angle = shared_joints[j_idx, 1]
        else:
            start_angle = thetastart[j_idx]
            delta_angle = thetaend[j_idx] - start_angle

        # Process VECTOR_SIZE time steps
        for i in range(VECTOR_SIZE):
            t_idx = t_base + i
            if t_idx >= N:
                break

            # Compute time scaling
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic - optimized computation
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            # Store results
            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_memory_optimized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Memory-bandwidth optimized kernel with correct 8-parameter signature.
        Uses grid-stride loops for better memory utilization.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        # Grid-stride loop for better memory utilization
        stride_t = cuda.gridDim.x * cuda.blockDim.x
        stride_j = cuda.gridDim.y * cuda.blockDim.y

        t_start = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_start = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        # Shared memory for coefficients and joint data
        shared_data = cuda.shared.array(
            (32, 4), dtype=float32
        )  # [start, delta, s, s_dot]

        # Process joints in chunks
        for j_idx in range(j_start, thetastart.shape[0], stride_j):
            # Load joint data to shared memory
            local_j = cuda.threadIdx.y
            if local_j < 32 and j_idx < thetastart.shape[0]:
                start_val = thetastart[j_idx]
                shared_data[local_j, 0] = start_val
                shared_data[local_j, 1] = thetaend[j_idx] - start_val

            cuda.syncthreads()

            # Process time steps with grid-stride
            for t_idx in range(t_start, N, stride_t):
                # Compute time scaling
                tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

                if method == 5:  # Quintic
                    tau_sq = tau * tau
                    tau_cb = tau_sq * tau
                    s = tau_cb * (10.0 + tau * (-15.0 + 6.0 * tau))
                    s_dot = tau_sq * (30.0 + tau * (-60.0 + 30.0 * tau)) / Tf
                    s_ddot = tau * (60.0 + tau * (-180.0 + 120.0 * tau)) / (Tf * Tf)
                elif method == 3:  # Cubic
                    tau_sq = tau * tau
                    s = tau_sq * (3.0 - 2.0 * tau)
                    s_dot = 6.0 * tau * (1.0 - tau) / Tf
                    s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
                else:  # Linear
                    s = tau
                    s_dot = 1.0 / Tf
                    s_ddot = 0.0

                # Use shared memory data if available
                if local_j < 32 and j_idx < thetastart.shape[0]:
                    start_angle = shared_data[local_j, 0]
                    delta_angle = shared_data[local_j, 1]
                else:
                    start_angle = (
                        thetastart[j_idx] if j_idx < thetastart.shape[0] else 0.0
                    )
                    delta_angle = (
                        (thetaend[j_idx] - start_angle)
                        if j_idx < thetastart.shape[0]
                        else 0.0
                    )

                # Store results
                if j_idx < thetastart.shape[0]:
                    traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
                    traj_vel[t_idx, j_idx] = s_dot * delta_angle
                    traj_acc[t_idx, j_idx] = s_ddot * delta_angle

            cuda.syncthreads()

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_warp_optimized(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Warp-level optimized kernel with correct 8-parameter signature.
        Uses warp-level primitives for maximum throughput.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        # Warp-level indexing
        warp_id = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) // 32
        lane_id = cuda.threadIdx.x % 32

        # Each warp processes 32 consecutive time steps
        t_base = warp_id * 32
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if j_idx >= thetastart.shape[0]:
            return

        # Load joint data (broadcast across warp)
        start_angle = thetastart[j_idx]
        delta_angle = thetaend[j_idx] - start_angle

        # Each thread in warp processes one time step
        t_idx = t_base + lane_id

        if t_idx < N:
            # Optimized time scaling computation
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            # Coalesced memory writes
            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    @cuda.jit(**jit_kwargs)
    def trajectory_kernel_cache_friendly(
        thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """
        FIXED: Cache-friendly kernel with correct 8-parameter signature.
        Uses tiled computation to maximize cache utilization.

        Args:
            thetastart: (num_joints,) device array of starting joint angles, radians.
            thetaend: (num_joints,) device array of ending joint angles, radians.
            traj_pos: (N, num_joints) device array, in-place output buffer for
                joint positions, radians.
            traj_vel: (N, num_joints) device array, in-place output buffer for
                joint velocities, radians/s.
            traj_acc: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        TILE_SIZE_T = 64  # Time tile size
        TILE_SIZE_J = 8  # Joint tile size

        # Block-level tiling
        t_tile_start = cuda.blockIdx.x * TILE_SIZE_T
        j_tile_start = cuda.blockIdx.y * TILE_SIZE_J

        # Thread indices within tile
        t_local = cuda.threadIdx.x
        j_local = cuda.threadIdx.y

        # Global indices
        t_idx = t_tile_start + t_local
        j_idx = j_tile_start + j_local

        # Shared memory for tile data
        shared_joints = cuda.shared.array((TILE_SIZE_J, 2), dtype=float32)
        shared_time = cuda.shared.array((TILE_SIZE_T, 3), dtype=float32)

        # Load joint data to shared memory
        if t_local == 0 and j_idx < thetastart.shape[0]:
            start_val = thetastart[j_idx]
            shared_joints[j_local, 0] = start_val
            shared_joints[j_local, 1] = thetaend[j_idx] - start_val

        # Load time scaling data to shared memory
        if j_local == 0 and t_idx < N:
            tau = 0.0 if N <= 1 else t_idx / (N - 1.0)

            if method == 5:  # Quintic
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = tau3 * (10.0 - 15.0 * tau + 6.0 * tau2)
                s_dot = tau2 * (30.0 - 60.0 * tau + 30.0 * tau2) / Tf
                s_ddot = tau * (60.0 - 180.0 * tau + 120.0 * tau2) / (Tf * Tf)
            elif method == 3:  # Cubic
                tau2 = tau * tau
                s = tau2 * (3.0 - 2.0 * tau)
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
            else:  # Linear
                s = tau
                s_dot = 1.0 / Tf
                s_ddot = 0.0

            shared_time[t_local, 0] = s
            shared_time[t_local, 1] = s_dot
            shared_time[t_local, 2] = s_ddot

        cuda.syncthreads()

        # Compute trajectory using shared memory data
        if t_idx < N and j_idx < thetastart.shape[0]:
            start_angle = shared_joints[j_local, 0]
            delta_angle = shared_joints[j_local, 1]
            s = shared_time[t_local, 0]
            s_dot = shared_time[t_local, 1]
            s_ddot = shared_time[t_local, 2]

            traj_pos[t_idx, j_idx] = start_angle + s * delta_angle
            traj_vel[t_idx, j_idx] = s_dot * delta_angle
            traj_acc[t_idx, j_idx] = s_ddot * delta_angle

    # DYNAMICS KERNELS - FIXED SIGNATURES
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
    ) -> None:
        """
        FIXED: Inverse dynamics kernel with correct 10-parameter signature.
        Removed the problematic 'stream' parameter that was causing the mismatch.

        Uses a simplified per-joint dynamics model (diagonal inertia, linear
        Coriolis term, scalar gravity contribution) rather than full recursive
        Newton-Euler.

        Args:
            thetalist_trajectory: (N, num_joints) device array of joint angles,
                radians.
            dthetalist_trajectory: (N, num_joints) device array of joint
                velocities, radians/s.
            ddthetalist_trajectory: (N, num_joints) device array of joint
                accelerations, radians/s^2.
            gravity_vector: (3,) device array, gravitational acceleration; only
                the z component is used.
            Ftip: External wrench at the tip (unused in this simplified kernel).
            Glist: (num_joints, *, *) device array of spatial inertia matrices;
                its diagonal supplies the effective inertia term.
            Slist: (>=num_joints, >=num_joints) device array of screw axes; its
                diagonal supplies the velocity-coupling term.
            M: Home configuration matrix (unused in this simplified kernel).
            torques_trajectory: (N, num_joints) device array, in-place output
                buffer for computed joint torques, clamped to ``torque_limits``.
            torque_limits: (num_joints, 2) device array of ``[min, max]`` torque
                bounds per joint.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if (
            t_idx >= thetalist_trajectory.shape[0]
            or j_idx >= thetalist_trajectory.shape[1]
        ):
            return

        # Load trajectory data
        theta_j = thetalist_trajectory[t_idx, j_idx]
        dtheta_j = dthetalist_trajectory[t_idx, j_idx]
        ddtheta_j = ddthetalist_trajectory[t_idx, j_idx]

        # Simplified dynamics computation with bounds checking
        M_contrib = (
            Glist[j_idx, j_idx, j_idx]
            if (
                j_idx < Glist.shape[0]
                and j_idx < Glist.shape[1]
                and j_idx < Glist.shape[2]
            )
            else 1.0
        )

        c_j = (
            Slist[j_idx, j_idx] * dtheta_j
            if (j_idx < Slist.shape[0] and j_idx < Slist.shape[1])
            else 0.0
        )

        g_j = gravity_vector[2] * 0.1 if gravity_vector.shape[0] > 2 else 0.0

        # Compute torque
        tau = M_contrib * ddtheta_j + c_j + g_j

        # Apply torque limits
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
        joint_limits,
    ) -> None:
        """Forward dynamics kernel.

        Each thread integrates from the initial state up to its own ``t_idx``,
        avoiding the temporal data race in the previous version (which read
        ``thetamat[t_idx-1]`` while parallel threads at lower ``t_idx`` may
        not have written that row yet). Cost is O(t_idx * intRes) per
        thread instead of O(intRes), but correctness no longer depends on
        warp scheduling.

        Uses a simplified per-joint dynamics model (diagonal inertia, scalar
        gravity) rather than full recursive Newton-Euler.

        Args:
            thetalist: (num_joints,) device array of initial joint angles, radians.
            dthetalist: (num_joints,) device array of initial joint velocities,
                radians/s.
            taumat: (N, num_joints) device array of applied joint torques per
                time step; ``taumat[i]`` advances state into row ``i``.
            g: (3,) device array, gravitational acceleration; only the z
                component is used.
            Ftipmat: External tip wrench per step (unused in this simplified
                kernel).
            dt: Time step between trajectory rows, seconds.
            intRes: Number of Euler sub-integration steps per ``dt``.
            Glist: (num_joints, *, *) device array of spatial inertia matrices;
                its diagonal supplies the inverse-inertia term.
            Slist: Screw axes (unused in this simplified kernel).
            M: Home configuration matrix (unused in this simplified kernel).
            thetamat: (N, num_joints) device array, in-place output buffer for
                integrated joint angles, radians.
            dthetamat: (N, num_joints) device array, in-place output buffer for
                integrated joint velocities, radians/s.
            ddthetamat: (N, num_joints) device array, in-place output buffer for
                joint accelerations, radians/s^2.
            joint_limits: (num_joints, 2) device array of ``[min, max]`` angle
                limits used to clamp integrated positions.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= taumat.shape[0] or j_idx >= thetalist.shape[0]:
            return

        current_theta = thetalist[j_idx]
        current_dtheta = dthetalist[j_idx]
        dt_step = dt / intRes
        ddtheta = 0.0

        if t_idx == 0:
            thetamat[t_idx, j_idx] = current_theta
            dthetamat[t_idx, j_idx] = current_dtheta
            ddthetamat[t_idx, j_idx] = 0.0
            return

        # Walk from t=1 to this thread's t_idx, matching the CPU path's
        # convention that row 0 is the initial state and taumat[i] advances
        # state row i. This remains independent per thread.
        for step in range(1, t_idx + 1):
            tau = taumat[step, j_idx]
            for _ in range(intRes):
                M_inv = (
                    1.0 / Glist[j_idx, j_idx, j_idx]
                    if (
                        j_idx < Glist.shape[0]
                        and j_idx < Glist.shape[1]
                        and j_idx < Glist.shape[2]
                        and Glist[j_idx, j_idx, j_idx] != 0.0
                    )
                    else 1.0
                )
                g_force = g[2] * 0.1 if g.shape[0] > 2 else 0.0
                ddtheta = (tau - g_force) * M_inv

                current_dtheta += ddtheta * dt_step
                current_theta += current_dtheta * dt_step

                if j_idx < joint_limits.shape[0]:
                    current_theta = max(
                        joint_limits[j_idx, 0],
                        min(current_theta, joint_limits[j_idx, 1]),
                    )

        thetamat[t_idx, j_idx] = current_theta
        dthetamat[t_idx, j_idx] = current_dtheta
        ddthetamat[t_idx, j_idx] = ddtheta

    @cuda.jit(**jit_kwargs)
    def cartesian_trajectory_kernel(
        pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method
    ) -> None:
        """Cartesian trajectory kernel.

        Each thread computes its own time scaling (no shared memory) so the
        scaling matches its own ``t_idx``. Quintic acceleration uses the
        full ``60 tau (1 - tau) (1 - 2 tau) / Tf^2`` form, and the linear
        method (1) is no longer silently zeroed.

        Args:
            pstart: (3,) device array, starting Cartesian position.
            pend: (3,) device array, ending Cartesian position.
            traj_pos: (N, 3) device array, in-place output buffer for Cartesian
                positions.
            traj_vel: (N, 3) device array, in-place output buffer for Cartesian
                velocities.
            traj_acc: (N, 3) device array, in-place output buffer for Cartesian
                accelerations.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        """
        t_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        coord_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        if t_idx >= N or coord_idx >= 3:
            return

        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)
        if method == 3:  # Cubic
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
        elif method == 5:  # Quintic
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
            s_ddot = 60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        dp = pend[coord_idx] - pstart[coord_idx]

        traj_pos[t_idx, coord_idx] = s * dp + pstart[coord_idx]
        traj_vel[t_idx, coord_idx] = s_dot * dp
        traj_acc[t_idx, coord_idx] = s_ddot * dp

    from .field_kernels import fused_potential_gradient_kernel

    @cuda.jit(**jit_kwargs)
    def batch_trajectory_kernel(
        thetastart_batch,  # (batch_size, num_joints)
        thetaend_batch,  # (batch_size, num_joints)
        traj_pos_batch,  # (batch_size, N, num_joints)
        traj_vel_batch,  # (batch_size, N, num_joints)
        traj_acc_batch,  # (batch_size, N, num_joints)
        Tf,
        N,
        method,
        batch_size,
    ) -> None:
        """Generate position, velocity, and acceleration for a batch of trajectories.

        Args:
            thetastart_batch: (batch_size, num_joints) device array of starting
                joint angles, radians.
            thetaend_batch: (batch_size, num_joints) device array of ending
                joint angles, radians.
            traj_pos_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint positions, radians.
            traj_vel_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint velocities, radians/s.
            traj_acc_batch: (batch_size, N, num_joints) device array, in-place
                output buffer for joint accelerations, radians/s^2.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
            batch_size: Number of trajectories in the batch.
        """
        # Compute global indices
        batch_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        t_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        j_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        # Bounds check—use shape[1] for num_joints
        if batch_idx >= batch_size or t_idx >= N or j_idx >= thetastart_batch.shape[1]:
            return

        # Per-thread time-scaling computation. Previous version wrote scaling
        # for thread (0,0,0)'s t_idx into shared memory and let every other
        # thread read it — so threads at different t_idx got the wrong scaling.
        tau = 0.0 if N <= 1 else t_idx / (N - 1.0)
        if method == 3:
            s = 3.0 * tau * tau - 2.0 * tau * tau * tau
            s_dot = 6.0 * tau * (1.0 - tau) / Tf
            s_ddot = 6.0 * (1.0 - 2.0 * tau) / (Tf * Tf)
        elif method == 5:
            tau2 = tau * tau
            tau3 = tau2 * tau
            tau4 = tau2 * tau2
            tau5 = tau4 * tau
            s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
            s_dot = 30.0 * tau2 * (1 - 2 * tau + tau2) / Tf
            s_ddot = 60.0 * tau * (1 - tau) * (1 - 2 * tau) / (Tf * Tf)
        else:  # Linear (method == 1) and any other value
            s = tau
            s_dot = 1.0 / Tf
            s_ddot = 0.0

        # Compute delta for this trajectory
        dtheta = thetaend_batch[batch_idx, j_idx] - thetastart_batch[batch_idx, j_idx]

        # Write results
        traj_pos_batch[batch_idx, t_idx, j_idx] = (
            s * dtheta + thetastart_batch[batch_idx, j_idx]
        )
        traj_vel_batch[batch_idx, t_idx, j_idx] = s_dot * dtheta
        traj_acc_batch[batch_idx, t_idx, j_idx] = s_ddot * dtheta

    @lru_cache(maxsize=64)
    def _auto_tune_kernel_config(N: int, num_joints: int) -> Optional[Dict[str, Any]]:
        """Auto-tune kernel configuration for specific problem size.

        Benchmarks each candidate kernel type on small test arrays and returns
        the fastest configuration. Results are memoized via ``lru_cache``.

        Args:
            N: Number of trajectory time steps for the target problem.
            num_joints: Number of joints for the target problem.

        Returns:
            Optional[Dict[str, Any]]: The best-performing kernel configuration
            dict (as returned by ``get_optimal_kernel_config``), or None when
            CUDA is unavailable.
        """
        if not CUDA_AVAILABLE:
            return None

        configs_to_test = [
            ("standard", {}),
            ("vectorized", {}),
            ("memory_optimized", {}),
            ("warp_optimized", {}),
            ("cache_friendly", {}),
        ]

        best_config = None
        best_time = float("inf")

        # Create small test arrays
        test_N = min(N, 1000)
        test_joints = min(num_joints, 8)

        try:
            d_start = cuda.device_array(test_joints, dtype=float32)
            d_end = cuda.device_array(test_joints, dtype=float32)
            d_pos = cuda.device_array((test_N, test_joints), dtype=float32)
            d_vel = cuda.device_array((test_N, test_joints), dtype=float32)
            d_acc = cuda.device_array((test_N, test_joints), dtype=float32)

            for kernel_type, params in configs_to_test:
                try:
                    config = get_optimal_kernel_config(test_N, test_joints, kernel_type)
                    if not config:
                        continue

                    kernel_func = config["kernel_func"]
                    grid = config["grid"]
                    block = config["block"]

                    # Warm-up
                    kernel_func[grid, block](
                        d_start, d_end, d_pos, d_vel, d_acc, 1.0, test_N, 3
                    )
                    cuda.synchronize()

                    # Timed run
                    start_time = perf_counter()
                    kernel_func[grid, block](
                        d_start, d_end, d_pos, d_vel, d_acc, 1.0, test_N, 3
                    )
                    cuda.synchronize()
                    elapsed = perf_counter() - start_time

                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = config

                except Exception:
                    continue

            return best_config or get_optimal_kernel_config(N, num_joints, "standard")

        except Exception:
            return get_optimal_kernel_config(N, num_joints, "standard")

    # HIGH-LEVEL OPTIMIZED FUNCTIONS
    def _optimized_trajectory_generation_monitored_cuda(
        thetastart: Any,
        thetaend: Any,
        Tf: float,
        N: int,
        method: int,
        use_pinned: bool = True,
        kernel_type: str = "auto",
        enable_monitoring: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trajectory with comprehensive performance monitoring.

        This is the main function for achieving 40x+ speedups. Falls back to the
        optimized CPU implementation if CUDA is unavailable or GPU execution
        fails.

        Args:
            thetastart: (num_joints,) array-like of starting joint angles, radians.
            thetaend: (num_joints,) array-like of ending joint angles, radians.
            Tf: Total trajectory duration, seconds.
            N: Number of trajectory time steps.
            method: Time-scaling order: 3 cubic, 5 quintic, else linear.
            use_pinned: If True, use pinned host memory for host/device transfers.
            kernel_type: Kernel selection strategy: "auto", "auto_tune", or an
                explicit kernel name ("standard", "vectorized",
                "memory_optimized", "warp_optimized", "cache_friendly").
            enable_monitoring: If True, log launch configuration and throughput
                and record per-kernel launch statistics.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos, traj_vel,
            traj_acc)``, each an ``(N, num_joints)`` float32 ndarray of joint
            positions (radians), velocities (radians/s), and accelerations
            (radians/s^2).
        """
        if not CUDA_AVAILABLE:
            return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

        num_joints = len(thetastart)
        total_work = N * num_joints

        # Performance recommendations
        if total_work < 50000:
            logger.warning(
                f"Problem size {total_work:,} may not achieve 40x speedup. "
                f"Recommend N ≥ {50000 // num_joints:,} for better GPU utilization."
            )

        try:
            # Convert to optimal data types
            thetastart_arr = np.ascontiguousarray(thetastart, dtype=np.float32)
            thetaend_arr = np.ascontiguousarray(thetaend, dtype=np.float32)

            # Get optimal configuration
            if kernel_type == "auto_tune":
                config = _auto_tune_kernel_config(N, num_joints)
            else:
                config = get_optimal_kernel_config(N, num_joints, kernel_type)

            if not config:
                raise RuntimeError("Failed to get kernel configuration")

            if enable_monitoring:
                logger.info(f"🚀 Using {config['kernel_type']} kernel:")
                logger.info(f"   Grid: {config['grid']}, Block: {config['block']}")
                logger.info(f"   Occupancy: {config['theoretical_occupancy']:.1f}%")
                logger.info(
                    f"   Expected speedup: {config['expected_speedup_range'][0]}-{config['expected_speedup_range'][1]}x"
                )
                if config.get("recommended_for_40x"):
                    logger.info("   ✅ Problem size optimal for 40x+ speedup!")
                else:
                    logger.info("   ⚠️  Consider larger N for maximum speedup")

            # Setup GPU memory with optimal transfers
            if use_pinned:
                d_thetastart = _h2d_pinned(thetastart_arr)
                d_thetaend = _h2d_pinned(thetaend_arr)
            else:
                d_thetastart = cuda.to_device(thetastart_arr)
                d_thetaend = cuda.to_device(thetaend_arr)

            # Allocate output arrays
            d_traj_pos = get_cuda_array((N, num_joints), dtype=np.float32)
            d_traj_vel = get_cuda_array((N, num_joints), dtype=np.float32)
            d_traj_acc = get_cuda_array((N, num_joints), dtype=np.float32)

            try:
                # Extract kernel configuration
                kernel_func = config["kernel_func"]
                grid = config["grid"]
                block = config["block"]

                # Record performance metrics
                if enable_monitoring:
                    _perf_monitor.record_kernel_launch(
                        config["kernel_type"], grid, block
                    )

                # Warm-up for large problems to eliminate JIT overhead
                if total_work > 100000:
                    warm_N = min(1000, N)
                    warm_grid = ((warm_N + block[0] - 1) // block[0], grid[1])
                    kernel_func[warm_grid, block](
                        d_thetastart,
                        d_thetaend,
                        d_traj_pos,
                        d_traj_vel,
                        d_traj_acc,
                        Tf,
                        warm_N,
                        method,
                    )
                    cuda.synchronize()

                # Main kernel launch - FIXED: Using 8 parameters instead of 9
                start_time = perf_counter()
                kernel_func[grid, block](
                    d_thetastart,
                    d_thetaend,
                    d_traj_pos,
                    d_traj_vel,
                    d_traj_acc,
                    Tf,
                    N,
                    method,
                )
                cuda.synchronize()
                gpu_time = perf_counter() - start_time

                # Copy results back with optimal memory transfer
                if use_pinned:
                    # Use pinned host arrays for faster transfer
                    traj_pos_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )
                    traj_vel_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )
                    traj_acc_pinned = cuda.pinned_array(
                        (N, num_joints), dtype=np.float32
                    )

                    d_traj_pos.copy_to_host(traj_pos_pinned)
                    d_traj_vel.copy_to_host(traj_vel_pinned)
                    d_traj_acc.copy_to_host(traj_acc_pinned)

                    # Convert to regular numpy arrays
                    traj_pos = np.array(traj_pos_pinned)
                    traj_vel = np.array(traj_vel_pinned)
                    traj_acc = np.array(traj_acc_pinned)
                else:
                    traj_pos = d_traj_pos.copy_to_host()
                    traj_vel = d_traj_vel.copy_to_host()
                    traj_acc = d_traj_acc.copy_to_host()

                if enable_monitoring:
                    throughput = (
                        total_work / gpu_time / 1e6
                    )  # Million elements per second
                    logger.info(f"⚡ GPU execution: {gpu_time*1000:.2f}ms")
                    logger.info(f"📊 Throughput: {throughput:.1f} M elements/sec")

                return traj_pos, traj_vel, traj_acc

            finally:
                # Always return arrays to pool
                return_cuda_array(d_traj_pos)
                return_cuda_array(d_traj_vel)
                return_cuda_array(d_traj_acc)

        except Exception as e:
            logger.warning(f"GPU trajectory generation failed: {e}")
            logger.info("Falling back to optimized CPU implementation")
            return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)

else:

    def trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA trajectory kernel is unavailable."""
        raise RuntimeError("CUDA trajectory kernel not available")

    def trajectory_kernel_vectorized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA vectorized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA vectorized trajectory kernel not available")

    def trajectory_kernel_memory_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA memory-optimized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA memory-optimized trajectory kernel not available")

    def trajectory_kernel_warp_optimized(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA warp-optimized trajectory kernel is unavailable."""
        raise RuntimeError("CUDA warp-optimized trajectory kernel not available")

    def trajectory_kernel_cache_friendly(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA cache-friendly trajectory kernel is unavailable."""
        raise RuntimeError("CUDA cache-friendly trajectory kernel not available")

    def inverse_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA inverse dynamics kernel is unavailable."""
        raise RuntimeError("CUDA inverse dynamics kernel not available")

    def forward_dynamics_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA forward dynamics kernel is unavailable."""
        raise RuntimeError("CUDA forward dynamics kernel not available")

    def cartesian_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA Cartesian trajectory kernel is unavailable."""
        raise RuntimeError("CUDA Cartesian trajectory kernel not available")

    def batch_trajectory_kernel(*args: Any, **kwargs: Any) -> NoReturn:
        """Raise because the CUDA batch trajectory kernel is unavailable."""
        raise RuntimeError("CUDA batch trajectory kernel not available")

    def _optimized_trajectory_generation_monitored_cuda(
        *args: Any, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use the CPU trajectory fallback when CUDA is unavailable."""
        return trajectory_cpu_fallback(args[0], args[1], args[2], args[3], args[4])


def optimized_trajectory_generation_monitored(
    thetastart: Any,
    thetaend: Any,
    Tf: float,
    N: int,
    method: int,
    use_pinned: bool = True,
    kernel_type: str = "auto",
    enable_monitoring: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a trajectory through the active backend's dispatch boundary."""
    if not _cuda_routing_enabled():
        return trajectory_cpu_fallback(thetastart, thetaend, Tf, N, method)
    return _optimized_trajectory_generation_monitored_cuda(
        thetastart,
        thetaend,
        Tf,
        N,
        method,
        use_pinned,
        kernel_type,
        enable_monitoring=enable_monitoring,
    )


# HIGH-LEVEL WRAPPER FUNCTIONS
def optimized_trajectory_generation(
    thetastart: Any,
    thetaend: Any,
    Tf: float,
    N: int,
    method: int,
    use_pinned: bool = True,
    kernel_type: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main entry point for optimized trajectory generation.

    This function automatically selects the best kernel and configuration
    for maximum performance and 40x+ speedups.

    Args:
        thetastart, thetaend: Start and end joint angles
        Tf: Final time
        N: Number of trajectory points
        method: Time scaling method (3=cubic, 5=quintic)
        use_pinned: Use pinned memory for faster transfers
        kernel_type: Kernel selection ("auto", "standard", "vectorized", etc.)
    """
    return optimized_trajectory_generation_monitored(
        thetastart,
        thetaend,
        Tf,
        N,
        method,
        use_pinned,
        kernel_type,
        enable_monitoring=True,
    )


def optimized_batch_trajectory_generation(
    thetastart_batch: np.ndarray,
    thetaend_batch: np.ndarray,
    Tf: float,
    N: int,
    method: int,
    use_pinned: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized batch trajectory generation for multiple trajectories.

    Args:
        thetastart_batch: (batch_size, num_joints) ndarray of starting joint
            angles, radians.
        thetaend_batch: (batch_size, num_joints) ndarray of ending joint angles,
            radians.
        Tf: Total trajectory duration, seconds.
        N: Number of trajectory time steps.
        method: Time-scaling order: 3 cubic, 5 quintic, else linear.
        use_pinned: If True, use pinned host memory for host-to-device transfers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(traj_pos_batch,
        traj_vel_batch, traj_acc_batch)``, each a ``(batch_size, N, num_joints)``
        float32 ndarray of joint positions (radians), velocities (radians/s), and
        accelerations (radians/s^2).

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not _cuda_routing_enabled():
        raise RuntimeError("CUDA not available for batch trajectory generation")

    batch_size, num_joints = thetastart_batch.shape

    # Use pinned memory for faster transfers
    if use_pinned:
        d_thetastart_batch = _h2d_pinned(
            np.ascontiguousarray(thetastart_batch, dtype=np.float32)
        )
        d_thetaend_batch = _h2d_pinned(
            np.ascontiguousarray(thetaend_batch, dtype=np.float32)
        )
    else:
        d_thetastart_batch = cuda.to_device(
            np.ascontiguousarray(thetastart_batch, dtype=np.float32)
        )
        d_thetaend_batch = cuda.to_device(
            np.ascontiguousarray(thetaend_batch, dtype=np.float32)
        )

    # Allocate output arrays
    d_traj_pos_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_vel_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)
    d_traj_acc_batch = get_cuda_array((batch_size, N, num_joints), dtype=np.float32)

    try:
        # 3D grid for batch processing
        grid = ((batch_size + 7) // 8, (N + 15) // 16, (num_joints + 7) // 8)
        block = (8, 16, 8)

        # Launch batch kernel - FIXED: Using 9 parameters instead of 10
        batch_trajectory_kernel[grid, block](
            d_thetastart_batch,
            d_thetaend_batch,
            d_traj_pos_batch,
            d_traj_vel_batch,
            d_traj_acc_batch,
            Tf,
            N,
            method,
            batch_size,
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
