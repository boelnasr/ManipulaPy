from numba import cuda, float32
import numpy as np

@cuda.jit
def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
    """
    CUDA kernel to compute positions, velocities, and accelerations using cubic or quintic time scaling.
    """
    idx = cuda.grid(1)
    if idx < N:
        t = idx * (Tf / (N - 1))
        if method == 3:  # Cubic time scaling
            s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
            s_dot = 6 * (t / Tf) * (1 - t / Tf)
            s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
        elif method == 5:  # Quintic time scaling
            s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
            s_dot = 30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + t / Tf**2)
            s_ddot = 60 / (Tf**2) * (t / Tf) * (1 - 2 * (t / Tf))
        else:
            s = s_dot = s_ddot = 0

        for j in range(thetastart.shape[0]):
            traj_pos[idx, j] = s * (thetaend[j] - thetastart[j]) + thetastart[j]
            traj_vel[idx, j] = s_dot * (thetaend[j] - thetastart[j])
            traj_acc[idx, j] = s_ddot * (thetaend[j] - thetastart[j])

@cuda.jit
def inverse_dynamics_kernel(
    thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory, gravity_vector, Ftip,
    Glist, Slist, M, torques_trajectory, torque_limits):
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
                    M_temp[row, col] += Glist[i, row, col]  # Simplified for demonstration

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
            tau_temp[j] = max(torque_limits[j, 0], min(tau_temp[j], torque_limits[j, 1]))
            torques_trajectory[idx, j] = tau_temp[j]

@cuda.jit
def forward_dynamics_kernel(
    thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes,
    Glist, Slist, M, thetamat, dthetamat, ddthetamat, joint_limits):
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
                ddthetalist_local[i] = (current_tau[i] - c_temp[i] - g_temp[i]) / M_temp[i, i]  # Simplified

            # Integrate to get velocities and positions
            for i in range(len(thetalist)):
                current_dthetalist[i] += ddthetalist_local[i] * (dt / intRes)
                current_thetalist[i] += current_dthetalist[i] * (dt / intRes)

            # Enforce joint limits
            for i in range(len(thetalist)):
                current_thetalist[i] = max(joint_limits[i, 0], min(current_thetalist[i], joint_limits[i, 1]))

        # Store results
        for i in range(len(thetalist)):
            thetamat[idx, i] = current_thetalist[i]
            dthetamat[idx, i] = current_dthetalist[i]
            ddthetamat[idx, i] = ddthetalist_local[i]

@cuda.jit
def cartesian_trajectory_kernel(pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method):
    """
    CUDA kernel to compute Cartesian trajectory positions, velocities, and accelerations.
    """
    idx = cuda.grid(1)
    if idx < N:
        t = idx * (Tf / (N - 1))
        if method == 3:
            s = 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3
            s_dot = 6 * (t / Tf) * (1 - t / Tf)
            s_ddot = 6 / (Tf**2) * (1 - 2 * (t / Tf))
        elif method == 5:
            s = 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
            s_dot = 30 * (t / Tf) ** 2 * (1 - 2 * (t / Tf) + t / Tf**2)
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
            potential[idx] += 0.5 * (positions[idx, i] - goal[i])**2

@cuda.jit
def repulsive_potential_kernel(positions, obstacles, potential, influence_distance):
    """
    CUDA kernel to compute repulsive potential field.
    """
    idx = cuda.grid(1)
    if idx < positions.shape[0]:
        for obs in range(obstacles.shape[0]):
            dist = 0
            for i in range(3):
                dist += (positions[idx, i] - obstacles[obs, i])**2
            dist = np.sqrt(dist)
            if dist < influence_distance:
                for i in range(3):
                    potential[idx] += 0.5 * (1/dist - 1/influence_distance)**2

@cuda.jit
def gradient_kernel(potential, gradient):
    """
    CUDA kernel to compute the gradient of the potential field.
    """
    idx = cuda.grid(1)
    if idx < potential.shape[0] - 1:
        gradient[idx] = potential[idx + 1] - potential[idx]
