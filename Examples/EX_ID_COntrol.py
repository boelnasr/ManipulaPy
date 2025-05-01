#!/usr/bin/env python3
# Auto-tune & run computed-torque control on all XArm joints

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning
from ManipulaPy.control import ManipulatorController

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 1) Build robot, planner, controller
urdf_proc = URDFToSerialManipulator(xarm_urdf_file)
robot     = urdf_proc.serial_manipulator
dynamics  = urdf_proc.dynamics
planner   = TrajectoryPlanning(robot, xarm_urdf_file, dynamics, robot.joint_limits, None)
ctrl      = ManipulatorController(dynamics)

# 2) Trajectory: quintic from start to end
thetastart = np.deg2rad([0, -45,  30,   0, 45,   0])
thetaend   = np.deg2rad([30,  45, -30,   0, 60,   0])
Tf, N = 2.0, 300
traj = planner.joint_trajectory(thetastart, thetaend, Tf, N, method=5)

time = np.linspace(0, Tf, N, dtype=np.float32)
dt   = Tf / (N-1)
pos  = traj["positions"]    # (N×6)
vel  = traj["velocities"]
acc  = traj["accelerations"]

# 3) Control gains
g   = cp.asarray([0, 0, -9.81], dtype=cp.float32)
Kp  = cp.asarray([100, 100, 2000,  10,  100,  10], dtype=cp.float32)
Ki  = cp.asarray([10, 5, 100,  10,  8,  10], dtype=cp.float32)
Kd = cp.asarray([0, 0, 50,  15,  0,  20], dtype=cp.float32)

# logs
log_q   = np.zeros((N, 6), dtype=np.float32)
log_tau = np.zeros((N, 6), dtype=np.float32)

# 4) Run control + forward dynamics
q  = thetastart.copy()
dq = np.zeros(6, dtype=np.float32)

for i in range(N):
    q_d   = pos[i]
    dq_d  = vel[i]
    ddq_d = acc[i]

    # 4a) compute torque
    tau_cp = ctrl.computed_torque_control(
        q_d, dq_d, ddq_d,
        q,   dq,
        g, dt, Kp, Ki, Kd
    )
    log_tau[i] = tau_cp.get()

    # 4b) forward simulate
    ddq = dynamics.forward_dynamics(q, dq, log_tau[i].tolist(), [0,0,-9.81], [0]*6)
    dq += ddq * dt
    q  += dq * dt
    log_q[i] = q

# 5) Plot position tracking for all joints
fig1, axs1 = plt.subplots(6, 1, sharex=True, figsize=(8, 12))
for j in range(6):
    axs1[j].plot(time, np.rad2deg(pos[:, j]), '--', label='desired')
    axs1[j].plot(time, np.rad2deg(log_q[:, j]),     label='actual')
    axs1[j].set_ylabel(f'J{j+1} [°]')
    axs1[j].legend(loc='upper right')
    axs1[j].grid(True)
axs1[-1].set_xlabel('Time [s]')
fig1.suptitle('Computed-Torque Tracking (All Joints)', y=0.94)
plt.tight_layout(rect=[0,0,1,0.96])
fig1.savefig('ctc_tracking_all.png', dpi=200)

# 6) Plot torque profiles for all joints
fig2, axs2 = plt.subplots(6, 1, sharex=True, figsize=(8, 12))
for j in range(6):
    axs2[j].plot(time, log_tau[:, j])
    axs2[j].set_ylabel(f'τ{j+1} [Nm]')
    axs2[j].grid(True)
axs2[-1].set_xlabel('Time [s]')
fig2.suptitle('Computed-Torque Command (All Joints)', y=0.94)
plt.tight_layout(rect=[0,0,1,0.96])
fig2.savefig('ctc_tau_all.png', dpi=200)

print("Wrote ctc_tracking_all.png and ctc_tau_all.png")
