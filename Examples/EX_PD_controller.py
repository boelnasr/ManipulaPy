#!/usr/bin/env python3
# Auto-tune & run PD on all XArm joints, moving **all** joints 45°

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.control import ManipulatorController

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 1) Build robot + controller
urdf_proc = URDFToSerialManipulator(xarm_urdf_file)
ctrl      = ManipulatorController(urdf_proc.dynamics)

# 2) Simulation parameters
n_steps, dt = 300, 0.01
time        = np.linspace(0, (n_steps-1)*dt, n_steps, dtype=np.float32)

# 3) Auto-tune Ku & Tu on a 10° step **for each** joint
Kp_vals = np.zeros(6, dtype=np.float32)
Kd_vals = np.zeros(6, dtype=np.float32)

for j in range(6):
    initial = cp.zeros(6, dtype=cp.float32)
    target  = cp.zeros(6, dtype=cp.float32)
    target[j] = np.deg2rad(10.0)  # small step for identification

    # positional call: (initial, target, dt, max_steps)
    Ku, Tu, _, _ = ctrl.find_ultimate_gain_and_period(
        initial, target, dt, n_steps
    )
    log.info(f"Joint {j+1}: Ku={Ku:.4f}, Tu={Tu:.4f}s")

    if np.isfinite(Ku) and np.isfinite(Tu) and Tu > 0:
        Kp, Ki, Kd = ctrl.tune_controller(Ku, Tu)
        # drop Ki for pure PD
    else:
        log.warning(f"J{j+1} auto-tune failed → fallback PD gains")
        Kp, Kd = 50.0, 1.0

    Kp_vals[j] = float(Kp)
    Kd_vals[j] = float(Kd)
    log.info(f" → Kp={Kp_vals[j]:.1f}, Kd={Kd_vals[j]:.1f}")

# 4) Build a **simultaneous** 0→45° ramp for ALL joints
ramp    = np.deg2rad(45.0) * (time / time[-1])   # shape (n_steps,)
thetad  = np.tile(ramp[:,None], (1,6))          # shape (n_steps, 6)
dthetad = np.zeros_like(thetad, dtype=np.float32)

# move Kp/Kd to GPU
Kp_gpu = cp.asarray(Kp_vals)
Kd_gpu = cp.asarray(Kd_vals)

# 5) "Unit-inertia" simulation: θ¨ = τ, integrate twice
theta   = cp.zeros(6, dtype=cp.float32)  # Use CuPy arrays
dtheta  = cp.zeros(6, dtype=cp.float32)  # Use CuPy arrays
history = np.zeros((n_steps,6), dtype=np.float32)

for i in range(n_steps):
    des_pos = cp.asarray(thetad[i])
    des_vel = cp.asarray(dthetad[i])

    # PD torque (all in CuPy)
    tau = ctrl.pd_control(
        desired_position = des_pos,
        desired_velocity = des_vel,
        current_position = theta,
        current_velocity = dtheta,
        Kp = Kp_gpu,
        Kd = Kd_gpu
    )

    # integrate θ¨ = τ  (I=1)
    dtheta += tau * dt
    theta  += dtheta * dt
    history[i] = theta.get()  # Convert from CuPy to NumPy for storage

# 6) Plot all six joints
fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 12))
for j in range(6):
    axs[j].plot(time, np.rad2deg(thetad[:, j]), '--', label='desired')
    axs[j].plot(time, np.rad2deg(history[:, j]),    label='actual')
    axs[j].set_ylabel(f'J{j+1} [°]')
    axs[j].grid(True)
    axs[j].legend(loc='upper right')

axs[-1].set_xlabel('Time [s]')
fig.suptitle('Auto-Tuned PD Tracking: All 6 Joints to 45°', y=0.94)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('auto_pd_all_joints_45deg.png', dpi=200)
print("Wrote auto_pd_all_joints_45deg.png")