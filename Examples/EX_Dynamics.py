#!/usr/bin/env python3
"""
ManipulaPy Dynamics Module Example (concise prints)
--------------------------------------------------
- Mass matrix
- Coriolis / centrifugal vector
- Gravity joint forces
- Inverse‑dynamics torque
- One forward‑dynamics Euler step
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

np_fmt = lambda arr: np.array2string(arr, separator=",", precision=4, suppress_small=True)


def main():
    # Build dynamics model from URDF
    urdf_proc = URDFToSerialManipulator(xarm_urdf_file)
    dyn   = urdf_proc.dynamics

    # Demo joint state
    theta  = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.0])
    dtheta = np.array([0.1, -0.05, 0.2, 0.0, 0.0, 0.05])
    dd_des = np.zeros_like(theta)
    g_vec  = np.array([0, 0, -9.81])
    Ftip   = np.zeros(6)

    # Mass matrix
    M = dyn.mass_matrix(theta)
    print("Mass matrix (6×6):\n", np_fmt(M))

    # Coriolis + centrifugal
    c = dyn.velocity_quadratic_forces(theta, dtheta)
    print("Coriolis/centrifugal forces:", np_fmt(c))

    # Gravity joint forces
    g_forces = dyn.gravity_forces(theta, g=g_vec)
    print("Gravity joint forces:", np_fmt(g_forces))

    # Inverse dynamics torque for zero desired accel
    tau = dyn.inverse_dynamics(theta, dtheta, dd_des, g_vec, Ftip)
    print("Inverse‑dynamics torque:", np_fmt(tau))

    # One forward‑dynamics Euler step
    dt = 0.001
    ddtheta = dyn.forward_dynamics(theta, dtheta, tau, g_vec, Ftip)
    theta_next  = theta  + dtheta * dt + 0.5 * ddtheta * dt**2
    dtheta_next = dtheta + ddtheta * dt
    print("θ_next:",  np_fmt(theta_next))
    print("θ̇_next:", np_fmt(dtheta_next))


if __name__ == "__main__":
    main()
