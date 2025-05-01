#!/usr/bin/env python3
"""
ManipulaPy Kinematics Example (headless‑friendly)
================================================
- Load URDF into a SerialManipulator
- Forward kinematics (space frame)
- Spatial Jacobian
- End‑effector spatial velocity
- Damped iterative IK with residuals saved to PNG
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / CI‑safe backend

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

# Helper for neat array printing
np_print = lambda arr: np.array2string(arr, separator=',', precision=5, suppress_small=True)


def main():
    # --- 1. Build model from URDF -------------------------------------------------
    manip = URDFToSerialManipulator(xarm_urdf_file).serial_manipulator

    # --- 2. Demo joint state ------------------------------------------------------
    theta  = [0.1, 0.2, 0.3, -0.1, 0.0, 0.1]      # rad
    dtheta = [0.05, -0.02, 0.1, 0.0, -0.05, 0.02]  # rad/s

    # --- 3. Forward kinematics ----------------------------------------------------
    T_end = manip.forward_kinematics(theta, frame="space")
    print("(End‑Effector Pose 4×4):\n", np_print(T_end))

    # --- 4. Spatial Jacobian ------------------------------------------------------
    J_space = manip.jacobian(theta, frame="space")
    print("(Spatial Jacobian 6×6):\n", np_print(J_space))

    # --- 5. EE spatial velocity ---------------------------------------------------
    twist = manip.end_effector_velocity(theta, dtheta, frame="space")
    print(" (EE Twist [ω_x,ω_y,ω_z,v_x,v_y,v_z]):\n", np_print(twist))

    # --- 6. Iterative IK ----------------------------------------------------------
    theta0 = np.zeros(len(theta))
    sol, ok, iters = manip.iterative_inverse_kinematics(
        T_end,
        thetalist0=theta0,
        eomg=1e-6,
        ev=1e-6,
        max_iterations=5000,
        plot_residuals=True,          # PNG saved by IK method
    )
    if ok:
        print(f"\nIK converged in {iters} iterations. Solution:\n", np_print(sol))
    else:
        print(f"\nIK failed after {iters} iterations. Last estimate:\n", np_print(sol))

    print("Residual plot saved as 'ik_residuals.png' (see working directory).")


if __name__ == "__main__":
    main()
