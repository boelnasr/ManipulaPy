#!/usr/bin/env python3
"""
Enhanced ManipulaPy Usage Example:
- Plans and saves joint trajectory plot
- Suppresses Numba performance warnings
- Configures logging to hide CUDA info
"""

import os
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from numba.core.errors import NumbaPerformanceWarning
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning
# import the data for the xarm
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

# Suppress Numba performance warnings
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
# Reduce CUDA driver log verbosity
logging.getLogger('numba.cuda.cudadrv.driver').setLevel(logging.WARNING)
# Optionally disable PyBullet info logs via environment
os.environ['PYBULLET_LOG_LEVEL'] = 'WARN'

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main(output_file="xarm_trajectory.png", method=5, Tf=5.0, N=100):
    # 1. Initialize manipulator from URDF
    xarm = URDFToSerialManipulator(xarm_urdf_file)

    # 2. Extract model and dynamics
    xarm_manipulator = xarm.serial_manipulator
    xarm_dynamics = xarm.dynamics
    joint_limits = xarm_manipulator.joint_limits

    # 3. Create trajectory planner
    planner = TrajectoryPlanning(
        serial_manipulator=xarm_manipulator,
        urdf_path=xarm_urdf_file,
        dynamics=xarm_dynamics,
        joint_limits=joint_limits,
    )

    # 4. Define start and end joint configs
    theta_start = [0, 0, 0, 0, 0, 0]
    theta_end = [1.0, 0.5, 0.7, 0.3, 0.2, 0.1]

    logging.info("Planning joint trajectory (method=%d, Tf=%.1f, N=%d)", method, Tf, N)
    trajectory = planner.joint_trajectory(
        thetastart=theta_start,
        thetaend=theta_end,
        Tf=Tf,
        N=N,
        method=method,
    )

    # 5. Extract trajectory data
    positions = trajectory["positions"]

    # 6. Plot positions
    time = np.linspace(0, Tf, positions.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(time, positions[:, i], label=f"Joint {i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.title(f"xArm Joint Trajectory (Method {method})")
    plt.legend()
    plt.grid(True)

    # 7. Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    logging.info("Trajectory plot saved to %s", output_file)

if __name__ == "__main__":
    main()
