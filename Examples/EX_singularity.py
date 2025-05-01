#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("Agg")
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
from ManipulaPy.singularity import Singularity  

def main():
    # 1) Load robot
    urdf_proc = URDFToSerialManipulator(xarm_urdf_file)
    robot = urdf_proc.serial_manipulator

    # determine DOF correctly from the S_list
    dof = robot.S_list.shape[1]

    # 2) Singularity helper
    sing = Singularity(robot)

    # 3) Home pose (all zeros, length = DOF)
    thetalist = np.zeros(dof, dtype=np.float32)

    # 4) Check for exact singularity
    if sing.singularity_analysis(thetalist):
        print("⚠️ Robot is at a singularity at the home pose!")
    else:
        print("✅ Home pose is non-singular.")

    # 5) Plot manipulability ellipsoids at joint1 = 30°
    test_pose = thetalist.copy()
    test_pose[0] = np.deg2rad(30)
    fig = plt.figure(figsize=(10,5))
    sing.manipulability_ellipsoid(test_pose)
    plt.suptitle("Manipulability Ellipsoids at Joint 1 = 30°")
    plt.savefig("manipulability_ellipsoid.png", dpi=200)
    print("→ Saved manipulability_ellipsoid.png")

    # 6) Estimate & plot workspace via Monte Carlo
    joint_limits = urdf_proc.robot_data["joint_limits"]
    fig = plt.figure(figsize=(8,8))
    sing.plot_workspace_monte_carlo(joint_limits, num_samples=5000)
    plt.title("Estimated Workspace (Convex Hull)")
    plt.savefig("workspace_convex_hull.png", dpi=200)
    print("→ Saved workspace_convex_hull.png")

if __name__ == "__main__":
    main()
