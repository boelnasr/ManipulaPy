#!/usr/bin/env python3
"""
Example: joint-space trajectory generation + combined JS + Cartesian PF avoidance
--------------------------------------------------------------------
Run as:
    python example_trajectory_planning.py
Produces:
    - joint_trajectory.png        # original joint plots
    - ee_path.png                 # TCP path with obstacles & avoidance
    - torque_profile.png          # torques for avoidance trajectory
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import TrajectoryPlanning
from ManipulaPy.utils import TransInv

# ------------------------------------------------------------------
# 1.  Robot model
# ------------------------------------------------------------------
urdf_proc          = URDFToSerialManipulator(xarm_urdf_file)
serial_manipulator = urdf_proc.serial_manipulator
dynamics           = urdf_proc.dynamics
joint_limits       = serial_manipulator.joint_limits      # list[(min,max)]
torque_limits      = [(-np.inf, np.inf)] * len(joint_limits)

planner = TrajectoryPlanning(
    serial_manipulator,
    xarm_urdf_file,
    dynamics,
    joint_limits,
    torque_limits
)
pf = planner.potential_field

# ------------------------------------------------------------------
# 2.  Define motion
# ------------------------------------------------------------------
thetastart = np.deg2rad([  0, -90,  90,   0,  90,   0])
thetaend   = np.deg2rad([ 45, -45,  60,   0, 110,  45])
Tf, N, method = 10.0, 200, 5

traj = planner.joint_trajectory(thetastart, thetaend, Tf, N, method)

# ------------------------------------------------------------------
# 3.  Plot joint position/velocity/acceleration (original)
# ------------------------------------------------------------------
planner.plot_trajectory(
    trajectory_data=traj,
    Tf=Tf,
    title="xArm joint trajectory (quintic)",
    labels=[f"Joint {i+1}" for i in range(len(joint_limits))]
)
plt.gcf().savefig("joint_trajectory.png", dpi=400)

# ------------------------------------------------------------------
# 4.  Define joint-space obstacles
# ------------------------------------------------------------------
obs_degrees = [
    [ 30, -30,  45,   0,  60,   0],
    [ 15,  15,  90,   0,  45,   0],
]
obs_js = [np.deg2rad(q) for q in obs_degrees]

# ------------------------------------------------------------------
# 4b. Cartesian obstacles *relative to robot base*
# ------------------------------------------------------------------
# If your robot base in the world is not at the origin, set world_T_base accordingly:
world_T_base = np.eye(4, dtype=np.float32)
# e.g. world_T_base[:3,3] = [0.1, 0.0, 0.2]

# define in world coordinates
cart_obs_world = np.array([
    [1.20,  0.40, 0.80],
    [0.40,  0.20, 0.30],
], dtype=np.float32)

# transform into base frame
h       = np.hstack((cart_obs_world, np.ones((len(cart_obs_world),1),dtype=np.float32)))
rel_h   = (TransInv(world_T_base) @ h.T).T
cart_obs = rel_h[:,:3]   # in base frame

# ------------------------------------------------------------------
# 5.  Build avoided JS path via combined PF (JS + Cartesian)
# ------------------------------------------------------------------
alpha       = 0.05      # step size
max_iters   = 50        # inner PF iterations per waypoint
safety_dist = 0.05      # leave this clearance in Cartesian

avoided = []
x_goal  = serial_manipulator.forward_kinematics(thetaend)[:3,3]

for q in traj["positions"]:
    q_new = q.copy()
    for _ in range(max_iters):
        # (1) JS potential-field gradient (attractive to thetaend, repulsive from obs_js)
        grad_js = pf.compute_gradient(q_new, thetaend, obs_js)

        # (2) Cartesian PF on the TCP
        T       = serial_manipulator.forward_kinematics(q_new)
        x_cur   = T[:3,3]
        # repulsive only (we already attract in joint space)
        grad_cart_xyz = np.zeros(3, dtype=np.float32)
        for obs in cart_obs:
            d = np.linalg.norm(x_cur - obs)
            if d <= pf.influence_distance:
                grad_cart_xyz += (
                    pf.repulsive_gain
                    * (1.0/d - 1.0/pf.influence_distance)
                    * (1.0/(d**3))
                    * (x_cur - obs)
                )

        # map that Cartesian gradient back to joint space via J^T
        J = serial_manipulator.jacobian(q_new)  # 6×n
        grad_cart_js = J[:3,:].T.dot(grad_cart_xyz)

        # (3) combine
        total_grad = grad_js + grad_cart_js

        # (4) take a step
        q_new -= alpha * total_grad

        # (5) check if TCP now clears all world‐frame obstacles in Cartesian
        tcp_world = (world_T_base @ np.r_[T[:3,3],1])[:3]
        dists = np.linalg.norm(cart_obs_world - tcp_world, axis=1)
        if np.all(dists > safety_dist):
            break

    avoided.append(q_new)

avoided = np.array(avoided, dtype=np.float32)

# compute TCP paths back in world frame for plotting
ee_orig     = np.array([(world_T_base @ np.r_[serial_manipulator.forward_kinematics(q)[:3,3],1])[:3]
                        for q in traj["positions"]])
ee_avoid    = np.array([(world_T_base @ np.r_[serial_manipulator.forward_kinematics(q)[:3,3],1])[:3]
                        for q in avoided])
ee_js_obs   = np.array([(world_T_base @ np.r_[serial_manipulator.forward_kinematics(q)[:3,3],1])[:3]
                        for q in obs_js])
ee_cart_obs = cart_obs_world

# ------------------------------------------------------------------
# 6.  Plot TCP path, obstacles & avoidance
# ------------------------------------------------------------------
fig_ee = plt.figure(figsize=(8,6))
ax = fig_ee.add_subplot(111, projection="3d")
ax.plot(*ee_orig.T,  label="original", lw=2)
ax.plot(*ee_avoid.T, label="avoided", lw=2, ls="--")

ax.scatter(*ee_js_obs.T,  c="r", s=80, marker="X", label="JS obstacles")
ax.scatter(*ee_cart_obs.T,c="k", s=100, marker="o", label="Cartesian obstacles")

ax.scatter(*ee_orig[0],  c="g", s=80, label="start")
ax.scatter(*ee_orig[-1], c="m", s=80, label="goal")

ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
ax.set_title("TCP path: original vs. PF avoidance")
ax.legend()
fig_ee.savefig("ee_path.png", dpi=200)

# ------------------------------------------------------------------
# 7.  Torque profile for avoided path
# ------------------------------------------------------------------
dt = Tf/(N-1)
d_avoided  = np.vstack((avoided[0], np.diff(avoided,axis=0)))  / dt
dd_avoided = np.vstack((avoided[0:2], np.diff(avoided,2,axis=0))) / dt**2

tau = planner.inverse_dynamics_trajectory(
    thetalist_trajectory   = avoided,
    dthetalist_trajectory  = d_avoided,
    ddthetalist_trajectory = dd_avoided,
)

time     = np.linspace(0, Tf, N)
n_joints = tau.shape[1]
fig_tau, axs = plt.subplots(n_joints,1,sharex=True,figsize=(8,2*n_joints))
for j in range(n_joints):
    axs[j].plot(time, tau[:,j])
    axs[j].set_ylabel(f"τ{j+1} [Nm]")
axs[-1].set_xlabel("time [s]")
fig_tau.suptitle("Torques for avoided trajectory")
fig_tau.tight_layout()
fig_tau.savefig("torque_profile.png", dpi=200)

print("Plots written to: joint_trajectory.png, ee_path.png, torque_profile.png")
