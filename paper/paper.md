---
title: "ManipulaPy: Theoretical Foundations and System Integration"
tags:
  - robotics
  - manipulator
  - simulation
  - kinematics
  - dynamics
  - perception
  - cuda
  - trajectory-planning
  - computer-vision
authors:
  - name: Mohamed Ibrahim
    orcid: 0000-0002-1768-2031
    affiliation: 1
affiliations:
  - name: Universität Duisburg-Essen
    index: 1
date: 2024-06-01
bibliography: paper.bib
---

# Summary

**ManipulaPy** is an open‑source Python toolbox that stitches together *the entire manipulation pipeline*—from URDF parsing to GPU‑accelerated dynamics, vision‑based perception, planning and control—inside a single, consistent API. Built on the Product‑of‑Exponentials (PoE) model [@lynch2017modern], PyBullet [@coumans2019], CuPy [@cupy2021] and a small set of custom CUDA kernels, the library lets researchers move quickly from an abstract robot description to simulation, analysis and real‑time control.

Typical use‑cases include:

- kinematic/dynamic benchmarking of new robot designs  
- rapid prototyping of computed‑torque, robust or adaptive controllers  
- perception‑aware motion planning with on‑board stereo cameras  
- reinforcement‑learning environments that need millisecond‑level dynamics  

In internal benchmarks on a 6‑DOF UR5, the GPU kernels provide up to **40×** speed‑ups for large‑batch inverse dynamics compared with NumPy‑only baselines.

# Statement of Need

Modern robotic research demands *tight integration* of geometry, physics, vision and control. Existing libraries like MoveIt [@chitta2012moveit], Orocos KDL [@smits2009kdl], and the Python Robotics Toolbox [@corke2021] cover parts of this stack but require non‑trivial glue code or lack GPU support. **ManipulaPy**:

- converts a URDF to PoE screw parameters **and** realistic joint/torque limits in one call  
- exposes *CUDA‑backed* kernels for time‑scaling, forward/inverse dynamics and trajectory rollout  
- bundles stereo vision & DBSCAN‑based obstacle clustering that feed directly into the planner [@chu2023clustering]  
- ships a Simulation wrapper that synchronises PyBullet, cameras, planners and controllers  

This “batteries‑included” design removes weeks of boilerplate for graduate‑level projects and provides a reproducible platform for publications that rely on high‑frequency dynamics or perception‑loop experiments.
# Library Architecture
| Module | Purpose (selected highlights) |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `urdf_processor.py` | Parse URDF → screw axes $(S_i)$, home pose $(M)$, link inertias $(G_i)$; query PyBullet for joint/torque limits → `SerialManipulator`, `ManipulatorDynamics`. |
| `kinematics.py`     | PoE forward & inverse kinematics, Jacobians, hybrid (NN + iterative) IK. |
| `dynamics.py`       | Mass matrix $(M(\theta))$, Coriolis/gravity, inverse & forward dynamics (RNEA) with on‑GPU cache. |
| `path_planning.py`  | CUDA‑accelerated cubic/quintic joint and Cartesian trajectories; potential‑field collision shaping. |
| `potential_field.py`| Attractive/repulsive potentials + gradient computation for online obstacle avoidance. |
| `control.py`        | PD/PID, computed‑torque, robust, adaptive and Kalman‑filter controllers plus Ziegler–Nichols auto‑tune. |
| `vision.py`         | Camera abstraction, stereo rectification, disparity → depth, PyBullet debug sliders [@Febrianto2022]. |
| `perception.py`     | Depth→point‑cloud, DBSCAN clustering → obstacle list for the planner/RL. |
| `singularity.py`    | Jacobian determinant/condition‑number tests, manipulability ellipsoid and Monte‑Carlo workspace hull (CUDA). |
| `sim.py`            | One‑liner PyBullet world (ground, robot, sliders, logging, collision hooks). |
| `cuda_kernels.py`   | Raw GPU kernels (trajectory, dynamics, potential‑field) tuned for 256‑thread blocks. |
| `utils.py`          | Lie‑group helpers, cubic/quintic time scaling, matrix log/exp, SE(3) ↔ se(3). |

# Theory Highlights

## PoE Kinematics

A pose is obtained with[@lynch2017modern] 
$$ T(\theta)=e^{S_1\theta_1}\ldots e^{S_n\theta_n}M, $$
while the space Jacobian stacks each transformed screw axis
$$ J(\theta)=\left[\operatorname{Ad}_{T_1}S_1,\ldots,S_n\right]. $$

## Dynamics

Mass matrix via spatial inertia:
$$ M(\theta)=\sum_{i=1}^{n}\operatorname{Ad}_{T_i}^T G_i\,\operatorname{Ad}_{T_i}, \qquad \tau=M\ddot{\theta}+C(\theta,\dot\theta)+g(\theta). $$

GPU kernels solve batches of these equations in parallel.

## Stereo Depth

With focal length $f$ and baseline $B$:
$$ Z=\dfrac{fB}{d}. $$

## CUDA Kernels for Acceleration

ManipulaPy includes a custom CUDA backend to speed up the most time-critical operations:

- **Trajectory Kernel** (`trajectory_kernel`)[@shahid2024]
  This kernel computes joint trajectories using cubic or quintic time scaling. It parallelizes over each timestep using 256 threads per block:
  $$
  \theta_i(t) = s(t) \cdot (\theta_{\text{end}} - \theta_{\text{start}}) + \theta_{\text{start}}.
  $$

- **Forward Dynamics Kernel** (`forward_dynamics_kernel`)[@liang2018gpu]
  Solves $\ddot{\theta} = M^{-1}(\tau - C - g)$ for multiple time steps in parallel. Each thread processes one trajectory point, leveraging shared memory for intermediate matrix storage.

- **Inverse Dynamics Kernel** (`inverse_dynamics_kernel`)[@shahid2024]
  Computes the required torque $\tau$ from desired accelerations. This is used inside computed-torque controllers or for logging in training.

- **Cartesian Trajectory Kernel** (`cartesian_trajectory_kernel`)[@shahid2024]  
  Generates position/velocity/acceleration trajectories in SE(3) by interpolating position and rotating frames via exponential maps.

All kernels are compiled with Numba's @cuda.jit decorator and optimized for 256-thread blocks. This balances occupancy and avoids register spilling. CuPy arrays wrap all inputs/outputs so that dynamics and control modules operate natively on the GPU.

In trajectory benchmarks with 1000 steps, GPU rollout reduced latency from ~80ms to <4ms on an RTX 3060.


Depth points feed the obstacle detector which in turn perturbs the potential‑field planner.

# Minimal Example

```python
from ManipulaPy import urdf_processor, path_planning, control, sim
import numpy as np

# build model & CUDA‑ready dynamics
proc   = urdf_processor.URDFToSerialManipulator("xarm.urdf")
robot  = proc.serial_manipulator
dyn    = proc.dynamics
ctrl   = control.ManipulatorController(dyn)

# 45° joint ramp
Tf, N  = 3.0, 300
goal   = np.deg2rad([45]*6)
traj   = path_planning.TrajectoryPlanning(robot, "xarm.urdf",
        dyn, proc.robot_data["joint_limits"]).joint_trajectory(
        np.zeros(6), goal, Tf, N, method=3)

# PyBullet sim with PD control
simu   = sim.Simulation("xarm.urdf", proc.robot_data["joint_limits"])
simu.initialize_robot()
Kp, Kd = np.full(6, 80.0), np.full(6, 8.0)
simu.run_controller(ctrl, traj["positions"], traj["velocities"],
        traj["accelerations"], g=[0,0,-9.81], Ftip=np.zeros(6),
        Kp=Kp, Ki=np.zeros(6), Kd=Kd)
```

# Acknowledgements

Work supported by **Universität Duisburg‑Essen** and inspired by *Modern Robotics* [@lynch2017modern], PyBullet [@coumans2019], and Ultralytics YOLO [@jocher2022yolo] projects.

# References
