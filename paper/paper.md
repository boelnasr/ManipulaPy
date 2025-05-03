---
title: "ManipulaPy: A GPU-Accelerated Python Framework for Robotic Manipulation, Perception, and Control"
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
  - name: M.I.M. Abo El Nasr
    orcid: 0000-0002-1768-2031
    affiliation: 1
affiliations:
  - name: Universität Duisburg-Essen
    index: 1
date: 2025-05-03
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

| Module              | Purpose (selected highlights)                                                                                                                                             |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `urdf_processor.py`     | Parses URDF files into screw axes $(S_i)$, home pose $(M)$, and link inertias $(G_i)$; queries PyBullet for joint/torque limits → `SerialManipulator`, `ManipulatorDynamics`. |
| `kinematics.py`         | Implements PoE-based forward and inverse kinematics, Jacobians, and hybrid (neural + iterative) inverse kinematics solvers.                                              |
| `dynamics.py`           | Computes mass matrix $M(\theta)$, Coriolis and gravity terms; supports forward and inverse dynamics (RNEA), with optional GPU caching.                                  |
| `path_planning.py`      | CUDA-accelerated cubic/quintic joint and Cartesian trajectory generation; includes potential-field shaping for collision-aware paths.                                    |
| `potential_field.py`    | Computes attractive and repulsive potentials and their gradients for real-time obstacle avoidance.                                                                      |
| `control.py`            | Offers PD/PID, computed-torque, robust, adaptive, and Kalman-filter-based controllers; supports Ziegler–Nichols tuning.                                                  |
| `vision.py`             | Handles camera modeling, stereo rectification, disparity-to-depth conversion; includes PyBullet-based GUI tuning interface .                             |
| `perception.py`         | Converts depth maps to point clouds; clusters points using DBSCAN to detect obstacles for planning and reinforcement learning.                                            |
| `singularity.py`        | Analyzes Jacobian singularities via condition numbers and manipulability ellipsoids; estimates workspace volume via Monte Carlo (GPU).                                   |
| `sim.py`                | Sets up a PyBullet environment in one line; loads robots, applies joint sliders, runs control loops, and performs collision monitoring.                                  |
| `cuda_kernels.py`       | Hosts low-level GPU kernels (trajectory rollout, dynamics, potential fields), tuned for 256-thread CUDA blocks.                                                          |
| `utils.py`              | Provides utility functions for Lie group operations, time scaling, matrix logarithms/exponentials, and SE(3) ↔ se(3) conversions.                                         |
# Theory Highlights

## URDF Processing and Robot Model Generation

A core capability of **ManipulaPy** is converting a robot's URDF (Unified Robot Description Format) file into an internal computational model suitable for kinematics, dynamics, and control. This transformation is handled by the `URDFToSerialManipulator` class.

When a user loads a robot model:

```python
from ManipulaPy.urdf_processor import URDFToSerialManipulator

urdf_proc = URDFToSerialManipulator("xarm.urdf")
```

the following steps are executed:

### 1. URDF Parsing

- The URDF is parsed using PyBullet’s internal loader to access link names, joint types, limits, masses, and inertia tensors.
- The joint hierarchy is extracted as a tree of revolute/prismatic joints with parent-child link relations.

### 2. Screw Axis Extraction

- Each joint is converted into a **screw axis** $S_i \in \mathbb{R}^6$, using the joint’s origin, axis, and type.
- The screw axis encodes both rotation and translation components:
  
  $$
  S = \begin{bmatrix} \omega \\ v \end{bmatrix}, \quad \text{with } \omega = \text{axis},\quad v = -\omega \times q
  $$

  where $q$ is the joint position in the base frame.

### 3. Home Configuration Matrix $M$

- The default pose of the end-effector with all joint angles at zero is computed as a homogeneous transformation matrix $M \in SE(3)$.
- This serves as the base pose in PoE kinematics:

  $$
  T(\theta) = e^{S_1 \theta_1} \cdots e^{S_n \theta_n} M
  $$
while the space Jacobian stacks each transformed screw axis
$$ J(\theta)=\left[\operatorname{Ad}_{T_1}S_1,\ldots,S_n\right]. $$


### 4. Inertial Property Extraction

- Each link’s mass and spatial inertia tensor are wrapped into a $6 \times 6$ **spatial inertia matrix** $G_i$ for use in dynamic calculations.
- These matrices are used to construct the mass matrix $M(\theta)$ and Coriolis/gravity terms.
$$ M(\theta)=\sum_{i=1}^{n}\operatorname{Ad}_{T_i}^T G_i\,\operatorname{Ad}_{T_i}, \qquad \tau=M\ddot{\theta}+C(\theta,\dot\theta)+g(\theta). $$


### 5. Limit and Metadata Mapping

- PyBullet is queried to extract joint limits, damping/friction, and torque constraints.
- This metadata is stored in `robot_data` and injected into the planner and controller to enforce safety and realism.

### 6. Model Object Output

- Finally, two primary Python objects are constructed:
  - `SerialManipulator`: A pure kinematic model with screw axes and link transforms.
  - `ManipulatorDynamics`: A dynamic model with mass, inertia, and external force computations.

These are returned via:

```python
robot = urdf_proc.serial_manipulator
dynamics = urdf_proc.dynamics
```

This one-call setup bridges URDF semantics with analytical modeling, enabling immediate simulation, control, and planning.


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

All kernels are compiled with Numba's **\@cuda.jit** decorator and optimized for 256-thread blocks. This balances occupancy and avoids register spilling. CuPy arrays wrap all inputs/outputs so that dynamics and control modules operate natively on the GPU.

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
