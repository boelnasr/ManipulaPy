---
title: "ManipulaPy: A GPU‑Accelerated Python Framework for Robotic Manipulation, Perception, and Control"
tags: [robotics, manipulator, simulation, kinematics, dynamics, perception, cuda, trajectory-planning, computer-vision]
authors:
  - name: M.I.M. Abo El Nasr
    orcid: 0000-0002-1768-2031
    affiliation: 1
affiliations:
  - name: Universität Duisburg‑Essen
    index: 1
date: "2025-05-03"
bibliography: paper.bib
---

# Summary

**ManipulaPy** is an open‑source Python toolbox that stitches together the entire manipulation pipeline—from URDF parsing to GPU‑accelerated dynamics, vision‑based perception, planning and control—within a single API.  Built on the Product‑of‑Exponentials model [@lynch2017modern], PyBullet [@coumans2019], CuPy [@cupy2021] and custom CUDA kernels [@liang2018gpu], the library lets researchers move from robot description to real‑time control with up to **40 ×** faster inverse‑dynamics on a 6‑DOF UR5 than a NumPy baseline.

## Statement of Need

Robotics research needs tight couplings of geometry, physics, vision and control. Existing stacks—MoveIt [@chitta2012moveit], Orocos KDL [@smits2009kdl] and the Python Robotics Toolbox [@corke2021]—cover parts of this but require glue code or lack GPU paths. **ManipulaPy** instead:

* converts a URDF to PoE screws **and** realistic joint limits in one call,  
* exposes CUDA kernels for time‑scaling and (inverse) dynamics [@liang2018gpu],  
* pipes stereo vision through DBSCAN obstacle clustering into the planner¹,  
* wraps PyBullet so cameras, planners and controllers stay synchronised at 1 kHz.

Implementation mirrors the clustering in [@chu2023clustering].

## Library Architecture

* **urdf_processor.py** – URDF → $(S_i,M,G_i)$ & limits → `SerialManipulator`, `ManipulatorDynamics`  
* **kinematics.py** – PoE FK/IK + Jacobians  
* **dynamics.py** – Mass matrix, Coriolis, gravity (GPU‑optional)  
* **path_planning.py** – CUDA cubic/quintic & SE(3) trajectories  
* **control.py** – PD/PID, computed‑torque, robust, adaptive controllers  
* **vision.py / perception.py** – Stereo → depth → DBSCAN obstacles  
* **singularity.py** – Jacobian condition, workspace Monte‑Carlo  
* **sim.py** – One‑line PyBullet setup & loop  
* **cuda_kernels.py** – Trajectory & dynamics kernels tuned for 256‑thread blocks  
* **utils.py** – Lie‑group and SE(3) helpers

## Theory Highlights

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

### CUDA Acceleration

Custom CUDA kernels optimize critical operations:

- **Trajectory Kernel**: Computes joint paths with cubic/quintic scaling
- **Forward Dynamics Kernel**: Solves equations of motion in parallel
- **Inverse Dynamics Kernel**: Calculates required torques from accelerations
- **Cartesian Trajectory Kernel**: Generates SE(3) trajectories with rotation interpolation

These kernels are optimized for 256-thread blocks, reducing trajectory generation latency.

## Minimal Example

```python
from ManipulaPy import urdf_processor, path_planning, control, sim
import numpy as np

# build model & CUDA-ready dynamics
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
