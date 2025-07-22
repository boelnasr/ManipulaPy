---
title: "ManipulaPy: A GPU‑Accelerated Python Framework for Robotic Manipulation, Perception, and Control"
tags: [robotics, manipulator, simulation, kinematics, dynamics, perception, cuda, trajectory-planning, computer-vision]
authors:
  - name: M.I.M. AboElNasr
    orcid: 0000-0002-1768-2031
    affiliation: 1
affiliations:
  - name: Universität Duisburg‑Essen
    index: 1
date: "2025-05-03"
bibliography: paper.bib
---

# Summary

**ManipulaPy** is an open‑source Python toolbox that unifies the entire manipulation pipeline—from URDF parsing to GPU‑accelerated dynamics, vision‑based perception, planning and control—within a single API. Built on the Product‑of‑Exponentials model [@lynch2017modern] (similar to Pinocchio [@Pinocchio2025] but with GPU acceleration), PyBullet [@coumans2019], CuPy [@cupy2021] and custom CUDA kernels [@liang2018gpu], the library enables researchers to move from robot description to real‑time control with up to **13× overall performance improvement** and **3600× faster inverse dynamics** on a 6‑DOF UR5 compared to NumPy baseline. Performance claims are reproducible via benchmarks in `benchmarks/README.md`.

# Statement of Need

Modern manipulation research requires tight integration of geometry, dynamics, perception, planning, and control—ideally within a single, real-time computational loop on GPU hardware. However, existing open-source tools address only portions of this pipeline, forcing researchers to write substantial integration code:

| Library | Core Strengths | Integration Challenges |
|---------|---------------|------------------------|
| **MoveIt** [@chitta2012moveit] | Mature sampling-based planners | Requires custom ROS nodes to bridge sensor data with planning; external plugins needed for real-time dynamics; no native GPU acceleration |
| **Pinocchio** [@Pinocchio2025] | High-performance PoE dynamics (C++) | CPU-only; separate perception and planning libraries must be manually synchronized; requires Python bindings for integration |
| **CuRobo** [@sundaralingam2023curobo] | GPU-accelerated collision checking and trajectory optimization | Planning-focused; lacks perception pipeline and closed-loop control; requires external sensor processing |
| **Python Robotics Toolbox** [@corke2021] | Educational algorithms with clear APIs | CPU-only implementation; users must implement their own simulators, controllers, and sensor processing |
| **PyRoKi** [@pyroki2025] | JAX-accelerated kinematics | Early development stage; limited dynamics and no perception support |
| **CBFPy** [@morton2025oscbf] | Control barrier functions with JAX | Specialized for safety-critical control; requires manual integration with perception and planning |

These integration challenges manifest as:
- **Sensor-planner gaps**: Converting camera data to collision geometries requires custom OpenCV → ROS → MoveIt pipelines
- **Dynamics-control mismatches**: Real-time controllers need consistent mass matrices, but most libraries compute dynamics separately from control loops  
- **GPU memory fragmentation**: Transferring data between CPU planners and GPU dynamics creates performance bottlenecks
- **Synchronization complexity**: Keeping sensors, planners, and controllers temporally aligned requires careful threading and message passing

**ManipulaPy** eliminates these integration burdens through a unified Python API that maintains data consistency across the entire manipulation pipeline:

![ManipulaPy manipulation pipeline architecture showing unified data flow from sensors through planning to control, with GPU acceleration throughout.](system_architecture.png)

Core design principles:

1. **Unified data structures**  
   All components share consistent representations (PoE screws, SE(3) transforms, GPU tensors)

2. **GPU-first architecture**  
   Trajectories, dynamics, and perception processing execute on GPU without CPU round-trips

3. **Temporal synchronization**  
   Built-in 1 kHz control loop keeps sensors, planners, and actuators phase-locked

4. **Extensible perception**  
   Multiple obstacle representations (primitives, point clouds, SDFs) supported simultaneously

Performance benchmarks demonstrating the claimed **13× overall speedup** are reproducible via ` Benchmarks/performance_benchmark.py` (requires CUDA-capable GPU).

![ManipulaPy PyBullet simulation showing GPU-accelerated trajectory execution with real-time collision avoidance. The robot smoothly navigates around dynamically detected obstacles while maintaining 1 kHz control rates.](manipulapy_trajectory.png)

# Library Architecture

ManipulaPy's architecture centers on a **unified manipulation pipeline** that maintains data consistency from sensor input to motor commands. Rather than loosely coupled modules, the system implements a coherent data flow where each component builds upon shared representations:


**Core Pipeline Components:**

**Robot Model Processing** converts URDF descriptions into Product-of-Exponentials representations, extracting screw axes, mass properties, and joint constraints through PyBullet integration. This creates the fundamental `SerialManipulator` and `ManipulatorDynamics` objects used throughout the system.

**Kinematics and Dynamics** implement GPU-accelerated forward/inverse kinematics, Jacobian computation, and Newton-Euler dynamics. Custom CUDA kernels optimize critical operations for 6-DOF manipulators, enabling real-time performance at 1 kHz control rates.

**Perception Integration** processes sensor data through a multi-stage pipeline supporting diverse input modalities. The `vision.py` module handles low-level camera operations (stereo rectification, calibration, image capture), while `perception.py` provides high-level semantic processing (object detection, clustering, obstacle representation). This separation allows users to plug in custom sensors while maintaining consistent 3D obstacle representations.

**Motion Planning** generates collision-free trajectories using GPU-accelerated time-scaling functions. The system supports both joint-space and Cartesian-space planning with real-time obstacle avoidance based on vision feedback.

**Control Systems** implement classical (PID, computed torque) and modern (adaptive, robust) control algorithms with automatic gain tuning. All controllers operate on the same dynamic model used in planning, ensuring consistency.

**Simulation Framework** provides PyBullet integration with synchronized camera rendering, physics simulation, and control execution. This enables seamless transition from simulation to real hardware.

**Key Architectural Decisions:**

- **Shared GPU Memory**: All components operate on GPU tensors, eliminating CPU-GPU transfer bottlenecks
- **Consistent Time Base**: 1 kHz control loop synchronizes all components  
- **Modular Perception**: Multiple obstacle representations coexist (geometric primitives, point clouds, signed distance fields)
- **Extensible Design**: New sensors, planners, and controllers integrate through well-defined interfaces

# Vision and Perception Pipeline

ManipulaPy's perception system addresses the challenge of converting raw sensor data into actionable robot knowledge through a five-stage pipeline that supports multiple obstacle representations:

![ManipulaPy perception pipeline showing sensor fusion, object detection, 3D integration, spatial clustering, and robot integration stages.](vision_pipeline.png)

**Stage 1: Sensor Fusion**
- **Stereo cameras**: RGB+depth via OpenCV rectification and SGBM disparity computation
- **RGB-D sensors**: Direct depth integration from RealSense, Kinect, or similar devices  
- **Point cloud input**: Direct processing of PCL/Open3D data structures
- **Multi-modal fusion**: Temporal alignment and calibration across sensor types

**Stage 2: Object Detection**  
- **YOLO v8 integration** [@Jocher_Ultralytics_YOLO_2023]: Real-time 2D bounding box detection at 30-50 FPS
- **Custom detector support**: Pluggable interface for domain-specific models
- **Geometric primitive detection**: Built-in recognition of spheres, boxes, cylinders from URDF specifications

**Stage 3: 3D Integration**
- **Depth projection**: Camera intrinsics $K$ transform pixel coordinates $(u,v)$ to 3D world positions  
- **Multi-frame fusion**: Temporal averaging reduces sensor noise and handles partial occlusions
- **Coordinate transformation**: Calibrated transforms $T_{base}^{cam}$ register sensor data to robot coordinates

**Stage 4: Spatial Clustering**
- **DBSCAN clustering** [@chu2021boundary]: Groups 3D points using $\epsilon$-neighborhoods for object segmentation
- **Hierarchical representations**: Octree/Octomap structures for large-scale environment mapping
- **Implicit surfaces**: Signed distance field generation for smooth collision checking

**Stage 5: Robot Integration**
- **Multi-representation support**: Simultaneously maintains geometric primitives, point clouds, and SDFs
- **Dynamic obstacle updates**: 5-15 Hz refresh rate during trajectory execution
- **Collision geometry generation**: Automatic conversion to convex hulls, bounding spheres, or custom shapes

**Supported Obstacle Representations:**

Unlike manipulation frameworks that handle only geometric primitives or require external mapping servers, ManipulaPy natively supports:
- **Geometric primitives**: Fast collision checking with spheres, boxes, cylinders
- **Unstructured point clouds**: Direct processing without conversion to meshes
- **Signed distance fields**: Smooth gradients for optimization-based planning
- **Octrees/Octomaps**: Hierarchical voxel representation for large environments
- **Hybrid representations**: Multiple formats coexist for different planning algorithms

This flexibility allows researchers to choose optimal representations for their specific applications while maintaining real-time performance through GPU acceleration.

# Theory and Implementation

## Product-of-Exponentials Kinematics

Like Pinocchio [@Pinocchio2025], ManipulaPy adopts the Product-of-Exponentials formulation for robot kinematics. However, while Pinocchio achieves performance through highly optimized C++ implementations, ManipulaPy provides GPU acceleration across the entire manipulation pipeline:

$$T(\theta) = e^{S_1 \theta_1} \cdots e^{S_n \theta_n} M$$

where each screw axis $S_i \in \mathbb{R}^6$ encodes joint motion and $M \in SE(3)$ represents the home configuration. The space-frame Jacobian becomes:

$$J(\theta) = \left[\operatorname{Ad}_{T_1}S_1, \ldots, S_n\right]$$

## GPU-Accelerated Dynamics

Custom CUDA kernels parallelize the recursive Newton-Euler algorithm for the fundamental dynamics equation:

$$\tau = M(\theta)\ddot{\theta} + C(\theta,\dot{\theta}) + G(\theta)$$

The mass matrix $M(\theta) = \sum_{i=1}^{n}\operatorname{Ad}_{T_i}^T G_i \operatorname{Ad}_{T_i}$ computation is optimized for 256-thread blocks, achieving up to **3600× speedup for inverse dynamics** and **8× speedup for trajectory generation** on 6-DOF manipulators compared to NumPy implementations.

# Acknowledgements

Work supported by **Universität Duisburg‑Essen** and inspired by *Modern Robotics* [@lynch2017modern], PyBullet [@coumans2019], Pinocchio [@Pinocchio2025], and Ultralytics YOLO [@Jocher_Ultralytics_YOLO_2023] projects.

# References