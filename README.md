# ManipulaPy

[![PyPI](https://img.shields.io/pypi/v/ManipulaPy)](https://pypi.org/project/ManipulaPy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![CI](https://github.com/boelnasr/ManipulaPy/actions/workflows/test.yml/badge.svg?branch=main)
![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)
[![status](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg)](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa)

**ManipulaPy** is a comprehensive, GPU-accelerated Python package for robotic manipulator analysis, simulation, planning, control, and perception. It provides a unified framework combining advanced kinematics, dynamics modeling, trajectory planning, control strategies, PyBullet simulation, and computer vision capabilities with optional CUDA acceleration.

---

## 🎯 Statement of Need

Modern robotics research and development requires sophisticated tools for manipulator analysis and control. Existing solutions often lack integration between kinematic analysis, dynamic modeling, trajectory planning, and perception systems. ManipulaPy addresses these critical gaps by providing:

- **🔧 Unified Framework**: Seamless integration of kinematics, dynamics, control, and perception
- **⚡ GPU Acceleration**: High-performance CUDA kernels for real-time applications
- **🔬 Research-Ready**: Comprehensive algorithms suitable for academic and industrial research
- **🧩 Modular Design**: Use individual components or the complete integrated system
- **📖 Well-Documented**: Extensive documentation with theoretical background and practical examples
- **🆓 Open Source**: AGPL-3.0 licensed for transparency, collaboration, and academic use

ManipulaPy fills the gap between basic robotics libraries and comprehensive research frameworks, providing researchers and developers with professional-grade tools for advanced robotic manipulation projects.

---

## 🚀 Key Features

- **🔧 Kinematic Analysis**: Forward and inverse kinematics for serial manipulators with Jacobian calculations
- **⚙️ Dynamic Modeling**: Complete dynamics analysis including mass matrix, Coriolis forces, and gravity compensation
- **🛤️ Path Planning**: CUDA-accelerated joint and Cartesian trajectory generation with collision avoidance
- **📊 Singularity Analysis**: Detect singularities, visualize manipulability ellipsoids, and compute workspace boundaries
- **📄 URDF Processing**: Convert URDF files into manipulatable Python models with PyBullet integration
- **🎮 Advanced Control**: PID, computed torque, adaptive, robust, and optimal control algorithms
- **🌐 Real-time Simulation**: PyBullet-based physics simulation with interactive visualization and GUI controls
- **👁️ Vision & Perception**: Stereo vision, depth processing, YOLO object detection, and 3D point cloud analysis
- **🚀 GPU Acceleration**: Optional CUDA kernels for high-performance trajectory planning and dynamics computation
- **📈 Visualization Tools**: Comprehensive plotting for trajectories, workspace analysis, and control performance

---

## 📦 Installation

### Basic Installation
```bash
pip install ManipulaPy
```


### Development Installation
```bash
git clone https://github.com/boelnasr/ManipulaPy.git
cd ManipulaPy
pip install -e .
```

### Verify Installation
```bash
python -c "import ManipulaPy; print('✅ ManipulaPy installed successfully')"
python -c "import ManipulaPy; ManipulaPy.check_dependencies()"
```

---

## 🛠️ Quick Start

### Basic Robot Setup
```python
import numpy as np
from ManipulaPy.urdf_processor import URDFToSerialManipulator

# Option 1: Use included robot models
try:
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
    print(f"Using xArm URDF: {urdf_file}")
except ImportError:
    # Option 2: Use your own URDF file
    urdf_file = "path/to/your/robot.urdf"

# Process URDF and create robot model
urdf_processor = URDFToSerialManipulator(urdf_file)
robot = urdf_processor.serial_manipulator
dynamics = urdf_processor.dynamics

print(f"✅ Robot loaded with {len(robot.joint_limits)} joints")
```

### Forward and Inverse Kinematics
```python
from math import pi

# Forward Kinematics
joint_angles = np.array([pi/6, pi/4, -pi/3, -pi/2, pi/4, pi/6])
end_effector_pose = robot.forward_kinematics(joint_angles)
print("End-effector pose:")
print(end_effector_pose)

# Inverse Kinematics
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.3, 0.4]  # Target position

solution, success, iterations = robot.iterative_inverse_kinematics(
    T_desired=target_pose,
    thetalist0=joint_angles,
    eomg=1e-6,  # Orientation tolerance
    ev=1e-6     # Position tolerance
)

if success:
    print(f"✅ IK converged in {iterations} iterations")
    print(f"Solution: {solution}")
else:
    print("❌ IK failed to converge")
```

---

## 🤖 Core Functionalities

### Advanced Trajectory Planning
```python
from ManipulaPy.path_planning import TrajectoryPlanning

# Initialize trajectory planner with collision checking
joint_limits = [(-pi, pi)] * 6
planner = TrajectoryPlanning(
    robot, urdf_file, dynamics, 
    joint_limits=joint_limits
)

# Generate smooth joint space trajectory
start_config = np.zeros(6)
end_config = np.array([pi/2, pi/4, pi/6, -pi/3, -pi/2, -pi/3])

trajectory = planner.joint_trajectory(
    thetastart=start_config,
    thetaend=end_config,
    Tf=5.0,  # 5 seconds
    N=100,   # 100 points
    method=5  # Quintic time scaling
)

print(f"Generated trajectory with {trajectory['positions'].shape[0]} points")

# Plot trajectory
planner.plot_trajectory(trajectory, Tf=5.0, title="Joint Space Trajectory")
```

### Comprehensive Dynamics Analysis
```python
from ManipulaPy.dynamics import ManipulatorDynamics

# Compute dynamics quantities
joint_velocities = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
joint_accelerations = np.array([0.1, 0.2, 0.1, 0.05, 0.05, 0.02])

# Mass matrix (configuration-dependent)
M = dynamics.mass_matrix(joint_angles)
print(f"Mass matrix condition number: {np.linalg.cond(M):.2f}")

# Coriolis and centrifugal forces
C = dynamics.velocity_quadratic_forces(joint_angles, joint_velocities)

# Gravity forces
G = dynamics.gravity_forces(joint_angles, g=[0, 0, -9.81])

# Inverse dynamics: τ = M(q)q̈ + C(q,q̇) + G(q)
required_torques = dynamics.inverse_dynamics(
    joint_angles, joint_velocities, joint_accelerations,
    [0, 0, -9.81], np.zeros(6)  # gravity, external forces
)
print(f"Required torques: {required_torques}")
```

### Advanced Control Systems
```python
from ManipulaPy.control import ManipulatorController

controller = ManipulatorController(dynamics)

# PID Control with tuned gains
Kp = np.diag([100, 100, 50, 50, 30, 30])
Ki = np.diag([10, 10, 5, 5, 3, 3])
Kd = np.diag([20, 20, 10, 10, 6, 6])

desired_pos = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])
current_pos = np.array([0.9, 0.4, 0.35, 0.25, 0.05, 0.05])
current_vel = np.zeros(6)

# PID control
control_torque = controller.pid_control(
    thetalistd=desired_pos,
    dthetalistd=np.zeros(6),
    thetalist=current_pos,
    dthetalist=current_vel,
    dt=0.01,
    Kp=Kp, Ki=Ki, Kd=Kd
)

# Computed torque control (model-based)
desired_acc = np.array([0.1, 0.1, 0.05, 0.05, 0.02, 0.02])
model_torque = controller.computed_torque_control(
    thetalistd=desired_pos,
    dthetalistd=np.zeros(6),
    ddthetalistd=desired_acc,
    thetalist=current_pos,
    dthetalist=current_vel,
    g=[0, 0, -9.81],
    dt=0.01,
    Kp=Kp, Ki=Ki, Kd=Kd
)
```

### Singularity Analysis and Workspace Computation
```python
from ManipulaPy.singularity import Singularity

singularity_analyzer = Singularity(robot)

# Check for singularities
is_singular = singularity_analyzer.singularity_analysis(joint_angles)
condition_num = singularity_analyzer.condition_number(joint_angles)

print(f"Configuration singular: {is_singular}")
print(f"Condition number: {condition_num:.2f}")

# Visualize manipulability ellipsoid
singularity_analyzer.manipulability_ellipsoid(joint_angles)

# Compute and visualize workspace
print("Computing robot workspace...")
singularity_analyzer.plot_workspace_monte_carlo(
    joint_limits=joint_limits,
    num_samples=5000
)
```

### Real-time Simulation with PyBullet
```python
from ManipulaPy.sim import Simulation

# Create simulation environment
sim = Simulation(
    urdf_file_path=urdf_file,
    joint_limits=joint_limits,
    time_step=0.01,
    real_time_factor=1.0
)

try:
    # Initialize simulation
    sim.initialize_robot()
    sim.initialize_planner_and_controller()
    sim.add_joint_parameters()  # Add GUI sliders

    print("🎮 Simulation started! Use GUI to control the robot.")
    
    # Execute trajectory
    if 'trajectory' in locals():
        final_pose = sim.run_trajectory(trajectory["positions"])
        print(f"Trajectory completed. Final pose: {final_pose}")
    
    # Manual control mode
    # sim.manual_control()  # Uncomment for interactive control
    
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    sim.close_simulation()
```

### Vision and Perception
```python
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception

# Setup camera configuration
camera_config = {
    "name": "main_camera",
    "intrinsic_matrix": np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32),
    "translation": [0, 0, 1.5],
    "rotation": [0, -30, 0],  # degrees
    "fov": 60
}

# Initialize vision and perception systems
vision = Vision([camera_config])
perception = Perception(vision)

# Detect and cluster obstacles
try:
    obstacles, labels = perception.detect_and_cluster_obstacles(
        depth_threshold=3.0,
        eps=0.1,
        min_samples=5
    )
    
    if len(obstacles) > 0:
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {len(obstacles)} obstacle points in {num_clusters} clusters")
    else:
        print("No obstacles detected")
        
except Exception as e:
    print(f"Vision system: {e} (normal if no camera available)")
```

### Performance Monitoring and GPU Acceleration
```python
from ManipulaPy.cuda_kernels import check_cuda_availability
import time

# Check GPU availability
if check_cuda_availability():
    print("🚀 CUDA acceleration available!")
    
    # Performance comparison
    start_time = time.time()
    large_trajectory = planner.joint_trajectory(
        start_config, end_config, 
        Tf=10.0, N=10000, method=5
    )
    gpu_time = time.time() - start_time
    
    print(f"GPU trajectory generation: {gpu_time:.3f} seconds")
    print(f"Performance: {10000/gpu_time:.0f} points/second")
    
else:
    print("⚠️ CUDA not available - using CPU fallback")
    print("Install GPU support: pip install ManipulaPy[gpu-cuda11]")
```

---

## 📁 Examples and Tutorials

The `Examples/` directory contains comprehensive demonstrations:

### Available Examples
- **`basic_kinematics.py`**: Forward/inverse kinematics with visualization
- **`trajectory_planning_demo.py`**: Advanced path planning with obstacles
- **`dynamics_analysis.py`**: Complete dynamics modeling and control
- **`simulation_demo.py`**: Real-time PyBullet simulation
- **`perception_demo.py`**: Stereo vision and object detection
- **`control_comparison.py`**: Various control strategies
- **`workspace_analysis.py`**: Singularity analysis and workspace computation

### Running Examples
```bash
cd Examples/
python basic_kinematics.py
python trajectory_planning_demo.py
python simulation_demo.py
```

---

## 🧪 Testing and Validation

ManipulaPy includes comprehensive tests covering all modules:

### Run Tests
```bash
# Install test dependencies
pip install ManipulaPy[dev]

# Run all tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_kinematics.py -v
python -m pytest tests/test_dynamics.py -v
python -m pytest tests/test_control.py -v

# Check installation and dependencies
python -c "import ManipulaPy; ManipulaPy.check_dependencies(verbose=True)"
```

### Performance Testing
```bash
# GPU performance test (if CUDA available)
python -c "from ManipulaPy.cuda_kernels import check_cuda_availability; print(f'CUDA: {check_cuda_availability()}')"

# Run performance benchmarks
python Examples/performance_test.py
```

---

## 🤝 Contributing

We welcome contributions! ManipulaPy is a community-driven project that benefits from diverse input.

### Quick Start for Contributors
```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/ManipulaPy.git
cd ManipulaPy

# Install in development mode
pip install -e .[dev]

# Run tests to verify setup
python -m pytest tests/ -x

# Make your changes and submit a pull request
```

### Contribution Guidelines
- **Follow PEP8** style guidelines
- **Add comprehensive tests** for new features
- **Update documentation** for API changes
- **Include examples** for new functionality
- **Maintain backward compatibility** when possible

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.

### Key Dependencies and Their Licenses

**Core Dependencies** (All compatible with AGPL-3.0):
- **`numpy`**, **`scipy`**, **`matplotlib`** → BSD License
- **`opencv-python`** → Apache 2.0 License  
- **`torch`** → BSD License
- **`pybullet`** → Zlib License
- **`scikit-learn`** → BSD License
- **`ultralytics`** → AGPL-3.0 License (YOLO object detection)

**Optional GPU Dependencies**:
- **`cupy`** → MIT License (GPU acceleration)
- **`numba`** → BSD License (JIT compilation)

All dependencies are compatible with AGPL-3.0 licensing. See [LICENSE](LICENSE) for complete terms.

### License Summary

ManipulaPy is free and open-source software. You can use, modify, and distribute it under AGPL-3.0 terms. If you distribute modified versions or use ManipulaPy in network services, you must make your source code available under the same license.

```
ManipulaPy - Comprehensive robotic manipulator analysis and control
Copyright (c) 2025 Mohamed Aboelnar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
```

---

## 🏆 Citation

If you use ManipulaPy in your research, please cite our work:

```bibtex
@software{manipulapy2025,
  title={ManipulaPy: A Comprehensive Python Package for Robotic Manipulator Analysis and Control},
  author={Mohamed Aboelnar},
  year={2025},
  url={https://github.com/boelnasr/ManipulaPy},
  version={1.2.0},
  license={AGPL-3.0-or-later}
}
```

For academic papers, please also consider citing the underlying mathematical frameworks and algorithms used in your specific application.

---

## 📬 Contact & Support

**Created and maintained by Mohamed Aboelnar**

### Contact Information
- 📧 **Email**: [aboelnasr1997@gmail.com](mailto:aboelnasr1997@gmail.com)
- 🐙 **GitHub**: [@boelnasr](https://github.com/boelnasr)
- 📖 **Documentation**: [manipulapy.readthedocs.io](https://manipulapy.readthedocs.io/)

### Getting Help
1. **📚 Check Documentation**: Comprehensive guides and API reference
2. **🔍 Browse Examples**: Complete working examples in `Examples/` directory
3. **🐛 Report Issues**: [GitHub Issues](https://github.com/boelnasr/ManipulaPy/issues)
4. **💬 Discussions**: [GitHub Discussions](https://github.com/boelnasr/ManipulaPy/discussions)
5. **📧 Direct Contact**: For complex questions or collaboration opportunities

Feel free to reach out with questions, bug reports, feature requests, or collaboration ideas!

---

## 🔄 Version History

### v1.2.0 (Current - January 2025)
- ✅ **AGPL-3.0 Licensing**: Full license compliance and transparency
- 🚀 **Enhanced GPU Support**: Improved CUDA kernels with graceful CPU fallbacks
- 🔧 **Better Error Handling**: Robust handling of missing dependencies
- 📦 **Improved Installation**: Cleaner dependency management with optional extras
- 📚 **Comprehensive Documentation**: Enhanced docstrings, examples, and JOSS compliance
- 🧪 **Extended Testing**: Broader test coverage and validation scenarios

### v1.1.0 (2024)
- 👁️ **Computer Vision**: Stereo vision and YOLO-based perception capabilities
- 🌐 **PyBullet Integration**: Real-time physics simulation and visualization
- ⚡ **GPU Acceleration**: CUDA kernels for trajectory planning and dynamics
- 🎮 **Advanced Control**: Additional control algorithms and strategies

### v1.0.0 (2024)
- 🔧 **Core Kinematics**: Forward/inverse kinematics and Jacobian calculations
- ⚙️ **Basic Dynamics**: Mass matrix and gravity force computation
- 🛤️ **Trajectory Planning**: Basic joint space trajectory generation
- 🎮 **Control Systems**: PID and basic control implementations

---

## ✅ Tested

ManipulaPy includes a comprehensive suite of unit tests covering:
- **Kinematics**: Forward/inverse kinematics accuracy and convergence
- **Dynamics**: Mass matrix properties and dynamics equations
- **Control**: Controller stability and performance
- **Perception**: Vision system functionality and robustness
- **Integration**: End-to-end system testing

We run tests with Python's built-in `unittest` and `pytest`. See the [`tests/`](./tests) folder for details.

**Test Coverage**: > 85% across all core modules  
**Continuous Integration**: Automated testing on multiple Python versions  
**Performance Validation**: Benchmarks for GPU acceleration and algorithm efficiency

---

## 🌟 Why Choose ManipulaPy?

### For Researchers
- **🔬 Academic Ready**: Comprehensive algorithms with solid mathematical foundations
- **📊 Extensible**: Modular design allows easy integration of new methods
- **📚 Well Documented**: Extensive documentation with theoretical background
- **🏆 Citable**: Proper citation format for academic publications

### For Developers  
- **⚡ High Performance**: GPU acceleration for computationally intensive tasks
- **🐍 Pythonic**: Clean, readable code following Python best practices
- **🔧 Modular**: Use only the components you need
- **🧪 Thoroughly Tested**: Comprehensive test suite ensuring reliability

### For Industry
- **🏭 Production Ready**: Robust error handling and graceful degradation
- **📈 Scalable**: Efficient algorithms suitable for real-time applications
- **🔒 Licensed**: Clear AGPL-3.0 licensing for commercial understanding
- **🤝 Actively Supported**: Regular updates and community support

---

*🤖 **ManipulaPy**: Empowering robotics research and development with comprehensive, high-performance tools for manipulator analysis and control.*

> 📌 **Latest Version**: `v1.2.0` - JOSS-compliant, GPU-accelerated, and ready for serious robotics work!

[![GitHub stars](https://img.shields.io/github/stars/boelnasr/ManipulaPy?style=social)](https://github.com/boelnasr/ManipulaPy)