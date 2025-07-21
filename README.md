# ManipulaPy

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ManipulaPy)](https://pypi.org/project/ManipulaPy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![CI](https://github.com/boelnasr/ManipulaPy/actions/workflows/test.yml/badge.svg?branch=main)
![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)
[![status](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg)](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa)

**A comprehensive, GPU-accelerated Python package for robotic manipulator analysis, simulation, planning, control, and perception.**

[Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Installation](#-installation) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

ManipulaPy is a modern, comprehensive framework that bridges the gap between basic robotics libraries and sophisticated research tools. It provides seamless integration of kinematics, dynamics, control, and perception systems with optional CUDA acceleration for real-time applications.

### Why ManipulaPy?

**🔧 Unified Framework**: Complete integration from low-level kinematics to high-level perception  
**⚡ GPU Accelerated**: CUDA kernels for trajectory planning and dynamics computation  
**🔬 Research Ready**: Mathematical rigor with practical implementation  
**🧩 Modular Design**: Use individual components or the complete system  
**📖 Well Documented**: Comprehensive guides with theoretical foundations  
**🆓 Open Source**: AGPL-3.0 licensed for transparency and collaboration

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔧 **Core Robotics**
- **Kinematics**: Forward/inverse kinematics with Jacobian analysis
- **Dynamics**: Mass matrix, Coriolis forces, gravity compensation
- **Control**: PID, computed torque, adaptive, robust algorithms
- **Singularity Analysis**: Detect singularities and workspace boundaries

</td>
<td width="50%">

### 🚀 **Advanced Capabilities**
- **Path Planning**: CUDA-accelerated trajectory generation
- **Simulation**: Real-time PyBullet physics simulation
- **Vision**: Stereo vision, YOLO detection, point clouds
- **URDF Processing**: Convert robot models to Python objects

</td>
</tr>
</table>

---



## 🚀 Quick Start

### Prerequisites

Before installing ManipulaPy, make sure your system has:

1. **NVIDIA Drivers & CUDA Toolkit**  
   - `nvcc` on your `PATH` (e.g. via `sudo apt install nvidia-cuda-toolkit` or the [official NVIDIA CUDA installer](https://developer.nvidia.com/cuda-downloads)).  
   - Verify with:
     ```bash
     nvidia-smi       # should list your GPU(s) and driver version
     nvcc --version   # should print CUDA version
     ```

2. **cuDNN**  
   - Download and install cuDNN for your CUDA version from NVIDIA’s website.  
   - Verify headers/libs under `/usr/include` and `/usr/lib/x86_64-linux-gnu` (or your distro’s equivalent).

---

### Installation

```bash
# Basic installation (CPU-only)
pip install ManipulaPy

# With GPU support (CUDA 11.x)
pip install ManipulaPy[gpu-cuda11]

# With GPU support (CUDA 12.x)
pip install ManipulaPy[gpu-cuda12]

# Development installation (with dev extras)
git clone https://github.com/boelnasr/ManipulaPy.git
cd ManipulaPy
pip install -e .[dev]
````

---

### Post‐Install Check

After installation, confirm that ManipulaPy can see your GPU:

```bash
# Check that CUDA is available to ManipulaPy
python3 - <<EOF
import manipulapy
if manipulapy.cuda.is_available():
    print("✅ CUDA is available:", manipulapy.cuda.get_device_name())
else:
    raise RuntimeError("❌ CUDA not detected by ManipulaPy")
EOF
```

If you see the ✅ message with your GPU name, you’re all set! Otherwise, double‑check the CUDA Toolkit and cuDNN installation steps above. \`\`\`


### 30-Second Demo

```python
import numpy as np
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
from ManipulaPy.control import ManipulatorController

# Load robot model
try:
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
except ImportError:
    urdf_file = "path/to/your/robot.urdf"

# Initialize robot
urdf_processor = URDFToSerialManipulator(urdf_file)
robot = urdf_processor.serial_manipulator
dynamics = urdf_processor.dynamics

# Forward kinematics
joint_angles = np.array([0.1, 0.2, -0.3, -0.5, 0.2, 0.1])
end_effector_pose = robot.forward_kinematics(joint_angles)
print(f"End-effector position: {end_effector_pose[:3, 3]}")

# Trajectory planning
joint_limits = [(-np.pi, np.pi)] * 6
planner = OptimizedTrajectoryPlanning(robot, urdf_file, dynamics, joint_limits)

trajectory = planner.joint_trajectory(
    thetastart=np.zeros(6),
    thetaend=joint_angles,
    Tf=5.0, N=100, method=5
)

print(f"✅ Generated {trajectory['positions'].shape[0]} trajectory points")
```

---

## 📚 Core Modules

### 🔧 Kinematics & Dynamics

<details>
<summary><b>Forward & Inverse Kinematics</b></summary>

```python
# Forward kinematics
pose = robot.forward_kinematics(joint_angles, frame="space")

# Inverse kinematics with advanced solver
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.3, 0.4]

solution, success, iterations = robot.iterative_inverse_kinematics(
    T_desired=target_pose,
    thetalist0=joint_angles,
    eomg=1e-6, ev=1e-6,
    max_iterations=5000,
    plot_residuals=True
)
```

</details>

<details>
<summary><b>Dynamic Analysis</b></summary>

```python
from ManipulaPy.dynamics import ManipulatorDynamics

# Compute dynamics quantities
M = dynamics.mass_matrix(joint_angles)
C = dynamics.velocity_quadratic_forces(joint_angles, joint_velocities)
G = dynamics.gravity_forces(joint_angles, g=[0, 0, -9.81])

# Inverse dynamics: τ = M(q)q̈ + C(q,q̇) + G(q)
torques = dynamics.inverse_dynamics(
    joint_angles, joint_velocities, joint_accelerations,
    [0, 0, -9.81], np.zeros(6)
)
```

</details>

### 🛤️ Path Planning & Control

<details>
<summary><b>Advanced Trajectory Planning</b></summary>

```python
# GPU-accelerated trajectory planning
planner = OptimizedTrajectoryPlanning(
    robot, urdf_file, dynamics, joint_limits,
    use_cuda=True,  # Enable GPU acceleration
    cuda_threshold=200,  # Auto-switch threshold
    enable_profiling=True
)

# Joint space trajectory
trajectory = planner.joint_trajectory(
    thetastart=start_config,
    thetaend=end_config,
    Tf=5.0, N=1000, method=5  # Quintic time scaling
)

# Cartesian space trajectory
cartesian_traj = planner.cartesian_trajectory(
    Xstart=start_pose, Xend=end_pose,
    Tf=3.0, N=500, method=3  # Cubic time scaling
)

# Performance monitoring
stats = planner.get_performance_stats()
print(f"GPU usage: {stats['gpu_usage_percent']:.1f}%")
```

</details>

<details>
<summary><b>Advanced Control Systems</b></summary>

```python
from ManipulaPy.control import ManipulatorController

controller = ManipulatorController(dynamics)

# Auto-tuned PID control using Ziegler-Nichols
Ku, Tu = 50.0, 0.5  # Ultimate gain and period
Kp, Ki, Kd = controller.ziegler_nichols_tuning(Ku, Tu, kind="PID")

# Computed torque control
control_torque = controller.computed_torque_control(
    thetalistd=desired_positions,
    dthetalistd=desired_velocities,
    ddthetalistd=desired_accelerations,
    thetalist=current_positions,
    dthetalist=current_velocities,
    g=[0, 0, -9.81], dt=0.01,
    Kp=Kp, Ki=Ki, Kd=Kd
)

# Adaptive control
adaptive_torque = controller.adaptive_control(
    thetalist=current_positions,
    dthetalist=current_velocities,
    ddthetalist=desired_accelerations,
    g=[0, 0, -9.81], Ftip=np.zeros(6),
    measurement_error=position_error,
    adaptation_gain=0.1
)
```

</details>

### 🌐 Simulation & Visualization

<details>
<summary><b>Real-time PyBullet Simulation</b></summary>

```python
from ManipulaPy.sim import Simulation

# Create simulation environment
sim = Simulation(
    urdf_file_path=urdf_file,
    joint_limits=joint_limits,
    time_step=0.01,
    real_time_factor=1.0
)

# Initialize and run
sim.initialize_robot()
sim.initialize_planner_and_controller()
sim.add_joint_parameters()  # GUI sliders

# Execute trajectory
final_pose = sim.run_trajectory(trajectory["positions"])

# Manual control with collision detection
sim.manual_control()
```

</details>

<details>
<summary><b>Singularity & Workspace Analysis</b></summary>

```python
from ManipulaPy.singularity import Singularity

analyzer = Singularity(robot)

# Singularity detection
is_singular = analyzer.singularity_analysis(joint_angles)
condition_number = analyzer.condition_number(joint_angles)

# Manipulability ellipsoid
analyzer.manipulability_ellipsoid(joint_angles)

# Workspace visualization with GPU acceleration
analyzer.plot_workspace_monte_carlo(
    joint_limits=joint_limits,
    num_samples=10000
)
```

</details>

### 👁️ Vision & Perception

<details>
<summary><b>Computer Vision Pipeline</b></summary>

```python
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception

# Camera configuration
camera_config = {
    "name": "main_camera",
    "intrinsic_matrix": np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
    "translation": [0, 0, 1.5],
    "rotation": [0, -30, 0],  # degrees
    "fov": 60,
    "use_opencv": True,  # Real camera
    "device_index": 0
}

# Stereo vision setup
left_cam = {**camera_config, "translation": [-0.1, 0, 1.5]}
right_cam = {**camera_config, "translation": [0.1, 0, 1.5]}

vision = Vision(
    camera_configs=[camera_config],
    stereo_configs=(left_cam, right_cam)
)

# Object detection and clustering
perception = Perception(vision)
obstacles, labels = perception.detect_and_cluster_obstacles(
    depth_threshold=3.0,
    eps=0.1, min_samples=5
)

# 3D point cloud from stereo
if vision.stereo_enabled:
    left_img, _ = vision.capture_image(0)
    right_img, _ = vision.capture_image(1)
    point_cloud = vision.get_stereo_point_cloud(left_img, right_img)
```

</details>

---

## 📊 Performance Features

### GPU Acceleration

ManipulaPy includes highly optimized CUDA kernels for performance-critical operations:

```python
from ManipulaPy.cuda_kernels import check_cuda_availability

if check_cuda_availability():
    print("🚀 CUDA acceleration available!")
    
    # Automatic GPU/CPU switching based on problem size
    planner = OptimizedTrajectoryPlanning(
        robot, urdf_file, dynamics, joint_limits,
        use_cuda=None,  # Auto-detect
        cuda_threshold=200,  # Switch threshold
        memory_pool_size_mb=512  # GPU memory pool
    )
    
    # Batch processing for multiple trajectories
    batch_trajectories = planner.batch_joint_trajectory(
        thetastart_batch=start_configs,  # (batch_size, n_joints)
        thetaend_batch=end_configs,
        Tf=5.0, N=1000, method=5
    )
else:
    print("CPU mode - install GPU support for acceleration")
```

### Performance Monitoring

```python
# Benchmark different implementations
results = planner.benchmark_performance([
    {"N": 1000, "joints": 6, "name": "Medium"},
    {"N": 5000, "joints": 6, "name": "Large"},
    {"N": 1000, "joints": 12, "name": "Many joints"}
])

for name, result in results.items():
    print(f"{name}: {result['total_time']:.3f}s, GPU: {result['used_gpu']}")
```

---

## 📁 Examples & Tutorials

The `Examples/` directory contains comprehensive demonstrations organized into three levels:

### 🎯 Basic Examples (⭐)
Perfect for getting started with ManipulaPy fundamentals.

| Example | Description | Output |
|---------|-------------|--------|
| `kinematics_basic_demo.py` | Forward/inverse kinematics with visualization | Manipulability analysis plots |
| `dynamics_basic_demo.py` | Mass matrix, Coriolis forces, gravity compensation | Complete robot analysis |
| `control_basic_demo.py` | PID, computed torque, feedforward control | Control strategy comparison |
| `urdf_processing_basic_demo.py` | URDF to SerialManipulator conversion | Configuration space analysis |
| `visualization_basic_demo.py` | End-effector paths and workspace visualization | 3D trajectory plots |

### 🔧 Intermediate Examples (⭐⭐)
Advanced features and integrated systems.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `trajectory_planning_intermediate_demo.py` | Multi-segment trajectories and optimization | GPU acceleration, smoothing |
| `singularity_analysis_intermediate_demo.py` | Workspace analysis and singularity avoidance | Manipulability ellipsoids |
| `control_comparison_intermediate_demo.py` | Multiple control strategies benchmarking | Real-time monitoring |
| `perception_intermediate_demo.py` | Computer vision pipeline with clustering | YOLO detection, stereo vision |
| `simulation_intermediate_demo.py` | Complete PyBullet integration | Real-time physics simulation |

### 🚀 Advanced Examples (⭐⭐⭐)
Research-grade implementations and high-performance computing.

| Example | Description | Advanced Features |
|---------|-------------|-------------------|
| `gpu_acceleration_advanced_demo.py` | CUDA kernels and performance optimization | Memory efficiency analysis |
| `batch_processing_advanced_demo.py` | Large-scale trajectory generation | Batch scaling analysis |
| `collision_avoidance_advanced_demo.py` | Real-time obstacle avoidance | Potential field visualization |
| `optimal_control_advanced_demo.py` | Advanced control algorithms | Performance statistics |
| `stereo_vision_advanced_demo.py` | 3D perception and point cloud processing | Advanced perception analysis |
| `real_robot_integration_advanced_demo.py` | Hardware integration examples | Real-time simulation |

### 🏃‍♂️ Running Examples

```bash
cd Examples/

# Basic Examples - Start here!
cd basic_examples/
python kinematics_basic_demo.py
python dynamics_basic_demo.py
python control_basic_demo.py

# Intermediate Examples - Integrated systems
cd ../intermediate_examples/
python trajectory_planning_intermediate_demo.py
python perception_intermediate_demo.py --enable-yolo
python simulation_intermediate_demo.py --urdf simple_arm.urdf

# Advanced Examples - Research-grade
cd ../advanced_examples/
python gpu_acceleration_advanced_demo.py --benchmark
python batch_processing_advanced_demo.py --size 1000
python collision_avoidance_advanced_demo.py --visualize
```

### 📊 Example Outputs

The examples generate various outputs:
- **📈 Analysis Reports**: `.txt` files with detailed performance metrics
- **📊 Visualizations**: `.png` plots for trajectories, workspaces, and analysis
- **📝 Logs**: `.log` files for debugging and monitoring
- **🎯 Models**: Pre-trained YOLO models and URDF files

### 🎨 Generated Visualizations

Examples create rich visualizations including:
- **Trajectory Analysis**: Multi-segment paths and optimization results
- **Workspace Visualization**: 3D manipulability and reachability analysis  
- **Control Performance**: Real-time monitoring and comparison plots
- **Perception Results**: Object detection, clustering, and stereo vision
- **Performance Benchmarks**: GPU vs CPU timing and memory usage



### 🔍 Example Selection Guide

**New to ManipulaPy?** → Start with `basic_examples/kinematics_basic_demo.py`

**Need trajectory planning?** → Try `intermediate_examples/trajectory_planning_intermediate_demo.py`

**Working with vision?** → Check `intermediate_examples/perception_intermediate_demo.py`

**Performance optimization?** → Explore `advanced_examples/gpu_acceleration_advanced_demo.py`

**Research applications?** → Dive into `advanced_examples/batch_processing_advanced_demo.py`

---
## 🧪 Testing & Validation

### Test Suite

```bash
# Install test dependencies
pip install ManipulaPy[dev]

# Run all tests
python -m pytest tests/ -v --cov=ManipulaPy

# Test specific modules
python -m pytest tests/test_kinematics.py -v
python -m pytest tests/test_dynamics.py -v
python -m pytest tests/test_control.py -v

```
<!-- 
### Coverage Statistics

- **Kinematics**: >95% test coverage
- **Dynamics**: >90% test coverage  
- **Control**: >88% test coverage
- **Perception**: >85% test coverage
- **GPU Kernels**: >80% test coverage 
python -m pytest tests/test_cuda_kernels.py -v  # GPU tests
-->

---

## 🧪 Benchmarking & Validation

ManipulaPy includes a comprehensive benchmarking suite to validate performance and accuracy across different hardware configurations.

### Benchmark Suite

Located in the `Benchmark/` directory, the suite provides three key tools:

| Benchmark | Purpose | Use Case |
|-----------|---------|----------|
| `performance_benchmark.py` | Comprehensive performance analysis | Full system evaluation and optimization |
| `accuracy_benchmark.py` | Numerical precision validation | Algorithm correctness verification |
| `quick_benchmark.py` | Fast development testing | CI/CD integration and regression testing |

### Real Performance Results

**Latest benchmark on 16-core CPU, 31.1GB RAM, NVIDIA GPU (30 SMs):**

```bash
=== ManipulaPy Performance Benchmark Results ===
Hardware: 16-core CPU, 31.1GB RAM, NVIDIA GPU (30 SMs, 1024 threads/block)
Test Configuration: Large-scale problems (10K-100K trajectory points)

Overall Performance:
  Total Tests: 36 scenarios
  Success Rate: 91.7% (33/36) ✅
  Overall Speedup: 13.02× average acceleration
  CPU Mean Time: 6.88s → GPU Mean Time: 0.53s

🚀 EXCEPTIONAL PERFORMANCE HIGHLIGHTS:

Inverse Dynamics (CUDA Accelerated):
  Mean GPU Speedup: 3,624× (3.6K times faster!)
  Peak Performance: 5,563× speedup achieved
  Real-time Impact: 7s → 0.002s computation

Joint Trajectory Planning:
  Mean GPU Speedup: 2.29×
  Best Case: 7.96× speedup
  Large Problems: Consistent GPU acceleration

Cartesian Trajectories:
  Mean GPU Speedup: 1.02× (CPU competitive)
  Consistent Performance: ±0.04 variance
```

### Performance Recommendations

**🎯 OPTIMAL GPU USE CASES:**
- ✅ Inverse dynamics computation (**1000×-5000× speedup**)
- ✅ Large trajectory generation (>10K points)
- ✅ Batch processing multiple trajectories
- ✅ Real-time control applications

**⚠️ CPU-OPTIMAL SCENARIOS:**
- Small trajectories (<1K points)
- Cartesian space interpolation
- Single-shot computations
- Development and debugging

### Running Benchmarks

```bash
# Quick performance check (< 60 seconds)
cd Benchmark/
python quick_benchmark.py

# Comprehensive GPU vs CPU analysis
python performance_benchmark.py --gpu --plot --save-results

# Validate numerical accuracy
python accuracy_benchmark.py --tolerance 1e-8
```


## 📖 Documentation

### Online Documentation
- **[Complete API Reference](https://manipulapy.readthedocs.io/)**
- **[User Guide](https://manipulapy.readthedocs.io/en/latest/api/index.html)**
- **[API Reference](https://manipulapy.readthedocs.io/en/latest/theory.html)**
- **[GPU Programming Guide](https://manipulapy.readthedocs.io/en/latest/user_guide/CUDA_Kernels.html)**

### Quick Reference

```python
# Check installation and dependencies
import ManipulaPy
ManipulaPy.check_dependencies(verbose=True)

# Module overview
print(ManipulaPy.__version__)  # Current version
print(ManipulaPy.__all__)     # Available modules

# GPU capabilities
from ManipulaPy.cuda_kernels import get_gpu_properties
props = get_gpu_properties()
if props:
    print(f"GPU: {props['multiprocessor_count']} SMs")
```

---

## 🤝 Contributing

We love your input! Whether you’re reporting a bug, proposing a new feature, or improving our docs, here’s how to get started:

### 1. Report an Issue
Please open a GitHub Issue with:
- A descriptive title  
- Steps to reproduce  
- Expected vs. actual behavior  
- Any relevant logs or screenshots  

### 2. Submit a Pull Request
1. Fork this repository and create your branch:
   ```bash
   git clone https://github.com/<your-username>/ManipulaPy.git
   cd ManipulaPy
   git checkout -b feature/my-feature
   ```
2. Install and set up the development environment:
   ```bash
   pip install -e .[dev]
   pre-commit install     # to run formatters and linters
   ```
3. Make your changes, then run tests and quality checks:
   ```bash
   # Run the full test suite
   python -m pytest tests/ -v

   # Lint and format
   black ManipulaPy/
   flake8 ManipulaPy/
   mypy ManipulaPy/
   ```
4. Commit with clear, focused messages and push your branch:
   ```bash
   git add .
   git commit -m "Add awesome new feature"
   git push origin feature/my-feature
   ```
5. Open a Pull Request against `main` describing your changes.

### 3. Seek Support
- **Design questions:** [GitHub Discussions](https://github.com/boelnasr/ManipulaPy/discussions)  
- **Bug reports:** [GitHub Issues](https://github.com/boelnasr/ManipulaPy/issues)  
- **Email:** aboelnasr1997@gmail.com  

### 4. Code of Conduct
Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep this community welcoming.  


### Contribution Areas

- 🐛 **Bug Reports**: Issues and edge cases
- ✨ **New Features**: Algorithms and capabilities
- 📚 **Documentation**: Guides and examples
- 🚀 **Performance**: CUDA kernels and optimizations
- 🧪 **Testing**: Test coverage and validation
- 🎨 **Visualization**: Plotting and animation tools

### Guidelines

- Follow **PEP 8** style guidelines
- Add **comprehensive tests** for new features
- Update **documentation** for API changes
- Include **working examples** for new functionality
- Maintain **backward compatibility** when possible

---

## 📄 License & Citation

### License

ManipulaPy is licensed under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.

**Key Points:**
- ✅ **Free to use** for research and education
- ✅ **Modify and distribute** under same license
- ✅ **Commercial use** allowed under AGPL terms
- ⚠️ **Network services** must provide source code
- 📜 **See [LICENSE](LICENSE)** for complete terms

### Citation

If you use ManipulaPy in your research, please cite:

```bibtex
@software{manipulapy2025,
  title={ManipulaPy: A Comprehensive Python Package for Robotic Manipulator Analysis and Control},
  author={Mohamed Aboelnar},
  year={2025},
  url={https://github.com/boelnasr/ManipulaPy},
  version={1.1.0},
  license={AGPL-3.0-or-later},

}
```

### Dependencies

All dependencies are AGPL-3.0 compatible:
- **Core**: `numpy`, `scipy`, `matplotlib` (BSD)
- **Vision**: `opencv-python` (Apache 2.0), `ultralytics` (AGPL-3.0)
- **GPU**: `cupy` (MIT), `numba` (BSD)
- **Simulation**: `pybullet` (Zlib), `urchin` (MIT)

---

## 📞 Support & Community

### Getting Help

1. **📚 Documentation**: [manipulapy.readthedocs.io](https://manipulapy.readthedocs.io/)
2. **💡 Examples**: Check the `Examples/` directory
3. **🐛 Issues**: [GitHub Issues](https://github.com/boelnasr/ManipulaPy/issues)
4. **💬 Discussions**: [GitHub Discussions](https://github.com/boelnasr/ManipulaPy/discussions)
5. **📧 Contact**: [aboelnasr1997@gmail.com](mailto:aboelnasr1997@gmail.com)

### Community

- **🌟 Star** the project if you find it useful
- **🍴 Fork** to contribute improvements
- **📢 Share** with the robotics community
- **📝 Cite** in your academic work

### Contact Information

**Created and maintained by Mohamed Aboelnar**

- 📧 **Email**: [aboelnasr1997@gmail.com](mailto:aboelnasr1997@gmail.com)
- 🐙 **GitHub**: [@boelnasr](https://github.com/boelnasr)
- 🔗 **LinkedIn**: Connect for collaboration opportunities

---

## 🏆 Why Choose ManipulaPy?

<table>
<tr>
<td width="33%">

### 🔬 **For Researchers**
- Comprehensive algorithms with solid mathematical foundations
- Extensible modular design for new methods
- Well-documented with theoretical background
- Proper citation format for publications
- AGPL-3.0 license for open science

</td>
<td width="33%">

### 👩‍💻 **For Developers**
- High-performance GPU acceleration
- Clean, readable Python code
- Modular architecture
- Comprehensive test suite
- Active development and support

</td>
<td width="33%">

### 🏭 **For Industry**
- Production-ready with robust error handling
- Scalable for real-time applications
- Clear licensing for commercial use
- Professional documentation
- Regular updates and maintenance

</td>
</tr>
</table>

---

<div align="center">

**🤖 ManipulaPy v1.1.0: Professional robotics tools for the Python ecosystem**

[![GitHub stars](https://img.shields.io/github/stars/boelnasr/ManipulaPy?style=social)](https://github.com/boelnasr/ManipulaPy)
[![Downloads](https://img.shields.io/pypi/dm/ManipulaPy)](https://pypi.org/project/ManipulaPy/)

*Empowering robotics research and development with comprehensive, GPU-accelerated tools*

[⭐ Star on GitHub](https://github.com/boelnasr/ManipulaPy) • [📦 Install from PyPI](https://pypi.org/project/ManipulaPy/) • [📖 Read the Docs](https://manipulapy.readthedocs.io/)

</div>