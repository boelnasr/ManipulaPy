# ManipulaPy

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ManipulaPy)](https://pypi.org/project/ManipulaPy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![CI](https://github.com/boelnasr/ManipulaPy/actions/workflows/test.yml/badge.svg?branch=main)
![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)
[![codecov](https://codecov.io/gh/boelnasr/ManipulaPy/branch/main/graph/badge.svg)](https://codecov.io/gh/boelnasr/ManipulaPy)
[![status](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg)](https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa)

**A comprehensive, GPU-accelerated Python package for robotic manipulator analysis, simulation, planning, control, and perception.**

[Quick Start](#quick-start) • [Documentation](https://manipulapy.readthedocs.io/en/latest/index.html) • [Examples](#examples) • [Installation](#installation) • [Contributing](#contributing)

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

## 📋 Feature Availability Matrix

ManipulaPy automatically enables features based on available dependencies. Here's what you can expect:

### Core Features (Always Available)

| Feature | CPU Performance | Dependencies | Notes |
|---------|----------------|--------------|-------|
| **Kinematics** | Excellent | numpy, scipy | Forward/inverse kinematics, Jacobians |
| **Basic Dynamics** | Good | numpy, scipy | Mass matrix, Coriolis, gravity |
| **Control Systems** | Excellent | numpy, scipy | PID, computed torque, adaptive |
| **URDF Processing** | Fast | pybullet, urchin | Robot model conversion |
| **Small Trajectories** | Good | numba | N < 1000 points, auto-optimized |

### GPU-Accelerated Features (Optional)

| Feature | CPU vs GPU | Requirements | Speedup |
|---------|------------|--------------|---------|
| **Large Trajectories** | 40x+ faster | CUDA, cupy | N > 1000 points |
| **Batch Processing** | 20x+ faster | CUDA, cupy | Multiple trajectories |
| **Inverse Dynamics** | 100x+ faster | CUDA, cupy | Large datasets |
| **Workspace Analysis** | 10x+ faster | CUDA, cupy | Monte Carlo sampling |

### Vision Features (Requires System Dependencies)

| Feature | Requirements | Common Issues | Solutions |
|---------|-------------|---------------|-----------|
| **Camera Capture** | OpenCV, libGL.so.1 | ImportError: libGL.so.1 | `apt install libgl1-mesa-glx` |
| **Object Detection** | ultralytics, internet | YOLO download fails | Check network, manual download |
| **Stereo Vision** | OpenCV, calibration | Poor depth quality | Camera calibration required |
| **3D Point Clouds** | OpenCV, numpy | Memory issues | Reduce point cloud density |

### Installation Check Commands

```python
import ManipulaPy

# Quick check - shows ✅/❌ for each feature
ManipulaPy.check_dependencies()

# Detailed system information
ManipulaPy.print_system_info()

# Get missing dependencies install commands
print(ManipulaPy.get_installation_command())

# Check specific feature
try:
    ManipulaPy.require_feature('cuda')
    print("GPU acceleration ready!")
except ImportError as e:
    print(f"GPU not available: {e}")
```

---

## <a id="quick-start"></a>🚀 Quick Start

### Installation
Before installing ManipulaPy, make sure your system has:

1. **NVIDIA Drivers & CUDA Toolkit**  
   - `nvcc` on your `PATH` (e.g. via `sudo apt install nvidia-cuda-toolkit` or the [official NVIDIA CUDA installer](https://developer.nvidia.com/cuda-downloads)).  
   - Verify with:
     ```bash
     nvidia-smi       # should list your GPU(s) and driver version
     nvcc --version   # should print CUDA version
     ```

2. **cuDNN**  
   - Download and install cuDNN for your CUDA version from [NVIDIA's cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/).  
   - Verify headers/libs under `/usr/include` and `/usr/lib/x86_64-linux-gnu` (or your distro’s equivalent).

---
ManipulaPy attempts to install all dependencies by default for the best user experience. Missing dependencies are handled gracefully - the package will still work with available features.

```bash
# One command installs everything (recommended)
pip install ManipulaPy
```

**What gets installed automatically:**

- ✅ **Core robotics** (always): kinematics, dynamics, control, basic trajectory planning
- 🚀 **GPU acceleration** (if CUDA available): 40x+ speedups for large problems (N > 1000)
- 👁️ **Vision features** (if system supports): camera capture, object detection, stereo vision
- 🎮 **Simulation** (if compatible): PyBullet physics simulation and visualization

### Check Your Installation

After installation, verify which features are available:

```python
import ManipulaPy

# Quick feature availability check
ManipulaPy.check_dependencies()

# Detailed system information
ManipulaPy.print_system_info()
```

### Alternative Installation Options

```bash
# Minimal installation (core features only)
pip install ManipulaPy[minimal]

# Specific CUDA version if auto-detection fails
pip install ManipulaPy[gpu-cuda12]  # For CUDA 12.x

# Headless environments (CI/Docker)
pip install ManipulaPy[vision-headless]

# Development environment
pip install ManipulaPy[dev]
```

### Troubleshooting Common Issues

**🚀 GPU Acceleration Not Working:**
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA toolkit
nvcc --version

# Install specific CUDA version if needed
pip install cupy-cuda11x  # or cupy-cuda12x
```

**👁️ Vision Features Not Working:**
```bash
# Ubuntu/Debian - fix libGL.so.1 error
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL libglib2.0

# Test OpenCV
python -c "import cv2; print('OpenCV OK')"
```

**🔧 Verify Installation:**
```python
# Test core functionality (always works)
from ManipulaPy.kinematics import SerialManipulator
print("✅ Core features working")

# Test GPU acceleration (if available)  
try:
    from ManipulaPy.cuda_kernels import check_cuda_availability
    if check_cuda_availability():
        print("🚀 GPU acceleration available")
    else:
        print("⚠️ GPU acceleration not available")
except ImportError:
    print("⚠️ GPU acceleration not installed")

# Test vision (if available)
try:
    from ManipulaPy.vision import Vision
    print("👁️ Vision features available")
except ImportError:
    print("⚠️ Vision features not available")
```

### What Works Without Additional Setup

**✅ Always Available (CPU-only):**

- Forward/inverse kinematics and Jacobians
- PID, computed torque, adaptive, and robust control
- URDF processing and robot model conversion  
- PyBullet simulation and visualization
- Small trajectory planning (N < 1000 points)
- Singularity analysis and workspace computation

**🚀 GPU-Accelerated (optional 40x+ speedup):**

- Large trajectory planning (N > 1000 points)
- Batch processing multiple trajectories
- Monte Carlo workspace analysis
- Inverse dynamics for long trajectories

**👁️ Requires System Dependencies:**

- Vision features: OpenCV, system graphics libraries (libGL.so.1)
- Object detection: YOLO models (auto-downloaded on first use)
- Stereo processing: Camera calibration data

---

### 30-Second Demo

```python
import numpy as np
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

# Load robot model (works with any URDF)
try:
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
except ImportError:
    urdf_file = "path/to/your/robot.urdf"

# Initialize robot
urdf_processor = URDFToSerialManipulator(urdf_file)
robot = urdf_processor.serial_manipulator
dynamics = urdf_processor.dynamics

# Forward kinematics (always available)
joint_angles = np.array([0.1, 0.2, -0.3, -0.5, 0.2, 0.1])
end_effector_pose = robot.forward_kinematics(joint_angles)
print(f"End-effector position: {end_effector_pose[:3, 3]}")

# GPU-accelerated trajectory planning (40x+ faster if GPU available)
joint_limits = [(-np.pi, np.pi)] * 6
planner = OptimizedTrajectoryPlanning(robot, urdf_file, dynamics, joint_limits)

trajectory = planner.joint_trajectory(
    thetastart=np.zeros(6),
    thetaend=joint_angles,
    Tf=5.0, N=1000, method=5  # Quintic time scaling
)

print(f"✅ Generated {trajectory['positions'].shape[0]} trajectory points")

# Check what features are available
import ManipulaPy
features = ManipulaPy.get_available_features()
print(f"Available features: {', '.join(features)}")

# Get performance stats
stats = planner.get_performance_stats()
if stats['gpu_calls'] > 0:
    speedup = stats.get('speedup_achieved', 0)
    print(f"🚀 GPU acceleration achieved {speedup:.1f}x speedup!")
else:
    print("🖥️ Using CPU computation")
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

The `Examples/` directory contains comprehensive demonstrations that work with different feature combinations:

### 🎯 Basic Examples (⭐) - CPU Only

These examples work immediately after `pip install ManipulaPy` with no additional setup required.

| Example | Description | Requirements | Output |
|---------|-------------|--------------|--------|
| `kinematics_basic_demo.py` | Forward/inverse kinematics | Core only | Manipulability plots |
| `dynamics_basic_demo.py` | Mass matrix, forces | Core only | Robot analysis |
| `control_basic_demo.py` | PID, computed torque | Core only | Control comparison |
| `urdf_processing_basic_demo.py` | URDF conversion | Core + PyBullet | Config analysis |
| `small_trajectory_demo.py` | CPU trajectory planning | Core only | Path visualization |

### 🔧 Intermediate Examples (⭐⭐) - Optional GPU

These examples automatically use GPU acceleration if available, gracefully fall back to CPU.

| Example | Description | Auto-Detects | Performance Boost |
|---------|-------------|--------------|-------------------|
| `trajectory_planning_intermediate_demo.py` | Large-scale trajectories | GPU available | 40x+ if N>1000 |
| `batch_processing_intermediate_demo.py` | Multiple trajectory generation | GPU available | 20x+ for batches |
| `workspace_analysis_intermediate_demo.py` | Monte Carlo workspace | GPU available | 10x+ for sampling |
| `dynamics_comparison_intermediate_demo.py` | CPU vs GPU dynamics | GPU available | 100x+ for large datasets |

### 🚀 Advanced Examples (⭐⭐⭐) - Full Features

These examples demonstrate complete integration with all available features.

| Example | Description | Requirements | Notes |
|---------|-------------|--------------|-------|
| `perception_advanced_demo.py` | Vision + planning | OpenCV, YOLO | Auto-downloads models |
| `stereo_vision_advanced_demo.py` | 3D perception | OpenCV, calibration | Requires camera setup |
| `real_robot_integration_advanced_demo.py` | Hardware control | All features | Hardware-dependent |
| `performance_optimization_advanced_demo.py` | Benchmarking suite | GPU recommended | Comprehensive analysis |

### 🏃‍♂️ Running Examples with Feature Detection

```bash
cd Examples/

# Basic examples - always work
cd basic_examples/
python kinematics_basic_demo.py  # ✅ Always works
python dynamics_basic_demo.py    # ✅ Always works

# Intermediate examples - auto-detect features
cd ../intermediate_examples/
python trajectory_planning_intermediate_demo.py  # 🚀 GPU if available
python batch_processing_intermediate_demo.py --size 1000  # 📊 Scales with hardware

# Advanced examples - check requirements first
cd ../advanced_examples/
python -c "import ManipulaPy; ManipulaPy.check_dependencies()"  # Check first
python perception_advanced_demo.py --enable-yolo  # 👁️ Needs vision
python stereo_vision_advanced_demo.py --camera-pair 0,1  # 📷 Needs cameras
```

### 📊 Example Output Management

Examples automatically adapt their output based on available features:

```python
# Example auto-adaptation pattern
def run_trajectory_example():
    import ManipulaPy
    
    # Check what's available
    features = ManipulaPy.get_available_features()
    
    if 'cuda' in features:
        print("🚀 Using GPU acceleration for large trajectories")
        N = 10000  # Large problem size
    else:
        print("🖥️ Using CPU computation for moderate trajectories")
        N = 1000   # Smaller problem size
    
    if 'vision' in features:
        print("👁️ Including vision-based obstacle detection")
        enable_vision = True
    else:
        print("⚠️ Vision features not available, using pre-defined obstacles")
        enable_vision = False
    
    # Run example with appropriate settings...
```

### 🎯 Example Selection Guide

**New to ManipulaPy?**
```bash
# Start here - guaranteed to work
python basic_examples/kinematics_basic_demo.py
```

**Have a GPU?**
```bash
# Check GPU first
python -c "from ManipulaPy.cuda_kernels import check_cuda_availability; print('GPU:', check_cuda_availability())"

# If True, try these for massive speedups
python intermediate_examples/trajectory_planning_intermediate_demo.py --large
python advanced_examples/performance_optimization_advanced_demo.py
```

**Working with cameras?**
```bash
# Check vision first
python -c "import ManipulaPy; print('Vision:', 'vision' in ManipulaPy.get_available_features())"

# If True, try perception examples
python intermediate_examples/perception_intermediate_demo.py
python advanced_examples/stereo_vision_advanced_demo.py
```

**Need maximum performance?**
```bash
# Full system check
python -c "import ManipulaPy; ManipulaPy.check_dependencies(verbose=True)"

# Run comprehensive benchmarks
python advanced_examples/performance_optimization_advanced_demo.py --full-benchmark
```

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
python -m pytest tests/test_cuda_kernels.py -v  # GPU tests
```

### ✅ High-Coverage Modules

| Module              | Coverage | Notes                             |
| ------------------- | -------- | --------------------------------- |
| `kinematics.py`     | **98%**  | Excellent — near full coverage    |
| `dynamics.py`       | **100%** | Fully tested                      |
| `perception.py`     | **92%**  | Very solid coverage               |
| `vision.py`         | **83%**  | Good; some PyBullet paths skipped |
| `urdf_processor.py` | **81%**  | Strong test coverage              |

---

### ⚠️ Needs More Testing

| Module           | Coverage | Notes                                                    |
| ---------------- | -------- | -------------------------------------------------------- |
| `control.py`     | **81%**  | Many skipped due to CuPy mock — test with GPU to improve |
| `sim.py`         | **77%**  | Manual control & GUI parts partially tested              |
| `singularity.py` | **64%**  | Workspace plots & CUDA sampling untested                 |
| `utils.py`       | **61%**  | Some math utils & decorators untested                    |

---

### 🚨 Low/No Coverage

| Module               | Coverage | Notes                                                 |
| -------------------- | -------- | ----------------------------------------------------- |
| `path_planning.py`   | **39%**  | Large gaps in CUDA-accelerated and plotting logic     |
| `cuda_kernels.py`    | **16%**  | Most tests skipped — `NUMBA_DISABLE_CUDA=1`           |
| `transformations.py` | **0%**   | Not tested at all — consider adding basic SE(3) tests |

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

---

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

We love your input! Whether you're reporting a bug, proposing a new feature, or improving our docs, here's how to get started:

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
  author={Mohamed Aboelnasr},
  year={2025},
  url={https://github.com/boelnasr/ManipulaPy},
  version={1.1.3},
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

**Created and maintained by Mohamed Aboelnasr**

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

**🤖 ManipulaPy v1.1.3: Professional robotics tools for the Python ecosystem**

[![GitHub stars](https://img.shields.io/github/stars/boelnasr/ManipulaPy?style=social)](https://github.com/boelnasr/ManipulaPy)
[![PyPI Downloads](https://static.pepy.tech/badge/manipulapy)](https://pepy.tech/projects/manipulapy)

*Empowering robotics research and development with comprehensive, GPU-accelerated tools*

[⭐ Star on GitHub](https://github.com/boelnasr/ManipulaPy) • [📦 Install from PyPI](https://pypi.org/project/ManipulaPy/) • [📖 Read the Docs](https://manipulapy.readthedocs.io/)

</div>