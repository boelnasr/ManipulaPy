# ManipulaPy Examples

This directory contains comprehensive examples demonstrating the capabilities of the ManipulaPy robotics library. The examples are organized into three difficulty levels: basic, intermediate, and advanced, each showcasing different features and use cases.

## 📁 Directory Structure

```
Examples/
├── basic_examples/          # Foundational examples for beginners
├── intermediate_examples/   # More complex multi-module demonstrations
├── advanced_examples/       # Production-ready scenarios and optimizations
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

Before running the examples, ensure you have ManipulaPy installed with all dependencies:

```bash
# Basic installation
pip install ManipulaPy

# For GPU acceleration (CUDA 11.x)
pip install ManipulaPy[gpu-cuda11]

# For GPU acceleration (CUDA 12.x)
pip install ManipulaPy[gpu-cuda12]

# For complete feature set including vision
pip install ManipulaPy[complete]
```

### Required Additional Packages

Some examples require additional packages:

```bash
# For vision examples
pip install ultralytics opencv-python

# For advanced examples
pip install scikit-learn matplotlib seaborn

# For simulation examples
pip install pybullet urchin
```

## 📚 Example Categories

### 🟢 Basic Examples

Perfect for newcomers to robotics and ManipulaPy. These examples demonstrate core concepts with clear, well-commented code.

#### `kinematics_basic_demo.py`
- **Purpose**: Introduction to forward/inverse kinematics
- **Features**: 
  - Forward kinematics computation
  - Inverse kinematics with multiple solutions
  - Jacobian analysis
  - End-effector pose visualization
- **Output**: `complete_robot_analysis.png`, `workspace_visualization.png`

#### `dynamics_basic_demo.py`
- **Purpose**: Fundamental dynamics computations
- **Features**:
  - Mass matrix computation
  - Coriolis and centrifugal forces
  - Gravity compensation
  - Inverse/forward dynamics
- **Output**: `plots/` directory with comprehensive analysis

#### `control_basic_demo.py`
- **Purpose**: Basic control algorithms
- **Features**:
  - PID control implementation
  - Computed torque control
  - Feedforward control
  - Controller tuning with Ziegler-Nichols
- **Output**: Multiple control strategy comparison plots

#### `urdf_processing_basic_demo.py`
- **Purpose**: URDF file processing and robot model generation
- **Features**:
  - URDF parsing and validation
  - SerialManipulator object creation
  - Joint limit extraction
  - Robot visualization

#### `visualization_basic_demo.py`
- **Purpose**: Visualization capabilities overview
- **Features**:
  - 3D robot visualization
  - Trajectory plotting
  - Workspace analysis
  - Manipulability ellipsoids

### 🟡 Intermediate Examples

More sophisticated examples combining multiple modules and real-world scenarios.

#### `trajectory_planning_intermediate_demo.py`
- **Purpose**: Advanced trajectory planning techniques
- **Features**:
  - Multiple trajectory generation methods
  - Performance benchmarking (CPU vs GPU)
  - Collision avoidance integration
  - Real-time trajectory optimization
- **Output**: Comprehensive trajectory analysis plots

#### `singularity_analysis_intermediate_demo.py`
- **Purpose**: Singularity detection and avoidance
- **Features**:
  - Condition number analysis
  - Manipulability ellipsoid visualization
  - Singularity avoidance strategies
  - Workspace dexterity optimization
- **Output**: `singularity_*.png` analysis plots

#### `control_comparison_intermediate_demo.py`
- **Purpose**: Comparative analysis of control strategies
- **Features**:
  - Multiple controller implementations
  - Performance metrics comparison
  - Real-time monitoring capabilities
  - Adaptive parameter tuning
- **Output**: `control_comparison.png`, `performance_benchmark.png`

#### `perception_intermediate_demo.py`
- **Purpose**: Computer vision and perception integration
- **Features**:
  - YOLO-based object detection
  - Stereo vision processing
  - 3D point cloud generation
  - Obstacle detection and clustering
- **Output**: Extensive vision analysis with timestamped images
- **Log**: `perception_demo.log`, `perception_demo_report.txt`

#### `simulation_intermediate_demo.py`
- **Purpose**: PyBullet simulation integration
- **Features**:
  - Real-time robot simulation
  - Physics-based dynamics
  - Interactive control interfaces
  - Trajectory execution in simulation
- **Output**: `simulation_demo.log`, `simulation_demo_report.txt`

### 🔴 Advanced Examples

Production-ready examples showcasing optimization, scaling, and real-world integration.

#### `gpu_acceleration_advanced_demo.py`
- **Purpose**: GPU acceleration and performance optimization
- **Features**:
  - CUDA kernel utilization
  - Memory management optimization
  - Performance profiling and analysis
  - Batch processing capabilities
- **Output**: `gpu_performance_analysis.png`, `memory_efficiency_analysis.png`

#### `batch_processing_advanced_demo.py`
- **Purpose**: Large-scale batch trajectory processing
- **Features**:
  - Multi-trajectory generation
  - Statistical analysis
  - Scaling behavior evaluation
  - Resource utilization monitoring
- **Output**: `batch_scaling_analysis.png`, `batch_statistics_analysis_100.png`

#### `collision_avoidance_advanced_demo.py`
- **Purpose**: Advanced collision avoidance strategies
- **Features**:
  - Potential field path planning
  - Real-time obstacle detection
  - Dynamic environment adaptation
  - Safety-critical applications
- **Output**: `potential_field_visualization.png`, `environment_3d.png`

#### `optimal_control_advanced_demo.py`
- **Purpose**: Optimal control and trajectory optimization
- **Features**:
  - Cost function optimization
  - Constrained trajectory planning
  - Multi-objective optimization
  - Performance comparison analysis
- **Output**: `trajectory_comparison_scenario_0.png`

#### `stereo_vision_advanced_demo.py`
- **Purpose**: Advanced stereo vision processing
- **Features**:
  - Stereo camera calibration
  - 3D reconstruction
  - Real-time depth estimation
  - Point cloud processing

#### `real_robot_integration_advanced_demo.py`
- **Purpose**: Real robot hardware integration
- **Features**:
  - Hardware abstraction layer
  - Real-time control implementation
  - Safety monitoring systems
  - Production deployment strategies
- **Output**: `real_time_simulation.png`

## 📊 Output Files and Analysis

### Generated Plots and Visualizations

The examples generate various types of analytical outputs:

- **Trajectory Analysis**: Path planning, velocity profiles, acceleration curves
- **Performance Metrics**: CPU vs GPU timing, memory usage, scaling behavior
- **Robot Analysis**: Workspace visualization, manipulability maps, singularity detection
- **Control Analysis**: PID tuning, controller comparison, stability analysis
- **Vision Processing**: Object detection results, stereo reconstruction, depth maps

### Log Files and Reports

Several examples generate detailed logs and reports:

- `perception_demo_report.txt`: Comprehensive vision system analysis
- `simulation_demo_report.txt`: Simulation performance and behavior analysis
- `batch_processing_analysis_report.txt`: Statistical analysis of batch operations
- `collision_avoidance_analysis_report.txt`: Safety and path planning analysis

### Timestamped Outputs

Vision and perception examples generate timestamped files with format:
`YYYYMMDD_HHMMSS_category_description.png`

This allows tracking of processing results over multiple runs.

## 🛠 Running the Examples

### Basic Usage

```bash
cd Examples/basic_examples
python kinematics_basic_demo.py
```

### With GPU Acceleration

```bash
# Set environment variables for optimal GPU performance
export MANIPULAPY_FASTMATH=1
export MANIPULAPY_USE_FP16=0  # Use FP32 for better precision

cd Examples/advanced_examples
python gpu_acceleration_advanced_demo.py
```

### For Vision Examples

Ensure you have the YOLO model file:

```bash
cd Examples/intermediate_examples
# The yolov8m.pt file should be present
python perception_intermediate_demo.py
```

## 🔧 Customization and Extension

### Modifying Examples

All examples are designed to be easily customizable:

1. **Robot Models**: Change URDF files in robot configuration sections
2. **Parameters**: Modify control gains, trajectory parameters, or vision settings
3. **Output Paths**: Customize where plots and logs are saved
4. **Performance Settings**: Adjust GPU/CPU thresholds and memory limits

### Creating New Examples

Use the existing examples as templates:

1. Start with a basic example for the core functionality
2. Add intermediate features for multi-module integration
3. Implement advanced optimizations for production use

### Integration with External Tools

Examples demonstrate integration with:

- **PyBullet**: Physics simulation and visualization
- **OpenCV**: Computer vision and image processing
- **YOLO**: Object detection and recognition
- **Matplotlib/Seaborn**: Scientific visualization
- **NumPy/SciPy**: Numerical computation

## 📈 Performance Considerations

### Hardware Requirements

- **Basic Examples**: Any modern CPU, 4GB+ RAM
- **Intermediate Examples**: Multi-core CPU, 8GB+ RAM, optional GPU
- **Advanced Examples**: High-performance CPU, 16GB+ RAM, CUDA-capable GPU recommended

### GPU Acceleration

When GPU acceleration is available:

- Automatic CPU/GPU selection based on problem size
- Memory pooling for efficient resource utilization
- Adaptive batch sizing for optimal performance

### Memory Management

Examples demonstrate:

- Efficient memory allocation and deallocation
- Batch processing for large datasets
- Resource cleanup and error handling

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: Check GPU drivers and CUDA toolkit installation
3. **Memory Issues**: Reduce batch sizes or enable memory pooling
4. **Visualization Issues**: Ensure display is available for matplotlib

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Debugging

Use the built-in profiling tools:

```python
from ManipulaPy.cuda_kernels import profile_start, profile_stop

profile_start()
# Your code here
profile_stop()
```

## 📖 Further Reading

### Documentation

- [ManipulaPy API Documentation](https://github.com/your-repo/ManipulaPy/docs)
- [User Guide](https://github.com/your-repo/ManipulaPy/wiki)
- [Tutorial Series](https://github.com/your-repo/ManipulaPy/tutorials)

### Research Papers

The algorithms and methods demonstrated in these examples are based on:

- Modern Robotics: Mechanics, Planning, and Control (Lynch & Park)
- Robotics: Modelling, Planning and Control (Siciliano et al.)
- Computer Vision: Algorithms and Applications (Szeliski)

### Community

- [GitHub Issues](https://github.com/your-repo/ManipulaPy/issues)
- [Discussion Forum](https://github.com/your-repo/ManipulaPy/discussions)
- [Contributing Guidelines](https://github.com/your-repo/ManipulaPy/CONTRIBUTING.md)

## 📄 License

These examples are provided under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later), same as the main ManipulaPy library.

Copyright (c) 2025 Mohamed Aboelnar

---

🤖 **Happy Robotics Programming!** 🤖

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-repo/ManipulaPy).