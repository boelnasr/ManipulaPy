#!/bin/bash

# Script to create index.rst with proper User Guide tabs
# Run this from: /path/to/ManipulaPy/docs/source/

echo "Creating ManipulaPy index.rst with User Guide tabs..."
echo "Directory: $(pwd)"

# Check if we're in the right directory
if [ ! -d "_static" ] && [ ! -d "api" ]; then
    echo "❌ Please run this script from the docs/source/ directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /path/to/ManipulaPy/docs/source/"
    exit 1
fi

echo "✅ Running from docs/source directory: $(pwd)"

# Create the main index.rst with proper tabbed structure
cat > index.rst << 'EOL'
ManipulaPy Documentation
========================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-AGPL--3.0-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0
   :alt: License

.. image:: https://img.shields.io/badge/version-0.2.0-blue.svg
   :alt: Version

.. image:: https://img.shields.io/badge/CUDA-Accelerated-brightgreen.svg
   :alt: CUDA Support

**ManipulaPy** is a comprehensive Python library for robotic manipulator analysis, simulation, and control. 
It provides high-performance tools for kinematics, dynamics, trajectory planning, computer vision, and 
perception for robotic applications with optional CUDA acceleration.

🚀 Key Features
---------------

* **Kinematics**: Forward and inverse kinematics with neural network solutions
* **Dynamics**: Mass matrix computation, inverse/forward dynamics with caching
* **Trajectory Planning**: CUDA-accelerated trajectory generation with collision avoidance
* **Control**: PID, computed torque, adaptive, and robust control algorithms
* **Vision**: Monocular/stereo camera support with PyBullet integration
* **Perception**: YOLO object detection, 3D point cloud processing, clustering
* **Simulation**: Real-time PyBullet integration with visualization
* **CUDA Acceleration**: GPU-accelerated computations for high performance

🏃 Quick Start
--------------

.. code-block:: bash

   # Install ManipulaPy
   pip install manipulapy[all]

.. code-block:: python

   # Basic usage
   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   # Load robot
   processor = URDFToSerialManipulator("robot.urdf")
   robot = processor.serial_manipulator

   # Forward kinematics
   joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
   pose = robot.forward_kinematics(joint_angles)

📚 Documentation
----------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/installation
   tutorials/quickstart
   tutorials/basic_concepts

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   tutorials/kinematics_guide
   tutorials/dynamics_guide
   tutorials/trajectory_planning_guide
   tutorials/control_guide
   tutorials/urdf_processing_guide

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   tutorials/vision_perception_guide
   tutorials/simulation_guide
   tutorials/cuda_acceleration
   tutorials/collision_avoidance

.. toctree::
   :maxdepth: 1
   :caption: Examples & Use Cases

   tutorials/robot_control_example
   tutorials/trajectory_optimization
   tutorials/multi_robot_systems

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/kinematics
   api/dynamics
   api/path_planning
   api/control
   api/vision
   api/perception
   api/simulation
   api/urdf_processor
   api/utils
   api/cuda_kernels
   api/potential_field
   api/singularity
   api/transformations

.. toctree::
   :maxdepth: 1
   :caption: Reference

   tutorials/troubleshooting
   tutorials/performance_tips
   tutorials/best_practices

🏗️ Module Overview
------------------

**Core Modules**

* :mod:`ManipulaPy.kinematics` - Forward and inverse kinematics (:doc:`tutorials/kinematics_guide`)
* :mod:`ManipulaPy.dynamics` - Robot dynamics computations (:doc:`tutorials/dynamics_guide`)
* :mod:`ManipulaPy.path_planning` - Trajectory generation (:doc:`tutorials/trajectory_planning_guide`)
* :mod:`ManipulaPy.control` - Control algorithms (:doc:`tutorials/control_guide`)

**Vision and Perception**

* :mod:`ManipulaPy.vision` - Camera support (:doc:`tutorials/vision_perception_guide`)
* :mod:`ManipulaPy.perception` - Object detection (:doc:`tutorials/vision_perception_guide`)

**Simulation and Processing**

* :mod:`ManipulaPy.sim` - PyBullet simulation (:doc:`tutorials/simulation_guide`)
* :mod:`ManipulaPy.urdf_processor` - URDF processing (:doc:`tutorials/urdf_processing_guide`)

🎯 Popular Learning Paths
--------------------------

**🆕 New Users**
   :doc:`tutorials/installation` → :doc:`tutorials/quickstart` → :doc:`tutorials/basic_concepts`

**🤖 Robot Kinematics**
   :doc:`tutorials/kinematics_guide` → :doc:`tutorials/dynamics_guide`

**🛤️ Motion Planning**
   :doc:`tutorials/trajectory_planning_guide` → :doc:`tutorials/collision_avoidance`

**🎮 Robot Control**
   :doc:`tutorials/control_guide` → :doc:`tutorials/robot_control_example`

**👁️ Computer Vision**
   :doc:`tutorials/vision_perception_guide` → :doc:`tutorials/simulation_guide`

**⚡ Performance**
   :doc:`tutorials/cuda_acceleration` → :doc:`tutorials/performance_tips`

📄 License
----------

ManipulaPy is released under the **AGPL-3.0 License**:

* ✅ **Open Source**: Source code freely available
* 🔄 **Copyleft**: Derivative works must be open source
* 🌐 **Network Use**: Modified network services must provide source
* 💼 **Commercial Use**: Permitted with AGPL compliance

📊 Indices and Tables
---------------------

* :ref:`genindex` - Complete index of all functions and classes
* :ref:`modindex` - Module index for quick navigation  
* :ref:`search` - Search the documentation
EOL

echo ""
echo "✅ index.rst with User Guide tabs created successfully!"
echo ""
echo "🏷️  Navigation structure created:"
echo "   📖 Getting Started (3 pages)"
echo "   📚 User Guides (5 individual guide pages)" 
echo "   🚀 Advanced Features (4 pages)"
echo "   💡 Examples & Use Cases (3 pages)"
echo "   📖 API Reference (13 module pages)"
echo "   📋 Reference (3 pages)"
echo ""
echo "📋 Next steps:"
echo "   1. Build: cd .. && make clean && make html"
echo "   2. View: open build/html/index.html"
