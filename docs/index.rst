ManipulaPy Documentation
========================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/manipulapy/badge/?version=latest
   :target: https://manipulapy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

ManipulaPy is a comprehensive Python library for robotics manipulation, offering high-performance tools for kinematics, dynamics, trajectory planning, and control of robotic manipulators. Built with CUDA acceleration and modern robotics algorithms, ManipulaPy enables researchers and engineers to develop sophisticated robotic applications with ease.

Key Features
------------

🚀 **High Performance**
   - CUDA-accelerated computations for real-time applications
   - Optimized algorithms for large-scale robotic simulations
   - Efficient memory management and parallel processing

🤖 **Comprehensive Robotics Suite**
   - Forward and inverse kinematics with multiple solvers
   - Complete dynamics modeling and simulation
   - Advanced trajectory planning with collision avoidance
   - Multiple control strategies (PID, computed torque, adaptive)

🎯 **Production Ready**
   - URDF integration for real robot deployment
   - PyBullet simulation environment support
   - Computer vision and perception modules
   - Extensive testing and validation

🔧 **Developer Friendly**
   - Clean, intuitive API design
   - Comprehensive documentation and examples
   - Modular architecture for easy extension
   - Type hints and extensive error handling

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install manipulapy

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from ManipulaPy.kinematics import SerialManipulator
   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   # Load robot from URDF
   urdf_processor = URDFToSerialManipulator("robot.urdf")
   robot = urdf_processor.serial_manipulator

   # Forward kinematics
   joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
   end_effector_pose = robot.forward_kinematics(joint_angles)

   # Inverse kinematics
   target_pose = np.eye(4)
   target_pose[:3, 3] = [0.5, 0.3, 0.7]  # Target position
   solution, success, iterations = robot.iterative_inverse_kinematics(
       target_pose, joint_angles
   )

   print(f"IK Solution found: {success}")
   print(f"Joint angles: {solution}")

What's in the Docs
------------------

This documentation provides comprehensive guides and references for using ManipulaPy:

- **Getting Started**: Installation, basic concepts, and first examples
- **User Guide**: Detailed tutorials covering all major functionality
- **API Reference**: Complete documentation of all classes and functions
- **Examples**: Real-world applications and case studies
- **Contributing**: Guidelines for contributing to the project

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/kinematics
   user_guide/dynamics
   user_guide/trajectory_planning
   user_guide/control
   user_guide/simulation
   user_guide/vision_perception
   user_guide/urdf_processing

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_manipulation
   examples/trajectory_optimization
   examples/control_systems
   examples/simulation_environments
   examples/computer_vision
   examples/jupyter_notebooks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/kinematics
   api/dynamics
   api/path_planning
   api/control
   api/simulation
   api/vision
   api/perception
   api/urdf_processor
   api/utils
   api/cuda_kernels

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   testing
   performance
   roadmap

.. toctree::
   :maxdepth: 1
   :caption: Community

   changelog
   license
   citing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Community and Support
=====================

- **GitHub Repository**: `https://github.com/moabdelgaber/ManipulaPy <https://github.com/moabdelgaber/ManipulaPy>`_
- **Issue Tracker**: `GitHub Issues <https://github.com/moabdelgaber/ManipulaPy/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/moabdelgaber/ManipulaPy/discussions>`_

License
=======

ManipulaPy is released under the MIT License. See the :doc:`license` page for details.

Citation
========

If you use ManipulaPy in your research, please cite our work:

.. code-block:: bibtex

   @article{manipulapy2024,
     title={ManipulaPy: A High-Performance Python Library for Robotics Manipulation},
     author={Mohamed Aboelnar},
     journal={Journal of Open Source Software},
     year={2024},
     publisher={The Open Journal},
     doi={10.21105/joss.xxxxx}
   }