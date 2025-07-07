ManipulaPy Documentation
========================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/version-0.2.0-blue.svg
   :alt: Version

**ManipulaPy** is a comprehensive Python library for robotic manipulator analysis, simulation, and control.

Features
--------

* **Kinematics**: Forward and inverse kinematics for serial manipulators
* **Dynamics**: Mass matrix computation, inverse dynamics, forward dynamics  
* **Trajectory Planning**: CUDA-accelerated trajectory generation and execution
* **Computer Vision**: Camera calibration, stereo vision, and object detection
* **Perception**: Obstacle detection and 3D point cloud processing
* **Control**: Various control algorithms including PID, computed torque, and adaptive control
* **Simulation**: PyBullet integration for realistic robot simulation
* **URDF Processing**: Automatic robot model generation from URDF files

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
   processor = URDFToSerialManipulator("path/to/robot.urdf")
   robot = processor.serial_manipulator

   # Forward kinematics
   joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
   end_effector_pose = robot.forward_kinematics(joint_angles)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Modules Overview
----------------

Core Modules
~~~~~~~~~~~~

* :mod:`ManipulaPy.kinematics` - Forward and inverse kinematics
* :mod:`ManipulaPy.dynamics` - Robot dynamics computations
* :mod:`ManipulaPy.path_planning` - Trajectory planning and execution
* :mod:`ManipulaPy.control` - Control algorithms and implementations

Vision and Perception
~~~~~~~~~~~~~~~~~~~~~

* :mod:`ManipulaPy.vision` - Computer vision capabilities
* :mod:`ManipulaPy.perception` - 3D perception and obstacle detection

Simulation and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

* :mod:`ManipulaPy.sim` - PyBullet simulation interface
* :mod:`ManipulaPy.urdf_processor` - URDF file processing
* :mod:`ManipulaPy.utils` - Utility functions and transformations

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
