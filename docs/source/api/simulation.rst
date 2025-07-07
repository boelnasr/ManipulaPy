.. _api-simulation:

============================
Simulation API Reference
============================

This page documents **ManipulaPy.sim**, the module for PyBullet-based simulation capabilities with real-time visualization, physics simulation, and interactive control.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Simulation`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

Simulation Class
================

.. currentmodule:: ManipulaPy.sim

.. autoclass:: Simulation
   :members:
   :show-inheritance:

   Main class for PyBullet-based simulation of robotic manipulators with real-time physics, visualization, and interactive control capabilities.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **urdf_file_path** (*str*) -- Path to the URDF file describing the robot
   - **joint_limits** (*list*) -- List of tuples representing joint angle limits
   - **torque_limits** (*list, optional*) -- List of tuples representing torque limits
   - **time_step** (*float, optional*) -- Simulation time step (default: 0.01)
   - **real_time_factor** (*float, optional*) -- Real-time scaling factor (default: 1.0)
   - **physics_client** (*int, optional*) -- PyBullet physics client ID

---

Simulation Management
=====================

Environment Setup
-----------------

.. automethod:: Simulation.setup_simulation
.. automethod:: Simulation.connect_simulation
.. automethod:: Simulation.disconnect_simulation
.. automethod:: Simulation.close_simulation

Robot Initialization
--------------------

.. automethod:: Simulation.initialize_robot
.. automethod:: Simulation.initialize_planner_and_controller

---

Robot Control and State
========================

Joint Control
-------------

.. automethod:: Simulation.set_joint_positions
.. automethod:: Simulation.get_joint_positions
.. automethod:: Simulation.get_joint_parameters

Trajectory Execution
--------------------

.. automethod:: Simulation.run_trajectory
.. automethod:: Simulation.simulate_robot_motion
.. automethod:: Simulation.simulate_robot_with_desired_angles

Controller Integration
----------------------

.. automethod:: Simulation.run_controller

---

Interactive Controls
====================

GUI Elements
------------

.. automethod:: Simulation.add_joint_parameters
.. automethod:: Simulation.add_reset_button
.. automethod:: Simulation.add_additional_parameters

Manual Control
--------------

.. automethod:: Simulation.manual_control
.. automethod:: Simulation.update_simulation_parameters

---

Visualization and Analysis
==========================

Trajectory Visualization
------------------------

.. automethod:: Simulation.plot_trajectory
.. automethod:: Simulation.plot_trajectory_in_scene

Data Management
---------------

.. automethod:: Simulation.save_joint_states

---

Safety and Monitoring
=====================

.. automethod:: Simulation.check_collisions
.. automethod:: Simulation.step_simulation

---

Main Execution
==============

.. automethod:: Simulation.run

---

Logging and Utilities
=====================

.. automethod:: Simulation.setup_logger

---

Usage Examples
==============

Basic Simulation Setup
----------------------

.. code-block:: python

   from ManipulaPy.sim import Simulation
   import numpy as np

   # Define joint limits for a 6-DOF robot
   joint_limits = [(-np.pi, np.pi)] * 6
   torque_limits = [(-50, 50)] * 6

   # Create simulation instance
   sim = Simulation(
       urdf_file_path="robot.urdf",
       joint_limits=joint_limits,
       torque_limits=torque_limits,
       time_step=0.01,
       real_time_factor=1.0
   )

   # Initialize robot and components
   sim.initialize_robot()
   sim.initialize_planner_and_controller()

Running a Trajectory
--------------------

.. code-block:: python

   # Define a simple trajectory
   trajectory = [
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start position
       [0.5, 0.3, -0.2, 0.1, 0.4, 0.0], # Intermediate
       [1.0, 0.5, -0.5, 0.2, 0.6, 0.1]  # End position
   ]

   # Run the trajectory
   final_ee_pos = sim.run_trajectory(trajectory)
   print(f"Final end-effector position: {final_ee_pos}")

Interactive Manual Control
--------------------------

.. code-block:: python

   # Add GUI controls
   sim.add_joint_parameters()
   sim.add_reset_button()
   sim.add_additional_parameters()

   # Start manual control mode
   sim.manual_control()

Controller Integration
---------------------

.. code-block:: python

   import cupy as cp

   # Define control parameters
   desired_positions = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])
   desired_velocities = np.zeros_like(desired_positions)
   desired_accelerations = np.zeros_like(desired_positions)

   # Control gains
   Kp = np.array([100, 100, 100, 50, 50, 50])
   Ki = np.array([1, 1, 1, 0.5, 0.5, 0.5])
   Kd = np.array([10, 10, 10, 5, 5, 5])

   # Gravity and external forces
   g = [0, 0, -9.81]
   Ftip = [0, 0, 0, 0, 0, 0]

   # Run controller
   final_pos = sim.run_controller(
       sim.controller,
       desired_positions,
       desired_velocities,
       desired_accelerations,
       g, Ftip, Kp, Ki, Kd
   )

Collision Monitoring
-------------------

.. code-block:: python

   # Enable collision checking during simulation
   while True:
       # Update robot state
       joint_positions = sim.get_joint_parameters()
       sim.set_joint_positions(joint_positions)
       
       # Check for collisions
       sim.check_collisions()
       
       # Step simulation
       sim.step_simulation()

Data Logging and Analysis
------------------------

.. code-block:: python

   # Save joint states during simulation
   sim.save_joint_states("robot_states.csv")

   # Visualize trajectory in 3D
   joint_trajectory = [
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
       [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
   ]
   
   ee_trajectory = []
   for joints in joint_trajectory:
       # Calculate end-effector position for each joint configuration
       ee_pos = sim.robot.forward_kinematics(joints)[:3, 3]
       ee_trajectory.append(ee_pos)
   
   sim.plot_trajectory_in_scene(joint_trajectory, ee_trajectory)

Complete Simulation Loop
-----------------------

.. code-block:: python

   # Complete example with trajectory execution and manual control
   try:
       # Generate trajectory using path planner
       thetastart = np.zeros(6)
       thetaend = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.0])
       
       trajectory_data = sim.trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=5.0, N=100, method=3
       )
       
       # Run the main simulation loop
       sim.run(trajectory_data["positions"])
       
   except KeyboardInterrupt:
       print("Simulation interrupted by user")
   finally:
       sim.close_simulation()

---

Configuration Options
=====================

Simulation Parameters
--------------------

The simulation can be configured with various parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``time_step``
     - 0.01
     - Physics simulation time step in seconds
   * - ``real_time_factor``
     - 1.0
     - Speed multiplier for real-time simulation
   * - ``joint_limits``
     - Required
     - List of (min, max) tuples for each joint
   * - ``torque_limits``
     - None
     - Optional torque limits for each joint

Physics Settings
----------------

The simulation uses PyBullet's physics engine with the following default settings:

- **Gravity**: [0, 0, -9.81] m/s²
- **Solver iterations**: PyBullet default
- **Contact breaking threshold**: PyBullet default
- **Collision detection**: Enabled for all robot links

GUI Controls
------------

When using interactive mode, the following GUI elements are available:

- **Joint sliders**: Individual control for each robot joint
- **Reset button**: Return robot to home position
- **Gravity control**: Adjust gravitational acceleration
- **Time step control**: Modify simulation time step

---

Performance Considerations
==========================

GPU Acceleration
---------------

The simulation module integrates with CuPy for GPU-accelerated computations:

- Controller calculations use CuPy arrays for improved performance
- Large trajectory computations benefit from GPU acceleration
- Memory management is handled automatically

Real-time Performance
--------------------

For optimal real-time performance:

- Use appropriate ``time_step`` values (0.001-0.01 seconds)
- Adjust ``real_time_factor`` based on computational load
- Monitor collision detection frequency for complex robots
- Consider reducing visualization quality for faster simulation

---

Error Handling
==============

Common Issues and Solutions
--------------------------

.. note::
   The simulation module includes comprehensive error handling for common issues:

**URDF Loading Errors**
   - Verify URDF file path and format
   - Check for missing mesh files or textures
   - Ensure joint limits are properly defined

**Physics Instability**
   - Reduce time step for better numerical stability
   - Check for unrealistic joint limits or masses
   - Verify contact parameters are reasonable

**GUI Issues**
   - Ensure PyBullet GUI mode is properly initialized
   - Check for conflicting parameter names
   - Verify graphics drivers support OpenGL

**Memory Issues**
   - Monitor CuPy memory usage for large trajectories
   - Use batch processing for very long simulations
   - Clear visualization lines periodically

---

Integration with Other Modules
==============================

Path Planning Integration
------------------------

.. code-block:: python

   # Using trajectory planner with simulation
   from ManipulaPy.path_planning import TrajectoryPlanning
   
   planner = TrajectoryPlanning(
       sim.robot, 
       sim.urdf_file_path, 
       sim.dynamics,
       sim.joint_limits,
       sim.torque_limits
   )
   
   # Generate smooth trajectory
   trajectory = planner.joint_trajectory(
       thetastart, thetaend, Tf=3.0, N=150, method=5
   )
   
   # Execute in simulation
   sim.run_trajectory(trajectory["positions"])

Control System Integration
-------------------------

.. code-block:: python

   # Advanced controller setup
   from ManipulaPy.control import ManipulatorController
   
   controller = ManipulatorController(sim.dynamics)
   
   # Auto-tune controller gains
   Ku, Tu = controller.find_ultimate_gain_and_period(
       thetalist=np.zeros(6),
       desired_joint_angles=np.array([0.1, 0.2, 0.1, 0.0, 0.1, 0.0]),
       dt=sim.time_step
   )
   
   Kp, Ki, Kd = controller.tune_controller(Ku, Tu, kind="PID")
   
   # Use tuned gains in simulation
   sim.run_controller(controller, positions, velocities, accelerations,
                     g, Ftip, Kp, Ki, Kd)

Vision System Integration
------------------------

.. code-block:: python

   # Camera-based simulation feedback
   from ManipulaPy.vision import Vision
   from ManipulaPy.perception import Perception
   
   # Setup vision system (if available)
   vision = Vision(use_pybullet_debug=True)
   perception = Perception(vision_instance=vision)
   
   # Use perception for obstacle avoidance in simulation
   obstacles, labels = perception.detect_and_cluster_obstacles()
   
   # Modify trajectory based on detected obstacles
   # (Implementation depends on specific requirements)

---

See Also
========

* :doc:`control` -- Controller implementations for simulation
* :doc:`path_planning` -- Trajectory generation for simulation  
* :doc:`urdf_processor` -- URDF loading and robot model creation
* :doc:`dynamics` -- Robot dynamics for physics simulation
* :doc:`kinematics` -- Forward and inverse kinematics
* :doc:`vision` -- Computer vision integration
* :doc:`perception` -- Environmental perception capabilities

External References
===================

* `PyBullet Documentation <https://pybullet.org/>`_
* `CuPy Documentation <https://cupy.dev/>`_
* `URDF Specification <http://wiki.ros.org/urdf/XML>`_