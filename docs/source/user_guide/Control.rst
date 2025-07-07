==========================
Control Module User Guide
==========================

The ManipulaPy Control Module provides comprehensive control algorithms for robotic manipulators with GPU acceleration using CuPy. This guide covers all available control methods, their parameters, and usage examples.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The control module implements various control strategies:

- **PID Control**: Classic feedback control
- **Computed Torque Control**: Model-based linearizing control
- **Adaptive Control**: Online parameter adaptation
- **Robust Control**: Disturbance rejection
- **Feedforward Control**: Open-loop compensation
- **Kalman Filter**: State estimation and control

All algorithms are GPU-accelerated and designed for real-time performance.

Mathematical Foundation
=======================

This section summarizes the key control-law equations implemented in the ``ManipulatorController`` class.

Computed-Torque (Inverse Dynamics) Control
-------------------------------------------

Compute the required joint-torques to achieve a desired trajectory while compensating for the robot's nonlinear dynamics:

.. math::

   e &= q_d - q, \quad \dot{e} = \dot{q}_d - \dot{q}, \quad \int e\,dt = \sum e\,dt \\
   \tau_{ff} &= M(q)\,\bigl(\ddot{q}_d + K_d\,\dot{e} + K_p\,e + K_i\,\!\!\int e\,dt\bigr) + C(q,\dot{q})\,\dot{q} + G(q)

Where:

- :math:`q,\dot{q},\ddot{q}` – actual joint angles, velocities, accelerations  
- :math:`q_d,\dot{q}_d,\ddot{q}_d` – desired joint trajectory  
- :math:`M(q)` – mass-inertia matrix  
- :math:`C(q,\dot{q})` – Coriolis/centrifugal matrix  
- :math:`G(q)` – gravity vector  
- :math:`K_p,K_i,K_d` – proportional, integral, derivative gains  

PID Control
-----------

A simpler joint-space PID law (no dynamics compensation):

.. math::

   \tau_{PID} = K_p\,e + K_i\,\int e\,dt + K_d\,\dot{e}

PD Feedforward Control
----------------------

Combine a PD feedback term with pure feedforward dynamics:

.. math::

   \tau = K_p\,e + K_d\,\dot{e} + \underbrace{M(q)\,\ddot{q}_d + C(q,\dot{q})\,\dot{q}_d + G(q)}_{\text{feedforward torque}}

Adaptive Control
----------------

Adjusts internal dynamic parameters online to compensate for modelling errors:

.. math::

   \hat{\theta}_{k+1} &= \hat{\theta}_k + \gamma \,e \\
   \tau &= M(q)\,\ddot{q}_d + C(q,\dot{q})\,\dot{q}_d + G(q) + Y(q,\dot{q},\ddot{q})\,\hat{\theta}

Where :math:`\hat{\theta}` is the estimated parameter vector and :math:`\gamma` the adaptation gain.

Robust Control
--------------

Adds a disturbance-rejection term to a computed-torque core:

.. math::

   \tau = M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + G(q) + J(q)^T\,F_{ext} + \alpha\,\hat{d}

With :math:`\hat{d}` the estimated disturbance and :math:`\alpha` its gain.

Kalman-Filter State Estimation
-------------------------------

Predict-update loop for joint state :math:`x = [q;\dot{q}]`:

1. **Prediction**  
   
   .. math::
   
      \hat{x}^- = F\,\hat{x} + B\,u, \quad P^- = F\,P\,F^T + Q

2. **Update**  
   
   .. math::
   
      K &= P^-\,H^T(H\,P^-\,H^T + R)^{-1} \\
      \hat{x} &= \hat{x}^- + K(y - H\,\hat{x}^-), \quad P = (I - K\,H)P^-

Where :math:`Q,R` are process/measurement covariances.

These formulas correspond exactly to the methods in :class:`~ManipulaPy.control.ManipulatorController`.

Installation and Setup
=======================

Prerequisites
-------------

.. code-block:: bash

   pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12
   pip install numpy matplotlib

Basic Import
------------

.. code-block:: python

   import numpy as np
   import cupy as cp
   from ManipulaPy.control import ManipulatorController
   from ManipulaPy.dynamics import ManipulatorDynamics

Quick Start
===========

Initialize Controller
---------------------

.. code-block:: python

   # Assuming you have a dynamics object
   controller = ManipulatorController(dynamics)

   # Basic PID control example
   Kp = np.array([10.0, 8.0, 5.0, 3.0, 2.0, 1.0])  # Proportional gains
   Ki = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])   # Integral gains
   Kd = np.array([1.0, 0.8, 0.5, 0.3, 0.2, 0.1])   # Derivative gains

   # Desired and current states
   thetalistd = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0])  # Desired angles
   thetalist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # Current angles
   dthetalistd = np.zeros(6)  # Desired velocities
   dthetalist = np.zeros(6)   # Current velocities

   # Control signal
   tau = controller.pid_control(
       thetalistd, dthetalistd, thetalist, dthetalist, 
       dt=0.01, Kp=Kp, Ki=Ki, Kd=Kd
   )

Controller Types
================

PID Control
-----------

**Description**: Classic Proportional-Integral-Derivative control for joint space regulation.

.. automethod:: ManipulaPy.control.ManipulatorController.pid_control

**Example**:

.. code-block:: python

   # Define gains for 6-DOF manipulator
   Kp = np.array([15.0, 12.0, 8.0, 5.0, 3.0, 2.0])
   Ki = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1])
   Kd = np.array([2.0, 1.5, 1.0, 0.8, 0.5, 0.3])

   tau = controller.pid_control(
       thetalistd=desired_angles,
       dthetalistd=desired_velocities,
       thetalist=current_angles,
       dthetalist=current_velocities,
       dt=0.01,
       Kp=Kp, Ki=Ki, Kd=Kd
   )

PD Control
----------

**Description**: Proportional-Derivative control without integral term.

.. automethod:: ManipulaPy.control.ManipulatorController.pd_control

**Example**:

.. code-block:: python

   Kp = np.array([20.0, 15.0, 10.0, 8.0, 5.0, 3.0])
   Kd = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])

   tau = controller.pd_control(
       desired_position=desired_angles,
       desired_velocity=desired_velocities,
       current_position=current_angles,
       current_velocity=current_velocities,
       Kp=Kp, Kd=Kd
   )

Computed Torque Control
-----------------------

**Description**: Model-based control that linearizes the nonlinear robot dynamics.

.. automethod:: ManipulaPy.control.ManipulatorController.computed_torque_control

**Example**:

.. code-block:: python

   # Higher gains can be used due to linearization
   Kp = np.array([50.0, 40.0, 30.0, 20.0, 15.0, 10.0])
   Ki = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
   Kd = np.array([8.0, 6.0, 4.0, 3.0, 2.0, 1.0])

   gravity = np.array([0, 0, -9.81])

   tau = controller.computed_torque_control(
       thetalistd=desired_angles,
       dthetalistd=desired_velocities,
       ddthetalistd=desired_accelerations,
       thetalist=current_angles,
       dthetalist=current_velocities,
       g=gravity,
       dt=0.01,
       Kp=Kp, Ki=Ki, Kd=Kd
   )

Feedforward Control
-------------------

**Description**: Open-loop control based on desired trajectory dynamics.

.. automethod:: ManipulaPy.control.ManipulatorController.feedforward_control

**Example**:

.. code-block:: python

   gravity = np.array([0, 0, -9.81])
   external_forces = np.zeros(6)  # No external forces

   tau_ff = controller.feedforward_control(
       desired_position=trajectory_positions[i],
       desired_velocity=trajectory_velocities[i],
       desired_acceleration=trajectory_accelerations[i],
       g=gravity,
       Ftip=external_forces
   )

PD + Feedforward Control
------------------------

**Description**: Combines feedback PD control with feedforward compensation.

.. automethod:: ManipulaPy.control.ManipulatorController.pd_feedforward_control

**Example**:

.. code-block:: python

   tau = controller.pd_feedforward_control(
       desired_position=trajectory_positions[i],
       desired_velocity=trajectory_velocities[i],
       desired_acceleration=trajectory_accelerations[i],
       current_position=current_angles,
       current_velocity=current_velocities,
       Kp=Kp, Kd=Kd,
       g=gravity,
       Ftip=external_forces
   )

Robust Control
--------------

**Description**: Control with disturbance rejection capabilities.

.. automethod:: ManipulaPy.control.ManipulatorController.robust_control

**Example**:

.. code-block:: python

   disturbance_estimate = np.array([0.1, -0.05, 0.08, 0.0, 0.02, -0.01])
   adaptation_gain = 0.5

   tau = controller.robust_control(
       thetalist=current_angles,
       dthetalist=current_velocities,
       ddthetalist=desired_accelerations,
       g=gravity,
       Ftip=external_forces,
       disturbance_estimate=disturbance_estimate,
       adaptation_gain=adaptation_gain
   )

Adaptive Control
----------------

**Description**: Control with online parameter adaptation.

.. automethod:: ManipulaPy.control.ManipulatorController.adaptive_control

**Example**:

.. code-block:: python

   measurement_error = current_angles - estimated_angles
   adaptation_gain = 0.1

   tau = controller.adaptive_control(
       thetalist=current_angles,
       dthetalist=current_velocities,
       ddthetalist=desired_accelerations,
       g=gravity,
       Ftip=external_forces,
       measurement_error=measurement_error,
       adaptation_gain=adaptation_gain
   )

Advanced Features
=================

Joint and Torque Limit Enforcement
-----------------------------------

.. automethod:: ManipulaPy.control.ManipulatorController.enforce_limits

**Example**:

.. code-block:: python

   # Define limits
   joint_limits = np.array([
       [-np.pi, np.pi],     # Joint 1
       [-np.pi/2, np.pi/2], # Joint 2
       [-np.pi, np.pi],     # Joint 3
       [-np.pi, np.pi],     # Joint 4
       [-np.pi/2, np.pi/2], # Joint 5
       [-np.pi, np.pi]      # Joint 6
   ])

   torque_limits = np.array([
       [-50, 50],  # Joint 1 torque limits (Nm)
       [-40, 40],  # Joint 2
       [-30, 30],  # Joint 3
       [-20, 20],  # Joint 4
       [-15, 15],  # Joint 5
       [-10, 10]   # Joint 6
   ])

   # Apply limits
   safe_angles, safe_velocities, safe_torques = controller.enforce_limits(
       thetalist=current_angles,
       dthetalist=current_velocities,
       tau=computed_torques,
       joint_limits=joint_limits,
       torque_limits=torque_limits
   )

Kalman Filter State Estimation
-------------------------------

.. automethod:: ManipulaPy.control.ManipulatorController.kalman_filter_predict

.. automethod:: ManipulaPy.control.ManipulatorController.kalman_filter_update

.. automethod:: ManipulaPy.control.ManipulatorController.kalman_filter_control

**Example**:

.. code-block:: python

   # Define noise covariances
   Q = np.eye(12) * 0.01  # Process noise (12x12 for 6 positions + 6 velocities)
   R = np.eye(12) * 0.1   # Measurement noise

   # Use Kalman filter for state estimation
   estimated_angles, estimated_velocities = controller.kalman_filter_control(
       thetalistd=desired_angles,
       dthetalistd=desired_velocities,
       thetalist=measured_angles,
       dthetalist=measured_velocities,
       taulist=applied_torques,
       g=gravity,
       Ftip=external_forces,
       dt=0.01,
       Q=Q, R=R
   )

Tuning Methods
==============

Ziegler-Nichols Tuning
----------------------

.. automethod:: ManipulaPy.control.ManipulatorController.ziegler_nichols_tuning

**Example**:

.. code-block:: python

   # Find ultimate gain and period first
   ultimate_gain, ultimate_period, gain_history, error_history = controller.find_ultimate_gain_and_period(
       thetalist=initial_angles,
       desired_joint_angles=target_angles,
       dt=0.01,
       max_steps=1000
   )

   # Tune controller gains
   Kp, Ki, Kd = controller.ziegler_nichols_tuning(
       Ku=ultimate_gain,
       Tu=ultimate_period,
       kind="PID"
   )

   print(f"Tuned gains - Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")

Automatic Ultimate Gain Finding
-------------------------------

.. automethod:: ManipulaPy.control.ManipulatorController.find_ultimate_gain_and_period

.. automethod:: ManipulaPy.control.ManipulatorController.tune_controller

**Example**:

.. code-block:: python

   # Automatically find critical parameters
   results = controller.find_ultimate_gain_and_period(
       thetalist=np.zeros(6),
       desired_joint_angles=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
       dt=0.01,
       max_steps=500
   )

   ultimate_gain, ultimate_period, gain_history, error_history = results

   # Use results for tuning
   Kp, Ki, Kd = controller.tune_controller(ultimate_gain, ultimate_period, kind="PID")

Plotting and Analysis
=====================

Steady-State Response Analysis
------------------------------

.. automethod:: ManipulaPy.control.ManipulatorController.plot_steady_state_response

**Example**:

.. code-block:: python

   # Simulate control response
   time_steps = np.linspace(0, 5, 500)
   response_data = []  # Your simulation data
   set_point = 0.5

   controller.plot_steady_state_response(
       time=time_steps,
       response=response_data,
       set_point=set_point,
       title="Joint 1 Step Response"
   )

Performance Metrics
-------------------

Calculate control performance metrics:

.. automethod:: ManipulaPy.control.ManipulatorController.calculate_rise_time

.. automethod:: ManipulaPy.control.ManipulatorController.calculate_percent_overshoot

.. automethod:: ManipulaPy.control.ManipulatorController.calculate_settling_time

.. automethod:: ManipulaPy.control.ManipulatorController.calculate_steady_state_error

**Example**:

.. code-block:: python

   # Calculate individual metrics
   rise_time = controller.calculate_rise_time(time, response, set_point)
   overshoot = controller.calculate_percent_overshoot(response, set_point)
   settling_time = controller.calculate_settling_time(time, response, set_point)
   steady_error = controller.calculate_steady_state_error(response, set_point)

   print(f"Rise Time: {rise_time:.3f}s")
   print(f"Overshoot: {overshoot:.2f}%")
   print(f"Settling Time: {settling_time:.3f}s")
   print(f"Steady-State Error: {steady_error:.4f}")

Cartesian Space Control
=======================

Joint Space Control
-------------------

.. automethod:: ManipulaPy.control.ManipulatorController.joint_space_control

Cartesian Space Control
-----------------------

.. automethod:: ManipulaPy.control.ManipulatorController.cartesian_space_control

Examples
========

Complete Control Loop Example
------------------------------

.. code-block:: python

   import numpy as np
   import cupy as cp
   from ManipulaPy.control import ManipulatorController

   def control_loop_example(controller, dynamics):
       # Initialize states
       current_angles = np.zeros(6)
       current_velocities = np.zeros(6)
       desired_angles = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
       
       # Control parameters
       Kp = np.array([20.0, 15.0, 12.0, 8.0, 5.0, 3.0])
       Ki = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1])
       Kd = np.array([2.0, 1.5, 1.2, 0.8, 0.5, 0.3])
       
       dt = 0.01
       simulation_time = 5.0
       steps = int(simulation_time / dt)
       
       # Storage for plotting
       time_history = []
       angle_history = []
       torque_history = []
       
       for step in range(steps):
           # Compute control signal
           tau = controller.computed_torque_control(
               thetalistd=desired_angles,
               dthetalistd=np.zeros(6),
               ddthetalistd=np.zeros(6),
               thetalist=current_angles,
               dthetalist=current_velocities,
               g=np.array([0, 0, -9.81]),
               dt=dt,
               Kp=Kp, Ki=Ki, Kd=Kd
           )
           
           # Simulate dynamics (simplified)
           accelerations = dynamics.forward_dynamics(
               current_angles, current_velocities, 
               cp.asnumpy(tau), np.array([0, 0, -9.81]), 
               np.zeros(6)
           )
           
           # Update states
           current_velocities += accelerations * dt
           current_angles += current_velocities * dt
           
           # Store data
           time_history.append(step * dt)
           angle_history.append(current_angles.copy())
           torque_history.append(cp.asnumpy(tau))
       
       return np.array(time_history), np.array(angle_history), np.array(torque_history)

Trajectory Tracking Example
----------------------------

.. code-block:: python

   def trajectory_tracking_example(controller, trajectory):
       """
       Example of tracking a predefined trajectory using PD+Feedforward control
       """
       positions = trajectory["positions"]
       velocities = trajectory["velocities"] 
       accelerations = trajectory["accelerations"]
       
       # Control gains
       Kp = np.array([25.0, 20.0, 15.0, 10.0, 8.0, 5.0])
       Kd = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.8])
       
       current_state = np.zeros(6)
       current_velocity = np.zeros(6)
       
       tracking_errors = []
       
       for i in range(len(positions)):
           # Compute control with feedforward
           tau = controller.pd_feedforward_control(
               desired_position=positions[i],
               desired_velocity=velocities[i],
               desired_acceleration=accelerations[i],
               current_position=current_state,
               current_velocity=current_velocity,
               Kp=Kp, Kd=Kd,
               g=np.array([0, 0, -9.81]),
               Ftip=np.zeros(6)
           )
           
           # Calculate tracking error
           error = np.linalg.norm(positions[i] - current_state)
           tracking_errors.append(error)
           
           # Update state (simplified integration)
           # In practice, you would use your robot's dynamics
           current_state = positions[i]  # Perfect tracking for demo
       
       return tracking_errors

Troubleshooting
===============

Common Issues and Solutions
---------------------------

1. **Controller Instability**
   
   - **Cause**: Gains too high or improper tuning
   - **Solution**: Use Ziegler-Nichols tuning or reduce gains systematically

2. **CuPy Memory Errors**
   
   - **Cause**: GPU memory exhaustion
   - **Solution**: Use smaller batch sizes or convert to CPU arrays when needed

3. **Numerical Issues**
   
   - **Cause**: Ill-conditioned matrices or extreme values
   - **Solution**: Add regularization or use more robust numerical methods

4. **Slow Performance**
   
   - **Cause**: Frequent CPU-GPU transfers
   - **Solution**: Keep computations on GPU as much as possible

Best Practices
--------------

1. **Gain Tuning**:
   
   - Start with conservative gains and increase gradually
   - Use automatic tuning methods when possible
   - Test stability margins before deployment

2. **GPU Usage**:
   
   - Convert to CuPy arrays early in the pipeline
   - Minimize data transfers between CPU and GPU
   - Use batch operations when possible

3. **Safety**:
   
   - Always enforce joint and torque limits
   - Implement emergency stops
   - Validate control signals before application

4. **Performance**:
   
   - Profile your control loop to identify bottlenecks
   - Use appropriate time steps for your application
   - Consider predictive control for better performance

Error Handling
--------------

.. code-block:: python

   try:
       tau = controller.computed_torque_control(...)
       
       # Check for NaN or infinite values
       if not np.all(np.isfinite(cp.asnumpy(tau))):
           raise ValueError("Control signal contains invalid values")
           
       # Enforce safety limits
       safe_tau = cp.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
       
   except Exception as e:
       print(f"Control error: {e}")
       # Implement fallback strategy
       tau = np.zeros(6)  # Safe fallback

API Reference
=============

.. autoclass:: ManipulaPy.control.ManipulatorController
   :members:
   :undoc-members:
   :show-inheritance:

Additional Resources
====================

- :doc:`../tutorials/kinematics_guide` - Learn about robot kinematics
- :doc:`../tutorials/dynamics_guide` - Understand robot dynamics
- :doc:`Trajectory_Planning` - Motion planning and trajectory generation
- :doc:`../api/control` - Complete API reference

For issues and questions:

- GitHub Issues: `ManipulaPy Issues <https://github.com/boelnasr/ManipulaPy/issues>`_
- Documentation: `ManipulaPy Docs <https://boelnasr.github.io/ManipulaPy/>`_