_api-control:

=========================
Control API Reference
=========================

This page documents **ManipulaPy.control**, the module for manipulator control algorithms with GPU acceleration.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Control`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

ManipulatorController Class
===========================

.. currentmodule:: ManipulaPy.control

.. autoclass:: ManipulatorController
   :members:
   :show-inheritance:

   Main class for control of robotic manipulators using CuPy-accelerated algorithms.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **dynamics** (*ManipulatorDynamics*) -- Instance providing dynamics computations

---

Control Strategies
==================

Basic Controllers
-----------------

.. automethod:: ManipulatorController.pid_control
.. automethod:: ManipulatorController.pd_control

Advanced Controllers
--------------------

.. automethod:: ManipulatorController.computed_torque_control
.. automethod:: ManipulatorController.adaptive_control
.. automethod:: ManipulatorController.robust_control

Feedforward Control
-------------------

.. automethod:: ManipulatorController.feedforward_control
.. automethod:: ManipulatorController.pd_feedforward_control

Space-Specific Control
----------------------

.. automethod:: ManipulatorController.joint_space_control
.. automethod:: ManipulatorController.cartesian_space_control

---

State Estimation
================

.. automethod:: ManipulatorController.kalman_filter_predict
.. automethod:: ManipulatorController.kalman_filter_update
.. automethod:: ManipulatorController.kalman_filter_control

---

Performance Analysis Tools
==========================

.. automethod:: ManipulatorController.plot_steady_state_response
.. automethod:: ManipulatorController.calculate_rise_time
.. automethod:: ManipulatorController.calculate_percent_overshoot
.. automethod:: ManipulatorController.calculate_settling_time
.. automethod:: ManipulatorController.calculate_steady_state_error

---

Auto-Tuning and Limits
======================

.. automethod:: ManipulatorController.ziegler_nichols_tuning
.. automethod:: ManipulatorController.tune_controller
.. automethod:: ManipulatorController.find_ultimate_gain_and_period
.. automethod:: ManipulatorController.enforce_limits

---

See Also
========

* :doc:`dynamics` -- Robot dynamics for model-based control
* :doc:`kinematics` -- Kinematic models for Cartesian control
* :doc:`path_planning` -- Trajectory reference generation
* :doc:`simulation` -- Simulator integration and testing tools