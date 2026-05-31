_api-control:

=========================
Control API Reference
=========================

This page documents **ManipulaPy.control**, the module for manipulator control algorithms.

.. note::
   As of v1.3.2, the control module is **GPU-optional / backend-agnostic**: it
   runs on NumPy by default and accepts CuPy arrays where available. Public type
   hints no longer reference ``cp.ndarray`` — inputs are coerced to NumPy
   internally, so callers can mix backends freely.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Control`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2
   :backlinks: none

---

ManipulatorController Class
===========================

.. currentmodule:: ManipulaPy.control

.. autoclass:: ManipulatorController
   :no-members:
   :show-inheritance:

   Main class for control of robotic manipulators. Backend-agnostic: works
   with NumPy or CuPy arrays (CuPy inputs are coerced to NumPy internally
   as of v1.3.2).

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

   :raises ValueError: if ``kalman_filter_predict`` has not been called
       (``self.x_hat is None``).
   :raises ValueError: if ``self.P`` is ``None`` or does not have shape
       ``(n, n)`` where ``n == self.x_hat.shape[0]``.

   .. versionchanged:: 1.3.2
      Both the ``x_hat`` and ``P`` preconditions are now validated up-front
      and raise ``ValueError`` with a descriptive message instead of failing
      inside the matrix algebra.

.. automethod:: ManipulatorController.kalman_filter_control

---

Performance Analysis Tools
==========================

.. automethod:: ManipulatorController.plot_steady_state_response
.. automethod:: ManipulatorController.calculate_rise_time
.. automethod:: ManipulatorController.calculate_percent_overshoot
.. automethod:: ManipulatorController.calculate_settling_time

   :param time: Array of time steps.
   :type time: np.ndarray
   :param response: Array of response values.
   :type response: np.ndarray
   :param set_point: Desired set point value (may be negative).
   :type set_point: float
   :param tolerance: Fractional tolerance band (default ``0.02`` = 2 %).
   :type tolerance: float
   :returns: First time at which the response enters the tolerance band
       and never leaves it again, or ``float('inf')`` if it never settles.
   :rtype: float

   .. versionchanged:: 1.3.2
      Returns the *first* time the response enters the settling band
      and stays there (was returning the last cross). Negative
      setpoints are now handled correctly.

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
