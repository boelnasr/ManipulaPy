.. _api-dynamics:

=========================
Dynamics API Reference
=========================

This page documents **``ManipulaPy.dynamics``**, the module that
adds full rigid-body dynamics to :pyclass:`~ManipulaPy.kinematics.SerialManipulator`.

.. tip::
   Looking for a conceptual tour?  See :doc:`../user_guide/Dynamics`.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

.. ---------------------------------------------------------------------------
.. ManipulatorDynamics
.. ---------------------------------------------------------------------------

ManipulatorDynamics Class
-------------------------

.. currentmodule:: ManipulaPy.dynamics

.. autoclass:: ManipulatorDynamics
   :members:
   :show-inheritance:

   **Inheritance:**  derives from :pyclass:`~ManipulaPy.kinematics.SerialManipulator`
   and therefore exposes *all* forward-kinematics & Jacobian helpers.

   .. rubric:: Constructor

   .. automethod:: __init__

      **Arguments**

      ===================== ===========================================================
      ``M_list``            4 × 4 home-configuration matrix :math:`\mathbf{M}`
      ``omega_list``        3 × *n* joint axes :math:`\boldsymbol{\omega}_i`
      ``r_list``            3 × *n* position vectors in the space frame
      ``b_list``            3 × *n* position vectors in the body frame
      ``S_list``            6 × *n* screw axes in the space frame
      ``B_list``            6 × *n* screw axes in the body frame
      ``Glist``             list of 6 × 6 spatial inertia matrices
      ===================== ===========================================================

   .. rubric:: Mass matrix :math:`\mathbf{M}(\boldsymbol{\theta})`

   .. automethod:: mass_matrix

      *Caching:*  results are memoised per joint configuration.

   .. rubric:: Partial-derivative helper

   .. automethod:: partial_derivative

   .. rubric:: Force components

   .. automethod:: velocity_quadratic_forces
   .. automethod:: gravity_forces

   .. rubric:: Inverse Dynamics

   .. automethod:: inverse_dynamics

   .. rubric:: Forward Dynamics

   .. automethod:: forward_dynamics


---------------
Full Equations
---------------

.. rubric:: **Inverse dynamics**

.. math::

   \boldsymbol{\tau}\;=\;
   \mathbf{M}\!\bigl(\boldsymbol{\theta}\bigr)\,
   \ddot{\boldsymbol{\theta}}
   \;+\;
   \mathbf{C}\!\bigl(\boldsymbol{\theta},\dot{\boldsymbol{\theta}}\bigr)\,
   \dot{\boldsymbol{\theta}}
   \;+\;
   \mathbf{G}\!\bigl(\boldsymbol{\theta}\bigr)
   \;+\;
   \mathbf{J}^{\mathsf{T}}\!\bigl(\boldsymbol{\theta}\bigr)\,
   \mathbf{F}_{\text{ext}}

.. rubric:: **Forward dynamics**

.. math::

   \ddot{\boldsymbol{\theta}}
   \;=\;
   \mathbf{M}^{-1}\!\bigl(\boldsymbol{\theta}\bigr)
   \Bigl(
     \boldsymbol{\tau}
     \;-\;
     \mathbf{C}\!\bigl(\boldsymbol{\theta},\dot{\boldsymbol{\theta}}\bigr)\,
     \dot{\boldsymbol{\theta}}
     \;-\;
     \mathbf{G}\!\bigl(\boldsymbol{\theta}\bigr)
     \;-\;
     \mathbf{J}^{\mathsf{T}}\!\bigl(\boldsymbol{\theta}\bigr)\,
     \mathbf{F}_{\text{ext}}
   \Bigr)

-------------
Usage Examples
-------------

**Basic set-up**

.. code-block:: python

   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   # Load a URDF and grab its pre-built dynamics model
   processor = URDFToSerialManipulator("robot.urdf")
   dynamics  = processor.dynamics        # instance of ManipulatorDynamics

**Mass matrix**

.. code-block:: python

   M = dynamics.mass_matrix([0.1, 0.2, 0.3, 0, 0, 0])

**Individual force terms**

.. code-block:: python

   q   = np.array([0.1, 0.2, 0.3, 0, 0, 0])
   dq  = np.array([0.05, 0.1, 0, 0, 0, 0])

   C   = dynamics.velocity_quadratic_forces(q, dq)
   G   = dynamics.gravity_forces(q)

**Inverse dynamics**

.. code-block:: python

   ddq = np.array([1.0, 0.5, 0, 0, 0, 0])
   tau = dynamics.inverse_dynamics(q, dq, ddq,
                                   g=[0, 0, -9.81],
                                   Ftip=[0, 0, -10, 0, 0, 0])

**Forward dynamics**

.. code-block:: python

   ddq_sim = dynamics.forward_dynamics(
       q, dq, tau,
       g=[0, 0, -9.81],
       Ftip=[0, 0, 0, 0, 0, 0]
   )

-------------
Key Features
-------------

* **Automatic caching** of expensive mass-matrix evaluations  
* **Complete force model** (Coriolis, centrifugal, gravity, externals)  
* **Gravity override** for moon/Mars/zero-g simulations  
* **Joint-limit aware** helpers inherited from kinematics  
* **Designed for GPU** – the same formulas power the CUDA kernels

---------------
Troubleshooting
---------------

*Large torques?*  Check your gravity vector and link inertias.  
*Singularities?*  Watch the Jacobian condition number (see Kinematics guide).  
*Slow first call?*  Numba JIT compiles on demand; subsequent calls are much faster.

-----------------
See Also
-----------------

* :doc:`kinematics` – forward kinematics & Jacobians  
* :doc:`control` – model-based controllers that rely on this API  
* :doc:`path_planning` – trajectory generation with dynamics constraints  
* :doc:`../user_guide/Dynamics` – physics background and best-practice tips
