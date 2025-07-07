# Kinematics User Guide

This chapter walks you through **everyday kinematics workflows** in ManipulaPy:
from building a *SerialManipulator* object, to computing forward & inverse
kinematics, Jacobians, workspace envelopes, and velocity mappings.  It assumes
you already ran the \:doc:`../installation` guide and have a working Python
interpreter.

.. \_ug-kinematics-prereq:

## Prerequisites

* Python ≥ 3.8 with `numpy` & `matplotlib`
* ManipulaPy ≥ |release|
* A basic grasp of screw theory / the Product‑of‑Exponentials (PoE) model.

If you need a refresher, see `Modern Robotics, Ch 3`\_ (free online PDF) or our
\:doc:`../theory/screw_theory` page.

.. \_Modern Robotics, Ch 3: [http://hades.mech.northwestern.edu/index.php/Modern\_Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)

## What is robot kinematics?

*Robot kinematics* studies the geometry of a manipulator **without** worrying
about forces or torques.  The central problems are:

* **Forward kinematics (FK):** joint angles → pose of the tool frame \:math:`T \in
  \mathrm{SE}(3)`.
* **Inverse kinematics (IK):** desired pose → one or more joint angle
  solutions.
* **Velocity kinematics:** joint‑space rates ↔ spatial twist of the
  end‑effector.

ManipulaPy models every rigid motion as an element of
\:math:`\mathrm{SE}(3) \cong \mathbb{R}^3 \times \mathrm{SO}(3)` using
*homogeneous transforms* and encodes joint axes as **screws**
\:math:`S_i \in \mathfrak{se}(3)`.

Forward kinematics via PoE

```

With screws arranged column‑wise in :math:`S_{list} \in \mathbb{R}^{6\times n}`
and a *home configuration* matrix :math:`M`, the FK map is

.. math::

   T(\theta)\;=\;e^{[S_1] \theta_1}\;e^{[S_2] \theta_2}\;\dots
   e^{[S_n] \theta_n}\,M.

ManipulaPy evaluates this with numerically stable *Rodrigues* formulas.

--------------------------------------------------------------
Quick‑start example
--------------------------------------------------------------

.. code-block:: python
   :caption: Define a 6‑DoF UR5‑like arm in 10 lines

   import numpy as np
   from ManipulaPy.kinematics import SerialManipulator

   # Home pose of tool frame (millimetres → metres)
   M = np.array([[1, 0, 0, 0.817],
                 [0, 1, 0, 0.000],
                 [0, 0, 1, 0.191],
                 [0, 0, 0, 1     ]])

   # Screw axes in the *space* frame (ω | v)
   S_list = np.array([
       [0,  0,  1,  0.000,  0.000, 0.000],
       [0, -1,  0, -0.089,  0.000, 0.000],
       [0, -1,  0, -0.089,  0.000, 0.425],
       [0, -1,  0, -0.089,  0.000, 0.817],
       [1,  0,  0,  0.000,  0.109, 0.000],
       [0, -1,  0, -0.089,  0.000, 0.817],
   ]).T

   robot = SerialManipulator(M_list=M, S_list=S_list)  # B_list autocomputed

   thetalist = np.deg2rad([30, -45, 10, 120, 0, 90])
   T = robot.forward_kinematics(thetalist)
   print(np.round(T, 3))

--------------------------------------------------------------
Creating a robot from URDF
--------------------------------------------------------------

Most users will load a *real* robot description:

.. code-block:: python

   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   urdf_path = "resources/urdfs/panda_arm.urdf"
   robot = URDFToSerialManipulator(urdf_path).serial_manipulator

   print(robot.n_joints)          # 7
   print(robot.joint_limits[:3])  # per‑joint lower/upper bounds

ManipulaPy parses ``<limit>`` tags, inertias, and collision geometry so the
same object can be reused for dynamics and planning.

--------------------------------------------------------------
Forward kinematics recipes
--------------------------------------------------------------

Single pose
~~~~~~~~~~~

.. code-block:: python

   T_B = robot.forward_kinematics(thetalist, frame="body")
   xyz = T_B[:3, 3]
   R   = T_B[:3, :3]      # orientation matrix

Batch evaluation
~~~~~~~~~~~~~~~~

If you need thousands of FK calls per frame, wrap them in NumPy arrays:

.. code-block:: python

   thetas = np.random.uniform(-np.pi, np.pi, (1024, robot.n_joints))
   poses  = robot.batch_forward_kinematics(thetas)  # returns (1024, 4, 4)

(The batch routine is CuPy‑accelerated when installed with
``pip install ManipulaPy[gpu-cuda11]``.)

--------------------------------------------------------------
Inverse kinematics
--------------------------------------------------------------

ManipulaPy supplies **four flavours**:

1. *Newton–Raphson* (`iterative_inverse_kinematics`) – default, robust.
2. *Levenberg–Marquardt* (`lm_inverse_kinematics`) – higher success near
   singularities.
3. *Position‑only IK* (`position_inverse_kinematics`) – ignores orientation.
4. *Null‑space IK* (redundancy resolution) – exposed via
   ``NullSpaceIKSolver``.

Example (basic Newton):

.. code-block:: python

   success, sol, _ = robot.iterative_inverse_kinematics(
       T_desired=T, thetalist0=np.zeros(robot.n_joints))

   assert success


--------------------------------------------------------------
Jacobian & singularities
--------------------------------------------------------------

.. code-block:: python

   J = robot.jacobian(thetalist, frame="space")  # (6 × n)
   manipulability = np.sqrt(np.linalg.det(J @ J.T))
   cond_number   = np.linalg.cond(J)

A warning is raised if ``cond_number > 1e6`` (config near singularity).

--------------------------------------------------------------
Velocity kinematics helper
--------------------------------------------------------------

.. code-block:: python

   θdot   = np.deg2rad([10, 5, 0, 0, 0, 0])      # joint rates [rad s⁻¹]
   V_s    = robot.end_effector_velocity(thetalist, θdot)  # spatial twist

   # Solve inverse velocity problem
   V_des  = np.array([0.1, 0, 0, 0, 0, 0.2])     # move +X & rotate +Z
   θdot_r = robot.joint_velocity(thetalist, V_des)

--------------------------------------------------------------
Workspace visualisation snippet
--------------------------------------------------------------

.. code-block:: python
   :caption: Scatter plot of reachable positions (CPU‑only)

   import matplotlib.pyplot as plt

   pts = robot.sample_workspace(n_samples=500)
   fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
   ax.scatter(*pts.T, s=3, alpha=.4)
   ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

--------------------------------------------------------------
Troubleshooting
--------------------------------------------------------------

.. _ug-kinematics-troubleshoot:

================  ===========================================
Symptom           Fix
================  ===========================================
IK fails to       * Check if the target pose is outside joint
converge            limits or workspace.
                  * Provide a better initial guess; use the
                    *smart_initial_guess* helper below.
Numerical         * Call IK with ``damping`` > 1e‑4 or switch to
instability         ``lm_inverse_kinematics``.
Joint limit       * Run :pyfunc:`ManipulaPy.kinematics.utils.enforce_joint_limits`.
violation
================  ===========================================

.. code-block:: python
   :caption: Smart initial guess helper

   def smart_initial_guess(robot, T_desired):
       try:
           return robot.current_configuration   # live robot state
       except AttributeError:
           # fallback: mid‑range of each joint
           return np.mean(robot.joint_limits, axis=1)

--------------------------------------------------------------
