Potential Field Module User Guide
=================================

This comprehensive guide covers the potential field path-planning tools in ManipulaPy, optimized for Python 3.10.12.

.. note::
   This guide is written for Python 3.10.12 users and includes version-specific examples 
   and performance tips.

Introduction to Potential Fields
-------------------------------

Potential field methods treat the robot as a point moving under the influence of an artificial
potential surface.  The robot is “pulled” toward the goal by an _attractive_ potential and
“pushed” away from obstacles by _repulsive_ potentials.  By following the negative gradient
of the combined field, the robot can find a collision-free path.

Key components:

- **Attractive potential**  draws the robot toward the goal.  
- **Repulsive potential**   pushes the robot away from obstacles within an influence radius.  
- **Gradient**             direction of steepest descent in the combined potential.  
- **Collision checking**   rapid test whether a given configuration is in collision.  

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\mathbf q \in \mathbb R^n` be the robot’s configuration (e.g., joint angles).  
Let :math:`\mathbf q_{\mathrm{goal}}` be the goal configuration, and let  
:math:`\{\,\mathbf o_i\in\mathbb R^n\}` be a set of obstacle points.

Attractive Potential
^^^^^^^^^^^^^^^^^^^^

A common quadratic attractive potential:

.. math::
   U_{\mathrm{att}}(\mathbf q)
     = \tfrac12\,k_{\mathrm{att}}\,\|\mathbf q - \mathbf q_{\mathrm{goal}}\|^2

where :math:`k_{\mathrm{att}}>0` is the attractive gain.

Repulsive Potential
^^^^^^^^^^^^^^^^^^^

An inverse-distance repulsive potential, active only within an influence distance :math:`d_0`:

.. math::
   U_{\mathrm{rep}}(\mathbf q)
     = \sum_{i}
       \begin{cases}
         \tfrac12\,k_{\mathrm{rep}}\!\bigl(\tfrac{1}{d_i} - \tfrac{1}{d_0}\bigr)^2
           & d_i \le d_0,\\[6pt]
         0 & d_i > d_0,
       \end{cases}
\quad
   d_i = \|\mathbf q - \mathbf o_i\|,
   \;k_{\mathrm{rep}}>0.

Total Potential and Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The combined potential:

.. math::
   U(\mathbf q)
     = U_{\mathrm{att}}(\mathbf q)
       + U_{\mathrm{rep}}(\mathbf q).

The control “force” is the negative gradient:

.. math::
   \mathbf F(\mathbf q)
     = -\,\nabla U(\mathbf q)
     = -\nabla U_{\mathrm{att}}(\mathbf q)
       -\nabla U_{\mathrm{rep}}(\mathbf q).

Closed-form gradients:

.. math::
   \nabla U_{\mathrm{att}}(\mathbf q)
     = k_{\mathrm{att}}\,(\mathbf q - \mathbf q_{\mathrm{goal}}),

.. math::
   \nabla U_{\mathrm{rep}}(\mathbf q)
     = \sum_{i:\,d_i \le d_0}
       k_{\mathrm{rep}}
       \bigl(\tfrac{1}{d_i} - \tfrac{1}{d_0}\bigr)
       \tfrac{1}{d_i^3}\,(\mathbf q - \mathbf o_i).

Class Reference
---------------

ManipulaPy provides two main classes in this module:

- **`PotentialField`**  
  Compute attractive & repulsive potentials and their gradients.

- **`CollisionChecker`**  
  Use URDF geometry to build convex hulls and test for self-collision.

Installation
~~~~~~~~~~~~

Ensure required packages are installed:

.. code-block:: bash

   pip install ManipulaPy[core] scipy urchin

Usage Examples
--------------

1. **Basic potential and gradient**

.. code-block:: python

   import numpy as np
   from ManipulaPy.potential_field import PotentialField

   # Define goal and obstacles in configuration space
   q_goal    = np.array([0.5, 0.2, -0.3])
   obstacles = [np.array([0.1, 0.0, 0.0]), np.array([0.4, 0.1, -0.2])]

   pf = PotentialField(
       attractive_gain=1.5,
       repulsive_gain=50.0,
       influence_distance=0.4
   )

   q = np.array([0.0, 0.0, 0.0])
   U_att = pf.compute_attractive_potential(q, q_goal)
   U_rep = pf.compute_repulsive_potential(q, obstacles)
   grad  = pf.compute_gradient(q, q_goal, obstacles)

   print(f"U_att = {U_att:.3f}, U_rep = {U_rep:.3f}")
   print(f"Total gradient = {grad}")

2. **Collision checking**

.. code-block:: python

   from ManipulaPy.potential_field import CollisionChecker

   cc = CollisionChecker("robot.urdf")
   q_test = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])

   if cc.check_collision(q_test):
       print("Configuration is in collision!")
   else:
       print("Configuration is collision-free.")

3. **Gradient descent path planning**

.. code-block:: python

   path = []
   q = np.zeros(6)
   for _ in range(100):
       grad = pf.compute_gradient(q, q_goal, obstacles)
       q    = q - 0.05*grad  # step size
       path.append(q.copy())
       if np.linalg.norm(q - q_goal) < 1e-3:
           break

   print(f"Planned {len(path)} steps to goal")

Advanced Topics
---------------

### Performance Tips

- **Vectorize obstacle list**: stack obstacles into an :math:`(m\times n)` array and
  compute all distances at once for large :math:`m`.  
- **Tune gains**: high :math:`k_{\mathrm{rep}}` produces stronger obstacle avoidance but
  may create local minima.  
- **Cache gradient**: if you repeatedly query similar :math:`\mathbf q`, memoize the result.

### Combining with Trajectory Planning

You can integrate the gradient from `PotentialField` into your
`TrajectoryPlanning` loop to adjust intermediate waypoints:

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning

   planner = TrajectoryPlanning(robot, "robot.urdf", dynamics, joint_limits)
   traj    = planner.joint_trajectory(q_start, q_goal, Tf=2.0, N=500, method=5)

   for idx, q in enumerate(traj["positions"]):
       if cc.check_collision(q):
           grad = pf.compute_gradient(q, q_goal, obstacles)
           traj["positions"][idx] -= 0.01*grad

Troubleshooting
---------------

- **Zero repulsive gradient**  
  If your robot never “feels” obstacles, check that
  `influence_distance` is larger than the minimum :math:`d_i`.  

- **Local minima**  
  Potential fields can trap in local minima.  Hybridize with RRT or rapidly-exploring
  random tree to escape.  

- **Performance bottleneck**  
  For many obstacles, vectorize distance computations or implement a CUDA kernel
  (see CUDA Kernels guide).

References
----------

- Latombe, J.-C., _Robot Motion Planning_, Kluwer, 1991.  
- Khatib, O., “Real-time obstacle avoidance for manipulators and mobile robots,”
  _IEEE IJRR_, 1986.  
- urchin.urdf — URDF parser for Python (used for mesh loading and convex hulls).  
