.. _user_guide_collision_checker:

Collision Checker Module User Guide
===================================

This guide covers the :class:`ManipulaPy.potential_field.CollisionChecker` class,
which performs geometry-based **self-collision detection** for serial manipulators
using URDF meshes.

.. contents:: **Quick navigation**
   :local:
   :depth: 2

.. note::
   Examples target **Python 3.10+**, **SciPy 1.10+**, and the ``urchin.urdf`` parser
   for loading URDF meshes and building convex hulls for collision tests.

Introduction
------------

Collision checking is essential for safe motion planning. The
``CollisionChecker`` builds **convex-hull** approximations of each link’s visual mesh
and tests for pairwise intersection at a given robot configuration.

Key features
~~~~~~~~~~~~

- **Convex hull construction** from URDF mesh vertices
- **Fast pairwise collision tests** using convex polytope intersection checks
- **Pose transforms** of hulls via forward kinematics
- **Self-collision detection** between any two **non-adjacent** links

Mathematical Background
-----------------------

Given a robot configuration :math:`\mathbf q` (joint angles), the forward-kinematics
function :math:`T_i(\mathbf q)\in SE(3)` returns the homogeneous transform of link *i*.
Each link’s mesh is approximated by its convex hull :math:`H_i`. Under transform
:math:`T_i`, hull vertices map to

.. math::
   \{\,T_i(\mathbf q)\,p \mid p\in H_i\,\}.

Two convex polyhedra intersect iff their convex hulls intersect. In practice, the
check can be implemented by half-space evaluations or separating-axis tests on the
transformed hulls.

Workflow
~~~~~~~~

1. **Load URDF** → extract mesh vertices.  
2. **Build hulls** :math:`\{H_i\}` offline.  
3. For each configuration :math:`\mathbf q`:
   - Compute :math:`T_i(\mathbf q)` for each link.
   - Transform :math:`H_i \rightarrow T_i(H_i)`.
   - Test all pairs *(i, j)* with *j > i + 1* (skip adjacent links) for intersection.
   - If any pair intersects → **collision**.

API Reference
-------------

.. currentmodule:: ManipulaPy.potential_field

.. autoclass:: CollisionChecker
   :members:
   :undoc-members:
   :inherited-members:

Installation
------------

Ensure dependencies are installed:

.. code-block:: bash

   pip install manipulapy[core] scipy urchin

Usage Examples
--------------

Basic collision check
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.potential_field import CollisionChecker

   # Initialize with your robot URDF
   cc = CollisionChecker("robot.urdf")

   # Test a single joint configuration
   q = [0.0, -0.5, 0.3, 0.0, 0.2, -0.1]
   if cc.check_collision(q):
       print("In collision!")
   else:
       print("Collision-free.")

Batch checking
~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.potential_field import CollisionChecker
   import numpy as np

   cc    = CollisionChecker("robot.urdf")
   poses = np.random.uniform(-0.5, 0.5, size=(100, 6))

   collisions = [cc.check_collision(q) for q in poses]
   print(f"{sum(collisions)} / {len(poses)} configurations in collision")

Integration with trajectory planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning
   from ManipulaPy.potential_field import CollisionChecker

   planner = TrajectoryPlanning(robot, "robot.urdf", dynamics, joint_limits)
   cc      = CollisionChecker("robot.urdf")

   traj = planner.joint_trajectory(q_start, q_goal, Tf=2.0, N=200, method=3)
   safe = []
   for q in traj["positions"]:
       if not cc.check_collision(q):
           safe.append(q)
   print(f"{len(safe)} collision-free waypoints")

Advanced Topics
---------------

Skipping adjacent links
~~~~~~~~~~~~~~~~~~~~~~~

By default, ``check_collision`` skips link pairs that are mechanically **adjacent**
to avoid false positives at shared joints.

Convex hull caching
~~~~~~~~~~~~~~~~~~~

Hulls are built once at initialization. For dynamic meshes, you can rebuild via
``cc._create_convex_hulls()`` (may be expensive; prefer offline caching).

Custom mesh precision
~~~~~~~~~~~~~~~~~~~~~

Simplify (decimate) meshes before hull construction to reduce hull size and speed
up intersection tests.

Troubleshooting
---------------

Mesh loading errors
~~~~~~~~~~~~~~~~~~~

Ensure your URDF’s ``<mesh>`` elements point to files with valid vertex arrays.

False negatives/positives
~~~~~~~~~~~~~~~~~~~~~~~~~

Convex hulls approximate concave geometry. For high precision, refine meshes or
augment with additional proxy geometry.

Performance bottlenecks
~~~~~~~~~~~~~~~~~~~~~~~

- Precompute and cache hulls.  
- Use fewer sample configurations in look-ahead checks.  
- Parallelize ``check_collision`` calls with multiprocessing.

See Also
--------

- :doc:`Trajectory_Planning` — generating joint trajectories
- :doc:`Perception` — environment sensing and obstacle extraction
- :doc:`Simulation` — PyBullet-based simulation

References
----------

- SciPy *ConvexHull* documentation:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
- ``urchin.urdf`` — URDF parser for Python
- Latombe, J.-C., *Robot Motion Planning*, Kluwer, 1991
- Ericson, C., *Real-Time Collision Detection*, Morgan Kaufmann, 2005
