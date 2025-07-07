Collision Checker Module User Guide
===================================

This guide covers the `CollisionChecker` class in ManipulaPy, which performs
geometry-based self-collision detection for serial manipulators using URDF meshes.

.. note::
   This guide uses Python 3.10.12, SciPy 1.10+, and the `urchin.urdf` parser
   to load URDF meshes and compute convex hulls for collision tests.

Introduction
------------

Collision checking is essential for safe motion planning.  The
`CollisionChecker` builds convex‐hull approximations of each link’s visual mesh
and tests for pairwise intersection at a given robot configuration.

Key features:

- **Convex hull construction** from URDF mesh vertices  
- **Fast pairwise collision tests** using SciPy’s `ConvexHull.intersects`  
- **Transforms hulls** by forward‐kinematics link poses  
- **Self-collision detection** between any two non-adjacent links  

Mathematical Background
-----------------------

Given a robot configuration :math:`\mathbf q` (joint angles), the forward‐kinematics
function :math:`T_i(\mathbf q)\in SE(3)` returns the homogeneous transform of link *i*.
Each link’s mesh is approximated by its convex hull :math:`H_i`.  Under transform
:math:`T_i`, the hull vertices move to

.. math::
   \{\,T_i(\mathbf q)\,p \mid p\in H_i.\}

Two convex polyhedra intersect iff their convex hulls intersect.  SciPy’s
`ConvexHull` can test intersection via oriented‐bounding‐boxes or
half‐space tests under the hood.

Workflow:

1. **Load URDF** → extract mesh vertices.  
2. **Build hulls** :math:`\{H_i\}` offline.  
3. For each configuration :math:`\mathbf q`:  
   - Compute :math:`T_i(\mathbf q)` for each link.  
   - Transform :math:`H_i` → :math:`T_i(H_i)`.  
   - Test all pairs *(i,j)*, *j>i+1* (skip adjacent links) for intersection.  
   - If any pair intersects → collision.  

Class Reference
---------------

.. autoclass:: ManipulaPy.potential_field.CollisionChecker
   :members:
   :undoc-members:
   :inherited-members:

Installation
------------

Ensure dependencies are installed:

.. code-block:: bash

   pip install ManipulaPy[core] scipy urchin

Usage Examples
--------------

1. **Basic collision check**

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

2. **Batch checking**

   .. code-block:: python

      from ManipulaPy.potential_field import CollisionChecker
      import numpy as np

      cc   = CollisionChecker("robot.urdf")
      poses = np.random.uniform(-0.5, 0.5, size=(100, 6))

      collisions = [cc.check_collision(q) for q in poses]
      print(f"{sum(collisions)} / {len(poses)} configurations in collision")

3. **Integration with path planning**

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

- **Skipping Adjacent Links**  
  By default, `check_collision` skips link pairs that are mechanically adjacent to
  avoid false positives at shared joints.

- **Convex Hull Caching**  
  Hulls are built once at initialization.  For dynamic meshes, you can rebuild
  via `cc._create_convex_hulls()`.

- **Custom Mesh Precision**  
  Simplify the mesh before hull construction (e.g., via Blender decimation) to
  speed up hull generation.

Troubleshooting
---------------

- **Mesh loading errors**  
  Ensure your URDF’s `<mesh>` elements point to files with valid vertex arrays.

- **False negatives/positives**  
  Convex hulls approximate concave geometry.  For high‐precision needs,
  subdivide large faces or use finer mesh resolution.

- **Performance bottleneck**  
  - Precompute all hulls offline.  
  - Use fewer sample configurations in look-ahead checks.  
  - Parallelize `check_collision` calls with multiprocessing.

References
----------

- SciPy ConvexHull documentation:  
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html  
- urchin.urdf — URDF parser for Python  
- Latombe, J.-C., _Robot Motion Planning_, Kluwer, 1991  
- Ericson, C., _Real-Time Collision Detection_, Morgan Kaufmann, 2005  
