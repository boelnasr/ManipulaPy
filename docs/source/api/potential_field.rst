.. _api-potential-field:

====================================
Potential Field API Reference
====================================

This page documents **ManipulaPy.potential_field**, the module for potential field path planning with attractive and repulsive potential computations, gradient calculations, and collision checking for robotic manipulator motion planning.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Potential_Field`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

PotentialField Class
====================

.. currentmodule:: ManipulaPy.potential_field

.. autoclass:: PotentialField
   :members:
   :show-inheritance:

   Main class for artificial potential field computations including attractive and repulsive potential calculations with configurable gain parameters.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **attractive_gain** (*float, optional*) -- Gain coefficient for attractive potential (default: 1.0)
   - **repulsive_gain** (*float, optional*) -- Gain coefficient for repulsive potential (default: 100.0)
   - **influence_distance** (*float, optional*) -- Maximum distance for repulsive influence (default: 0.5)

---

Potential Field Computations
============================

Attractive Potential
-------------------

.. automethod:: PotentialField.compute_attractive_potential

   Computes quadratic attractive potential energy toward goal configuration.
   
   **Parameters:**
   
   - **q** (*numpy.ndarray*) -- Current configuration vector
   - **q_goal** (*numpy.ndarray*) -- Goal configuration vector

   **Returns:**
   
   - **float** -- Attractive potential energy

   **Mathematical Implementation:**
   
   .. math::
      U_{att}(q) = \frac{1}{2} \xi \|q - q_{goal}\|^2

   Where ξ is the attractive_gain parameter.

Repulsive Potential
------------------

.. automethod:: PotentialField.compute_repulsive_potential

   Computes repulsive potential energy from obstacle configurations.
   Uses inverse distance squared repulsion with influence distance cutoff.

   **Parameters:**
   
   - **q** (*numpy.ndarray*) -- Current configuration vector
   - **obstacles** (*list*) -- List of obstacle configuration vectors

   **Returns:**
   
   - **float** -- Total repulsive potential energy

   **Mathematical Implementation:**
   
   .. math::
      U_{rep}(q) = 10 \sum_{i} 2\eta \left(\frac{1}{\rho_i} - \frac{1}{\rho_0}\right)^2

   Where:
   - η is the repulsive_gain parameter
   - ρᵢ is the distance to obstacle i
   - ρ₀ is the influence_distance parameter
   - Scaling factor of 10 applied to final result

Gradient Computation
-------------------

.. automethod:: PotentialField.compute_gradient

   Computes the total gradient vector of the potential field for navigation.
   Combines attractive and repulsive gradient components.

   **Parameters:**
   
   - **q** (*numpy.ndarray*) -- Current configuration vector
   - **q_goal** (*numpy.ndarray*) -- Goal configuration vector
   - **obstacles** (*list*) -- List of obstacle configuration vectors

   **Returns:**
   
   - **numpy.ndarray** -- Total gradient vector

   **Implementation Details:**
   
   **Attractive Gradient:**
   
   .. math::
      \nabla U_{att}(q) = \xi (q - q_{goal})

   **Repulsive Gradient:**
   
   .. math::
      \nabla U_{rep}(q) = \sum_{i} 5\eta \left(\frac{1}{\rho_i} - \frac{1}{\rho_0}\right) \frac{1}{\rho_i^3} (q - q_{obs,i})

   **Total Gradient:**
   
   .. math::
      \nabla U_{total}(q) = \nabla U_{att}(q) + \nabla U_{rep}(q)

---

CollisionChecker Class
======================

.. autoclass:: CollisionChecker
   :members:
   :show-inheritance:

   URDF-based collision detection system using convex hull approximations for robotic manipulator self-collision and environment collision checking.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **urdf_path** (*str*) -- File path to robot URDF description

   **Initialization Process:**
   
   1. Loads URDF using urchin.urdf.URDF.load()
   2. Extracts visual geometry from robot links
   3. Generates convex hulls for collision approximation
   4. Stores hull dictionary indexed by link names

---

Collision Detection Methods
===========================

Hull Generation
--------------

.. automethod:: CollisionChecker._create_convex_hulls

   Generates convex hull approximations from URDF visual mesh data.
   
   **Returns:**
   
   - **dict** -- Dictionary mapping link names to ConvexHull objects

   **Processing Pipeline:**
   
   1. Iterates through robot.links
   2. Extracts visual geometry meshes
   3. Validates mesh.vertices attribute existence
   4. Constructs scipy.spatial.ConvexHull from vertices
   5. Associates hulls with link names

Geometric Transformation
-----------------------

.. automethod:: CollisionChecker._transform_convex_hull

   Transforms convex hull points using homogeneous transformation matrices.

   **Parameters:**
   
   - **convex_hull** (*ConvexHull*) -- Original convex hull object
   - **transform** (*numpy.ndarray*) -- 4×4 homogeneous transformation matrix

   **Returns:**
   
   - **ConvexHull** -- Transformed convex hull object

   **Mathematical Implementation:**
   
   .. math::
      P_{transformed} = R \cdot P_{original}^T + t

   Where:
   - R is the 3×3 rotation matrix (transform[:3, :3])
   - t is the 3×1 translation vector (transform[:3, 3])
   - P represents point coordinates

Collision Detection
------------------

.. automethod:: CollisionChecker.check_collision

   Performs comprehensive self-collision detection for given joint configuration.

   **Parameters:**
   
   - **thetalist** (*numpy.ndarray*) -- Joint angle configuration vector

   **Returns:**
   
   - **bool** -- True if collision detected, False otherwise

   **Detection Algorithm:**
   
   1. **Forward Kinematics**: Compute link transforms using robot.link_fk(cfg=thetalist)
   2. **Hull Transformation**: Transform each link's convex hull to world coordinates
   3. **Pairwise Intersection**: Check all link pairs for hull intersections
   4. **Self-Exclusion**: Skip collision checks between identical links
   5. **Boolean Result**: Return True on first intersection found

---

Data Structures and Internal Representation
===========================================

Convex Hull Storage
------------------

Internal convex_hulls dictionary structure:

.. code-block:: python

   convex_hulls = {
       "link_name_1": ConvexHull(vertices_1),
       "link_name_2": ConvexHull(vertices_2),
       # ... additional links
   }

ConvexHull Properties
--------------------

Each ConvexHull object contains:

- **points**: Original vertex coordinates
- **vertices**: Indices of vertices forming the hull
- **simplices**: Triangular faces of the hull
- **equations**: Hyperplane equations for faces

Forward Kinematics Integration
-----------------------------

Integration with urchin URDF processing:

- **Input**: Joint configuration (thetalist)
- **Method**: robot.link_fk(cfg=thetalist)
- **Output**: Dictionary {link_name: 4×4_transform_matrix}

---

Computational Complexity Analysis
=================================

Potential Field Methods
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Method
     - Complexity
     - Dominant Operations
   * - ``compute_attractive_potential``
     - O(n)
     - Vector subtraction + norm
   * - ``compute_repulsive_potential``
     - O(k×n)
     - k obstacles × distance computation
   * - ``compute_gradient``
     - O(k×n)
     - k obstacles × gradient calculation

Collision Detection Methods
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Method
     - Complexity
     - Dominant Operations
   * - ``_create_convex_hulls``
     - O(m×v³)
     - m links × ConvexHull(v vertices)
   * - ``_transform_convex_hull``
     - O(v)
     - v vertices × matrix multiplication
   * - ``check_collision``
     - O(L²×v)
     - L² link pairs × hull intersection

Where:
- n: Configuration space dimensions
- k: Number of obstacles
- m: Number of robot links
- v: Average vertices per mesh
- L: Number of links with visual geometry

---

Numerical Implementation Details
===============================

Potential Field Scaling
-----------------------

Applied scaling factors in implementation:

- **Repulsive potential**: 10× multiplier on final result
- **Repulsive gradient**: 5× multiplier on gradient magnitude
- **Distance threshold**: influence_distance parameter cutoff

Distance Computations
--------------------

All distance calculations use numpy.linalg.norm():

- **Method**: Euclidean L2 norm
- **Input**: Configuration space vectors
- **Precision**: Double precision floating point

Matrix Operations
----------------

Convex hull transformations use optimized NumPy operations:

- **Rotation**: 3×3 matrix multiplication with broadcasted points
- **Translation**: Vector addition with reshaped translation vector
- **Memory layout**: Contiguous arrays for efficient computation

Intersection Detection
---------------------

Hull intersection testing relies on scipy.spatial.ConvexHull:

- **Algorithm**: Built-in intersection testing methods
- **Precision**: Computational geometry tolerance handling
- **Robustness**: Handles degenerate cases automatically

---

Error Handling and Edge Cases
=============================

URDF Processing Errors
----------------------

- **Missing files**: URDF.load() exception handling
- **Invalid geometry**: hasattr() validation for mesh.vertices
- **Empty meshes**: Automatic skipping of links without valid geometry

Convex Hull Failures
--------------------

- **Coplanar points**: scipy.spatial.ConvexHull internal handling
- **Insufficient vertices**: Minimum 4 points required for 3D hulls
- **Numerical precision**: Tolerance-based geometric computations

Potential Field Singularities
-----------------------------

- **Zero distance**: Protected by influence_distance > 0 requirement
- **Infinite gradients**: Prevented by distance threshold checking
- **Numerical overflow**: Managed through finite gain parameters

---

Memory Management
================

Convex Hull Storage
------------------

Memory allocation for collision checking:

- **Hull objects**: Persistent storage throughout object lifetime
- **Transformed hulls**: Temporary allocation during collision checks
- **Point arrays**: Copy operations for transformation safety

Potential Field Computations
----------------------------

Memory usage characteristics:

- **Gradient arrays**: Allocated once, reused for multiple obstacles
- **Distance scalars**: Temporary variables with automatic cleanup
- **Configuration vectors**: Input parameter references (no copying)

---

See Also
========

* :doc:`path_planning` -- Trajectory planning integration with potential fields
* :doc:`kinematics` -- Forward kinematics for collision checking
* :doc:`utils` -- Mathematical utilities for vector operations
* :doc:`urdf_processor` -- URDF loading and processing capabilities

External Dependencies
=====================

* `urchin <https://github.com/fishbotics/urchin>`_ -- URDF processing library
* `scipy.spatial <https://docs.scipy.org/doc/scipy/reference/spatial.html>`_ -- Convex hull computations
* `NumPy <https://numpy.org/doc/stable/>`_ -- Numerical array operations