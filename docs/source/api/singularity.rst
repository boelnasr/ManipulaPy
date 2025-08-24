.. _api-singularity:

===============================
Singularity API Reference
===============================

This page documents **ManipulaPy.singularity**, the module for comprehensive singularity analysis of robotic manipulators including detection, visualization, and workspace analysis with optional CUDA acceleration.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Singularity_Analysis `.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

Singularity Class
=================

.. currentmodule:: ManipulaPy.singularity

.. autoclass:: Singularity
   :members:
   :show-inheritance:

   Main class for singularity analysis of robotic manipulators providing detection, manipulability analysis, workspace visualization, and condition number computations.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **serial_manipulator** (*SerialManipulator*) -- Instance of SerialManipulator for kinematic computations

---

Singularity Detection
=====================

Primary Detection Methods
-------------------------

.. automethod:: Singularity.singularity_analysis

   Analyzes if the manipulator is at a singularity by computing the determinant of the Jacobian matrix.
   Uses a threshold of 1e-4 for determinant comparison.

   **Parameters:**
   
   - **thetalist** (*numpy.ndarray*) -- Array of joint angles in radians

   **Returns:**
   
   - **bool** -- True if at singularity (|det(J)| < 1e-4), False otherwise

.. automethod:: Singularity.near_singularity_detection

   Detects proximity to singularities using condition number analysis.
   
   **Parameters:**
   
   - **thetalist** (*numpy.ndarray*) -- Array of joint angles in radians
   - **threshold** (*float, optional*) -- Condition number threshold (default: 1e-2)

   **Returns:**
   
   - **bool** -- True if near singularity (condition number > threshold), False otherwise

Condition Analysis
-----------------

.. automethod:: Singularity.condition_number

   Computes the condition number of the Jacobian matrix using singular value decomposition.
   
   **Parameters:**
   
   - **thetalist** (*numpy.ndarray*) -- Array of joint angles in radians

   **Returns:**
   
   - **float** -- Condition number of the Jacobian matrix

---

Manipulability Analysis
======================

Ellipsoid Visualization
----------------------

.. automethod:: Singularity.manipulability_ellipsoid

   Generates and plots manipulability ellipsoids for linear and angular velocity components.
   Uses SVD to compute ellipsoid radii and orientations from Jacobian submatrices.

   **Parameters:**
   
   - **thetalist** (*numpy.ndarray*) -- Array of joint angles in radians
   - **ax** (*matplotlib.axes._subplots.Axes3DSubplot, optional*) -- 3D axis for plotting

   **Technical Details:**
   
   - Separates Jacobian into linear (J[:3,:]) and angular (J[3:,:]) components
   - Computes SVD for each component: U, S, V = svd(J_component)
   - Ellipsoid radii calculated as 1/sqrt(singular_values)
   - Generates sphere points and transforms using U and radii
   - Creates dual plot with blue (linear) and red (angular) ellipsoids

---

Workspace Analysis
==================

Monte Carlo Methods
-------------------

.. automethod:: Singularity.plot_workspace_monte_carlo

   Estimates robot workspace using CUDA-accelerated Monte Carlo sampling.
   Generates random joint configurations and computes reachable end-effector positions.

   **Parameters:**
   
   - **joint_limits** (*list*) -- List of (min, max) tuples for each joint
   - **num_samples** (*int, optional*) -- Number of Monte Carlo samples (default: 10000)

   **CUDA Implementation Details:**
   
   - Uses numba.cuda for GPU acceleration
   - Random number generation via xoroshiro128p_uniform_float32
   - Parallel joint angle sampling with configurable thread blocks
   - Thread block size: 256 threads per block
   - Grid size: (num_samples + 255) // 256 blocks

   **Processing Pipeline:**
   
   1. Initialize CUDA device arrays for joint samples
   2. Generate random states with create_xoroshiro128p_states
   3. Launch CUDA kernel for parallel sampling
   4. Copy samples to host and compute forward kinematics
   5. Generate convex hull from workspace points
   6. Visualize using matplotlib plot_trisurf

---

Mathematical Implementation
===========================

Jacobian Analysis
----------------

All methods rely on the Jacobian matrix computation from the SerialManipulator:

- **Frame**: Uses "space" frame for consistency

- **Dimensions**: 6×n matrix (6 DOF × n joints)

- **Components**: Upper 3 rows (linear), lower 3 rows (angular)

Determinant Computation
----------------------

Singularity detection uses numpy.linalg.det() with absolute value comparison:

- **Threshold**: 1e-4 (configurable in source)

- **Method**: Direct determinant calculation

- **Precision**: Double precision floating point

Condition Number Calculation
---------------------------

Uses numpy.linalg.cond() which implements:

- **Method**: Ratio of largest to smallest singular values

- **Algorithm**: SVD-based computation

- **Stability**: Numerically stable for ill-conditioned matrices

SVD Implementation
-----------------

Manipulability ellipsoid generation uses numpy.linalg.svd():

- **Decomposition**: J = U @ diag(S) @ V.T

- **Radii computation**: 1.0 / sqrt(S)

- **Transformation**: Ellipsoid points = U @ diag(radii) @ sphere_points

Convex Hull Generation
---------------------

Workspace visualization uses scipy.spatial.ConvexHull:

- **Algorithm**: Quickhull algorithm

- **Input**: N×3 array of workspace points

- **Output**: Triangulated surface mesh

- **Visualization**: Matplotlib plot_trisurf with viridis colormap

---

Performance Characteristics
===========================

Computational Complexity
------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Method
     - Complexity
     - Primary Operations
   * - ``singularity_analysis``
     - O(n³)
     - Jacobian + determinant
   * - ``condition_number``
     - O(n³)
     - Jacobian + SVD
   * - ``manipulability_ellipsoid``
     - O(n³ + m)
     - 2×SVD + sphere generation
   * - ``plot_workspace_monte_carlo``
     - O(k×n²)
     - k×forward_kinematics + convex_hull

Memory Requirements
------------------

CUDA kernel memory allocation:

- **joint_samples**: num_samples × num_joints × float32

- **rng_states**: num_samples × state_size

- **device_joint_limits**: num_joints × 2 × float32

Thread Configuration
-------------------

CUDA kernel execution parameters:

- **threads_per_block**: 256 (fixed)

- **blocks_per_grid**: ceil(num_samples / 256)

- **total_threads**: blocks_per_grid × threads_per_block

---

Data Flow Architecture
=====================

Singularity Analysis Pipeline
----------------------------

1. **Input**: Joint angles (numpy.ndarray)

2. **Jacobian Computation**: SerialManipulator.jacobian()

3. **Analysis**: Determinant or condition number calculation

4. **Output**: Boolean or float result

Manipulability Ellipsoid Pipeline
---------------------------------

1. **Input**: Joint angles + optional axis

2. **Jacobian Decomposition**: Split into linear/angular components

3. **SVD Analysis**: Compute singular values and vectors

4. **Ellipsoid Generation**: Transform unit sphere using SVD results

5. **Visualization**: Matplotlib 3D surface plotting

Workspace Analysis Pipeline
---------------------------

1. **CUDA Setup**: Initialize device memory and random states

2. **Parallel Sampling**: Generate random joint configurations

3. **Host Transfer**: Copy samples from GPU to CPU

4. **Forward Kinematics**: Batch computation of end-effector positions

5. **Convex Hull**: Geometric analysis of workspace boundary

6. **Visualization**: 3D triangulated surface rendering

---

Error Handling and Validation
=============================

Input Validation
----------------

- **Joint angles**: Verified as numpy arrays with correct dimensions

- **Joint limits**: Validated as list of tuples with numeric values

- **Thresholds**: Checked for positive numeric values

Numerical Stability
------------------

- **Singular matrices**: Handled gracefully in condition number computation

- **Zero determinants**: Managed with absolute value thresholding

- **Ill-conditioned systems**: SVD provides robust decomposition

CUDA Error Management
--------------------

- **Memory allocation**: Automatic cleanup on kernel completion

- **Thread synchronization**: Implicit synchronization after kernel launch

- **Device compatibility**: Runtime detection of CUDA availability

---

See Also
========

* :doc:`kinematics` -- SerialManipulator class for Jacobian computation
* :doc:`utils` -- Mathematical utilities and matrix operations
* :doc:`path_planning` -- Trajectory planning with singularity considerations
* :doc:`control` -- Control algorithms affected by singularities