.. _api-perception:

===============================
Perception API Reference
===============================

This page documents **ManipulaPy.perception**, the module for higher-level perception capabilities including obstacle detection, 3D point cloud generation, clustering, and environmental understanding for robotic systems.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Perception`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

Perception Class
================

.. currentmodule:: ManipulaPy.perception

.. autoclass:: Perception
   :members:
   :show-inheritance:

   Higher-level perception module that integrates Vision capabilities with machine learning algorithms for comprehensive environmental understanding and obstacle analysis.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **vision_instance** (*Vision*) -- Required Vision instance for camera operations and low-level processing
   - **logger_name** (*str, optional*) -- Logger identifier (default: "PerceptionLogger")

   **Validation:**
   
   - Raises ValueError if vision_instance is None
   - Initializes logging with configurable debug levels
   - Establishes Vision module dependency relationship

---

Primary Perception Methods
==========================

Obstacle Detection Pipeline
---------------------------

.. automethod:: Perception.detect_and_cluster_obstacles

   Complete environmental analysis pipeline combining vision-based detection with machine learning clustering.

   **Parameters:**
   
   - **camera_index** (*int, optional*) -- Camera identifier for image capture (default: 0)
   - **depth_threshold** (*float, optional*) -- Maximum depth for obstacle consideration in meters (default: 5.0)
   - **step** (*int, optional*) -- Depth image downsampling stride (default: 2)
   - **eps** (*float, optional*) -- DBSCAN neighbor distance threshold (default: 0.1)
   - **min_samples** (*int, optional*) -- DBSCAN minimum cluster size (default: 3)

   **Returns:**
   
   - **obstacle_points** (*np.ndarray*) -- Shape (N, 3) array of 3D obstacle coordinates
   - **labels** (*np.ndarray*) -- Shape (N,) cluster labels with -1 indicating noise

   **Processing Pipeline:**
   
   1. **Image Acquisition**: vision.capture_image(camera_index)
   2. **Depth Validation**: Checks depth.ndim >= 2 and non-None status
   3. **Obstacle Detection**: vision.detect_obstacles() with YOLO integration
   4. **Point Validation**: Verifies non-None and non-empty point arrays
   5. **Clustering Analysis**: DBSCAN application to 3D point coordinates
   6. **Result Logging**: Debug information and cluster statistics

---

Stereo Vision Integration
========================

Disparity Computation
--------------------

.. automethod:: Perception.compute_stereo_disparity

   Wrapper for stereo disparity computation with error handling and logging.

   **Parameters:**
   
   - **left_img** (*np.ndarray*) -- Left camera image array
   - **right_img** (*np.ndarray*) -- Right camera image array

   **Returns:**
   
   - **disparity** (*np.ndarray*) -- Float32 disparity map

   **Dependencies:**
   
   - Requires vision.stereo_enabled = True
   - Delegates to vision.rectify_stereo_images()
   - Delegates to vision.compute_disparity()

Point Cloud Generation
---------------------

.. automethod:: Perception.get_stereo_point_cloud

   3D point cloud generation from stereo image pairs with comprehensive error handling.

   **Parameters:**
   
   - **left_img** (*np.ndarray*) -- Left camera image array
   - **right_img** (*np.ndarray*) -- Right camera image array

   **Returns:**
   
   - **point_cloud** (*np.ndarray*) -- Shape (N, 3) 3D point coordinates

   **Error Handling:**
   
   - Returns empty (0, 3) array if stereo not enabled
   - Logs point cloud size for debugging
   - Delegates processing to vision.get_stereo_point_cloud()

---

Machine Learning Components
===========================

Clustering Analysis
------------------

.. automethod:: Perception.cluster_obstacles

   DBSCAN-based clustering of 3D obstacle points for spatial grouping and noise filtering.

   **Parameters:**
   
   - **points** (*np.ndarray*) -- Shape (N, 3) array of 3D coordinates
   - **eps** (*float, optional*) -- Maximum neighbor distance (default: 0.1)
   - **min_samples** (*int, optional*) -- Minimum cluster density (default: 3)

   **Returns:**
   
   - **labels** (*np.ndarray*) -- Shape (N,) cluster assignments
   - **num_clusters** (*int*) -- Count of identified clusters excluding noise

   **DBSCAN Implementation:**
   
   - Uses sklearn.cluster.DBSCAN with L2 distance metric
   - Noise points labeled as -1
   - Cluster counting excludes noise category
   - Comprehensive logging of cluster statistics

   **Algorithm Details:**
   
   1. **Input Validation**: Empty array handling with early return
   2. **Model Instantiation**: DBSCAN(eps=eps, min_samples=min_samples)
   3. **Fitting**: dbscan_model.fit(points) on 3D coordinates
   4. **Label Extraction**: dbscan_model.labels_ array
   5. **Statistics Computation**: Unique label counting and noise analysis

---

Resource Management
===================

Cleanup Methods
--------------

.. automethod:: Perception.release

   Explicit resource cleanup for Vision instance dependencies.

   **Implementation:**
   
   - Safely calls vision.release() with exception handling
   - Logs cleanup operations for debugging
   - Handles None vision instances gracefully

.. automethod:: Perception.__del__

   Destructor ensuring automatic resource cleanup on object destruction.

   **Safety Features:**
   
   - hasattr() validation before vision access
   - Silent exception handling to prevent destructor failures
   - Automatic cleanup without user intervention

---

Logging Infrastructure
=====================

Logger Configuration
-------------------

.. automethod:: Perception._setup_logger

   Configures structured logging with consistent formatting across the perception pipeline.

   **Parameters:**
   
   - **name** (*str*) -- Logger identifier for message routing

   **Returns:**
   
   - **logger** (*logging.Logger*) -- Configured logger instance

   **Configuration Details:**
   
   - **Level**: DEBUG for comprehensive information
   - **Handler**: StreamHandler for console output
   - **Format**: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   - **Duplicate Prevention**: Checks existing handlers before addition

---

Error Handling Hierarchy
------------------------

1. **Vision Instance Validation**: Constructor-level checking
2. **Image Availability**: Depth image validation with fallbacks
3. **Detection Results**: None-checking with empty array returns
4. **Clustering Input**: Empty point array handling
5. **Resource Cleanup**: Exception-safe cleanup operations

---

Dependencies and Integration
===========================

Required Dependencies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Module
     - Purpose
     - Integration Point
   * - ``sklearn.cluster.DBSCAN``
     - Point clustering
     - cluster_obstacles() method
   * - ``numpy``
     - Array operations
     - All coordinate manipulations
   * - ``logging``
     - Debug/status output
     - Logger configuration and usage
   * - ``ManipulaPy.vision``
     - Low-level vision
     - vision_instance dependency

Vision Module Interface
----------------------

Perception requires Vision instance with specific methods:

- **capture_image()**: RGB/depth image acquisition
- **detect_obstacles()**: YOLO-based object detection
- **stereo_enabled**: Boolean stereo capability flag
- **rectify_stereo_images()**: Stereo rectification
- **compute_disparity()**: Disparity map computation
- **get_stereo_point_cloud()**: 3D reconstruction
- **release()**: Resource cleanup

---


Error Recovery and Robustness
=============================

Vision Failure Handling
-----------------------

- **Missing Depth**: Empty array return with warning logs
- **Detection Failures**: None-result handling with error logs
- **Stereo Unavailability**: Runtime exception with clear messaging

Clustering Robustness
---------------------

- **Empty Input**: Graceful handling with early return
- **Insufficient Points**: DBSCAN handles minimum sample requirements
- **Parameter Validation**: Implicit through sklearn parameter checking

Resource Management Safety
--------------------------

- **Constructor Validation**: Immediate failure for invalid Vision instances
- **Cleanup Exceptions**: Silent handling in destructor context
- **Multiple Cleanup Calls**: Safe repeated release() invocation

---

See Also
========

* :doc:`vision` -- Low-level computer vision and camera management
* :doc:`path_planning` -- Trajectory planning with obstacle avoidance
* :doc:`potential_field` -- Obstacle-aware potential field methods
* :doc:`control` -- Control systems with environmental feedback

External Dependencies
=====================

* `scikit-learn <https://scikit-learn.org/stable/modules/clustering.html#dbscan>`_ -- DBSCAN clustering algorithm
* `NumPy <https://numpy.org/doc/stable/>`_ -- Numerical array operations
* `Python logging <https://docs.python.org/3/library/logging.html>`_ -- Structured logging framework