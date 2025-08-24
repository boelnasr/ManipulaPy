.. _api-vision:

===========================
Vision API Reference
===========================

This page documents **ManipulaPy.vision**, the module for computer vision capabilities including stereo vision, object detection, camera calibration, and PyBullet integration with optional YOLO object detection.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/vision`.

---

Quick Navigation
================

.. contents::
   :local:
   :depth: 2

---

Vision Class
============

.. currentmodule:: ManipulaPy.vision

.. autoclass:: Vision
   :members:
   :show-inheritance:

   Unified vision system for monocular/stereo cameras with PyBullet integration, stereo processing pipeline, and YOLO-based object detection capabilities.

   .. rubric:: Constructor

   .. automethod:: __init__

   **Parameters:**
   
   - **camera_configs** (*list of dict, optional*) -- Camera configuration dictionaries
   - **stereo_configs** (*tuple(dict, dict), optional*) -- Left and right camera configurations for stereo
   - **use_pybullet_debug** (*bool, optional*) -- Enable PyBullet debug sliders (default: False)
   - **show_plot** (*bool, optional*) -- Display debug camera feed in matplotlib (default: True)
   - **logger_name** (*str, optional*) -- Logger identifier (default: "VisionSystemLogger")
   - **physics_client** (*int, optional*) -- PyBullet client ID for simulation integration

   **Camera Configuration Structure:**
   
   Each camera_config dictionary contains:
   
   - **name** (*str*) -- Camera identifier
   - **translation** (*list*) -- [x, y, z] position in world coordinates
   - **rotation** (*list*) -- [roll_deg, pitch_deg, yaw_deg] orientation angles
   - **fov** (*float*) -- Field of view in degrees
   - **near** (*float*) -- Near clipping plane distance
   - **far** (*float*) -- Far clipping plane distance
   - **intrinsic_matrix** (*np.ndarray*) -- 3×3 camera intrinsic matrix
   - **distortion_coeffs** (*np.ndarray*) -- 5-element distortion coefficient array
   - **use_opencv** (*bool*) -- Enable OpenCV device capture
   - **device_index** (*int*) -- OpenCV device identifier

---

Camera Management
=================

Configuration Methods
--------------------

.. automethod:: Vision._configure_camera

   Internal method for storing camera configurations and initializing OpenCV capture devices.

   **Parameters:**
   
   - **idx** (*int*) -- Camera index for internal storage
   - **cfg** (*dict*) -- Camera configuration dictionary

   **Processing:**
   
   1. Extracts configuration parameters with defaults
   2. Builds extrinsic transformation matrix
   3. Stores camera parameters in self.cameras[idx]
   4. Initializes OpenCV VideoCapture if use_opencv=True
   5. Validates device accessibility and logs status

.. automethod:: Vision._validate_stereo_config

   Validates stereo camera configuration completeness.

   **Parameters:**
   
   - **left_cfg** (*dict*) -- Left camera configuration
   - **right_cfg** (*dict*) -- Right camera configuration

   **Required Keys:**
   
   - intrinsic_matrix, distortion_coeffs, translation, rotation

Extrinsic Matrix Computation
---------------------------

.. automethod:: Vision._make_extrinsic_matrix

   Constructs 4×4 homogeneous transformation matrix from translation and Euler angles.

   **Parameters:**
   
   - **translation** (*list*) -- [x, y, z] translation vector
   - **rotation_deg** (*list*) -- [roll, pitch, yaw] in degrees

   **Returns:**
   
   - **np.ndarray** -- 4×4 extrinsic transformation matrix

.. automethod:: Vision._euler_to_rotation_matrix

   Converts Euler angles to 3×3 rotation matrix using ZYX convention.

   **Parameters:**
   
   - **euler_deg** (*list*) -- [roll_deg, pitch_deg, yaw_deg]

   **Returns:**
   
   - **np.ndarray** -- 3×3 rotation matrix (float32)

   **Mathematical Implementation:**
   
   .. math::
      R = R_z(\text{yaw}) \cdot R_y(\text{pitch}) \cdot R_x(\text{roll})

---

Image Acquisition
=================

PyBullet Integration
-------------------

.. automethod:: Vision.capture_image

   Captures RGB and depth images from PyBullet cameras with automatic depth scaling.

   **Parameters:**
   
   - **camera_index** (*int, optional*) -- Camera identifier (default: 0)

   **Returns:**
   
   - **rgb** (*np.ndarray*) -- RGB image array (H, W, 3) uint8
   - **depth** (*np.ndarray*) -- Depth image array (H, W) float32

   **Processing Pipeline:**
   
   1. Validates camera_index existence in self.cameras
   2. Constructs view matrix from camera translation and fixed target
   3. Builds projection matrix from camera parameters
   4. Calls pb.getCameraImage() with computed matrices
   5. Processes RGBA to RGB conversion
   6. Applies depth scaling: depth = near + (far - near) × normalized_depth

---

Object Detection
================

YOLO Integration
---------------

.. automethod:: Vision.detect_obstacles

   YOLO-based object detection with 3D position estimation using depth information.

   **Parameters:**
   
   - **depth_image** (*np.ndarray*) -- Depth image for 3D positioning
   - **rgb_image** (*np.ndarray*) -- RGB image for YOLO detection
   - **depth_threshold** (*float, optional*) -- Maximum depth consideration (default: 0.0)
   - **camera_index** (*int, optional*) -- Camera for intrinsic parameters (default: 0)
   - **step** (*int, optional*) -- Depth downsampling stride (default: 5)

   **Returns:**
   
   - **positions** (*np.ndarray*) -- Shape (N, 3) 3D object positions
   - **orientations** (*np.ndarray*) -- Shape (N,) XY-plane orientation angles

   **Detection Pipeline:**
   
   1. **YOLO Inference**: self.yolo_model(rgb_image, conf=0.3)

   2. **Bounding Box Extraction**: box.xyxy[0].tolist() coordinates

   3. **Depth Analysis**: Median depth within bounding box region

   4. **3D Reprojection**: Camera intrinsics-based coordinate transformation

   5. **Orientation Estimation**: np.arctan2(y_cam, x_cam) in XY plane

   **3D Transformation:**
   
   .. math::
      \begin{align}
      x_{cam} &= (c_x - c_{x,intrinsic}) \cdot \frac{d}{f_x} \\
      y_{cam} &= (c_y - c_{y,intrinsic}) \cdot \frac{d}{f_y} \\
      z_{cam} &= d
      \end{align}

   Where (c_x, c_y) is bounding box center, d is median depth, and f_x, f_y are focal lengths.

---

Stereo Vision Pipeline
=====================

Rectification Setup
------------------

.. automethod:: Vision.compute_stereo_rectification_maps

   Computes stereo rectification maps using OpenCV stereoRectify algorithm.

   **Parameters:**
   
   - **image_size** (*tuple, optional*) -- (width, height) for rectification (default: (640, 480))

   **Implementation Details:**
   
   1. **Parameter Extraction**: Intrinsics K1, K2 and distortions D1, D2

   2. **Relative Geometry**: R_lr = R_right @ R_left.T, t_lr = t_right - R_lr @ t_left

   3. **Type Unification**: Convert all inputs to float64 for cv2.stereoRectify

   4. **Rectification Computation**: cv2.stereoRectify with CALIB_ZERO_DISPARITY

   5. **Map Generation**: cv2.initUndistortRectifyMap for both cameras

   6. **Q Matrix Storage**: Disparity-to-depth transformation matrix

Image Processing
---------------

.. automethod:: Vision.rectify_stereo_images

   Applies rectification maps to stereo image pairs.

   **Parameters:**
   
   - **left_img** (*np.ndarray*) -- Left camera image
   - **right_img** (*np.ndarray*) -- Right camera image

   **Returns:**
   
   - **left_rect** (*np.ndarray*) -- Rectified left image
   - **right_rect** (*np.ndarray*) -- Rectified right image

   **Requirements:**
   
   - stereo_enabled = True
   - Rectification maps computed via compute_stereo_rectification_maps()

.. automethod:: Vision.compute_disparity

   Computes disparity map using StereoSGBM algorithm.

   **Parameters:**
   
   - **left_rect** (*np.ndarray*) -- Rectified left image
   - **right_rect** (*np.ndarray*) -- Rectified right image

   **Returns:**
   
   - **disparity** (*np.ndarray*) -- Float32 disparity map

   **StereoSGBM Configuration:**
   
   - minDisparity: 0
   - numDisparities: 64
   - blockSize: 7
   - P1: 8 × 3 × 7²
   - P2: 32 × 3 × 7²
   - Fixed-point scaling: division by 16.0

3D Reconstruction
----------------

.. automethod:: Vision.disparity_to_pointcloud

   Reprojects disparity map to 3D point cloud using Q matrix.

   **Parameters:**
   
   - **disparity** (*np.ndarray*) -- Disparity map from compute_disparity()

   **Returns:**
   
   - **cloud_filtered** (*np.ndarray*) -- Shape (N, 3) valid 3D points

   **Filtering Criteria:**
   
   - Positive disparity values (disp_flat > 0)
   - Finite coordinates (np.isfinite(cloud[:, 0]))
   - Reasonable depth (cloud[:, 2] < 10.0)

.. automethod:: Vision.get_stereo_point_cloud

   High-level stereo processing pipeline combining rectification, disparity, and reprojection.

   **Parameters:**
   
   - **left_img** (*np.ndarray*) -- Left camera image
   - **right_img** (*np.ndarray*) -- Right camera image

   **Returns:**
   
   - **point_cloud** (*np.ndarray*) -- Shape (N, 3) 3D point coordinates

   **Pipeline Sequence:**
   
   1. rectify_stereo_images(left_img, right_img)
   2. compute_disparity(left_rect, right_rect)
   3. disparity_to_pointcloud(disparity)

---

PyBullet Debug Interface
=======================

Debug Slider Setup
-----------------

.. automethod:: Vision._setup_pybullet_debug_sliders

   Creates PyBullet debug sliders for interactive virtual camera control.

   **Slider Categories:**
   
   **View Matrix Parameters:**
   
   - target_x, target_y, target_z: Camera target position
   - distance: Camera distance from target
   - yaw, pitch, roll: Camera orientation angles
   - upAxisIndex: Up axis selection (0 or 1)

   **Projection Matrix Parameters:**
   
   - width, height: Image resolution
   - fov: Field of view angle
   - near_val, far_val: Clipping plane distances
   - print: Debug output trigger

.. automethod:: Vision._get_pybullet_view_proj

   Reads debug slider values and constructs view/projection matrices.

   **Returns:**
   
   - **view_mtx** (*list*) -- PyBullet view matrix
   - **proj_mtx** (*list*) -- PyBullet projection matrix  
   - **width** (*int*) -- Image width
   - **height** (*int*) -- Image height

   **Matrix Construction:**
   
   - View: pb.computeViewMatrixFromYawPitchRoll()
   - Projection: pb.computeProjectionMatrixFOV()
   - Debug logging on print button toggle

---

Resource Management
==================

Cleanup Methods
--------------

.. automethod:: Vision.release

   Releases OpenCV capture device resources.

   **Operations:**
   
   - Iterates through self.capture_devices
   - Calls cap.release() for each VideoCapture object
   - Clears capture_devices dictionary
   - Logs cleanup status for each device

.. automethod:: Vision.__del__

   Destructor with robust error handling for graceful resource cleanup.

   **Safety Features:**
   
   - Validates logger existence and handler availability
   - Checks stream closure status before logging
   - Silent exception handling to prevent destructor failures
   - Guaranteed resource cleanup regardless of logging success

---

Utility Functions
================

.. currentmodule:: ManipulaPy.vision

.. autofunction:: read_debug_parameters

   Utility function for reading PyBullet debug parameter values.

   **Parameters:**
   
   - **dbg_params** (*dict*) -- Dictionary mapping parameter names to PyBullet IDs

   **Returns:**
   
   - **values** (*dict*) -- Dictionary mapping parameter names to current values

   **Implementation:**
   
   Iterates through dbg_params and calls pb.readUserDebugParameter() for each ID.

---

Data Structures and Configuration
=================================

Internal Storage Format
----------------------

**Camera Storage (self.cameras):**

.. code-block:: python

   self.cameras[index] = {
       "name": str,
       "translation": [x, y, z],
       "rotation": [roll_deg, pitch_deg, yaw_deg],
       "fov": float,
       "near": float,
       "far": float,
       "intrinsic_matrix": np.ndarray(3, 3),
       "distortion_coeffs": np.ndarray(5,),
       "use_opencv": bool,
       "device_index": int,
       "extrinsic_matrix": np.ndarray(4, 4)
   }

**Stereo Configuration Attributes:**

.. code-block:: python

   # Stereo processing state
   self.stereo_enabled: bool
   self.left_cam_cfg: dict
   self.right_cam_cfg: dict
   
   # Rectification maps
   self.left_map_x: np.ndarray
   self.left_map_y: np.ndarray
   self.right_map_x: np.ndarray
   self.right_map_y: np.ndarray
   
   # 3D reconstruction
   self.Q: np.ndarray  # 4x4 disparity-to-depth matrix
   self.stereo_matcher: cv2.StereoSGBM

Default Camera Configuration
---------------------------

When no camera_configs provided, uses:

.. code-block:: python

   default_config = {
       "name": "default_camera",
       "translation": [0, 0, 0],
       "rotation": [0, 0, 0],
       "fov": 60,
       "near": 0.1,
       "far": 5.0,
       "intrinsic_matrix": [[500, 0, 320],
                           [0, 500, 240],
                           [0, 0, 1]],
       "distortion_coeffs": [0, 0, 0, 0, 0],
       "use_opencv": False,
       "device_index": 0
   }

---

Error Handling and Validation
=============================

YOLO Model Management
--------------------

- **Loading Failure**: Sets yolo_model = None, continues operation

- **Detection Fallback**: Returns empty arrays when YOLO unavailable

- **Input Validation**: Checks rgb_image and depth_image for None/invalid

Stereo Processing Errors
------------------------

- **Configuration Validation**: Checks required keys in stereo configs

- **Runtime State Checking**: Validates stereo_enabled before operations

- **Map Initialization**: Ensures rectification maps computed before use

OpenCV Device Handling
----------------------

- **Device Access Failure**: Raises RuntimeError with descriptive message

- **Capture Validation**: Uses cap.isOpened() to verify device accessibility

- **Resource Cleanup**: Automatic release in destructor and explicit method

PyBullet Integration Safety
--------------------------

- **Parameter Reading**: Safe handling of missing debug parameters

- **Matrix Validation**: Debug logging for view/projection matrix inspection

- **Client State**: No assumptions about PyBullet connection status

---

Performance Considerations
==========================

Memory Management
----------------

- **Image Arrays**: Contiguous memory layout for OpenCV operations

- **Rectification Maps**: Persistent storage for repeated stereo processing

- **Point Clouds**: Filtered arrays to reduce memory footprint

Computational Efficiency
-----------------------

- **YOLO Inference**: Single forward pass per image

- **Stereo Matching**: StereoSGBM optimized for quality/speed balance

- **Depth Scaling**: Vectorized operations for range conversion

Threading Considerations
-----------------------

- **OpenCV Capture**: Single-threaded device access

- **YOLO Processing**: GPU acceleration when available

- **PyBullet Integration**: Main thread simulation access required

---

See Also
========

* :doc:`perception` -- Higher-level perception capabilities using Vision
* :doc:`utils` -- Mathematical utilities for transformations
* :doc:`simulation` -- PyBullet simulation integration
* :doc:`potential_field` -- Obstacle avoidance using vision data

External Dependencies
=====================

* `OpenCV <https://opencv.org/>`_ -- Computer vision algorithms and stereo processing
* `PyBullet <https://pybullet.org/>`_ -- Physics simulation and camera rendering
* `Ultralytics YOLO <https://ultralytics.com/>`_ -- Object detection framework
* `NumPy <https://numpy.org/>`_ -- Numerical array operations
* `Matplotlib <https://matplotlib.org/>`_ -- Debug visualization