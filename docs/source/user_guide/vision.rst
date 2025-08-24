Vision User Guide
=================

.. _user_guide_vision:

This comprehensive guide covers the Vision module in ManipulaPy, which provides advanced computer vision capabilities for robotic perception, including stereo vision, object detection, and PyBullet integration.

.. contents:: **Quick Navigation**
   :local:
   :depth: 2

Overview
--------

The Vision module is a unified computer vision system that brings together:

- **Monocular and stereo camera support** with flexible configuration

- **YOLO-based object detection** for real-time obstacle identification

- **PyBullet virtual cameras** with interactive debugging sliders

- **Stereo vision pipeline** for 3D reconstruction and depth estimation

- **Camera calibration utilities** for precise geometric measurements

.. raw:: html

   <div class="feature-showcase">
      <div class="feature-card">
         <span class="feature-icon">📷</span>
         <h4>Multi-Camera Support</h4>
         <p>Configure multiple cameras with individual intrinsics, extrinsics, and distortion parameters</p>
      </div>
      <div class="feature-card">
         <span class="feature-icon">🤖</span>
         <h4>YOLO Integration</h4>
         <p>Real-time object detection with YOLOv8 for robust obstacle identification</p>
      </div>
      <div class="feature-card">
         <span class="feature-icon">🎮</span>
         <h4>PyBullet Debug</h4>
         <p>Interactive virtual cameras with real-time parameter adjustment</p>
      </div>
      <div class="feature-card">
         <span class="feature-icon">👁️</span>
         <h4>Stereo Vision</h4>
         <p>Complete stereo pipeline from rectification to 3D point cloud generation</p>
      </div>
   </div>

Getting Started
---------------

Basic Camera Setup
~~~~~~~~~~~~~~~~~~

The simplest way to start with the Vision module:

.. code-block:: python

   from ManipulaPy.vision import Vision
   import numpy as np
   
   # Create a basic vision system with default settings
   vision = Vision()
   
   # Capture an image (requires PyBullet environment)
   rgb_image, depth_image = vision.capture_image()
   
   print(f"📸 Captured RGB image: {rgb_image.shape}")
   print(f"📏 Captured depth image: {depth_image.shape}")

Custom Camera Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, configure your cameras explicitly:

.. code-block:: python

   # Define camera parameters
   camera_config = {
       "name": "workspace_camera",
       "translation": [0.0, 0.0, 1.0],  # 1 meter above workspace
       "rotation": [0, 45, 0],           # Look down at 45 degrees
       "fov": 60,                        # Field of view in degrees
       "near": 0.1,                      # Near clipping plane
       "far": 10.0,                      # Far clipping plane
       "intrinsic_matrix": np.array([
           [500, 0, 320],    # fx, 0, cx
           [0, 500, 240],    # 0, fy, cy
           [0, 0, 1]         # 0, 0, 1
       ], dtype=np.float32),
       "distortion_coeffs": np.zeros(5, dtype=np.float32),  # k1,k2,p1,p2,k3
       "use_opencv": False,              # Use PyBullet cameras
       "device_index": 0                 # Camera device index
   }
   
   # Create vision system with custom configuration
   vision = Vision(camera_configs=[camera_config])

Core Features
-------------

Object Detection with YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Vision module integrates YOLOv8 for robust object detection:

.. code-block:: python

   # Capture images
   rgb_image, depth_image = vision.capture_image(camera_index=0)
   
   # Detect obstacles with 3D positioning
   obstacle_positions, orientations = vision.detect_obstacles(
       depth_image=depth_image,
       rgb_image=rgb_image,
       depth_threshold=5.0,    # Only consider objects within 5 meters
       camera_index=0,
       step=2                  # Depth sampling step for efficiency
   )
   
   # Process detected obstacles
   print(f"🔍 Detected {len(obstacle_positions)} obstacles")
   
   for i, (pos, orientation) in enumerate(zip(obstacle_positions, orientations)):
       print(f"Obstacle {i+1}:")
       print(f"  📍 Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] meters")
       print(f"  🧭 Orientation: {orientation:.1f} degrees")

The object detection pipeline:

1. **YOLO Detection**: Identifies objects in RGB images with bounding boxes

2. **Depth Analysis**: Uses depth information within bounding boxes

3. **3D Positioning**: Converts 2D detections to 3D world coordinates

4. **Orientation Estimation**: Computes object orientation in the XY plane

PyBullet Virtual Cameras
~~~~~~~~~~~~~~~~~~~~~~~~

For simulation and debugging, use PyBullet's virtual cameras:

.. code-block:: python

   # Create an interactive debug camera system
   debug_vision = Vision(
       use_pybullet_debug=True,    # Enable PyBullet debug sliders
       show_plot=True              # Display camera feed in matplotlib
   )
   
   # The debug interface provides real-time sliders for:
   # - Camera position (target_x, target_y, target_z)
   # - Camera orientation (yaw, pitch, roll) 
   # - View parameters (distance, up axis)
   # - Projection settings (width, height, FOV, near/far planes)

**Debug Interface Features:**

- **Real-time parameter adjustment** via PyBullet GUI sliders

- **Live camera feed** displayed in matplotlib window

- **Matrix visualization** for view and projection matrices

- **Interactive positioning** for optimal camera placement

Stereo Vision Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

For 3D reconstruction, configure a stereo camera pair:

.. code-block:: python

   # Configure left camera
   left_camera_config = {
       "name": "left_camera",
       "translation": [0.0, 0.0, 0.5],
       "rotation": [0, 0, 0],
       "intrinsic_matrix": np.array([
           [600, 0, 320],
           [0, 600, 240],
           [0, 0, 1]
       ], dtype=np.float32),
       "distortion_coeffs": np.zeros(5, dtype=np.float32)
   }
   
   # Configure right camera (10cm baseline)
   right_camera_config = left_camera_config.copy()
   right_camera_config["name"] = "right_camera"
   right_camera_config["translation"] = [0.1, 0.0, 0.5]  # 10cm to the right
   
   # Create stereo vision system
   stereo_vision = Vision(stereo_configs=(left_camera_config, right_camera_config))
   
   # Compute rectification maps (do this once)
   stereo_vision.compute_stereo_rectification_maps(image_size=(640, 480))
   
   # Capture stereo images
   left_image, _ = stereo_vision.capture_image(0)  # Left camera
   right_image, _ = stereo_vision.capture_image(1) # Right camera
   
   # Process stereo pipeline
   left_rect, right_rect = stereo_vision.rectify_stereo_images(left_image, right_image)
   disparity_map = stereo_vision.compute_disparity(left_rect, right_rect)
   point_cloud = stereo_vision.disparity_to_pointcloud(disparity_map)
   
   print(f"🌐 Generated point cloud with {len(point_cloud)} 3D points")

**Stereo Pipeline Steps:**

1. **Image Rectification**: Align stereo images for disparity computation

2. **Disparity Calculation**: Use StereoSGBM for robust disparity estimation

3. **3D Reconstruction**: Convert disparity to 3D points using camera geometry

4. **Point Cloud Filtering**: Remove invalid and distant points

Advanced Usage
--------------

Multiple Camera Systems
~~~~~~~~~~~~~~~~~~~~~~~

Configure and manage multiple cameras simultaneously:

.. code-block:: python

   # Define multiple camera configurations
   camera_configs = [
       {  # Overview camera
           "name": "overview_camera",
           "translation": [0, 0, 2.0],
           "rotation": [0, 90, 0],     # Look straight down
           "fov": 80,
           "intrinsic_matrix": np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float32),
           "distortion_coeffs": np.zeros(5, dtype=np.float32)
       },
       {  # Side view camera
           "name": "side_camera", 
           "translation": [1.0, 0, 0.5],
           "rotation": [0, 0, 90],     # Look sideways
           "fov": 60,
           "intrinsic_matrix": np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
           "distortion_coeffs": np.zeros(5, dtype=np.float32)
       }
   ]
   
   # Create multi-camera vision system
   multi_vision = Vision(camera_configs=camera_configs)
   
   # Capture from different cameras
   overview_rgb, overview_depth = multi_vision.capture_image(camera_index=0)
   side_rgb, side_depth = multi_vision.capture_image(camera_index=1)
   
   # Detect obstacles from multiple viewpoints
   obstacles_overview, _ = multi_vision.detect_obstacles(overview_depth, overview_rgb, camera_index=0)
   obstacles_side, _ = multi_vision.detect_obstacles(side_depth, side_rgb, camera_index=1)
   
   print(f"📷 Overview camera detected {len(obstacles_overview)} obstacles")
   print(f"📷 Side camera detected {len(obstacles_side)} obstacles")

OpenCV Camera Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use real hardware cameras with OpenCV:

.. code-block:: python

   # Configure real camera with OpenCV
   real_camera_config = {
       "name": "usb_camera",
       "translation": [0, 0, 0],
       "rotation": [0, 0, 0],
       "fov": 60,
       "intrinsic_matrix": np.array([
           [800, 0, 320],    # Values from camera calibration
           [0, 800, 240],
           [0, 0, 1]
       ], dtype=np.float32),
       "distortion_coeffs": np.array([-0.1, 0.05, 0, 0, 0], dtype=np.float32),  # From calibration
       "use_opencv": True,        # Enable OpenCV capture
       "device_index": 0          # USB camera device ID
   }
   
   # Create vision system with real camera
   real_vision = Vision(camera_configs=[real_camera_config])
   
   # Note: capture_image() will use OpenCV for image acquisition
   # when use_opencv=True

Camera Calibration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding the camera configuration parameters:

.. code-block:: python

   # Intrinsic matrix format:
   # [fx  0  cx]
   # [0  fy  cy] 
   # [0   0   1]
   #
   # Where:
   # fx, fy = focal lengths in pixels
   # cx, cy = principal point (image center) in pixels
   
   intrinsic_matrix = np.array([
       [focal_x, 0, center_x],
       [0, focal_y, center_y],
       [0, 0, 1]
   ], dtype=np.float32)
   
   # Distortion coefficients: [k1, k2, p1, p2, k3]
   # k1, k2, k3 = radial distortion coefficients
   # p1, p2 = tangential distortion coefficients
   distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
   
   # Extrinsic parameters (pose in world coordinates):
   # translation = [x, y, z] position in meters
   # rotation = [roll, pitch, yaw] in degrees

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For long-running applications, manage resources carefully
   vision = Vision(camera_configs=configs)
   
   try:
       while True:
           # Capture and process images
           rgb, depth = vision.capture_image()
           obstacles, _ = vision.detect_obstacles(depth, rgb)
           
           # Process obstacles...
           
           # Clean up large arrays if needed
           del rgb, depth
           
   finally:
       # Always release resources
       vision.release()

Efficient Object Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize detection parameters for performance
   obstacles, orientations = vision.detect_obstacles(
       depth_image=depth,
       rgb_image=rgb,
       depth_threshold=3.0,     # Limit detection range
       camera_index=0,
       step=4                   # Increase step size for speed (lower accuracy)
   )
   
   # For real-time applications, consider:
   # - Reducing image resolution
   # - Increasing step size
   # - Limiting depth threshold
   # - Processing every nth frame

Error Handling and Debugging
----------------------------

Robust Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       # Create vision system
       vision = Vision(camera_configs=configs)
       
       # Attempt image capture
       rgb, depth = vision.capture_image(camera_index=0)
       
       if rgb is None or depth is None:
           print("❌ Failed to capture images")
           return
           
       # Attempt object detection
       obstacles, orientations = vision.detect_obstacles(depth, rgb)
       
       if len(obstacles) == 0:
           print("⚠️ No obstacles detected")
       else:
           print(f"✅ Detected {len(obstacles)} obstacles")
           
   except RuntimeError as e:
       print(f"❌ Vision system error: {e}")
   except Exception as e:
       print(f"❌ Unexpected error: {e}")
   finally:
       if 'vision' in locals():
           vision.release()

Debugging Tips
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Create vision with detailed logging
   vision = Vision(camera_configs=configs, logger_name="DebugVision")
   
   # Check YOLO model status
   if vision.yolo_model is None:
       print("⚠️ YOLO model not loaded - object detection disabled")
   else:
       print("✅ YOLO model loaded successfully")
   
   # Verify camera configuration
   for idx, camera in vision.cameras.items():
       print(f"📷 Camera {idx}: {camera['name']}")
       print(f"   Position: {camera['translation']}")
       print(f"   Rotation: {camera['rotation']}")
       print(f"   FOV: {camera['fov']}°")

Common Issues and Solutions
--------------------------------

**Issue: No objects detected by YOLO**

.. code-block:: python

   # Solutions:
   # 1. Check if YOLO model loaded properly
   if vision.yolo_model is None:
       print("Install ultralytics: pip install ultralytics")
   
   # 2. Verify image quality
   rgb, depth = vision.capture_image()
   if rgb.max() == 0:
       print("Image is completely black - check lighting/camera")
   
   # 3. Adjust detection confidence
   # Lower confidence threshold in detect_obstacles()

**Issue: Poor stereo reconstruction**

.. code-block:: python

   # Solutions:
   # 1. Ensure proper camera calibration
   # 2. Check baseline distance (should be 5-15% of working distance)
   # 3. Verify image rectification quality
   
   left_rect, right_rect = vision.rectify_stereo_images(left, right)
   # Rectified images should be aligned horizontally

**Issue: Inaccurate 3D positions**

.. code-block:: python

   # Solutions:
   # 1. Calibrate intrinsic matrix precisely
   # 2. Verify depth image scaling
   # 3. Check coordinate frame conventions
   
   # Debug depth values
   print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")
   print(f"Near/far planes: {camera['near']} - {camera['far']}")

Real-World Applications
---------------------------

Robot Navigation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning
   
   # Integrated obstacle detection for path planning
   def safe_navigation():
       # Detect current obstacles
       rgb, depth = vision.capture_image()
       obstacles, _ = vision.detect_obstacles(depth, rgb, depth_threshold=2.0)
       
       # Update robot's environmental model
       planner = TrajectoryPlanning(robot, urdf_file, dynamics, joint_limits)
       
       # Plan collision-free trajectory
       safe_trajectory = planner.joint_trajectory(
           thetastart=current_position,
           thetaend=target_position,
           Tf=5.0,
           N=100,
           method=5
       )
       
       return safe_trajectory, obstacles

Pick and Place Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def pick_and_place_with_vision():
       # Detect objects in workspace
       rgb, depth = vision.capture_image(camera_index=0)  # Overhead camera
       objects, orientations = vision.detect_obstacles(depth, rgb, depth_threshold=1.0)
       
       if len(objects) == 0:
           print("No objects found to pick")
           return
       
       # Select closest object
       closest_idx = np.argmin([np.linalg.norm(obj) for obj in objects])
       target_object = objects[closest_idx]
       target_orientation = orientations[closest_idx]
       
       print(f"🎯 Targeting object at: {target_object}")
       print(f"🧭 Object orientation: {target_orientation:.1f}°")
       
       # Plan approach trajectory
       # ... (integrate with kinematics and planning)

Best Practices
-----------------

1. **Camera Placement**

   - Position cameras for optimal workspace coverage
   - Avoid backlighting and reflective surfaces
   - Ensure sufficient lighting for object detection

2. **Calibration**

   - Use high-quality calibration patterns (checkerboards)
   - Capture calibration images from multiple angles
   - Verify calibration accuracy before deployment

3. **Performance**

   - Choose appropriate image resolutions for your application
   - Balance detection accuracy with processing speed
   - Use temporal filtering for stable object tracking

4. **Robustness**

   - Implement proper error handling for all vision operations
   - Use multiple cameras for redundancy when possible
   - Validate detection results before using in control loops

5. **Integration**

   - Coordinate vision frame rates with control loop timing
   - Transform coordinates to robot base frame consistently
   - Use vision confidence scores in decision making

See Also
-----------

- :doc:`../api/vision` - Complete Vision API reference
- :doc:`Perception` - Higher-level perception capabilities
- :doc:`../tutorials/index` - Vision and perception tutorials
- :doc:`Simulation` - PyBullet integration guide

.. raw:: html

   <style>
   .feature-showcase {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
      margin: 2rem 0;
   }
   
   .feature-card {
      background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
      border: 1px solid #e1e8ed;
      border-radius: 12px;
      padding: 1.5rem;
      text-align: center;
      transition: all 0.3s ease;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
   }
   
   .feature-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px rgba(0,0,0,0.1);
      border-color: #1da1f2;
   }
   
   .feature-icon {
      font-size: 2.5rem;
      display: block;
      margin-bottom: 1rem;
   }
   
   .feature-card h4 {
      margin: 0 0 0.5rem 0;
      color: #14171a;
      font-weight: 600;
   }
   
   .feature-card p {
      margin: 0;
      color: #657786;
      font-size: 0.9rem;
      line-height: 1.4;
   }
   </style>