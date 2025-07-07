Perception User Guide
====================

.. _user_guide_perception:

This comprehensive guide covers the Perception module in ManipulaPy, which provides high-level perception capabilities for robotic systems including obstacle detection, 3D point cloud processing, clustering, and environmental understanding.

.. contents:: **Quick Navigation**
   :local:
   :depth: 2

Overview
--------

The Perception module serves as a higher-level interface that builds upon the Vision module to provide sophisticated environmental understanding capabilities for robotic systems. It transforms raw visual data into actionable information for robot control and navigation.

.. raw:: html

   <div class="perception-hero">
      <div class="hero-content">
         <h3>🧠 Intelligent Scene Understanding</h3>
         <p>Transform raw camera data into meaningful environmental information for autonomous robotic decision-making.</p>
         
         <div class="capability-grid">
            <div class="capability">
               <span class="cap-icon">🔍</span>
               <strong>Object Detection</strong><br>
               YOLO-powered obstacle identification
            </div>
            <div class="capability">
               <span class="cap-icon">🌐</span>
               <strong>3D Clustering</strong><br>
               DBSCAN-based obstacle grouping
            </div>
            <div class="capability">
               <span class="cap-icon">👁️</span>
               <strong>Stereo Processing</strong><br>
               Point cloud generation and analysis
            </div>
            <div class="capability">
               <span class="cap-icon">🤖</span>
               <strong>Robot Integration</strong><br>
               Seamless planning and control integration
            </div>
         </div>
      </div>
   </div>

Key Features
------------

**Environmental Understanding**
  - Real-time obstacle detection and classification
  - 3D spatial clustering using DBSCAN algorithms
  - Multi-camera data fusion and integration

**Advanced Processing**
  - Stereo vision pipeline for depth reconstruction
  - Point cloud filtering and segmentation
  - Temporal consistency and object tracking

**Robot Integration**
  - Direct integration with path planning modules
  - Real-time safety monitoring and collision avoidance
  - Coordinate frame transformations for robot control

Getting Started
---------------

Basic Perception Setup
~~~~~~~~~~~~~~~~~~~~~

The Perception module requires a Vision instance to function:

.. code-block:: python

   from ManipulaPy.vision import Vision
   from ManipulaPy.perception import Perception
   import numpy as np
   
   # Create a vision system
   vision = Vision()
   
   # Create perception system with the vision instance
   perception = Perception(vision_instance=vision)
   
   print("🧠 Perception system initialized successfully!")

Simple Obstacle Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

Detect and cluster obstacles in the robot's environment:

.. code-block:: python

   # Detect and cluster obstacles
   obstacle_points, cluster_labels = perception.detect_and_cluster_obstacles(
       camera_index=0,           # Use first camera
       depth_threshold=5.0,      # Consider objects within 5 meters
       step=2,                   # Depth sampling step for efficiency
       eps=0.1,                  # DBSCAN clustering epsilon
       min_samples=3             # Minimum points per cluster
   )
   
   # Analyze results
   num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
   noise_points = np.sum(cluster_labels == -1)
   
   print(f"🔍 Detected {len(obstacle_points)} obstacle points")
   print(f"📊 Found {num_clusters} clusters with {noise_points} noise points")
   
   # Process each cluster
   for cluster_id in set(cluster_labels):
       if cluster_id == -1:  # Skip noise points
           continue
       
       cluster_points = obstacle_points[cluster_labels == cluster_id]
       cluster_center = np.mean(cluster_points, axis=0)
       cluster_size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
       
       print(f"Cluster {cluster_id}:")
       print(f"  📍 Center: [{cluster_center[0]:.2f}, {cluster_center[1]:.2f}, {cluster_center[2]:.2f}] m")
       print(f"  📏 Size: [{cluster_size[0]:.2f}, {cluster_size[1]:.2f}, {cluster_size[2]:.2f}] m")

Core Functionality
------------------

Obstacle Detection and Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary function of the Perception module is to detect and cluster obstacles:

.. code-block:: python

   # Advanced obstacle detection with custom parameters
   def detect_workspace_obstacles():
       """Detect obstacles in the robot workspace with optimized parameters."""
       
       obstacle_points, labels = perception.detect_and_cluster_obstacles(
           camera_index=0,
           depth_threshold=3.0,     # Limit to workspace range
           step=1,                  # High resolution for precision
           eps=0.05,                # Tight clustering for small objects
           min_samples=5            # Robust clusters only
       )
       
       # Filter clusters by size (remove tiny clusters)
       valid_clusters = []
       for cluster_id in set(labels):
           if cluster_id == -1:
               continue
           
           cluster_points = obstacle_points[labels == cluster_id]
           cluster_volume = np.prod(np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0))
           
           # Only keep clusters larger than 1 cubic centimeter
           if cluster_volume > 0.000001:  # 1 cm³
               valid_clusters.append({
                   'id': cluster_id,
                   'points': cluster_points,
                   'center': np.mean(cluster_points, axis=0),
                   'volume': cluster_volume
               })
       
       return valid_clusters

The detection pipeline follows these steps:

1. **Image Capture**: Acquire RGB and depth images from the vision system
2. **Object Detection**: Use YOLO to identify objects in RGB images
3. **Depth Integration**: Combine 2D detections with depth information
4. **3D Point Generation**: Convert detections to 3D world coordinates
5. **Clustering**: Group nearby points using DBSCAN algorithm
6. **Filtering**: Remove noise and invalid clusters

Stereo Vision Processing
~~~~~~~~~~~~~~~~~~~~~~~

For systems with stereo cameras, generate detailed 3D point clouds:

.. code-block:: python

   # Check if stereo vision is available
   if perception.vision.stereo_enabled:
       # Capture stereo image pair
       left_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # From left camera
       right_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) # From right camera
       
       # Compute disparity map
       disparity_map = perception.compute_stereo_disparity(left_image, right_image)
       
       # Generate 3D point cloud
       point_cloud = perception.get_stereo_point_cloud(left_image, right_image)
       
       print(f"🌐 Generated point cloud with {len(point_cloud)} 3D points")
       
       # Process point cloud for obstacles
       if len(point_cloud) > 0:
           # Cluster the full point cloud
           cloud_labels, num_cloud_clusters = perception.cluster_obstacles(
               point_cloud, 
               eps=0.02,      # Finer clustering for dense point clouds
               min_samples=10 # More points required for robust clusters
           )
           
           print(f"☁️ Point cloud contains {num_cloud_clusters} distinct objects")
   else:
       print("⚠️ Stereo vision not enabled - using monocular detection")

Advanced Clustering Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune clustering parameters for different environments:

.. code-block:: python

   def adaptive_clustering(obstacle_points, environment_type="indoor"):
       """Adapt clustering parameters based on environment type."""
       
       if environment_type == "indoor":
           # Indoor environments: smaller objects, higher precision
           eps = 0.05
           min_samples = 3
       elif environment_type == "outdoor":
           # Outdoor environments: larger objects, more noise tolerance
           eps = 0.15
           min_samples = 8
       elif environment_type == "industrial":
           # Industrial settings: structured objects, medium precision
           eps = 0.08
           min_samples = 5
       else:
           # Default parameters
           eps = 0.1
           min_samples = 3
       
       labels, num_clusters = perception.cluster_obstacles(
           obstacle_points, 
           eps=eps, 
           min_samples=min_samples
       )
       
       return labels, num_clusters

Data Flow Architecture
--------------------

Understanding the Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Perception module processes data through a sophisticated pipeline that transforms raw sensor input into actionable robotic intelligence. Understanding this flow is crucial for effective system integration and troubleshooting.

.. raw:: html

   <div class="dataflow-diagram">
      <div class="flow-stage">
         <div class="stage-icon">📷</div>
         <h4>1. Sensor Input</h4>
         <p>RGB + Depth cameras capture raw visual data</p>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-stage">
         <div class="stage-icon">🔍</div>
         <h4>2. Object Detection</h4>
         <p>YOLO identifies objects in RGB images</p>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-stage">
         <div class="stage-icon">🌐</div>
         <h4>3. 3D Integration</h4>
         <p>Depth data creates 3D obstacle points</p>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-stage">
         <div class="stage-icon">🧠</div>
         <h4>4. Clustering</h4>
         <p>DBSCAN groups related points</p>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-stage">
         <div class="stage-icon">🤖</div>
         <h4>5. Robot Control</h4>
         <p>Obstacle data enables safe navigation</p>
      </div>
   </div>

Detailed Data Flow Stages
~~~~~~~~~~~~~~~~~~~~~~~~~

**Stage 1: Sensor Data Acquisition**

.. code-block:: python

   # Raw sensor data flow
   def trace_sensor_input():
       """Trace the initial data acquisition stage."""
       
       # Vision system captures multi-modal data
       rgb_image, depth_image = vision.capture_image(camera_index=0)
       
       print("📷 Sensor Input Stage:")
       print(f"  RGB Image: {rgb_image.shape} - {rgb_image.dtype}")
       print(f"  Depth Image: {depth_image.shape} - {depth_image.dtype}")
       print(f"  Depth Range: {np.min(depth_image):.2f}m to {np.max(depth_image):.2f}m")
       
       return rgb_image, depth_image

**Stage 2: Object Detection Processing**

.. code-block:: python

   def trace_object_detection(rgb_image):
       """Trace the object detection stage."""
       
       print("\n🔍 Object Detection Stage:")
       
       if perception.vision.yolo_model:
           # YOLO inference on RGB image
           results = perception.vision.yolo_model(rgb_image, conf=0.3)
           
           if results and results[0].boxes is not None:
               boxes = results[0].boxes
               print(f"  Detected Objects: {len(boxes)}")
               
               for i, box in enumerate(boxes):
                   x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                   confidence = box.conf[0].item() if hasattr(box, 'conf') else 0.0
                   
                   print(f"    Object {i}: bbox=({x1},{y1},{x2},{y2}), conf={confidence:.2f}")
               
               return boxes
           else:
               print("  No objects detected")
               return []
       else:
           print("  YOLO model not available")
           return []

**Stage 3: 3D Point Generation**

.. code-block:: python

   def trace_3d_integration(boxes, depth_image, camera_index=0):
       """Trace the 3D point generation stage."""
       
       print("\n🌐 3D Integration Stage:")
       
       # Camera intrinsics for unprojection
       intrinsics = perception.vision.cameras[camera_index]["intrinsic_matrix"]
       fx, fy = intrinsics[0, 0], intrinsics[1, 1]
       cx, cy = intrinsics[0, 2], intrinsics[1, 2]
       
       points_3d = []
       
       for i, box in enumerate(boxes):
           x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
           
           # Extract depth in bounding box
           depth_roi = depth_image[y1:y2, x1:x2]
           valid_depths = depth_roi[depth_roi > 0]
           
           if len(valid_depths) > 0:
               median_depth = np.median(valid_depths)
               
               # Convert 2D detection to 3D point
               center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
               
               # Unproject to 3D using camera model
               x_3d = (center_x - cx) * median_depth / fx
               y_3d = (center_y - cy) * median_depth / fy
               z_3d = median_depth
               
               point_3d = np.array([x_3d, y_3d, z_3d])
               points_3d.append(point_3d)
               
               print(f"  Object {i} → 3D Point: [{x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}]m")
       
       return np.array(points_3d) if points_3d else np.empty((0, 3))

**Stage 4: Clustering and Segmentation**

.. code-block:: python

   def trace_clustering(points_3d, eps=0.1, min_samples=3):
       """Trace the clustering stage."""
       
       print("\n🧠 Clustering Stage:")
       
       if len(points_3d) == 0:
           print("  No points to cluster")
           return np.array([]), 0
       
       from sklearn.cluster import DBSCAN
       
       # Apply DBSCAN clustering
       dbscan = DBSCAN(eps=eps, min_samples=min_samples)
       labels = dbscan.fit_predict(points_3d)
       
       # Analyze clustering results
       unique_labels = set(labels)
       num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
       noise_points = np.sum(labels == -1)
       
       print(f"  Clustering Parameters: eps={eps}, min_samples={min_samples}")
       print(f"  Results: {num_clusters} clusters, {noise_points} noise points")
       
       # Detailed cluster analysis
       for cluster_id in unique_labels:
           if cluster_id == -1:
               continue
           
           cluster_points = points_3d[labels == cluster_id]
           cluster_center = np.mean(cluster_points, axis=0)
           cluster_spread = np.std(cluster_points, axis=0)
           
           print(f"    Cluster {cluster_id}:")
           print(f"      Points: {len(cluster_points)}")
           print(f"      Center: [{cluster_center[0]:.3f}, {cluster_center[1]:.3f}, {cluster_center[2]:.3f}]m")
           print(f"      Spread: [{cluster_spread[0]:.3f}, {cluster_spread[1]:.3f}, {cluster_spread[2]:.3f}]m")
       
       return labels, num_clusters

**Stage 5: Robot Integration Data**

.. code-block:: python

   def trace_robot_integration(points_3d, labels):
       """Trace how perception data integrates with robot control."""
       
       print("\n🤖 Robot Integration Stage:")
       
       # Transform to robot base frame (example transformation)
       def camera_to_robot_transform(points):
           """Transform points from camera frame to robot base frame."""
           # Example: camera mounted 0.5m above robot base, looking forward
           T_camera_to_robot = np.array([
               [0, 0, 1, 0.5],      # Camera X → Robot Z (forward)
               [-1, 0, 0, 0],       # Camera Y → Robot -X (left) 
               [0, -1, 0, 0.5],     # Camera Z → Robot -Y (up)
               [0, 0, 0, 1]
           ])
           
           # Convert points to homogeneous coordinates
           points_homo = np.column_stack([points, np.ones(len(points))])
           
           # Apply transformation
           points_robot = (T_camera_to_robot @ points_homo.T).T[:, :3]
           
           return points_robot
       
       if len(points_3d) > 0:
           # Transform to robot frame
           points_robot = camera_to_robot_transform(points_3d)
           
           print(f"  Coordinate Transformation: Camera → Robot Base Frame")
           print(f"  Original points (camera frame): {len(points_3d)}")
           print(f"  Transformed points (robot frame): {len(points_robot)}")
           
           # Generate obstacle data for path planning
           obstacles_for_planning = []
           
           for cluster_id in set(labels):
               if cluster_id == -1:  # Skip noise
                   continue
               
               cluster_points = points_robot[labels == cluster_id]
               
               # Create obstacle representation
               obstacle = {
                   'id': cluster_id,
                   'center': np.mean(cluster_points, axis=0),
                   'radius': np.max(np.linalg.norm(
                       cluster_points - np.mean(cluster_points, axis=0), axis=1
                   )) + 0.05,  # Add 5cm safety margin
                   'points': cluster_points,
                   'confidence': len(cluster_points) / len(points_3d)  # Relative size
               }
               
               obstacles_for_planning.append(obstacle)
               
               print(f"    Obstacle {cluster_id}:")
               print(f"      Center (robot frame): [{obstacle['center'][0]:.3f}, "
                     f"{obstacle['center'][1]:.3f}, {obstacle['center'][2]:.3f}]m")
               print(f"      Safety radius: {obstacle['radius']:.3f}m")
               print(f"      Confidence: {obstacle['confidence']:.2f}")
           
           return obstacles_for_planning
       else:
           print("  No obstacles to process for robot integration")
           return []

Complete Data Flow Demonstration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def demonstrate_complete_dataflow():
       """Demonstrate the complete perception data flow pipeline."""
       
       print("🔄 COMPLETE PERCEPTION DATA FLOW DEMONSTRATION")
       print("=" * 60)
       
       # Stage 1: Sensor Input
       rgb_image, depth_image = trace_sensor_input()
       
       # Stage 2: Object Detection  
       detected_boxes = trace_object_detection(rgb_image)
       
       # Stage 3: 3D Integration
       points_3d = trace_3d_integration(detected_boxes, depth_image)
       
       # Stage 4: Clustering
       labels, num_clusters = trace_clustering(points_3d)
       
       # Stage 5: Robot Integration
       robot_obstacles = trace_robot_integration(points_3d, labels)
       
       # Summary
       print("\n📊 PIPELINE SUMMARY:")
       print(f"  Raw Images Processed: 2 (RGB + Depth)")
       print(f"  Objects Detected: {len(detected_boxes)}")
       print(f"  3D Points Generated: {len(points_3d)}")
       print(f"  Clusters Formed: {num_clusters}")
       print(f"  Robot Obstacles: {len(robot_obstacles)}")
       
       return {
           'rgb_image': rgb_image,
           'depth_image': depth_image,
           'detected_boxes': detected_boxes,
           'points_3d': points_3d,
           'labels': labels,
           'robot_obstacles': robot_obstacles
       }

Data Flow Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from collections import defaultdict
   
   class DataFlowProfiler:
       """Profile the performance of each stage in the data flow."""
       
       def __init__(self):
           self.stage_times = defaultdict(list)
           self.stage_data_sizes = defaultdict(list)
       
       def profile_complete_pipeline(self, num_runs=10):
           """Profile the complete pipeline over multiple runs."""
           
           print(f"\n⏱️ PROFILING DATA FLOW PIPELINE ({num_runs} runs)")
           print("=" * 50)
           
           for run in range(num_runs):
               pipeline_start = time.time()
               
               # Stage 1: Sensor Input
               stage_start = time.time()
               rgb_image, depth_image = perception.vision.capture_image()
               stage_time = time.time() - stage_start
               self.stage_times['sensor_input'].append(stage_time)
               self.stage_data_sizes['sensor_input'].append(
                   rgb_image.nbytes + depth_image.nbytes if rgb_image is not None else 0
               )
               
               if rgb_image is None:
                   continue
               
               # Stage 2: Object Detection
               stage_start = time.time()
               obstacles, labels = perception.detect_and_cluster_obstacles()
               stage_time = time.time() - stage_start
               self.stage_times['detection_clustering'].append(stage_time)
               self.stage_data_sizes['detection_clustering'].append(
                   obstacles.nbytes + labels.nbytes if len(obstacles) > 0 else 0
               )
               
               # Stage 3: Robot Integration (simulated)
               stage_start = time.time()
               # Simulate coordinate transformation and obstacle processing
               if len(obstacles) > 0:
                   processed_obstacles = self._simulate_robot_integration(obstacles, labels)
               else:
                   processed_obstacles = []
               stage_time = time.time() - stage_start
               self.stage_times['robot_integration'].append(stage_time)
               self.stage_data_sizes['robot_integration'].append(
                   len(processed_obstacles) * 64  # Estimated bytes per obstacle
               )
               
               total_time = time.time() - pipeline_start
               self.stage_times['total_pipeline'].append(total_time)
               
               if (run + 1) % 5 == 0:
                   print(f"  Completed {run + 1}/{num_runs} runs...")
           
           self._print_performance_report()
       
       def _simulate_robot_integration(self, obstacles, labels):
           """Simulate robot integration processing."""
           processed = []
           for cluster_id in set(labels):
               if cluster_id != -1:
                   cluster_points = obstacles[labels == cluster_id]
                   processed.append({
                       'center': np.mean(cluster_points, axis=0),
                       'radius': np.max(np.linalg.norm(
                           cluster_points - np.mean(cluster_points, axis=0), axis=1
                       ))
                   })
           return processed
       
       def _print_performance_report(self):
           """Print detailed performance analysis."""
           
           print("\n📈 PERFORMANCE ANALYSIS:")
           print("-" * 40)
           
           for stage_name, times in self.stage_times.items():
               if len(times) > 0:
                   avg_time = np.mean(times) * 1000  # Convert to milliseconds
                   std_time = np.std(times) * 1000
                   max_time = np.max(times) * 1000
                   min_time = np.min(times) * 1000
                   
                   avg_size = np.mean(self.stage_data_sizes[stage_name]) / 1024  # KB
                   
                   print(f"\n{stage_name.replace('_', ' ').title()}:")
                   print(f"  Average Time: {avg_time:.2f} ± {std_time:.2f} ms")
                   print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
                   print(f"  Average Data Size: {avg_size:.1f} KB")
                   
                   if stage_name != 'total_pipeline':
                       percentage = (avg_time / (np.mean(self.stage_times['total_pipeline']) * 1000)) * 100
                       print(f"  Pipeline Percentage: {percentage:.1f}%")
       
       def get_bottlenecks(self):
           """Identify performance bottlenecks."""
           
           bottlenecks = []
           total_avg = np.mean(self.stage_times['total_pipeline']) * 1000
           
           for stage_name, times in self.stage_times.items():
               if stage_name != 'total_pipeline' and len(times) > 0:
                   avg_time = np.mean(times) * 1000
                   percentage = (avg_time / total_avg) * 100
                   
                   if percentage > 30:  # More than 30% of total time
                       bottlenecks.append({
                           'stage': stage_name,
                           'time_ms': avg_time,
                           'percentage': percentage
                       })
           
           return sorted(bottlenecks, key=lambda x: x['percentage'], reverse=True)

Data Flow Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimize_data_flow():
       """Demonstrate data flow optimization techniques."""
       
       print("\n🚀 DATA FLOW OPTIMIZATION STRATEGIES")
       print("=" * 45)
       
       # Strategy 1: Reduce data resolution for speed
       print("\n1. Resolution Optimization:")
       
       def downsample_for_speed(rgb_image, depth_image, factor=2):
           """Downsample images to reduce processing time."""
           if rgb_image is not None:
               h, w = rgb_image.shape[:2]
               new_h, new_w = h // factor, w // factor
               
               rgb_small = cv2.resize(rgb_image, (new_w, new_h))
               depth_small = cv2.resize(depth_image, (new_w, new_h))
               
               print(f"  Original: {w}x{h} → Downsampled: {new_w}x{new_h}")
               print(f"  Data reduction: {((w*h - new_w*new_h)/(w*h)*100):.1f}%")
               
               return rgb_small, depth_small
           return None, None
       
       # Strategy 2: Region of Interest (ROI) processing
       print("\n2. ROI-based Processing:")
       
       def process_roi_only(rgb_image, depth_image, roi_bounds):
           """Process only a region of interest."""
           x1, y1, x2, y2 = roi_bounds
           
           if rgb_image is not None:
               rgb_roi = rgb_image[y1:y2, x1:x2]
               depth_roi = depth_image[y1:y2, x1:x2]
               
               total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
               roi_pixels = (y2-y1) * (x2-x1)
               reduction = ((total_pixels - roi_pixels) / total_pixels) * 100
               
               print(f"  ROI: ({x1},{y1}) to ({x2},{y2})")
               print(f"  Processing reduction: {reduction:.1f}%")
               
               return rgb_roi, depth_roi
           return None, None
       
       # Strategy 3: Temporal filtering
       print("\n3. Temporal Filtering:")
       
       class TemporalFilter:
           """Filter obstacles over time to reduce noise."""
           
           def __init__(self, history_size=5, stability_threshold=0.3):
               self.obstacle_history = deque(maxlen=history_size)
               self.stability_threshold = stability_threshold
           
           def filter_obstacles(self, current_obstacles):
               """Apply temporal filtering to obstacles."""
               self.obstacle_history.append(current_obstacles)
               
               if len(self.obstacle_history) < 3:
                   return current_obstacles  # Need more history
               
               # Find stable obstacles (present in multiple frames)
               stable_obstacles = []
               
               for obstacle in current_obstacles:
                   stability_count = 0
                   
                   for past_obstacles in list(self.obstacle_history)[:-1]:
                       for past_obstacle in past_obstacles:
                           distance = np.linalg.norm(obstacle - past_obstacle)
                           if distance < self.stability_threshold:
                               stability_count += 1
                               break
                   
                   stability_ratio = stability_count / (len(self.obstacle_history) - 1)
                   if stability_ratio > 0.5:  # Present in >50% of recent frames
                       stable_obstacles.append(obstacle)
               
               print(f"    Temporal filtering: {len(current_obstacles)} → {len(stable_obstacles)} obstacles")
               return np.array(stable_obstacles) if stable_obstacles else np.empty((0, 3))


Advanced Applications
--------------------

Real-time Monitoring
~~~~~~~~~~~~~~~~~~~

Set up continuous environmental monitoring:

.. code-block:: python

   import time
   import threading
   from collections import deque
   
   class EnvironmentMonitor:
       """Real-time environment monitoring system."""
       
       def __init__(self, perception_system, update_rate=10):
           self.perception = perception_system
           self.update_rate = update_rate  # Hz
           self.obstacle_history = deque(maxlen=100)
           self.monitoring = False
           self.monitor_thread = None
       
       def start_monitoring(self):
           """Start the monitoring thread."""
           self.monitoring = True
           self.monitor_thread = threading.Thread(target=self._monitor_loop)
           self.monitor_thread.start()
           print("🔄 Environment monitoring started")
       
       def stop_monitoring(self):
           """Stop the monitoring thread."""
           self.monitoring = False
           if self.monitor_thread:
               self.monitor_thread.join()
           print("⏹️ Environment monitoring stopped")
       
       def _monitor_loop(self):
           """Main monitoring loop."""
           while self.monitoring:
               start_time = time.time()
               
               try:
                   # Detect current obstacles
                   obstacles, labels = self.perception.detect_and_cluster_obstacles()
                   
                   # Store in history
                   timestamp = time.time()
                   self.obstacle_history.append({
                       'timestamp': timestamp,
                       'obstacles': obstacles,
                       'labels': labels,
                       'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
                   })
                   
                   # Check for significant changes
                   if len(self.obstacle_history) > 1:
                       prev_count = self.obstacle_history[-2]['num_clusters']
                       curr_count = self.obstacle_history[-1]['num_clusters']
                       
                       if abs(curr_count - prev_count) > 1:
                           print(f"⚠️ Environment change detected: {prev_count} → {curr_count} clusters")
               
               except Exception as e:
                   print(f"❌ Monitoring error: {e}")
               
               # Maintain update rate
               elapsed = time.time() - start_time
               sleep_time = max(0, 1.0/self.update_rate - elapsed)
               time.sleep(sleep_time)
       
       def get_current_environment(self):
           """Get the latest environment state."""
           if self.obstacle_history:
               return self.obstacle_history[-1]
           return None
   
   # Usage
   monitor = EnvironmentMonitor(perception, update_rate=5)  # 5 Hz monitoring
   monitor.start_monitoring()
   
   # Let it run for a while
   time.sleep(10)
   
   # Check current state
   current_env = monitor.get_current_environment()
   if current_env:
       print(f"🌍 Current environment: {current_env['num_clusters']} clusters detected")
   
   monitor.stop_monitoring()

Integration with Path Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use perception data for safe robot navigation:

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning
   
   def plan_safe_trajectory(perception_system, robot, dynamics, start_config, goal_config):
       """Plan a trajectory that avoids detected obstacles."""
       
       # Get current obstacle configuration
       obstacle_points, labels = perception_system.detect_and_cluster_obstacles(
           depth_threshold=2.0,  # Only consider nearby obstacles
           eps=0.1,
           min_samples=5
       )
       
       # Extract cluster centers as obstacles for planning
       obstacles = []
       for cluster_id in set(labels):
           if cluster_id == -1:  # Skip noise
               continue
           
           cluster_points = obstacle_points[labels == cluster_id]
           cluster_center = np.mean(cluster_points, axis=0)
           cluster_radius = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
           
           obstacles.append({
               'center': cluster_center,
               'radius': cluster_radius + 0.1  # Add safety margin
           })
       
       print(f"🚧 Planning around {len(obstacles)} obstacles")
       
       # Create trajectory planner
       joint_limits = [(-np.pi, np.pi)] * len(start_config)
       planner = TrajectoryPlanning(robot, "robot.urdf", dynamics, joint_limits)
       
       # Generate collision-free trajectory
       trajectory = planner.joint_trajectory(
           thetastart=start_config,
           thetaend=goal_config,
           Tf=5.0,
           N=100,
           method=5  # Quintic time scaling
       )
       
       return trajectory, obstacles

Object Tracking and Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track objects over time for consistent identification:

.. code-block:: python

   class ObjectTracker:
       """Simple object tracking based on position proximity."""
       
       def __init__(self, max_distance=0.2, max_age=10):
           self.tracked_objects = []
           self.max_distance = max_distance  # Maximum distance for association
           self.max_age = max_age  # Maximum age before removing track
           self.next_id = 0
       
       def update(self, new_detections):
           """Update tracker with new detections."""
           # Age existing tracks
           for track in self.tracked_objects:
               track['age'] += 1
           
           # Associate new detections with existing tracks
           unmatched_detections = []
           
           for detection in new_detections:
               best_match = None
               best_distance = float('inf')
               
               for track in self.tracked_objects:
                   distance = np.linalg.norm(detection - track['position'])
                   if distance < self.max_distance and distance < best_distance:
                       best_match = track
                       best_distance = distance
               
               if best_match:
                   # Update existing track
                   best_match['position'] = detection
                   best_match['age'] = 0
               else:
                   # Create new track
                   unmatched_detections.append(detection)
           
           # Add new tracks
           for detection in unmatched_detections:
               self.tracked_objects.append({
                   'id': self.next_id,
                   'position': detection,
                   'age': 0
               })
               self.next_id += 1
           
           # Remove old tracks
           self.tracked_objects = [
               track for track in self.tracked_objects 
               if track['age'] < self.max_age
           ]
           
           return self.tracked_objects
   
   # Usage with perception system
   tracker = ObjectTracker()
   
   for frame in range(100):  # Process 100 frames
       # Get current detections
       obstacles, labels = perception.detect_and_cluster_obstacles()
       
       # Extract cluster centers
       detections = []
       for cluster_id in set(labels):
           if cluster_id != -1:
               cluster_points = obstacles[labels == cluster_id]
               center = np.mean(cluster_points, axis=0)
               detections.append(center)
       
       # Update tracker
       tracked_objects = tracker.update(detections)
       
       print(f"Frame {frame}: {len(tracked_objects)} tracked objects")

Performance Optimization
------------------------

Efficient Processing Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimized_perception_pipeline(perception_system, quality_level="medium"):
       """Optimized perception pipeline with adjustable quality levels."""
       
       if quality_level == "high":
           # High quality: full resolution, tight clustering
           params = {
               'depth_threshold': 5.0,
               'step': 1,
               'eps': 0.05,
               'min_samples': 5
           }
       elif quality_level == "medium":
           # Medium quality: balanced performance
           params = {
               'depth_threshold': 3.0,
               'step': 2,
               'eps': 0.1,
               'min_samples': 3
           }
       else:  # low quality
           # Low quality: fast processing
           params = {
               'depth_threshold': 2.0,
               'step': 4,
               'eps': 0.15,
               'min_samples': 2
           }
       
       # Execute detection with optimized parameters
       obstacles, labels = perception_system.detect_and_cluster_obstacles(**params)
       
       return obstacles, labels

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   def memory_efficient_processing(perception_system, batch_size=10):
       """Process perception data in batches to manage memory usage."""
       
       results = []
       
       for batch in range(batch_size):
           # Process one frame
           obstacles, labels = perception_system.detect_and_cluster_obstacles()
           
           # Store only essential information
           frame_result = {
               'timestamp': time.time(),
               'num_obstacles': len(obstacles),
               'num_clusters': len(set(labels)) - (1 if -1 in labels else 0),
               'cluster_centers': []
           }
           
           # Extract cluster centers only (not all points)
           for cluster_id in set(labels):
               if cluster_id != -1:
                   cluster_points = obstacles[labels == cluster_id]
                   center = np.mean(cluster_points, axis=0)
                   frame_result['cluster_centers'].append(center.tolist())
           
           results.append(frame_result)
           
           # Clean up large arrays
           del obstacles, labels
       
       return results

Error Handling and Robustness
-----------------------------

Robust Perception Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_perception_pipeline(perception_system, max_retries=3):
       """Robust perception pipeline with error handling and retries."""
       
       for attempt in range(max_retries):
           try:
               # Attempt to detect obstacles
               obstacles, labels = perception_system.detect_and_cluster_obstacles()
               
               # Validate results
               if obstacles is None or len(obstacles) == 0:
                   print(f"⚠️ No obstacles detected on attempt {attempt + 1}")
                   if attempt < max_retries - 1:
                       time.sleep(0.1)  # Brief pause before retry
                       continue
                   else:
                       print("❌ No valid obstacles detected after all retries")
                       return np.empty((0, 3)), np.array([])
               
               # Check for reasonable number of clusters
               num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
               if num_clusters > 50:  # Suspiciously high number
                   print(f"⚠️ Detected {num_clusters} clusters - may indicate noisy data")
               
               print(f"✅ Successfully detected {len(obstacles)} points in {num_clusters} clusters")
               return obstacles, labels
               
           except RuntimeError as e:
               print(f"❌ Runtime error on attempt {attempt + 1}: {e}")
               if attempt < max_retries - 1:
                   time.sleep(0.1)
               else:
                   print("❌ All attempts failed")
                   raise
           
           except Exception as e:
               print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
               if attempt < max_retries - 1:
                   time.sleep(0.1)
               else:
                   print("❌ All attempts failed")
                   raise
       
       return np.empty((0, 3)), np.array([])

System Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class PerceptionHealthMonitor:
       """Monitor the health and performance of the perception system."""
       
       def __init__(self, perception_system):
           self.perception = perception_system
           self.stats = {
               'successful_detections': 0,
               'failed_detections': 0,
               'average_processing_time': 0.0,
               'processing_times': deque(maxlen=100)
           }
       
       def monitored_detection(self, **kwargs):
           """Perform detection with health monitoring."""
           start_time = time.time()
           
           try:
               obstacles, labels = self.perception.detect_and_cluster_obstacles(**kwargs)
               
               # Record success
               self.stats['successful_detections'] += 1
               processing_time = time.time() - start_time
               self.stats['processing_times'].append(processing_time)
               self.stats['average_processing_time'] = np.mean(self.stats['processing_times'])
               
               return obstacles, labels
               
           except Exception as e:
               # Record failure
               self.stats['failed_detections'] += 1
               print(f"❌ Detection failed: {e}")
               raise
       
       def get_health_report(self):
           """Generate a health report."""
           total_attempts = self.stats['successful_detections'] + self.stats['failed_detections']
           success_rate = (self.stats['successful_detections'] / max(1, total_attempts)) * 100
           
           report = {
               'success_rate': success_rate,
               'total_attempts': total_attempts,
               'average_processing_time': self.stats['average_processing_time'],
               'status': 'healthy' if success_rate > 90 else 'degraded' if success_rate > 70 else 'critical'
           }
           
           return report

Best Practices
--------------

1. **Environment Adaptation**
   - Adjust clustering parameters based on environment type
   - Use appropriate depth thresholds for workspace size
   - Consider lighting conditions and camera placement

2. **Performance Optimization**
   - Balance detection quality with processing speed
   - Use appropriate step sizes for depth sampling
   - Implement frame skipping for real-time applications

3. **Robustness**
   - Always validate detection results before use
   - Implement proper error handling and recovery
   - Use temporal filtering to reduce noise

4. **Integration**
   - Coordinate perception timing with control loops
   - Transform coordinates to robot base frame
   - Validate obstacle data before path planning

5. **Maintenance**
   - Monitor system performance regularly
   - Log detection statistics for analysis
   - Update clustering parameters based on performance

Common Issues and Solutions
--------------------------

**Issue: Too many small clusters detected**

.. code-block:: python

   # Solution: Increase min_samples parameter
   obstacles, labels = perception.detect_and_cluster_obstacles(
       eps=0.1,
       min_samples=8  # Increase from default 3 to 8
   )

**Issue: Large objects split into multiple clusters**

.. code-block:: python

   # Solution: Increase eps parameter
   obstacles, labels = perception.detect_and_cluster_obstacles(
       eps=0.2,  # Increase from default 0.1 to 0.2
       min_samples=3
   )

**Issue: Poor stereo reconstruction**

.. code-block:: python

   # Solution: Check stereo configuration and calibration
   if perception.vision.stereo_enabled:
       # Verify stereo cameras are properly calibrated
       perception.vision.compute_stereo_rectification_maps()
   else:
       print("Stereo not enabled - check stereo_configs")

See Also
--------

- :doc:`../api/perception` - Complete Perception API reference
- :doc:`/user_guide/vision` - Vision module user guide  
- :doc:`Trajectory_Planning` - Path planning integration
- :doc:`../tutorials/index` - Perception tutorials and examples

.. raw:: html

   <style>
   .perception-hero {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      border-radius: 12px;
      margin: 2rem 0;
      text-align: center;
   }
   1
   .perception-hero h3 {
      margin-top: 0;
      font-size: 1.8rem;
   }
   
   .capability-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 1.5rem;
   }
   
   .capability {
      background: rgba(255,255,255,0.1);
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
   }
   
   .cap-icon {
      font-size: 2rem;
      display: block;
      margin-bottom: 0.5rem;
   }
   </style>