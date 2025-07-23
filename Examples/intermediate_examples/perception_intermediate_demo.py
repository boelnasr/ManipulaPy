#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Perception Intermediate Demo - ManipulaPy

This script demonstrates the perception and vision capabilities of the ManipulaPy library including:
- Vision system setup and camera configuration
- Image capture and processing
- Stereo vision and 3D point cloud generation
- Obstacle detection using YOLO and depth information
- Clustering and analysis of detected objects
- PyBullet integration for simulated perception
- Advanced perception features and motion analysis
- Visualization with automatic plot saving

All generated plots and images are saved in the same folder as this script.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import cv2
import matplotlib
matplotlib.use('TkAgg')
# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import ManipulaPy modules
try:
    # Add the parent directory to the path to find ManipulaPy
    current_dir = Path(__file__).parent.absolute()
    manipulapy_path = current_dir.parent.parent
    if str(manipulapy_path) not in sys.path:
        sys.path.insert(0, str(manipulapy_path))
    
    from ManipulaPy.vision import Vision
    from ManipulaPy.perception import Perception
    import pybullet as pb
    import pybullet_data
    
    print("✅ ManipulaPy perception modules imported successfully")
    
except ImportError as e:
    print(f"❌ Error importing ManipulaPy modules: {e}")
    print("Please ensure ManipulaPy is properly installed and accessible.")
    print("Note: Some optional dependencies may be missing (YOLO, pybullet, etc.)")
    
    # Try to continue with available modules
    try:
        import cv2
        import numpy as np
        print("✅ Basic OpenCV and NumPy available")
    except ImportError:
        print("❌ Critical dependencies missing. Exiting.")
        sys.exit(1)

# Configure matplotlib for non-interactive plotting
plt.ioff()  # Turn off interactive mode for automated saving
plt.rcParams['figure.max_open_warning'] = 50

class PerceptionIntermediateDemo:
    """
    Comprehensive demonstration of ManipulaPy's perception and vision features
    with automatic image and plot saving.
    """
    
    def __init__(self, use_real_camera=False, save_outputs=True):
        """
        Initialize the perception demonstration.
        
        Args:
            use_real_camera (bool): Whether to attempt using real camera devices
            save_outputs (bool): Whether to save generated images and plots
        """
        self.use_real_camera = use_real_camera
        self.save_outputs = save_outputs
        self.script_dir = Path(__file__).parent.absolute()
        self.outputs_saved = []
        
        # Setup logging
        self.setup_logging()
        
        # Initialize PyBullet for simulated perception
        self.pybullet_client = None
        self.robot_id = None
        self.camera_data = {}
        
        # Vision and Perception systems
        self.vision_system = None
        self.perception_system = None
        
        # Demo data storage
        self.demo_results = {}
        
        self.logger.info("Perception demo initialized")
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.script_dir / 'perception_demo.log'
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='w')
            ]
        )
        self.logger = logging.getLogger('PerceptionDemo')
    
    def save_image(self, image, filename, title=None):
        """Save image with timestamp and title."""
        if not self.save_outputs:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if title:
            filename = f"{timestamp}_{title}_{filename}"
        else:
            filename = f"{timestamp}_{filename}"
            
        filepath = self.script_dir / filename
        try:
            # Handle different image formats
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA image
                cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))
            else:
                # Grayscale or depth image
                cv2.imwrite(str(filepath), image)
            
            self.outputs_saved.append(str(filepath))
            self.logger.info(f"Image saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save image {filepath}: {e}")
    
    def save_plot(self, filename, title=None):
        """Save current plot with timestamp and title."""
        if not self.save_outputs:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if title:
            filename = f"{timestamp}_{title}_{filename}"
        else:
            filename = f"{timestamp}_{filename}"
            
        filepath = self.script_dir / filename
        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            self.outputs_saved.append(str(filepath))
            self.logger.info(f"Plot saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save plot {filepath}: {e}")
    
    def setup_pybullet_simulation(self):
        """Setup PyBullet simulation environment for perception testing."""
        self.logger.info("=== Setting up PyBullet Simulation ===")
        
        try:
            # Connect to PyBullet
            self.pybullet_client = pb.connect(pb.GUI)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.81)
            
            # Load ground plane
            plane_id = pb.loadURDF("plane.urdf")
            
            # Load some objects for perception testing
            # Cube
            cube_id = pb.loadURDF("cube_small.urdf", [0.5, 0.0, 0.5])
            pb.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])  # Red
            
            # Sphere
            try:
                sphere_id = pb.loadURDF("sphere2.urdf", [0.3, 0.3, 0.5])
                pb.changeVisualShape(sphere_id, -1, rgbaColor=[0, 1, 0, 1])  # Green
            except:
                self.logger.warning("Could not load sphere, using another cube")
                sphere_id = pb.loadURDF("cube_small.urdf", [0.3, 0.3, 0.5])
                pb.changeVisualShape(sphere_id, -1, rgbaColor=[0, 1, 0, 1])
            
            # Another object
            try:
                obj_id = pb.loadURDF("duck_vhacd.urdf", [0.2, -0.2, 0.5])
                pb.changeVisualShape(obj_id, -1, rgbaColor=[0, 0, 1, 1])  # Blue
            except:
                self.logger.warning("Could not load duck, using cube")
                obj_id = pb.loadURDF("cube_small.urdf", [0.2, -0.2, 0.5])
                pb.changeVisualShape(obj_id, -1, rgbaColor=[0, 0, 1, 1])
            
            self.logger.info("PyBullet simulation setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup PyBullet simulation: {e}")
            return False
    
    def setup_vision_system(self):
        """Setup vision system with different camera configurations."""
        self.logger.info("=== Setting up Vision System ===")
        
        try:
            # Define camera configurations
            # Monocular camera configuration
            mono_camera_config = {
                "name": "mono_camera",
                "translation": [0, 0, 1.0],  # 1m above origin
                "rotation": [0, -30, 0],      # Looking down at 30 degrees
                "fov": 60,
                "near": 0.1,
                "far": 5.0,
                "intrinsic_matrix": np.array([
                    [525.0, 0, 320.0],
                    [0, 525.0, 240.0],
                    [0, 0, 1.0]
                ], dtype=np.float32),
                "distortion_coeffs": np.zeros(5, dtype=np.float32),
                "use_opencv": self.use_real_camera,
                "device_index": 0
            }
            
            # Stereo camera configurations
            left_camera_config = {
                "name": "left_camera",
                "translation": [-0.1, 0, 1.0],  # Left camera
                "rotation": [0, -30, 0],
                "fov": 60,
                "near": 0.1,
                "far": 5.0,
                "intrinsic_matrix": np.array([
                    [525.0, 0, 320.0],
                    [0, 525.0, 240.0],
                    [0, 0, 1.0]
                ], dtype=np.float32),
                "distortion_coeffs": np.zeros(5, dtype=np.float32)
            }
            
            right_camera_config = {
                "name": "right_camera",
                "translation": [0.1, 0, 1.0],   # Right camera
                "rotation": [0, -30, 0],
                "fov": 60,
                "near": 0.1,
                "far": 5.0,
                "intrinsic_matrix": np.array([
                    [525.0, 0, 320.0],
                    [0, 525.0, 240.0],
                    [0, 0, 1.0]
                ], dtype=np.float32),
                "distortion_coeffs": np.zeros(5, dtype=np.float32)
            }
            
            # Initialize vision system
            if self.use_real_camera:
                self.logger.info("Attempting to use real camera...")
                camera_configs = [mono_camera_config]
                stereo_configs = None
            else:
                self.logger.info("Using PyBullet virtual cameras...")
                camera_configs = [mono_camera_config]
                stereo_configs = (left_camera_config, right_camera_config)
            
            # Create vision system
            self.vision_system = Vision(
                camera_configs=camera_configs,
                stereo_configs=stereo_configs,
                use_pybullet_debug=True,
                show_plot=False,  # We'll handle plotting manually
                physics_client=self.pybullet_client
            )
            
            # Initialize perception system
            self.perception_system = Perception(
                vision_instance=self.vision_system,
                logger_name="PerceptionDemo"
            )
            
            self.logger.info("Vision and Perception systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup vision system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demonstrate_basic_image_capture(self):
        """Demonstrate basic image capture capabilities."""
        self.logger.info("=== Basic Image Capture Demonstration ===")
        
        try:
            if not self.vision_system:
                self.logger.warning("Vision system not available, skipping image capture")
                return
            
            # Capture images from different camera positions
            camera_positions = [
                {"translation": [0, 0, 1.0], "rotation": [0, -30, 0], "name": "overhead"},
                {"translation": [1.0, 0, 0.5], "rotation": [0, 0, -90], "name": "side"},
                {"translation": [0, 1.0, 0.5], "rotation": [0, 0, 180], "name": "front"}
            ]
            
            captured_images = []
            
            for i, pos_config in enumerate(camera_positions):
                try:
                    self.logger.info(f"Capturing image from {pos_config['name']} position")
                    
                    # For PyBullet cameras, we need to use the camera interface
                    rgb_image, depth_image = self.vision_system.capture_image(camera_index=0)
                    
                    if rgb_image is not None:
                        captured_images.append({
                            'rgb': rgb_image,
                            'depth': depth_image,
                            'position': pos_config['name']
                        })
                        
                        # Save individual images
                        self.save_image(rgb_image, f"rgb_{pos_config['name']}.png", "capture")
                        if depth_image is not None:
                            # Normalize depth for visualization
                            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            self.save_image(depth_normalized, f"depth_{pos_config['name']}.png", "capture")
                        
                        self.logger.info(f"Captured {pos_config['name']} image: {rgb_image.shape}")
                    else:
                        self.logger.warning(f"Failed to capture image from {pos_config['name']} position")
                
                except Exception as e:
                    self.logger.warning(f"Image capture failed for {pos_config['name']}: {e}")
            
            # Create comparison plot
            if captured_images:
                fig, axes = plt.subplots(2, len(captured_images), figsize=(5*len(captured_images), 8))
                if len(captured_images) == 1:
                    axes = axes.reshape(-1, 1)
                
                for i, img_data in enumerate(captured_images):
                    # RGB image
                    axes[0, i].imshow(img_data['rgb'])
                    axes[0, i].set_title(f"RGB - {img_data['position']}")
                    axes[0, i].axis('off')
                    
                    # Depth image
                    if img_data['depth'] is not None:
                        depth_display = axes[1, i].imshow(img_data['depth'], cmap='jet')
                        axes[1, i].set_title(f"Depth - {img_data['position']}")
                        axes[1, i].axis('off')
                        plt.colorbar(depth_display, ax=axes[1, i], fraction=0.046)
                    else:
                        axes[1, i].text(0.5, 0.5, 'No Depth Data', ha='center', va='center', 
                                       transform=axes[1, i].transAxes)
                        axes[1, i].set_title(f"Depth - {img_data['position']}")
                        axes[1, i].axis('off')
                
                plt.tight_layout()
                self.save_plot('image_capture_comparison.png', 'capture')
                plt.close()
            
            self.demo_results['image_capture'] = {
                'images_captured': len(captured_images),
                'positions': [img['position'] for img in captured_images]
            }
            
            self.logger.info(f"Image capture demonstration completed: {len(captured_images)} images")
            
        except Exception as e:
            self.logger.error(f"Image capture demonstration failed: {e}")
            plt.close()
    
    def demonstrate_obstacle_detection(self):
        """Demonstrate obstacle detection using depth and YOLO."""
        self.logger.info("=== Obstacle Detection Demonstration ===")
        
        try:
            if not self.perception_system:
                self.logger.warning("Perception system not available, skipping obstacle detection")
                return
            
            # Capture current scene
            rgb_image, depth_image = self.vision_system.capture_image(camera_index=0)
            
            if rgb_image is None:
                self.logger.warning("No image captured for obstacle detection")
                return
            
            # Detect obstacles
            self.logger.info("Running obstacle detection...")
            obstacle_points, labels = self.perception_system.detect_and_cluster_obstacles(
                camera_index=0,
                depth_threshold=3.0,  # 3 meters max
                step=2,               # Downsample by factor of 2
                eps=0.1,              # DBSCAN epsilon
                min_samples=3         # DBSCAN min samples
            )
            
            self.logger.info(f"Detected {len(obstacle_points)} obstacle points")
            self.logger.info(f"Found {len(np.unique(labels))} clusters (including noise)")
            
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            
            # Original RGB image
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(rgb_image)
            ax1.set_title('Original RGB Image')
            ax1.axis('off')
            
            # Depth image
            ax2 = fig.add_subplot(2, 3, 2)
            if depth_image is not None:
                depth_display = ax2.imshow(depth_image, cmap='jet')
                ax2.set_title('Depth Image')
                ax2.axis('off')
                plt.colorbar(depth_display, ax=ax2, fraction=0.046)
            
            # 3D obstacle points
            if len(obstacle_points) > 0:
                ax3 = fig.add_subplot(2, 3, 3, projection='3d')
                
                # Color points by cluster
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    if label == -1:
                        # Noise points in black
                        mask = labels == label
                        ax3.scatter(obstacle_points[mask, 0], 
                                   obstacle_points[mask, 1], 
                                   obstacle_points[mask, 2], 
                                   c='black', s=20, alpha=0.5, label='Noise')
                    else:
                        mask = labels == label
                        ax3.scatter(obstacle_points[mask, 0], 
                                   obstacle_points[mask, 1], 
                                   obstacle_points[mask, 2], 
                                   c=[colors[i]], s=30, alpha=0.8, label=f'Cluster {label}')
                
                ax3.set_xlabel('X (m)')
                ax3.set_ylabel('Y (m)')
                ax3.set_zlabel('Z (m)')
                ax3.set_title('3D Obstacle Points')
                ax3.legend()
                
                # Top-down view
                ax4 = fig.add_subplot(2, 3, 4)
                for i, label in enumerate(unique_labels):
                    if label == -1:
                        mask = labels == label
                        ax4.scatter(obstacle_points[mask, 0], 
                                   obstacle_points[mask, 1], 
                                   c='black', s=20, alpha=0.5, label='Noise')
                    else:
                        mask = labels == label
                        ax4.scatter(obstacle_points[mask, 0], 
                                   obstacle_points[mask, 1], 
                                   c=[colors[i]], s=30, alpha=0.8, label=f'Cluster {label}')
                
                ax4.set_xlabel('X (m)')
                ax4.set_ylabel('Y (m)')
                ax4.set_title('Top-Down View')
                ax4.grid(True)
                ax4.axis('equal')
                
                # Cluster analysis
                ax5 = fig.add_subplot(2, 3, 5)
                cluster_labels = labels[labels != -1]  # Exclude noise
                if len(cluster_labels) > 0:
                    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
                    ax5.bar(unique_clusters, cluster_counts)
                    ax5.set_xlabel('Cluster ID')
                    ax5.set_ylabel('Number of Points')
                    ax5.set_title('Cluster Sizes')
                    ax5.grid(True)
                
                # Distance histogram
                ax6 = fig.add_subplot(2, 3, 6)
                distances = np.linalg.norm(obstacle_points, axis=1)
                ax6.hist(distances, bins=20, alpha=0.7, edgecolor='black')
                ax6.set_xlabel('Distance from Camera (m)')
                ax6.set_ylabel('Number of Points')
                ax6.set_title('Distance Distribution')
                ax6.grid(True)
            
            plt.tight_layout()
            self.save_plot('obstacle_detection_analysis.png', 'obstacles')
            plt.close()
            
            # Save obstacle data
            if len(obstacle_points) > 0:
                # Save as separate images for better visualization
                self.save_image(rgb_image, 'obstacle_detection_rgb.png', 'obstacles')
                if depth_image is not None:
                    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.save_image(depth_normalized, 'obstacle_detection_depth.png', 'obstacles')
            
            self.demo_results['obstacle_detection'] = {
                'points_detected': len(obstacle_points),
                'clusters_found': len(np.unique(labels)) - (1 if -1 in labels else 0),
                'noise_points': np.sum(labels == -1) if len(labels) > 0 else 0
            }
            
            self.logger.info("Obstacle detection demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Obstacle detection demonstration failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def demonstrate_stereo_vision(self):
        """Demonstrate stereo vision and 3D point cloud generation."""
        self.logger.info("=== Stereo Vision Demonstration ===")
        
        try:
            if not self.vision_system or not self.vision_system.stereo_enabled:
                self.logger.warning("Stereo vision not available, skipping demonstration")
                return
            
            # Generate synthetic stereo images for demonstration
            self.logger.info("Generating synthetic stereo pair...")
            
            # Create simple synthetic stereo images
            height, width = 240, 320
            
            # Left image - simple scene with objects at different depths
            left_img = np.zeros((height, width, 3), dtype=np.uint8)
            # Add some geometric shapes
            cv2.rectangle(left_img, (50, 50), (150, 150), (255, 0, 0), -1)   # Red square
            cv2.circle(left_img, (250, 120), 40, (0, 255, 0), -1)            # Green circle
            cv2.rectangle(left_img, (100, 180), (220, 220), (0, 0, 255), -1) # Blue rectangle
            
            # Right image - shifted for stereo effect
            right_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(right_img, (45, 50), (145, 150), (255, 0, 0), -1)   # Red square (shifted)
            cv2.circle(right_img, (245, 120), 40, (0, 255, 0), -1)            # Green circle (shifted)
            cv2.rectangle(right_img, (95, 180), (215, 220), (0, 0, 255), -1)  # Blue rectangle (shifted)
            
            # Save stereo pair
            self.save_image(left_img, 'stereo_left.png', 'stereo')
            self.save_image(right_img, 'stereo_right.png', 'stereo')
            
            # Compute stereo rectification maps
            try:
                self.vision_system.compute_stereo_rectification_maps()
                self.logger.info("Stereo rectification maps computed")
                
                # Rectify images
                left_rect, right_rect = self.vision_system.rectify_stereo_images(left_img, right_img)
                
                # Compute disparity
                disparity = self.vision_system.compute_disparity(left_rect, right_rect)
                
                # Generate point cloud
                point_cloud = self.vision_system.disparity_to_pointcloud(disparity)
                
                self.logger.info(f"Generated point cloud with {len(point_cloud)} points")
                
                # Visualize results
                fig = plt.figure(figsize=(15, 10))
                
                # Original stereo pair
                ax1 = fig.add_subplot(2, 3, 1)
                ax1.imshow(left_img)
                ax1.set_title('Left Image')
                ax1.axis('off')
                
                ax2 = fig.add_subplot(2, 3, 2)
                ax2.imshow(right_img)
                ax2.set_title('Right Image')
                ax2.axis('off')
                
                # Rectified images
                ax3 = fig.add_subplot(2, 3, 3)
                ax3.imshow(left_rect)
                ax3.set_title('Left Rectified')
                ax3.axis('off')
                
                # Disparity map
                ax4 = fig.add_subplot(2, 3, 4)
                disp_display = ax4.imshow(disparity, cmap='jet')
                ax4.set_title('Disparity Map')
                ax4.axis('off')
                plt.colorbar(disp_display, ax=ax4, fraction=0.046)
                
                # 3D point cloud
                if len(point_cloud) > 0:
                    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
                    # Subsample for visualization
                    step = max(1, len(point_cloud) // 1000)
                    pc_sample = point_cloud[::step]
                    
                    ax5.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2], 
                               c=pc_sample[:, 2], cmap='viridis', s=1)
                    ax5.set_xlabel('X')
                    ax5.set_ylabel('Y')
                    ax5.set_zlabel('Z')
                    ax5.set_title('3D Point Cloud')
                
                # Point cloud statistics
                ax6 = fig.add_subplot(2, 3, 6)
                if len(point_cloud) > 0:
                    depths = point_cloud[:, 2]
                    ax6.hist(depths, bins=30, alpha=0.7, edgecolor='black')
                    ax6.set_xlabel('Depth (Z)')
                    ax6.set_ylabel('Number of Points')
                    ax6.set_title('Depth Distribution')
                    ax6.grid(True)
                
                plt.tight_layout()
                self.save_plot('stereo_vision_analysis.png', 'stereo')
                plt.close()
                
                self.demo_results['stereo_vision'] = {
                    'point_cloud_size': len(point_cloud),
                    'disparity_range': [float(np.min(disparity)), float(np.max(disparity))],
                    'depth_range': [float(np.min(point_cloud[:, 2])), float(np.max(point_cloud[:, 2]))] if len(point_cloud) > 0 else [0, 0]
                }
                
            except Exception as e:
                self.logger.warning(f"Stereo processing failed: {e}")
                self.demo_results['stereo_vision'] = {'error': str(e)}
            
            self.logger.info("Stereo vision demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Stereo vision demonstration failed: {e}")
            plt.close()
    
    def demonstrate_advanced_perception(self):
        """Demonstrate advanced perception features and analysis."""
        self.logger.info("=== Advanced Perception Demonstration ===")
        
        try:
            # Create synthetic scene data for analysis
            self.logger.info("Generating synthetic perception data...")
            
            # Simulate multiple object detections over time
            num_frames = 20
            detection_history = []
            
            for frame in range(num_frames):
                # Simulate moving objects
                objects = []
                
                # Object 1: Moving in a circle
                angle = frame * 0.3
                obj1_pos = [0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.5]
                objects.append({'id': 1, 'position': obj1_pos, 'size': 0.1, 'type': 'cube'})
                
                # Object 2: Moving linearly
                obj2_pos = [0.2 + frame * 0.02, 0.0, 0.3]
                objects.append({'id': 2, 'position': obj2_pos, 'size': 0.08, 'type': 'sphere'})
                
                # Object 3: Stationary
                obj3_pos = [-0.3, -0.2, 0.4]
                objects.append({'id': 3, 'position': obj3_pos, 'size': 0.12, 'type': 'cylinder'})
                
                detection_history.append({
                    'frame': frame,
                    'timestamp': frame * 0.1,  # 10 Hz simulation
                    'objects': objects
                })
            
            # Analyze detection history
            self.logger.info("Analyzing object tracking and motion...")
            
            # Extract trajectories
            trajectories = {}
            for detection in detection_history:
                for obj in detection['objects']:
                    obj_id = obj['id']
                    if obj_id not in trajectories:
                        trajectories[obj_id] = {
                            'positions': [],
                            'timestamps': [],
                            'type': obj['type']
                        }
                    trajectories[obj_id]['positions'].append(obj['position'])
                    trajectories[obj_id]['timestamps'].append(detection['timestamp'])
            
            # Convert to numpy arrays
            for obj_id in trajectories:
                trajectories[obj_id]['positions'] = np.array(trajectories[obj_id]['positions'])
                trajectories[obj_id]['timestamps'] = np.array(trajectories[obj_id]['timestamps'])
            
            # Calculate velocities and accelerations
            motion_analysis = {}
            for obj_id, traj in trajectories.items():
                positions = traj['positions']
                timestamps = traj['timestamps']
                
                # Calculate velocities
                velocities = []
                for i in range(1, len(positions)):
                    dt = timestamps[i] - timestamps[i-1]
                    if dt > 0:
                        vel = (positions[i] - positions[i-1]) / dt
                        velocities.append(vel)
                
                # Calculate accelerations
                accelerations = []
                for i in range(1, len(velocities)):
                    dt = timestamps[i+1] - timestamps[i]
                    if dt > 0:
                        acc = (velocities[i] - velocities[i-1]) / dt
                        accelerations.append(acc)
                
                motion_analysis[obj_id] = {
                    'velocities': np.array(velocities) if velocities else np.array([]),
                    'accelerations': np.array(accelerations) if accelerations else np.array([]),
                    'avg_speed': np.mean([np.linalg.norm(v) for v in velocities]) if velocities else 0,
                    'max_speed': np.max([np.linalg.norm(v) for v in velocities]) if velocities else 0
                }
            
            # Visualize advanced perception analysis
            fig = plt.figure(figsize=(18, 12))
            
            # 3D trajectories
            ax1 = fig.add_subplot(2, 4, 1, projection='3d')
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, (obj_id, traj) in enumerate(trajectories.items()):
                positions = traj['positions']
                color = colors[i % len(colors)]
                ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                        color=color, marker='o', label=f'Object {obj_id} ({traj["type"]})')
                # Mark start and end
                ax1.scatter(*positions[0], color=color, s=100, marker='s', alpha=0.7)
                ax1.scatter(*positions[-1], color=color, s=100, marker='^', alpha=0.7)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Object Trajectories')
            ax1.legend()
            
            # Top-down view
            ax2 = fig.add_subplot(2, 4, 2)
            for i, (obj_id, traj) in enumerate(trajectories.items()):
                positions = traj['positions']
                color = colors[i % len(colors)]
                ax2.plot(positions[:, 0], positions[:, 1], 
                        color=color, marker='o', label=f'Object {obj_id}')
                ax2.scatter(positions[0, 0], positions[0, 1], color=color, s=100, marker='s')
                ax2.scatter(positions[-1, 0], positions[-1, 1], color=color, s=100, marker='^')
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Top-Down Trajectories')
            ax2.grid(True)
            ax2.legend()
            ax2.axis('equal')
            
            # Speed analysis
            ax3 = fig.add_subplot(2, 4, 3)
            obj_ids = list(motion_analysis.keys())
            avg_speeds = [motion_analysis[obj_id]['avg_speed'] for obj_id in obj_ids]
            max_speeds = [motion_analysis[obj_id]['max_speed'] for obj_id in obj_ids]
            
            x = np.arange(len(obj_ids))
            width = 0.35
            ax3.bar(x - width/2, avg_speeds, width, label='Average Speed', alpha=0.7)
            ax3.bar(x + width/2, max_speeds, width, label='Max Speed', alpha=0.7)
            ax3.set_xlabel('Object ID')
            ax3.set_ylabel('Speed (m/s)')
            ax3.set_title('Object Speed Analysis')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'Obj {id}' for id in obj_ids])
            ax3.legend()
            ax3.grid(True)
            
            # Velocity profiles over time
            ax4 = fig.add_subplot(2, 4, 4)
            for i, (obj_id, analysis) in enumerate(motion_analysis.items()):
                if len(analysis['velocities']) > 0:
                    timestamps = trajectories[obj_id]['timestamps'][1:]  # Skip first point
                    speeds = [np.linalg.norm(v) for v in analysis['velocities']]
                    color = colors[i % len(colors)]
                    ax4.plot(timestamps, speeds, color=color, marker='o', label=f'Object {obj_id}')
            
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Speed (m/s)')
            ax4.set_title('Speed vs Time')
            ax4.legend()
            ax4.grid(True)
            
            # Object detection statistics
            ax5 = fig.add_subplot(2, 4, 5)
            detection_counts = {}
            for detection in detection_history:
                frame = detection['frame']
                count = len(detection['objects'])
                detection_counts[frame] = count
            
            frames = list(detection_counts.keys())
            counts = list(detection_counts.values())
            ax5.plot(frames, counts, 'b-o')
            ax5.set_xlabel('Frame Number')
            ax5.set_ylabel('Objects Detected')
            ax5.set_title('Detection Count Over Time')
            ax5.grid(True)
            
            # Distance from camera analysis
            ax6 = fig.add_subplot(2, 4, 6)
            all_distances = []
            for detection in detection_history:
                for obj in detection['objects']:
                    distance = np.linalg.norm(obj['position'])
                    all_distances.append(distance)
            
            ax6.hist(all_distances, bins=15, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Distance from Camera (m)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Object Distance Distribution')
            ax6.grid(True)
            
            # Object size analysis
            ax7 = fig.add_subplot(2, 4, 7)
            obj_types = {}
            for detection in detection_history:
                for obj in detection['objects']:
                    obj_type = obj['type']
                    if obj_type not in obj_types:
                        obj_types[obj_type] = []
                    obj_types[obj_type].append(obj['size'])
            
            type_names = list(obj_types.keys())
            avg_sizes = [np.mean(obj_types[t]) for t in type_names]
            ax7.bar(type_names, avg_sizes, alpha=0.7)
            ax7.set_xlabel('Object Type')
            ax7.set_ylabel('Average Size (m)')
            ax7.set_title('Object Size by Type')
            ax7.grid(True)
            
            # Prediction accuracy simulation
            ax8 = fig.add_subplot(2, 4, 8)
            # Simulate prediction vs actual positions
            prediction_errors = []
            for obj_id, traj in trajectories.items():
                positions = traj['positions']
                if len(positions) > 2:
                    # Simple linear prediction
                    for i in range(2, len(positions)):
                        # Predict next position based on velocity
                        velocity = positions[i-1] - positions[i-2]
                        predicted = positions[i-1] + velocity
                        actual = positions[i]
                        error = np.linalg.norm(predicted - actual)
                        prediction_errors.append(error)
            
            if prediction_errors:
                ax8.hist(prediction_errors, bins=10, alpha=0.7, edgecolor='black')
                ax8.set_xlabel('Prediction Error (m)')
                ax8.set_ylabel('Frequency')
                ax8.set_title('Motion Prediction Accuracy')
                ax8.grid(True)
            
            plt.tight_layout()
            self.save_plot('advanced_perception_analysis.png', 'advanced')
            plt.close()
            
            # Generate synthetic clustering demonstration
            self.demonstrate_clustering_analysis()
            
            self.demo_results['advanced_perception'] = {
                'objects_tracked': len(trajectories),
                'total_detections': sum(len(d['objects']) for d in detection_history),
                'avg_objects_per_frame': np.mean([len(d['objects']) for d in detection_history]),
                'motion_analysis': {obj_id: {'avg_speed': float(analysis['avg_speed']), 
                                           'max_speed': float(analysis['max_speed'])} 
                                  for obj_id, analysis in motion_analysis.items()}
            }
            
            self.logger.info("Advanced perception demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Advanced perception demonstration failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def demonstrate_clustering_analysis(self):
        """Demonstrate clustering analysis on synthetic 3D point data."""
        self.logger.info("=== Clustering Analysis Demonstration ===")
        
        try:
            # Generate synthetic 3D point cloud data
            np.random.seed(42)  # For reproducible results
            
            # Create clusters of points
            cluster_centers = [
                [0.5, 0.2, 0.3],
                [0.8, -0.1, 0.4],
                [-0.3, 0.4, 0.5],
                [0.1, -0.5, 0.2]
            ]
            
            all_points = []
            true_labels = []
            
            for i, center in enumerate(cluster_centers):
                # Generate points around each center
                n_points = np.random.randint(20, 50)
                cluster_points = np.random.normal(center, 0.08, (n_points, 3))
                all_points.extend(cluster_points)
                true_labels.extend([i] * n_points)
            
            # Add some noise points
            n_noise = 15
            noise_points = np.random.uniform(-1, 1, (n_noise, 3))
            all_points.extend(noise_points)
            true_labels.extend([-1] * n_noise)
            
            points = np.array(all_points)
            true_labels = np.array(true_labels)
            
            self.logger.info(f"Generated {len(points)} points with {len(cluster_centers)} clusters")
            
            # Test different clustering parameters
            eps_values = [0.1, 0.15, 0.2, 0.25]
            min_samples_values = [3, 5, 8]
            
            clustering_results = []
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    # Perform clustering
                    try:
                        if hasattr(self.perception_system, 'cluster_obstacles'):
                            labels, num_clusters = self.perception_system.cluster_obstacles(
                                points, eps=eps, min_samples=min_samples
                            )
                        else:
                            # Fallback to sklearn directly
                            from sklearn.cluster import DBSCAN
                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = dbscan.fit_predict(points)
                            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        # Calculate metrics
                        noise_points = np.sum(labels == -1)
                        silhouette_score = 0  # Placeholder
                        
                        clustering_results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'labels': labels,
                            'num_clusters': num_clusters,
                            'noise_points': noise_points,
                            'silhouette_score': silhouette_score
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Clustering failed for eps={eps}, min_samples={min_samples}: {e}")
            
            # Visualize clustering results
            if clustering_results:
                # Find best result (most clusters with least noise)
                best_result = max(clustering_results, 
                                key=lambda x: x['num_clusters'] - x['noise_points']/10)
                
                fig = plt.figure(figsize=(20, 12))
                
                # Original data with true labels
                ax1 = fig.add_subplot(2, 4, 1, projection='3d')
                unique_true = np.unique(true_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_true)))
                
                for i, label in enumerate(unique_true):
                    mask = true_labels == label
                    if label == -1:
                        ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                   c='black', s=20, alpha=0.5, label='True Noise')
                    else:
                        ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                   c=[colors[i]], s=30, alpha=0.8, label=f'True Cluster {label}')
                
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.set_title('True Clusters')
                ax1.legend()
                
                # Best clustering result
                ax2 = fig.add_subplot(2, 4, 2, projection='3d')
                best_labels = best_result['labels']
                unique_pred = np.unique(best_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pred)))
                
                for i, label in enumerate(unique_pred):
                    mask = best_labels == label
                    if label == -1:
                        ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                   c='black', s=20, alpha=0.5, label='Predicted Noise')
                    else:
                        ax2.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                   c=[colors[i]], s=30, alpha=0.8, label=f'Pred Cluster {label}')
                
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                ax2.set_title(f'Best Clustering (eps={best_result["eps"]}, min_samples={best_result["min_samples"]})')
                ax2.legend()
                
                # Parameter sensitivity analysis
                ax3 = fig.add_subplot(2, 4, 3)
                eps_vals = [r['eps'] for r in clustering_results]
                num_clusters_vals = [r['num_clusters'] for r in clustering_results]
                scatter = ax3.scatter(eps_vals, num_clusters_vals, 
                                    c=[r['min_samples'] for r in clustering_results], 
                                    cmap='viridis', s=60, alpha=0.7)
                ax3.set_xlabel('Epsilon')
                ax3.set_ylabel('Number of Clusters')
                ax3.set_title('Parameter Sensitivity')
                ax3.grid(True)
                plt.colorbar(scatter, ax=ax3, label='Min Samples')
                
                # Noise vs clusters
                ax4 = fig.add_subplot(2, 4, 4)
                noise_vals = [r['noise_points'] for r in clustering_results]
                ax4.scatter(num_clusters_vals, noise_vals, alpha=0.7)
                ax4.set_xlabel('Number of Clusters')
                ax4.set_ylabel('Noise Points')
                ax4.set_title('Clusters vs Noise')
                ax4.grid(True)
                
                # Show different eps values
                selected_results = [r for r in clustering_results if r['min_samples'] == 5][:4]
                
                for idx, result in enumerate(selected_results):
                    ax = fig.add_subplot(2, 4, 5 + idx, projection='3d')
                    labels = result['labels']
                    unique_labels = np.unique(labels)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = labels == label
                        if label == -1:
                            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                     c='black', s=15, alpha=0.5)
                        else:
                            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                                     c=[colors[i]], s=25, alpha=0.8)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'eps={result["eps"]}\n{result["num_clusters"]} clusters')
                
                plt.tight_layout()
                self.save_plot('clustering_analysis.png', 'clustering')
                plt.close()
                
                # Performance comparison chart
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Clusters found vs parameters
                eps_unique = sorted(set(r['eps'] for r in clustering_results))
                min_samples_unique = sorted(set(r['min_samples'] for r in clustering_results))
                
                cluster_matrix = np.zeros((len(min_samples_unique), len(eps_unique)))
                for r in clustering_results:
                    i = min_samples_unique.index(r['min_samples'])
                    j = eps_unique.index(r['eps'])
                    cluster_matrix[i, j] = r['num_clusters']
                
                im1 = ax1.imshow(cluster_matrix, cmap='viridis', aspect='auto')
                ax1.set_xticks(range(len(eps_unique)))
                ax1.set_xticklabels([f'{e:.2f}' for e in eps_unique])
                ax1.set_yticks(range(len(min_samples_unique)))
                ax1.set_yticklabels(min_samples_unique)
                ax1.set_xlabel('Epsilon')
                ax1.set_ylabel('Min Samples')
                ax1.set_title('Number of Clusters')
                plt.colorbar(im1, ax=ax1)
                
                # Noise matrix
                noise_matrix = np.zeros((len(min_samples_unique), len(eps_unique)))
                for r in clustering_results:
                    i = min_samples_unique.index(r['min_samples'])
                    j = eps_unique.index(r['eps'])
                    noise_matrix[i, j] = r['noise_points']
                
                im2 = ax2.imshow(noise_matrix, cmap='Reds', aspect='auto')
                ax2.set_xticks(range(len(eps_unique)))
                ax2.set_xticklabels([f'{e:.2f}' for e in eps_unique])
                ax2.set_yticks(range(len(min_samples_unique)))
                ax2.set_yticklabels(min_samples_unique)
                ax2.set_xlabel('Epsilon')
                ax2.set_ylabel('Min Samples')
                ax2.set_title('Noise Points')
                plt.colorbar(im2, ax=ax2)
                
                # Best parameters for each eps
                best_per_eps = {}
                for eps in eps_unique:
                    eps_results = [r for r in clustering_results if r['eps'] == eps]
                    if eps_results:
                        best = max(eps_results, key=lambda x: x['num_clusters'] - x['noise_points']/10)
                        best_per_eps[eps] = best
                
                eps_list = list(best_per_eps.keys())
                clusters_list = [best_per_eps[eps]['num_clusters'] for eps in eps_list]
                noise_list = [best_per_eps[eps]['noise_points'] for eps in eps_list]
                
                ax3.plot(eps_list, clusters_list, 'b-o', label='Clusters')
                ax3_twin = ax3.twinx()
                ax3_twin.plot(eps_list, noise_list, 'r-s', label='Noise Points')
                ax3.set_xlabel('Epsilon')
                ax3.set_ylabel('Number of Clusters', color='b')
                ax3_twin.set_ylabel('Noise Points', color='r')
                ax3.set_title('Best Results per Epsilon')
                ax3.grid(True)
                
                # Summary statistics
                ax4.axis('off')
                summary_text = f"""Clustering Analysis Summary:
                
Total Points: {len(points)}
True Clusters: {len(cluster_centers)}
True Noise Points: {n_noise}

Best Result:
- Epsilon: {best_result['eps']}
- Min Samples: {best_result['min_samples']}
- Clusters Found: {best_result['num_clusters']}
- Noise Points: {best_result['noise_points']}

Parameter Ranges Tested:
- Epsilon: {min(eps_values):.2f} - {max(eps_values):.2f}
- Min Samples: {min(min_samples_values)} - {max(min_samples_values)}
"""
                ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                plt.tight_layout()
                self.save_plot('clustering_performance.png', 'clustering')
                plt.close()
            
            self.demo_results['clustering_analysis'] = {
                'total_points': len(points),
                'true_clusters': len(cluster_centers),
                'parameter_combinations_tested': len(clustering_results),
                'best_result': {
                    'eps': float(best_result['eps']),
                    'min_samples': int(best_result['min_samples']),
                    'clusters_found': int(best_result['num_clusters']),
                    'noise_points': int(best_result['noise_points'])
                } if clustering_results else None
            }
            
            self.logger.info("Clustering analysis demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Clustering analysis demonstration failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def demonstrate_vision_features(self):
        """Demonstrate various vision system features."""
        self.logger.info("=== Vision Features Demonstration ===")
        
        try:
            # Test different image processing techniques
            self.logger.info("Testing image processing features...")
            
            # Create test images
            test_images = self.create_test_images()
            
            # Process each test image
            processed_results = []
            
            for i, (name, image) in enumerate(test_images.items()):
                result = {
                    'name': name,
                    'original': image,
                    'processed': {}
                }
                
                # Edge detection
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                edges = cv2.Canny(gray, 50, 150)
                result['processed']['edges'] = edges
                
                # Gaussian blur
                blurred = cv2.GaussianBlur(image, (15, 15), 0)
                result['processed']['blurred'] = blurred
                
                # Histogram equalization (for grayscale)
                if len(gray.shape) == 2:
                    equalized = cv2.equalizeHist(gray)
                    result['processed']['equalized'] = equalized
                
                # Morphological operations
                kernel = np.ones((5, 5), np.uint8)
                if len(image.shape) == 3:
                    morph_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    morph_img = image
                
                # Threshold first
                _, thresh = cv2.threshold(morph_img, 127, 255, cv2.THRESH_BINARY)
                
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                result['processed']['opening'] = opening
                result['processed']['closing'] = closing
                
                processed_results.append(result)
            
            # Visualize processing results
            fig = plt.figure(figsize=(20, 15))
            
            for i, result in enumerate(processed_results):
                row = i * 2
                
                # Original image
                ax = plt.subplot(len(processed_results)*2, 4, row*4 + 1)
                if len(result['original'].shape) == 3:
                    plt.imshow(result['original'])
                else:
                    plt.imshow(result['original'], cmap='gray')
                plt.title(f"Original - {result['name']}")
                plt.axis('off')
                
                # Edge detection
                ax = plt.subplot(len(processed_results)*2, 4, row*4 + 2)
                plt.imshow(result['processed']['edges'], cmap='gray')
                plt.title(f"Edges - {result['name']}")
                plt.axis('off')
                
                # Blurred
                ax = plt.subplot(len(processed_results)*2, 4, row*4 + 3)
                if len(result['processed']['blurred'].shape) == 3:
                    plt.imshow(result['processed']['blurred'])
                else:
                    plt.imshow(result['processed']['blurred'], cmap='gray')
                plt.title(f"Blurred - {result['name']}")
                plt.axis('off')
                
                # Morphology
                ax = plt.subplot(len(processed_results)*2, 4, row*4 + 4)
                plt.imshow(result['processed']['opening'], cmap='gray')
                plt.title(f"Morphology - {result['name']}")
                plt.axis('off')
                
                # Second row with additional processing
                if 'equalized' in result['processed']:
                    ax = plt.subplot(len(processed_results)*2, 4, (row+1)*4 + 1)
                    plt.imshow(result['processed']['equalized'], cmap='gray')
                    plt.title(f"Equalized - {result['name']}")
                    plt.axis('off')
                
                ax = plt.subplot(len(processed_results)*2, 4, (row+1)*4 + 2)
                plt.imshow(result['processed']['closing'], cmap='gray')
                plt.title(f"Closing - {result['name']}")
                plt.axis('off')
                
                # Feature analysis
                ax = plt.subplot(len(processed_results)*2, 4, (row+1)*4 + 3)
                # Simple feature analysis - count edge pixels
                edge_count = np.sum(result['processed']['edges'] > 0)
                total_pixels = result['processed']['edges'].size
                edge_density = edge_count / total_pixels
                
                plt.bar(['Edge Density'], [edge_density])
                plt.title(f"Features - {result['name']}")
                plt.ylabel('Density')
                
                # Histogram
                ax = plt.subplot(len(processed_results)*2, 4, (row+1)*4 + 4)
                if len(result['original'].shape) == 3:
                    hist_img = cv2.cvtColor(result['original'], cv2.COLOR_RGB2GRAY)
                else:
                    hist_img = result['original']
                
                hist, bins = np.histogram(hist_img.flatten(), 256, [0, 256])
                plt.plot(hist)
                plt.title(f"Histogram - {result['name']}")
                plt.xlabel('Intensity')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            self.save_plot('vision_features_analysis.png', 'vision')
            plt.close()
            
            # Save processed images
            for result in processed_results:
                name = result['name']
                for proc_name, proc_img in result['processed'].items():
                    filename = f"vision_{name}_{proc_name}.png"
                    self.save_image(proc_img, filename, 'vision')
            
            self.demo_results['vision_features'] = {
                'images_processed': len(processed_results),
                'processing_techniques': ['edges', 'blur', 'equalization', 'morphology', 'histogram']
            }
            
            self.logger.info("Vision features demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Vision features demonstration failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
    
    def create_test_images(self):
        """Create synthetic test images for vision processing."""
        test_images = {}
        
        # Geometric shapes image
        height, width = 240, 320
        shapes_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add various shapes
        cv2.rectangle(shapes_img, (50, 50), (150, 150), (255, 0, 0), -1)   # Red square
        cv2.circle(shapes_img, (250, 120), 40, (0, 255, 0), -1)            # Green circle
        cv2.rectangle(shapes_img, (100, 180), (220, 220), (0, 0, 255), -1) # Blue rectangle
        
        # Draw some lines
        cv2.line(shapes_img, (0, height//2), (width, height//2), (255, 255, 0), 3)
        cv2.line(shapes_img, (width//2, 0), (width//2, height), (255, 0, 255), 3)
        
        test_images['geometric_shapes'] = shapes_img
        
        # Noise image
        noise_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        test_images['noise'] = noise_img
        
        # Gradient image
        gradient_img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width):
            intensity = int(255 * i / width)
            gradient_img[:, i] = [intensity, intensity, intensity]
        test_images['gradient'] = gradient_img
        
        # Textured image
        textured_img = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                textured_img[i, j] = int(127 + 127 * np.sin(i/10) * np.cos(j/10))
        test_images['textured'] = textured_img
        
        return test_images
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        self.logger.info("=== Generating Summary Report ===")
        
        try:
            # Collect all demonstration data
            summary_data = {
                'system_info': {
                    'pybullet_available': self.pybullet_client is not None,
                    'vision_system_available': self.vision_system is not None,
                    'perception_system_available': self.perception_system is not None,
                    'outputs_generated': len(self.outputs_saved)
                },
                'demonstrations': self.demo_results
            }
            
            # Create summary visualization
            fig = plt.figure(figsize=(16, 12))
            
            # System status
            ax1 = plt.subplot(3, 3, 1)
            systems = ['PyBullet', 'Vision', 'Perception']
            status = [
                summary_data['system_info']['pybullet_available'],
                summary_data['system_info']['vision_system_available'],
                summary_data['system_info']['perception_system_available']
            ]
            colors = ['green' if s else 'red' for s in status]
            bars = ax1.bar(systems, [1 if s else 0 for s in status], color=colors)
            ax1.set_title('System Components Status')
            ax1.set_ylabel('Available')
            ax1.set_ylim(0, 1.2)
            
            # Add status labels
            for bar, stat in zip(bars, status):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        'Available' if stat else 'Not Available',
                        ha='center', va='bottom', fontsize=8)
            
            # Demonstration coverage
            ax2 = plt.subplot(3, 3, 2)
            demo_names = list(self.demo_results.keys())
            demo_success = [1 if demo else 0 for demo in self.demo_results.values()]
            
            if demo_names:
                ax2.barh(demo_names, demo_success, color='lightblue')
                ax2.set_title('Demonstrations Completed')
                ax2.set_xlabel('Success')
                ax2.set_xlim(0, 1.2)
            else:
                ax2.text(0.5, 0.5, 'No demonstrations\ncompleted', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Demonstrations Status')
            
            # Output files generated
            ax3 = plt.subplot(3, 3, 3)
            if self.outputs_saved:
                file_types = {}
                for filepath in self.outputs_saved:
                    ext = os.path.splitext(filepath)[1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                
                types = list(file_types.keys())
                counts = list(file_types.values())
                ax3.pie(counts, labels=types, autopct='%1.1f%%')
                ax3.set_title('Output Files by Type')
            else:
                ax3.text(0.5, 0.5, 'No output files\ngenerated', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Output Files')
            
            # Perception metrics (if available)
            ax4 = plt.subplot(3, 3, 4)
            if 'obstacle_detection' in self.demo_results:
                obs_data = self.demo_results['obstacle_detection']
                metrics = ['Points Detected', 'Clusters Found', 'Noise Points']
                values = [
                    obs_data.get('points_detected', 0),
                    obs_data.get('clusters_found', 0),
                    obs_data.get('noise_points', 0)
                ]
                ax4.bar(metrics, values)
                ax4.set_title('Obstacle Detection Metrics')
                ax4.set_ylabel('Count')
                plt.setp(ax4.get_xticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No obstacle detection\ndata available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Obstacle Detection')
            
            # Stereo vision metrics (if available)
            ax5 = plt.subplot(3, 3, 5)
            if 'stereo_vision' in self.demo_results and 'error' not in self.demo_results['stereo_vision']:
                stereo_data = self.demo_results['stereo_vision']
                point_cloud_size = stereo_data.get('point_cloud_size', 0)
                depth_range = stereo_data.get('depth_range', [0, 0])
                
                ax5.bar(['Point Cloud Size'], [point_cloud_size])
                ax5.set_title('Stereo Vision Metrics')
                ax5.set_ylabel('Points Generated')
                
                # Add depth range as text
                if depth_range[1] > depth_range[0]:
                    ax5.text(0.5, 0.8, f'Depth Range:\n{depth_range[0]:.2f} - {depth_range[1]:.2f} m',
                            transform=ax5.transAxes, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            else:
                ax5.text(0.5, 0.5, 'No stereo vision\ndata available', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Stereo Vision')
            
            # Advanced perception metrics (if available)
            ax6 = plt.subplot(3, 3, 6)
            if 'advanced_perception' in self.demo_results:
                adv_data = self.demo_results['advanced_perception']
                metrics = ['Objects Tracked', 'Total Detections', 'Avg/Frame']
                values = [
                    adv_data.get('objects_tracked', 0),
                    adv_data.get('total_detections', 0),
                    adv_data.get('avg_objects_per_frame', 0)
                ]
                ax6.bar(metrics, values)
                ax6.set_title('Advanced Perception Metrics')
                ax6.set_ylabel('Count')
                plt.setp(ax6.get_xticklabels(), rotation=45)
            else:
                ax6.text(0.5, 0.5, 'No advanced perception\ndata available', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Advanced Perception')
            
            # Clustering analysis (if available)
            ax7 = plt.subplot(3, 3, 7)
            if 'clustering_analysis' in self.demo_results:
                clust_data = self.demo_results['clustering_analysis']
                best_result = clust_data.get('best_result')
                if best_result:
                    metrics = ['Clusters Found', 'Noise Points']
                    values = [best_result['clusters_found'], best_result['noise_points']]
                    ax7.bar(metrics, values, color=['green', 'red'])
                    ax7.set_title(f'Clustering Results\n(eps={best_result["eps"]:.2f})')
                    ax7.set_ylabel('Count')
                else:
                    ax7.text(0.5, 0.5, 'Clustering analysis\nfailed', 
                            ha='center', va='center', transform=ax7.transAxes)
                    ax7.set_title('Clustering Analysis')
            else:
                ax7.text(0.5, 0.5, 'No clustering\ndata available', 
                        ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Clustering Analysis')
            
            # Vision features (if available)
            ax8 = plt.subplot(3, 3, 8)
            if 'vision_features' in self.demo_results:
                vis_data = self.demo_results['vision_features']
                techniques = vis_data.get('processing_techniques', [])
                if techniques:
                    ax8.barh(range(len(techniques)), [1]*len(techniques))
                    ax8.set_yticks(range(len(techniques)))
                    ax8.set_yticklabels(techniques)
                    ax8.set_title('Vision Processing Techniques')
                    ax8.set_xlabel('Applied')
                else:
                    ax8.text(0.5, 0.5, 'No vision features\ndata available', 
                            ha='center', va='center', transform=ax8.transAxes)
                    ax8.set_title('Vision Features')
            else:
                ax8.text(0.5, 0.5, 'No vision features\ndata available', 
                        ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Vision Features')
            
            # Summary statistics
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            # Compile summary text
            total_outputs = len(self.outputs_saved)
            demos_completed = len([d for d in self.demo_results.values() if d])
            
            summary_text = f"""Perception Demo Summary

System Components:
• PyBullet: {'✓' if summary_data['system_info']['pybullet_available'] else '✗'}
• Vision System: {'✓' if summary_data['system_info']['vision_system_available'] else '✗'}
• Perception System: {'✓' if summary_data['system_info']['perception_system_available'] else '✗'}

Demonstrations:
• Completed: {demos_completed}
• Total Outputs: {total_outputs}

Key Results:
• Images Captured: {self.demo_results.get('image_capture', {}).get('images_captured', 'N/A')}
• Obstacles Detected: {self.demo_results.get('obstacle_detection', {}).get('points_detected', 'N/A')}
• Point Cloud Size: {self.demo_results.get('stereo_vision', {}).get('point_cloud_size', 'N/A')}
"""
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            # Add title and timestamp
            fig.suptitle('ManipulaPy Perception Demo - Comprehensive Report', 
                         fontsize=16, fontweight='bold')
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                    ha='right', va='bottom', fontsize=8)
            
            plt.tight_layout()
            self.save_plot('perception_summary_report.png', 'summary')
            plt.close()
            
            # Save detailed text report
            report_file = self.script_dir / 'perception_demo_report.txt'
            with open(report_file, 'w') as f:
                f.write("ManipulaPy Perception Demo - Detailed Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {timestamp}\n\n")
                
                f.write("System Information:\n")
                f.write(f"  PyBullet Available: {summary_data['system_info']['pybullet_available']}\n")
                f.write(f"  Vision System Available: {summary_data['system_info']['vision_system_available']}\n")
                f.write(f"  Perception System Available: {summary_data['system_info']['perception_system_available']}\n")
                f.write(f"  Total Outputs Generated: {summary_data['system_info']['outputs_generated']}\n\n")
                
                if self.outputs_saved:
                    f.write("Generated Output Files:\n")
                    for output_path in self.outputs_saved:
                        f.write(f"  - {os.path.basename(output_path)}\n")
                    f.write("\n")
                
                f.write("Demonstration Results:\n")
                for demo_name, demo_data in self.demo_results.items():
                    f.write(f"  {demo_name}:\n")
                    if isinstance(demo_data, dict):
                        for key, value in demo_data.items():
                            f.write(f"    {key}: {value}\n")
                    else:
                        f.write(f"    Status: {demo_data}\n")
                    f.write("\n")
            
            if self.outputs_saved:
                self.outputs_saved.append(str(report_file))
            
            self.logger.info(f"Summary report saved to: {report_file}")
            self.logger.info("Summary report generation completed")
            
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Summary report generation failed: {e}")
            import traceback
            traceback.print_exc()
            plt.close()
            return {}
    
    def cleanup_resources(self):
        """Clean up PyBullet and other resources."""
        self.logger.info("Cleaning up resources...")
        
        try:
            # Release vision system resources
            if self.vision_system:
                self.vision_system.release()
            
            # Release perception system resources
            if self.perception_system:
                self.perception_system.release()
            
            # Disconnect PyBullet
            if self.pybullet_client is not None:
                pb.disconnect(self.pybullet_client)
                self.pybullet_client = None
            
            self.logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def run_complete_demonstration(self):
        """Run the complete perception demonstration sequence."""
        self.logger.info("Starting ManipulaPy Perception Intermediate Demo")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Setup phase
            self.logger.info("Setting up demonstration environment...")
            
            pybullet_ok = self.setup_pybullet_simulation()
            vision_ok = self.setup_vision_system()
            
            if not pybullet_ok:
                self.logger.warning("PyBullet setup failed, some demonstrations may be limited")
            
            if not vision_ok:
                self.logger.warning("Vision system setup failed, some demonstrations may be skipped")
            
            # Run demonstrations
            self.logger.info("Running demonstration sequence...")
            
            # Basic demonstrations
            self.demonstrate_basic_image_capture()
            self.demonstrate_vision_features()
            
            # Advanced demonstrations (if systems are available)
            if self.perception_system:
                self.demonstrate_obstacle_detection()
                self.demonstrate_advanced_perception()
            else:
                self.logger.warning("Skipping perception-based demonstrations")
            
            if self.vision_system and self.vision_system.stereo_enabled:
                self.demonstrate_stereo_vision()
            else:
                self.logger.warning("Skipping stereo vision demonstration")
            
            # Generate summary
            summary = self.generate_summary_report()
            
            total_time = time.time() - start_time
            self.logger.info(f"Demo completed successfully in {total_time:.2f} seconds")
            self.logger.info(f"Generated {len(self.outputs_saved)} output files")
            
            if self.save_outputs:
                self.logger.info("All outputs saved to: " + str(self.script_dir))
                self.logger.info("Output files:")
                for output_file in self.outputs_saved:
                    self.logger.info(f"  - {os.path.basename(output_file)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup resources
            self.cleanup_resources()
            plt.close('all')


def main():
    """Main function to run the perception demonstration."""
    print("ManipulaPy Perception Intermediate Demo")
    print("======================================")
    print("This demo will generate perception analysis plots and images.")
    print("Note: Some features require optional dependencies (YOLO, PyBullet, etc.)")
    print()
    
    # Configuration options
    use_real_camera = False  # Set to True to attempt using real camera devices
    save_outputs = True      # Set to False to disable file saving
    
    try:
        # Create and run demo
        demo = PerceptionIntermediateDemo(
            use_real_camera=use_real_camera, 
            save_outputs=save_outputs
        )
        summary = demo.run_complete_demonstration()
        
        print("\n" + "="*50)
        print("DEMO SUMMARY")
        print("="*50)
        print(f"PyBullet available: {summary.get('system_info', {}).get('pybullet_available', False)}")
        print(f"Vision system available: {summary.get('system_info', {}).get('vision_system_available', False)}")
        print(f"Perception system available: {summary.get('system_info', {}).get('perception_system_available', False)}")
        print(f"Output files generated: {summary.get('system_info', {}).get('outputs_generated', 0)}")
        print(f"Results saved to: {demo.script_dir}")
        
        if demo.outputs_saved:
            print("\nGenerated files:")
            for output_file in demo.outputs_saved:
                print(f"  - {os.path.basename(output_file)}")
        
        print(f"\nDemonstrations completed:")
        for demo_name, demo_data in summary.get('demonstrations', {}).items():
            status = "✓" if demo_data else "✗"
            print(f"  {status} {demo_name}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo finished. Check the generated files for detailed results.")


if __name__ == "__main__":
    main()