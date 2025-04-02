#!/usr/bin/env python3

import pybullet as pb
import pybullet_data
import numpy as np
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception
import random
import os
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import logging
import matplotlib

matplotlib.use('Agg')  # or 'TkAgg' if Tk is installed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PerceptionTest")

class ScenePerception:
    def __init__(self):
        """Initialize PyBullet and setup the scene."""
        # Initialize PyBullet
        self.physics_client = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.loadURDF("plane.urdf")
        
        # Initialize scene objects
        self.object_ids = []
        self.cameras = {}
        self.vision = None
        self.perception = None

    def setup_cameras(self):
        """Configure and setup stereo cameras."""
        cam_height = 1.2
        cam_distance = 2.0
        baseline = 0.2
        
        common_params = {
            "fov": 60,
            "near": 0.1,
            "far": 10.0,
            "intrinsic_matrix": np.array([
                [1000,    0,  320],
                [   0, 1000,  240],
                [   0,    0,    1]
            ], dtype=np.float32),
            "distortion_coeffs": np.zeros(5, dtype=np.float32),
            "use_opencv": False
        }
        
        left_config = common_params.copy()
        left_config.update({
            "name": "left_camera",
            "translation": [cam_distance, -baseline/2, cam_height],
            "rotation": [-30, 0, 0],
        })

        right_config = common_params.copy()
        right_config.update({
            "name": "right_camera",
            "translation": [cam_distance, baseline/2, cam_height],
            "rotation": [-30, 0, 0],
        })

        # Initialize Vision and Perception
        self.vision = Vision(
            camera_configs=[left_config, right_config],
            stereo_configs=(left_config, right_config),
            use_pybullet_debug=False,
            show_plot=True,
            physics_client=self.physics_client
        )
        
        self.perception = Perception(vision_instance=self.vision)
        
        # Create visual camera representations
        self.create_camera_visual(left_config, [1, 0, 0, 1])  # Red for left camera
        self.create_camera_visual(right_config, [0, 0, 1, 1])  # Blue for right camera
        
        # Add baseline visualization
        pb.addUserDebugLine(
            lineFromXYZ=left_config["translation"],
            lineToXYZ=right_config["translation"],
            lineColorRGB=[0, 1, 0],  # Green line connecting cameras
            lineWidth=1
        )

        return left_config, right_config

    def create_camera_visual(self, config, color):
        """Create visual representation of a camera."""
        size = 0.05
        vis_shape = pb.createVisualShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[size/2, size, size/2],
            rgbaColor=color
        )
        
        col_shape = pb.createCollisionShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[size/2, size, size/2]
        )
        
        orientation = pb.getQuaternionFromEuler([np.radians(r) for r in config["rotation"]])
        
        camera_id = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=config["translation"],
            baseOrientation=orientation
        )
        
        # Add viewing direction line
        direction = Rotation.from_euler('xyz', [np.radians(r) for r in config["rotation"]]).apply([1, 0, 0])
        end_point = np.array(config["translation"]) + direction * 0.2
        
        pb.addUserDebugLine(
            lineFromXYZ=config["translation"],
            lineToXYZ=end_point,
            lineColorRGB=color[:3],
            lineWidth=2
        )
        
        return camera_id

    def add_objects(self, num_objects=8):
        """Add random objects to the scene in a grid pattern."""
        min_pos, max_pos = (-0.5, -0.5, 0), (0.5, 0.5, 0.3)
        object_types = ["duck_vhacd.urdf", "cube_small.urdf", "sphere_small.urdf"]
        scales = [2.0, 0.5, 0.5]  # Scale factors for each object type
        
        x_positions = np.linspace(min_pos[0], max_pos[0], 3)
        y_positions = np.linspace(min_pos[1], max_pos[1], 3)
        
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                if len(self.object_ids) >= num_objects:
                    break
                    
                obj_idx = random.randint(0, len(object_types)-1)
                obj_type = object_types[obj_idx]
                scale = scales[obj_idx]
                
                z = random.uniform(min_pos[2], max_pos[2])
                pos = [x, y, z]
                
                # Random orientation
                orient = pb.getQuaternionFromEuler([
                    random.uniform(0, np.pi/4),  # Limited rotation for better visibility
                    random.uniform(0, np.pi/4),
                    random.uniform(0, 2*np.pi)
                ])
                
                obj_id = pb.loadURDF(obj_type, 
                                   basePosition=pos, 
                                   baseOrientation=orient,
                                   globalScaling=scale,
                                   useFixedBase=True)
                
                # Random color for visibility
                color = [random.random(), random.random(), random.random(), 1]
                pb.changeVisualShape(obj_id, -1, rgbaColor=color)
                self.object_ids.append(obj_id)

    def get_object_states(self):
        """Get ground truth states of all objects."""
        states = {}
        for obj_id in self.object_ids:
            pos, orn = pb.getBasePositionAndOrientation(obj_id)
            rot = Rotation.from_quat(orn)
            euler = rot.as_euler('xyz', degrees=True)
            states[obj_id] = {
                'position': np.array(pos),
                'orientation': np.array(euler),
                'type': pb.getBodyInfo(obj_id)[1].decode('utf-8')
            }
        return states

    def process_stereo(self):
        """Process stereo vision and generate point cloud."""
        # Compute rectification maps for stereo
        self.vision.compute_stereo_rectification_maps(image_size=(640, 480))
        
        # Capture stereo images
        logger.info("\nCapturing stereo images...")
        left_rgb, left_depth = self.vision.capture_image(camera_index=0)
        right_rgb, right_depth = self.vision.capture_image(camera_index=1)
        
        # Compute disparity and point cloud
        logger.info("Computing disparity...")
        disparity = self.perception.compute_stereo_disparity(left_rgb, right_rgb)
        
        logger.info("Generating point cloud...")
        point_cloud = self.perception.get_stereo_point_cloud(left_rgb, right_rgb)
        
        return left_rgb, right_rgb, disparity, point_cloud

    def run(self):
        """Run the perception pipeline."""
        try:
            # Setup scene
            logger.info("Setting up scene...")
            self.setup_cameras()
            self.add_objects()
            
            # Let physics settle
            for _ in range(100):
                pb.stepSimulation()
            time.sleep(1)
            
            # Get ground truth
            ground_truth = self.get_object_states()
            logger.info("\n=== Ground Truth Object Positions ===")
            for obj_id, state in ground_truth.items():
                logger.info(f"\nObject {obj_id} ({state['type']})")
                logger.info(f"Position: [{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}]")
                logger.info(f"Orientation: [{state['orientation'][0]:.1f}°, {state['orientation'][1]:.1f}°, {state['orientation'][2]:.1f}°]")
            
            # Process stereo
            left_rgb, right_rgb, disparity, point_cloud = self.process_stereo()
            
            if len(point_cloud) > 0:
                # Filter point cloud
                mask = (point_cloud[:, 2] > 0.01) & (np.linalg.norm(point_cloud, axis=1) < 5.0)
                filtered_cloud = point_cloud[mask]
                
                # Cluster objects
                logger.info("Clustering objects...")
                labels, num_clusters = self.perception.cluster_obstacles(
                    filtered_cloud, 
                    eps=0.1,
                    min_samples=5
                )
                
                logger.info(f"\n=== Perception Results ===")
                logger.info(f"Detected {num_clusters} distinct objects")
                logger.info(f"Point cloud size: {len(filtered_cloud)} points")
                
                # Visualize results
                self.visualize_results(left_rgb, right_rgb, disparity, filtered_cloud, labels, num_clusters)
            else:
                logger.warning("No points generated in point cloud!")
            
            # Keep simulation running
            logger.info("\nPress Ctrl+C to exit...")
            while True:
                pb.stepSimulation()
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("\nExiting...")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.perception:
                self.perception.release()
            pb.disconnect()

    def visualize_results(self, left_rgb, right_rgb, disparity, point_cloud, labels, num_clusters):
        """Visualize stereo and clustering results."""
        # Stereo results
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(left_rgb)
        plt.title('Left Camera')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(right_rgb)
        plt.title('Right Camera')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(disparity, cmap='jet')
        plt.title('Disparity Map')
        plt.colorbar(label='Disparity')
        plt.axis('off')
        plt.show()
        
        # Clustered point cloud
        colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot clusters
        for i in range(num_clusters):
            cluster_points = point_cloud[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                ax.scatter(cluster_points[:, 0], 
                          cluster_points[:, 1], 
                          cluster_points[:, 2], 
                          c=[colors[i]], 
                          label=f'Object {i}')
                ax.text(centroid[0], centroid[1], centroid[2], 
                       f'Obj {i}\n({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})',
                       color='black')
        
        # Plot noise points if any
        noise_points = point_cloud[labels == -1]
        if len(noise_points) > 0:
            ax.scatter(noise_points[:, 0], 
                      noise_points[:, 1], 
                      noise_points[:, 2], 
                      c='black', 
                      label='Noise',
                      alpha=0.1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Detected Objects')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    scene = ScenePerception()
    scene.run()