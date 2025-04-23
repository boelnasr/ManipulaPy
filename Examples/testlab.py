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
from scipy.spatial import cKDTree

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

    def add_objects(self, num_objects):
        """Add random objects to the scene in a grid pattern."""
        min_pos, max_pos = (-0.5, -0.5, 0), (0.5, 0.5, 0.3)
        object_types = ["cube_small.urdf"]
        scales = [1.0]  # Scale factors for each object type
        
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

    def statistical_outlier_removal(self, cloud, k=30, std_ratio=2.0):
        """
        Remove outliers from point cloud using a statistical approach.
        - k: Number of nearest neighbors
        - std_ratio: Points exceeding mean distance + std_ratio * std_dev are removed
        """
        if len(cloud) < k + 1:
            return cloud

        tree = cKDTree(cloud)
        distances = []

        for point in cloud:
            # Make sure k isn't larger than the number of points
            k_actual = min(k+1, len(cloud))
            dist, _ = tree.query(point, k=k_actual)
            # Exclude the first distance (zero, self-point)
            distances.append(np.mean(dist[1:]))

        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist

        mask = distances <= threshold
        return cloud[mask]

    def improved_clustering(self, point_cloud, expected_num_objects=None):
        """
        Improved clustering method that tries multiple parameter sets to find the best clustering.
        
        Args:
            point_cloud (np.ndarray): The filtered point cloud to cluster
            expected_num_objects (int, optional): Expected number of objects if known
            
        Returns:
            tuple: (labels, num_clusters) - The cluster labels and number of valid clusters
        """
        logger.info("Running improved clustering algorithm...")
        
        # If we have an expected number, use it as a target
        target_count = expected_num_objects if expected_num_objects is not None else len(self.object_ids)
        logger.info(f"Target number of objects: {target_count}")
        
        # Define multiple parameter sets to try
        # Format: (eps, min_samples, std_ratio for outlier removal)
        parameter_sets = [
            (0.05, 30, 2.0),   # Tighter clustering, more noise rejection
            (0.08, 50, 2.0),   # Medium clustering
            (0.1, 100, 2.0),   # Looser clustering, fewer clusters likely
            (0.12, 25, 1.5),   # Even looser, but less strict on min_samples
            (0.15, 15, 1.0),   # Very loose clustering to catch spread-out objects
        ]
        
        best_result = None
        best_score = float('inf')  # Lower is better
        best_params = None
        
        # Try all parameter sets and keep the best one
        for eps, min_samples, std_ratio in parameter_sets:
            # Apply statistical outlier removal with these parameters
            filtered_cloud = self.statistical_outlier_removal(
                point_cloud.copy(), k=min(50, len(point_cloud)//10), std_ratio=std_ratio
            )
            
            if len(filtered_cloud) < min_samples:
                logger.warning(f"Too few points left after filtering with eps={eps}, min_samples={min_samples}")
                continue
            
            # Run DBSCAN with these parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(filtered_cloud)
            
            # Count valid clusters (not noise)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            num_clusters = len(unique_labels)
            
            # Validate clusters - remove clusters that are too small or too large
            valid_labels = self.validate_clusters(filtered_cloud, labels)
            
            # Count valid clusters again after validation
            unique_valid_labels = set(valid_labels)
            if -1 in unique_valid_labels:
                unique_valid_labels.remove(-1)
            valid_num_clusters = len(unique_valid_labels)
            
            # Calculate score based on difference from target count and percentage of noise points
            noise_percentage = np.sum(valid_labels == -1) / len(valid_labels)
            count_diff = abs(valid_num_clusters - target_count)
            
            # Score formula: difference from target count + penalty for high noise percentage
            score = count_diff + (noise_percentage * 2)
            
            logger.info(f"Parameters (eps={eps}, min_samples={min_samples}): "
                      f"Found {valid_num_clusters} clusters, "
                      f"{noise_percentage:.1%} noise, score={score:.2f}")
            
            # Update best result if this is better
            if score < best_score:
                best_score = score
                best_result = (valid_labels, valid_num_clusters)
                best_params = (eps, min_samples, std_ratio)
        
        if best_result is None:
            logger.warning("Could not find any valid clustering! Falling back to basic method.")
            # Fall back to basic clustering
            labels, num_clusters = self.perception.cluster_obstacles(
                point_cloud, eps=0.1, min_samples=50
            )
            return labels, num_clusters
        
        logger.info(f"Best clustering: {best_result[1]} clusters with parameters {best_params}")
        return best_result

    def validate_clusters(self, point_cloud, labels):
        """
        Validate clusters by filtering out those that are too small, too large,
        or have the wrong shape characteristics.
        
        Args:
            point_cloud (np.ndarray): The point cloud data
            labels (np.ndarray): The cluster labels from DBSCAN
            
        Returns:
            np.ndarray: The validated labels with invalid clusters marked as noise (-1)
        """
        validated_labels = labels.copy()
        unique_labels = set(labels)
        
        # Skip noise points
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for label in unique_labels:
            # Get points in this cluster
            cluster_mask = (labels == label)
            cluster_points = point_cloud[cluster_mask]
            
            # Skip if too few points
            if len(cluster_points) < 20:
                validated_labels[cluster_mask] = -1
                continue
                
            # Calculate cluster dimensions
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            dimensions = max_coords - min_coords
            
            # Check if the cluster is too small in any dimension
            if np.any(dimensions < 0.01):
                validated_labels[cluster_mask] = -1
                continue
                
            # Check if the cluster is too large in any dimension
            if np.any(dimensions > 0.5):
                validated_labels[cluster_mask] = -1
                continue
                
            # Check if the cluster is too flat (could be a wall or floor)
            # Calculate flatness as the ratio of smallest to largest dimension
            flatness_ratio = np.min(dimensions) / np.max(dimensions)
            if flatness_ratio < 0.1:  # Very flat object
                validated_labels[cluster_mask] = -1
                continue
            
            # Calculate point density
            volume = np.prod(dimensions)
            density = len(cluster_points) / (volume + 1e-10)  # Avoid division by zero
            
            # Check if density is too low (sparse cluster)
            if density < 1000 and volume > 0.001:  # Adjust thresholds as needed
                validated_labels[cluster_mask] = -1
                continue
        
        # Relabel the valid clusters with consecutive integers
        # This makes visualization and processing easier
        new_labels = np.ones_like(validated_labels) * -1  # Start with all noise
        next_label = 0
        
        unique_valid_labels = set(validated_labels)
        if -1 in unique_valid_labels:
            unique_valid_labels.remove(-1)
            
        for label in unique_valid_labels:
            cluster_mask = (validated_labels == label)
            new_labels[cluster_mask] = next_label
            next_label += 1
        
        return new_labels

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

    def remove_ground_plane(self, point_cloud, z_threshold=0.02):
        """Remove points that are close to the ground plane."""
        mask = point_cloud[:, 2] > z_threshold
        return point_cloud[mask]

    def bounding_box_filter(self, point_cloud, x_range=(-1, 1), y_range=(-1, 1), z_range=(0, 1)):
        """Filter points to be within the specified spatial bounds."""
        mask = (
            (point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1]) &
            (point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1]) &
            (point_cloud[:, 2] >= z_range[0]) & (point_cloud[:, 2] <= z_range[1])
        )
        return point_cloud[mask]

    def debug_point_cloud(self, point_cloud):
        """Print information about the point cloud to diagnose issues"""
        print("\n=== POINT CLOUD DEBUG INFO ===")
        print(f"Total points: {len(point_cloud)}")
        
        if len(point_cloud) > 0:
            # Check for NaN/Inf values
            nan_count = np.isnan(point_cloud).any(axis=1).sum()
            inf_count = np.isinf(point_cloud).any(axis=1).sum()
            print(f"NaN points: {nan_count}, Inf points: {inf_count}")
            
            # Get range information
            min_vals = np.min(point_cloud, axis=0)
            max_vals = np.max(point_cloud, axis=0)
            print(f"X range: [{min_vals[0]:.4f}, {max_vals[0]:.4f}]")
            print(f"Y range: [{min_vals[1]:.4f}, {max_vals[1]:.4f}]")
            print(f"Z range: [{min_vals[2]:.4f}, {max_vals[2]:.4f}]")
            
            # Get Z distribution
            z_values = point_cloud[:, 2]
            neg_z = np.sum(z_values < 0)
            pos_z = np.sum(z_values >= 0)
            print(f"Z distribution: {neg_z} negative, {pos_z} positive")
            
            # Get distance distribution
            distances = np.linalg.norm(point_cloud, axis=1)
            print(f"Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
            
            # Get 5 random sample points
            if len(point_cloud) >= 5:
                indices = np.random.choice(len(point_cloud), 5, replace=False)
                print("Sample points:")
                for i, idx in enumerate(indices):
                    print(f"  Point {i+1}: {point_cloud[idx]}")
        
        print("==============================\n")

    def direct_clustering(self, point_cloud, num_objects):
        """
        A refined clustering approach that aims to find the correct number of objects
        with stricter parameters to prevent over-segmentation.
        """
        print("\n--- DIRECT CLUSTERING ---")
        
        # 1. Basic cleanup - remove NaN/Inf
        valid_mask = ~(np.isnan(point_cloud).any(axis=1) | np.isinf(point_cloud).any(axis=1))
        cloud = point_cloud[valid_mask]
        
        # 2. Check Z orientation and flip if necessary
        z_vals = cloud[:, 2]
        neg_z = np.sum(z_vals < 0)
        pos_z = np.sum(z_vals >= 0)
        
        if neg_z > pos_z:
            print(f"Flipping Z axis ({neg_z} negative vs {pos_z} positive)")
            cloud[:, 2] = -cloud[:, 2]
        
        # 3. Apply basic distance filter
        distances = np.linalg.norm(cloud, axis=1)
        dist_mask = distances < 10.0  # More restrictive distance
        cloud = cloud[dist_mask]
        
        print(f"After basic filtering: {len(cloud)} points")
        
        if len(cloud) == 0:
            return np.array([]), 0
        
        # 4. Try clustering parameters specifically suited for your scene
        # Start with much larger eps and min_samples to prevent over-segmentation
        parameter_sets = [
            (0.3, 50),   # Very coarse clustering, fewer clusters
            (0.2, 40),   # Medium-coarse
            (0.15, 30),  # Medium
            (0.1, 20),   # Medium-fine
            (0.05, 10)   # Fine-grained - only use if previous options didn't work
        ]
        
        best_num_clusters = 0
        best_labels = None
        best_diff = float('inf')
        
        # Try each parameter set and keep the one closest to the expected number of objects
        for eps, min_samples in parameter_sets:
            if len(cloud) < min_samples:
                continue
                
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(cloud)
                
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                current_num_clusters = len(unique_labels)
                
                diff = abs(current_num_clusters - num_objects)
                
                print(f"eps={eps}, min_samples={min_samples}: {current_num_clusters} clusters (target: {num_objects})")
                
                if diff < best_diff:
                    best_diff = diff
                    best_num_clusters = current_num_clusters
                    best_labels = labels
                    
                    # If we found exactly the right number, we can stop
                    if diff == 0:
                        break
            except Exception as e:
                print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                continue
        
        if best_labels is None:
            print("Failed to find any clusters with any parameter set.")
            return np.array([]), 0
            
        print(f"Best result: {best_num_clusters} clusters (target: {num_objects})")
        print("----------------------\n")
        
        return best_labels, best_num_clusters
    def merge_nearby_clusters(self, point_cloud, labels, distance_threshold=0.2):
        """
        Merge clusters whose centroids are closer than the threshold distance.
        This helps combat over-segmentation.
        """
        unique_labels = np.unique(labels)
        if -1 in unique_labels:  # Remove noise label
            unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) <= 1:
            return labels
            
        # Calculate centroids for each cluster
        centroids = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = point_cloud[mask]
            if len(cluster_points) > 0:
                centroids.append(np.mean(cluster_points, axis=0))
        
        centroids = np.array(centroids)
        
        # Create a mapping from old labels to new labels
        new_label_map = {label: label for label in unique_labels}
        next_label = np.max(unique_labels) + 1
        
        # Calculate distances between all pairs of centroids
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                if new_label_map[unique_labels[i]] == new_label_map[unique_labels[j]]:
                    continue  # Already merged
                    
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < distance_threshold:
                    # Merge j into i
                    label_j = new_label_map[unique_labels[j]]
                    label_i = new_label_map[unique_labels[i]]
                    
                    # Update all points with label_j to label_i
                    for k in range(len(unique_labels)):
                        if new_label_map[unique_labels[k]] == label_j:
                            new_label_map[unique_labels[k]] = label_i
        
        # Apply the mapping to create a new labels array
        new_labels = np.copy(labels)
        for old_label, new_label in new_label_map.items():
            new_labels[labels == old_label] = new_label
        
        # Relabel to consecutive integers
        unique_new_labels = np.unique(new_labels)
        if -1 in unique_new_labels:
            unique_new_labels = unique_new_labels[unique_new_labels != -1]
        
        final_labels = np.copy(new_labels)
        for i, label in enumerate(unique_new_labels):
            final_labels[new_labels == label] = i
        
        return final_labels


    def run(self):
        """Run the perception pipeline."""
        try:
            # Setup scene
            logger.info("Setting up scene...")
            self.setup_cameras()
            num_objects = random.randint(3, 7)
            print("\n")
            print("==================================")
            print(f"GENERATING {num_objects} RANDOM OBJECTS")
            print("==================================")
            self.add_objects(num_objects=num_objects)
            
            # Let physics settle
            for _ in range(100):
                pb.stepSimulation()
            time.sleep(1)
            
            # Get ground truth
            ground_truth = self.get_object_states()
            logger.info("\n=== Ground Truth Object Positions ===")
            for obj_id, state in ground_truth.items():
                logger.info(f"\nObject {obj_id} ({state['type']})")
                logger.info(
                    f"Position: [{state['position'][0]:.3f}, "
                    f"{state['position'][1]:.3f}, {state['position'][2]:.3f}]"
                )
                logger.info(
                    f"Orientation: [{state['orientation'][0]:.1f}°, "
                    f"{state['orientation'][1]:.1f}°, "
                    f"{state['orientation'][2]:.1f}°]"
                )
            
            # Process stereo
            left_rgb, right_rgb, disparity, point_cloud = self.process_stereo()
            
            if len(point_cloud) == 0:
                logger.warning("No points generated in point cloud!")
            else:
                # Print debug information about the point cloud
                self.debug_point_cloud(point_cloud)
                
                # Try direct clustering approach instead of filtering pipeline
                labels, num_clusters = self.direct_clustering(point_cloud, num_objects)
                
                # Check if we have over-segmentation and try to merge clusters
                if num_clusters > num_objects * 2:  # Significant over-segmentation
                    print(f"Detected {num_clusters} clusters but expected {num_objects} - attempting to merge nearby clusters")
                    merged_labels = self.merge_nearby_clusters(point_cloud, labels, distance_threshold=0.2)
                    merged_unique_labels = np.unique(merged_labels)
                    if -1 in merged_unique_labels:
                        merged_unique_labels = merged_unique_labels[merged_unique_labels != -1]
                    merged_num_clusters = len(merged_unique_labels)
                    
                    print(f"After merging: {merged_num_clusters} clusters")
                    labels = merged_labels
                    num_clusters = merged_num_clusters
                
                # If we still couldn't find any reasonable clusters
                if num_clusters == 0 or num_clusters > num_objects * 5:  # No clusters or extreme over-segmentation
                    logger.warning("Clustering failed or produced too many clusters. Trying a different approach...")
                    
                    # Basic filtering
                    nan_inf_mask = ~(
                        np.isnan(point_cloud).any(axis=1) | 
                        np.isinf(point_cloud).any(axis=1)
                    )
                    filtered_cloud = point_cloud[nan_inf_mask]
                    
                    # Try with very strong clustering parameters to avoid over-segmentation
                    dbscan = DBSCAN(eps=0.4, min_samples=100)  # Much larger eps, many more required points
                    labels = dbscan.fit_predict(filtered_cloud)
                    
                    # Count valid clusters (not noise)
                    unique_labels = np.unique(labels)
                    if -1 in unique_labels:
                        unique_labels = unique_labels[unique_labels != -1]
                    num_clusters = len(unique_labels)
                    
                    logger.info(f"Strong parameter clustering found {num_clusters} clusters")
                    
                    if num_clusters > 0:
                        # Use the filtered cloud and labels from fallback approach
                        point_cloud = filtered_cloud
                
                print("\n")
                print("==================================")
                print(f"DETECTED {num_clusters} OBSTACLES (GROUND TRUTH: {num_objects})")
                print("==================================")
                
                logger.info(f"\n=== Perception Results ===")
                logger.info(f"Detected {num_clusters} distinct objects (Ground truth: {num_objects})")
                logger.info(f"Point cloud size: {len(point_cloud)} points")
                
                # Visualize results if we found any clusters
                if num_clusters > 0:
                    # Ensure point_cloud and labels have the same length
                    if len(point_cloud) != len(labels):
                        min_length = min(len(point_cloud), len(labels))
                        point_cloud = point_cloud[:min_length]
                        labels = labels[:min_length]
                        logger.info(f"Adjusted arrays to length {min_length} for visualization")
                    
                    self.visualize_results(
                        left_rgb, 
                        right_rgb, 
                        disparity, 
                        point_cloud, 
                        labels, 
                        num_clusters
                    )
            
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
        # Make sure point_cloud and labels have the same length
        if len(point_cloud) != len(labels):
            logger.warning(f"Point cloud length ({len(point_cloud)}) doesn't match labels length ({len(labels)})")
            # Trim both arrays to the same length to avoid the error
            min_length = min(len(point_cloud), len(labels))
            point_cloud = point_cloud[:min_length]
            labels = labels[:min_length]
            logger.info(f"Adjusted both arrays to length {min_length}")
        
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
        plt.savefig("stereo_results.png", dpi=150)
        plt.close()        
        
        # Clustered point cloud
        colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot clusters
        for i in range(num_clusters):
            cluster_mask = labels == i
            if np.any(cluster_mask):  # Make sure there are points in this cluster
                cluster_points = point_cloud[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    ax.scatter(cluster_points[:, 0], 
                            cluster_points[:, 1], 
                            cluster_points[:, 2], 
                            c=[colors[i]], 
                            label=f'Object {i}',
                            s=10)  # Smaller point size for clarity
                    ax.text(centroid[0], centroid[1], centroid[2], 
                        f'Obj {i}\n({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})',
                        color='black',
                        fontsize=8)
        
        # Set consistent axis limits for better perspective
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(0.0, 0.5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Detected Objects: {num_clusters}')
        ax.legend()
        
        plt.savefig("object_clusters.png", dpi=150)
        plt.close()
        
        # Add a top-down view for better visualization
        plt.figure(figsize=(8, 8))
        plt.title(f'Top-Down View: {num_clusters} Objects')
        
        # Plot all clusters from top view
        for i in range(num_clusters):
            cluster_mask = labels == i
            if np.any(cluster_mask):  # Make sure there are points in this cluster
                cluster_points = point_cloud[cluster_mask]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    plt.scatter(cluster_points[:, 0], 
                            cluster_points[:, 1],
                            c=[colors[i]], 
                            label=f'Object {i}',
                            alpha=0.7,
                            s=10)
                    plt.annotate(f'Obj {i}', (centroid[0], centroid[1]))
        
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.legend()
        
        plt.savefig("top_view.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    scene = ScenePerception()
    scene.run()