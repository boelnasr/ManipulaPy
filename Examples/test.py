#!/usr/bin/env python3

import logging
import os
import time
import random
import numpy as np
import pybullet as pb
import pybullet_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

# ManipulaPy modules
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScenePerceptionNode")


class ScenePerception:
    def __init__(self):
        """
        Constructor for the ScenePerception object.
        Replaces the ROS node setup/teardown logic.
        """
        logger.info("ScenePerception: starting up...")

        # PyBullet connection (GUI mode for visualization)
        self.physics_client = None

        self.vision = None
        self.perception = None
        self.camera_ids = []
        self.spawned_objects = []

        # Initialize the environment and run main scene logic
        self._init_pybullet_gui()
        self._setup_cameras()

        # Spawn random objects
        self.spawned_objects = self._spawn_random_objects(num_objects=random.randint(4,7))

        # Let objects settle (or just a short pause, but they are fixed anyway)
        for _ in range(100):
            pb.stepSimulation()
        time.sleep(1.0)

        # Log ground truth (debug)
        self._log_ground_truth()

        # Process stereo vision
        self._process_stereo_vision()

    # ---------------------------
    # Initialization: Connect to PyBullet GUI
    # ---------------------------
    def _init_pybullet_gui(self):
        logger.info("Connecting to PyBullet (GUI mode)...")
        self.physics_client = pb.connect(pb.GUI)
        if self.physics_client >= 0:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
            try:
                pb.loadURDF(plane_path, useFixedBase=True)
            except Exception as e:
                logger.error(f"Failed to load plane URDF '{plane_path}': {e}")
            pb.setGravity(0, 0, -9.81)
            logger.info("PyBullet (GUI) connected. Ground plane loaded, gravity set.")
        else:
            logger.error("Failed to connect to PyBullet in GUI mode.")

    # ---------------------------
    # Camera Setup
    # ---------------------------
    def _setup_cameras(self):
        cam_height = 1.0
        cam_distance = 2.0
        baseline = 0.2

        common_params = {
            "fov": 120,
            "near": 0.1,
            "far": 10.0,
            "intrinsic_matrix": np.array([
                [1000,    0, 320],
                [   0, 1000, 240],
                [   0,    0,   1]
            ], dtype=np.float32),
            "distortion_coeffs": np.zeros(5, dtype=np.float32),
            "use_opencv": False
        }

        left_cam = common_params.copy()
        left_cam.update({
            "name": "left_camera",
            "translation": [cam_distance, -baseline/2, cam_height],
            "rotation": [-25, -5, 0],  # in degrees
        })

        right_cam = common_params.copy()
        right_cam.update({
            "name": "right_camera",
            "translation": [cam_distance, baseline/2, cam_height],
            "rotation": [-25, -5, 0],
        })

        try:
            self.vision = Vision(
                camera_configs=[left_cam, right_cam],
                stereo_configs=(left_cam, right_cam),
                use_pybullet_debug=False,
                show_plot=False,
                physics_client=self.physics_client
            )
            self.perception = Perception(vision_instance=self.vision)

            # Add visual markers for the cameras
            self._add_camera_visual(left_cam, [1, 0, 0, 1])   # Red marker
            self._add_camera_visual(right_cam, [0, 0, 1, 1])  # Blue marker

            # Draw a baseline line between the cameras
            pb.addUserDebugLine(
                lineFromXYZ=left_cam["translation"],
                lineToXYZ=right_cam["translation"],
                lineColorRGB=[0, 1, 0],
                lineWidth=1
            )

            # Compute stereo rectification maps
            self.vision.compute_stereo_rectification_maps(image_size=(640, 480))
            logger.info("Vision & Perception are set up.")
        except Exception as e:
            logger.error(f"Error setting up cameras: {e}")

    def _add_camera_visual(self, cam_cfg, color):
        size = 0.05
        # Convert rotation (degrees) to quaternion
        q = pb.getQuaternionFromEuler([np.radians(r) for r in cam_cfg["rotation"]])
        vis_shape = pb.createVisualShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[size/2, size, size/2],
            rgbaColor=color
        )
        col_shape = pb.createCollisionShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=[size/2, size, size/2]
        )
        cam_id = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=cam_cfg["translation"],
            baseOrientation=q
        )
        # Compute forward direction vector
        direction = R.from_euler('xyz', cam_cfg["rotation"], degrees=True).apply([1, 0, 0])
        end_pt = np.array(cam_cfg["translation"]) + direction * 0.2
        pb.addUserDebugLine(
            lineFromXYZ=cam_cfg["translation"],
            lineToXYZ=end_pt.tolist(),
            lineColorRGB=color[:3],
            lineWidth=2
        )
        self.camera_ids.append(cam_id)

    # ---------------------------
    # Object Spawning (Directly in PyBullet)
    # ---------------------------
    def _spawn_random_objects(self, num_objects=8):
        """
        Spawns objects in a structured grid pattern with useFixedBase=True.
        """
        logger.info(f"Spawning {num_objects} structured objects in PyBullet.")

        # Define object types and scaling factors
        object_types = ["cube_small.urdf", "sphere_small.urdf"]
        scales = [2.0, 1, 1]  # Scale factors

        # Define placement ranges
        x_range = (-0.5, 0.5)
        y_range = (-0.5, 0.5)
        z_range = (0.1, 0.4)

        grid_size = int(np.ceil(np.sqrt(num_objects)))
        x_positions = np.linspace(x_range[0], x_range[1], grid_size)
        y_positions = np.linspace(y_range[0], y_range[1], grid_size)
        
        spawned = []

        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                if len(spawned) >= num_objects:
                    break

                obj_idx = random.randint(0, len(object_types) - 1)
                obj_type = object_types[obj_idx]
                scale = scales[obj_idx]

                z = random.uniform(*z_range)
                pos = [
                    x + random.uniform(-0.1, 0.1),
                    y + random.uniform(-0.1, 0.1),
                    z
                ]

                orient = pb.getQuaternionFromEuler([
                    random.uniform(-np.pi / 8, np.pi / 8),
                    random.uniform(-np.pi / 8, np.pi / 8),
                    random.uniform(0, 2 * np.pi)
                ])

                obj_id = pb.loadURDF(
                    obj_type,
                    basePosition=pos,
                    baseOrientation=orient,
                    globalScaling=scale,
                    useFixedBase=True
                )

                # Apply random color
                color = [random.random(), random.random(), random.random(), 1]
                pb.changeVisualShape(obj_id, -1, rgbaColor=color)
                spawned.append(obj_id)

                logger.info(
                    f"✅ Spawned object {len(spawned)}: {obj_type} at "
                    f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), ID={obj_id}"
                )

        for _ in range(20):
            pb.stepSimulation()
        time.sleep(0.5)

        return spawned

    def _evaluate_position_accuracy(self, obstacles):
        """
        Evaluate the accuracy of detected obstacle positions by comparing with ground truth.
        
        Parameters:
            obstacles (list): List of detected obstacles with their positions
            
        Returns:
            dict: Statistics about position errors
        """
        if not self.spawned_objects:
            logger.warning("⚠️ No spawned objects available for evaluation")
            return {}
        
        # Get ground truth positions from PyBullet
        ground_truth = []
        for obj_id in self.spawned_objects:
            pos, _ = pb.getBasePositionAndOrientation(obj_id)
            ground_truth.append({
                "id": obj_id,
                "position": pos
            })
        
        logger.info("\n=== Position Estimation Accuracy ===")
        logger.info(f"Ground truth objects: {len(ground_truth)}")
        logger.info(f"Detected obstacles: {len(obstacles)}")
        
        # For each ground truth object, find the closest detected obstacle
        matching_errors = []
        matched_obstacles = set()
        
        for gt_obj in ground_truth:
            gt_pos = np.array(gt_obj["position"])
            min_dist = float('inf')
            closest_obs = None
            closest_idx = -1
            
            for i, obs in enumerate(obstacles):
                obs_pos = np.array(obs["position"])
                dist = np.linalg.norm(gt_pos - obs_pos)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_obs = obs
                    closest_idx = i
            
            if closest_obs:
                matched_obstacles.add(closest_idx)
                matching_errors.append(min_dist)
                
                # Output detailed comparison
                logger.info(
                    f"Ground truth ID {gt_obj['id']}: pos={gt_pos} -> "
                    f"Matched with {closest_obs['name']}: pos={closest_obs['position']} "
                    f"(error={min_dist:.4f}m)"
                )
        
        # Report unmatched obstacles (potentially false positives)
        for i, obs in enumerate(obstacles):
            if i not in matched_obstacles:
                logger.warning(f"⚠️ Unmatched obstacle: {obs['name']} at position {obs['position']}")
        
        # Calculate error statistics
        if matching_errors:
            avg_error = np.mean(matching_errors)
            max_error = np.max(matching_errors)
            min_error = np.min(matching_errors)
            
            logger.info(f"Position estimation metrics:")
            logger.info(f"Average error: {avg_error:.4f}m")
            logger.info(f"Maximum error: {max_error:.4f}m")
            logger.info(f"Minimum error: {min_error:.4f}m")
            
            return {
                "average_error": avg_error,
                "max_error": max_error,
                "min_error": min_error,
                "matched_objects": len(matching_errors),
                "unmatched_obstacles": len(obstacles) - len(matched_obstacles)
            }
        else:
            logger.warning("⚠️ No matches found between ground truth and detected obstacles")
            return {
                "average_error": float('inf'),
                "matched_objects": 0,
                "unmatched_obstacles": len(obstacles)
            }
    # ---------------------------
    # Ground Truth Logging
    # ---------------------------
    def _log_ground_truth(self):
        nb = pb.getNumBodies()
        logger.info(f"\n=== Ground Truth: {nb} total bodies in the environment ===")
        for i in range(nb):
            info = pb.getBodyInfo(i)
            name = info[1].decode('utf-8')
            pos, orn = pb.getBasePositionAndOrientation(i)
            eul = R.from_quat(orn).as_euler('xyz', degrees=True)
            logger.info(
                f" Body[{i}]: {name}, pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                f"euler=({eul[0]:.1f}, {eul[1]:.1f}, {eul[2]:.1f})"
            )

    # ---------------------------
    # Stereo Processing & Obstacle Detection
    # ---------------------------

    def _merge_nearby_clusters(self, cloud, labels, max_merge_distance=0.25):
        """
        Merge clusters that are close to each other based on centroid distance.

        Parameters:
            cloud (np.ndarray): Point cloud data (Nx3).
            labels (np.ndarray): Cluster labels for each point.
            max_merge_distance (float): Max distance between cluster centroids to merge.

        Returns:
            np.ndarray: Updated labels after merging.
        """
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels != -1]  # exclude noise

        if len(valid_labels) <= 1:
            # If there are 0 or 1 valid clusters, no merge is needed
            return labels

        # 1) Compute centroids for each cluster
        centroids = {}
        for label in valid_labels:
            cluster_points = cloud[labels == label]
            centroids[label] = np.mean(cluster_points, axis=0)

        # 2) Initialize mapping so each label maps to itself
        merge_map = {label: label for label in valid_labels}

        # 3) Find clusters to merge based on centroid distance
        for i in range(len(valid_labels)):
            label_i = valid_labels[i]
            if label_i not in merge_map:
                continue

            for j in range(i + 1, len(valid_labels)):
                label_j = valid_labels[j]
                # Skip if already merged
                if label_j not in merge_map or merge_map[label_j] != label_j:
                    continue

                dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
                if dist < max_merge_distance:
                    logger.info(f"Merging clusters {label_j} into {label_i} (distance: {dist:.4f})")
                    merged_label = merge_map[label_i]
                    merge_map[label_j] = merged_label

        # 4) Apply the merge map to the label array
        new_labels = np.copy(labels)
        for old_label, new_label in merge_map.items():
            if old_label != new_label:
                new_labels[labels == old_label] = new_label

        # 5) Log result
        merged_unique_labels = np.unique(new_labels)
        merged_valid_labels = merged_unique_labels[merged_unique_labels != -1]
        num_merged_clusters = len(merged_valid_labels)
        logger.info(f"After merging: {num_merged_clusters} clusters (reduced from {len(valid_labels)})")

        return new_labels


    def _process_stereo_vision(self):
        """
        Capture stereo images, compute disparity, generate point cloud,
        filter points, run DBSCAN, merge nearby clusters, and log final obstacles.
        """
        logger.info("📷 Capturing stereo images from left & right cameras...")
        left_rgb, _ = self.vision.capture_image(camera_index=0)
        right_rgb, _ = self.vision.capture_image(camera_index=1)

        if left_rgb is None or right_rgb is None:
            logger.error("❌ Failed to capture images from cameras.")
            return

        # 1) Compute stereo disparity
        logger.info("🔍 Computing stereo disparity...")
        disparity = self.perception.compute_stereo_disparity(left_rgb, right_rgb)

        # 2) Generate the 3D point cloud from the stereo pair
        logger.info("🌍 Generating point cloud...")
        point_cloud = self.perception.get_stereo_point_cloud(left_rgb, right_rgb)
        if point_cloud is None or len(point_cloud) == 0:
            logger.warning("⚠️ No valid points in the point cloud. Skipping processing.")
            return

        logger.info(f"✅ Point cloud generated with {len(point_cloud)} points.")

        # 3) Flip negative Z-values if needed (some stereo configs produce inverted depth)
        min_z, max_z = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
        logger.info(f"🔎 Point cloud Z-range: min={min_z:.2f}, max={max_z:.2f}")
        if min_z < 0 and abs(min_z) > abs(max_z):
            logger.info("🔄 Detected negative Z-values; flipping Z-coordinates.")
            point_cloud[:, 2] = -point_cloud[:, 2]
            min_z, max_z = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
            logger.info(f"✅ After flipping, Z-range: min={min_z:.2f}, max={max_z:.2f}")

        # 4) Remove NaN/Inf points
        valid_mask = ~(
            np.isnan(point_cloud).any(axis=1) | np.isinf(point_cloud).any(axis=1)
        )
        valid_cloud = point_cloud[valid_mask]
        logger.info(f"✅ Valid points after NaN/Inf removal: {len(valid_cloud)}")

        # 5) Distance filtering (e.g., remove points > 50 meters away)
        max_distance = 50.0
        mask = np.linalg.norm(valid_cloud, axis=1) < max_distance
        filtered_cloud = valid_cloud[mask]
        logger.info(f"🔎 Filtered point cloud: {len(filtered_cloud)} points.")

        if len(filtered_cloud) == 0:
            logger.warning("⚠️ No valid points left after filtering. Skipping clustering.")
            return

        # 6) DBSCAN Clustering
        logger.info("🔍 Running DBSCAN clustering on filtered point cloud...")
        labels, num_clusters, processed_cloud = self._dbscan_clustering(
            filtered_cloud,
            min_samples=15
        )
        if num_clusters == 0:
            logger.warning("⚠️ No valid clusters found. Skipping obstacle logic.")
            return

        # 7) Post-Processing Merge Step
        logger.info("🔍 Performing post-processing cluster merging...")
        merged_labels = self._merge_nearby_clusters(processed_cloud, labels, max_merge_distance=0.08)

        # Re-count how many clusters remain after merging
        merged_unique_labels = np.unique(merged_labels)
        merged_valid_labels = merged_unique_labels[merged_unique_labels != -1]
        num_merged_clusters = len(merged_valid_labels)

        # 8) Convert final clusters to obstacles
        obstacles = self._make_obstacle_array(processed_cloud, merged_labels, num_merged_clusters)
        logger.info(f"✅ Found {num_merged_clusters} cluster(s) after merging. Final obstacle results:")
        for obs in obstacles:
            logger.info(f"   - {obs['name']}: Position = {obs['position']}")

        accuracy_metrics = self._evaluate_position_accuracy(obstacles)


        # 9) Visualization
        self._visualize_results(left_rgb, right_rgb, disparity, processed_cloud, merged_labels, num_merged_clusters)



    def _dbscan_clustering(
        self, cloud, min_samples=15,
        min_cluster_volume=0.005, max_cluster_volume=2.0,
        min_points_per_cluster=100, max_points_per_cluster=1000
    ):
        """
        Improved DBSCAN clustering with dynamic `eps`, volume constraints, and density filtering.
        Returns filtered labels, number of clusters, and the processed point cloud.
        """
        if len(cloud) == 0:
            logger.warning("⚠️ Empty point cloud! No clustering performed.")
            return np.array([]), 0, np.array([])
        
        # Store original cloud for reference
        original_cloud = cloud
        
        # Downsample the cloud if it's very large
        if len(cloud) > 10000:
            step = max(1, len(cloud) // 10000)
            cloud = cloud[::step]
            logger.info(f"🔄 Downsampled point cloud to {len(cloud)} points (step={step})")
        
        # Remove outliers before clustering
        cloud = self._statistical_outlier_removal(cloud, k=30, std_ratio=2)
        
        # Dynamically estimate `eps` using KNN
        try:
            neighbors = NearestNeighbors(n_neighbors=min_samples).fit(cloud)
            distances, _ = neighbors.kneighbors(cloud)
            avg_distance = np.mean(distances[:, -1])
            eps = avg_distance * 0.8  # Reduced multiplier for tighter clusters
        except Exception as e:
            logger.error(f"❌ KNN estimation failed: {e}")
            eps = 0.1

        logger.info(f"📏 Estimated DBSCAN `eps` = {eps:.3f}")

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(cloud)
        labels = db.labels_

        # Filter clusters based on size and volume
        filtered_labels = np.copy(labels)
        unique_labels = np.unique(labels)
        valid_clusters = []

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = cloud[labels == label]
            # Size constraints
            if len(cluster_points) < min_points_per_cluster or len(cluster_points) > max_points_per_cluster:
                filtered_labels[labels == label] = -1
                continue

            # Check dimensionality of the cluster
            cluster_points_np = np.array(cluster_points)
            x_range = np.max(cluster_points_np[:, 0]) - np.min(cluster_points_np[:, 0])
            y_range = np.max(cluster_points_np[:, 1]) - np.min(cluster_points_np[:, 1])
            z_range = np.max(cluster_points_np[:, 2]) - np.min(cluster_points_np[:, 2])
            
            # Check if the cluster is roughly flat (one dimension is much smaller than others)
            is_flat = False
            min_range = min(x_range, y_range, z_range)
            max_range = max(x_range, y_range, z_range)
            if min_range < 0.01 * max_range or min_range < 0.001:  # Threshold for "flatness"
                is_flat = True
                logger.info(f"Cluster {label} appears to be flat (ranges: x={x_range:.4f}, y={y_range:.4f}, z={z_range:.4f})")
            
            # Compute volume based on whether the cluster is flat or not
            try:
                if is_flat or len(cluster_points) < 4:
                    # For flat clusters, use the area times a small thickness
                    if z_range < min(x_range, y_range):
                        cluster_volume = x_range * y_range * 0.01  # Arbitrary small thickness
                    elif y_range < min(x_range, z_range):
                        cluster_volume = x_range * z_range * 0.01
                    else:
                        cluster_volume = y_range * z_range * 0.01
                else:
                    # For 3D clusters, try ConvexHull with exception handling
                    try:
                        hull = ConvexHull(cluster_points)
                        cluster_volume = hull.volume
                    except Exception as e:
                        # Fallback to bounding box volume
                        logger.warning(f"⚠️ ConvexHull failed for cluster {label}: {e}")
                        cluster_volume = x_range * y_range * z_range
            except Exception as e:
                logger.warning(f"⚠️ Volume calculation failed for cluster {label}: {e}")
                cluster_volume = 0.0  # Default to zero volume
                
            logger.info(f"Cluster {label}: {len(cluster_points)} points, volume = {cluster_volume:.6f}")

            if cluster_volume < min_cluster_volume or cluster_volume > max_cluster_volume:
                filtered_labels[labels == label] = -1
                continue

            valid_clusters.append(label)

        n_clusters = len(valid_clusters)
        noise_count = np.sum(filtered_labels == -1)

        logger.info(f"✅ DBSCAN => {n_clusters} valid clusters, {noise_count} noise points.")
        return filtered_labels, n_clusters, cloud  # Return the processed cloud
    def _statistical_outlier_removal(self, cloud, k=10, std_ratio=2.0):
        """
        Remove statistical outliers from point cloud.
        """
        if len(cloud) < k + 1:
            return cloud
            
        tree = cKDTree(cloud)
        distances = []
        for point in cloud:
            dist, _ = tree.query(point, k=k+1)
            distances.append(np.mean(dist[1:]))

        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist

        indices = np.where(distances <= threshold)[0]
        return cloud[indices]

    def _make_obstacle_array(self, cloud, labels, num_clusters):
        """
        Convert cluster data into a simple list of obstacles with mapped positions.
        """
        obstacles_list = []

        expected_x_range = (-0.5, 0.5)
        expected_y_range = (-0.5, 0.5)
        expected_z_range = (0.1, 0.4)

        percentile_min, percentile_max = 5, 95
        x_vals = cloud[:, 0]
        y_vals = cloud[:, 1]
        z_vals = cloud[:, 2]

        detected_x_range = (np.percentile(x_vals, percentile_min), np.percentile(x_vals, percentile_max))
        detected_y_range = (np.percentile(y_vals, percentile_min), np.percentile(y_vals, percentile_max))
        detected_z_range = (np.percentile(z_vals, percentile_min), np.percentile(z_vals, percentile_max))

        logger.info(f"Detected ranges - X: {detected_x_range}, Y: {detected_y_range}, Z: {detected_z_range}")

        valid_labels = np.unique(labels)
        valid_labels = valid_labels[valid_labels != -1]

        idx = 0
        for label in valid_labels:
            cluster_pts = cloud[np.where(labels == label)]
            if len(cluster_pts) == 0:
                continue

            idx += 1
            centroid_camera = np.mean(cluster_pts, axis=0)

            mapped_x = self._map_value(centroid_camera[0], detected_x_range, expected_x_range)
            mapped_y = self._map_value(centroid_camera[1], detected_y_range, expected_y_range)
            mapped_z = self._map_value(centroid_camera[2], detected_z_range, expected_z_range)

            obs = {
                "name": f"obstacle_{idx}",
                "position": (float(mapped_x), float(mapped_y), float(mapped_z))
            }
            obstacles_list.append(obs)

            logger.info(
                f"✅ Obstacle {idx}: World Position "
                f"({mapped_x:.2f}, {mapped_y:.2f}, {mapped_z:.2f})"
            )
        
        return obstacles_list


    def _map_value(self, value, from_range, to_range, clip=True):
        """
        Maps a value from one range to another with a simple linear scaling.
        """
        from_min, from_max = from_range
        to_min, to_max = to_range

        if np.isclose(from_min, from_max, atol=1e-6):
            return (to_min + to_max) / 2

        # Simple linear mapping
        normalized_value = (value - from_min) / (from_max - from_min)
        mapped_value = to_min + normalized_value * (to_max - to_min) 

        if clip:
            mapped_value = np.clip(mapped_value, to_min, to_max)

        return mapped_value

    def _visualize_results(self, left_rgb, right_rgb, disparity, pc, labels, num_clusters):
        """
        Save stereo images and clustering to disk for debugging.
        """
        try:
            # Stereo images and disparity map
            fig1 = plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(left_rgb)
            plt.title("Left Camera")
            plt.axis("off")
            
            plt.subplot(132)
            plt.imshow(right_rgb)
            plt.title("Right Camera")
            plt.axis("off")
            
            plt.subplot(133)
            plt.imshow(disparity, cmap="jet")
            plt.title("Disparity Map")
            plt.colorbar(label="Disparity")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("stereo_results.png", dpi=150)
            plt.close(fig1)
            logger.info("Saved stereo_results.png")

            # Clustered objects (noise removed)
            fig2 = plt.figure(figsize=(10, 8))
            ax = fig2.add_subplot(111, projection="3d")

            valid_labels = np.unique(labels)
            valid_labels = valid_labels[valid_labels != -1]

            if len(valid_labels) == 0:
                logger.warning("⚠️ No valid clusters to plot!")
                return
            
            colors = plt.cm.jet(np.linspace(0, 1, len(valid_labels)))

            for c, label in enumerate(valid_labels):
                cluster_points = pc[labels == label]
                if len(cluster_points) == 0:
                    continue

                ax.scatter(
                    cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                    color=colors[c % len(colors)],
                    label=f"Cluster {label}",
                    s=8
                )

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("Clustered Objects (Noise Removed)")
            ax.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            plt.savefig("cluster_results.png", dpi=150)
            plt.close(fig2)
            logger.info("Saved cluster_results.png (NO NOISE)")

        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")


def main():
    """
    Entry point for running the ScenePerception logic as a normal script.
    """
    try:
        # Instantiate ScenePerception, which runs all logic on init
        scene = ScenePerception()
    except KeyboardInterrupt:
        logger.info("Shutting down ScenePerception...")
    finally:
        # Safely disconnect from PyBullet if needed
        if pb.getConnectionInfo() is not None:
            pb.disconnect()


if __name__ == "__main__":
    main()
