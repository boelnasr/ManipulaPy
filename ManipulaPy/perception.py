#!/usr/bin/env python3
import logging
import numpy as np
from sklearn.cluster import DBSCAN


class Perception:
    """
    A higher-level perception module that uses a Vision instance to handle
    tasks like obstacle detection, 3D point cloud generation, and clustering.

    Attributes
    ----------
    vision : Vision
        A Vision instance for camera tasks (capturing images, stereo, etc.).
    logger : logging.Logger
        Logger for debugging and status messages.
    """

    def __init__(self, vision_instance=None, logger_name="PerceptionLogger"):
        """
        Initialize the Perception system with a Vision instance.

        Parameters
        ----------
        vision_instance : Vision, optional
            A Vision instance for camera tasks (monocular/stereo).
            Must be provided or else a ValueError is raised.
        logger_name : str
            The name for this Perception logger.
        """
        self.logger = self._setup_logger(logger_name)
        if vision_instance is None:
            raise ValueError("A valid Vision instance must be provided.")
        self.vision = vision_instance
        self.logger.info("Perception initialized successfully.")

    def _setup_logger(self, name):
        """
        Configure and return a logger for this Perception module.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    # --------------------------------------------------------------------------
    # Primary Methods
    # --------------------------------------------------------------------------
    def detect_and_cluster_obstacles(
        self, camera_index=0, depth_threshold=5.0, step=2, eps=0.1, min_samples=3
    ):
        """
        Capture an image from the Vision instance, detect 3D obstacle points,
        and then cluster those points using DBSCAN.

        Parameters
        ----------
        camera_index : int
            Index of the camera to capture from.
        depth_threshold : float
            Maximum depth (in meters) to consider for obstacles.
        step : int
            Step size for downsampling the depth image.
        eps : float
            Maximum distance between two points to be considered neighbors in DBSCAN.
        min_samples : int
            Minimum number of points required to form a cluster in DBSCAN.

        Returns
        -------
        obstacle_points : np.ndarray of shape (N, 3)
            3D points representing detected obstacles.
        labels : np.ndarray
            Cluster labels for each point (with -1 for noise).
        """
        rgb, depth = self.vision.capture_image(camera_index=camera_index)
        if depth is None or depth.ndim < 2:
            self.logger.warning(
                f"❌ Depth image not available from camera {camera_index}; returning empty arrays."
            )
            return np.empty((0, 3)), np.array([])

        self.logger.debug("Calling vision.detect_obstacles()...")
        obstacle_points, labels = self.vision.detect_obstacles(
            depth_image=depth,
            rgb_image=rgb,
            depth_threshold=depth_threshold,
            camera_index=camera_index,
            step=step,
        )

        if obstacle_points is None or labels is None:
            self.logger.error(
                "🚨 detect_obstacles() returned None! Returning empty arrays to prevent crash."
            )
            return np.empty((0, 3)), np.array([])

        if obstacle_points.shape[0] == 0:
            self.logger.info("⚠️ No obstacle points detected to cluster.")
            return obstacle_points, np.array([])

        self.logger.debug(
            f"Detected {obstacle_points.shape[0]} obstacle points before clustering."
        )
        labels, num_clusters = self.cluster_obstacles(
            obstacle_points, eps=eps, min_samples=min_samples
        )
        self.logger.info(
            f"✅ Clustering complete. Found {num_clusters} clusters (excluding noise)."
        )
        return obstacle_points, labels

    # --------------------------------------------------------------------------
    # Stereo Methods
    # --------------------------------------------------------------------------
    def compute_stereo_disparity(self, left_img, right_img):
        """
        Compute a stereo disparity map from two images.

        Parameters
        ----------
        left_img : np.ndarray
            Image from the left camera.
        right_img : np.ndarray
            Image from the right camera.

        Returns
        -------
        disparity : np.ndarray of type float32
            Computed disparity map.
        """
        if not self.vision.stereo_enabled:
            raise RuntimeError("Stereo is not enabled in the Vision instance.")
        left_rect, right_rect = self.vision.rectify_stereo_images(left_img, right_img)
        disparity = self.vision.compute_disparity(left_rect, right_rect)
        self.logger.debug("Stereo disparity computed.")
        return disparity

    def get_stereo_point_cloud(self, left_img, right_img):
        """
        Generate a 3D point cloud from a stereo pair of images.

        Parameters
        ----------
        left_img : np.ndarray
            Image from the left camera.
        right_img : np.ndarray
            Image from the right camera.

        Returns
        -------
        point_cloud : np.ndarray of shape (N, 3)
            The 3D point cloud in world coordinates.
        """
        if not self.vision.stereo_enabled:
            self.logger.error("Stereo is not enabled in the Vision instance.")
            return np.empty((0, 3))
        point_cloud = self.vision.get_stereo_point_cloud(left_img, right_img)
        self.logger.debug(
            f"Stereo point cloud generated with {point_cloud.shape[0]} points."
        )
        return point_cloud

    # --------------------------------------------------------------------------
    # Clustering
    # --------------------------------------------------------------------------
    def cluster_obstacles(self, points, eps=0.1, min_samples=3):
        """
        Cluster the 3D points using DBSCAN.

        Parameters
        ----------
        points : np.ndarray of shape (N, 3)
            The 3D points to cluster.
        eps : float
            The maximum distance between two points for them to be considered neighbors.
        min_samples : int
            The minimum number of points required to form a dense region.

        Returns
        -------
        labels : np.ndarray of shape (N,)
            The cluster label for each point (-1 indicates noise).
        num_clusters : int
            The number of clusters found (excluding noise).
        """
        if points.shape[0] == 0:
            self.logger.info(
                "⚠️ No points provided to cluster_obstacles; returning empty results."
            )
            return np.array([]), 0

        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_model.fit(points)
        labels = dbscan_model.labels_
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = np.sum(labels == -1)
        self.logger.info(
            f"DBSCAN clustering: {num_clusters} clusters found with {noise_count} noise points."
        )
        return labels, num_clusters

    # --------------------------------------------------------------------------
    # Resource Management
    # --------------------------------------------------------------------------
    def release(self):
        """
        Release resources held by the Vision instance.
        """
        if self.vision:
            try:
                self.logger.info("Releasing Vision resources...")
                self.vision.release()
            except Exception as e:
                self.logger.error(f"Error releasing Vision resources: {e}")

    def __del__(self):
        """
        Destructor to ensure Vision resources are released.
        """
        try:
            if hasattr(self, "vision") and self.vision is not None:
                self.vision.release()
        except Exception:
            pass
