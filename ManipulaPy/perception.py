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
        Configure a logger for this Perception module.

        Returns
        -------
        logging.Logger
        """
        logger = logging.getLogger(name)
        # Avoid adding multiple handlers (e.g., in Jupyter or repeated imports)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    # --------------------------------------------------------------------------
    # Primary Methods
    # --------------------------------------------------------------------------

    def detect_and_cluster_obstacles(self, camera_index=0, depth_threshold=0.5, step=10, eps=0.05, min_samples=5):
        """
        Capture an image from Vision, detect 3D obstacle points, and cluster them.
        """
        rgb, depth = self.vision.capture_image(camera_index=camera_index)
        if depth is None or len(depth.shape) < 2:
            self.logger.warning(f"❌ Depth image not available from camera {camera_index}; returning empty arrays.")
            return np.empty((0, 3)), np.array([])

        obstacle_points, labels = self.vision.detect_obstacles(
            depth_image=depth,
            rgb_image=rgb,
            depth_threshold=depth_threshold,
            camera_index=camera_index,
            step=step
        )

        if obstacle_points is None or labels is None:
            self.logger.error("🚨 detect_obstacles() returned None! Fixing...")
            return np.empty((0, 3)), np.array([])  # Prevents crash

        # Clustering
        if obstacle_points.shape[0] == 0:
            self.logger.info("⚠️ No obstacles detected to cluster.")
            return obstacle_points, np.array([])

        labels, num_clusters = self.cluster_obstacles(obstacle_points, eps=eps, min_samples=min_samples)
        self.logger.info(f"✅ Obstacles clustered. Found {num_clusters} clusters.")

        return obstacle_points, labels

    # --------------------------------------------------------------------------
    # Stereo Methods
    # --------------------------------------------------------------------------
    def compute_stereo_disparity(self, left_img, right_img):
        """
        Compute a stereo disparity map from two images (rectified or not).

        Parameters
        ----------
        left_img : np.ndarray
            Image from the left camera.
        right_img : np.ndarray
            Image from the right camera.

        Returns
        -------
        disparity : np.ndarray (float32)
            Computed disparity map.
        """
        if not self.vision.stereo_enabled:
            raise RuntimeError("Stereo is not enabled in the attached Vision instance.")
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
            3D point cloud in world coordinates.
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
    # Resource Management
    # --------------------------------------------------------------------------
    def release(self):
        """
        Release any resources (e.g., camera interfaces) held by the Vision instance.
        """
        if self.vision:
            try:
                self.logger.info("Releasing Perception resources (Vision).")
                self.vision.release()
            except Exception as e:
                self.logger.error(f"Error while releasing Vision resources: {e}")

    def __del__(self):
        """
        Destructor for Perception.
        """
        try:
            if hasattr(self, 'vision') and self.vision is not None:
                self.vision.release()
        except Exception:
            pass
