#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Vision Module - ManipulaPy

This module provides computer vision capabilities for robotic perception including
stereo vision, object detection, and camera calibration with optional YOLO integration.

Performance Note:
    YOLO models are cached globally to avoid reinstantiation overhead. The first call to
    detect_objects() will download and load the model (~200MB, 2-3 seconds), but subsequent
    calls reuse the cached instance for real-time performance. Use clear_yolo_cache() to
    free memory when needed.

    For production use, the Vision class provides better model lifecycle management.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import numpy as np
import cv2
import pybullet as pb
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Any
from .utils import euler_to_rotation_matrix
import warnings

# Optional imports - handle gracefully if not available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None  # Define YOLO as None when not available
    warnings.warn(
        "ultralytics not available. Vision features requiring YOLO will be disabled. "
        "Install with: pip install ultralytics",
        ImportWarning
    )

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    warnings.warn(
        "OpenCV not available. Some vision features will be disabled. "
        "Install with: pip install opencv-python",
        ImportWarning
    )

# Global model cache to avoid reinstantiation
_YOLO_MODEL_CACHE: Dict[str, Any] = {}

def detect_objects(
    image: Union[np.ndarray, str],
    model_path: Optional[str] = None
) -> Any:
    """
    Detect objects in an image using YOLO with model caching.

    Note: For real-time/repeated detection, use the Vision class instead,
    which provides better model lifecycle management.

    Args:
        image: Input image (numpy array or file path)
        model_path: Path to YOLO model (optional, defaults to 'yolov8n.pt')

    Returns:
        Detection results from YOLO model

    Raises:
        ImportError: If ultralytics is not available

    Performance Note:
        This function caches the YOLO model globally. The first call will
        download/load the model (~200MB, 2-3 seconds), but subsequent calls
        with the same model_path will reuse the cached instance.

    Example:
        >>> import numpy as np
        >>> from ManipulaPy.vision import detect_objects
        >>>
        >>> # First call: loads model (slow)
        >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        >>> results1 = detect_objects(image)
        >>>
        >>> # Second call: reuses cached model (fast)
        >>> results2 = detect_objects(image)
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "ultralytics is required for object detection. "
            "Install with: pip install ultralytics"
        )

    # Use model caching to avoid reinstantiation bottleneck
    model_key = model_path or 'yolov8n.pt'

    if model_key not in _YOLO_MODEL_CACHE:
        # First call: load and cache the model
        _YOLO_MODEL_CACHE[model_key] = YOLO(model_key)

    # Reuse cached model
    model = _YOLO_MODEL_CACHE[model_key]
    results = model(image)
    return results

def clear_yolo_cache(model_path: Optional[str] = None) -> int:
    """
    Clear cached YOLO models to free memory.

    Args:
        model_path: Specific model to clear, or None to clear all cached models

    Returns:
        Number of models cleared from cache

    Example:
        >>> from ManipulaPy.vision import detect_objects, clear_yolo_cache
        >>>
        >>> # Use model
        >>> results = detect_objects(image)
        >>>
        >>> # Free memory when done
        >>> count = clear_yolo_cache()
        >>> print(f"Cleared {count} cached models")
    """
    global _YOLO_MODEL_CACHE

    if model_path is None:
        # Clear all cached models
        count = len(_YOLO_MODEL_CACHE)
        _YOLO_MODEL_CACHE.clear()
        return count
    else:
        # Clear specific model
        model_key = model_path
        if model_key in _YOLO_MODEL_CACHE:
            del _YOLO_MODEL_CACHE[model_key]
            return 1
        return 0

def process_image(image):
    """
    Basic image processing function.

    Args:
        image: Input image (numpy array)

    Returns:
        Processed image
    """
    if not OPENCV_AVAILABLE:
        # Fallback to basic numpy operations
        if isinstance(image, str):
            raise ImportError(
                "OpenCV is required to load images from file paths. "
                "Install with: pip install opencv-python"
            )
        return image

    # Use OpenCV if available
    if isinstance(image, str):
        return cv2.imread(image)
    return image

# Add this to ensure the module can be imported even without dependencies
__all__ = [
    'detect_objects',
    'clear_yolo_cache',
    'process_image',
    'Vision',
    'ULTRALYTICS_AVAILABLE',
    'OPENCV_AVAILABLE'
]
def read_debug_parameters(dbg_params):
    """
    Utility to read current slider values from PyBullet debug interface.
    """
    values = {}
    for name, param_id in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param_id)
    return values


class Vision:
    """
    A unified vision class for monocular/stereo inputs and PyBullet debugging.

    Features:
    ---------
      - Monocular camera(s) with user-specified intrinsics/extrinsics or OpenCV capture devices.
      - Optional PyBullet debug sliders for a "virtual" camera.
      - Stereo pipeline: rectification, disparity, 3D point cloud generation (if stereo configs provided).
      - Basic obstacle detection via depth thresholding + intrinsics-based unprojection.

    Parameters
    ----------
    camera_configs : list of dict, optional
        Each dict describes one camera (monocular or part of a stereo pair):
            - "name" (str)
            - "translation" ([x, y, z]) in world or local coords
            - "rotation" ([roll_deg, pitch_deg, yaw_deg]) in degrees
            - "fov" (float)
            - "near" (float)
            - "far" (float)
            - "intrinsic_matrix" (3x3 np.array)
            - "distortion_coeffs" (1D np.array of length=5)
            - "use_opencv" (bool)
            - "device_index" (int) => for OpenCV
    stereo_configs : tuple(dict, dict), optional
        (left_cam_cfg, right_cam_cfg) for stereo. Must have 'intrinsic_matrix',
        'distortion_coeffs', 'translation', 'rotation', etc.
    use_pybullet_debug : bool
        If True, create debug sliders in PyBullet for a single "virtual" camera.
    show_plot : bool
        If True (and use_pybullet_debug=True), display the debug camera feed in a Matplotlib window.
    logger_name : str
        Logger name for this class.
    physics_client : int or None
        PyBullet client ID if controlling a simulation environment.
    """

    def __init__(
        self,
        camera_configs=None,
        stereo_configs=None,
        use_pybullet_debug=False,
        show_plot=True,
        logger_name="VisionSystemLogger",
        physics_client=None,
    ):
        """
        Initializes the Vision system with optional monocular/stereo cameras, PyBullet debug tools, and YOLO object detection.

        Parameters:
            - camera_configs: List of dictionaries for monocular cameras or stereo pairs.
            - stereo_configs: Tuple(left_cam_config, right_cam_config) for stereo processing.
            - use_pybullet_debug: Enables PyBullet sliders to modify virtual camera parameters.
            - show_plot: Displays PyBullet debug images using Matplotlib.
            - logger_name: Name for logging information.
            - physics_client: PyBullet client ID for interacting with a simulation.
        """
        self.logger = self._setup_logger(logger_name)
        self.logger.info("Initializing Vision system...")

        # Store external references/flags
        self.physics_client = physics_client
        self.use_pybullet_debug = use_pybullet_debug
        self.show_plot = show_plot

        # YOLO Model (lazy loading - will be loaded on first use)
        self.yolo_model = None
        self._yolo_model_path = "yolov8m.pt"  # Default model
        self._yolo_load_attempted = False


        # Camera configuration (Monocular or Stereo)
        self.cameras = {}
        self.capture_devices = {}

        if not camera_configs:
            self.logger.info("⚠️ No camera_configs provided; using default settings.")
            camera_configs = [
                {
                    "name": "default_camera",
                    "translation": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "fov": 60,
                    "near": 0.1,
                    "far": 5.0,
                    "intrinsic_matrix": np.array(
                        [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32
                    ),
                    "distortion_coeffs": np.zeros(5, dtype=np.float32),
                    "use_opencv": False,
                    "device_index": 0,
                }
            ]

        # Stereo configuration
        self.stereo_enabled = stereo_configs is not None
        self.left_cam_cfg = None
        self.right_cam_cfg = None
        self.left_map_x = None
        self.left_map_y = None
        self.right_map_x = None
        self.right_map_y = None
        self.Q = None
        self.stereo_matcher = None

        if self.stereo_enabled:
            left_cfg, right_cfg = stereo_configs
            self._validate_stereo_config(left_cfg, right_cfg)
            self.left_cam_cfg = left_cfg
            self.right_cam_cfg = right_cfg

            # Stereo disparity matcher (StereoSGBM for better quality)
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,
                blockSize=7,
                P1=8 * 3 * 7**2,
                P2=32 * 3 * 7**2,
            )
            self.logger.info("✅ Stereo processing enabled.")

        # If using PyBullet debug, initialize sliders
        if self.use_pybullet_debug:
            self.logger.info("📌 Using PyBullet debug sliders for virtual camera.")
            self._setup_pybullet_debug_sliders()

            if self.show_plot:
                self.fig = plt.figure("PyBullet Debug Camera", figsize=(5, 4))
                self.ax = self.fig.add_subplot(111)
                init_img = np.zeros((240, 320, 4), dtype=np.uint8)
                self.img_display = self.ax.imshow(init_img, origin="upper")
                plt.axis("off")
                plt.tight_layout()
        else:
            # Initialize regular cameras
            for i, cfg in enumerate(camera_configs):
                self._configure_camera(i, cfg)

    # --------------------------------------------------------------------------
    # Logger
    # --------------------------------------------------------------------------
    def _setup_logger(self, name):
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
    # Camera Config
    # --------------------------------------------------------------------------
    def _configure_camera(self, idx, cfg):
        """
        Store a camera configuration and open an OpenCV capture device if requested.
        """
        name = cfg.get("name", f"camera_{idx}")
        translation = cfg.get("translation", [0, 0, 0])
        rotation_deg = cfg.get("rotation", [0, 0, 0])
        fov = cfg.get("fov", 60)
        near_val = cfg.get("near", 0.1)
        far_val = cfg.get("far", 5.0)
        intrinsic = cfg.get("intrinsic_matrix", np.eye(3, dtype=np.float32))
        distortion = cfg.get("distortion_coeffs", np.zeros(5, dtype=np.float32))
        use_opencv = cfg.get("use_opencv", False)
        device_index = cfg.get("device_index", idx)

        # Build extrinsic from translation + Euler angles
        extrinsic = self._make_extrinsic_matrix(translation, rotation_deg)

        self.cameras[idx] = {
            "name": name,
            "translation": translation,
            "rotation": rotation_deg,
            "fov": fov,
            "near": near_val,
            "far": far_val,
            "intrinsic_matrix": intrinsic,
            "distortion_coeffs": distortion,
            "use_opencv": use_opencv,
            "device_index": device_index,
            "extrinsic_matrix": extrinsic,
        }

        # OpenCV capture if requested
        if use_opencv:
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                self.capture_devices[idx] = cap
                self.logger.info(f"{name}: Opened OpenCV device {device_index}.")
            else:
                msg = f"{name}: Failed to open OpenCV device {device_index}."
                self.logger.error(msg)
                raise RuntimeError(msg)

    def _validate_stereo_config(self, left_cfg, right_cfg):
        """
        Basic validation that each stereo cam config has required keys.
        """
        required = ["intrinsic_matrix", "distortion_coeffs", "translation", "rotation"]
        for i, cfg in enumerate([left_cfg, right_cfg]):
            for key in required:
                if key not in cfg:
                    msg = f"Stereo config for cam {i} missing '{key}'."
                    self.logger.error(msg)
                    raise ValueError(msg)

    # --------------------------------------------------------------------------
    # Extrinsic & Rotation Utility
    # --------------------------------------------------------------------------
    def _make_extrinsic_matrix(self, translation, rotation_deg):
        """
        Create a 4x4 extrinsic matrix from translation and Euler angles in degrees.
        Uses cv2.Rodrigues for improved numerical stability.
        """
        # Convert Euler angles (roll, pitch, yaw) -> rotation matrix
        # We'll do it by first converting Euler -> axis-angle or direct matrix.
        # For simplicity, let's do manual Euler -> rotation matrix, or combine them:
        R = self._euler_to_rotation_matrix(rotation_deg)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T

    def _euler_to_rotation_matrix(self, euler_deg):
        """
        Convert [roll_deg, pitch_deg, yaw_deg] to a rotation matrix via manual multiplication,
        or you can chain cv2.Rodrigues calls. We'll do a direct approach here for clarity.
        """
        roll, pitch, yaw = np.radians(euler_deg)

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ],
            dtype=np.float32,
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ],
            dtype=np.float32,
        )

        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
            dtype=np.float32,
        )

        # Final rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx

    # --------------------------------------------------------------------------
    # PyBullet Debug Camera
    # --------------------------------------------------------------------------
    def _setup_pybullet_debug_sliders(self):
        """
        Creates PyBullet debug sliders for a single 'virtual' camera.
        """
        self.dbg_params = {}
        # View matrix sliders
        self.dbg_params["target_x"] = pb.addUserDebugParameter("target_x", -1, 1, 0)
        self.dbg_params["target_y"] = pb.addUserDebugParameter("target_y", -1, 1, 0)
        self.dbg_params["target_z"] = pb.addUserDebugParameter("target_z", -1, 1, 0)
        self.dbg_params["distance"] = pb.addUserDebugParameter("distance", 0, 10, 2)
        self.dbg_params["yaw"] = pb.addUserDebugParameter("yaw", -180, 180, 0)
        self.dbg_params["pitch"] = pb.addUserDebugParameter("pitch", -180, 180, -40)
        self.dbg_params["roll"] = pb.addUserDebugParameter("roll", -180, 180, 0)
        self.dbg_params["upAxisIndex"] = pb.addUserDebugParameter(
            "upAxisIndex", 0, 1, 1
        )

        # Projection matrix sliders
        self.dbg_params["width"] = pb.addUserDebugParameter("width", 100, 1000, 320)
        self.dbg_params["height"] = pb.addUserDebugParameter("height", 100, 1000, 240)
        self.dbg_params["fov"] = pb.addUserDebugParameter("fov", 1, 180, 60)
        self.dbg_params["near_val"] = pb.addUserDebugParameter(
            "near_val", 1e-3, 1.0, 0.1
        )
        self.dbg_params["far_val"] = pb.addUserDebugParameter("far_val", 1.0, 50.0, 5.0)

        # Print button
        self.dbg_params["print"] = pb.addUserDebugParameter("print_params", 1, 0, 1)
        self.old_print_val = 1

    def _get_pybullet_view_proj(self):
        """
        Reads debug sliders, returns (view_mtx, proj_mtx, width, height).
        """
        vals = read_debug_parameters(self.dbg_params)

        # Build view matrix
        target_pos = [vals["target_x"], vals["target_y"], vals["target_z"]]
        up_axis_idx = int(vals["upAxisIndex"])
        view_mtx = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_pos,
            distance=vals["distance"],
            yaw=vals["yaw"],
            pitch=vals["pitch"],
            roll=vals["roll"],
            upAxisIndex=up_axis_idx,
        )

        # Build projection matrix
        width = int(vals["width"])
        height = int(vals["height"])
        aspect = width / float(height)
        proj_mtx = pb.computeProjectionMatrixFOV(
            fov=vals["fov"],
            aspect=aspect,
            nearVal=vals["near_val"],
            farVal=vals["far_val"],
        )

        # Debug print if button clicked
        if self.old_print_val != vals["print"]:
            self.old_print_val = vals["print"]
            vm_np = np.array(view_mtx).reshape((4, 4), order="F")
            pm_np = np.array(proj_mtx).reshape((4, 4), order="F")
            self.logger.info("===== PYBULLET DEBUG CAMERA PARAMS =====")
            self.logger.info(f"View Matrix:\n{vm_np}")
            self.logger.info(f"Projection Matrix:\n{pm_np}")
            self.logger.info("========================================\n")

        return view_mtx, proj_mtx, width, height

    # --------------------------------------------------------------------------
    # YOLO Lazy Loading
    # --------------------------------------------------------------------------
    def _ensure_yolo_loaded(self):
        """
        Lazy load YOLO model on first use.

        Returns:
            bool: True if YOLO is loaded successfully, False otherwise
        """
        if self.yolo_model is not None:
            return True

        if self._yolo_load_attempted:
            return False

        self._yolo_load_attempted = True

        if not ULTRALYTICS_AVAILABLE:
            self.logger.warning("YOLO not available - ultralytics not installed.")
            return False

        try:
            self.logger.info(f"Loading YOLO model ({self._yolo_model_path})...")
            self.yolo_model = YOLO(self._yolo_model_path)
            self.logger.info("✅ YOLO model loaded successfully.")
            return True
        except Exception as e:
            self.logger.warning(f"YOLO load failed ({e}); continuing without it.")
            return False

    # --------------------------------------------------------------------------
    # Image Capture
    # --------------------------------------------------------------------------
    def capture_image(self, camera_index=0):
        """
        Captures an RGB and depth image from PyBullet cameras.
        """
        if camera_index not in self.cameras:
            self.logger.error(f"Camera index {camera_index} not found.")
            return None, None

        cfg = self.cameras[camera_index]
        width, height = 640, 480

        target = [0, 0, 0.5]
        up_vector = [0, 0, 1]

        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=cfg["translation"],
            cameraTargetPosition=target,
            cameraUpVector=up_vector,
        )
        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=cfg["fov"],
            aspect=width / float(height),
            nearVal=cfg["near"],
            farVal=cfg["far"],
        )

        _, _, rgba, depth_buf, _ = pb.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix
        )

        rgb = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        depth = np.array(depth_buf, dtype=np.float32).reshape((height, width))

        # Fix depth range scaling
        near, far = cfg["near"], cfg["far"]
        depth = near + (far - near) * depth

        return rgb, depth

    # --------------------------------------------------------------------------
    # Basic Obstacle Detection
    # --------------------------------------------------------------------------
    def detect_obstacles(
        self,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        depth_threshold: float = 0.0,
        camera_index: int = 0,
        step: int = 5,
    ):
        """
        Detect obstacles with YOLO (if available) and estimate 3D positions via depth.

        Returns
        -------
        positions : (N, 3) float32 array
        orientations : (N,) float32 array  (yaw angles in degrees in XY plane)
        """
        # --------- Basic sanity checks ----------
        if depth_image is None or rgb_image is None:
            self.logger.error("❌ Invalid depth or RGB input")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if camera_index not in self.cameras:
            self.logger.error(f"Camera index {camera_index} not found.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Lazy load YOLO on first use
        if not self._ensure_yolo_loaded():
            self.logger.warning("⚠️ YOLO model is not available! Skipping object detection.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # --------- YOLO inference (guarded) ----------
        try:
            results = self.yolo_model(rgb_image, conf=0.3)
        except Exception as e:
            self.logger.warning(f"YOLO inference failed ({e}); returning empty.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if not results or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
            self.logger.info("No objects detected by YOLO.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = results[0].boxes

        fx = self.cameras[camera_index]["intrinsic_matrix"][0, 0]
        fy = self.cameras[camera_index]["intrinsic_matrix"][1, 1]
        cx_i = self.cameras[camera_index]["intrinsic_matrix"][0, 2]
        cy_i = self.cameras[camera_index]["intrinsic_matrix"][1, 2]

        positions = []
        orientations = []

        for box in boxes:
            # xyxy may be tensor/ndarray; make sure it’s a flat list of 4 ints
            xyxy = getattr(box, "xyxy", None)
            if xyxy is None:
                continue
            coords = np.array(xyxy[0]).astype(int).tolist()
            if len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords

            # Subsample ROI to speed up median depth if 'step' > 1
            depth_roi = depth_image[y1:y2:step, x1:x2:step]
            valid_depths = depth_roi[depth_roi > 0]
            if valid_depths.size == 0:
                continue

            mean_depth = float(np.median(valid_depths))
            if mean_depth > depth_threshold:
                # Too far -> ignore
                continue

            # Center pixel
            cx_pix = (x1 + x2) // 2
            cy_pix = (y1 + y2) // 2

            x_cam = (cx_pix - cx_i) * mean_depth / fx
            y_cam = (cy_pix - cy_i) * mean_depth / fy
            z_cam = mean_depth

            positions.append([x_cam, y_cam, z_cam])
            orientations.append(np.degrees(np.arctan2(y_cam, x_cam)))

        if not positions:
            self.logger.info("No obstacles kept after depth filtering.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return np.asarray(positions, dtype=np.float32), np.asarray(orientations, dtype=np.float32)

    # --------------------------------------------------------------------------
    # Stereo Methods
    # --------------------------------------------------------------------------

    def compute_stereo_rectification_maps(self, image_size=(640, 480)):
        if not self.stereo_enabled:
            self.logger.warning("Stereo not enabled.")
            return

        # Extract intrinsics/distortions
        K1 = self.left_cam_cfg["intrinsic_matrix"]
        D1 = self.left_cam_cfg["distortion_coeffs"]
        K2 = self.right_cam_cfg["intrinsic_matrix"]
        D2 = self.right_cam_cfg["distortion_coeffs"]

        # Build rotation/translation between left & right
        R_l = euler_to_rotation_matrix(
            self.left_cam_cfg["rotation"]
        )  # might be float32
        t_l = np.array(self.left_cam_cfg["translation"], dtype=np.float32)
        R_r = euler_to_rotation_matrix(self.right_cam_cfg["rotation"])
        t_r = np.array(self.right_cam_cfg["translation"], dtype=np.float32)
        R_lr = R_r @ R_l.T
        t_lr = t_r - R_lr @ t_l

        # -- NEW LINES: unify types to float64 for all stereo inputs --
        K1 = K1.astype(np.float64)
        D1 = D1.astype(np.float64)
        K2 = K2.astype(np.float64)
        D2 = D2.astype(np.float64)
        R_lr = R_lr.astype(np.float64)
        t_lr = t_lr.astype(np.float64)

        # Now call stereoRectify with consistent 64-bit floats
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=K1,
            distCoeffs1=D1,
            cameraMatrix2=K2,
            distCoeffs2=D2,
            imageSize=image_size,
            R=R_lr,
            T=t_lr,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        # Generate undistort rectification maps (still fine to use float32 maps)
        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, image_size, cv2.CV_32FC1
        )
        self.Q = Q
        self.logger.info("Stereo rectification maps computed successfully.")

    def rectify_stereo_images(self, left_img, right_img):
        """
        Remap left and right images to their rectified forms.
        """
        if not self.stereo_enabled:
            raise RuntimeError("Stereo is disabled; cannot rectify images.")
        if self.left_map_x is None or self.right_map_x is None:
            raise RuntimeError(
                "Rectification maps not computed. Call compute_stereo_rectification_maps first."
            )

        left_rect = cv2.remap(
            left_img, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR
        )
        right_rect = cv2.remap(
            right_img, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR
        )
        return left_rect, right_rect

    def compute_disparity(self, left_rect, right_rect):
        """
        Compute a disparity map from rectified stereo images using StereoSGBM (or BM).
        """
        if self.stereo_matcher is None:
            raise RuntimeError(
                "Stereo matcher not initialized (stereo configs missing)."
            )

        # Convert to grayscale if needed
        if len(left_rect.shape) == 3:
            left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        if len(right_rect.shape) == 3:
            right_rect = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo_matcher.compute(left_rect, right_rect).astype(
            np.float32
        )
        # Most OpenCV stereo matchers produce fixed-point disparities in Q16.
        # Typically we divide by 16.0 to get real disparity values:
        disparity /= 16.0

        return disparity

    def disparity_to_pointcloud(self, disparity):
        """
        Reproject a disparity map to 3D points using the Q matrix from stereoRectify.
        """
        if self.Q is None:
            raise RuntimeError(
                "No Q matrix found. Did you call compute_stereo_rectification_maps?"
            )

        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        h, w, _ = points_3D.shape
        cloud = points_3D.reshape(-1, 3)

        # Filter out invalid/inf points and too far points
        disp_flat = disparity.reshape(-1)
        valid_mask = (disp_flat > 0) & np.isfinite(cloud[:, 0]) & (cloud[:, 2] < 10.0)
        cloud_filtered = cloud[valid_mask]

        return cloud_filtered

    def get_stereo_point_cloud(self, left_img, right_img):
        """
        High-level pipeline: rectify, compute disparity, reproject to 3D.
        """
        if not self.stereo_enabled:
            raise RuntimeError(
                "Stereo is not enabled. Provide stereo_configs in constructor."
            )

        left_rect, right_rect = self.rectify_stereo_images(left_img, right_img)
        disparity = self.compute_disparity(left_rect, right_rect)
        point_cloud = self.disparity_to_pointcloud(disparity)
        return point_cloud

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    def release(self):
        """
        Release resources (e.g., OpenCV capture devices).
        """
        for idx, cap in self.capture_devices.items():
            cap.release()
            self.logger.info(f"Released OpenCV camera {idx}.")
        self.capture_devices.clear()

# Replace the __del__ method in ManipulaPy/vision.py (around line 648)

    def __del__(self):
        """
        Destructor: ensure we release resources gracefully.
        """
        try:
            # Only attempt logging if we have a valid logger with open handlers
            if (hasattr(self, 'logger') and 
                self.logger and 
                hasattr(self.logger, 'handlers')):
                
                # Check if any handlers are still valid (not closed)
                has_valid_handler = False
                for handler in self.logger.handlers:
                    try:
                        # Check if handler has a stream and if it's open
                        if hasattr(handler, 'stream'):
                            if hasattr(handler.stream, 'closed'):
                                if not handler.stream.closed:
                                    has_valid_handler = True
                                    break
                            else:
                                # Stream exists but no 'closed' attribute, assume it's valid
                                has_valid_handler = True
                                break
                        else:
                            # Handler without stream (e.g., NullHandler), assume valid
                            has_valid_handler = True
                            break
                    except (AttributeError, ValueError):
                        # Skip this handler if we can't determine its state
                        continue
                
                # Only log if we found at least one valid handler
                if has_valid_handler:
                    try:
                        self.logger.debug("Vision destructor called; releasing resources.")
                    except (ValueError, OSError):
                        # Even if we thought the handler was valid, it might have closed
                        # between our check and the log call. Just ignore it.
                        pass
            
            # Always attempt to release resources, regardless of logging success
            if hasattr(self, 'release'):
                self.release()
                
        except Exception:
            # Silently ignore ALL exceptions during destruction
            # The destructor should never raise exceptions
            pass
