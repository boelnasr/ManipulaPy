#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Perception Intermediate Demo - ManipulaPy

End-to-end tour of ManipulaPy's vision and perception stack, driven entirely by
the real ``ManipulaPy.vision.Vision`` and ``ManipulaPy.perception.Perception``
APIs against a headless PyBullet scene:

- Virtual-camera RGB + metric-depth capture (``Vision.capture_image``).
- Depth-based obstacle extraction with optional YOLO detection, falling back to a
  pure-depth point cloud when YOLO weights are unavailable
  (``Perception.detect_and_cluster_obstacles`` / ``Vision.detect_obstacles``).
- DBSCAN clustering of 3D points (``Perception.cluster_obstacles``).
- Stereo rectification, disparity and point-cloud reconstruction
  (``Vision.compute_stereo_rectification_maps`` / ``rectify_stereo_images`` /
  ``compute_disparity`` / ``get_stereo_point_cloud``).

The demo runs cleanly headless (no display, no GPU, no camera). PyBullet is used
in DIRECT mode and YOLO is pinned to CPU so it never crashes on a GPU-less box.
All plots are saved next to this script.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pin YOLO to CPU *before* importing ManipulaPy.vision so the model never tries
# to return CUDA tensors on a GPU-less / driver-mismatched machine.
os.environ.setdefault("MANIPULAPY_YOLO_DEVICE", "cpu")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; honour MPLBACKEND=Agg
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Make the package importable when run directly from the examples folder.
_CURRENT_DIR = Path(__file__).parent.absolute()
_REPO_ROOT = _CURRENT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import pybullet as pb  # noqa: E402
import pybullet_data  # noqa: E402

from ManipulaPy.vision import Vision  # noqa: E402
from ManipulaPy.perception import Perception  # noqa: E402

plt.ioff()


class PerceptionIntermediateDemo:
    """Showcase ManipulaPy's Vision/Perception API on a headless PyBullet scene."""

    def __init__(self, save_outputs: bool = True) -> None:
        """Initialise logging and output bookkeeping.

        Args:
            save_outputs: If True, save figures next to this script.
        """
        self.save_outputs = save_outputs
        self.script_dir = _CURRENT_DIR
        self.outputs_saved: List[str] = []
        self.demo_results: Dict[str, dict] = {}

        self.physics_client: Optional[int] = None
        self.vision: Optional[Vision] = None
        self.perception: Optional[Perception] = None

        self._setup_logging()
        self.logger.info("Perception demo initialized")

    # ------------------------------------------------------------------ infra
    def _setup_logging(self) -> None:
        """Configure a stdout logger for the demo."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger("PerceptionDemo")

    def _save_plot(self, filename: str) -> None:
        """Save the current matplotlib figure into the script directory."""
        if not self.save_outputs:
            return
        filepath = self.script_dir / filename
        try:
            plt.savefig(filepath, dpi=130, bbox_inches="tight", facecolor="white")
            self.outputs_saved.append(str(filepath))
            self.logger.info(f"Plot saved: {filepath.name}")
        except Exception as exc:  # pragma: no cover - IO edge case
            self.logger.error(f"Failed to save plot {filepath}: {exc}")

    # ------------------------------------------------------------------ setup
    def setup_scene(self) -> bool:
        """Build a small headless PyBullet scene with a few obstacles."""
        self.logger.info("=== Setting up PyBullet scene (DIRECT mode) ===")
        try:
            self.physics_client = pb.connect(pb.DIRECT)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.81)
            pb.loadURDF("plane.urdf")

            # A cluster of small cubes the camera will look down on.
            obstacle_xy = [(0.35, 0.0), (0.30, 0.25), (-0.25, 0.20), (-0.30, -0.25)]
            for x, y in obstacle_xy:
                cube = pb.loadURDF("cube_small.urdf", [x, y, 0.05])
                pb.changeVisualShape(cube, -1, rgbaColor=[0.9, 0.2, 0.2, 1])

            # Settle the scene.
            for _ in range(20):
                pb.stepSimulation()

            self.logger.info("Scene ready with %d obstacles", len(obstacle_xy))
            return True
        except Exception as exc:
            self.logger.error(f"PyBullet scene setup failed: {exc}")
            return False

    def setup_vision(self) -> bool:
        """Create the Vision (mono + stereo) and Perception systems."""
        self.logger.info("=== Setting up Vision and Perception ===")
        try:
            intrinsic = np.array(
                [[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]], dtype=np.float32
            )
            distortion = np.zeros(5, dtype=np.float32)

            def cam(name: str, translation: List[float]) -> dict:
                return {
                    "name": name,
                    "translation": translation,
                    "rotation": [0, -90, 0],  # looking straight down
                    "fov": 60,
                    "near": 0.1,
                    "far": 5.0,
                    "intrinsic_matrix": intrinsic,
                    "distortion_coeffs": distortion,
                    "use_opencv": False,
                    "device_index": 0,
                }

            mono = cam("overhead", [0.1, 0.0, 1.2])
            left = cam("left", [0.0, 0.0, 1.2])
            right = cam("right", [0.2, 0.0, 1.2])

            # use_pybullet_debug=False so cameras are actually configured and
            # capture_image() works through the virtual-camera path (no GUI).
            self.vision = Vision(
                camera_configs=[mono],
                stereo_configs=(left, right),
                use_pybullet_debug=False,
                show_plot=False,
                physics_client=self.physics_client,
            )
            self.perception = Perception(
                vision_instance=self.vision, logger_name="PerceptionDemo"
            )
            self.logger.info("Vision and Perception systems initialized")
            return True
        except Exception as exc:
            self.logger.error(f"Vision setup failed: {exc}")
            return False

    # ------------------------------------------------------------- capture
    def demo_image_capture(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture an RGB + metric-depth frame from the virtual camera."""
        self.logger.info("=== Virtual-Camera Capture ===")
        rgb, depth = self.vision.capture_image(camera_index=0)
        if rgb is None or depth is None:
            self.logger.warning("Capture returned no data")
            return None, None

        self.logger.info(
            "Captured RGB %s, depth %s (range %.2f-%.2f m)",
            rgb.shape,
            depth.shape,
            float(depth.min()),
            float(depth.max()),
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.imshow(rgb)
        ax1.set_title("RGB (overhead virtual camera)")
        ax1.axis("off")
        dimg = ax2.imshow(depth, cmap="jet")
        ax2.set_title("Metric Depth")
        ax2.axis("off")
        fig.colorbar(dimg, ax=ax2, fraction=0.046, label="meters")
        plt.tight_layout()
        self._save_plot("01_camera_capture.png")
        plt.close(fig)

        self.demo_results["image_capture"] = {
            "rgb_shape": list(rgb.shape),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
        }
        return rgb, depth

    # ------------------------------------------------------- obstacle pipeline
    def _depth_pointcloud(self, depth: np.ndarray, step: int) -> np.ndarray:
        """Back-project foreground depth pixels to 3D camera-frame points.

        Used as a YOLO-free fallback so the perception pipeline always has data
        to cluster, even without object-detection weights. Foreground (obstacle)
        pixels are isolated relative to the background plane: PyBullet's depth
        buffer is non-linear, so a scene-relative percentile cut is far more
        robust than a fixed metric threshold.
        """
        cfg = self.vision.cameras[0]
        K = cfg["intrinsic_matrix"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        h, w = depth.shape
        vs = np.arange(0, h, step)
        us = np.arange(0, w, step)
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]

        # Foreground = closer than the 10th-percentile of the depth image.
        foreground = z < np.percentile(depth, 10.0)
        z = z[foreground]
        uu = uu[foreground]
        vv = vv[foreground]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def demo_obstacle_detection(self, depth: np.ndarray) -> None:
        """Cluster scene obstacles from the captured depth frame.

        The deterministic, weight-free path back-projects foreground depth
        pixels and clusters them with ``Perception.cluster_obstacles`` (DBSCAN).
        YOLO-based detection is attempted as a supplement and reported when
        weights are available, but is never required.
        """
        self.logger.info("=== Obstacle Detection and Clustering ===")

        # Supplementary: YOLO + depth back-projection (optional, may be empty).
        yolo_detections = 0
        try:
            yolo_points, _ = self.perception.detect_and_cluster_obstacles(
                camera_index=0, depth_threshold=5.0, step=4, eps=0.1, min_samples=3
            )
            yolo_detections = int(len(yolo_points))
        except Exception as exc:
            self.logger.info("YOLO detection unavailable (%s)", exc)
        self.logger.info("YOLO contributed %d detection(s)", yolo_detections)

        # Primary: foreground depth point cloud + DBSCAN clustering.
        points = self._depth_pointcloud(depth, step=6)
        if len(points) > 0:
            labels, _ = self.perception.cluster_obstacles(
                points, eps=0.08, min_samples=5
            )
        else:
            labels = np.empty((0,), int)

        n_clusters = int(np.sum(np.unique(labels) != -1))
        n_noise = int(np.sum(labels == -1)) if len(labels) else 0
        self.logger.info(
            "Foreground depth cloud: %d points -> %d clusters, %d noise",
            len(points),
            n_clusters,
            n_noise,
        )

        self._plot_clusters(
            points, labels, title="Depth-Clustered Obstacles",
            filename="02_obstacle_clusters.png",
        )
        self.demo_results["obstacle_detection"] = {
            "yolo_detections": yolo_detections,
            "depth_points": int(len(points)),
            "clusters": n_clusters,
            "noise": n_noise,
        }

    def demo_clustering_api(self) -> None:
        """Exercise Perception.cluster_obstacles on a known synthetic cloud."""
        self.logger.info("=== DBSCAN Clustering API ===")
        rng = np.random.default_rng(42)
        centers = [[0.5, 0.2, 0.3], [0.8, -0.1, 0.4], [-0.3, 0.4, 0.5]]
        clouds = [rng.normal(c, 0.05, (40, 3)) for c in centers]
        noise = rng.uniform(-1, 1, (15, 3))
        points = np.vstack(clouds + [noise]).astype(np.float32)

        labels, n_clusters = self.perception.cluster_obstacles(
            points, eps=0.15, min_samples=5
        )
        n_noise = int(np.sum(labels == -1))
        self.logger.info(
            "Recovered %d/%d clusters (%d noise) from synthetic cloud",
            n_clusters,
            len(centers),
            n_noise,
        )

        self._plot_clusters(
            points, labels,
            title=f"DBSCAN: {n_clusters} clusters / {len(centers)} planted",
            filename="03_clustering_api.png",
        )
        self.demo_results["clustering_api"] = {
            "planted": len(centers),
            "found": int(n_clusters),
            "noise": n_noise,
        }

    # ----------------------------------------------------------------- stereo
    def demo_stereo_vision(self) -> None:
        """Stereo rectification, disparity and point-cloud reconstruction."""
        self.logger.info("=== Stereo Vision ===")
        if not self.vision.stereo_enabled:
            self.logger.warning("Stereo not enabled; skipping")
            return

        # Synthetic stereo pair: shapes shifted right->left to encode disparity.
        h, w = 480, 640
        left = np.zeros((h, w, 3), np.uint8)
        right = np.zeros((h, w, 3), np.uint8)
        shapes = [
            ((150, 150), (260, 260), 24),  # near (large disparity)
            ((380, 120), (470, 210), 12),  # mid
            ((250, 320), (340, 410), 6),   # far (small disparity)
        ]
        for (p1, p2, d) in shapes:
            cv2.rectangle(left, p1, p2, (255, 255, 255), -1)
            cv2.rectangle(right, (p1[0] - d, p1[1]), (p2[0] - d, p2[1]),
                          (255, 255, 255), -1)

        try:
            self.vision.compute_stereo_rectification_maps(image_size=(w, h))
            left_rect, right_rect = self.vision.rectify_stereo_images(left, right)
            disparity = self.vision.compute_disparity(left_rect, right_rect)
            cloud = self.vision.get_stereo_point_cloud(left, right)
        except Exception as exc:
            self.logger.warning("Stereo processing failed: %s", exc)
            self.demo_results["stereo_vision"] = {"error": str(exc)}
            return

        self.logger.info(
            "Disparity range %.1f-%.1f, point cloud %d points",
            float(disparity.min()),
            float(disparity.max()),
            len(cloud),
        )

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(left)
        ax1.set_title("Left image")
        ax1.axis("off")
        ax2 = fig.add_subplot(2, 2, 2)
        dd = ax2.imshow(disparity, cmap="jet")
        ax2.set_title("Disparity map")
        ax2.axis("off")
        fig.colorbar(dd, ax=ax2, fraction=0.046)
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        if len(cloud) > 0:
            sub = cloud[:: max(1, len(cloud) // 2000)]
            ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2],
                        cmap="viridis", s=2)
        ax3.set_title("Reconstructed point cloud")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax4 = fig.add_subplot(2, 2, 4)
        if len(cloud) > 0:
            ax4.hist(cloud[:, 2], bins=30, edgecolor="black", alpha=0.75)
        ax4.set_title("Depth (Z) distribution")
        ax4.set_xlabel("Z")
        ax4.set_ylabel("points")
        ax4.grid(True)
        plt.tight_layout()
        self._save_plot("04_stereo_vision.png")
        plt.close(fig)

        self.demo_results["stereo_vision"] = {
            "point_cloud_size": int(len(cloud)),
            "disparity_range": [float(disparity.min()), float(disparity.max())],
        }

    # ------------------------------------------------------------- plotting
    def _plot_clusters(
        self, points: np.ndarray, labels: np.ndarray, title: str, filename: str
    ) -> None:
        """Render a 3D + top-down scatter of clustered points."""
        fig = plt.figure(figsize=(12, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        if len(points) == 0:
            ax3d.text2D(0.5, 0.5, "No points", ha="center", va="center",
                        transform=ax3d.transAxes)
            ax2d.text(0.5, 0.5, "No points", ha="center", va="center",
                      transform=ax2d.transAxes)
        else:
            unique = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
            for i, lab in enumerate(unique):
                mask = labels == lab
                if lab == -1:
                    c, name = "black", "noise"
                else:
                    c, name = [colors[i]], f"cluster {lab}"
                ax3d.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                             c=c, s=18, alpha=0.8, label=name)
                ax2d.scatter(points[mask, 0], points[mask, 1], c=c, s=18,
                             alpha=0.8, label=name)
            ax3d.legend(loc="upper right", fontsize=8)
            ax2d.legend(loc="upper right", fontsize=8)

        ax3d.set_title(title)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        ax2d.set_title("Top-down view")
        ax2d.set_xlabel("X (m)")
        ax2d.set_ylabel("Y (m)")
        ax2d.grid(True)
        ax2d.axis("equal")
        plt.tight_layout()
        self._save_plot(filename)
        plt.close(fig)

    # ------------------------------------------------------------- lifecycle
    def cleanup(self) -> None:
        """Release vision/perception and disconnect PyBullet."""
        self.logger.info("Cleaning up resources...")
        try:
            if self.perception is not None:
                self.perception.release()
            elif self.vision is not None:
                self.vision.release()
            if self.physics_client is not None:
                pb.disconnect(self.physics_client)
                self.physics_client = None
        except Exception as exc:  # pragma: no cover - cleanup best effort
            self.logger.warning(f"Cleanup warning: {exc}")

    def run(self) -> Dict[str, dict]:
        """Run the full perception demonstration sequence."""
        self.logger.info("Starting ManipulaPy Perception Intermediate Demo")
        self.logger.info("=" * 60)
        start = time.time()
        try:
            if not self.setup_scene() or not self.setup_vision():
                self.logger.error("Setup failed; aborting demonstrations")
                return self.demo_results

            _, depth = self.demo_image_capture()
            if depth is not None:
                self.demo_obstacle_detection(depth)
            self.demo_clustering_api()
            self.demo_stereo_vision()

            elapsed = time.time() - start
            self.logger.info("Demo completed in %.2f s", elapsed)
            self.logger.info("Saved %d output file(s) to %s",
                             len(self.outputs_saved), self.script_dir)
            return self.demo_results
        finally:
            self.cleanup()
            plt.close("all")


def main() -> None:
    """Entry point: run the perception demo and print a compact summary."""
    print("ManipulaPy Perception Intermediate Demo")
    print("=" * 40)

    demo = PerceptionIntermediateDemo(save_outputs=True)
    results = demo.run()

    print("\n" + "=" * 40)
    print("DEMO SUMMARY")
    print("=" * 40)
    for name, data in results.items():
        print(f"  {name}: {data}")
    print(f"\nOutput files ({len(demo.outputs_saved)}):")
    for path in demo.outputs_saved:
        print(f"  - {os.path.basename(path)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
