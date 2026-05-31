#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced Stereo Vision Demo - ManipulaPy

This demo exercises the full ManipulaPy 3D-perception pipeline on a synthetic,
fully reproducible stereo scene (no cameras, no GPU, no YOLO weights required):

  1. Configure a calibrated stereo rig (left/right intrinsics + baseline) and
     build OpenCV rectification maps via ``Vision.compute_stereo_rectification_maps``.
  2. Synthesize a rectified left/right image pair containing several textured
     foreground "obstacle" blobs at known depths.
  3. Recover geometry with the real library calls:
       - ``Vision.compute_disparity``        (StereoSGBM)
       - ``Vision.disparity_to_pointcloud``  (reproject via the Q matrix)
       - ``Vision.get_stereo_point_cloud``   (one-shot rectify->disparity->3D)
  4. Segment the reconstructed point cloud into discrete obstacles with
     ``Perception.cluster_obstacles`` (DBSCAN) and report per-cluster centroids.
  5. Demonstrate optional YOLO obstacle detection through ``Vision.detect_obstacles``,
     degrading gracefully to the synthetic pipeline when weights/cameras are absent.

Every external dependency (CUDA, PyBullet cameras, YOLO weights, scikit-learn)
is guarded; the demo always runs to completion on a headless CPU.

Usage:
    NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg python stereo_vision_advanced_demo.py

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - OpenCV ships with ManipulaPy
    print(f"OpenCV is required for this demo: {exc}")
    raise SystemExit(1)

try:
    from ManipulaPy.vision import Vision
    from ManipulaPy.perception import Perception
except ImportError as exc:  # pragma: no cover
    print(f"Error importing ManipulaPy modules: {exc}")
    raise SystemExit(1)

# Keep all artifacts next to this script and make synthetic data reproducible.
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)

IMAGE_SIZE = (640, 480)  # (width, height)
FOCAL = 600.0  # px
BASELINE = 0.10  # m, left-to-right camera offset along +x


# ---------------------------------------------------------------------------
# Stereo rig configuration
# ---------------------------------------------------------------------------
def build_stereo_configs() -> Tuple[Dict, Dict]:
    """Return (left_cfg, right_cfg) for a horizontally-offset calibrated rig.

    Both cameras share intrinsics; the right camera is translated by ``BASELINE``
    metres along +x, which is exactly the geometry stereo matching assumes.
    """
    w, h = IMAGE_SIZE
    K = np.array(
        [[FOCAL, 0.0, w / 2.0], [0.0, FOCAL, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist = np.zeros(5, dtype=np.float32)
    left_cfg = {
        "name": "left",
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "intrinsic_matrix": K,
        "distortion_coeffs": dist,
        "fov": 60,
        "near": 0.1,
        "far": 10.0,
    }
    right_cfg = dict(left_cfg)
    right_cfg["name"] = "right"
    right_cfg["translation"] = [BASELINE, 0.0, 0.0]
    return left_cfg, right_cfg


# ---------------------------------------------------------------------------
# Synthetic rectified stereo pair
# ---------------------------------------------------------------------------
def synthesize_stereo_pair(
    obstacles: List[Tuple[float, float, float, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a textured left/right image pair from a list of obstacle blobs.

    Each obstacle is ``(X, Y, Z, radius_px)`` in the left-camera frame. The same
    blob is drawn into both images; in the right image it is shifted left by the
    stereo disparity ``d = FOCAL * BASELINE / Z`` (px), so a correct matcher will
    recover the planted depth ``Z``.

    Returns BGR uint8 images of shape (H, W, 3).
    """
    w, h = IMAGE_SIZE
    cx, cy = w / 2.0, h / 2.0

    # Mild background texture so StereoSGBM has features to lock onto.
    base = RNG.integers(40, 70, size=(h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0)
    left = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    right = left.copy()

    for (X, Y, Z, radius) in obstacles:
        u = int(round(cx + FOCAL * X / Z))
        v = int(round(cy + FOCAL * Y / Z))
        disparity = FOCAL * BASELINE / Z  # px shift between left and right
        u_right = int(round(u - disparity))

        # A filled disc plus a speckle pattern keeps the block matcher confident.
        color = tuple(int(c) for c in RNG.integers(150, 240, size=3))
        cv2.circle(left, (u, v), radius, color, -1)
        cv2.circle(right, (u_right, v), radius, color, -1)
        for img, uc in ((left, u), (right, u_right)):
            speckle = RNG.integers(0, 60, size=(2 * radius, 2 * radius, 3), dtype=np.uint8)
            y0, x0 = v - radius, uc - radius
            y1, x1 = y0 + 2 * radius, x0 + 2 * radius
            if 0 <= y0 and y1 <= h and 0 <= x0 and x1 <= w:
                roi = img[y0:y1, x0:x1].astype(np.int16) + speckle.astype(np.int16) - 30
                img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

    return left, right


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_pipeline_figure(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    cloud: np.ndarray,
    labels: np.ndarray,
    out_path: str,
) -> None:
    """Save a 2x2 panel: stereo pair, disparity map, and top-down clustered cloud."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Left image")
    axes[0, 1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Right image")

    disp_vis = np.ma.masked_less_equal(disparity, 0.0)
    im = axes[1, 0].imshow(disp_vis, cmap="plasma")
    axes[1, 0].set_title("Disparity (px)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    ax = axes[1, 1]
    if cloud.shape[0] > 0:
        cmap = plt.get_cmap("tab10")
        for lab in np.unique(labels):
            mask = labels == lab
            col = "0.6" if lab == -1 else cmap(int(lab) % 10)
            # Plot |Z| so the depth axis reads as positive metres.
            ax.scatter(cloud[mask, 0], np.abs(cloud[mask, 2]), s=4, color=col,
                       label=("noise" if lab == -1 else f"obstacle {lab}"))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z / depth (m)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Clustered point cloud (top-down)")

    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional YOLO probe (graceful)
# ---------------------------------------------------------------------------
def try_yolo_detection(vision: Vision, rgb: np.ndarray) -> None:
    """Attempt YOLO-based obstacle detection; report and continue if unavailable.

    ``detect_obstacles`` needs a configured PyBullet camera AND YOLO weights, and
    pins to CPU here to avoid CUDA aborts. Any failure degrades to a printed note.
    """
    os.environ.setdefault("MANIPULAPY_YOLO_DEVICE", "cpu")
    print("\n[5] Optional YOLO obstacle detection")
    try:
        depth = np.full(rgb.shape[:2], 2.0, dtype=np.float32)
        positions, orientations = vision.detect_obstacles(
            depth_image=depth, rgb_image=rgb, depth_threshold=5.0, camera_index=0
        )
        if positions.shape[0] == 0:
            print("    YOLO available but no objects passed the depth filter "
                  "(expected for a synthetic scene).")
        else:
            print(f"    YOLO detected {positions.shape[0]} object(s); "
                  f"first centroid ~ {positions[0].round(3)} m")
    except Exception as exc:  # weights/cameras absent -> graceful skip
        print(f"    YOLO path unavailable ({type(exc).__name__}); "
              "synthetic stereo pipeline already covers obstacle geometry.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the synthetic stereo reconstruction + clustering pipeline end to end."""
    print("=" * 64)
    print("ManipulaPy: Advanced Stereo Vision Demo (headless / CPU-safe)")
    print("=" * 64)

    # Planted obstacles: (X, Y, Z, radius_px) in the left-camera frame.
    obstacles = [
        (-0.30, 0.00, 1.5, 34),
        (0.35, -0.10, 2.5, 30),
        (0.05, 0.20, 4.0, 22),
    ]
    print("\n[1] Scene: 3 obstacles planted at depths "
          f"{[o[2] for o in obstacles]} m (baseline {BASELINE} m, f={FOCAL:.0f}px)")

    left_cfg, right_cfg = build_stereo_configs()
    vision = Vision(
        camera_configs=[left_cfg],
        stereo_configs=(left_cfg, right_cfg),
        show_plot=False,
    )
    print("[2] Stereo Vision configured -> "
          f"stereo_enabled={vision.stereo_enabled}, matcher={type(vision.stereo_matcher).__name__}")

    # Build rectification maps + Q reprojection matrix (real OpenCV calibration).
    vision.compute_stereo_rectification_maps(image_size=IMAGE_SIZE)
    assert vision.Q is not None, "stereoRectify did not produce a Q matrix"

    # Synthesize a rectified pair (the rig is aligned, so images are already rectified).
    left_img, right_img = synthesize_stereo_pair(obstacles)

    # --- Real disparity + reprojection through the library ---
    left_rect, right_rect = vision.rectify_stereo_images(left_img, right_img)
    disparity = vision.compute_disparity(left_rect, right_rect)
    valid = disparity[disparity > 0]
    print(f"[3] Disparity computed: {valid.size} valid px, "
          f"range [{(valid.min() if valid.size else 0):.1f}, "
          f"{(valid.max() if valid.size else 0):.1f}] px")

    # One-shot high-level pipeline (rectify -> disparity -> 3D) for the cloud.
    cloud = vision.get_stereo_point_cloud(left_img, right_img)
    print(f"    Reconstructed point cloud: {cloud.shape[0]} 3D points")

    # --- Cluster the cloud into discrete obstacles with DBSCAN ---
    print("\n[4] Obstacle segmentation via Perception.cluster_obstacles (DBSCAN)")
    perception = Perception(vision_instance=vision)
    labels = np.array([])
    if cloud.shape[0] > 0:
        # Downsample for speed; clustering scales with point count.
        if cloud.shape[0] > 6000:
            idx = RNG.choice(cloud.shape[0], 6000, replace=False)
            cloud = cloud[idx]
        try:
            labels, n_clusters = perception.cluster_obstacles(
                cloud, eps=0.15, min_samples=20
            )
            print(f"    Found {n_clusters} cluster(s) "
                  f"({np.sum(labels == -1)} noise points)")
            for lab in sorted(set(labels) - {-1}):
                centroid = cloud[labels == lab].mean(axis=0)
                # OpenCV's reprojectImageTo3D returns Z with a sign set by the
                # baseline-direction convention; |Z| is the metric depth.
                depth_m = abs(centroid[2])
                print(f"      cluster {lab}: centroid X={centroid[0]:+.2f} "
                      f"Y={centroid[1]:+.2f} depth={depth_m:.2f} m "
                      f"({np.sum(labels == lab)} pts)")
        except ImportError as exc:
            print(f"    scikit-learn unavailable ({exc}); skipping clustering.")
            labels = np.full(cloud.shape[0], -1)
    else:
        print("    Empty cloud; nothing to cluster.")

    # --- Optional YOLO probe (always graceful) ---
    try_yolo_detection(vision, left_img)

    # --- Save the pipeline figure ---
    out_path = os.path.join(HERE, "stereo_vision_pipeline.png")
    save_pipeline_figure(left_img, right_img, disparity, cloud, labels, out_path)
    print(f"\n[6] Saved pipeline figure -> {out_path}")

    vision.release()
    print("\nStereo vision demo complete.")


if __name__ == "__main__":
    main()
