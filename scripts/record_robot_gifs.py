#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Record robot GIFs for README.md using ManipulaPy's URDF parser +
PyBullet headless renderer.

Produces:

    ur5_pick_motion.gif   - UR5 executing a quintic-timed reach trajectory
    robot_zoo.gif         - 4 robots (UR5, Panda, iiwa14, xArm6) holding their
                            home poses, camera orbiting

Both are rendered through PyBullet's DIRECT mode (no X server, no GUI window
required) and captured via p.getCameraImage. Stitched into GIFs with imageio.

Run after a release to refresh README visuals:

    python scripts/record_robot_gifs.py

Sizes are tuned for GitHub embed (each GIF is <3 MB at the listed resolution).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
from ManipulaPy.urdf import URDF, PackageResolver


def _pybullet_compatible_urdf(urdf_path: Path) -> Path:
    """Rewrite a URDF with every package:// URI swapped for the absolute
    mesh path that ManipulaPy.urdf.PackageResolver resolves.
    PyBullet's loader has no notion of ROS packages, so this is the cleanest
    way to render bundled robots end-to-end. If the URDF has no package://
    URIs (e.g. xarm6 uses raw relative paths) the original is returned so
    PyBullet keeps resolving relative paths from the URDF's directory."""
    text = Path(urdf_path).read_text(encoding="utf-8")
    if "package://" not in text:
        return urdf_path

    resolver = PackageResolver.for_urdf(urdf_path)

    def _swap(match: re.Match[str]) -> str:
        uri = match.group(1)
        resolved = resolver._resolve_package_uri(uri)
        if not resolved or resolved == uri:
            return match.group(0)
        return f'filename="{resolved}"'

    rewritten = re.sub(r'filename="(package://[^"]+)"', _swap, text)

    tmp = Path(tempfile.mkstemp(suffix=".urdf")[1])
    tmp.write_text(rewritten, encoding="utf-8")
    return tmp

# Force matplotlib-free, fully headless. Without this PyBullet can still try
# to open an X11 connection on some systems when capturing images.
os.environ.setdefault("DISPLAY", "")

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "gifs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 480, 360


def _connect():
    """Headless PyBullet client. Returns the client id."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    return client


def _disconnect():
    p.disconnect()


def _movable_joints(robot_id):
    """Return the list of non-fixed joint indices (matches ManipulaPy.sim)."""
    return [
        i for i in range(p.getNumJoints(robot_id))
        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED
    ]


def _set_pose(robot_id, joints, q):
    for idx, angle in zip(joints, q):
        p.resetJointState(robot_id, idx, float(angle))


def _capture(view_target, distance, yaw, pitch=-20.0):
    """Take a single RGB frame from the configured viewpoint."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=view_target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55.0, aspect=IMG_W / IMG_H, nearVal=0.05, farVal=10.0,
    )
    _, _, rgba, *_ = p.getCameraImage(
        width=IMG_W, height=IMG_H,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3].copy()


def record_ur5_pick_motion():
    """UR5 executes a quintic-timed reach via OptimizedTrajectoryPlanning."""
    _connect()
    try:
        p.loadURDF("plane.urdf")
        urdf_path = get_robot_urdf("ur5")
        # Parse with our native parser (regression coverage — confirms the
        # planner and the rendered model agree on the same URDF).
        URDF.load(urdf_path)
        urdf_for_pb = _pybullet_compatible_urdf(Path(urdf_path))
        robot_id = p.loadURDF(str(urdf_for_pb), [0, 0, 0], useFixedBase=True)
        joints = _movable_joints(robot_id)
        n = len(joints)

        # Build a planner-driven trajectory just like the README demo does.
        urdf_obj = URDF.load(urdf_path)
        serial = urdf_obj.to_serial_manipulator()
        dynamics = urdf_obj.to_manipulator_dynamics()
        planner = OptimizedTrajectoryPlanning(
            serial, urdf_path, dynamics,
            joint_limits=[(-np.pi, np.pi)] * n,
        )
        traj = planner.joint_trajectory(
            thetastart=np.zeros(n),
            thetaend=np.array([0.6, -0.7, 1.0, -0.6, 1.3, 0.0])[:n],
            Tf=4.0, N=100, method=5,
        )
        positions = np.asarray(traj["positions"])

        frames = []
        for i, q in enumerate(positions):
            _set_pose(robot_id, joints, q)
            yaw = 35 + 60.0 * (i / len(positions))   # slow orbit
            frames.append(_capture([0, 0, 0.35], distance=1.6, yaw=yaw))

        out = OUT_DIR / "ur5_pick_motion.gif"
        imageio.mimsave(out, frames, format="GIF", fps=20, loop=0)
        return out
    finally:
        _disconnect()


def _safe_home_pose(n_joints):
    """A small, safe initial pose that shows off arm structure without
    self-colliding on most arms."""
    pose = np.zeros(n_joints)
    if n_joints >= 2:
        pose[1] = -0.6
    if n_joints >= 3:
        pose[2] = 0.6
    if n_joints >= 5:
        pose[4] = 0.5
    return pose


def record_robot_zoo():
    """Side-by-side strip of four robots holding a clean pose, camera
    orbiting each in turn. Lives as one continuous GIF rather than four
    separate files — easier README embed."""
    robot_keys = ["ur5", "panda", "iiwa14", "xarm6"]
    frames_per_robot = 30
    all_frames = []

    for key in robot_keys:
        _connect()
        try:
            p.loadURDF("plane.urdf")
            urdf_path = Path(get_robot_urdf(key))
            URDF.load(urdf_path)                              # smoke-test parse
            # Search paths so PyBullet can resolve relative mesh refs
            # (e.g. xarm6 references xarm_description/meshes/... directly).
            for parent in (urdf_path.parent, *urdf_path.parents[:3]):
                p.setAdditionalSearchPath(str(parent))
            urdf_for_pb = _pybullet_compatible_urdf(urdf_path)
            robot_id = p.loadURDF(str(urdf_for_pb), [0, 0, 0], useFixedBase=True)
            joints = _movable_joints(robot_id)
            _set_pose(robot_id, joints, _safe_home_pose(len(joints)))

            for i in range(frames_per_robot):
                yaw = i * (360 / frames_per_robot)
                frame = _capture([0, 0, 0.4], distance=1.6, yaw=yaw)
                # Stamp the robot name on each frame so the GIF self-narrates.
                from PIL import Image, ImageDraw, ImageFont
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        20,
                    )
                except OSError:
                    font = ImageFont.load_default()
                draw.text((12, 10), key, fill=(20, 20, 20), font=font)
                all_frames.append(np.asarray(img))
        finally:
            _disconnect()

    out = OUT_DIR / "robot_zoo.gif"
    imageio.mimsave(out, all_frames, format="GIF", fps=15, loop=0)
    return out


def main():
    print("Recording ur5_pick_motion.gif...")
    p1 = record_ur5_pick_motion()
    print(f"  -> {p1}  ({p1.stat().st_size / 1024:.0f} KB)")

    print("Recording robot_zoo.gif...")
    p2 = record_robot_zoo()
    print(f"  -> {p2}  ({p2.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
