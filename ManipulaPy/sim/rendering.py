"""Rendering concern definitions for the simulation class."""

# ruff: noqa: F841, SIM108, UP006, UP035 - moved bodies are intentionally exact.

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import _runtime


class _RenderingConcern:
    def _capsule_line(
        self,
        a: Sequence[float],
        b: Sequence[float],
        radius: float = 0.006,
        rgba: Sequence[float] = (1, 0.5, 0, 1),
    ) -> int:
        """
        Create a thin capsule between point a and b; returns body-id.
        This creates REAL GEOMETRY that appears in getCameraImage() screenshots.

        Args:
            a: Start point [x, y, z]
            b: End point [x, y, z]
            radius: Capsule radius in world units
            rgba: Color as [r, g, b, a] where values are 0-1

        Returns:
            int: PyBullet body ID, or -1 if failed
        """
        a, b = np.array(a), np.array(b)
        v = b - a
        L = np.linalg.norm(v)

        if L < 1e-6:
            return -1

        # Calculate orientation to align capsule with the line direction
        z = v / L  # Direction vector

        # Find perpendicular vectors
        x = np.cross([0, 0, 1], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([0, 1, 0], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        # Calculate proper orientation for capsule
        # PyBullet capsules are aligned with Z-axis by default
        if abs(z[2]) > 0.99:  # Nearly vertical
            orn = _runtime.p.getQuaternionFromEuler([0, 0, 0])
        else:
            # Calculate rotation to align Z-axis with direction vector
            angle = np.arccos(np.clip(z[2], -1, 1))
            if angle > 1e-6:
                axis = np.cross([0, 0, 1], z)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    # Inline axis-angle to quaternion to avoid relying on
                    # _runtime.p.getQuaternionFromAxisAngle, which is missing from
                    # several pybullet builds (cross-version portability).
                    half = angle / 2.0
                    s = np.sin(half)
                    orn = (axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half))
                else:
                    orn = _runtime.p.getQuaternionFromEuler([0, 0, 0])
            else:
                orn = _runtime.p.getQuaternionFromEuler([0, 0, 0])

        # Midpoint of the line segment
        mid = (a + b) / 2

        try:
            # Create collision and visual shapes
            col = _runtime.p.createCollisionShape(
                _runtime.p.GEOM_CAPSULE, radius=radius, height=L
            )
            vis = _runtime.p.createVisualShape(
                _runtime.p.GEOM_CAPSULE, radius=radius, length=L, rgbaColor=rgba
            )

            # Create static body (mass=0)
            body_id = _runtime.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=mid,
                baseOrientation=orn,
            )

            return body_id

        except Exception as e:
            self.logger.error(f"Failed to create capsule line: {e}")
            return -1

    def plot_trajectory(
        self,
        ee_positions: Sequence[Sequence[float]],
        line_width: int = 3,
        color: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Plots the end-effector trajectory in PyBullet using REAL GEOMETRY.

        This method now creates actual 3D capsules that will appear in screenshots
        taken with getCameraImage(), unlike the previous addUserDebugLine() approach.

        Args:
            ee_positions: List of end-effector positions [[x,y,z], ...]
            line_width: Width factor for trajectory visualization
            color: RGB color as [r, g, b] where values are 0-1

        Returns:
            list: Body IDs of created trajectory geometry (for cleanup)
        """
        _runtime._check_pybullet_available()
        # Clear any existing trajectory bodies
        self.clear_trajectory_visualization()

        if len(ee_positions) < 2:
            self.logger.warning("Not enough positions to plot trajectory")
            return []

        if color is None:
            color = [1, 0, 0]

        # Convert color to RGBA
        if len(color) == 3:
            rgba_color = color + [1.0]  # Add alpha
        else:
            rgba_color = color

        # Calculate radius based on line_width (convert to world scale)
        base_radius = 0.003  # Base radius in world units
        radius = base_radius * (line_width / 3.0)  # Scale with line_width

        trajectory_bodies = []

        self.logger.info(
            f"Creating trajectory visualization with {len(ee_positions)} points"
        )

        # Create capsule segments between consecutive points
        for i in range(1, len(ee_positions)):
            try:
                # Get consecutive points
                start_pos = ee_positions[i - 1]
                end_pos = ee_positions[i]

                # Create multiple parallel capsules for thickness effect
                for j in range(max(1, line_width // 2)):
                    # Slight offset for thickness
                    offset = j * 0.002  # Small offset in world units

                    start_offset = [start_pos[0] + offset, start_pos[1], start_pos[2]]
                    end_offset = [end_pos[0] + offset, end_pos[1], end_pos[2]]

                    # Create capsule segment
                    body_id = self._capsule_line(
                        start_offset, end_offset, radius=radius, rgba=rgba_color
                    )

                    if body_id != -1:
                        trajectory_bodies.append(body_id)

            except Exception as e:
                self.logger.error(f"Failed to create trajectory segment {i}: {e}")

        # Store body IDs for cleanup
        self.trajectory_body_ids.extend(trajectory_bodies)

        # Add trajectory markers
        marker_bodies = self._add_trajectory_markers(ee_positions, rgba_color)
        self.trajectory_body_ids.extend(marker_bodies)

        self.logger.info(
            f"✅ Created trajectory visualization: {len(trajectory_bodies)} segments + {len(marker_bodies)} markers"
        )
        self.logger.info("🎯 Trajectory will now appear in screenshots as 3D geometry!")

        return trajectory_bodies

    def _add_trajectory_markers(
        self, ee_positions: Sequence[Sequence[float]], color: Sequence[float]
    ) -> List[int]:
        """
        Add START/END markers using real geometry.

        Args:
            ee_positions: List of end-effector positions
            color: RGBA color for markers

        Returns:
            list: Body IDs of created markers
        """
        marker_bodies = []

        try:
            # START marker (green sphere)
            start_visual = _runtime.p.createVisualShape(
                shapeType=_runtime.p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0.0, 1.0, 0.0, 1.0],  # Green
            )
            start_collision = _runtime.p.createCollisionShape(
                shapeType=_runtime.p.GEOM_SPHERE, radius=0.02
            )
            start_marker = _runtime.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=start_collision,
                baseVisualShapeIndex=start_visual,
                basePosition=[
                    ee_positions[0][0],
                    ee_positions[0][1],
                    ee_positions[0][2] + 0.05,
                ],
            )
            marker_bodies.append(start_marker)

            # END marker (red sphere)
            end_visual = _runtime.p.createVisualShape(
                shapeType=_runtime.p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],  # Red
            )
            end_collision = _runtime.p.createCollisionShape(
                shapeType=_runtime.p.GEOM_SPHERE, radius=0.02
            )
            end_marker = _runtime.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=end_collision,
                baseVisualShapeIndex=end_visual,
                basePosition=[
                    ee_positions[-1][0],
                    ee_positions[-1][1],
                    ee_positions[-1][2] + 0.05,
                ],
            )
            marker_bodies.append(end_marker)

            # Add intermediate waypoints if trajectory is long enough
            if len(ee_positions) > 10:
                waypoint_indices = [
                    len(ee_positions) // 4,
                    len(ee_positions) // 2,
                    3 * len(ee_positions) // 4,
                ]

                for idx in waypoint_indices:
                    if 0 <= idx < len(ee_positions):
                        waypoint_visual = _runtime.p.createVisualShape(
                            shapeType=_runtime.p.GEOM_SPHERE,
                            radius=0.015,
                            rgbaColor=[0.0, 0.0, 1.0, 1.0],  # Blue
                        )
                        waypoint_collision = _runtime.p.createCollisionShape(
                            shapeType=_runtime.p.GEOM_SPHERE, radius=0.015
                        )
                        waypoint_marker = _runtime.p.createMultiBody(
                            baseMass=0,
                            baseCollisionShapeIndex=waypoint_collision,
                            baseVisualShapeIndex=waypoint_visual,
                            basePosition=[
                                ee_positions[idx][0],
                                ee_positions[idx][1],
                                ee_positions[idx][2] + 0.03,
                            ],
                        )
                        marker_bodies.append(waypoint_marker)

        except Exception as e:
            self.logger.error(f"Failed to create trajectory markers: {e}")

        return marker_bodies

    def clear_trajectory_visualization(self) -> None:
        """
        Clear all trajectory visualization bodies from the simulation.
        """
        _runtime._check_pybullet_available()
        if hasattr(self, "trajectory_body_ids"):
            removed_count = 0
            for body_id in self.trajectory_body_ids:
                try:
                    _runtime.p.removeBody(body_id)
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(
                        f"Could not remove trajectory body {body_id}: {e}"
                    )

            if removed_count > 0:
                self.logger.info(
                    f"🧹 Removed {removed_count} trajectory visualization bodies"
                )

            self.trajectory_body_ids = []

    def plot_trajectory_in_scene(
        self,
        joint_trajectory: Sequence[Sequence[float]],
        end_effector_trajectory: Sequence[Sequence[float]],
    ) -> None:
        """
        Plots the trajectory in the simulation scene.

        Renders the end-effector path as a 3-D Matplotlib line plot, then replays
        the joint trajectory in the PyBullet simulation.

        Args:
            joint_trajectory: Sequence of joint-angle configurations (one per
                simulation step) to replay, each a sequence of joint angles in
                radians.
            end_effector_trajectory: Sequence of end-effector positions to plot,
                each an (x, y, z) world-frame coordinate.
        """
        _runtime._check_pybullet_available()
        self.logger.info("Plotting trajectory in simulation scene...")
        ee_positions = np.array(end_effector_trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            label="End-Effector Trajectory",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()

        self.run_trajectory(joint_trajectory)
        self.logger.info("Trajectory plotted and simulation completed.")
