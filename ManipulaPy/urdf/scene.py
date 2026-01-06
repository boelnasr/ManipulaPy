#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Scene Module - Multi-Robot Support

Provides a Scene class for managing multiple robots in a shared workspace.
Supports base transforms, namespacing, and world-frame kinematics.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .types import Origin


@dataclass
class RobotInstance:
    """
    A robot instance within a scene.

    Contains a reference to the URDF and its placement in the world.
    """

    name: str
    urdf: "URDF"  # Forward reference
    base_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    namespace: str = ""

    def __post_init__(self):
        self.base_transform = np.asarray(self.base_transform, dtype=np.float64)
        if self.base_transform.shape != (4, 4):
            raise ValueError("base_transform must be a 4x4 matrix")

    @property
    def prefixed_link_names(self) -> List[str]:
        """Get link names with namespace prefix."""
        prefix = f"{self.namespace}/" if self.namespace else ""
        return [f"{prefix}{link.name}" for link in self.urdf.links]

    @property
    def prefixed_joint_names(self) -> List[str]:
        """Get joint names with namespace prefix."""
        prefix = f"{self.namespace}/" if self.namespace else ""
        return [f"{prefix}{joint.name}" for joint in self.urdf.joints]


class Scene:
    """
    A scene containing multiple robots with world-frame kinematics.

    Supports:
    - Multiple robots with different base transforms
    - Namespacing to avoid name collisions
    - World-frame forward kinematics
    - Collision geometry aggregation

    Example:
        >>> scene = Scene()
        >>> scene.add_robot("robot1", URDF.load("ur5.urdf"), base_xyz=[0, 0, 0])
        >>> scene.add_robot("robot2", URDF.load("ur5.urdf"), base_xyz=[1.5, 0, 0])
        >>> world_fk = scene.world_link_fk({"robot1": [0]*6, "robot2": [0]*6})
    """

    def __init__(self, name: str = "scene"):
        """
        Initialize an empty scene.

        Args:
            name: Scene name
        """
        self.name = name
        self._robots: Dict[str, RobotInstance] = {}

    def add_robot(
        self,
        name: str,
        urdf: "URDF",
        base_transform: Optional[np.ndarray] = None,
        base_xyz: Optional[Union[List[float], np.ndarray]] = None,
        base_rpy: Optional[Union[List[float], np.ndarray]] = None,
        namespace: Optional[str] = None,
    ) -> "Scene":
        """
        Add a robot to the scene.

        Args:
            name: Unique name for this robot instance
            urdf: URDF object for the robot
            base_transform: 4x4 base transformation matrix (overrides xyz/rpy)
            base_xyz: Base position [x, y, z]
            base_rpy: Base orientation [roll, pitch, yaw]
            namespace: Optional namespace prefix for links/joints

        Returns:
            self for method chaining
        """
        if name in self._robots:
            raise ValueError(f"Robot '{name}' already exists in scene")

        # Build base transform
        if base_transform is not None:
            T = np.asarray(base_transform, dtype=np.float64)
        else:
            origin = Origin(
                xyz=np.asarray(base_xyz or [0, 0, 0], dtype=np.float64),
                rpy=np.asarray(base_rpy or [0, 0, 0], dtype=np.float64),
            )
            T = origin.matrix

        self._robots[name] = RobotInstance(
            name=name,
            urdf=urdf,
            base_transform=T,
            namespace=namespace or name,
        )

        return self

    def remove_robot(self, name: str) -> "Scene":
        """
        Remove a robot from the scene.

        Args:
            name: Name of robot to remove

        Returns:
            self for method chaining
        """
        if name not in self._robots:
            raise ValueError(f"Robot '{name}' not found in scene")

        del self._robots[name]
        return self

    def get_robot(self, name: str) -> RobotInstance:
        """
        Get a robot instance by name.

        Args:
            name: Robot name

        Returns:
            RobotInstance
        """
        if name not in self._robots:
            raise ValueError(f"Robot '{name}' not found in scene")
        return self._robots[name]

    @property
    def robots(self) -> Dict[str, RobotInstance]:
        """Get all robots in the scene."""
        return dict(self._robots)

    @property
    def robot_names(self) -> List[str]:
        """Get names of all robots."""
        return list(self._robots.keys())

    def set_base_transform(
        self,
        name: str,
        base_transform: Optional[np.ndarray] = None,
        base_xyz: Optional[Union[List[float], np.ndarray]] = None,
        base_rpy: Optional[Union[List[float], np.ndarray]] = None,
    ) -> "Scene":
        """
        Update the base transform of a robot.

        Args:
            name: Robot name
            base_transform: 4x4 base transformation matrix
            base_xyz: Base position [x, y, z]
            base_rpy: Base orientation [roll, pitch, yaw]

        Returns:
            self for method chaining
        """
        if name not in self._robots:
            raise ValueError(f"Robot '{name}' not found in scene")

        if base_transform is not None:
            T = np.asarray(base_transform, dtype=np.float64)
        else:
            origin = Origin(
                xyz=np.asarray(base_xyz or [0, 0, 0], dtype=np.float64),
                rpy=np.asarray(base_rpy or [0, 0, 0], dtype=np.float64),
            )
            T = origin.matrix

        self._robots[name].base_transform = T
        return self

    def world_link_fk(
        self,
        configurations: Dict[str, Union[np.ndarray, Dict[str, float]]],
        links: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute world-frame forward kinematics for all robots.

        Args:
            configurations: {robot_name: joint_configuration}
            links: Optional {robot_name: [link_names]} to compute FK for

        Returns:
            {robot_name: {link_name: 4x4 world transform}}
        """
        result = {}

        for robot_name, robot in self._robots.items():
            cfg = configurations.get(robot_name)
            robot_links = links.get(robot_name) if links else None

            # Get FK in robot frame
            local_fk = robot.urdf.link_fk(cfg=cfg, links=robot_links, use_names=True)

            # Transform to world frame
            base_T = robot.base_transform
            world_fk = {}
            for link_name, T_local in local_fk.items():
                world_fk[link_name] = base_T @ T_local

            result[robot_name] = world_fk

        return result

    def world_end_effector_fk(
        self,
        configurations: Dict[str, Union[np.ndarray, Dict[str, float]]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute world-frame end effector positions for all robots.

        Args:
            configurations: {robot_name: joint_configuration}

        Returns:
            {robot_name: 4x4 world transform of end effector}
        """
        result = {}

        for robot_name, robot in self._robots.items():
            cfg = configurations.get(robot_name)

            # Get EE FK in robot frame
            ee_name = robot.urdf.end_effector_link.name
            local_fk = robot.urdf.link_fk(cfg=cfg, links=[ee_name], use_names=True)

            # Transform to world frame
            base_T = robot.base_transform
            result[robot_name] = base_T @ local_fk[ee_name]

        return result

    def get_all_collision_geometry(
        self,
        configurations: Optional[Dict[str, Union[np.ndarray, Dict[str, float]]]] = None,
    ) -> List[Dict]:
        """
        Get collision geometry for all robots in world frame.

        Args:
            configurations: Optional joint configurations

        Returns:
            List of collision geometry dictionaries with world transforms
        """
        geometries = []

        for robot_name, robot in self._robots.items():
            cfg = configurations.get(robot_name) if configurations else None

            # Get FK for all links
            local_fk = robot.urdf.link_fk(cfg=cfg, use_names=True)
            base_T = robot.base_transform

            for link in robot.urdf.links:
                link_T_world = base_T @ local_fk[link.name]

                for collision in link.collisions:
                    if collision.geometry is None:
                        continue

                    # Compute collision geometry world transform
                    collision_T_local = collision.origin.matrix
                    collision_T_world = link_T_world @ collision_T_local

                    geometries.append({
                        "robot": robot_name,
                        "link": link.name,
                        "geometry": collision.geometry,
                        "transform": collision_T_world,
                    })

        return geometries

    def get_all_visual_geometry(
        self,
        configurations: Optional[Dict[str, Union[np.ndarray, Dict[str, float]]]] = None,
    ) -> List[Dict]:
        """
        Get visual geometry for all robots in world frame.

        Args:
            configurations: Optional joint configurations

        Returns:
            List of visual geometry dictionaries with world transforms
        """
        geometries = []

        for robot_name, robot in self._robots.items():
            cfg = configurations.get(robot_name) if configurations else None

            # Get FK for all links
            local_fk = robot.urdf.link_fk(cfg=cfg, use_names=True)
            base_T = robot.base_transform

            for link in robot.urdf.links:
                link_T_world = base_T @ local_fk[link.name]

                for visual in link.visuals:
                    if visual.geometry is None:
                        continue

                    # Compute visual geometry world transform
                    visual_T_local = visual.origin.matrix
                    visual_T_world = link_T_world @ visual_T_local

                    geometries.append({
                        "robot": robot_name,
                        "link": link.name,
                        "geometry": visual.geometry,
                        "material": visual.material,
                        "transform": visual_T_world,
                    })

        return geometries

    def check_inter_robot_collision(
        self,
        configurations: Dict[str, Union[np.ndarray, Dict[str, float]]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Check for collisions between robots (not self-collision).

        This is a simple bounding box check. For accurate collision detection,
        use a dedicated collision library.

        Args:
            configurations: {robot_name: joint_configuration}

        Returns:
            List of collision pairs: [(robot1, link1, robot2, link2), ...]
        """
        geometries = self.get_all_collision_geometry(configurations)
        collisions = []

        # Simple O(n^2) pairwise check
        for i, geom1 in enumerate(geometries):
            for geom2 in geometries[i + 1:]:
                # Skip same robot (self-collision)
                if geom1["robot"] == geom2["robot"]:
                    continue

                # Simple bounding box overlap check
                if self._bboxes_overlap(geom1, geom2):
                    collisions.append((
                        geom1["robot"],
                        geom1["link"],
                        geom2["robot"],
                        geom2["link"],
                    ))

        return collisions

    def _bboxes_overlap(self, geom1: Dict, geom2: Dict) -> bool:
        """Simple axis-aligned bounding box overlap check."""
        # Get bounding boxes in world frame
        bbox1 = self._get_geometry_bbox(geom1["geometry"], geom1["transform"])
        bbox2 = self._get_geometry_bbox(geom2["geometry"], geom2["transform"])

        if bbox1 is None or bbox2 is None:
            return False

        # Check overlap
        min1, max1 = bbox1
        min2, max2 = bbox2

        return np.all(max1 >= min2) and np.all(max2 >= min1)

    def _get_geometry_bbox(
        self, geometry, transform: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get axis-aligned bounding box for geometry in world frame."""
        from .types import Box, Cylinder, Sphere, Mesh

        center = transform[:3, 3]

        if isinstance(geometry, Box):
            half_size = geometry.size / 2
            # Rotate half-size by transform rotation and take abs for AABB
            R = transform[:3, :3]
            half_extent = np.abs(R @ half_size)
            return center - half_extent, center + half_extent

        elif isinstance(geometry, Sphere):
            r = geometry.radius
            return center - r, center + r

        elif isinstance(geometry, Cylinder):
            # Conservative AABB for cylinder
            r = max(geometry.radius, geometry.length / 2)
            return center - r, center + r

        elif isinstance(geometry, Mesh):
            # If mesh has vertices, compute AABB
            if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                vertices = geometry.vertices
                # Transform vertices
                ones = np.ones((len(vertices), 1))
                homogeneous = np.hstack([vertices, ones])
                world_vertices = (transform @ homogeneous.T).T[:, :3]
                return world_vertices.min(axis=0), world_vertices.max(axis=0)

        return None

    @classmethod
    def from_urdfs(
        cls,
        urdfs: Dict[str, Union[str, Path, "URDF"]],
        base_transforms: Optional[Dict[str, np.ndarray]] = None,
        name: str = "scene",
    ) -> "Scene":
        """
        Create a scene from multiple URDFs.

        Args:
            urdfs: {robot_name: urdf_path_or_object}
            base_transforms: Optional {robot_name: 4x4_transform}
            name: Scene name

        Returns:
            Scene with all robots added
        """
        from .core import URDF

        scene = cls(name=name)

        for robot_name, urdf_source in urdfs.items():
            # Load URDF if path provided
            if isinstance(urdf_source, (str, Path)):
                urdf = URDF.load(urdf_source)
            else:
                urdf = urdf_source

            # Get base transform
            base_T = None
            if base_transforms and robot_name in base_transforms:
                base_T = base_transforms[robot_name]

            scene.add_robot(robot_name, urdf, base_transform=base_T)

        return scene

    def __repr__(self) -> str:
        robot_list = ", ".join(self.robot_names)
        return f"Scene(name='{self.name}', robots=[{robot_list}])"
