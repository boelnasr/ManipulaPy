#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Path Resolver Module

Handles resolution of package:// URIs and relative paths in URDF files.
Supports ROS package paths, environment-based search, and configurable
package maps.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union

logger = logging.getLogger(__name__)


class PackageResolver:
    """
    Resolve package:// URIs and relative paths for URDF resources.

    Supports multiple resolution strategies:
    1. Explicit package map (package_name -> path)
    2. ROS package paths (via rospack or ament_index)
    3. Environment-based search paths
    4. Base path relative resolution

    Example:
        >>> resolver = PackageResolver()
        >>> resolver.add_package("ur_description", "/opt/ros/melodic/share/ur_description")
        >>> resolved = resolver.resolve("package://ur_description/meshes/ur5/visual/base.dae")
        "/opt/ros/melodic/share/ur_description/meshes/ur5/visual/base.dae"
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        package_map: Optional[Dict[str, Union[str, Path]]] = None,
        search_paths: Optional[List[Union[str, Path]]] = None,
        use_ros: bool = True,
    ):
        """
        Initialize PackageResolver.

        Args:
            base_path: Base directory for relative path resolution
            package_map: Dictionary mapping package names to paths
            search_paths: Additional directories to search for packages
            use_ros: Whether to use ROS package discovery (if available)
        """
        self.base_path = Path(base_path) if base_path else None
        self._package_map: Dict[str, Path] = {}
        self._search_paths: List[Path] = []
        self._use_ros = use_ros

        # Add initial package map
        if package_map:
            for name, path in package_map.items():
                self.add_package(name, path)

        # Add search paths
        if search_paths:
            for path in search_paths:
                self.add_search_path(path)

        # Add paths from environment
        self._init_from_environment()

    def _init_from_environment(self) -> None:
        """Initialize from environment variables."""
        # ROS package paths
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
        if ros_package_path:
            for path_str in ros_package_path.split(os.pathsep):
                path = Path(path_str)
                if path.exists():
                    self._search_paths.append(path)

        # Ament prefix path (ROS 2)
        ament_prefix_path = os.environ.get("AMENT_PREFIX_PATH", "")
        if ament_prefix_path:
            for prefix in ament_prefix_path.split(os.pathsep):
                share_path = Path(prefix) / "share"
                if share_path.exists():
                    self._search_paths.append(share_path)

        # Custom ManipulaPy package path
        manipulapy_package_path = os.environ.get("MANIPULAPY_PACKAGE_PATH", "")
        if manipulapy_package_path:
            for path_str in manipulapy_package_path.split(os.pathsep):
                path = Path(path_str)
                if path.exists():
                    self._search_paths.append(path)

        # Optional explicit package map (JSON file path or JSON string)
        package_map_env = os.environ.get("MANIPULAPY_PACKAGE_MAP", "")
        if package_map_env:
            map_data = None
            map_path = Path(package_map_env)
            if map_path.exists():
                try:
                    map_data = json.loads(map_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"MANIPULAPY_PACKAGE_MAP file is not valid JSON: {exc}"
                    )
            else:
                try:
                    map_data = json.loads(package_map_env)
                except json.JSONDecodeError:
                    logger.warning(
                        "MANIPULAPY_PACKAGE_MAP must be a JSON file path or JSON string."
                    )

            if isinstance(map_data, dict):
                for name, path in map_data.items():
                    self.add_package(name, path)
            elif map_data is not None:
                logger.warning(
                    "MANIPULAPY_PACKAGE_MAP must be a JSON object mapping package name to path."
                )

    def add_package(self, name: str, path: Union[str, Path]) -> None:
        """
        Add a package to the resolver.

        Args:
            name: Package name
            path: Path to package root directory
        """
        path = Path(path)
        if path.exists():
            self._package_map[name] = path
            logger.debug(f"Added package '{name}' at {path}")
        else:
            logger.warning(f"Package path does not exist: {path}")

    def add_search_path(self, path: Union[str, Path]) -> None:
        """
        Add a search path for package discovery.

        Args:
            path: Directory to search for packages
        """
        path = Path(path)
        if path.exists() and path not in self._search_paths:
            self._search_paths.append(path)
            logger.debug(f"Added search path: {path}")

    def resolve(self, uri: str) -> str:
        """
        Resolve a URI to an absolute file path.

        Handles:
        - package://package_name/path/to/file
        - file:///absolute/path
        - Relative paths
        - Absolute paths

        Args:
            uri: URI or path to resolve

        Returns:
            Resolved absolute path (or original if unresolvable)
        """
        if not uri:
            return uri

        # Handle package:// URIs
        if uri.startswith("package://"):
            return self._resolve_package_uri(uri)

        # Handle file:// URIs
        if uri.startswith("file://"):
            return uri[7:]

        # Handle absolute paths
        if Path(uri).is_absolute():
            return uri

        # Handle relative paths
        return self._resolve_relative_path(uri)

    def _resolve_package_uri(self, uri: str) -> str:
        """
        Resolve package:// URI.

        Args:
            uri: URI starting with package://

        Returns:
            Resolved absolute path
        """
        # Parse URI: package://package_name/relative/path
        path_part = uri[10:]  # Remove "package://"
        parts = path_part.split("/", 1)

        if len(parts) < 2:
            logger.warning(f"Invalid package URI: {uri}")
            return uri

        package_name, relative_path = parts

        # Try explicit package map first
        if package_name in self._package_map:
            resolved = self._package_map[package_name] / relative_path
            if resolved.exists():
                return str(resolved)

        # Try search paths
        for search_path in self._search_paths:
            # Try package_name as direct subdirectory
            candidate = search_path / package_name / relative_path
            if candidate.exists():
                self._package_map[package_name] = search_path / package_name
                return str(candidate)

            # Try without package name (for flat structures)
            candidate = search_path / relative_path
            if candidate.exists():
                return str(candidate)

        # Try ROS package discovery
        if self._use_ros:
            ros_path = self._find_ros_package(package_name)
            if ros_path:
                self._package_map[package_name] = ros_path
                resolved = ros_path / relative_path
                if resolved.exists():
                    return str(resolved)

        # Try base path as fallback
        if self.base_path:
            candidate = self.base_path / relative_path
            if candidate.exists():
                return str(candidate)

            candidate = self.base_path / package_name / relative_path
            if candidate.exists():
                return str(candidate)

        logger.warning(f"Could not resolve package URI: {uri}")
        return uri

    def _resolve_relative_path(self, path: str) -> str:
        """
        Resolve a relative path.

        Args:
            path: Relative path

        Returns:
            Resolved absolute path
        """
        # Try base path first
        if self.base_path:
            candidate = self.base_path / path
            if candidate.exists():
                return str(candidate)

        # Try search paths
        for search_path in self._search_paths:
            candidate = search_path / path
            if candidate.exists():
                return str(candidate)

        # Return as-is if not found
        return path

    def _find_ros_package(self, package_name: str) -> Optional[Path]:
        """
        Find a ROS package using rospack or ament_index.

        Args:
            package_name: Name of the package

        Returns:
            Path to package or None if not found
        """
        # Try ament_index_python (ROS 2)
        try:
            from ament_index_python.packages import get_package_share_directory

            return Path(get_package_share_directory(package_name))
        except (ImportError, Exception):
            pass

        # Try rospkg (ROS 1)
        try:
            import rospkg

            rospack = rospkg.RosPack()
            return Path(rospack.get_path(package_name))
        except (ImportError, Exception):
            pass

        # Try catkin_find (ROS 1)
        try:
            import subprocess

            result = subprocess.run(
                ["catkin_find", package_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return None

    def create_handler(self) -> Callable[[str], str]:
        """
        Create a filename handler function for the parser.

        Returns:
            Function that resolves URIs to paths
        """
        return self.resolve

    def list_packages(self) -> Dict[str, Path]:
        """
        List all known packages.

        Returns:
            Dictionary of package names to paths
        """
        return dict(self._package_map)

    def list_search_paths(self) -> List[Path]:
        """
        List all search paths.

        Returns:
            List of search paths
        """
        return list(self._search_paths)

    @classmethod
    def for_urdf(cls, urdf_path: Union[str, Path]) -> "PackageResolver":
        """
        Create a resolver configured for a specific URDF file.

        Automatically sets base_path to the URDF's directory and
        adds common relative package locations.

        Args:
            urdf_path: Path to the URDF file

        Returns:
            Configured PackageResolver
        """
        urdf_path = Path(urdf_path).resolve()
        base_path = urdf_path.parent

        resolver = cls(base_path=base_path)

        # Add parent directories as potential package roots
        # Common structures:
        # - package/urdf/robot.urdf -> package is 2 levels up
        # - package/robots/model.urdf -> package is 2 levels up
        for parent in [base_path.parent, base_path.parent.parent]:
            if parent.exists():
                resolver.add_search_path(parent)
                # If it looks like a package directory, add it
                if (parent / "package.xml").exists():
                    resolver.add_package(parent.name, parent)

        return resolver
