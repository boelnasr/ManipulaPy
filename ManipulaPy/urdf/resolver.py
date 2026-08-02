#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Path Resolver Module

Handles resolution of package:// URIs and relative paths in URDF files.
Supports ROS package paths, environment-based search, and configurable
package maps.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import json
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse
from urllib.request import url2pathname
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

_PACKAGE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")
_MAX_DISCOVERY_UPWARD_DEPTH = 16
_MAX_DISCOVERY_DOWNWARD_DEPTH = 8
_MAX_DISCOVERY_CANDIDATES = 256
_MAX_DISCOVERY_DIRECTORIES = 4096
_MAX_DISCOVERY_ENTRIES = 16384
_MAX_DISCOVERY_BOUNDARIES = 64


def _path_escapes_base(candidate: Path, base: Path) -> bool:
    """Return True if ``candidate`` resolves outside ``base``.

    Guards against symlink/traversal escapes for relative mesh references:
    a resolved path that is not contained within the resolved base directory
    must be refused. Unresolvable paths are treated as escaping.
    """
    try:
        candidate.resolve().relative_to(base.resolve())
        return False
    except (ValueError, OSError, RuntimeError):
        return True


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
        >>> resolver.add_package(
        ...     "ur_description", "/opt/ros/melodic/share/ur_description"
        ... )
        >>> resolved = resolver.resolve(
        ...     "package://ur_description/meshes/ur5/visual/base.dae"
        ... )
        "/opt/ros/melodic/share/ur_description/meshes/ur5/visual/base.dae"
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        package_map: Optional[Dict[str, Union[str, Path]]] = None,
        search_paths: Optional[List[Union[str, Path]]] = None,
        use_ros: bool = True,
    ) -> None:
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
        self._deep_search_paths: List[Path] = []
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
        # ROS package paths — only when ROS discovery is enabled, otherwise
        # use_ros=False would still leak host ROS packages via search paths.
        if self._use_ros:
            ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
            if ros_package_path:
                for path_str in ros_package_path.split(os.pathsep):
                    path = Path(path_str)
                    if path.exists():
                        self._add_direct_search_path(path)

            ament_prefix_path = os.environ.get("AMENT_PREFIX_PATH", "")
            if ament_prefix_path:
                for prefix in ament_prefix_path.split(os.pathsep):
                    share_path = Path(prefix) / "share"
                    if share_path.exists():
                        self._add_direct_search_path(share_path)

        # The custom ManipulaPy path is an explicit user-controlled discovery
        # root, so unlike ambient ROS/Ament paths it permits bounded indexing.
        manipulapy_package_path = os.environ.get("MANIPULAPY_PACKAGE_PATH", "")
        if manipulapy_package_path:
            for path_str in manipulapy_package_path.split(os.pathsep):
                path = Path(path_str)
                if path.exists():
                    self.add_search_path(path)

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
                        "MANIPULAPY_PACKAGE_MAP must be a JSON file path "
                        "or JSON string."
                    )

            if isinstance(map_data, dict):
                for name, path in map_data.items():
                    self.add_package(name, path)
            elif map_data is not None:
                logger.warning(
                    "MANIPULAPY_PACKAGE_MAP must be a JSON object mapping "
                    "package name to path."
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
        if path.exists():
            self._add_direct_search_path(path)
            if path not in self._deep_search_paths:
                self._deep_search_paths.append(path)
            logger.debug(f"Added search path: {path}")

    def _add_direct_search_path(self, path: Path) -> None:
        """Add a permitted direct-lookup root without enabling recursive scan."""
        if path not in self._search_paths:
            self._search_paths.append(path)

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

        # Handle file:// URIs (use url2pathname so file:///C:/... works on Windows)
        if uri.startswith("file://"):
            return url2pathname(urlparse(uri).path)

        # Handle absolute paths
        if Path(uri).is_absolute():
            return uri

        # Handle relative paths
        return self._resolve_relative_path(uri)

    def _resolve_package_uri(self, uri: str) -> Optional[str]:
        """Resolve package://pkg/path with ambiguity detection.

        Strategy 1 (explicit ``add_package`` mapping) short-circuits — the
        documented escape hatch must always win. Strategies 2-5 (search paths,
        ROS lookup, base path, ancestor heuristic) collect candidates whose
        canonical (symlink-resolved) paths are deduped before the ambiguity
        check, so symlinked or case-insensitive workspaces don't trigger false
        ambiguity.
        """
        if not uri.startswith("package://"):
            return uri
        rest = uri[len("package://") :]
        if not rest:
            logger.warning(f"Malformed package URI {uri!r}: missing package name")
            return uri
        parts = rest.split("/", 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            logger.warning(
                f"Malformed package URI {uri!r}: expected 'package://<name>/<path>'"
            )
            return uri
        package_name, relative_path = parts[0], parts[1]

        if _PACKAGE_NAME_RE.fullmatch(package_name) is None:
            logger.warning(
                "Refusing to resolve %r: invalid package name %r",
                uri,
                package_name,
            )
            return uri

        rel_parts = Path(relative_path).parts
        if ".." in rel_parts or Path(relative_path).is_absolute():
            logger.warning(
                f"Refusing to resolve {uri!r}: relative path contains traversal "
                "or is absolute"
            )
            return uri

        # Strategy 1: explicit package map (highest precedence — short-circuits
        # ambiguity detection so add_package() remains a working escape hatch).
        # If the caller pinned this package and the file is missing under the
        # pinned root, do NOT fall through to other strategies: silently
        # returning a different package would defeat the explicit override.
        if package_name in self._package_map:
            pinned_root = Path(self._package_map[package_name])
            cand = pinned_root / relative_path
            if cand.exists() and not _path_escapes_base(cand, pinned_root):
                return str(cand)
            logger.warning(
                "Explicit package mapping for %r did not contain %r under %r; "
                "refusing to fall back to auto-discovery to honor the override.",
                package_name,
                relative_path,
                str(self._package_map[package_name]),
            )
            return uri

        candidates: List[Path] = []

        def accept(candidate: Path, root: Path) -> None:
            """Collect ``candidate`` only if it stays inside its own ``root``.

            Containment is per-strategy: each candidate is checked against the
            root that produced it, not against ``base_path``. ``..`` is already
            refused above, but the ancestor heuristic walks upward on the
            resolver's own initiative, so a URI needs no ``..`` to land outside
            the description directory — and ``_path_escapes_base`` resolves
            symlinks, so an in-tree link to an out-of-tree target is refused
            too.
            """
            if candidate.exists() and not _path_escapes_base(candidate, root):
                candidates.append(candidate)

        # Strategy 2: search paths — try both the package-rooted and the flat
        # forms (regression: prior code only tried search_path/pkg/relative).
        for search_path in self._search_paths:
            root = Path(search_path)
            accept(root / package_name / relative_path, root)
            accept(root / relative_path, root)

        # Strategy 3: ROS package discovery (only when use_ros is enabled).
        if self._use_ros:
            ros_pkg_root = self._find_ros_package(package_name)
            if ros_pkg_root is not None:
                accept(ros_pkg_root / relative_path, ros_pkg_root)
            ros_paths = os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep)
            for ros_path in ros_paths:
                if not ros_path:
                    continue
                root = Path(ros_path) / package_name
                accept(root / relative_path, root)

        # Strategy 4: base path fallback
        if self.base_path:
            accept(Path(self.base_path) / relative_path, Path(self.base_path))

            # Strategy 5: ancestor heuristic.
            #
            # Walking upward is what makes the ordinary ROS layout work — the
            # URDF sits in <pkg>/urdf/ while its meshes sit in <pkg>/meshes/,
            # so the mesh is only reachable from an ancestor of base_path. An
            # ancestor is outside base_path by construction, so it may serve as
            # a root only when the URI's own package name justifies it, and the
            # two forms do NOT deserve the same trust:
            #
            #   b) <ancestor>/... where the ancestor IS the named package.
            #      The URDF being resolved lives inside that directory, so it
            #      is reading from its own package. A name match is enough.
            #
            #   a) <ancestor>/<package_name>/... — a SIBLING directory that
            #      does not contain the URDF. A bare name match here is not
            #      evidence of anything: `package://.ssh/id_rsa` from an
            #      ordinary ~/robots/urdf/ layout would read the private key.
            #      Require proof it is a real package, i.e. a package.xml.
            #
            # Layouts that are not ROS packages should be pinned explicitly
            # with add_package() or add_search_path(), which are permitted
            # roots by definition and are unaffected by this check.
            for ancestor in [
                self.base_path,
                self.base_path.parent,
                self.base_path.parent.parent,
            ]:
                pkg_root = ancestor / package_name
                if (pkg_root / "package.xml").is_file():
                    accept(pkg_root / relative_path, pkg_root)
                if ancestor.name == package_name:
                    accept(ancestor / relative_path, ancestor)

        discovered_roots = self._discover_package_roots(package_name)
        if discovered_roots is None:
            logger.warning(
                "Refusing to resolve %r because package discovery was incomplete",
                uri,
            )
            return uri
        for discovered_root in discovered_roots:
            accept(
                discovered_root / relative_path,
                discovered_root,
            )

        # Dedup by canonical (symlink/case-resolved) path before ambiguity check.
        seen_canonical = set()
        unique_paths: List[str] = []
        for cand in candidates:
            try:
                canonical = cand.resolve(strict=True)
            except (OSError, RuntimeError):
                canonical = cand.absolute()
            key = str(canonical)
            if key in seen_canonical:
                continue
            seen_canonical.add(key)
            unique_paths.append(str(cand))

        if not unique_paths:
            logger.warning(f"Package URI {uri!r} could not be resolved (no candidates)")
            return uri
        if len(unique_paths) == 1:
            return unique_paths[0]

        logger.warning(
            f"Multiple package paths matched for {uri!r}: {sorted(unique_paths)}. "
            "Refusing to auto-resolve to avoid the wrong choice. Add explicit "
            "package mapping with resolver.add_package(name, path)."
        )
        return uri

    @staticmethod
    def _package_name_from_manifest(manifest: Path) -> Optional[str]:
        """Return a validated package name from a real ``package.xml`` file."""
        try:
            if manifest.is_symlink() or not manifest.is_file():
                return None
            document_root = ElementTree.parse(manifest).getroot()
            if document_root.tag != "package":
                return None
            name = document_root.findtext("name")
        except ElementTree.ParseError:
            return None
        if name is None:
            return None
        normalized = name.strip()
        if _PACKAGE_NAME_RE.fullmatch(normalized) is None:
            return None
        return normalized

    def _has_workspace_marker(self, directory: Path) -> bool:
        """Return whether ``directory`` has a usable discovery boundary marker."""
        try:
            if self._package_name_from_manifest(directory / "package.xml"):
                return True
            colcon_marker = directory / "COLCON_IGNORE"
            if colcon_marker.is_file() and not colcon_marker.is_symlink():
                return True
            git_marker = directory / ".git"
            if git_marker.is_dir() and (git_marker / "HEAD").is_file():
                return True
            if git_marker.is_file() and not git_marker.is_symlink():
                with git_marker.open(encoding="utf-8") as marker_file:
                    return marker_file.read(256).startswith("gitdir:")
        except (OSError, RuntimeError, UnicodeError):
            return False
        return False

    def _discovery_boundaries(self) -> Optional[List[Path]]:
        """Find explicit or marker-terminated roots for bounded discovery."""
        boundaries: List[Path] = []
        boundaries.extend(Path(path) for path in self._deep_search_paths)

        if self.base_path is not None:
            current = Path(self.base_path).resolve()
            for _depth in range(_MAX_DISCOVERY_UPWARD_DEPTH + 1):
                if self._has_workspace_marker(current):
                    boundaries.append(current)
                    break
                parent = current.parent
                if parent == current:
                    break
                current = parent

        unique: List[Path] = []
        seen = set()
        for boundary in boundaries:
            try:
                canonical = boundary.resolve(strict=True)
            except (OSError, RuntimeError):
                return None
            key = str(canonical)
            if key not in seen and canonical.is_dir():
                seen.add(key)
                unique.append(canonical)
                if len(unique) > _MAX_DISCOVERY_BOUNDARIES:
                    logger.warning("URDF package discovery exceeded the root cap")
                    return None
        return unique

    def _index_packages(
        self, boundary: Path, budget: Dict[str, int]
    ) -> Optional[Dict[str, List[Path]]]:
        """Index validated package manifests under one bounded workspace root.

        ``None`` means a safety cap was exhausted. Callers must discard the
        partial index because unseen packages could make a result ambiguous.
        """
        packages: Dict[str, List[Path]] = {}
        queue = deque([(boundary, 0)])

        while queue:
            directory, depth = queue.popleft()
            budget["directories"] += 1
            if budget["directories"] > _MAX_DISCOVERY_DIRECTORIES:
                logger.warning("URDF package discovery exceeded the directory cap")
                return None

            manifest = directory / "package.xml"
            try:
                manifest_exists = manifest.exists()
            except OSError:
                return None
            if manifest_exists:
                budget["candidates"] += 1
                if budget["candidates"] > _MAX_DISCOVERY_CANDIDATES:
                    logger.warning("URDF package discovery exceeded the candidate cap")
                    return None
                try:
                    package_name = self._package_name_from_manifest(manifest)
                except OSError:
                    return None
                if package_name is not None and not _path_escapes_base(
                    directory, boundary
                ):
                    packages.setdefault(package_name, []).append(directory)

            if depth >= _MAX_DISCOVERY_DOWNWARD_DEPTH:
                continue
            try:
                entries = os.scandir(directory)
            except OSError:
                return None
            with entries:
                for entry in entries:
                    budget["entries"] += 1
                    if budget["entries"] > _MAX_DISCOVERY_ENTRIES:
                        logger.warning(
                            "URDF package discovery exceeded the filesystem entry cap"
                        )
                        return None
                    try:
                        if entry.is_symlink() or not entry.is_dir(
                            follow_symlinks=False
                        ):
                            continue
                    except OSError:
                        return None
                    queue.append((Path(entry.path), depth + 1))

        return packages

    def _discover_package_roots(self, package_name: str) -> Optional[List[Path]]:
        """Return contained roots, or ``None`` when discovery was incomplete."""
        discovered: List[Path] = []
        boundaries = self._discovery_boundaries()
        if boundaries is None:
            return None
        budget = {"candidates": 0, "directories": 0, "entries": 0}
        for boundary in boundaries:
            index = self._index_packages(boundary, budget)
            if index is None:
                return None
            discovered.extend(index.get(package_name, []))
        return discovered

    def _resolve_relative_path(self, path: str) -> str:
        """
        Resolve a relative path.

        A relative mesh reference must stay within the robot-description
        directory. Paths containing a ``..`` component, or whose resolved
        location escapes the search root, are refused and returned unchanged
        (mirroring the ``package://`` traversal guard).

        Args:
            path: Relative path

        Returns:
            Resolved absolute path
        """
        if ".." in Path(path).parts:
            logger.warning(
                f"Refusing to resolve relative path {path!r}: contains '..' traversal"
            )
            return path

        # Try base path first
        if self.base_path:
            candidate = self.base_path / path
            if candidate.exists() and not _path_escapes_base(candidate, self.base_path):
                return str(candidate)

        # Try search paths
        for search_path in self._search_paths:
            candidate = search_path / path
            if candidate.exists() and not _path_escapes_base(candidate, search_path):
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

        Sets ``base_path`` to the URDF's directory. Package roots are then
        found by the resolver's bounded, manifest-based discovery.

        Args:
            urdf_path: Path to the URDF file

        Returns:
            Configured PackageResolver
        """
        urdf_path = Path(urdf_path).resolve()
        base_path = urdf_path.parent

        return cls(base_path=base_path)
