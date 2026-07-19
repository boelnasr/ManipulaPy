#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side collision checking for potential-field planning."""

import itertools
import logging
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from ..backend import get_backend
from ..urdf import URDF
from .adjacency import build_link_adjacency

_logger = logging.getLogger("ManipulaPy.potential_field.fields")


def _to_host_numpy(backend: Any, value: Any) -> NDArray[Any]:
    """Materialize backend-native values without round-tripping host arrays."""
    if backend.is_backend_array(value):
        value = backend.to_numpy(value)
    return np.asarray(value)


class CollisionChecker:
    """
    Collision checker using URDF visual/collision geometry and convex hulls.

    Supports multiple URDF parser backends:
        - "builtin": Native ManipulaPy parser (NumPy 2.0 compatible, default)
        - "pybullet": PyBullet-based parser (requires pybullet)
    """

    def __init__(
        self, urdf_path: str, backend: str = "builtin", load_meshes: bool = True
    ) -> None:
        """
        Initializes a CollisionChecker object.

        Args:
            urdf_path (str): The path to the URDF file.
            backend (str): Parser backend - "builtin" (default) or "pybullet"
            load_meshes (bool): Whether to load mesh geometry data (default: True)
        """
        self.robot = URDF.load(urdf_path, backend=backend, load_meshes=load_meshes)
        # ACM derived from URDF topology to avoid adjacent-link false positives
        self._acm = build_link_adjacency(self.robot, exclude_grandparents=True)
        self._visual_fallback_warned = set()
        self.convex_hulls = self._create_convex_hulls()

    def _warn_visual_fallback_once(self, link_name: str) -> None:
        """Log one visual-geometry fallback warning per link."""
        if link_name not in self._visual_fallback_warned:
            self._visual_fallback_warned.add(link_name)
            _logger.warning(
                "Link %r has no collision geometry; falling back to visual geometry "
                "for collision checking — results may be inaccurate.",
                link_name,
            )

    def _create_convex_hulls(self) -> Dict[str, ConvexHull]:
        """
        Creates a dictionary of convex hulls for each link, preferring collision
        geometry and falling back to visual geometry with a one-shot warning.

        Returns:
            dict: A dictionary where the keys are the names of the robot links
                  and the values are the corresponding convex hulls.
        """
        backend = get_backend()
        convex_hulls = {}
        for link in self.robot.links:
            # Prefer collision geometry; fall back to visuals with a warning
            sources = link.collisions if link.collisions else None
            if sources is None and link.visuals:
                self._warn_visual_fallback_once(link.name)
                sources = link.visuals
            if not sources:
                continue

            # Aggregate vertices from EVERY geometry element on the link, so a
            # link with multiple collision meshes is enclosed by a single hull
            # spanning all of them. (Previously the loop overwrote the entry on
            # each iteration, keeping only the last mesh and silently dropping
            # the rest — which could miss self-collisions.)
            link_vertices = []
            for geom_element in sources:
                if geom_element.geometry is None:
                    continue
                geom = geom_element.geometry

                vertices = None
                if hasattr(geom, "mesh_data") and geom.mesh_data is not None:
                    vertices = geom.mesh_data.vertices
                elif hasattr(geom, "mesh") and geom.mesh is not None:
                    mesh = geom.mesh
                    if hasattr(mesh, "vertices"):
                        vertices = mesh.vertices

                if vertices is None:
                    continue
                # Mesh data feeds host-only NumPy/SciPy geometry operations.
                vertices = np.asarray(_to_host_numpy(backend, vertices), dtype=float)
                if vertices.ndim != 2 or vertices.shape[0] < 1:
                    continue

                # Transform vertices into the link frame via the element's
                # <origin> so meshes with different offsets are combined in the
                # correct relative positions (not all stacked at the link frame).
                origin = getattr(geom_element, "origin", None)
                T = getattr(origin, "matrix", None) if origin is not None else None
                if T is not None:
                    T = _to_host_numpy(backend, T)
                if (
                    isinstance(T, np.ndarray)
                    and T.shape == (4, 4)
                    and vertices.shape[1] == 3
                ):
                    vertices = vertices @ T[:3, :3].T + T[:3, 3]

                link_vertices.append(vertices)

            if not link_vertices:
                continue

            all_vertices = np.vstack(link_vertices)
            if len(all_vertices) < 4:
                continue  # need >= 4 points for a 3-D convex hull
            try:
                convex_hulls[link.name] = ConvexHull(all_vertices)
            except Exception as exc:
                _logger.warning(
                    "Skipping convex hull for link %r (%d vertices): %s",
                    link.name,
                    len(all_vertices),
                    exc,
                )

        return convex_hulls

    def _transform_convex_hull(
        self, convex_hull: ConvexHull, transform: NDArray[np.float64]
    ) -> ConvexHull:
        """Apply a 4x4 transform to a ConvexHull and return a NEW ConvexHull.

        Retained for backwards compatibility with existing tests
        (test_potential_field_coverage.py, test_potential_field_extended.py)
        that call this method directly. Internal callers in check_collision
        transform cached vertices directly via matrix multiply and skip the
        ConvexHull rebuild — prefer that path for new code.
        """
        backend = get_backend()
        # SciPy ConvexHull is host-only: materialize the transform explicitly.
        transform = _to_host_numpy(backend, transform)
        transformed_points = transform[:3, :3] @ convex_hull.points.T + transform[
            :3, 3
        ].reshape(-1, 1)
        return ConvexHull(transformed_points.T)

    def check_collision(self, thetalist: Any) -> bool:
        """
        Check for self-collision at a given joint configuration.

        Args:
            thetalist: Joint configuration (array or dict)

        Returns:
            bool: True if collision detected, False otherwise
        """
        backend = get_backend()
        # URDF FK is host-only for array configurations; named dicts are part
        # of its public API and must pass through unchanged.
        configuration = (
            thetalist
            if isinstance(thetalist, dict)
            else _to_host_numpy(backend, thetalist)
        )
        fk_results = self.robot.link_fk(cfg=configuration, use_names=True)

        hull_names = [n for n in self.convex_hulls if n in fk_results]
        for name_a, name_b in itertools.combinations(hull_names, 2):
            # Skip adjacent-link pairs — they always overlap at joints
            if frozenset({name_a, name_b}) in self._acm:
                continue

            # NumPy/SciPy collision geometry is an explicit host boundary.
            T_a = _to_host_numpy(backend, fk_results[name_a])
            T_b = _to_host_numpy(backend, fk_results[name_b])
            pts_a = (T_a[:3, :3] @ self.convex_hulls[name_a].points.T + T_a[:3, 3:4]).T
            pts_b = (T_b[:3, :3] @ self.convex_hulls[name_b].points.T + T_b[:3, 3:4]).T
            if self._points_intersect(pts_a, pts_b):
                return True
        return False

    def _points_intersect(
        self, pts_a: NDArray[np.float64], pts_b: NDArray[np.float64]
    ) -> bool:
        """
        Check if two point clouds' bounding boxes intersect.

        This is a simplified check — for production use, consider a proper
        collision detection library like fcl or trimesh.

        Args:
            pts_a: (N, 3) array of points
            pts_b: (M, 3) array of points

        Returns:
            bool: True if bounding boxes overlap
        """
        backend = get_backend()
        # Bounding-box checks are NumPy host operations for direct callers too.
        pts_a = _to_host_numpy(backend, pts_a)
        pts_b = _to_host_numpy(backend, pts_b)
        min_a = np.min(pts_a, axis=0)
        max_a = np.max(pts_a, axis=0)
        min_b = np.min(pts_b, axis=0)
        max_b = np.max(pts_b, axis=0)
        return bool(np.all(max_a >= min_b) and np.all(max_b >= min_a))

    def _hulls_intersect(self, hull1: ConvexHull, hull2: ConvexHull) -> bool:
        """
        Check if two convex hulls intersect.

        Thin wrapper around _points_intersect for backwards compatibility
        with external callers that pass ConvexHull objects.

        Args:
            hull1: First ConvexHull
            hull2: Second ConvexHull

        Returns:
            bool: True if hulls potentially intersect
        """
        return self._points_intersect(hull1.points, hull2.points)
