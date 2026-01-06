#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Primitive Geometry Generation

Generate vertices and faces for box, cylinder, sphere primitives.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import numpy as np
from typing import Tuple


def create_box_vertices(size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create box mesh vertices and faces.

    Args:
        size: Box dimensions (x, y, z)

    Returns:
        vertices: (8, 3) array of vertex positions
        faces: (12, 3) array of triangle face indices
    """
    sx, sy, sz = size / 2

    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            # Bottom
            [0, 2, 1],
            [0, 3, 2],
            # Top
            [4, 5, 6],
            [4, 6, 7],
            # Front
            [0, 1, 5],
            [0, 5, 4],
            # Back
            [2, 3, 7],
            [2, 7, 6],
            # Left
            [0, 4, 7],
            [0, 7, 3],
            # Right
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )

    return vertices, faces


def create_cylinder_vertices(
    radius: float, length: float, segments: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cylinder mesh vertices and faces.

    Cylinder is centered at origin, aligned with Z axis.

    Args:
        radius: Cylinder radius
        length: Cylinder length
        segments: Number of segments around circumference

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
    """
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    hz = length / 2

    # Bottom circle vertices
    bottom = np.column_stack([radius * cos_a, radius * sin_a, np.full(segments, -hz)])

    # Top circle vertices
    top = np.column_stack([radius * cos_a, radius * sin_a, np.full(segments, hz)])

    # Center vertices for caps
    bottom_center = np.array([[0, 0, -hz]])
    top_center = np.array([[0, 0, hz]])

    vertices = np.vstack([bottom, top, bottom_center, top_center])

    # Faces
    faces = []
    bottom_center_idx = 2 * segments
    top_center_idx = 2 * segments + 1

    for i in range(segments):
        next_i = (i + 1) % segments

        # Side triangles
        faces.append([i, next_i, segments + next_i])
        faces.append([i, segments + next_i, segments + i])

        # Bottom cap
        faces.append([bottom_center_idx, next_i, i])

        # Top cap
        faces.append([top_center_idx, segments + i, segments + next_i])

    faces = np.array(faces, dtype=np.int64)

    return vertices, faces


def create_sphere_vertices(
    radius: float, subdivisions: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sphere mesh vertices and faces using icosphere subdivision.

    Args:
        radius: Sphere radius
        subdivisions: Number of subdivision iterations

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle face indices
    """
    # Start with icosahedron
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Subdivide
    for _ in range(subdivisions):
        vertices, faces = _subdivide_icosphere(vertices, faces)

    # Scale to desired radius
    vertices = vertices * radius

    return vertices, faces


def _subdivide_icosphere(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide icosphere by splitting each triangle into 4."""
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []

    def get_midpoint(i1, i2):
        """Get or create midpoint vertex index."""
        key = (min(i1, i2), max(i1, i2))
        if key in edge_midpoints:
            return edge_midpoints[key]

        v1, v2 = vertices[i1], vertices[i2]
        mid = (v1 + v2) / 2
        mid = mid / np.linalg.norm(mid)  # Project to sphere

        idx = len(new_vertices)
        new_vertices.append(mid)
        edge_midpoints[key] = idx
        return idx

    for face in faces:
        v0, v1, v2 = face

        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)

        new_faces.extend(
            [
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c],
            ]
        )

    return np.array(new_vertices, dtype=np.float64), np.array(new_faces, dtype=np.int64)
