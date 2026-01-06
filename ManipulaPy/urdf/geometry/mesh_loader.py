#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Mesh File Loader

Load mesh files (STL, OBJ, DAE) with optional trimesh fallback.

Copyright (c) 2025 Mohamed Aboelnasr
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import struct
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Container for mesh geometry data."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None


def load_mesh(filename: str, scale: np.ndarray = None) -> Optional[MeshData]:
    """
    Load mesh from file.

    Supports STL (binary and ASCII), OBJ.
    Falls back to trimesh for DAE and complex formats.

    Args:
        filename: Path to mesh file
        scale: Scale factors (x, y, z)

    Returns:
        MeshData or None if loading fails
    """
    path = Path(filename)

    if not path.exists():
        logger.warning(f"Mesh file not found: {filename}")
        return None

    suffix = path.suffix.lower()

    try:
        if suffix == ".stl":
            mesh = _load_stl(path)
        elif suffix == ".obj":
            mesh = _load_obj(path)
        elif suffix in (".dae", ".collada"):
            mesh = _load_with_trimesh(path)
        else:
            mesh = _load_with_trimesh(path)

        if mesh is not None and scale is not None:
            mesh.vertices = mesh.vertices * scale

        return mesh

    except Exception as e:
        logger.warning(f"Failed to load mesh {filename}: {e}")
        return None


def _load_stl(path: Path) -> Optional[MeshData]:
    """Load STL file (binary or ASCII)."""
    with open(path, "rb") as f:
        header = f.read(80)

        # Check if ASCII
        f.seek(0)
        first_line = f.readline()
        if first_line.strip().lower().startswith(b"solid"):
            # Could be ASCII, check further
            f.seek(0)
            content = f.read()
            if b"facet normal" in content:
                return _load_stl_ascii(path)

        # Binary STL
        f.seek(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]

        vertices = []
        normals = []

        for _ in range(num_triangles):
            # Normal (3 floats)
            nx, ny, nz = struct.unpack("<3f", f.read(12))
            normals.extend([[nx, ny, nz]] * 3)

            # 3 vertices
            for _ in range(3):
                x, y, z = struct.unpack("<3f", f.read(12))
                vertices.append([x, y, z])

            # Attribute byte count (unused)
            f.read(2)

        vertices = np.array(vertices, dtype=np.float64)
        normals = np.array(normals, dtype=np.float64)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)

        return MeshData(vertices=vertices, faces=faces, normals=normals)


def _load_stl_ascii(path: Path) -> Optional[MeshData]:
    """Load ASCII STL file."""
    vertices = []
    normals = []

    with open(path, "r") as f:
        current_normal = None

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "facet" and parts[1] == "normal":
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]

            elif parts[0] == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if current_normal:
                    normals.append(current_normal)

    if not vertices:
        return None

    vertices = np.array(vertices, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64) if normals else None
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)

    return MeshData(vertices=vertices, faces=faces, normals=normals)


def _load_obj(path: Path) -> Optional[MeshData]:
    """Load OBJ file."""
    vertices = []
    faces = []
    normals = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif parts[0] == "vn":
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif parts[0] == "f":
                # Handle different face formats: v, v/vt, v/vt/vn, v//vn
                face_vertices = []
                for p in parts[1:]:
                    v_idx = int(p.split("/")[0])
                    # OBJ indices are 1-based
                    face_vertices.append(v_idx - 1 if v_idx > 0 else v_idx)

                # Triangulate if more than 3 vertices
                for i in range(1, len(face_vertices) - 1):
                    faces.append(
                        [face_vertices[0], face_vertices[i], face_vertices[i + 1]]
                    )

    if not vertices:
        return None

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    normals = np.array(normals, dtype=np.float64) if normals else None

    return MeshData(vertices=vertices, faces=faces, normals=normals)


def _load_with_trimesh(path: Path) -> Optional[MeshData]:
    """Load mesh using trimesh library."""
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh")

        return MeshData(
            vertices=np.array(mesh.vertices, dtype=np.float64),
            faces=np.array(mesh.faces, dtype=np.int64),
            normals=np.array(mesh.vertex_normals, dtype=np.float64)
            if hasattr(mesh, "vertex_normals")
            else None,
        )

    except ImportError:
        logger.warning("trimesh not available for loading complex mesh formats")
        return None
    except Exception as e:
        logger.warning(f"trimesh failed to load {path}: {e}")
        return None
