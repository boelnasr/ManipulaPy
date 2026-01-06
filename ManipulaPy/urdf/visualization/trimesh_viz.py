#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Trimesh-based URDF Visualization

Copyright (c) 2025 Mohamed Aboelnasr
"""

from typing import TYPE_CHECKING, Optional, Dict, Union
import numpy as np

if TYPE_CHECKING:
    from ..core import URDF

try:
    import trimesh
    import trimesh.transformations as tra

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def _geometry_to_trimesh(geometry, color=None):
    """Convert URDF geometry to trimesh object."""
    from ..types import Box, Cylinder, Sphere, Mesh

    if geometry is None:
        return None

    mesh = None

    if isinstance(geometry, Box):
        mesh = trimesh.creation.box(extents=geometry.size)

    elif isinstance(geometry, Cylinder):
        mesh = trimesh.creation.cylinder(
            radius=geometry.radius, height=geometry.length
        )

    elif isinstance(geometry, Sphere):
        mesh = trimesh.creation.icosphere(radius=geometry.radius, subdivisions=3)

    elif isinstance(geometry, Mesh):
        if geometry.filename:
            try:
                mesh = trimesh.load(geometry.filename, force="mesh")
                # Apply scale
                if not np.allclose(geometry.scale, 1.0):
                    mesh.apply_scale(geometry.scale)
            except Exception:
                # Create placeholder if mesh loading fails
                mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])

    if mesh is not None and color is not None:
        mesh.visual.face_colors = color

    return mesh


def _build_scene(urdf: "URDF", cfg=None, use_collision: bool = False):
    """Build trimesh scene from URDF."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for visualization")

    # Get FK
    fk = urdf.link_fk(cfg, use_names=True)

    scene = trimesh.Scene()

    for link_name, transform in fk.items():
        link = urdf.get_link(link_name)
        if link is None:
            continue

        geometries = link.collisions if use_collision else link.visuals

        for i, geom_item in enumerate(geometries):
            geom = geom_item.geometry
            if geom is None:
                continue

            # Get color from material (for visuals)
            color = None
            if hasattr(geom_item, "material") and geom_item.material:
                if geom_item.material.color is not None:
                    color = (geom_item.material.color * 255).astype(np.uint8)

            mesh = _geometry_to_trimesh(geom, color)
            if mesh is None:
                continue

            # Apply geometry origin transform
            geom_transform = geom_item.origin.matrix
            full_transform = transform @ geom_transform

            mesh_name = f"{link_name}_{i}"
            scene.add_geometry(mesh, node_name=mesh_name, transform=full_transform)

    return scene


def show_trimesh(
    urdf: "URDF",
    cfg: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    use_collision: bool = False,
) -> None:
    """Show robot using trimesh viewer."""
    scene = _build_scene(urdf, cfg, use_collision)
    scene.show()


def animate_trimesh(
    urdf: "URDF",
    cfg_trajectory: Union[Dict[str, np.ndarray], np.ndarray],
    loop_time: float = 3.0,
    use_collision: bool = False,
) -> None:
    """Animate robot using trimesh."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for animation")

    # Convert trajectory to array format
    if isinstance(cfg_trajectory, dict):
        # Get joint order
        joint_names = urdf.actuated_joint_names
        n_steps = None
        for name in joint_names:
            if name in cfg_trajectory:
                n_steps = len(cfg_trajectory[name])
                break

        if n_steps is None:
            raise ValueError("Empty trajectory")

        cfgs = np.zeros((n_steps, len(joint_names)), dtype=np.float64)
        for i, name in enumerate(joint_names):
            if name in cfg_trajectory:
                cfgs[:, i] = cfg_trajectory[name]
    else:
        cfgs = np.asarray(cfg_trajectory, dtype=np.float64)
        if cfgs.ndim == 1:
            cfgs = cfgs.reshape(1, -1)

    n_steps = cfgs.shape[0]
    dt = loop_time / n_steps

    # Build initial scene
    scene = _build_scene(urdf, cfgs[0], use_collision)

    def callback(scene):
        """Animation callback."""
        # Get current frame from time
        import time

        t = time.time() % loop_time
        frame = int((t / loop_time) * n_steps) % n_steps

        cfg = cfgs[frame]
        fk = urdf.link_fk(cfg, use_names=True)

        # Update node transforms
        for link_name, transform in fk.items():
            link = urdf.get_link(link_name)
            if link is None:
                continue

            geometries = link.collisions if use_collision else link.visuals

            for i, geom_item in enumerate(geometries):
                mesh_name = f"{link_name}_{i}"
                if mesh_name in scene.graph.nodes:
                    geom_transform = geom_item.origin.matrix
                    full_transform = transform @ geom_transform
                    scene.graph.update(frame_to=mesh_name, matrix=full_transform)

    scene.show(callback=callback)
