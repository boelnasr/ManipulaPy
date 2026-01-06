#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Visualization

Lazy-loaded visualization using trimesh or PyBullet.

Copyright (c) 2025 Mohamed Aboelnasr
"""

from typing import TYPE_CHECKING, Optional, Dict, Union
import numpy as np

if TYPE_CHECKING:
    from ..core import URDF


def show_robot(
    urdf: "URDF",
    cfg: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    use_collision: bool = False,
) -> None:
    """
    Visualize robot using available backend.

    Tries trimesh first, falls back to PyBullet.
    """
    # Try trimesh first
    try:
        from .trimesh_viz import show_trimesh

        show_trimesh(urdf, cfg, use_collision)
        return
    except ImportError:
        pass

    # Try PyBullet
    try:
        from .pybullet_viz import show_pybullet

        show_pybullet(urdf, cfg)
        return
    except ImportError:
        pass

    raise ImportError(
        "No visualization backend available. "
        "Install trimesh or pybullet for visualization support."
    )


def animate_robot(
    urdf: "URDF",
    cfg_trajectory: Union[Dict[str, np.ndarray], np.ndarray],
    loop_time: float = 3.0,
    use_collision: bool = False,
) -> None:
    """
    Animate robot along trajectory using available backend.
    """
    # Try trimesh first
    try:
        from .trimesh_viz import animate_trimesh

        animate_trimesh(urdf, cfg_trajectory, loop_time, use_collision)
        return
    except ImportError:
        pass

    # Try PyBullet
    try:
        from .pybullet_viz import animate_pybullet

        animate_pybullet(urdf, cfg_trajectory, loop_time)
        return
    except ImportError:
        pass

    raise ImportError(
        "No visualization backend available. "
        "Install trimesh or pybullet for animation support."
    )


__all__ = ["show_robot", "animate_robot"]
