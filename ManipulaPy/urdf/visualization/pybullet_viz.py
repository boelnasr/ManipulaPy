#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PyBullet-based URDF Visualization

Copyright (c) 2025 Mohamed Aboelnasr
"""

from typing import TYPE_CHECKING, Optional, Dict, Union
import numpy as np
import time

if TYPE_CHECKING:
    from ..core import URDF

try:
    import pybullet as p
    import pybullet_data

    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


def show_pybullet(
    urdf: "URDF",
    cfg: Optional[Union[np.ndarray, Dict[str, float]]] = None,
) -> None:
    """
    Show robot using PyBullet GUI.

    Note: Requires URDF file path to be available.
    """
    if not PYBULLET_AVAILABLE:
        raise ImportError("pybullet is required for visualization")

    # This requires the original URDF file path
    # For now, we'll use a simplified approach
    raise NotImplementedError(
        "PyBullet visualization requires URDF file path. "
        "Use trimesh backend or urdf.show() from URDFToSerialManipulator."
    )


def animate_pybullet(
    urdf: "URDF",
    cfg_trajectory: Union[Dict[str, np.ndarray], np.ndarray],
    loop_time: float = 3.0,
) -> None:
    """
    Animate robot using PyBullet.

    Note: Requires URDF file path to be available.
    """
    if not PYBULLET_AVAILABLE:
        raise ImportError("pybullet is required for animation")

    raise NotImplementedError(
        "PyBullet animation requires URDF file path. "
        "Use trimesh backend or urdf.animate() from URDFToSerialManipulator."
    )
