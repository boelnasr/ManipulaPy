#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
UR5 robot data module.

Provides access to Universal Robots UR5 URDF files.
"""

from pathlib import Path

_MODULE_PATH = Path(__file__).parent.parent

# UR5 URDF file path (located in universal_robots subdirectory)
urdf_file = str(_MODULE_PATH / "universal_robots" / "ur5" / "ur5.urdf")

__all__ = ['urdf_file']
