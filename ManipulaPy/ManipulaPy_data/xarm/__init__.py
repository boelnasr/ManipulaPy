#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
xArm robot data module.

Provides access to xArm URDF files.
"""

from pathlib import Path

_MODULE_PATH = Path(__file__).parent

# Default xArm6 URDF file path
urdf_file = str(_MODULE_PATH / "xarm6_robot.urdf")

__all__ = ['urdf_file']
