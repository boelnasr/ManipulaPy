#!/usr/bin/env python3
"""xArm Robot Model Data"""
import os
from pathlib import Path

# Path to this robot's data directory
DATA_DIR = Path(__file__).parent

# Find the main URDF file (try common names)
_urdf_candidates = [
    "xarm6_robot.urdf",
    "xarm6_robot_white.urdf",
    "xarm6_with_gripper.urdf",
    "xarm_robot.urdf", 
    "xarm.urdf",
]

urdf_file = None
for candidate in _urdf_candidates:
    candidate_path = DATA_DIR / candidate
    if candidate_path.exists():
        urdf_file = str(candidate_path)
        break

if urdf_file is None:
    # Fallback to first .urdf file found
    urdf_files = list(DATA_DIR.glob("*.urdf"))
    if urdf_files:
        urdf_file = str(urdf_files[0])

# Export the URDF path for easy access
__all__ = ['urdf_file', 'DATA_DIR']