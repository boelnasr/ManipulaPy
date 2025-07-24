#!/usr/bin/env python3
"""UR5 Robot Model Data"""
import os
from pathlib import Path

# Path to this robot's data directory
DATA_DIR = Path(__file__).parent

# Main URDF file path
urdf_file = str(DATA_DIR / "ur5.urdf")

# Export the URDF path for easy access
__all__ = ['urdf_file', 'DATA_DIR']