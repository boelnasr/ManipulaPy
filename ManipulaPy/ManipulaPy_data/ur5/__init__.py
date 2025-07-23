#!/usr/bin/env python3
"""
UR5 Robot Model Data

This module provides access to UR5 robot URDF files and related assets.
"""

import os
import pkg_resources

# Get the path to this module
_module_path = os.path.dirname(__file__)

def get_urdf_path():
    """Get the path to the UR5 URDF file."""
    try:
        # Try pkg_resources first (works with pip installs)
        return pkg_resources.resource_filename(__name__, 'ur5.urdf')
    except:
        # Fallback to direct path
        return os.path.join(_module_path, 'ur5.urdf')

def get_visual_path():
    """Get the path to the UR5 visual directory."""
    try:
        return pkg_resources.resource_filename(__name__, 'visual')
    except:
        return os.path.join(_module_path, 'visual')

def get_collision_path():
    """Get the path to the UR5 collision directory."""
    try:
        return pkg_resources.resource_filename(__name__, 'collision')
    except:
        return os.path.join(_module_path, 'collision')

# Export the URDF path for easy access
urdf_file = get_urdf_path()

__all__ = ['get_urdf_path', 'get_visual_path', 'get_collision_path', 'urdf_file']
