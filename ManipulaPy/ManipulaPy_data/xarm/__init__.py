#!/usr/bin/env python3
"""
XARM Robot Model Data

This module provides access to XARM robot URDF files and related assets.
"""

import os
import pkg_resources

# Get the path to this module
_module_path = os.path.dirname(__file__)

def get_urdf_path(model='xarm6_robot'):
    """
    Get the path to an XARM URDF file.
    
    Args:
        model (str): URDF model name. Options:
            - 'xarm6_robot' (default)
            - 'xarm6_robot_white' 
            - 'xarm6_with_gripper'
            - 'base', 'link1', 'link2', etc.
    
    Returns:
        str: Path to the URDF file
    """
    urdf_filename = f"{model}.urdf"
    try:
        # Try pkg_resources first (works with pip installs)
        return pkg_resources.resource_filename(__name__, urdf_filename)
    except:
        # Fallback to direct path
        return os.path.join(_module_path, urdf_filename)

def get_description_path():
    """Get the path to the XARM description directory."""
    try:
        return pkg_resources.resource_filename(__name__, 'xarm_description')
    except:
        return os.path.join(_module_path, 'xarm_description')

def get_gripper_path():
    """Get the path to the XARM gripper directory."""
    try:
        return pkg_resources.resource_filename(__name__, 'xarm_gripper')
    except:
        return os.path.join(_module_path, 'xarm_gripper')

def list_available_models():
    """List all available URDF models."""
    models = []
    try:
        # List all .urdf files in the directory
        for file in os.listdir(_module_path):
            if file.endswith('.urdf'):
                models.append(file[:-5])  # Remove .urdf extension
    except:
        pass
    return models

# Export the main URDF path for easy access
urdf_file = get_urdf_path('xarm6_robot')

__all__ = [
    'get_urdf_path', 
    'get_description_path', 
    'get_gripper_path', 
    'list_available_models',
    'urdf_file'
]