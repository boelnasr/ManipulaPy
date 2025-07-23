#!/usr/bin/env python3
"""
ManipulaPy Robot Data

This module provides access to robot models, URDF files, and related assets
included with ManipulaPy.
"""

import os
import pkg_resources
from typing import List, Dict, Optional

# Get the path to this module
_module_path = os.path.dirname(__file__)

def get_robot_path(robot_name: str, model: Optional[str] = None) -> str:
    """
    Get the path to a robot's URDF file.
    
    Args:
        robot_name (str): Robot name ('ur5', 'xarm')
        model (str, optional): Specific model variant
        
    Returns:
        str: Path to the URDF file
        
    Raises:
        ValueError: If robot_name is not supported
        FileNotFoundError: If the requested file doesn't exist
    """
    if robot_name.lower() == 'ur5':
        try:
            from .ur5 import get_urdf_path
            return get_urdf_path()
        except ImportError:
            # Fallback
            return os.path.join(_module_path, 'ur5', 'ur5.urdf')
            
    elif robot_name.lower() == 'xarm':
        try:
            from .xarm import get_urdf_path
            return get_urdf_path(model or 'xarm6_robot')
        except ImportError:
            # Fallback
            model_file = f"{model or 'xarm6_robot'}.urdf"
            return os.path.join(_module_path, 'xarm', model_file)
    else:
        raise ValueError(f"Unsupported robot: {robot_name}. Available: ur5, xarm")

def list_available_robots() -> List[str]:
    """
    List all available robot models.
    
    Returns:
        List[str]: List of available robot names
    """
    robots = []
    try:
        for item in os.listdir(_module_path):
            item_path = os.path.join(_module_path, item)
            if os.path.isdir(item_path) and not item.startswith('__'):
                robots.append(item)
    except:
        pass
    return robots

def get_robot_info() -> Dict[str, Dict]:
    """
    Get information about available robots.
    
    Returns:
        Dict[str, Dict]: Dictionary with robot information
    """
    info = {}
    
    # UR5 info
    try:
        from .ur5 import get_urdf_path
        info['ur5'] = {
            'name': 'Universal Robots UR5',
            'dof': 6,
            'urdf_path': get_urdf_path(),
            'description': '6-DOF collaborative robot arm'
        }
    except:
        pass
    
    # XARM info  
    try:
        from .xarm import get_urdf_path, list_available_models
        info['xarm'] = {
            'name': 'UFactory XARM6',
            'dof': 6,
            'urdf_path': get_urdf_path(),
            'models': list_available_models(),
            'description': '6-DOF industrial robot arm'
        }
    except:
        pass
        
    return info

def check_data_integrity() -> Dict[str, bool]:
    """
    Check if all expected data files are present.
    
    Returns:
        Dict[str, bool]: Status of each robot's data
    """
    status = {}
    
    # Check UR5
    try:
        ur5_path = get_robot_path('ur5')
        status['ur5'] = os.path.exists(ur5_path)
    except:
        status['ur5'] = False
    
    # Check XARM
    try:
        xarm_path = get_robot_path('xarm')
        status['xarm'] = os.path.exists(xarm_path)
    except:
        status['xarm'] = False
        
    return status

# Convenience exports
def get_ur5_urdf():
    """Get UR5 URDF path (convenience function)."""
    return get_robot_path('ur5')

def get_xarm_urdf(model='xarm6_robot'):
    """Get XARM URDF path (convenience function)."""
    return get_robot_path('xarm', model)

# Module-level constants
AVAILABLE_ROBOTS = list_available_robots()
ROBOT_INFO = get_robot_info()

__all__ = [
    'get_robot_path',
    'list_available_robots', 
    'get_robot_info',
    'check_data_integrity',
    'get_ur5_urdf',
    'get_xarm_urdf',
    'AVAILABLE_ROBOTS',
    'ROBOT_INFO'
]
