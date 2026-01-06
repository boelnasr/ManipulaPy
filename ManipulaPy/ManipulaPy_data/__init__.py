#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ManipulaPy Robot Data

This module provides access to robot models, URDF files, and related assets
included with ManipulaPy.

Supported Robot Families:
    - Universal Robots: UR3, UR5, UR10, UR3e, UR5e, UR10e, UR16e
    - Franka Emika: Panda
    - KUKA: LBR iiwa 7, LBR iiwa 14
    - Kinova: Gen3, Jaco
    - Fanuc: LRMate 200iB, M16iB
    - ABB: IRB 2400
    - UFactory: xArm 5, 6, 7
    - Robotiq: 2F-85, 2F-140 grippers

Example:
    >>> from ManipulaPy.ManipulaPy_data import get_robot_urdf, list_robots
    >>> print(list_robots())
    >>> urdf_path = get_robot_urdf('ur5')
    >>> urdf_path = get_robot_urdf('panda')
    >>> urdf_path = get_robot_urdf('iiwa14')

Copyright (c) 2025 Mohamed Aboelnasr
"""

import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Get the path to this module
_MODULE_PATH = Path(__file__).parent

# =============================================================================
# Robot Database
# =============================================================================

ROBOT_DATABASE: Dict[str, Dict] = {
    # Universal Robots
    'ur3': {
        'name': 'Universal Robots UR3',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '3 kg',
        'reach': '500 mm',
        'urdf': 'universal_robots/ur3/ur3.urdf',
        'description': '6-DOF collaborative robot, compact version',
    },
    'ur5': {
        'name': 'Universal Robots UR5',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '5 kg',
        'reach': '850 mm',
        'urdf': 'universal_robots/ur5/ur5.urdf',
        'description': '6-DOF collaborative robot, medium payload',
    },
    'ur10': {
        'name': 'Universal Robots UR10',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '10 kg',
        'reach': '1300 mm',
        'urdf': 'universal_robots/ur10/ur10.urdf',
        'description': '6-DOF collaborative robot, high payload',
    },
    'ur3e': {
        'name': 'Universal Robots UR3e',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '3 kg',
        'reach': '500 mm',
        'urdf': 'universal_robots/ur3e/ur3e.urdf',
        'description': '6-DOF e-Series collaborative robot, compact',
    },
    'ur5e': {
        'name': 'Universal Robots UR5e',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '5 kg',
        'reach': '850 mm',
        'urdf': 'universal_robots/ur5e/ur5e.urdf',
        'description': '6-DOF e-Series collaborative robot, medium payload',
    },
    'ur10e': {
        'name': 'Universal Robots UR10e',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '12.5 kg',
        'reach': '1300 mm',
        'urdf': 'universal_robots/ur10e/ur10e.urdf',
        'description': '6-DOF e-Series collaborative robot, high payload',
    },
    'ur16e': {
        'name': 'Universal Robots UR16e',
        'manufacturer': 'Universal Robots',
        'dof': 6,
        'payload': '16 kg',
        'reach': '900 mm',
        'urdf': 'universal_robots/ur16e/ur16e.urdf',
        'description': '6-DOF e-Series collaborative robot, heavy payload',
    },

    # Franka Emika
    'panda': {
        'name': 'Franka Emika Panda',
        'manufacturer': 'Franka Emika',
        'dof': 7,
        'payload': '3 kg',
        'reach': '855 mm',
        'urdf': 'franka_panda/panda.urdf',
        'description': '7-DOF research robot with torque sensing',
    },
    'franka_panda': {
        'name': 'Franka Emika Panda',
        'manufacturer': 'Franka Emika',
        'dof': 7,
        'payload': '3 kg',
        'reach': '855 mm',
        'urdf': 'franka_panda/panda.urdf',
        'description': '7-DOF research robot with torque sensing (alias)',
    },

    # KUKA LBR iiwa
    'iiwa7': {
        'name': 'KUKA LBR iiwa 7 R800',
        'manufacturer': 'KUKA',
        'dof': 7,
        'payload': '7 kg',
        'reach': '800 mm',
        'urdf': 'kuka_iiwa/iiwa7/iiwa7.urdf',
        'description': '7-DOF collaborative robot with torque sensors',
    },
    'iiwa14': {
        'name': 'KUKA LBR iiwa 14 R820',
        'manufacturer': 'KUKA',
        'dof': 7,
        'payload': '14 kg',
        'reach': '820 mm',
        'urdf': 'kuka_iiwa/iiwa14/iiwa14.urdf',
        'description': '7-DOF collaborative robot with torque sensors',
    },
    'kuka_iiwa': {
        'name': 'KUKA LBR iiwa 14 R820',
        'manufacturer': 'KUKA',
        'dof': 7,
        'payload': '14 kg',
        'reach': '820 mm',
        'urdf': 'kuka_iiwa/iiwa14/iiwa14.urdf',
        'description': '7-DOF collaborative robot (alias for iiwa14)',
    },

    # Kinova
    'gen3': {
        'name': 'Kinova Gen3',
        'manufacturer': 'Kinova',
        'dof': 7,
        'payload': '4 kg',
        'reach': '902 mm',
        'urdf': 'kinova/gen3/gen3.urdf',
        'description': '7-DOF lightweight robot arm',
    },
    'kinova_gen3': {
        'name': 'Kinova Gen3',
        'manufacturer': 'Kinova',
        'dof': 7,
        'payload': '4 kg',
        'reach': '902 mm',
        'urdf': 'kinova/gen3/gen3.urdf',
        'description': '7-DOF lightweight robot arm (alias)',
    },
    'jaco_6dof': {
        'name': 'Kinova Jaco 6-DOF',
        'manufacturer': 'Kinova',
        'dof': 6,
        'payload': '1.6 kg',
        'reach': '900 mm',
        'urdf': 'kinova/jaco/jaco_6dof.urdf',
        'description': '6-DOF assistive robot arm',
    },
    'jaco_7dof': {
        'name': 'Kinova Jaco 7-DOF',
        'manufacturer': 'Kinova',
        'dof': 7,
        'payload': '1.6 kg',
        'reach': '900 mm',
        'urdf': 'kinova/jaco/jaco_7dof.urdf',
        'description': '7-DOF assistive robot arm',
    },

    # Fanuc Industrial Robots
    'fanuc_lrmate': {
        'name': 'Fanuc LR Mate 200iB',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '5 kg',
        'reach': '704 mm',
        'urdf': 'fanuc/lrmate200ib.urdf',
        'description': '6-DOF compact industrial robot',
    },
    'fanuc_m16ib': {
        'name': 'Fanuc M-16iB',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '16 kg',
        'reach': '1885 mm',
        'urdf': 'fanuc/m16ib.urdf',
        'description': '6-DOF industrial robot',
    },

    # Fanuc CRX Collaborative Robots
    'crx5ia': {
        'name': 'Fanuc CRX-5iA',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '5 kg',
        'reach': '994 mm',
        'urdf': 'fanuc_crx/crx5ia.urdf',
        'description': '6-DOF collaborative robot, compact',
    },
    'crx10ia': {
        'name': 'Fanuc CRX-10iA',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '10 kg',
        'reach': '1249 mm',
        'urdf': 'fanuc_crx/crx10ia.urdf',
        'description': '6-DOF collaborative robot, medium payload',
    },
    'crx10ia_l': {
        'name': 'Fanuc CRX-10iA/L',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '10 kg',
        'reach': '1418 mm',
        'urdf': 'fanuc_crx/crx10ia_l.urdf',
        'description': '6-DOF collaborative robot, long reach',
    },
    'crx20ia_l': {
        'name': 'Fanuc CRX-20iA/L',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '20 kg',
        'reach': '1418 mm',
        'urdf': 'fanuc_crx/crx20ia_l.urdf',
        'description': '6-DOF collaborative robot, high payload',
    },
    'crx30ia': {
        'name': 'Fanuc CRX-30iA',
        'manufacturer': 'Fanuc',
        'dof': 6,
        'payload': '30 kg',
        'reach': '1252 mm',
        'urdf': 'fanuc_crx/crx30ia.urdf',
        'description': '6-DOF collaborative robot, heavy payload',
    },

    # ABB
    'abb_irb2400': {
        'name': 'ABB IRB 2400',
        'manufacturer': 'ABB',
        'dof': 6,
        'payload': '7-20 kg',
        'reach': '1550 mm',
        'urdf': 'abb/irb2400.urdf',
        'description': '6-DOF industrial robot',
    },

    # xArm (existing)
    'xarm6': {
        'name': 'UFactory xArm6',
        'manufacturer': 'UFactory',
        'dof': 6,
        'payload': '5 kg',
        'reach': '700 mm',
        'urdf': 'xarm/xarm6_robot.urdf',
        'description': '6-DOF robot arm',
    },
    'xarm6_gripper': {
        'name': 'UFactory xArm6 with Gripper',
        'manufacturer': 'UFactory',
        'dof': 6,
        'payload': '5 kg',
        'reach': '700 mm',
        'urdf': 'xarm/xarm6_with_gripper.urdf',
        'description': '6-DOF robot arm with gripper',
    },

    # Robotiq Grippers
    'robotiq_2f_85': {
        'name': 'Robotiq 2F-85',
        'manufacturer': 'Robotiq',
        'dof': 1,
        'payload': 'N/A',
        'reach': '85 mm stroke',
        'urdf': 'robotiq/robotiq_2f_85.urdf',
        'description': 'Adaptive parallel gripper, 85mm stroke',
    },
    'robotiq_2f_140': {
        'name': 'Robotiq 2F-140',
        'manufacturer': 'Robotiq',
        'dof': 1,
        'payload': 'N/A',
        'reach': '140 mm stroke',
        'urdf': 'robotiq/robotiq_2f_140.urdf',
        'description': 'Adaptive parallel gripper, 140mm stroke',
    },
}

# =============================================================================
# Public API Functions
# =============================================================================


def get_robot_urdf(robot_name: str) -> str:
    """
    Get the path to a robot's URDF file.

    Args:
        robot_name: Robot identifier (e.g., 'ur5', 'panda', 'iiwa14')

    Returns:
        Absolute path to the URDF file

    Raises:
        ValueError: If robot_name is not supported
        FileNotFoundError: If URDF file doesn't exist

    Example:
        >>> urdf_path = get_robot_urdf('ur5')
        >>> urdf_path = get_robot_urdf('panda')
    """
    robot_key = robot_name.lower().replace('-', '_').replace(' ', '_')

    if robot_key not in ROBOT_DATABASE:
        available = list_robots()
        raise ValueError(
            f"Unknown robot: '{robot_name}'. "
            f"Available robots: {', '.join(available[:10])}..."
        )

    urdf_rel_path = ROBOT_DATABASE[robot_key]['urdf']
    urdf_path = _MODULE_PATH / urdf_rel_path

    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF file not found: {urdf_path}. "
            f"The robot model may not be installed."
        )

    return str(urdf_path)


def get_robot_info(robot_name: str) -> Dict:
    """
    Get detailed information about a robot.

    Args:
        robot_name: Robot identifier

    Returns:
        Dictionary with robot information

    Example:
        >>> info = get_robot_info('ur5')
        >>> print(f"DOF: {info['dof']}, Reach: {info['reach']}")
    """
    robot_key = robot_name.lower().replace('-', '_').replace(' ', '_')

    if robot_key not in ROBOT_DATABASE:
        raise ValueError(f"Unknown robot: '{robot_name}'")

    info = ROBOT_DATABASE[robot_key].copy()
    info['urdf_path'] = get_robot_urdf(robot_name)
    info['available'] = os.path.exists(info['urdf_path'])

    return info


def list_robots(manufacturer: Optional[str] = None) -> List[str]:
    """
    List all available robot models.

    Args:
        manufacturer: Filter by manufacturer (e.g., 'Universal Robots', 'KUKA')

    Returns:
        List of robot identifiers

    Example:
        >>> all_robots = list_robots()
        >>> ur_robots = list_robots('Universal Robots')
    """
    robots = []

    for key, info in ROBOT_DATABASE.items():
        # Skip aliases
        if 'alias' in info.get('description', '').lower():
            continue

        if manufacturer is None:
            robots.append(key)
        elif info.get('manufacturer', '').lower() == manufacturer.lower():
            robots.append(key)

    return sorted(robots)


def list_manufacturers() -> List[str]:
    """
    List all robot manufacturers.

    Returns:
        List of manufacturer names
    """
    manufacturers = set()
    for info in ROBOT_DATABASE.values():
        if 'manufacturer' in info:
            manufacturers.add(info['manufacturer'])
    return sorted(manufacturers)


def get_robots_by_dof(dof: int) -> List[str]:
    """
    Get robots with a specific number of DOFs.

    Args:
        dof: Number of degrees of freedom

    Returns:
        List of robot identifiers

    Example:
        >>> six_dof_robots = get_robots_by_dof(6)
        >>> seven_dof_robots = get_robots_by_dof(7)
    """
    robots = []
    for key, info in ROBOT_DATABASE.items():
        if 'alias' in info.get('description', '').lower():
            continue
        if info.get('dof') == dof:
            robots.append(key)
    return sorted(robots)


def check_robot_available(robot_name: str) -> bool:
    """
    Check if a robot's URDF file is available.

    Args:
        robot_name: Robot identifier

    Returns:
        True if URDF file exists
    """
    try:
        path = get_robot_urdf(robot_name)
        return os.path.exists(path)
    except (ValueError, FileNotFoundError):
        return False


def get_all_available_robots() -> Dict[str, Dict]:
    """
    Get information about all available robots.

    Returns:
        Dictionary mapping robot names to their info
    """
    available = {}
    for key in list_robots():
        try:
            urdf_path = get_robot_urdf(key)
            if os.path.exists(urdf_path):
                available[key] = get_robot_info(key)
        except (ValueError, FileNotFoundError):
            continue
    return available


def print_robot_catalog():
    """
    Print a formatted catalog of all available robots.
    """
    print("=" * 70)
    print("ManipulaPy Robot Catalog")
    print("=" * 70)

    for manufacturer in list_manufacturers():
        robots = list_robots(manufacturer)
        if not robots:
            continue

        print(f"\n{manufacturer}")
        print("-" * len(manufacturer))

        for robot in robots:
            info = ROBOT_DATABASE[robot]
            available = check_robot_available(robot)
            status = "✓" if available else "✗"
            print(f"  {status} {robot:20s} {info['dof']}-DOF  {info.get('payload', 'N/A'):>10s}  {info.get('description', '')[:35]}")

    print("\n" + "=" * 70)


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================


def get_robot_path(robot_name: str, model: Optional[str] = None) -> str:
    """
    Get the path to a robot's URDF file (legacy API).

    This function is provided for backward compatibility.
    Use get_robot_urdf() for new code.
    """
    if model:
        robot_name = f"{robot_name}_{model}"
    return get_robot_urdf(robot_name)


def list_available_robots() -> List[str]:
    """List all available robots (legacy API)."""
    return list_robots()


def get_ur5_urdf() -> str:
    """Get UR5 URDF path (convenience function)."""
    return get_robot_urdf('ur5')


def get_xarm_urdf(model: str = 'xarm6') -> str:
    """Get xArm URDF path (convenience function)."""
    return get_robot_urdf(model)


def get_panda_urdf() -> str:
    """Get Franka Panda URDF path (convenience function)."""
    return get_robot_urdf('panda')


def get_iiwa_urdf(variant: str = 'iiwa14') -> str:
    """Get KUKA iiwa URDF path (convenience function)."""
    return get_robot_urdf(variant)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Primary API
    'get_robot_urdf',
    'get_robot_info',
    'list_robots',
    'list_manufacturers',
    'get_robots_by_dof',
    'check_robot_available',
    'get_all_available_robots',
    'print_robot_catalog',

    # Convenience functions
    'get_ur5_urdf',
    'get_xarm_urdf',
    'get_panda_urdf',
    'get_iiwa_urdf',

    # Legacy API
    'get_robot_path',
    'list_available_robots',

    # Data
    'ROBOT_DATABASE',
]
