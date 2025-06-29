#!/usr/bin/env python3
"""
xArm Robot Data Module

This module provides URDF files and mesh data for the xArm robotic manipulator.
"""

import os
from pathlib import Path

# Get the directory containing this module
_MODULE_DIR = Path(__file__).parent

# Available URDF files in the xarm directory
AVAILABLE_URDFS = {
    'base': _MODULE_DIR / "base.urdf",
    'base_com': _MODULE_DIR / "base_com.urdf", 
    'xarm6_robot': _MODULE_DIR / "xarm6_robot.urdf",
    'xarm6_robot_white': _MODULE_DIR / "xarm6_robot_white.urdf",
    'xarm6_with_gripper': _MODULE_DIR / "xarm6_with_gripper.urdf",
    'link1': _MODULE_DIR / "link1.urdf",
    'link1_com': _MODULE_DIR / "link1_com.urdf",
    'link2': _MODULE_DIR / "link2.urdf",
    'link2_com': _MODULE_DIR / "link2_com.urdf",
    'link3': _MODULE_DIR / "link3.urdf",
    'link3_com': _MODULE_DIR / "link3_com.urdf",
    'link4': _MODULE_DIR / "link4.urdf",
    'link4_com': _MODULE_DIR / "link4_com.urdf",
    'link5': _MODULE_DIR / "link5.urdf",
    'link5_com': _MODULE_DIR / "link5_com.urdf",
    'link6': _MODULE_DIR / "link6.urdf",
    'link6_com': _MODULE_DIR / "link6_com.urdf",
}

# Default URDF file (the main complete robot)
urdf_file = str(AVAILABLE_URDFS['xarm6_robot'])

# Fallback if the main URDF doesn't exist
if not AVAILABLE_URDFS['xarm6_robot'].exists():
    # Try alternative URDFs
    for name, path in AVAILABLE_URDFS.items():
        if path.exists():
            urdf_file = str(path)
            print(f"Warning: Using fallback URDF: {name}")
            break
    else:
        # If no URDF files exist, create a basic one
        urdf_file = str(_MODULE_DIR / "basic_xarm.urdf")
        _create_basic_xarm_urdf()

def _create_basic_xarm_urdf():
    """Create a basic xArm URDF file for testing purposes."""
    basic_urdf_content = '''<?xml version="1.0"?>
<robot name="xarm6">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Link 1 -->
  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.267"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.28" upper="6.28" effort="50" velocity="3.14"/>
  </joint>

  <!-- Link 2 -->
  <link name="link2">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.289"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.267" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.09" upper="2.09" effort="50" velocity="3.14"/>
  </joint>

  <!-- Link 3 -->
  <link name="link3">
    <visual>
      <geometry>
        <cylinder radius="0.045" length="0.077"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0 0 0.289" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-6.28" upper="6.28" effort="30" velocity="3.14"/>
  </joint>

  <!-- Link 4 -->
  <link name="link4">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3425"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0.2 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="joint4" type="revolute">
    <parent link="link3"/>
    <child link="link4"/>
    <origin xyz="0.077 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-6.28" upper="6.28" effort="30" velocity="3.14"/>
  </joint>

  <!-- Link 5 -->
  <link name="link5">
    <visual>
      <geometry>
        <cylinder radius="0.035" length="0.076"/>
      </geometry>
      <material name="purple">
        <color rgba="0.8 0.2 0.8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="joint5" type="revolute">
    <parent link="link4"/>
    <child link="link5"/>
    <origin xyz="0.3425 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.09" upper="2.09" effort="20" velocity="3.14"/>
  </joint>

  <!-- Link 6 (End Effector) -->
  <link name="link6">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.076"/>
      </geometry>
      <material name="cyan">
        <color rgba="0.2 0.8 0.8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="joint6" type="revolute">
    <parent link="link5"/>
    <child link="link6"/>
    <origin xyz="0.076 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-6.28" upper="6.28" effort="20" velocity="3.14"/>
  </joint>

</robot>'''
    
    with open(_MODULE_DIR / "basic_xarm.urdf", 'w') as f:
        f.write(basic_urdf_content)
    print(f"Created basic xArm URDF at: {_MODULE_DIR / 'basic_xarm.urdf'}")

def get_urdf_path(urdf_name='xarm6_robot'):
    """
    Get the path to a specific xArm URDF file.
    
    Parameters
    ----------
    urdf_name : str
        Name of the URDF file to retrieve. Available options:
        'base', 'base_com', 'xarm6_robot', 'xarm6_robot_white', 
        'xarm6_with_gripper', 'link1', 'link1_com', etc.
    
    Returns
    -------
    str
        Absolute path to the requested URDF file
        
    Raises
    ------
    FileNotFoundError
        If the requested URDF file doesn't exist
    """
    if urdf_name in AVAILABLE_URDFS:
        urdf_path = AVAILABLE_URDFS[urdf_name]
        if urdf_path.exists():
            return str(urdf_path.absolute())
        else:
            raise FileNotFoundError(f"URDF file '{urdf_name}' not found at {urdf_path}")
    else:
        available = list(AVAILABLE_URDFS.keys())
        raise ValueError(f"Unknown URDF name '{urdf_name}'. Available: {available}")

def get_mesh_directory():
    """
    Get the directory containing mesh files.
    
    Returns
    -------
    str
        Path to the mesh directory
    """
    return str(_MODULE_DIR / "xarm_description")

def list_available_urdfs():
    """
    List all available URDF files in the xarm directory.
    
    Returns
    -------
    dict
        Dictionary mapping URDF names to their availability status
    """
    status = {}
    for name, path in AVAILABLE_URDFS.items():
        status[name] = path.exists()
    return status

# Make sure urdf_file points to an existing file
if not Path(urdf_file).exists():
    print(f"Warning: Default URDF {urdf_file} not found. Available URDFs:")
    for name, exists in list_available_urdfs().items():
        print(f"  {name}: {'✓' if exists else '✗'}")
