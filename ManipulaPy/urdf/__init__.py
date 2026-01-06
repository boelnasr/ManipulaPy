#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ManipulaPy Native URDF Parser

A modern, optimized URDF parser built specifically for ManipulaPy.
Combines best practices from yourdfpy, urchin, and urdfpy while being
tailored for robotics kinematics and dynamics workflows.

Features:
    - Zero external URDF dependencies (only numpy required for core)
    - NumPy 2.0+ compatible
    - Direct SerialManipulator/ManipulatorDynamics conversion
    - Optional visualization (lazy-loaded trimesh/pybullet)
    - Xacro macro expansion support

Example:
    >>> from ManipulaPy.urdf import URDF
    >>> robot = URDF.load("robot.urdf")
    >>> manipulator = robot.to_serial_manipulator()
    >>> robot.show()

Copyright (c) 2025 Mohamed Aboelnasr
"""

from .core import URDF
from .types import (
    Link,
    Joint,
    JointType,
    Inertial,
    Visual,
    Collision,
    Origin,
    JointLimit,
    JointDynamics,
    JointMimic,
    Material,
    Box,
    Cylinder,
    Sphere,
    Mesh,
    # Transmission types
    Transmission,
    TransmissionJoint,
    Actuator,
)
from .resolver import PackageResolver
from .validation import (
    URDFValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_urdf,
)
from .scene import Scene, RobotInstance
from .modifiers import URDFModifier, load_calibration, save_calibration

__all__ = [
    # Main class
    "URDF",
    # Path resolution
    "PackageResolver",
    # Validation
    "URDFValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_urdf",
    # Scene/Multi-robot
    "Scene",
    "RobotInstance",
    # Data types
    "Link",
    "Joint",
    "JointType",
    "Inertial",
    "Visual",
    "Collision",
    "Origin",
    "JointLimit",
    "JointDynamics",
    "JointMimic",
    "Material",
    "Box",
    "Cylinder",
    "Sphere",
    "Mesh",
    # Transmission types
    "Transmission",
    "TransmissionJoint",
    "Actuator",
    # Modifiers
    "URDFModifier",
    "load_calibration",
    "save_calibration",
]

__version__ = "1.3.0"
