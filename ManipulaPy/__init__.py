#!/usr/bin/env python3

"""
ManipulaPy Package

This package provides tools for the analysis and manipulation of robotic systems, including kinematics,
dynamics, singularity analysis, path planning, control, simulation, and vision-based perception.
"""

# Import main modules for easier access
from ManipulaPy.kinematics import *
from ManipulaPy.dynamics import *
from ManipulaPy.singularity import *
from ManipulaPy.path_planning import *
from ManipulaPy.utils import *
from ManipulaPy.urdf_processor import *
from ManipulaPy.vision import *
from ManipulaPy.perception import *
from ManipulaPy.control import *
from ManipulaPy.sim import *
from ManipulaPy.potential_field import *
from ManipulaPy.cuda_kernels import *

# Define package-level variables
__version__ = "1.1.0"
__author__ = "Mohamed Aboelnar"

__all__ = [
    "kinematics",
    "dynamics",
    "singularity",
    "path_planning",
    "utils",
    "urdf_processor",
    "vision",
    "perception",
    "control",
    "sim",
    "potential_field",
    "cuda_kernels"
]
