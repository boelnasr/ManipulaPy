"""
Manipulapy Package

This package provides tools for the analysis and manipulation of robotic systems, including kinematics,
dynamics, singularity analysis, path planning, and URDF processing utilities.
"""

# Import main modules for easier access
from .kinematics import *
from .dynamics import *
from .singularity import *
from .path_planning import *
from .utils import *
from .urdf_processor import *
from .controller import *
# Define package-level variables
__version__ = '1.0.1'
__author__ = 'Mohamed Aboelnar'

__all__ = [
    'kinematics',
    'dynamics',
    'singularity',
    'path_planning',
    'utils',
    'urdf_processor',
    'controller'

]