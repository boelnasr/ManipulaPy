#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ManipulaPy Package

This package provides tools for the analysis and manipulation of robotic systems,
including kinematics, dynamics, singularity analysis, path planning, and URDF processing utilities.

License: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
Copyright (c) 2025 Mohamed Aboelnasr

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import warnings
import sys

# Package metadata
__version__ = "1.1.2"
__author__ = "Mohamed Aboelnasr"
__license__ = "AGPL-3.0-or-later"

# Core modules that should always be available
_CORE_MODULES = []
_OPTIONAL_MODULES = []

# Import core utilities (should always work)
try:
    from ManipulaPy.utils import *
    from ManipulaPy import utils
    _CORE_MODULES.append("utils")
except ImportError as e:
    warnings.warn(f"Failed to import utils module: {e}", ImportWarning)

# Import core functionality that depends on torch
try:
    from ManipulaPy.kinematics import *
    from ManipulaPy import kinematics
    _CORE_MODULES.append("kinematics")
except ImportError as e:
    warnings.warn(f"Failed to import kinematics module: {e}. Ensure PyTorch is installed.", ImportWarning)

try:
    from ManipulaPy.dynamics import *
    from ManipulaPy import dynamics
    _CORE_MODULES.append("dynamics")
except ImportError as e:
    warnings.warn(f"Failed to import dynamics module: {e}. Ensure PyTorch is installed.", ImportWarning)

try:
    from ManipulaPy.singularity import *
    from ManipulaPy import singularity
    _CORE_MODULES.append("singularity")
except ImportError as e:
    warnings.warn(f"Failed to import singularity module: {e}", ImportWarning)

try:
    from ManipulaPy.path_planning import *
    from ManipulaPy import path_planning
    _CORE_MODULES.append("path_planning")
except ImportError as e:
    warnings.warn(f"Failed to import path_planning module: {e}", ImportWarning)

try:
    from ManipulaPy.urdf_processor import *
    from ManipulaPy import urdf_processor
    _CORE_MODULES.append("urdf_processor")
except ImportError as e:
    warnings.warn(f"Failed to import urdf_processor module: {e}", ImportWarning)

try:
    from ManipulaPy.control import *
    from ManipulaPy import control
    _CORE_MODULES.append("control")
except ImportError as e:
    warnings.warn(f"Failed to import control module: {e}", ImportWarning)

try:
    from ManipulaPy.potential_field import *
    from ManipulaPy import potential_field
    _CORE_MODULES.append("potential_field")
except ImportError as e:
    warnings.warn(f"Failed to import potential_field module: {e}", ImportWarning)

# Optional modules - these may have external dependencies
try:
    from ManipulaPy.vision import *
    from ManipulaPy import vision
    _OPTIONAL_MODULES.append("vision")
except ImportError as e:
    warnings.warn(
        f"Vision module not available: {e}. "
        "Install ultralytics and opencv-python for vision features: "
        "pip install ultralytics opencv-python",
        ImportWarning
    )

try:
    from ManipulaPy.perception import *
    from ManipulaPy import perception
    _OPTIONAL_MODULES.append("perception")
except ImportError as e:
    warnings.warn(
        f"Perception module not available: {e}. "
        "Install required dependencies for perception features.",
        ImportWarning
    )

try:
    from ManipulaPy.sim import *
    from ManipulaPy import sim
    _OPTIONAL_MODULES.append("sim")
except ImportError as e:
    warnings.warn(
        f"Simulation module not available: {e}. "
        "Install pybullet for simulation features: pip install pybullet",
        ImportWarning
    )

try:
    from ManipulaPy.cuda_kernels import *
    from ManipulaPy import cuda_kernels
    _OPTIONAL_MODULES.append("cuda_kernels")
except ImportError as e:
    warnings.warn(
        f"CUDA kernels module not available: {e}. "
        "Install CUDA-enabled PyTorch for GPU acceleration.",
        ImportWarning
    )

# Define what gets imported with "from ManipulaPy import *"
__all__ = _CORE_MODULES + _OPTIONAL_MODULES

# Print import summary (only in debug mode or if explicitly requested)
def _print_import_summary():
    """Print summary of successfully imported modules."""
    print(f"ManipulaPy {__version__} loaded successfully!")
    print(f"Core modules: {', '.join(_CORE_MODULES)}")
    if _OPTIONAL_MODULES:
        print(f"Optional modules: {', '.join(_OPTIONAL_MODULES)}")
    
    missing_core = set(["utils", "kinematics", "dynamics"]) - set(_CORE_MODULES)
    if missing_core:
        print(f"⚠️  Missing core modules: {', '.join(missing_core)}")
    
    missing_optional = set(["vision", "perception", "sim", "cuda_kernels"]) - set(_OPTIONAL_MODULES)
    if missing_optional:
        print(f"ℹ️  Unavailable optional modules: {', '.join(missing_optional)}")

# Only show summary if in debug mode or environment variable is set
import os
if os.getenv('MANIPULAPY_VERBOSE', '').lower() in ('1', 'true', 'yes') or __debug__:
    try:
        _print_import_summary()
    except Exception:
        pass  # Don't let summary printing break the import

# Clean up temporary variables
del warnings, sys, os