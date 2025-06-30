#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ManipulaPy Package

This package provides tools for the analysis and manipulation of robotic systems, including kinematics,
dynamics, singularity analysis, path planning, and URDF processing utilities.

License: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
Copyright (c) 2025 Mohamed Aboelnar

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




# ---------------------------------------------------------------------
# Re-export key submodules for convenient top-level access
# (import order chosen to avoid circular dependencies)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------
__version__ = "1.1.0"            
__author__  = "Mohamed Aboelnar"
__license__ = "AGPL-3.0-or-later"

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
    "cuda_kernels",
]
