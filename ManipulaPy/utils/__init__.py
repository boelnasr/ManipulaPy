#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Utilities Module - ManipulaPy

This module contains essential utility functions for working with rigid body motions,
transformations, and related operations in robotic manipulation. Functions cover
screw theory, matrix operations, Lie algebra computations, time scaling, and
coordinate transformations for kinematics and dynamics calculations.

The functions in this module support:
- Extracting and manipulating screw vectors and twists
- Computing transformation matrices from twists and joint angles
- Matrix logarithms and exponentials for SE(3) and SO(3)
- Converting between rotation matrices and Euler angles
- Skew-symmetric matrix operations
- Time scaling functions for trajectory generation

Copyright (c) 2025 Mohamed Aboelnasr

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""

from .screw import (
    extract_omega_list,
    extract_r_list,
    extract_screw_list,
    logm_to_twist,
)
from .time_scaling import CubicTimeScaling, QuinticTimeScaling
from .se3 import (
    MatrixExp6,
    MatrixLog6,
    TransInv,
    TransToRp,
    VecTose3,
    adjoint_transform,
    logm,
    se3ToVec,
    transform_from_twist,
)
from .so3 import (
    MatrixExp3,
    MatrixLog3,
    NearZero,
    VecToso3,
    euler_to_rotation_matrix,
    rotation_logm,
    rotation_matrix_to_euler_angles,
    skew_symmetric,
    skew_symmetric_to_vector,
)
