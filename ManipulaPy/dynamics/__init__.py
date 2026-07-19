#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dynamics Module - ManipulaPy

This module provides classes and functions for manipulator dynamics analysis
including mass matrix computation, Coriolis forces, gravity compensation, and
inverse/forward dynamics.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

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

from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

import numpy as np  # noqa: F401
from numpy.typing import NDArray  # noqa: F401

from ..kinematics import SerialManipulator  # noqa: F401
from ..utils import adjoint_transform as ad  # noqa: F401
from .manipulator_dynamics import ManipulatorDynamics  # noqa: F401

# Importing the facade loads its concern modules and registers them as package
# attributes. Keep those implementation details out of the historical namespace.
for _concern_module in ("cache", "forces", "id_fd", "mass_matrix"):
    globals().pop(_concern_module, None)
del _concern_module
