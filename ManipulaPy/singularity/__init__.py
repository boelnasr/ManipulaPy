#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Singularity Analysis Module - ManipulaPy

This module provides comprehensive singularity analysis capabilities for robotic
manipulators including singularity detection, manipulability ellipsoid visualization,
workspace analysis, and condition number computations with optional CUDA acceleration.

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

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.spatial import ConvexHull

from .singularity_analysis import Singularity
