#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Potential Field Module - ManipulaPy

This module provides potential field path planning capabilities including attractive
and repulsive potential computations, gradient calculations, and collision checking
for robotic manipulator motion planning in cluttered environments.

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

import itertools  # noqa: F401
import logging  # noqa: F401
from typing import Any, Dict, Iterable, Set  # noqa: F401

import numpy as np  # noqa: F401
from numpy.typing import NDArray  # noqa: F401
from scipy.spatial import ConvexHull  # noqa: F401

from ..urdf import URDF  # noqa: F401
from .adjacency import build_link_adjacency  # noqa: F401
from .collision import CollisionChecker  # noqa: F401
from .fields import PotentialField  # noqa: F401
