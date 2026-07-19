#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Simulation Module - ManipulaPy

This module provides PyBullet-based simulation capabilities for robotic manipulators
including real-time visualization, physics simulation, and interactive control.

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

from .simulation import (
    Any,
    List,
    ManipulatorController,
    Optional,
    Sequence,
    Simulation,
    Tuple,
    _PYBULLET_AVAILABLE,
    logging,
    np,
    os,
    p,
    plt,
    pybullet_data,
    time,
    tp,
)
