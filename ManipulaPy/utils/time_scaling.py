#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Time-Scaling Utilities - ManipulaPy

Trajectory time-scaling functions (cubic and quintic) used to parameterize
motion between waypoints.

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


def CubicTimeScaling(Tf: float, t: float) -> float:
    """
    Compute the cubic time scaling factor.

    Parameters:
        Tf (float): The total time of the motion.
        t (float): The current time.

    Returns:
        float: The cubic time scaling factor.
    """
    return 3 * (t / Tf) ** 2 - 2 * (t / Tf) ** 3


def QuinticTimeScaling(Tf: float, t: float) -> float:
    """
    Compute the quintic time scaling factor.

    Parameters:
        Tf (float): The total time of the motion.
        t (float): The current time.

    Returns:
        float: The quintic time scaling factor.
    """
    return 10 * (t / Tf) ** 3 - 15 * (t / Tf) ** 4 + 6 * (t / Tf) ** 5
