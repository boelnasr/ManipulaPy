#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Planning Module - ManipulaPy

This module provides optimized trajectory planning capabilities including joint
and Cartesian space trajectory generation with CUDA acceleration and collision
avoidance.

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

from . import trajectory_planning

# Re-export the implementation module's full namespace (public API plus the
# single-underscore helpers that existing callers/tests import) so that every name
# that used to live in the flat ManipulaPy.path_planning module stays importable
# from ManipulaPy.planning. This is done dynamically rather than with an explicit
# name list because part of that namespace (e.g. MockCuda and the CUDA-only kernel
# symbols) is defined conditionally on CUDA availability, so a hard-coded list
# would break imports on machines with a different CUDA configuration. Dunder names
# are skipped so this package keeps its own module machinery.
globals().update(
    {
        name: value
        for name, value in vars(trajectory_planning).items()
        if not name.startswith("__")
    }
)

# The trajectory planning implementation is split across internal helper modules
# (_kernels, trajectory, trajectory_dynamics, collision_host, _plotting). Importing
# trajectory_planning above pulls those submodules in and the import system binds
# them as attributes of this package. They are implementation detail, so drop them
# here to keep the historical public namespace unchanged.
for _internal in (
    "_kernels",
    "trajectory",
    "trajectory_dynamics",
    "collision_host",
    "_plotting",
    "_runtime",
    "_sys",
    "_ModuleType",
    "_FORWARDED_RUNTIME_NAMES",
    "_PlanningCompatibilityModule",
):
    globals().pop(_internal, None)
del _internal

__all__ = list(trajectory_planning.__all__)
