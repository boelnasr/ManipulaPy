#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Path Planning Compatibility Shim - ManipulaPy

The trajectory planning implementation now lives in the ManipulaPy.planning
package (ManipulaPy.planning.trajectory_planning). This module is kept so that
the historical import path ManipulaPy.path_planning keeps working unchanged.

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

from . import planning as _planning

# Re-export the planning package's full namespace so that every name that used to
# live in this module stays importable from ManipulaPy.path_planning. This includes
# the single-underscore helpers that existing callers/tests import (e.g.
# _traj_cpu_njit); only dunders are skipped so this module keeps its own machinery.
# Done dynamically to stay faithful to the CUDA-conditional public surface.
globals().update(
    {
        name: value
        for name, value in vars(_planning).items()
        if not name.startswith("__")
    }
)

__all__ = list(_planning.__all__)
