#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
IK Helper Functions - ManipulaPy (compatibility shim)

The implementation now lives in :mod:`ManipulaPy.ik.ik_helpers`. This module is
kept as a compatibility shim so the historical import path
``from ManipulaPy.ik_helpers import ...`` keeps working unchanged. It re-exports
the full public namespace of ``ManipulaPy.ik.ik_helpers``.

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

from .ik.ik_helpers import (
    Callable,
    IKInitialGuessCache,
    List,
    NDArray,
    Optional,
    Tuple,
    Union,
    __all__,
    _clip_to_limits,
    adaptive_multi_start_ik,
    extrapolate_from_current,
    midpoint_of_limits,
    np,
    random_in_limits,
    workspace_heuristic_guess,
)
