#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Control Module - ManipulaPy

This module provides various control algorithms for robotic manipulators including
PID, computed torque, adaptive, and robust control methods.

Control inputs, guarded dynamics results, and persistent controller state follow the
caller-selected active array backend; NumPy remains the default. Plotting and tuning
are explicit host-only boundaries, while response metrics return Python scalars after
backend-native array operations.

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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
from numpy.typing import NDArray  # noqa: F401

from ManipulaPy.backend import get_backend, use_backend  # noqa: F401

from . import _as_backend_array, _to_host_array, _validate_i_clamp  # noqa: F401

BackendArray = Any


@dataclass(frozen=True)
class _StateOwner:
    """Backend placement recorded for one persistent state value."""

    backend: Any
    token: Any
    value: Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from .computed_torque import _ComputedTorqueConcern  # noqa: E402
from .kalman import _KalmanConcern  # noqa: E402
from .metrics import _MetricsConcern  # noqa: E402
from .pid import _PidConcern  # noqa: E402
from .robust_adaptive import _RobustAdaptiveConcern  # noqa: E402


class ManipulatorController:
    """Manipulator controls with active-backend arrays and persistent state."""

    def __init__(self, dynamics: Any) -> None:
        """
        Initialize the ManipulatorController with the dynamics of the manipulator.

        Inputs and persistent state use the active backend selected by the caller.

        Parameters:
            dynamics (ManipulatorDynamics): An instance of ManipulatorDynamics.
        """
        self.dynamics = dynamics
        self.eint: Optional[BackendArray] = None
        self.parameter_estimate: Optional[BackendArray] = None
        self.P: Optional[BackendArray] = None
        self.x_hat: Optional[BackendArray] = None
        self._state_owners: Dict[str, _StateOwner] = {}

    def _normalize_state(self, name: str) -> Optional[BackendArray]:
        """Move persistent numeric state to the caller-selected backend."""
        value = getattr(self, name)
        if value is None:
            return None
        backend = get_backend()
        token = backend.cache_token()
        state_owners = getattr(self, "_state_owners", None)
        if state_owners is None:
            state_owners = {}
            self._state_owners = state_owners
        owner = state_owners.get(name)
        if owner is not None and owner.value is value:
            if owner.backend is not backend or owner.token != token:
                value = owner.backend.to_numpy(value)
        value = backend.asarray(value)
        setattr(self, name, value)
        state_owners[name] = _StateOwner(backend, token, value)
        return value

    def _set_state(self, name: str, value: BackendArray) -> BackendArray:
        """Store backend-native persistent numeric state."""
        if not hasattr(self, "_state_owners"):
            self._state_owners = {}
        setattr(self, name, value)
        backend = get_backend()
        self._state_owners[name] = _StateOwner(backend, backend.cache_token(), value)
        return value

    computed_torque_control = _ComputedTorqueConcern.__dict__["computed_torque_control"]
    feedforward_control = _ComputedTorqueConcern.__dict__["feedforward_control"]
    pd_feedforward_control = _ComputedTorqueConcern.__dict__["pd_feedforward_control"]
    enforce_limits = _ComputedTorqueConcern.__dict__["enforce_limits"]
    joint_space_control = _ComputedTorqueConcern.__dict__["joint_space_control"]
    cartesian_space_control = _ComputedTorqueConcern.__dict__["cartesian_space_control"]
    pd_control = _PidConcern.__dict__["pd_control"]
    pid_control = _PidConcern.__dict__["pid_control"]
    robust_control = _RobustAdaptiveConcern.__dict__["robust_control"]
    adaptive_control = _RobustAdaptiveConcern.__dict__["adaptive_control"]
    kalman_filter_predict = _KalmanConcern.__dict__["kalman_filter_predict"]
    kalman_filter_update = _KalmanConcern.__dict__["kalman_filter_update"]
    kalman_filter_control = _KalmanConcern.__dict__["kalman_filter_control"]
    plot_steady_state_response = _MetricsConcern.__dict__["plot_steady_state_response"]
    calculate_rise_time = _MetricsConcern.__dict__["calculate_rise_time"]
    calculate_percent_overshoot = _MetricsConcern.__dict__[
        "calculate_percent_overshoot"
    ]
    calculate_settling_time = _MetricsConcern.__dict__["calculate_settling_time"]
    calculate_steady_state_error = _MetricsConcern.__dict__[
        "calculate_steady_state_error"
    ]
    ziegler_nichols_tuning = _MetricsConcern.__dict__["ziegler_nichols_tuning"]
    tune_controller = _MetricsConcern.__dict__["tune_controller"]
    find_ultimate_gain_and_period = _MetricsConcern.__dict__[
        "find_ultimate_gain_and_period"
    ]


del (
    _ComputedTorqueConcern,
    _KalmanConcern,
    _MetricsConcern,
    _PidConcern,
    _RobustAdaptiveConcern,
)
