#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Robust and adaptive controller concerns."""

from __future__ import annotations

from typing import Any

from . import manipulator_controller as _runtime

BackendArray = Any


class _RobustAdaptiveConcern:
    """Descriptor container for the robust and adaptive controller concerns."""

    def robust_control(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        ddthetalist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        disturbance_estimate: BackendArray,
        adaptation_gain: BackendArray,
    ) -> BackendArray:
        """
        Robust Control.

        Inputs and guarded dynamics results follow the active backend.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            disturbance_estimate: Estimate of disturbances.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Robust torque (NumPy array under the default backend).
        """
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        ddthetalist = _runtime._as_backend_array(ddthetalist)
        g = _runtime._as_backend_array(g)
        Ftip = _runtime._as_backend_array(Ftip)
        disturbance_estimate = _runtime._as_backend_array(disturbance_estimate)
        adaptation_gain = _runtime._as_backend_array(adaptation_gain)

        M = _runtime._as_backend_array(self.dynamics.mass_matrix(thetalist))
        c = _runtime._as_backend_array(
            self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        )
        g_forces = _runtime._as_backend_array(
            self.dynamics.gravity_forces(thetalist, g)
        )
        J_transpose = _runtime._as_backend_array(self.dynamics.jacobian(thetalist)).T
        tau = (
            M @ ddthetalist
            + c
            + g_forces
            + J_transpose @ Ftip
            + adaptation_gain * disturbance_estimate
        )
        return tau

    def adaptive_control(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        ddthetalist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        measurement_error: BackendArray,
        adaptation_gain: float,
    ) -> BackendArray:
        """
        Adaptive Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            ddthetalist: Desired joint accelerations.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            measurement_error: Error in measurement.
            adaptation_gain: Gain for the adaptation term.

        Returns:
            NDArray: Adaptive torque (NumPy array under the default backend).
        """
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        ddthetalist = _runtime._as_backend_array(ddthetalist)
        g = _runtime._as_backend_array(g)
        Ftip = _runtime._as_backend_array(Ftip)
        measurement_error = _runtime._as_backend_array(measurement_error)
        adaptation_gain = _runtime._as_backend_array(adaptation_gain)

        backend = _runtime.get_backend()
        self._normalize_state("parameter_estimate")
        n = thetalist.size
        if self.parameter_estimate is None:
            self._set_state(
                "parameter_estimate", backend.zeros((n,), dtype=thetalist.dtype)
            )

        err = measurement_error.reshape(-1)
        gamma = adaptation_gain.reshape(-1)[0]

        self._set_state("parameter_estimate", self.parameter_estimate + gamma * err)

        M = _runtime._as_backend_array(self.dynamics.mass_matrix(thetalist))
        c = _runtime._as_backend_array(
            self.dynamics.velocity_quadratic_forces(thetalist, dthetalist)
        )
        g_forces = _runtime._as_backend_array(
            self.dynamics.gravity_forces(thetalist, g)
        )
        J_transpose = _runtime._as_backend_array(self.dynamics.jacobian(thetalist)).T

        tau = (
            M @ ddthetalist
            + c
            + g_forces
            + J_transpose @ Ftip
            + self.parameter_estimate
        )
        return tau
