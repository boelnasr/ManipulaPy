#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PD and PID controller concerns."""

from __future__ import annotations

from typing import Any, Optional

from . import manipulator_controller as _runtime

BackendArray = Any


class _PidConcern:
    """Descriptor container for the pd and pid controller concerns."""

    def pd_control(
        self,
        desired_position: BackendArray,
        desired_velocity: BackendArray,
        current_position: BackendArray,
        current_velocity: BackendArray,
        Kp: BackendArray,
        Kd: BackendArray,
    ) -> BackendArray:
        """
        PD Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            desired_position: Desired joint positions.
            desired_velocity: Desired joint velocities.
            current_position: Current joint positions.
            current_velocity: Current joint velocities.
            Kp: Proportional gain.
            Kd: Derivative gain.

        Returns:
            NDArray: PD signal (NumPy array under the default backend).
        """
        desired_position = _runtime._as_backend_array(desired_position)
        desired_velocity = _runtime._as_backend_array(desired_velocity)
        current_position = _runtime._as_backend_array(current_position)
        current_velocity = _runtime._as_backend_array(current_velocity)
        Kp = _runtime._as_backend_array(Kp)
        Kd = _runtime._as_backend_array(Kd)

        e = desired_position - current_position
        edot = desired_velocity - current_velocity
        pd_signal = Kp * e + Kd * edot
        return pd_signal

    def pid_control(
        self,
        thetalistd: BackendArray,
        dthetalistd: BackendArray,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        dt: float,
        Kp: BackendArray,
        Ki: BackendArray,
        Kd: BackendArray,
        i_clamp: Optional[float] = None,
    ) -> BackendArray:
        """
        PID Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalistd: Desired joint angles.
            dthetalistd: Desired joint velocities.
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            dt: Time step.
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.

        Returns:
            NDArray: PID signal (NumPy array under the default backend).
        """
        thetalistd = _runtime._as_backend_array(thetalistd)
        dthetalistd = _runtime._as_backend_array(dthetalistd)
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        Kp = _runtime._as_backend_array(Kp)
        Ki = _runtime._as_backend_array(Ki)
        Kd = _runtime._as_backend_array(Kd)

        backend = _runtime.get_backend()
        self._normalize_state("eint")
        if self.eint is None or self.eint.shape != thetalist.shape:
            if self.eint is not None and self.eint.shape != thetalist.shape:
                _runtime.logger.debug(
                    "Controller state shape mismatch (%s vs %s); resetting integral.",
                    self.eint.shape,
                    thetalist.shape,
                )
            self._set_state(
                "eint", backend.zeros(thetalist.shape, dtype=backend.float64)
            )

        e = thetalistd - thetalist
        self._set_state("eint", self.eint + e * dt)
        i_clamp = _runtime._validate_i_clamp(i_clamp)
        if i_clamp is not None:
            self._set_state("eint", backend.clip(self.eint, -i_clamp, i_clamp))

        e_dot = dthetalistd - dthetalist
        tau = Kp * e + Ki * self.eint + Kd * e_dot
        return tau
