#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kalman filter controller concerns."""

from __future__ import annotations

from typing import Any, Tuple

from . import manipulator_controller as _runtime

BackendArray = Any


class _KalmanConcern:
    """Descriptor container for the kalman filter controller concerns."""

    def kalman_filter_predict(
        self,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        taulist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        dt: float,
        Q: BackendArray,
    ) -> None:
        """
        Kalman Filter Prediction.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            taulist: Applied torques.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            dt: Time step.
            Q: Process noise covariance.

        Returns:
            None
        """
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)
        taulist = _runtime._as_backend_array(taulist)
        g = _runtime._as_backend_array(g)
        Ftip = _runtime._as_backend_array(Ftip)
        Q = _runtime._as_backend_array(Q)
        backend = _runtime.get_backend()
        self._normalize_state("x_hat")
        self._normalize_state("P")
        n = self.x_hat.shape[0] if self.x_hat is not None else len(thetalist) * 2
        if Q.shape != (n, n):
            raise ValueError(f"Q must have shape ({n}, {n}), got {Q.shape}")

        if self.x_hat is None:
            self._set_state("x_hat", backend.concatenate((thetalist, dthetalist)))

        thetalist_pred = (
            self.x_hat[: len(thetalist)] + self.x_hat[len(thetalist) :] * dt
        )
        dthetalist_pred = (
            _runtime._as_backend_array(
                self.dynamics.forward_dynamics(
                    self.x_hat[: len(thetalist)],
                    self.x_hat[len(thetalist) :],
                    taulist,
                    g,
                    Ftip,
                )
            )
            * dt
            + self.x_hat[len(thetalist) :]
        )
        x_hat_pred = backend.concatenate((thetalist_pred, dthetalist_pred))

        if self.P is None:
            self._set_state("P", backend.eye(len(x_hat_pred)))
        F = backend.eye(len(x_hat_pred))
        self._set_state("P", F @ self.P @ F.T + Q)

        self._set_state("x_hat", x_hat_pred)

    def kalman_filter_update(self, z: BackendArray, R: BackendArray) -> None:
        """
        Kalman Filter Update.

        Inputs follow the active backend selected by the caller.

        Parameters:
            z: Measurement vector.
            R: Measurement noise covariance.

        Returns:
            None
        """
        z = _runtime._as_backend_array(z)
        R = _runtime._as_backend_array(R)
        if self.x_hat is None:
            raise ValueError(
                "kalman_filter_update called before kalman_filter_predict; "
                "x_hat has not been initialized"
            )
        backend = _runtime.get_backend()
        self._normalize_state("x_hat")
        n = self.x_hat.shape[0]
        if self.P is None:
            raise ValueError(
                f"P must be initialized with shape ({n}, {n}) before update; "
                "got None"
            )
        self._normalize_state("P")
        if getattr(self.P, "shape", None) != (n, n):
            raise ValueError(
                f"P must be initialized with shape ({n}, {n}) before update; "
                f"got {self.P.shape}"
            )
        if z.shape != (n,):
            raise ValueError(f"z must have shape ({n},) to match x_hat, got {z.shape}")
        if R.shape != (n, n):
            raise ValueError(f"R must have shape ({n}, {n}), got {R.shape}")

        H = backend.eye(len(self.x_hat))
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ backend.inv(S)
        self._set_state("x_hat", self.x_hat + K @ y)
        self._set_state("P", (backend.eye(len(self.x_hat)) - K @ H) @ self.P)

    def kalman_filter_control(
        self,
        thetalistd: BackendArray,
        dthetalistd: BackendArray,
        thetalist: BackendArray,
        dthetalist: BackendArray,
        taulist: BackendArray,
        g: BackendArray,
        Ftip: BackendArray,
        dt: float,
        Q: BackendArray,
        R: BackendArray,
    ) -> Tuple[BackendArray, BackendArray]:
        """
        Kalman Filter Control.

        Inputs follow the active backend selected by the caller.

        Parameters:
            thetalistd: Desired joint angles.
            dthetalistd: Desired joint velocities.
            thetalist: Current joint angles.
            dthetalist: Current joint velocities.
            taulist: Applied torques.
            g: Gravity vector.
            Ftip: External forces applied at the end effector.
            dt: Time step.
            Q: Process noise covariance.
            R: Measurement noise covariance.

        Returns:
            tuple: Estimated angles and velocities (NumPy arrays by default).
        """
        thetalist = _runtime._as_backend_array(thetalist)
        dthetalist = _runtime._as_backend_array(dthetalist)

        self.kalman_filter_predict(thetalist, dthetalist, taulist, g, Ftip, dt, Q)
        self.kalman_filter_update(
            _runtime.get_backend().concatenate((thetalist, dthetalist)), R
        )
        return self.x_hat[: len(thetalist)], self.x_hat[len(thetalist) :]
