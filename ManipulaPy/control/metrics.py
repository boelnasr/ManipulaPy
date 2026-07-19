#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Controller plotting, response metrics, and tuning concerns."""

from __future__ import annotations

from typing import Any, List, Tuple, Union

from numpy.typing import NDArray

from . import manipulator_controller as _runtime

BackendArray = Any


class _MetricsConcern:
    """Controller plotting, response metrics, and tuning descriptors."""

    def plot_steady_state_response(
        self,
        time: Union[NDArray[_runtime.np.float64], List[float]],
        response: Union[NDArray[_runtime.np.float64], List[float]],
        set_point: float,
        title: str = "Steady State Response",
    ) -> None:
        """
        Plot the steady-state response of the controller.

        Parameters:
            time (_runtime.np.ndarray): Array of time steps.
            response (_runtime.np.ndarray): Array of response values.
            set_point (float): Desired set point value.
            title (str, optional): Title of the plot.

        Returns:
            None
        """
        time = _runtime._to_host_array(time)
        response = _runtime._to_host_array(response)

        with _runtime.use_backend("numpy"):
            _runtime.plt.figure(figsize=(10, 5))
            _runtime.plt.plot(time, response, label="Response")
            _runtime.plt.axhline(
                y=set_point, color="r", linestyle="--", label="Set Point"
            )

            # Calculate key metrics
            rise_time = self.calculate_rise_time(time, response, set_point)
            percent_overshoot = self.calculate_percent_overshoot(response, set_point)
            settling_time = self.calculate_settling_time(time, response, set_point)
            steady_state_error = self.calculate_steady_state_error(response, set_point)

            # Annotate metrics on the plot
            _runtime.plt.axvline(
                x=rise_time,
                color="g",
                linestyle="--",
                label=f"Rise Time: {rise_time:.2f}s",
            )
            _runtime.plt.axhline(
                y=set_point * (1 + percent_overshoot / 100),
                color="b",
                linestyle="--",
                label=f"Overshoot: {percent_overshoot:.2f}%",
            )
            _runtime.plt.axvline(
                x=settling_time,
                color="m",
                linestyle="--",
                label=f"Settling Time: {settling_time:.2f}s",
            )
            _runtime.plt.axhline(
                y=set_point + steady_state_error,
                color="c",
                linestyle="--",
                label=f"Steady State Error: {steady_state_error:.2f}",
            )

            _runtime.plt.xlabel("Time (s)")
            _runtime.plt.ylabel("Response")
            _runtime.plt.title(title)
            _runtime.plt.legend()
            _runtime.plt.grid(True)
            _runtime.plt.show()

    def calculate_rise_time(
        self,
        time: Union[NDArray[_runtime.np.float64], List[float]],
        response: Union[NDArray[_runtime.np.float64], List[float]],
        set_point: float,
    ) -> float:
        """
        Calculate the rise time.

        Parameters:
            time (_runtime.np.ndarray): Array of time steps.
            response (_runtime.np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Rise time.
        """
        backend = _runtime.get_backend()
        time = backend.asarray(time)
        response = backend.asarray(response)

        rise_start = 0.1 * set_point
        rise_end = 0.9 * set_point
        start_mask = response >= rise_start
        end_mask = response >= rise_end
        if not bool(_runtime._to_host_array(backend.any(start_mask))) or not bool(
            _runtime._to_host_array(backend.any(end_mask))
        ):
            return float("inf")
        start_idx = int(_runtime._to_host_array(backend.argmax(start_mask)))
        end_idx = int(_runtime._to_host_array(backend.argmax(end_mask)))
        return float(_runtime._to_host_array(time[end_idx] - time[start_idx]))

    def calculate_percent_overshoot(
        self,
        response: Union[NDArray[_runtime.np.float64], List[float]],
        set_point: float,
    ) -> float:
        """
        Calculate the percent overshoot.

        Parameters:
            response (_runtime.np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Percent overshoot.
        """
        if set_point == 0:
            return 0.0
        backend = _runtime.get_backend()
        response = backend.asarray(response)
        max_response = backend.amax(response)
        return float(
            _runtime._to_host_array(((max_response - set_point) / set_point) * 100)
        )

    def calculate_settling_time(
        self,
        time: Union[NDArray[_runtime.np.float64], List[float]],
        response: Union[NDArray[_runtime.np.float64], List[float]],
        set_point: float,
        tolerance: float = 0.02,
    ) -> float:
        """
        Calculate the settling time.

        Returns the first time at which the response enters the tolerance band
        and never leaves it again.  Returns ``float('inf')`` when the response
        never enters the band, or enters but does not remain settled through
        the end of the recorded data.

        Parameters:
            time (_runtime.np.ndarray): Array of time steps.
            response (_runtime.np.ndarray): Array of response values.
            set_point (float): Desired set point value.
            tolerance (float): Fractional tolerance band (default 0.02 = 2 %).

        Returns:
            float: Settling time, or inf if the response never settles.
        """
        backend = _runtime.get_backend()
        time = backend.asarray(time)
        response = backend.asarray(response)

        settling_threshold = abs(set_point) * tolerance
        in_band = backend.abs(response - set_point) <= settling_threshold
        if not bool(_runtime._to_host_array(backend.any(in_band))):
            return float("inf")
        in_band_host = _runtime._to_host_array(in_band)
        n = len(in_band_host)
        last_excursion = -1
        for i in range(n - 1, -1, -1):
            if not in_band_host[i]:
                last_excursion = i
                break
        if last_excursion == n - 1:
            return float("inf")  # never settles
        first_settled_idx = last_excursion + 1
        return float(_runtime._to_host_array(time[first_settled_idx]))

    def calculate_steady_state_error(
        self,
        response: Union[NDArray[_runtime.np.float64], List[float]],
        set_point: float,
    ) -> float:
        """
        Calculate the steady-state error.

        Parameters:
            response (_runtime.np.ndarray): Array of response values.
            set_point (float): Desired set point value.

        Returns:
            float: Steady-state error.
        """
        response = _runtime.get_backend().asarray(response)
        return float(_runtime._to_host_array(response[-1] - set_point))

    # ------------------------------------------------------------------------
    def ziegler_nichols_tuning(
        self,
        Ku: Union[float, NDArray[_runtime.np.float64], List[float]],
        Tu: Union[float, NDArray[_runtime.np.float64], List[float]],
        kind: str = "PID",
    ) -> Tuple[
        Union[float, NDArray[_runtime.np.float64]],
        Union[float, NDArray[_runtime.np.float64]],
        Union[float, NDArray[_runtime.np.float64]],
    ]:
        """
        Compute Ziegler-Nichols controller gains.

        Args:
            Ku: Ultimate gain as a scalar or vector.
            Tu: Ultimate period as a scalar or vector.
            kind: Controller type: ``"P"``, ``"PI"``, or ``"PID"``.

        Returns:
            Tuple of ``(Kp, Ki, Kd)`` gains.
        """
        Ku = _runtime._to_host_array(Ku).astype(float)
        kind = kind.upper()

        if kind == "P":
            Kp, Ki, Kd = 0.50 * Ku, 0.0 * Ku, 0.0 * Ku
        else:
            Tu = _runtime._to_host_array(Tu).astype(float)
            if not _runtime.np.all(_runtime.np.isfinite(Tu)) or _runtime.np.any(
                Tu <= 0
            ):
                raise ValueError(
                    f"Tu (ultimate period) must be positive and finite, got Tu={Tu!r}. "
                    "Tu == 0 typically indicates find_ultimate_gain_and_period "
                    "found no "
                    "sustained oscillation; check your gain sweep."
                )

            if kind == "PI":
                Kp, Ki, Kd = 0.45 * Ku, 1.2 * Ku / Tu, 0.0 * Ku
            elif kind == "PID":
                Kp = 0.60 * Ku
                Ki = 2.0 * Kp / Tu
                Kd = 0.125 * Kp * Tu
            else:
                raise ValueError("kind must be 'P', 'PI' or 'PID'")

        # Return scalars as plain floats so assertEqual passes exactly
        if Ku.size == 1:
            return float(Kp), float(Ki), float(Kd)
        return Kp, Ki, Kd

    # ------------------------------------------------------------------------
    def tune_controller(
        self,
        Ku: Union[float, NDArray[_runtime.np.float64], List[float]],
        Tu: Union[float, NDArray[_runtime.np.float64], List[float]],
        kind: str = "PID",
    ) -> Tuple[
        Union[float, NDArray[_runtime.np.float64]],
        Union[float, NDArray[_runtime.np.float64]],
        Union[float, NDArray[_runtime.np.float64]],
    ]:
        """
        Convenience wrapper that logs and returns NumPy arrays (length = DOF).
        """
        Kp, Ki, Kd = self.ziegler_nichols_tuning(Ku, Tu, kind)
        _runtime.logger.info(
            f"Tuned Z-N ({kind}) gains\n  Kp={Kp}\n  Ki={Ki}\n  Kd={Kd}"
        )
        return Kp, Ki, Kd

    # ------------------------------------------------------------------------
    def find_ultimate_gain_and_period(
        self,
        thetalist: Union[NDArray[_runtime.np.float64], List[float]],
        desired_joint_angles: Union[NDArray[_runtime.np.float64], List[float]],
        dt: float,
        max_steps: int = 1000,
    ) -> Tuple[float, float, List[float], List[NDArray[_runtime.np.float64]]]:
        """
        Find the ultimate gain and period using the Ziegler–Nichols method.

        This optimizer is an explicit NumPy host boundary.

        Parameters:
            thetalist: Initial joint angles (shape [6]).
            desired_joint_angles: Step target angles (shape [6]).
            dt: Simulation time step.
            max_steps: Number of integration steps to try.

        Returns:
            tuple:
              - ultimate_gain (float)
              - ultimate_period (float)
              - gain_history (list of float)
              - error_history (list of _runtime.np.ndarray)
        """
        thetalist = _runtime._to_host_array(thetalist)
        desired_joint_angles = _runtime._to_host_array(desired_joint_angles)

        with _runtime.use_backend("numpy"):
            Kp = 0.01
            increase = 1.1
            oscillation = False
            gain_history = []
            error_history = []

            while not oscillation and Kp < 1000:
                theta = thetalist.copy()
                omega = _runtime.np.zeros_like(theta)
                self._set_state("eint", _runtime.np.zeros_like(theta))
                errors = []

                for step in range(max_steps):
                    # pure-PD poke
                    tau = self.pd_control(
                        desired_joint_angles,
                        _runtime.np.zeros_like(theta),
                        theta,
                        omega,
                        Kp,
                        0.0,
                    )
                    # alpha = M⁻¹ (tau – C – G)
                    M = self.dynamics.mass_matrix(theta)
                    C = self.dynamics.velocity_quadratic_forces(theta, omega)
                    Gf = self.dynamics.gravity_forces(
                        theta, _runtime.np.array([0, 0, -9.81])
                    )
                    alpha = _runtime.np.linalg.solve(M, tau - C - Gf)

                    omega += alpha * dt
                    theta += omega * dt

                    err = _runtime.np.linalg.norm(theta - desired_joint_angles)
                    errors.append(err)
                    # blow-up guard
                    if step > 10 and err > 1e10:
                        break

                gain_history.append(Kp)
                error_history.append(_runtime.np.array(errors))

                # look for the first upward slope after initial increase
                if len(errors) >= 2 and errors[-2] < errors[-1] < errors[-2] * 1.2:
                    oscillation = True
                else:
                    Kp *= increase

            ultimate_gain = float(Kp)
            ultimate_period = (max_steps * dt) / max(
                1,
                _runtime.np.count_nonzero(
                    _runtime.np.diff(_runtime.np.sign(error_history[-1]))
                )
                // 2,
            )

            return ultimate_gain, ultimate_period, gain_history, error_history
