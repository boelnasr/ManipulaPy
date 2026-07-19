#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverse and forward manipulator dynamics."""

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from . import manipulator_dynamics as _runtime


class _InverseForwardDynamicsConcern:
    """Inverse/forward methods installed on the public dynamics class."""

    def inverse_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        ddthetalist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        Compute joint torques for the requested motion and end-effector wrench.

        Args:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            ddthetalist: Joint accelerations.
            g: Gravity vector.
            Ftip: End-effector wrench.

        Returns:
            Required joint torques.
        """
        backend = _runtime.get_backend()
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        taulist = (
            backend.matmul(M, backend.asarray(ddthetalist))
            + c
            + g_forces
            + backend.matmul(J_transpose, backend.asarray(Ftip))
        )
        return taulist

    def forward_dynamics(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        dthetalist: Union[NDArray[np.float64], List[float]],
        taulist: Union[NDArray[np.float64], List[float]],
        g: Union[NDArray[np.float64], List[float]],
        Ftip: Union[NDArray[np.float64], List[float]],
    ) -> NDArray[np.float64]:
        """
        Solve joint accelerations from applied torques and external loads.

        Args:
            thetalist: Joint angles.
            dthetalist: Joint velocities.
            taulist: Applied joint torques.
            g: Gravity vector.
            Ftip: End-effector wrench.

        Returns:
            Joint accelerations.
        """
        backend = _runtime.get_backend()
        M = self.mass_matrix(thetalist)
        c = self.velocity_quadratic_forces(thetalist, dthetalist)
        g_forces = self.gravity_forces(thetalist, g)
        J_transpose = self.jacobian(thetalist).T
        rhs = (
            backend.asarray(taulist)
            - c
            - g_forces
            - backend.matmul(J_transpose, backend.asarray(Ftip))
        )
        ddthetalist = backend.solve(M, rhs)
        return ddthetalist
