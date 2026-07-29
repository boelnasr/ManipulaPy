#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Singularity Analysis Module - ManipulaPy

This module provides comprehensive singularity analysis capabilities for robotic
manipulators including singularity detection, manipulability ellipsoid visualization,
workspace analysis, and condition number computations with optional CUDA acceleration.

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

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.spatial import ConvexHull

from ..backend import get_backend


class Singularity:
    """Singularity and manipulability analysis for a serial manipulator."""

    def __init__(self, serial_manipulator: Any) -> None:
        """
        Initialize the Singularity class with a SerialManipulator object.

        Parameters:
            serial_manipulator (SerialManipulator): An instance of SerialManipulator.
        """
        self.serial_manipulator = serial_manipulator

    def singularity_analysis(
        self, thetalist: Union[NDArray[np.float64], List[float]]
    ) -> bool:
        """Detect singularity using smallest singular value (works for any Jacobian shape).

        Computes the space-frame Jacobian at the given configuration and flags a
        singularity when its smallest singular value falls below ``1e-4``.

        Args:
            thetalist: Joint angles in radians, length-N array or list, where N
                is the number of joints.

        Returns:
            bool: True if the configuration is (near-)singular, otherwise False.
        """
        backend = get_backend()
        J = backend.asarray(self.serial_manipulator.jacobian(thetalist, frame="space"))
        singular_values = backend.svdvals(J)
        # Smallest singular value (SVD returns them in descending order). The
        # threshold comparison is the public bool contract, evaluated on host.
        smallest = backend.to_numpy(singular_values[-1])
        return bool(smallest < 1e-4)

    @staticmethod
    def _ellipsoid_axes(backend: Any, jacobian_part: Any) -> Tuple[Any, Any]:
        """Return the ellipsoid axes and guarded radii for a Jacobian block.

        Dispatches the SVD and radii math through the caller's backend (passed
        in so one backend instance serves the whole ellipsoid computation). The
        singular-value spectrum is right-padded with zeros to the axis count so
        ``radii`` scales every axis of ``U``, and the ``maximum(S, 1e-10)`` floor
        preserves the existing divide-by-zero guard around degenerate axes.

        Args:
            backend: The active ArrayBackend captured by the caller.
            jacobian_part: Linear or angular velocity block of the Jacobian.

        Returns:
            Tuple ``(U, radii)`` in the backend's array type: ``U`` are the
            ellipsoid axes and ``radii`` the reciprocal-root singular values.
        """
        U, S, _ = backend.svd(jacobian_part, full_matrices=True)
        pad = U.shape[1] - S.shape[0]
        if pad > 0:
            S = backend.concatenate([S, backend.zeros(pad, dtype=S.dtype)])
        radii = 1.0 / backend.sqrt(backend.maximum(S, 1e-10))
        return U, radii

    def manipulability_ellipsoid(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        ax: Optional[Any] = None,
    ) -> None:
        """
        Plot the manipulability ellipsoid for a given set of joint angles.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Matplotlib 3D axis to plot on. Defaults to None.
        """
        backend = get_backend()
        J = backend.asarray(self.serial_manipulator.jacobian(thetalist, frame="space"))
        J_v = J[:3, :]  # Linear velocity part of the Jacobian
        J_w = J[3:, :]  # Angular velocity part of the Jacobian

        # Ellipsoid axis math (SVD + guarded radii) is dispatched through the
        # active backend; converted to host arrays for the matplotlib plotting.
        U_v, radii_v = self._ellipsoid_axes(backend, J_v)
        U_w, radii_w = self._ellipsoid_axes(backend, J_w)
        U_v, radii_v = backend.to_numpy(U_v), backend.to_numpy(radii_v)
        U_w, radii_w = backend.to_numpy(U_w), backend.to_numpy(radii_w)

        # Generate points on a unit sphere
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        points = np.array([x.flatten(), y.flatten(), z.flatten()])

        # Transform points to ellipsoids
        ellipsoid_points_v = np.dot(U_v, np.diag(radii_v)).dot(points)
        ellipsoid_points_w = np.dot(U_w, np.diag(radii_w)).dot(points)

        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
        else:
            ax1 = ax2 = ax

        # Plot the linear velocity ellipsoid
        ax1.plot_surface(
            ellipsoid_points_v[0].reshape(x.shape),
            ellipsoid_points_v[1].reshape(y.shape),
            ellipsoid_points_v[2].reshape(z.shape),
            color="b",
            alpha=0.5,
        )
        ax1.set_title("Linear Velocity Ellipsoid")

        # Plot the angular velocity ellipsoid
        ax2.plot_surface(
            ellipsoid_points_w[0].reshape(x.shape),
            ellipsoid_points_w[1].reshape(y.shape),
            ellipsoid_points_w[2].reshape(z.shape),
            color="r",
            alpha=0.5,
        )
        ax2.set_title("Angular Velocity Ellipsoid")

        if ax is None:
            plt.show()

    def plot_workspace_monte_carlo(
        self, joint_limits: List[Tuple[float, float]], num_samples: int = 10000
    ) -> None:
        """
        Estimate the robot workspace using Monte Carlo sampling.

        Parameters:
            joint_limits (list): A list of tuples representing the joint limits.
            num_samples (int, optional): Number of samples for Monte Carlo simulation. Defaults to 10000.
        """
        # Initialize device arrays
        joint_samples = cuda.device_array(
            (num_samples, len(joint_limits)), dtype=np.float32
        )

        # Define the CUDA kernel for generating joint angles
        @cuda.jit
        def generate_joint_samples(rng_states, joint_limits, joint_samples) -> None:
            """CUDA kernel filling joint_samples with uniform random angles.

            Each thread maps to one sample row and draws a uniform random angle
            per joint within its lower/upper limit.

            Args:
                rng_states: xoroshiro128+ RNG state array (one state per thread),
                    as produced by ``create_xoroshiro128p_states``.
                joint_limits: (num_joints, 2) device array of (low, high) limits
                    per joint, in radians.
                joint_samples: (num_samples, num_joints) device array; in-place
                    OUTPUT buffer written with the sampled joint angles in radians.
            """
            pos = cuda.grid(1)
            if pos < joint_samples.shape[0]:
                for i in range(joint_samples.shape[1]):
                    low, high = joint_limits[i]
                    joint_samples[pos, i] = (
                        xoroshiro128p_uniform_float32(rng_states, pos) * (high - low)
                        + low
                    )

        # Setup random states
        rng_states = create_xoroshiro128p_states(num_samples, seed=1234)
        device_joint_limits = cuda.to_device(np.array(joint_limits, dtype=np.float32))

        # Launch kernel
        threadsperblock = 256
        blockspergrid = (num_samples + threadsperblock - 1) // threadsperblock
        generate_joint_samples[blockspergrid, threadsperblock](
            rng_states, device_joint_limits, joint_samples
        )

        # Copy joint samples to host and calculate workspace points
        host_joint_samples = joint_samples.copy_to_host()
        workspace_points = np.array(
            [
                self.serial_manipulator.forward_kinematics(thetas)[:3, 3]
                for thetas in host_joint_samples
            ]
        )

        # Compute convex hull
        hull = ConvexHull(workspace_points)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            workspace_points[:, 0],
            workspace_points[:, 1],
            workspace_points[:, 2],
            triangles=hull.simplices,
            cmap="viridis",
            edgecolor="none",
            alpha=0.5,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Workspace (Smooth Convex Hull)")
        plt.show()

    def condition_number(
        self, thetalist: Union[NDArray[np.float64], List[float]]
    ) -> float:
        """
        Calculate the condition number of the Jacobian for a given set of joint angles.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.

        Returns:
            float: The condition number of the Jacobian matrix.
        """
        backend = get_backend()
        J = backend.asarray(self.serial_manipulator.jacobian(thetalist, frame="space"))
        # np.linalg.cond parity requires the values-only SVD (its full-USV
        # LAPACK path can spin forever on non-finite input) and an errstate
        # guard: cond suppresses the 0/0 of a fully rank-deficient matrix even
        # when the caller escalates floating-point errors to exceptions.
        with np.errstate(all="ignore"):
            singular_values = backend.svdvals(J)
            ratio = backend.amax(singular_values) / backend.amin(singular_values)
        # Host public boundary: NaN ratio (0/0, or an inf-input spectrum) is
        # reported as an infinite condition number, preserving the input dtype
        # exactly as np.linalg.cond does.
        # Only a caller who passed a backend-native ``thetalist`` can be
        # differentiating through this metric, and only then must the result
        # stay native: reading a traced value onto the host raises under JAX and
        # silently detaches the graph under Torch. A host seed keeps the host
        # return on every backend, so the documented float contract (and the
        # np.linalg.cond parity above) is unchanged for existing callers.
        # Gating on ``is_concrete`` alone would be wrong here -- it is a
        # property of the backend, not of this value, and would hand a Tensor
        # back to every Torch caller passing plain NumPy.
        if backend.is_concrete or not backend.is_backend_array(thetalist):
            value = backend.to_numpy(ratio)[()]
            if np.isnan(value):
                value = type(value)(np.inf)
            return value
        # ``ratio != ratio`` is the NaN test because the protocol exposes
        # ``isfinite``, which cannot separate NaN from the infinity replacing it.
        return backend.where(ratio != ratio, backend.asarray(float("inf")), ratio)

    def near_singularity_detection(
        self,
        thetalist: Union[NDArray[np.float64], List[float]],
        threshold: float = 1e-2,
    ) -> bool:
        """
        Detect if the manipulator is near a singularity by comparing the condition number with a threshold.

        Parameters:
            thetalist (numpy.ndarray): Array of joint angles in radians.
            threshold (float, optional): Threshold value for the condition number. Defaults to 1e-2.

        Returns:
            bool: True if the manipulator is near a singularity, False otherwise.
        """
        cond_number = self.condition_number(thetalist)
        return cond_number > threshold
