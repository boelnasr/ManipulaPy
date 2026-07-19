#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trajectory plotting mixin - ManipulaPy"""

from . import _kernels as _runtime


class _PlottingMixin:
    @staticmethod
    def plot_trajectory(
        trajectory_data,
        Tf,
        title="Joint Trajectory",
        labels=None,
        performance_stats=None,
    ) -> None:
        """Plot joint position, velocity and acceleration trajectories.

        Draws a 3-by-num_joints grid of subplots (position, velocity,
        acceleration per joint) over a time axis spanning ``[0, Tf]`` and
        appends GPU speedup / kernel info to the title when
        ``performance_stats`` is provided. Displays the figure via
        ``plt.show()``.

        Args:
            trajectory_data (dict): Trajectory dict with keys ``"positions"``,
                ``"velocities"`` and ``"accelerations"``, each an
                ``(num_steps, num_joints)`` array.
            Tf (float): Total trajectory duration, seconds, defining the time
                axis.
            title (str): Base figure title. Defaults to ``"Joint Trajectory"``.
            labels (list, optional): Per-joint labels; used only when its length
                equals ``num_joints``, otherwise generic ``"Joint i"`` labels
                are generated.
            performance_stats (dict, optional): Optional stats with keys such as
                ``"speedup_achieved"`` and ``"best_kernel_used"`` used to
                annotate the title.
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        num_joints = positions.shape[1]
        time_steps = _runtime.np.linspace(0, Tf, num_steps)

        fig, axs = _runtime.plt.subplots(3, num_joints, figsize=(15, 10), sharex="col")

        # Add performance info to title
        if performance_stats:
            speedup = performance_stats.get("speedup_achieved", 0)
            kernel = performance_stats.get("best_kernel_used", "unknown")
            if speedup > 1:
                title += f" (GPU: {speedup:.1f}x speedup, {kernel} kernel)"
            else:
                title += " (CPU execution)"

        fig.suptitle(title)

        for i in range(num_joints):
            if labels and len(labels) == num_joints:
                label = labels[i]
            else:
                label = f"Joint {i+1}"

            axs[0, i].plot(time_steps, positions[:, i], label=f"{label} Position")
            axs[0, i].set_ylabel("Position")
            axs[0, i].legend()

            axs[1, i].plot(time_steps, velocities[:, i], label=f"{label} Velocity")
            axs[1, i].set_ylabel("Velocity")
            axs[1, i].legend()

            axs[2, i].plot(
                time_steps, accelerations[:, i], label=f"{label} Acceleration"
            )
            axs[2, i].set_ylabel("Acceleration")
            axs[2, i].legend()

        for ax in axs[-1]:
            ax.set_xlabel("Time (s)")

        _runtime.plt.tight_layout()
        _runtime.plt.show()

    def plot_tcp_trajectory(self, trajectory, dt) -> None:
        """
        Enhanced TCP trajectory plotting with performance monitoring.

        Args:
            trajectory (list): Joint angle configurations representing the
                trajectory.
            dt (float): The time step between consecutive points in the trajectory.

        Returns:
            None
        """
        start_time = _runtime.time.time()

        tcp_trajectory = [
            self.serial_manipulator.forward_kinematics(joint_angles)
            for joint_angles in trajectory
        ]
        tcp_positions = [pose[:3, 3] for pose in tcp_trajectory]

        velocity, acceleration, jerk = self.calculate_derivatives(tcp_positions, dt)
        time_array = _runtime.np.arange(0, len(tcp_positions) * dt, dt)

        elapsed = _runtime.time.time() - start_time

        _runtime.plt.figure(figsize=(12, 8))
        title = f"TCP Trajectory (FK computed in {elapsed:.3f}s)"
        _runtime.plt.suptitle(title)

        for i, label in enumerate(["X", "Y", "Z"]):
            _runtime.plt.subplot(4, 1, 1)
            _runtime.plt.plot(
                time_array,
                _runtime.np.array(tcp_positions)[:, i],
                label=f"TCP {label} Position",
            )
            _runtime.plt.ylabel("Position")
            _runtime.plt.legend()

            _runtime.plt.subplot(4, 1, 2)
            _runtime.plt.plot(
                time_array[:-1], velocity[:, i], label=f"TCP {label} Velocity"
            )
            _runtime.plt.ylabel("Velocity")
            _runtime.plt.legend()

            _runtime.plt.subplot(4, 1, 3)
            _runtime.plt.plot(
                time_array[:-2], acceleration[:, i], label=f"TCP {label} Acceleration"
            )
            _runtime.plt.ylabel("Acceleration")
            _runtime.plt.legend()

            _runtime.plt.subplot(4, 1, 4)
            _runtime.plt.plot(time_array[:-3], jerk[:, i], label=f"TCP {label} Jerk")
            _runtime.plt.xlabel("Time")
            _runtime.plt.ylabel("Jerk")
            _runtime.plt.legend()

        _runtime.plt.tight_layout()
        _runtime.plt.show()

    def plot_cartesian_trajectory(
        self, trajectory_data, Tf, title="Cartesian Trajectory", performance_stats=None
    ) -> None:
        """
        Enhanced Cartesian trajectory plotting with performance information.

        Args:
            trajectory_data (dict): A dictionary containing trajectory data.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot.
            performance_stats (dict, optional): Performance statistics to display.

        Returns:
            None
        """
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        accelerations = trajectory_data["accelerations"]

        num_steps = positions.shape[0]
        time_steps = _runtime.np.linspace(0, Tf, num_steps)

        # Add performance info to title
        if performance_stats:
            speedup = performance_stats.get("speedup_achieved", 0)
            if speedup > 1:
                title += f" (GPU: {speedup:.1f}x speedup)"
            else:
                title += " (CPU execution)"

        fig, axs = _runtime.plt.subplots(3, 1, figsize=(10, 15), sharex="col")
        fig.suptitle(title)

        axs[0].plot(time_steps, positions[:, 0], label="X Position")
        axs[0].plot(time_steps, positions[:, 1], label="Y Position")
        axs[0].plot(time_steps, positions[:, 2], label="Z Position")
        axs[0].set_ylabel("Position")
        axs[0].legend()

        axs[1].plot(time_steps, velocities[:, 0], label="X Velocity")
        axs[1].plot(time_steps, velocities[:, 1], label="Y Velocity")
        axs[1].plot(time_steps, velocities[:, 2], label="Z Velocity")
        axs[1].set_ylabel("Velocity")
        axs[1].legend()

        axs[2].plot(time_steps, accelerations[:, 0], label="X Acceleration")
        axs[2].plot(time_steps, accelerations[:, 1], label="Y Acceleration")
        axs[2].plot(time_steps, accelerations[:, 2], label="Z Acceleration")
        axs[2].set_ylabel("Acceleration")
        axs[2].legend()

        axs[2].set_xlabel("Time (s)")

        _runtime.plt.tight_layout()
        _runtime.plt.show()

    def plot_ee_trajectory(
        self, trajectory_data, Tf, title="End-Effector Trajectory"
    ) -> None:
        """
        Enhanced end-effector trajectory plotting.

        Args:
            trajectory_data (dict): A dictionary containing trajectory data.
            Tf (float): The final time of the trajectory.
            title (str, optional): The title of the plot.

        Returns:
            None
        """
        positions = trajectory_data["positions"]
        num_steps = positions.shape[0]
        time_steps = _runtime.np.linspace(0, Tf, num_steps)

        if "orientations" in trajectory_data:
            orientations = trajectory_data["orientations"]
        else:
            # Compute orientations using forward kinematics
            start_time = _runtime.time.time()
            orientations = _runtime.np.array(
                [
                    self.serial_manipulator.forward_kinematics(pos)[:3, :3]
                    for pos in positions
                ]
            )
            elapsed = _runtime.time.time() - start_time
            title += f" (FK for orientations: {elapsed:.3f}s)"

        fig = _runtime.plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.suptitle(title)

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label="EE Position",
            color="b",
            linewidth=2,
        )

        # Draw orientation frames at selected points
        frame_step = max(1, num_steps // 20)
        for i in range(0, num_steps, frame_step):
            R = orientations[i]
            pos = positions[i]
            scale = 0.01

            # X-axis (red)
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                R[0, 0],
                R[1, 0],
                R[2, 0],
                length=scale,
                color="r",
                alpha=0.8,
            )
            # Y-axis (green)
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                R[0, 1],
                R[1, 1],
                R[2, 1],
                length=scale,
                color="g",
                alpha=0.8,
            )
            # Z-axis (blue)
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                R[0, 2],
                R[1, 2],
                R[2, 2],
                length=scale,
                color="b",
                alpha=0.8,
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.legend()
        _runtime.plt.show()
