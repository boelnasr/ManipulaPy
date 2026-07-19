#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-side collision avoidance and planning mixin - ManipulaPy"""

from . import _kernels as _runtime
from ._kernels import List, np


class _CollisionMixin:
    def _apply_collision_avoidance_gpu(
        self, traj_pos: np.ndarray, thetaend: np.ndarray
    ) -> np.ndarray:
        """Apply potential-field collision avoidance on the CUDA-enabled path.

        Collision avoidance is a joint-space computation and has no GPU kernel
        (the GPU potential field is 3-D Cartesian, not joint-space), so this
        delegates to the validated joint-space CPU routine. Kept as a distinct
        dispatch target so the GPU/CPU planning code paths stay symmetric.

        Args:
            traj_pos: (N, num_joints) ndarray of joint positions (rad);
                modified in place as colliding points are nudged out.
            thetaend: (num_joints,) ndarray of goal joint angles (rad), used as
                the attractive target.

        Returns:
            np.ndarray: The (possibly adjusted) ``(N, num_joints)`` trajectory.
        """
        # Collision avoidance operates in JOINT space: ``traj_pos`` holds joint
        # configurations and each is nudged by ``step -= gradient``, so the
        # potential-field gradient must be joint-space too (see PotentialField and
        # _apply_collision_avoidance_cpu). The GPU ``optimized_potential_field``
        # kernel is a 3-D *Cartesian* field — feeding it an (N, num_joints) array
        # and an (num_joints,) goal is a dimension mismatch that yields a wrong
        # gradient and a 6-vs-3 broadcast error on update. There is no joint-space
        # GPU potential-field kernel, so we run the validated joint-space routine
        # rather than silently erroring per iteration and falling back anyway.
        return self._apply_collision_avoidance_cpu(traj_pos, thetaend)

    def _apply_collision_avoidance_cpu(
        self, traj_pos: np.ndarray, thetaend: np.ndarray
    ) -> np.ndarray:
        """Apply CPU-based potential field collision avoidance.

        For each colliding trajectory point, iteratively descends the
        potential-field gradient toward ``thetaend`` (up to 100 iterations)
        until the configuration is collision-free.

        Args:
            traj_pos: (N, num_joints) ndarray of joint positions (rad);
                modified in place as colliding points are nudged out.
            thetaend: (num_joints,) ndarray of goal joint angles (rad), used as
                the attractive target.

        Returns:
            np.ndarray: The (possibly adjusted) ``(N, num_joints)`` trajectory.
        """
        backend = _runtime.get_backend()
        if len(traj_pos) == 0:
            # Base iterated an empty trajectory and returned it unchanged;
            # ``stack([])`` would raise, so short-circuit the degenerate case.
            return traj_pos

        # The collision checker and potential field live in the host NumPy
        # ``potential_field`` module, so cross the boundary explicitly: keep the
        # goal on the host and integrate each row's gradient nudge entirely on
        # the host (no backend-native / NumPy mixed arithmetic), then re-enter
        # the backend. Rows are stacked (no in-place writes).
        q_goal = backend.to_numpy(thetaend)
        obstacles = []  # Define obstacles here as needed

        adjusted_rows = []
        for step in traj_pos:
            step_host = backend.to_numpy(step)
            if self.collision_checker.check_collision(step_host):
                for _ in range(100):  # Max iterations to adjust trajectory
                    gradient = self.potential_field.compute_gradient(
                        step_host, q_goal, obstacles
                    )
                    # Adjust step size as needed (host arithmetic, row dtype)
                    step_host = _runtime.np.asarray(
                        step_host - 0.01 * gradient, dtype=step_host.dtype
                    )
                    if not self.collision_checker.check_collision(step_host):
                        break
            adjusted_rows.append(backend.asarray(step_host))

        return backend.stack(adjusted_rows)

    def plan_trajectory(
        self, start_position, target_position, obstacle_points
    ) -> List[List[float]]:
        """
        Enhanced trajectory planning with collision avoidance.

        Args:
            start_position (list): Initial joint configuration.
            target_position (list): Desired joint configuration.
            obstacle_points (list): List of obstacle points in the environment.

        Returns:
            list: Joint trajectory as a list of joint configurations.
        """
        _runtime.logger.info(
            "Planning trajectory from %d to %d DOF",
            len(start_position),
            len(target_position),
        )

        # Enhanced trajectory planning with multiple waypoints
        # This is a simple interpolation - can be extended with RRT*, PRM, etc.
        num_waypoints = 5
        joint_trajectory = []

        backend = _runtime.get_backend()
        start_pos = backend.asarray(start_position)
        target_pos = backend.asarray(target_position)
        # The potential field and collision checker are host NumPy boundaries.
        # Keep host copies of the goal and obstacles so every field/checker call
        # and the gradient nudge stay on the host (no backend-native / NumPy
        # mixed arithmetic), then re-enter the backend with the result.
        target_host = backend.to_numpy(target_pos)

        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint = (1 - alpha) * start_pos + alpha * target_pos

            # Simple collision avoidance - move away from obstacles.
            if obstacle_points and self.potential_field:
                waypoint_host = backend.to_numpy(waypoint)
                obstacles_host = [backend.to_numpy(o) for o in obstacle_points]
                for _ in range(10):  # Max adjustment iterations
                    gradient = self.potential_field.compute_gradient(
                        waypoint_host, target_host, obstacles_host
                    )
                    # Host arithmetic at the waypoint dtype (no in-place writes)
                    waypoint_host = _runtime.np.asarray(
                        waypoint_host - 0.01 * gradient, dtype=waypoint_host.dtype
                    )

                    # Check if waypoint is collision-free
                    if self.collision_checker:
                        if not self.collision_checker.check_collision(waypoint_host):
                            break
                waypoint = backend.asarray(waypoint_host)

            joint_trajectory.append(backend.to_numpy(waypoint).tolist())

        _runtime.logger.info(
            f"Planned trajectory with {len(joint_trajectory)} waypoints"
        )
        return joint_trajectory
