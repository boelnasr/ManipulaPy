#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: F401, F841, SIM105, UP006, UP035 - preserve the historical API.
"""
Simulation Module - ManipulaPy

This module provides PyBullet-based simulation capabilities for robotic manipulators
including real-time visualization, physics simulation, and interactive control.

UPDATED VERSION with VISIBLE TRAJECTORY SPLINE:
- Replaced addUserDebugLine() with real capsule geometry
- Trajectory splines now appear in getCameraImage() screenshots
- Added proper cleanup for trajectory visualization

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


import logging
import os
import sys as _sys
import time
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ManipulaPy.control import ManipulatorController
from ManipulaPy.path_planning import TrajectoryPlanning as tp

from . import _runtime
from .controllers import _ControlConcern
from .rendering import _RenderingConcern

p = _runtime.p
pybullet_data = _runtime.pybullet_data
_PYBULLET_AVAILABLE = _runtime._PYBULLET_AVAILABLE
_check_pybullet_available = _runtime._check_pybullet_available
get_backend = _runtime.get_backend


class Simulation:
    """PyBullet-backed simulation and visualization helper for manipulators."""

    def __init__(
        self,
        urdf_file_path: str,
        joint_limits: Sequence[Tuple[float, float]],
        torque_limits: Optional[Sequence[Any]] = None,
        time_step: float = 0.01,
        real_time_factor: float = 1.0,
        physics_client: Optional[int] = None,
        enable_self_collision: bool = False,
        disable_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        """
        Initialize the simulation and set up the PyBullet world.

        Args:
            urdf_file_path: Path to the robot's URDF file.
            joint_limits: Per-joint (lower, upper) position limits.
            torque_limits: Optional per-joint torque limits.
            time_step: Simulation time step in seconds.
            real_time_factor: Real-time playback factor.
            physics_client: Existing PyBullet client id to reuse, if any.
            enable_self_collision: Whether to enable self-collision checking.
            disable_pairs: Link name pairs to exclude from self-collision.
        """
        if not _PYBULLET_AVAILABLE:
            raise ImportError(
                "Simulation requires pybullet. Install with: "
                "pip install 'ManipulaPy[simulation]'"
            )
        self.urdf_file_path = urdf_file_path
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.real_time_factor = real_time_factor
        self.enable_self_collision = enable_self_collision
        self._disable_pairs = disable_pairs or []
        self.logger = self.setup_logger()
        self.physics_client = physics_client
        self.joint_params = []
        self.reset_button = None
        self.home_position = None

        # NEW: Track trajectory visualization bodies for cleanup
        self.trajectory_body_ids = []

        self.setup_simulation()

    def setup_logger(self) -> logging.Logger:
        """
        Sets up the logger for the simulation.
        """
        logger = logging.getLogger("SimulationLogger")
        logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(ch)
            # Own our output; don't also bubble to the root handler (double logging)
            logger.propagate = False
        return logger

    def _connect_client(self) -> int:
        """Connect to PyBullet, preferring GUI but falling back to DIRECT.

        Honors ``MANIPULAPY_PYBULLET_CONNECT`` (``GUI`` or ``DIRECT``) so headless
        environments (CI, servers) can force a windowless client. Otherwise it
        tries GUI and transparently falls back to DIRECT when no display is
        available, instead of leaving an invalid client that later surfaces as
        out-of-range joint-index errors.

        Returns:
            int: The connected PyBullet physics client id.
        """
        mode = os.getenv("MANIPULAPY_PYBULLET_CONNECT", "").strip().upper()
        if mode == "DIRECT":
            return p.connect(p.DIRECT)
        if mode == "GUI":
            return p.connect(p.GUI)
        try:
            client = p.connect(p.GUI)
            if client < 0:
                raise RuntimeError("GUI connection returned an invalid client id")
            return client
        except Exception:
            self.logger.warning(
                "PyBullet GUI unavailable; falling back to DIRECT (headless) mode."
            )
            return p.connect(p.DIRECT)

    def connect_simulation(self) -> None:
        """
        Connects to the PyBullet simulation.
        """
        _check_pybullet_available()
        self.logger.info("Connecting to PyBullet simulation...")
        if self.physics_client is None:
            self.physics_client = self._connect_client()
        p.resetSimulation()  # Clear the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def disconnect_simulation(self) -> None:
        """
        Disconnects from the PyBullet simulation.
        """
        _check_pybullet_available()
        self.logger.info("Disconnecting from PyBullet simulation...")
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            self.logger.info("Disconnected successfully.")

    def setup_simulation(self) -> None:
        """
        Sets up the simulation environment.
        """
        _check_pybullet_available()
        if self.physics_client is None:
            self.physics_client = self._connect_client()
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load the robot, enabling self-collision detection if requested
        load_flags = p.URDF_USE_SELF_COLLISION if self.enable_self_collision else 0
        self.robot_id = p.loadURDF(
            self.urdf_file_path, useFixedBase=True, flags=load_flags
        )

        # Identify non-fixed joints
        self.non_fixed_joints = [
            i
            for i in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]
        self.home_position = np.zeros(len(self.non_fixed_joints))

        # Apply per-pair collision filter exclusions when self-collision is on
        if self.enable_self_collision and self._disable_pairs:
            # Build link-name -> pybullet link index map; joint info[12] is child link name
            link_name_to_idx = {
                p.getJointInfo(self.robot_id, i)[12].decode(): i
                for i in range(p.getNumJoints(self.robot_id))
            }
            for name_a, name_b in self._disable_pairs:
                idx_a = link_name_to_idx.get(name_a)
                idx_b = link_name_to_idx.get(name_b)
                if idx_a is not None and idx_b is not None:
                    p.setCollisionFilterPair(
                        self.robot_id, self.robot_id, idx_a, idx_b, 0
                    )
                else:
                    self.logger.warning(
                        "disable_pairs: could not resolve link names %r / %r to indices",
                        name_a,
                        name_b,
                    )

    def initialize_robot(self) -> None:
        """
        Initializes the robot using the URDF processor.
        """
        _check_pybullet_available()
        # Only skip URDF processing if self.robot is already set.
        if hasattr(self, "robot") and self.robot is not None:
            self.logger.warning("Robot already initialized. Skipping URDF processing.")
        else:
            # Even if self.robot_id is already set from setup_simulation(),
            # we need to process the URDF to set self.robot and self.dynamics.
            if not (hasattr(self, "robot_id") and self.robot_id is not None):
                load_flags = (
                    p.URDF_USE_SELF_COLLISION if self.enable_self_collision else 0
                )
                self.robot_id = p.loadURDF(
                    self.urdf_file_path,
                    [0, 0, 0.1],
                    useFixedBase=True,
                    flags=load_flags,
                )
            # Process the URDF to generate the robot model and dynamics.
            from ManipulaPy.urdf_processor import URDFToSerialManipulator

            urdf_processor = URDFToSerialManipulator(self.urdf_file_path)
            self.robot = urdf_processor.serial_manipulator
            self.dynamics = urdf_processor.dynamics
            # Identify non-fixed joints
            self.non_fixed_joints = [
                i
                for i in range(p.getNumJoints(self.robot_id))
                if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
            ]
            self.home_position = np.zeros(len(self.non_fixed_joints))

    set_robot_models = _ControlConcern.__dict__["set_robot_models"]
    initialize_planner_and_controller = _ControlConcern.__dict__[
        "initialize_planner_and_controller"
    ]
    add_joint_parameters = _ControlConcern.__dict__["add_joint_parameters"]
    add_reset_button = _ControlConcern.__dict__["add_reset_button"]
    set_joint_positions = _ControlConcern.__dict__["set_joint_positions"]
    get_joint_positions = _ControlConcern.__dict__["get_joint_positions"]

    _capsule_line = _RenderingConcern.__dict__["_capsule_line"]
    plot_trajectory = _RenderingConcern.__dict__["plot_trajectory"]
    _add_trajectory_markers = _RenderingConcern.__dict__["_add_trajectory_markers"]
    clear_trajectory_visualization = _RenderingConcern.__dict__[
        "clear_trajectory_visualization"
    ]

    def run_trajectory(
        self, joint_trajectory: Sequence[Sequence[float]]
    ) -> Tuple[float, float, float]:
        """
        Runs a joint trajectory in the simulation.

        Iterates over each waypoint, commands position control, steps the
        simulation, records the end-effector position, and finally renders the
        traced end-effector path in the scene.

        Args:
            joint_trajectory: Sequence of joint-angle configurations (one per
                simulation step), each a sequence of joint angles in radians.

        Returns:
            tuple[float, float, float]: The (x, y, z) world position of the
            end-effector at the final waypoint.
        """
        _check_pybullet_available()
        self.logger.info("Running trajectory...")
        ee_positions = []

        for joint_positions in joint_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Trajectory completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def run_controller(
        self, desired_positions: Sequence[Sequence[float]]
    ) -> Tuple[float, float, float]:
        """
        Drive the robot through ``desired_positions`` in open-loop position
        control, one configuration per simulation step.

        For real closed-loop torque control, drive PyBullet's
        ``p.TORQUE_CONTROL`` mode directly in your own loop. The previous
        signature accepted a controller object plus PID gains; those were
        removed in v1.3.2 because the loop body never produced honest
        closed-loop behavior. See CHANGELOG.

        Args:
            desired_positions: Waypoints to track, shape ``(N, DOF)`` where DOF
                must equal the number of non-fixed joints; joint angles in
                radians.

        Returns:
            tuple[float, float, float]: The (x, y, z) world position of the
            end-effector at the final waypoint.

        Raises:
            ValueError: If ``desired_positions`` is empty, is not 2-D, or its
                joint count does not match the number of non-fixed joints.
        """
        _check_pybullet_available()
        self.logger.info("Running controller...")
        ee_positions = []

        # PyBullet boundary: bring a possibly device-side trajectory to the
        # host through the active backend before feeding the native sim loop.
        # A backend-native array is converted whole (iterating it first would
        # yield device-side rows that cannot cross to the host); other
        # iterables (e.g. generator waypoints) are materialized directly.
        backend = get_backend()
        host = (
            backend.to_numpy(desired_positions)
            if backend.is_backend_array(desired_positions)
            else list(desired_positions)
        )
        positions_arr = np.asarray(host, dtype=float)
        if positions_arr.size == 0:
            raise ValueError("desired_positions is empty; nothing to track")
        if positions_arr.ndim != 2:
            raise ValueError(
                "desired_positions must have shape (N waypoints x DOF); "
                f"actual shape is {positions_arr.shape}"
            )
        expected_dof = len(self.non_fixed_joints)
        actual_dof = positions_arr.shape[1]
        if actual_dof != expected_dof:
            raise ValueError(
                "desired_positions joint count mismatch: "
                f"expected {expected_dof}, got {actual_dof}"
            )

        for pos in positions_arr:
            # Open-loop position tracking. Closed-loop torque control via
            # this method was always broken (treated torque as position delta).
            # For real closed-loop control, use p.TORQUE_CONTROL mode directly
            # in your own loop. See v1.3.2 CHANGELOG.
            self.set_joint_positions(pos)
            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Controller run completed.")
        return ee_positions[-1]  # Return the last end-effector position

    get_joint_parameters = _ControlConcern.__dict__["get_joint_parameters"]

    def simulate_robot_motion(
        self, desired_angles_trajectory: Sequence[Sequence[float]]
    ) -> Tuple[float, float, float]:
        """
        Simulates the robot's motion using a given trajectory of desired joint angles.

        Commands each configuration via position control, steps the simulation,
        collects the end-effector positions, and plots the traced path.

        Args:
            desired_angles_trajectory: Sequence of desired joint-angle
                configurations (one per simulation step), each a sequence of
                joint angles in radians.

        Returns:
            tuple[float, float, float]: The (x, y, z) world position of the
            end-effector at the final configuration.
        """
        _check_pybullet_available()
        self.logger.info("Simulating robot motion...")
        ee_positions = []

        for joint_positions in desired_angles_trajectory:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()

            # Get end-effector position
            ee_pos = p.getLinkState(self.robot_id, p.getNumJoints(self.robot_id) - 1)[4]
            ee_positions.append(ee_pos)

            time.sleep(self.time_step / self.real_time_factor)

        # Plot trajectory with REAL GEOMETRY that appears in screenshots
        self.plot_trajectory(ee_positions)
        self.logger.info("Robot motion simulation completed.")
        return ee_positions[-1]  # Return the last end-effector position

    def simulate_robot_with_desired_angles(
        self, desired_angles: Sequence[float]
    ) -> None:
        """
        Simulates the robot using PyBullet with desired joint angles.

        Args:
            desired_angles (np.ndarray): Desired joint angles.
        """
        _check_pybullet_available()
        self.logger.info("Simulating robot with desired joint angles...")

        p.setJointMotorControlArray(
            self.robot_id,
            self.non_fixed_joints,
            p.POSITION_CONTROL,
            targetPositions=desired_angles,
            forces=[1000] * len(desired_angles),
        )

        time_step = 0.00001
        p.setTimeStep(time_step)
        try:
            while True:
                p.stepSimulation()
                time.sleep(time_step / self.real_time_factor)
        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
            self.logger.info("Robot simulation with desired angles completed.")
            self.close_simulation()
        except Exception:
            self.close_simulation()
            raise

    def close_simulation(self) -> None:
        """
        Closes the simulation.
        """
        self.logger.info("Closing simulation...")

        # Clear trajectory visualization
        self.clear_trajectory_visualization()

        self.disconnect_simulation()
        self.logger.info("Simulation closed.")

    def check_collisions(self) -> List[Tuple[int, int, Tuple[float, float, float]]]:
        """
        Checks for self-collisions in the simulation and returns contacts.

        Returns:
            list of (linkA, linkB, position) tuples for each contact point.
            Empty list if no collisions or simulation not started.
        """
        _check_pybullet_available()
        if self.robot_id is None:
            self.logger.warning(
                "Cannot check for collisions before simulation is started."
            )
            return []
        contacts = []
        # PyBullet's per-joint linkIndexA filter excludes base-link contacts
        # (base index is -1, never a non-fixed joint). Query without filter to
        # catch base<->link pairs (the most common self-collision on folded arms).
        for pt in p.getContactPoints(self.robot_id, self.robot_id) or []:
            link_a, link_b, position = pt[3], pt[4], pt[5]
            contacts.append((link_a, link_b, position))
            self.logger.warning(
                "Self-collision: link %s <-> link %s at %s", link_a, link_b, position
            )
        return contacts

    def step_simulation(self) -> None:
        """
        Steps the simulation forward by one time step.
        """
        _check_pybullet_available()
        self.logger.info("Setting up the simulation environment...")
        self.connect_simulation()
        self.add_additional_parameters()

    add_additional_parameters = _ControlConcern.__dict__["add_additional_parameters"]
    update_simulation_parameters = _ControlConcern.__dict__[
        "update_simulation_parameters"
    ]

    def manual_control(self) -> None:
        """
        Allows manual control of the robot through the PyBullet UI sliders.
        """
        _check_pybullet_available()
        self.logger.info("Starting manual control...")
        if not self.joint_params:
            self.add_joint_parameters()  # Ensure sliders are created
        self.add_additional_parameters()  # Additional controls like gravity and time step

        # Add reset button if it doesn't exist
        if self.reset_button is None:
            self.add_reset_button()

        try:
            while True:
                joint_positions = self.get_joint_parameters()
                if len(joint_positions) != len(self.non_fixed_joints):
                    raise ValueError(
                        "Number of joint positions "
                        f"({len(joint_positions)}) does not match number of "
                        f"non-fixed joints ({len(self.non_fixed_joints)})."
                    )
                self.set_joint_positions(joint_positions)
                self.check_collisions()  # Check for collisions in each step
                self.update_simulation_parameters()  # Update simulation parameters

                p.stepSimulation()
                time.sleep(self.time_step / self.real_time_factor)

                # Check if reset button exists before reading it
                if (
                    self.reset_button is not None
                    and p.readUserDebugParameter(self.reset_button) == 1
                ):
                    self.logger.info("Resetting simulation state...")
                    self.set_joint_positions(self.home_position)
                    break  # Exit manual control to restart trajectory loop
        except KeyboardInterrupt:
            self.logger.info("Manual control stopped.")
            self.close_simulation()
        except Exception:
            self.close_simulation()
            raise

    save_joint_states = _ControlConcern.__dict__["save_joint_states"]

    plot_trajectory_in_scene = _RenderingConcern.__dict__["plot_trajectory_in_scene"]

    def run(self, joint_trajectory: Sequence[Sequence[float]]) -> None:
        """
        Main loop for running the simulation.

        Runs the given trajectory once, then waits for the GUI reset button and
        switches between trajectory, wait-for-reset, and manual control modes.

        Args:
            joint_trajectory: Sequence of joint-angle configurations (one per
                simulation step) to execute, each a sequence of joint angles in
                radians.
        """
        try:
            reset_pressed = False
            mode = "trajectory"  # Mode can be 'trajectory' or 'manual'

            while True:
                if mode == "trajectory":
                    end_pos = self.run_trajectory(joint_trajectory)
                    self.logger.info("Trajectory completed. Waiting for reset...")
                    mode = "wait_reset"

                while mode == "wait_reset" and not reset_pressed:
                    p.stepSimulation()
                    time.sleep(0.01)

                    if p.readUserDebugParameter(self.reset_button) > 0:
                        self.logger.info(
                            "Reset button pressed. Returning to home position and entering manual control..."
                        )
                        self.set_joint_positions(self.home_position)
                        mode = "manual"
                        break

                if mode == "manual":
                    self.manual_control()
                    reset_pressed = False  # Reset the flag to restart the trajectory
                    mode = "trajectory"  # Go back to trajectory mode

        except KeyboardInterrupt:
            self.logger.info("Simulation stopped by user.")
            self.close_simulation()
        except Exception:
            self.close_simulation()
            raise

    def __del__(self) -> None:
        """
        Destructor to clean up trajectory visualization when simulation is destroyed.
        """
        try:
            if hasattr(self, "trajectory_body_ids"):
                self.clear_trajectory_visualization()
        except Exception:
            logger = getattr(self, "logger", None)
            if logger is not None:
                try:
                    logger.debug(
                        "Failed to clear trajectory visualization during cleanup.",
                        exc_info=True,
                    )
                except Exception:
                    pass


del _RenderingConcern
del _ControlConcern
_runtime._install_compatibility_proxy(_sys.modules[__name__])
del _runtime
del _sys
