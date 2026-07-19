"""Controller concern definitions for the simulation class."""

# ruff: noqa: UP006, UP035 - annotations are frozen by the historical API.

from typing import Any, List, Optional, Sequence

import numpy as np

from ManipulaPy.control import ManipulatorController
from ManipulaPy.path_planning import TrajectoryPlanning as tp

from . import _runtime


class _ControlConcern:
    def set_robot_models(self, robot: Any, dynamics: Any) -> None:
        """
        Set pre-existing robot models to avoid reprocessing.

        Args:
            robot: SerialManipulator instance
            dynamics: ManipulatorDynamics instance
        """
        self.robot = robot
        self.dynamics = dynamics
        self.logger.info("Pre-existing robot models set successfully.")

    def initialize_planner_and_controller(self) -> None:
        """
        Initializes the trajectory planner and the manipulator controller.
        """
        self.trajectory_planner = tp(
            self.robot,
            self.urdf_file_path,
            self.dynamics,
            self.joint_limits,
            self.torque_limits,
        )
        self.controller = ManipulatorController(self.dynamics)

    def add_joint_parameters(self) -> None:
        """
        Adds GUI sliders for each joint.
        """
        _runtime._check_pybullet_available()
        if not self.joint_params:
            for i, joint_index in enumerate(self.non_fixed_joints):
                param_id = _runtime.p.addUserDebugParameter(
                    f"Joint {joint_index}",
                    self.joint_limits[i][0],
                    self.joint_limits[i][1],
                    0,
                )
                self.joint_params.append(param_id)

    def add_reset_button(self) -> None:
        """
        Adds a reset button to the simulation.
        """
        _runtime._check_pybullet_available()
        if self.reset_button is None:
            try:
                self.reset_button = _runtime.p.addUserDebugParameter("Reset", 1, 0, 1)
            except Exception as e:
                self.logger.error(f"Failed to add reset button: {e}")

    def set_joint_positions(
        self, joint_positions: Sequence[float], forces: Optional[Sequence[float]] = None
    ) -> None:
        """
        Sets the joint positions of the robot.

        Drives the non-fixed joints toward ``joint_positions`` using PyBullet's
        ``POSITION_CONTROL`` mode.

        Args:
            joint_positions: Target angles for the non-fixed joints, in radians,
                one entry per non-fixed joint.
            forces: Optional per-joint maximum motor force. When ``None``, forces
                are derived from ``self.torque_limits`` (collapsing each
                (min, max) pair to its largest absolute magnitude) or default to
                ``1000.0`` for every joint when no torque limits are configured.
        """
        _runtime._check_pybullet_available()
        n = len(self.non_fixed_joints)
        if forces is None:
            if getattr(self, "torque_limits", None) is not None:
                # PyBullet wants one scalar per joint, but torque_limits is
                # commonly a list of (min, max) pairs. Collapse each pair to
                # the maximum absolute magnitude so the motor can both push
                # and pull within the configured limits.
                torque_limits = np.asarray(self.torque_limits, dtype=float)
                if torque_limits.ndim == 2 and torque_limits.shape[1] == 2:
                    forces = np.max(np.abs(torque_limits), axis=1).tolist()
                else:
                    forces = torque_limits.tolist()
            else:
                forces = [1000.0] * n
        _runtime.p.setJointMotorControlArray(
            self.robot_id,
            self.non_fixed_joints,
            _runtime.p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=forces,
        )

    def get_joint_positions(self) -> np.ndarray:
        """
        Gets the current joint positions of the robot.
        """
        _runtime._check_pybullet_available()
        joint_positions = [
            _runtime.p.getJointState(self.robot_id, i)[0] for i in self.non_fixed_joints
        ]
        return np.array(joint_positions)

    def get_joint_parameters(self) -> List[float]:
        """
        Gets the current values of the GUI sliders.
        """
        _runtime._check_pybullet_available()
        return [
            _runtime.p.readUserDebugParameter(param_id)
            for param_id in self.joint_params
        ]

    def add_additional_parameters(self) -> None:
        """
        Adds additional GUI parameters for controlling physics properties like gravity and time step.
        """
        _runtime._check_pybullet_available()
        if not hasattr(self, "gravity_param"):
            self.gravity_param = _runtime.p.addUserDebugParameter(
                "Gravity", -20, 20, -9.81
            )
        if not hasattr(self, "time_step_param"):
            self.time_step_param = _runtime.p.addUserDebugParameter(
                "Time Step", 0.001, 0.1, self.time_step
            )

    def update_simulation_parameters(self) -> None:
        """
        Updates simulation parameters from GUI controls.
        """
        _runtime._check_pybullet_available()
        gravity = _runtime.p.readUserDebugParameter(self.gravity_param)
        time_step = _runtime.p.readUserDebugParameter(self.time_step_param)
        _runtime.p.setGravity(0, 0, gravity)
        _runtime.p.setTimeStep(time_step)
        self.time_step = time_step

    def save_joint_states(self, filename: str = "joint_states.csv") -> None:
        """
        Saves the joint states to a CSV file.

        Args:
            filename (str): The filename for the CSV file.
        """
        _runtime._check_pybullet_available()
        joint_states = [
            _runtime.p.getJointState(self.robot_id, i) for i in self.non_fixed_joints
        ]
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]

        data = np.column_stack((positions, velocities))
        np.savetxt(
            filename, data, delimiter=",", header="Position,Velocity", comments=""
        )
        self.logger.info(f"Joint states saved to {filename}.")
