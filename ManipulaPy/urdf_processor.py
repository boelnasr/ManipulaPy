#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Processor Module - ManipulaPy

This module provides comprehensive URDF (Unified Robot Description Format) processing
capabilities including conversion to SerialManipulator objects, extraction of kinematic
and dynamic parameters, and integration with PyBullet for simulation and visualization.

The processor uses the native ManipulaPy URDF parser which provides:
- NumPy 2.0+ compatibility
- Direct SerialManipulator/ManipulatorDynamics conversion
- Batch forward kinematics support
- Multi-robot scene management
- URDF modification and calibration

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

from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import numpy as np

# PyBullet is optional - only needed for PyBullet-based joint limits
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None

from .urdf import URDF, URDFModifier, Scene, validate_urdf
from .kinematics import SerialManipulator
from .dynamics import ManipulatorDynamics
from . import utils


class URDFToSerialManipulator:
    """
    A class to convert URDF files to SerialManipulator objects and simulate them using PyBullet.

    Supports multiple URDF parser backends:
        - "builtin": Native ManipulaPy parser (NumPy 2.0 compatible, default)
        - "urchin": Legacy urchin parser (requires urchin, not NumPy 2.0 compatible)
        - "pybullet": PyBullet-based parser (requires pybullet)

    Features:
        - Direct conversion to SerialManipulator and ManipulatorDynamics
        - Forward kinematics (single and batch)
        - URDF validation
        - Optional PyBullet joint limit extraction
        - Visualization and animation

    Example:
        >>> processor = URDFToSerialManipulator("robot.urdf")
        >>> manipulator = processor.serial_manipulator
        >>> T_ee = manipulator.forward_kinematics([0, 0, 0, 0, 0, 0])

        >>> # Batch FK for trajectory
        >>> configs = np.random.uniform(-np.pi, np.pi, (100, 6))
        >>> transforms = processor.batch_forward_kinematics(configs)
    """

    def __init__(
        self,
        urdf_name: Union[str, Path],
        use_pybullet_limits: bool = False,
        backend: str = "builtin",
        load_meshes: bool = False,
        validate: bool = False,
    ):
        """
        Initializes the object with the given urdf_name.

        Parameters:
            urdf_name (str | Path): Path to the URDF file.
            use_pybullet_limits (bool): Whether to override URDF limits with PyBullet's.
                                        Default False (use URDF limits directly).
            backend (str): Parser backend - "builtin" (default), "urchin", or "pybullet"
            load_meshes (bool): Whether to load mesh geometry. Default False.
            validate (bool): Whether to validate URDF structure. Default False.
        """
        self.urdf_name = str(urdf_name)
        self.backend = backend
        self.robot = URDF.load(urdf_name, backend=backend, load_meshes=load_meshes)

        # Optionally validate
        if validate:
            result = validate_urdf(self.robot)
            if not result.valid:
                import warnings
                warnings.warn(
                    f"URDF validation issues: {[str(i) for i in result.issues]}",
                    UserWarning,
                )

        # 1. Load URDF data (Slist, Blist, M, etc.) using native parser
        self.robot_data = self._extract_robot_data()

        # 2. Optionally retrieve limits from PyBullet and override
        if use_pybullet_limits:
            if not PYBULLET_AVAILABLE:
                import warnings
                warnings.warn(
                    "PyBullet not available, using URDF joint limits instead.",
                    UserWarning,
                )
                self.robot_data["joint_limits"] = self.robot.joint_limits
            else:
                pyb_joint_limits = self._get_joint_limits_from_pybullet()
                self.robot_data["joint_limits"] = pyb_joint_limits
        else:
            # Use limits from URDF parser or default to (-π, π)
            self.robot_data["joint_limits"] = self.robot.joint_limits

        # 3. Create SerialManipulator and dynamics
        self.serial_manipulator = self.initialize_serial_manipulator()
        self.dynamics = self.initialize_manipulator_dynamics()

    @staticmethod
    def transform_to_xyz(T: np.ndarray) -> np.ndarray:
        """
        Extracts the XYZ position from a 4x4 transformation matrix.
        Returns a 3-element NumPy array (x, y, z).
        """
        return np.array(T[0:3, 3])

    @staticmethod
    def get_link(robot: URDF, link_name: str):
        """
        Given a robot URDF and a link name, returns the link associated with that name.
        Returns None if not found.
        """
        return robot.get_link(link_name)

    @staticmethod
    def w_p_to_slist(w: np.ndarray, p: np.ndarray, robot_dof: int) -> np.ndarray:
        """
        Convert angular velocity (w) and position (p) vectors into screw axes (Slist).
        Slist has shape (6, robot_dof).
        """
        Slist = []
        for i in range(robot_dof):
            w_ = w[i]
            p_ = p[i]
            v_ = np.cross(-1 * w_, p_)
            Slist.append([w_[0], w_[1], w_[2], v_[0], v_[1], v_[2]])
        return np.transpose(Slist)

    def _extract_robot_data(self) -> dict:
        """
        Extract kinematic/dynamic parameters from the loaded URDF.

        Uses the native parser's extract_screw_axes() method which provides:
          - Home position matrix M
          - Slist (space-frame screw axes)
          - Blist (body-frame screw axes)
          - Glist (inertia/mass)
          - DOF count

        Returns:
            dict: Robot kinematic/dynamic parameters
        """
        # Use native parser's screw axis extraction
        params = self.robot.extract_screw_axes()

        return {
            "M": params["M"],
            "Slist": params["S_list"],
            "Blist": params["B_list"],
            "Glist": params["G_list"],
            "actuated_joints_num": self.robot.num_dofs,
            "joint_limits": params["joint_limits"],
        }

    def load_urdf(self, urdf_name: str) -> dict:
        """
        Load the URDF file and extract the necessary info for the robot model.

        DEPRECATED: Use _extract_robot_data() instead. This method is kept for
        backward compatibility but now delegates to the native parser.

        Parameters:
            urdf_name (str): Path to URDF file (ignored, uses self.robot)

        Returns:
            dict: Robot kinematic/dynamic parameters
        """
        import warnings
        warnings.warn(
            "load_urdf() is deprecated. Use _extract_robot_data() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._extract_robot_data()

    def _get_joint_limits_from_pybullet(self) -> List[Tuple[float, float]]:
        """
        Connect to PyBullet (in DIRECT mode, so no GUI),
        load the URDF, and retrieve per-joint limits for all revolute joints.

        Returns:
            List of (lower, upper) tuples in the order they appear as revolute in the URDF.

        Raises:
            RuntimeError: If PyBullet is not available.
        """
        if not PYBULLET_AVAILABLE:
            raise RuntimeError(
                "PyBullet not available. Install with: pip install pybullet"
            )

        # Connect in DIRECT mode so we don't open a GUI
        cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        try:
            # Load the URDF in PyBullet
            robot_id = p.loadURDF(self.urdf_name, useFixedBase=True)

            joint_limits = []

            total_joints = p.getNumJoints(robot_id)
            for i in range(total_joints):
                joint_info = p.getJointInfo(robot_id, i)
                joint_type = joint_info[2]  # e.g. p.JOINT_REVOLUTE, p.JOINT_FIXED, etc.
                if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    # If invalid (e.g., continuous), set to (-π, π)
                    if lower_limit > upper_limit:
                        lower_limit = -np.pi
                        upper_limit = np.pi
                    joint_limits.append((lower_limit, upper_limit))

        finally:
            p.disconnect(cid)

        return joint_limits

    def initialize_serial_manipulator(self) -> SerialManipulator:
        """
        Initializes a SerialManipulator object using the extracted URDF data.
        Overwrites the example joint limits with PyBullet-based ones if available.
        """
        data = self.robot_data

        # If we previously stored them from PyBullet, e.g. data["joint_limits"],
        # use them. Otherwise default to e.g. (-π, π).
        if "joint_limits" in data:
            jlimits = data["joint_limits"]
        else:
            jlimits = [(-np.pi, np.pi)] * data["actuated_joints_num"]

        return SerialManipulator(
            M_list=data["M"],
            omega_list=utils.extract_omega_list(data["Slist"]),
            S_list=data["Slist"],
            B_list=data["Blist"],
            G_list=data["Glist"],
            joint_limits=jlimits,
        )

    def initialize_manipulator_dynamics(self):
        """
        Initializes the ManipulatorDynamics object using the extracted URDF data.
        """
        data = self.robot_data
        self.manipulator_dynamics = ManipulatorDynamics(
            M_list=data["M"],
            omega_list=data["Slist"][:, :3],
            r_list=utils.extract_r_list(data["Slist"]),
            b_list=None,  # If needed, define or extract from URDF
            S_list=data["Slist"],
            B_list=data["Blist"],
            Glist=data["Glist"],
        )
        return self.manipulator_dynamics

    def visualize_robot(self, cfg=None):
        """
        Visualizes the URDF model.

        Parameters:
            cfg: Optional joint configuration (array or dict)
        """
        self.robot.show(cfg=cfg)

    def visualize_trajectory(
        self, cfg_trajectory=None, loop_time=3.0, use_collision=False
    ):
        """
        Animate robot along a trajectory.

        Parameters:
            cfg_trajectory: Joint trajectories as (N, DOF) array or dict {joint_name: array}
            loop_time: Animation duration in seconds
            use_collision: Use collision geometry instead of visual
        """
        actuated_joints = self.robot.actuated_joints

        # If a NumPy array, convert to a dictionary of joint_name -> configurations
        if cfg_trajectory is not None:
            if isinstance(cfg_trajectory, np.ndarray):
                expected_columns = len(actuated_joints)
                if cfg_trajectory.shape[1] != expected_columns:
                    raise ValueError(
                        f"Expected cfg_trajectory with {expected_columns} cols, got {cfg_trajectory.shape[1]}"
                    )
                cfg_trajectory = {
                    joint.name: cfg_trajectory[:, i]
                    for i, joint in enumerate(actuated_joints)
                    if i < cfg_trajectory.shape[1]
                }
            elif isinstance(cfg_trajectory, dict):
                if len(cfg_trajectory) != len(actuated_joints):
                    raise ValueError(
                        f"Expected {len(actuated_joints)} keys in cfg_trajectory, got {len(cfg_trajectory)}"
                    )
            else:
                raise TypeError(
                    "cfg_trajectory must be a numpy array or dict {joint_name: array([...])}."
                )
        else:
            # Default small motion
            cfg_trajectory = {joint.name: [0, np.pi / 2] for joint in actuated_joints}

        self.robot.animate(
            cfg_trajectory=cfg_trajectory,
            loop_time=loop_time,
            use_collision=use_collision,
        )

    def print_joint_info(self):
        """
        Returns the joint names instead of printing them to console.

        Returns:
            dict: Contains 'num_joints' and 'joint_names'
        """
        joint_names = [joint.name for joint in self.robot.joints]
        return {"num_joints": len(joint_names), "joint_names": joint_names}

    def get_serial_manipulator(self) -> SerialManipulator:
        """
        Get SerialManipulator directly from native URDF parser.

        This is a convenience method that uses the native parser's
        to_serial_manipulator() method directly.

        Returns:
            SerialManipulator instance
        """
        return self.robot.to_serial_manipulator()

    def get_manipulator_dynamics(self) -> ManipulatorDynamics:
        """
        Get ManipulatorDynamics directly from native URDF parser.

        This is a convenience method that uses the native parser's
        to_manipulator_dynamics() method directly.

        Returns:
            ManipulatorDynamics instance
        """
        return self.robot.to_manipulator_dynamics()

    # ==================== New Methods for Enhanced Functionality ====================

    def forward_kinematics(
        self,
        cfg: Union[np.ndarray, List[float]],
        frame: str = "space",
    ) -> np.ndarray:
        """
        Compute forward kinematics using SerialManipulator.

        Parameters:
            cfg: Joint configuration array
            frame: Reference frame ("space" or "body")

        Returns:
            4x4 transformation matrix of end-effector
        """
        return self.serial_manipulator.forward_kinematics(cfg, frame=frame)

    def link_fk(
        self,
        cfg: Optional[Union[np.ndarray, List[float], Dict[str, float]]] = None,
        use_names: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute forward kinematics for all links using native URDF parser.

        This is faster for getting transforms of all links, not just end-effector.

        Parameters:
            cfg: Joint configuration (None uses current config)
            use_names: Return dict with string keys (always True for compatibility)

        Returns:
            Dict mapping link names to 4x4 transformation matrices
        """
        return self.robot.link_fk(cfg, use_names=use_names)

    def batch_forward_kinematics(
        self,
        cfgs: np.ndarray,
        link_name: Optional[str] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute forward kinematics for multiple configurations (vectorized).

        This is 50x+ faster than calling forward_kinematics in a loop.

        Parameters:
            cfgs: Array of configurations with shape (N, num_dofs)
            link_name: If provided, return only transforms for this link.
                       If None, return transforms for all links.

        Returns:
            If link_name is provided: (N, 4, 4) array of transforms
            If link_name is None: Dict mapping link names to (N, 4, 4) arrays

        Example:
            >>> configs = np.random.uniform(-np.pi, np.pi, (100, 6))
            >>> ee_transforms = processor.batch_forward_kinematics(configs, "ee_link")
            >>> all_transforms = processor.batch_forward_kinematics(configs)
        """
        batch_fk = self.robot.link_fk_batch(cfgs)

        if link_name is not None:
            if link_name not in batch_fk:
                raise ValueError(
                    f"Unknown link: {link_name}. Available: {list(batch_fk.keys())}"
                )
            return batch_fk[link_name]

        return batch_fk

    def get_end_effector_transforms(
        self, cfgs: np.ndarray
    ) -> np.ndarray:
        """
        Get end-effector transforms for multiple configurations.

        Convenience method for batch FK of end-effector only.

        Parameters:
            cfgs: Array of configurations with shape (N, num_dofs)

        Returns:
            (N, 4, 4) array of end-effector transforms
        """
        ee_name = self.robot.end_effector_link.name
        return self.batch_forward_kinematics(cfgs, link_name=ee_name)

    def jacobian(
        self,
        cfg: Union[np.ndarray, List[float]],
        frame: str = "space",
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix at given configuration.

        Parameters:
            cfg: Joint configuration array
            frame: Reference frame ("space" or "body")

        Returns:
            6xN Jacobian matrix
        """
        return self.serial_manipulator.jacobian(cfg, frame=frame)

    def inverse_kinematics(
        self,
        T_desired: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "robust",
        **kwargs,
    ) -> Tuple[np.ndarray, bool, int]:
        """
        Compute inverse kinematics to reach desired pose.

        Parameters:
            T_desired: Desired 4x4 transformation matrix
            initial_guess: Starting joint configuration (None = zeros)
            method: IK method - "robust" (default), "smart", or "iterative"
            **kwargs: Additional arguments passed to IK solver

        Returns:
            Tuple of (theta, success, iterations)
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.robot.num_dofs)

        if method == "robust":
            theta, success, iters, _ = self.serial_manipulator.robust_inverse_kinematics(
                T_desired, **kwargs
            )
            return theta, success, iters
        elif method == "smart":
            return self.serial_manipulator.smart_inverse_kinematics(
                T_desired, **kwargs
            )
        else:  # iterative
            return self.serial_manipulator.iterative_inverse_kinematics(
                T_desired, initial_guess, **kwargs
            )

    @property
    def num_dofs(self) -> int:
        """Number of actuated degrees of freedom."""
        return self.robot.num_dofs

    @property
    def joint_names(self) -> List[str]:
        """Names of actuated joints in order."""
        return self.robot.actuated_joint_names

    @property
    def link_names(self) -> List[str]:
        """Names of all links."""
        return [link.name for link in self.robot.links]

    @property
    def end_effector_name(self) -> str:
        """Name of the end-effector link."""
        return self.robot.end_effector_link.name

    @property
    def joint_limits_array(self) -> np.ndarray:
        """
        Joint limits as (N, 2) array.

        Returns:
            Array with shape (num_dofs, 2) where [:, 0] is lower and [:, 1] is upper
        """
        return np.array(self.robot.joint_limits)

    def get_transform(
        self,
        frame_to: str,
        frame_from: str = "world",
        cfg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get transform between two frames.

        Parameters:
            frame_to: Target frame (link name)
            frame_from: Source frame (link name or "world")
            cfg: Joint configuration (None = current config)

        Returns:
            4x4 transformation matrix from frame_from to frame_to
        """
        return self.robot.get_transform(frame_to, frame_from, cfg)

    def create_modifier(self) -> URDFModifier:
        """
        Create a URDFModifier for calibration and payload simulation.

        The modifier creates a deep copy of the URDF, so modifications
        don't affect the original robot.

        Returns:
            URDFModifier instance

        Example:
            >>> modifier = processor.create_modifier()
            >>> modifier.offset_joint_zero("joint1", 0.01)
            >>> modifier.add_payload("ee_link", mass=1.0, com=[0, 0, 0.1])
            >>> calibrated_robot = modifier.urdf
        """
        return URDFModifier(self.robot)

    def validate(self) -> Dict:
        """
        Validate the URDF structure.

        Returns:
            Dict with 'valid' (bool) and 'issues' (list) keys
        """
        result = validate_urdf(self.robot)
        return {
            "valid": result.valid,
            "issues": [
                {"severity": i.severity.name, "message": i.message}
                for i in result.issues
            ],
        }

    def __repr__(self) -> str:
        return (
            f"URDFToSerialManipulator(urdf='{self.urdf_name}', "
            f"dofs={self.num_dofs}, backend='{self.backend}')"
        )


# ==================== Convenience Functions ====================


def load_robot(
    urdf_path: Union[str, Path],
    backend: str = "builtin",
    use_pybullet_limits: bool = False,
) -> URDFToSerialManipulator:
    """
    Load a robot from URDF file.

    Convenience function that creates a URDFToSerialManipulator.

    Parameters:
        urdf_path: Path to URDF file
        backend: Parser backend ("builtin", "urchin", or "pybullet")
        use_pybullet_limits: Whether to use PyBullet for joint limits

    Returns:
        URDFToSerialManipulator instance

    Example:
        >>> robot = load_robot("panda.urdf")
        >>> T = robot.forward_kinematics([0, 0, 0, -1.57, 0, 1.57, 0])
    """
    return URDFToSerialManipulator(
        urdf_path,
        backend=backend,
        use_pybullet_limits=use_pybullet_limits,
    )


def create_multi_robot_scene(
    name: str = "scene",
) -> Scene:
    """
    Create a multi-robot scene for managing multiple robots in a shared workspace.

    Parameters:
        name: Name for the scene

    Returns:
        Scene instance

    Example:
        >>> scene = create_multi_robot_scene("workcell")
        >>> scene.add_robot("left_arm", left_robot.robot, base_xyz=[0, 0.5, 0])
        >>> scene.add_robot("right_arm", right_robot.robot, base_xyz=[0, -0.5, 0])
        >>> world_fk = scene.world_link_fk({"left_arm": cfg1, "right_arm": cfg2})
    """
    return Scene(name)
