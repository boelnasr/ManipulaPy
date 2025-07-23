#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Processor Module - ManipulaPy

This module provides comprehensive URDF (Unified Robot Description Format) processing
capabilities including conversion to SerialManipulator objects, extraction of kinematic
and dynamic parameters, and integration with PyBullet for simulation and visualization.

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


from urchin.urdf import URDF
import numpy as np
import pybullet as p
import pybullet_data

from .kinematics import SerialManipulator
from .dynamics import ManipulatorDynamics
from . import utils


class URDFToSerialManipulator:
    """
    A class to convert URDF files to SerialManipulator objects and simulate them using PyBullet.
    """

    def __init__(self, urdf_name: str, use_pybullet_limits: bool = True):
        """
        Initializes the object with the given urdf_name.

        Parameters:
            urdf_name (str): The name of the URDF file.
            use_pybullet_limits (bool): Whether to override URDF or default joint limits
                                        with those from PyBullet.
        """
        self.urdf_name = urdf_name
        self.robot = URDF.load(urdf_name)

        # 1. Load URDF data (Slist, Blist, M, etc.)
        self.robot_data = self.load_urdf(urdf_name)

        # 2. Optionally retrieve limits from PyBullet and override
        if use_pybullet_limits:
            pyb_joint_limits = self._get_joint_limits_from_pybullet()
            # Store them in self.robot_data so the manipulator can pick them up
            self.robot_data["joint_limits"] = pyb_joint_limits
        else:
            # If not using PyBullet, fallback to a default
            # (Here, we just fill with (-π, π); adapt as needed)
            self.robot_data["joint_limits"] = [
                (-np.pi, np.pi) for _ in range(self.robot_data["actuated_joints_num"])
            ]

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
        for link in robot.links:
            if link.name == link_name:
                return link
        return None

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

    def load_urdf(self, urdf_name: str) -> dict:
        """
        Load the URDF file and extract the necessary info for the robot model:
          - Home position matrix M
          - Slist (space-frame screw axes)
          - Blist (body-frame screw axes)
          - Glist (inertia/mass)
          - DOF count
        """
        robot = URDF.load(urdf_name)
        joint_num = len(
            [joint for joint in robot.actuated_joints if joint.joint_type != "fixed"]
        )

        p_ = []  # Joint positions
        w_ = []  # Joint rotation axes
        M_list = np.eye(4)
        Glist = []

        link_fk = robot.link_fk()
        link_fk = {link.name: fk for link, fk in link_fk.items()}

        for joint in robot.actuated_joints:
            if joint.joint_type == "fixed":
                continue

            child_link_name = joint.child
            child_link = self.get_link(robot, child_link_name)
            if not child_link:
                continue

            # Convert 4x4 transform to position
            p_.append(self.transform_to_xyz(link_fk[joint.child]))

            # Build inertia matrix G
            G = np.eye(6)
            if child_link.inertial:
                G[0:3, 0:3] = child_link.inertial.inertia
                G[3:6, 3:6] = child_link.inertial.mass * np.eye(3)
            Glist.append(G)

            # Compute the rotation axis in the global frame
            child_M = link_fk[joint.child]
            child_w = np.array(child_M[0:3, 0:3] @ np.array(joint.axis).T)
            w_.append(child_w)

            # If there's an origin transform for the inertial, apply it
            if child_link.inertial and child_link.inertial.origin is not None:
                child_M = child_M @ child_link.inertial.origin
            M_list = np.dot(M_list, child_M)

        # Build Slist (6 x DOF) from w and p
        Slist = self.w_p_to_slist(w_, p_, joint_num)

        # Build Blist via Adjoint of inv(M_list)
        Tsb_inv = np.linalg.inv(M_list)
        Ad_Tsb_inv = utils.adjoint_transform(Tsb_inv)
        Blist = np.dot(Ad_Tsb_inv, Slist)

        return {
            "M": M_list,
            "Slist": Slist,
            "Blist": Blist,
            "Glist": Glist,
            "actuated_joints_num": joint_num,
        }

    def _get_joint_limits_from_pybullet(self):
        """
        Connect to PyBullet (in DIRECT mode, so no GUI),
        load the URDF, and retrieve per-joint limits for all revolute joints.
        No prints, just returns a list of (lower, upper) in the
        order they appear as revolute in the URDF.
        """
        # Connect in DIRECT mode so we don't open a GUI
        cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the URDF in PyBullet
        # We assume the file is in the local path or recognized by PyBullet
        robot_id = p.loadURDF(self.urdf_name, useFixedBase=True)

        joint_limits = []
        # We'll track how many revolve joints we've seen so far
        revolve_count = 0

        # We'll also count total joints to match the order from URDF's "actuated_joints"
        # But typically PyBullet's joint index ordering matches the URDF link ordering
        total_joints = p.getNumJoints(robot_id)
        for i in range(total_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_type = joint_info[2]  # e.g. p.JOINT_REVOLUTE, p.JOINT_FIXED, etc.
            if joint_type == p.JOINT_REVOLUTE:
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                # If invalid (e.g., continuous), set to (-π, π)
                if lower_limit > upper_limit:
                    lower_limit = -np.pi
                    upper_limit = np.pi
                joint_limits.append((lower_limit, upper_limit))
                revolve_count += 1

        p.disconnect(cid)

        # If revolve_count < self.robot_data["actuated_joints_num"], we might
        # have fewer revolve joints in PyBullet. Adjust or log as needed.
        # We'll simply return what we found:
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

    def visualize_robot(self):
        """
        Visualizes the URDF model using matplotlib (from urchin).
        """
        self.robot.show()

    def visualize_trajectory(
        self, cfg_trajectory=None, loop_time=3.0, use_collision=False
    ):
        actuated_joints = [
            joint for joint in self.robot.joints if joint.joint_type != "fixed"
        ]

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
        """
        joint_names = [joint.name for joint in self.robot.joints]
        # Return them in a list or dictionary, no printing
        return {"num_joints": len(joint_names), "joint_names": joint_names}
