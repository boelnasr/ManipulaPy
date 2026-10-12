#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDFToSerialManipulator configuration contract on branched robots.

The processor's configuration vector is the screw-axis chain vector (one
value per Slist column, in robot_data["joint_names"] order), not the whole
URDF's actuated-joint vector. Joint values are radians / metres; transforms
are world (root link) frame.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from ManipulaPy.urdf_processor import URDFToSerialManipulator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "ManipulaPy" / "ManipulaPy_data"
JACO_6DOF = DATA / "kinova" / "jaco" / "jaco_6dof.urdf"
UR5 = DATA / "universal_robots" / "ur5" / "ur5.urdf"

# "z_side" hangs off base_link on its own revolute joint, declared first so
# the whole-robot actuated order is [side, j1, j2] while the default chain
# (tip "arm_tip", first leaf by name) is [j1, j2]. A prefix slice of the
# whole-robot vector would therefore pick the wrong joints.
TWO_BRANCH_URDF = """<?xml version="1.0"?>
<robot name="two_branch">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="arm_tip"/>
  <link name="z_side"/>
  <joint name="side" type="revolute">
    <parent link="base_link"/><child link="z_side"/>
    <origin xyz="0 0.2 0"/><axis xyz="1 0 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="arm_tip"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _load(path: Path) -> URDFToSerialManipulator:
    with warnings.catch_warnings():
        # Branched robots warn about several end-effector candidates.
        warnings.simplefilter("ignore", UserWarning)
        return URDFToSerialManipulator(path)


@pytest.fixture(scope="module")
def isolated_dir(tmp_path_factory) -> Path:
    """Copies outside the checkout: in-place package:// discovery is slow."""
    return tmp_path_factory.mktemp("urdf_processor_chain")


@pytest.fixture(scope="module")
def two_branch(isolated_dir) -> URDFToSerialManipulator:
    path = isolated_dir / "two_branch.urdf"
    path.write_text(TWO_BRANCH_URDF)
    return _load(path)


@pytest.fixture(scope="module")
def jaco(isolated_dir) -> URDFToSerialManipulator:
    return _load(Path(shutil.copy(JACO_6DOF, isolated_dir / "jaco_6dof.urdf")))


@pytest.fixture(scope="module")
def ur5(isolated_dir) -> URDFToSerialManipulator:
    return _load(Path(shutil.copy(UR5, isolated_dir / "ur5.urdf")))


@pytest.fixture(params=["two_branch", "jaco"])
def branched(request) -> URDFToSerialManipulator:
    return request.getfixturevalue(request.param)


def _random_chain_cfgs(proc: URDFToSerialManipulator, n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    # From robot_data, not the properties under test.
    lim = np.array(proc.robot_data["joint_limits"])
    return rng.uniform(lim[:, 0], lim[:, 1], size=(n, len(lim)))


class TestBranchedContract:
    def test_robot_has_joints_off_the_chain(self, branched) -> None:
        assert branched.robot.num_dofs > branched.robot_data["Slist"].shape[1]

    def test_num_dofs_is_screw_columns(self, branched) -> None:
        assert branched.num_dofs == branched.robot_data["Slist"].shape[1]

    def test_joint_names_follow_chain(self, branched) -> None:
        assert branched.joint_names == branched.robot_data["joint_names"]

    def test_two_branch_chain_names(self, two_branch) -> None:
        assert two_branch.joint_names == ["j1", "j2"]

    def test_joint_limits_array_follows_chain(self, branched) -> None:
        limits = branched.joint_limits_array
        assert limits.shape == (branched.num_dofs, 2)
        np.testing.assert_allclose(limits, branched.robot_data["joint_limits"])

    def test_fk_accepts_zeros_of_num_dofs(self, branched) -> None:
        T = branched.serial_manipulator.forward_kinematics(np.zeros(branched.num_dofs))
        np.testing.assert_allclose(T, branched.robot_data["M"], atol=1e-12)

    def test_batch_fk_matches_serial_fk(self, branched) -> None:
        cfgs = _random_chain_cfgs(branched, 5)
        batch = branched.batch_forward_kinematics(
            cfgs, link_name=branched.end_effector_name
        )
        assert batch.shape == (5, 4, 4)
        for q, T in zip(cfgs, batch):
            np.testing.assert_allclose(T, branched.forward_kinematics(q), atol=1e-10)

    def test_batch_fk_all_links_accepts_chain_cfgs(self, branched) -> None:
        fk = branched.batch_forward_kinematics(_random_chain_cfgs(branched, 3))
        assert set(branched.link_names) <= set(fk)

    def test_end_effector_transforms_accept_chain_cfgs(self, branched) -> None:
        cfgs = _random_chain_cfgs(branched, 2)
        transforms = branched.get_end_effector_transforms(cfgs)
        np.testing.assert_allclose(
            transforms[0], branched.forward_kinematics(cfgs[0]), atol=1e-10
        )

    def test_batch_fk_rejects_whole_robot_width(self, branched) -> None:
        with pytest.raises(ValueError):
            branched.batch_forward_kinematics(np.zeros((1, branched.robot.num_dofs)))

    def test_link_fk_maps_chain_vector_by_name(self, branched) -> None:
        q = _random_chain_cfgs(branched, 1)[0]
        fk = branched.link_fk(q)
        np.testing.assert_allclose(
            fk[branched.end_effector_name],
            branched.forward_kinematics(q),
            atol=1e-10,
        )

    def test_link_fk_leaves_off_chain_joints_at_zero(self, two_branch) -> None:
        two_branch.link_fk({"side": 1.0})
        fk = two_branch.link_fk(np.zeros(2))
        np.testing.assert_allclose(fk["z_side"][:3, :3], np.eye(3), atol=1e-12)

    def test_get_transform_accepts_chain_vector(self, branched) -> None:
        q = _random_chain_cfgs(branched, 1)[0]
        T = branched.get_transform(branched.end_effector_name, cfg=q)
        np.testing.assert_allclose(T, branched.forward_kinematics(q), atol=1e-10)

    def test_iterative_ik_default_guess_is_chain_sized(self, branched) -> None:
        q_goal = 0.5 * _random_chain_cfgs(branched, 1)[0]
        theta, _, _ = branched.inverse_kinematics(
            branched.forward_kinematics(q_goal), method="iterative"
        )
        assert np.asarray(theta).shape == (branched.num_dofs,)


class TestSingleChainUnchanged:
    """On a single-chain robot the chain is the whole robot."""

    def test_members_equal_whole_robot(self, ur5) -> None:
        assert ur5.num_dofs == ur5.robot.num_dofs == 6
        assert ur5.joint_names == ur5.robot.actuated_joint_names
        np.testing.assert_array_equal(
            ur5.joint_limits_array, np.array(ur5.robot.joint_limits)
        )

    def test_batch_fk_equals_urdf_batch_fk(self, ur5) -> None:
        cfgs = _random_chain_cfgs(ur5, 4)
        ours = ur5.batch_forward_kinematics(cfgs)
        ref = ur5.robot.link_fk_batch(cfgs)
        assert ours.keys() == ref.keys()
        for name in ref:
            np.testing.assert_array_equal(ours[name], ref[name])

    def test_link_fk_equals_urdf_link_fk(self, ur5) -> None:
        q = _random_chain_cfgs(ur5, 1)[0]
        ours = ur5.link_fk(q)
        ref = ur5.robot.link_fk(q, use_names=True)
        for name in ref:
            np.testing.assert_array_equal(ours[name], ref[name])
