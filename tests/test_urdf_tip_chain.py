#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for end-effector selection and tip_link-restricted screw extraction.

Covers the 1.4.2 fixes:
- the default end effector no longer depends on PYTHONHASHSEED (set order);
- tip_link is validated against the root -> tip joint chain;
- extract_screw_axes only builds columns for actuated joints on that chain.

PyBullet is the independent reference for FK and Jacobians (world frame,
metres / radians, q = joint angles of panda_joint1..7).

Copyright (c) 2025 Mohamed Aboelnasr
"""

import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_PANDA = (
    REPO_ROOT / "ManipulaPy" / "ManipulaPy_data" / "franka_panda" / "panda.urdf"
)
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

# base_link -> link1 -> link2 is the arm; "sensor" hangs off base_link on a
# fixed joint (no actuated joint to the root, like the UR5 "base" leaf);
# "other_root" -> "other_child" is a second, disconnected tree.
BRANCHED_URDF = """<?xml version="1.0"?>
<robot name="branched">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="sensor"/>
  <link name="other_root"/>
  <link name="other_child"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="sensor_mount" type="fixed">
    <parent link="base_link"/><child link="sensor"/>
    <origin xyz="0 0.1 0"/>
  </joint>
  <joint name="other_fixed" type="fixed">
    <parent link="other_root"/><child link="other_child"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def isolated_dir(tmp_path_factory):
    """Copies of the Panda URDFs outside any package workspace.

    Only the kinematic tree matters here. Loading in place makes the
    package:// resolver scan the whole checkout once per mesh URI, which
    takes minutes on slow filesystems (e.g. WSL /mnt/c).
    """
    return tmp_path_factory.mktemp("urdf")


@pytest.fixture(scope="module")
def bundled_panda(isolated_dir):
    return Path(shutil.copy(BUNDLED_PANDA, isolated_dir / "bundled_panda.urdf"))


def _load_quiet(path):
    from ManipulaPy.urdf import URDF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return URDF.load(path)


def _branched():
    from ManipulaPy.urdf import URDF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # multi-root warning is expected here
        robot = URDF.from_xml_string(BRANCHED_URDF)
        robot.root_links  # build the (lazy) kinematic structure now
        return robot


# ---------------------------------------------------------------- PyBullet


@pytest.fixture(scope="module")
def pb_panda(isolated_dir):
    """PyBullet's Panda (has panda_grasptarget), loaded in DIRECT mode.

    "path" is an isolated copy for the native parser; PyBullet loads the
    original so it can find its meshes.
    """
    p = pytest.importorskip("pybullet")
    pybullet_data = pytest.importorskip("pybullet_data")
    path = Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"
    cid = p.connect(p.DIRECT)
    body = p.loadURDF(str(path), useFixedBase=True, physicsClientId=cid)
    links, joints, movable = {}, {}, []
    for i in range(p.getNumJoints(body, physicsClientId=cid)):
        info = p.getJointInfo(body, i, physicsClientId=cid)
        links[info[12].decode()] = i
        joints[info[1].decode()] = i
        if info[2] != p.JOINT_FIXED:
            movable.append(i)
    copy = Path(shutil.copy(path, isolated_dir / "pybullet_panda.urdf"))
    yield {
        "p": p,
        "cid": cid,
        "body": body,
        "path": copy,
        "links": links,
        "joints": joints,
        "movable": movable,
    }
    p.disconnect(cid)


def _pb_set(pb, q):
    for name, angle in zip(ARM_JOINTS, q):
        pb["p"].resetJointState(
            pb["body"], pb["joints"][name], angle, physicsClientId=pb["cid"]
        )


def _pb_link_pose(pb, link, q):
    """World-frame 4x4 pose of a URDF link frame at arm angles q (rad)."""
    p = pb["p"]
    _pb_set(pb, q)
    state = p.getLinkState(
        pb["body"],
        pb["links"][link],
        computeForwardKinematics=True,
        physicsClientId=pb["cid"],
    )
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    T[:3, 3] = state[4]
    return T


def _rotation_error_rad(Ra, Rb):
    """Angle (rad) of Ra^T Rb, accurate for small errors."""
    R = Ra.T @ Rb
    vee = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.arctan2(np.linalg.norm(vee), (np.trace(R) - 1.0) / 2.0))


def _random_q(limits, count, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([l for l, _ in limits])
    hi = np.array([h for _, h in limits])
    return rng.uniform(lo, hi, size=(count, len(limits)))


# ------------------------------------------------- F1: default end effector


class TestDefaultEndEffector:
    def test_default_tip_is_hash_seed_independent(self, bundled_panda) -> None:
        """Same tip and M under PYTHONHASHSEED 0..3 (set order varies)."""
        script = (
            "import warnings; warnings.simplefilter('ignore')\n"
            "from ManipulaPy.urdf import URDF\n"
            f"u = URDF.load({str(bundled_panda)!r})\n"
            "p = u.extract_screw_axes()\n"
            "print('RESULT', u.end_effector_link.name, p['joint_names'],\n"
            "      p['M'].round(9).tolist())\n"
        )
        outputs = set()
        for seed in range(4):
            env = dict(os.environ, PYTHONHASHSEED=str(seed), MANIPULAPY_QUIET="1")
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            lines = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")]
            assert len(lines) == 1, result.stdout
            outputs.add(lines[0])
        assert len(outputs) == 1, outputs
        assert outputs.pop().startswith("RESULT panda_leftfinger ")

    def test_ambiguous_default_warns_and_picks_sorted_first(
        self, bundled_panda
    ) -> None:
        from ManipulaPy.urdf import URDF

        with pytest.warns(UserWarning, match="panda_leftfinger.*panda_rightfinger"):
            robot = URDF.load(bundled_panda)
            ee = robot.end_effector_link
        assert ee.name == "panda_leftfinger"

    def test_leaf_without_actuated_joints_is_not_a_candidate(self) -> None:
        """'sensor' sorts after 'link2' but must never win; only one candidate."""
        robot = _branched()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert robot.end_effector_link.name == "link2"

    def test_roots_are_sorted(self) -> None:
        assert [r.name for r in _branched().root_links] == ["base_link", "other_root"]


# --------------------------------------------------- F1: tip_link validation


class TestTipLinkValidation:
    def test_unknown_tip_link_raises(self) -> None:
        with pytest.raises(ValueError, match="no_such_link"):
            _branched().extract_screw_axes(tip_link="no_such_link")

    def test_tip_without_actuated_joint_raises(self) -> None:
        with pytest.raises(ValueError, match="sensor"):
            _branched().extract_screw_axes(tip_link="sensor")

    @pytest.mark.parametrize("tip", ["other_child", "other_root"])
    def test_tip_unreachable_from_root_raises(self, tip) -> None:
        with pytest.raises(ValueError, match=tip):
            _branched().extract_screw_axes(tip_link=tip)

    def test_grasptarget_home_matches_pybullet(self, pb_panda) -> None:
        params = _load_quiet(pb_panda["path"]).extract_screw_axes(
            tip_link="panda_grasptarget"
        )
        T_pb = _pb_link_pose(pb_panda, "panda_grasptarget", np.zeros(7))
        np.testing.assert_allclose(params["M"][:3, 3], T_pb[:3, 3], atol=1e-6)
        assert _rotation_error_rad(params["M"][:3, :3], T_pb[:3, :3]) < 1e-6


# --------------------------------------------- F2: chain-restricted screws


class TestChainRestrictedScrews:
    def test_grasptarget_has_seven_arm_columns(self, pb_panda) -> None:
        params = _load_quiet(pb_panda["path"]).extract_screw_axes(
            tip_link="panda_grasptarget"
        )
        assert params["joint_names"] == ARM_JOINTS
        assert params["S_list"].shape == (6, 7)
        assert params["B_list"].shape == (6, 7)
        assert params["omega_list"].shape == (3, 7)
        assert params["r_list"].shape == (3, 7)
        assert len(params["joint_limits"]) == 7
        assert len(params["G_list"]) == 7
        assert len(params["Mlist_per_link"]) == 7

    def test_link3_has_three_columns(self, pb_panda) -> None:
        params = _load_quiet(pb_panda["path"]).extract_screw_axes(
            tip_link="panda_link3"
        )
        assert params["joint_names"] == ARM_JOINTS[:3]
        assert params["S_list"].shape == (6, 3)
        assert len(params["joint_limits"]) == 3

    def test_limits_follow_chain_order(self, pb_panda) -> None:
        robot = _load_quiet(pb_panda["path"])
        params = robot.extract_screw_axes(tip_link="panda_grasptarget")
        expected = [
            (robot.joint_map[n].limit.lower, robot.joint_map[n].limit.upper)
            for n in ARM_JOINTS
        ]
        assert params["joint_limits"] == expected

    def test_default_tip_keeps_finger_column(self, bundled_panda) -> None:
        """No tip_link: chain to panda_leftfinger, mimic finger excluded."""
        params = _load_quiet(bundled_panda).extract_screw_axes()
        assert params["joint_names"] == ARM_JOINTS + ["panda_finger_joint1"]
        assert params["S_list"].shape == (6, 8)

    def _manipulator(self, pb_panda):
        from ManipulaPy.kinematics import SerialManipulator

        params = _load_quiet(pb_panda["path"]).extract_screw_axes(
            tip_link="panda_grasptarget"
        )
        S = params["S_list"]
        robot = SerialManipulator(
            M_list=params["M"],
            omega_list=S[:3, :],
            S_list=S,
            B_list=params["B_list"],
            joint_limits=params["joint_limits"],
        )
        return robot, params

    def test_fk_matches_pybullet_at_random_q(self, pb_panda) -> None:
        robot, params = self._manipulator(pb_panda)
        for q in _random_q(params["joint_limits"], 20, seed=17):
            T_pb = _pb_link_pose(pb_panda, "panda_grasptarget", q)
            for frame in ("space", "body"):
                T = np.asarray(robot.forward_kinematics(q, frame=frame))
                np.testing.assert_allclose(T[:3, 3], T_pb[:3, 3], atol=1e-6)
                assert _rotation_error_rad(T[:3, :3], T_pb[:3, :3]) < 1e-6

    def test_to_serial_manipulator_matches_pybullet_at_home(self, pb_panda) -> None:
        """FK ignores omega_list when S/B are given, so check it directly too."""
        urdf = _load_quiet(pb_panda["path"])
        robot = urdf.to_serial_manipulator(tip_link="panda_grasptarget")
        S = urdf.extract_screw_axes(tip_link="panda_grasptarget")["S_list"]
        np.testing.assert_array_equal(np.asarray(robot.omega_list), S[:3, :])
        home = np.array(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
        )
        T = np.asarray(robot.forward_kinematics(home))
        T_pb = _pb_link_pose(pb_panda, "panda_grasptarget", home)
        np.testing.assert_allclose(T[:3, 3], T_pb[:3, 3], atol=1e-6)
        assert _rotation_error_rad(T[:3, :3], T_pb[:3, :3]) < 1e-6

    def test_space_jacobian_matches_pybullet(self, pb_panda) -> None:
        """PyBullet's J is [linear at the tip point; angular], world frame.

        The space Jacobian columns are twists [w; v_s] with v_s the velocity
        of the point at the world origin, so v_tip = v_s + w x p_tip.
        """
        p = pb_panda["p"]
        robot, params = self._manipulator(pb_panda)
        n_movable = len(pb_panda["movable"])
        for q in _random_q(params["joint_limits"], 3, seed=5):
            Js = np.asarray(robot.jacobian(q, frame="space"))
            p_tip = np.asarray(robot.forward_kinematics(q))[:3, 3]
            Jw, Jv = Js[:3], Js[3:]
            Jv_tip = Jv + np.cross(Jw.T, p_tip).T

            _pb_set(pb_panda, q)
            q_all = [
                p.getJointState(pb_panda["body"], j, physicsClientId=pb_panda["cid"])[0]
                for j in pb_panda["movable"]
            ]
            lin, ang = p.calculateJacobian(
                pb_panda["body"],
                pb_panda["links"]["panda_grasptarget"],
                [0, 0, 0],
                q_all,
                [0.0] * n_movable,
                [0.0] * n_movable,
                physicsClientId=pb_panda["cid"],
            )
            arm_cols = [
                pb_panda["movable"].index(pb_panda["joints"][n]) for n in ARM_JOINTS
            ]
            np.testing.assert_allclose(Jv_tip, np.asarray(lin)[:, arm_cols], atol=1e-6)
            np.testing.assert_allclose(Jw, np.asarray(ang)[:, arm_cols], atol=1e-6)
