#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for mimic joints on the root -> tip chain in screw-axis extraction.

The PoE model has one column per actuated, non-mimic chain joint, so a mimic
joint on the chain is frozen at q = 0 while native ``link_fk`` moves it with
its driver. An explicit ``tip_link`` behind a mimic joint must be rejected;
the default end effector prefers mimic-free chains and warns when it cannot
avoid one.

Units: metres / radians, world frame = URDF root frame.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "ManipulaPy" / "ManipulaPy_data"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

# base -> j1 -> link1 -> j_mimic (mimics j1) -> tip: the only leaf is behind
# a mimic joint, so the default has no mimic-free candidate.
MIMIC_ONLY_URDF = """<?xml version="1.0"?>
<robot name="mimic_only">
  <link name="base_link"/><link name="link1"/><link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j_mimic" type="revolute">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
    <mimic joint="j1" multiplier="2" offset="0"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def isolated_dir(tmp_path_factory):
    """Copies outside the checkout: loading in place makes the package://
    resolver scan the whole tree per mesh URI (minutes on WSL /mnt/c)."""
    return tmp_path_factory.mktemp("urdf")


@pytest.fixture(scope="module")
def panda_path(isolated_dir):
    src = DATA / "franka_panda" / "panda.urdf"
    return Path(shutil.copy(src, isolated_dir / "panda.urdf"))


@pytest.fixture(scope="module")
def robotiq_85_path(isolated_dir):
    src = DATA / "robotiq" / "robotiq_2f_85.urdf"
    return Path(shutil.copy(src, isolated_dir / "robotiq_2f_85.urdf"))


def _load_quiet(path):
    from ManipulaPy.urdf import URDF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return URDF.load(path)


def _poe_fk(S_list: np.ndarray, M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Space-frame PoE: T(q) = exp([S1]q1)...exp([Sn]qn) M."""
    from ManipulaPy.utils import MatrixExp6, VecTose3

    T = np.eye(4)
    for i in range(S_list.shape[1]):
        T = T @ MatrixExp6(VecTose3(S_list[:, i] * q[i]))
    return T @ M


class TestExplicitMimicTip:
    def test_panda_rightfinger_raises_naming_mimic_and_driver(self, panda_path) -> None:
        robot = _load_quiet(panda_path)
        with pytest.raises(
            ValueError,
            match=r"'panda_finger_joint2' \(driven by 'panda_finger_joint1'\)",
        ):
            robot.extract_screw_axes(tip_link="panda_rightfinger")

    def test_inline_mimic_tip_raises(self) -> None:
        from ManipulaPy.urdf import URDF

        robot = URDF.from_xml_string(MIMIC_ONLY_URDF)
        with pytest.raises(ValueError, match=r"'j_mimic' \(driven by 'j1'\)"):
            robot.extract_screw_axes(tip_link="tip")


class TestMimicFreePanda:
    def test_default_is_leftfinger_with_8_columns_and_no_warning(
        self, panda_path
    ) -> None:
        from ManipulaPy.urdf import URDF

        # rightfinger is behind a mimic joint, so leftfinger is the only
        # preferred candidate: neither ambiguity nor mimic warning applies.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            robot = URDF.load(panda_path)
            params = robot.extract_screw_axes()
        assert robot.end_effector_link.name == "panda_leftfinger"
        assert list(params["joint_names"]) == ARM_JOINTS + ["panda_finger_joint1"]
        assert params["S_list"].shape == (6, 8)

    def test_leftfinger_fk_matches_native_when_finger_moves(self, panda_path) -> None:
        robot = _load_quiet(panda_path)
        params = robot.extract_screw_axes(tip_link="panda_leftfinger")
        q = np.array([0.3, -0.5, 0.2, -1.8, 0.1, 1.4, 0.7, 0.03])  # rad..., m
        cfg = dict(zip(params["joint_names"], q))
        native = robot.link_fk(cfg, use_names=True)["panda_leftfinger"]
        np.testing.assert_allclose(
            _poe_fk(params["S_list"], params["M"], q), native, atol=1e-9
        )


class TestDefaultFallsBackToMimicChain:
    def test_inline_default_warns_and_keeps_tip(self) -> None:
        from ManipulaPy.urdf import URDF

        robot = URDF.from_xml_string(MIMIC_ONLY_URDF)
        with pytest.warns(UserWarning, match=r"'j_mimic'.*holds mimic joints at 0"):
            params = robot.extract_screw_axes()
        assert robot.end_effector_link.name == "tip"
        assert list(params["joint_names"]) == ["j1"]

    def test_robotiq_85_default_unchanged_with_mimic_warning(
        self, robotiq_85_path
    ) -> None:
        from ManipulaPy.urdf import URDF

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            robot = URDF.load(robotiq_85_path)
        with pytest.warns(
            UserWarning, match=r"'left_inner_finger_joint'.*holds mimic joints at 0"
        ):
            params = robot.extract_screw_axes()
        assert robot.end_effector_link.name == "left_inner_finger_pad"
        assert list(params["joint_names"]) == ["finger_joint"]
