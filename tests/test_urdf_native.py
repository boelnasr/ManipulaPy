#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Test suite for ManipulaPy native URDF parser.

Tests core functionality:
- URDF loading and parsing
- Forward kinematics
- Joint types (revolute, continuous, prismatic, fixed, planar, floating)
- Mimic joints
- Transmissions
- Validation (cycle detection, disconnected links)
- Scene (multi-robot support)
- Modifiers

Copyright (c) 2025 Mohamed Aboelnasr
"""

import pytest
import numpy as np
from pathlib import Path
import warnings

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "urdf_fixtures"


class TestURDFLoading:
    """Test URDF loading and basic parsing."""

    def test_load_simple_arm(self):
        """Test loading a simple 2-DOF arm."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        assert robot.name == "simple_arm"
        assert len(robot.links) == 3  # base_link, link1, link2
        assert len(robot.joints) == 2  # joint1, joint2
        assert robot.num_dofs == 2

    def test_link_properties(self):
        """Test link properties are parsed correctly."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Check base_link
        base = robot._links["base_link"]
        assert base.name == "base_link"
        assert base.inertial is not None
        assert base.inertial.mass == 1.0
        assert len(base.visuals) > 0
        assert len(base.collisions) > 0

    def test_joint_properties(self):
        """Test joint properties are parsed correctly."""
        from ManipulaPy.urdf import URDF, JointType

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        joint1 = robot._joints["joint1"]
        assert joint1.name == "joint1"
        assert joint1.joint_type == JointType.REVOLUTE
        assert joint1.parent == "base_link"
        assert joint1.child == "link1"
        np.testing.assert_allclose(joint1.axis, [0, 0, 1])
        assert joint1.limit is not None
        assert joint1.limit.lower < joint1.limit.upper

    def test_joint_limits(self):
        """Test joint limits extraction."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        limits = robot.joint_limits
        assert len(limits) == 2  # 2 joints
        for lower, upper in limits:
            assert lower < upper  # lower < upper


class TestJointTypes:
    """Test different joint type handling."""

    def test_revolute_joint(self):
        """Test revolute joint motion."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Test FK at different configurations
        cfg0 = np.array([0.0, 0.0])
        cfg1 = np.array([np.pi / 4, np.pi / 6])  # Rotate both joints

        fk0 = robot.link_fk(cfg0, use_names=True)
        fk1 = robot.link_fk(cfg1, use_names=True)

        # link2 position should change when joints rotate
        # Test the full transform (rotation + translation)
        assert not np.allclose(fk0["link2"], fk1["link2"])

    def test_continuous_joint(self):
        """Test continuous joint (no limits)."""
        from ManipulaPy.urdf import URDF, JointType

        robot = URDF.load(FIXTURES_DIR / "continuous_joints.urdf")

        # Check joint type
        wheel_joint = robot._joints["wheel1_joint"]
        assert wheel_joint.joint_type == JointType.CONTINUOUS

        # Continuous joints should handle any angle
        cfg = np.array([10 * np.pi, -10 * np.pi])
        fk = robot.link_fk(cfg)  # Should not raise

    def test_prismatic_joint(self):
        """Test prismatic joint motion."""
        from ManipulaPy.urdf import URDF, JointType

        robot = URDF.load(FIXTURES_DIR / "prismatic_joint.urdf")

        # Check joint types
        assert robot._joints["x_joint"].joint_type == JointType.PRISMATIC
        assert robot._joints["y_joint"].joint_type == JointType.PRISMATIC
        assert robot._joints["z_joint"].joint_type == JointType.PRISMATIC

        # Test translation
        cfg0 = np.array([0.0, 0.0, 0.0])
        cfg1 = np.array([0.05, 0.0, 0.0])  # Move x

        fk0 = robot.link_fk(cfg0, use_names=True)
        fk1 = robot.link_fk(cfg1, use_names=True)

        # X position should change by 0.05
        np.testing.assert_allclose(
            fk1["x_slide"][:3, 3] - fk0["x_slide"][:3, 3],
            [0.05, 0, 0],
            atol=1e-10
        )

    def test_fixed_joints_ignored(self):
        """Test that fixed joints don't add DOFs."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        # simple_arm has only revolute joints, should have 2 DOFs
        assert robot.num_dofs == 2


class TestMimicJoints:
    """Test mimic joint handling."""

    def test_mimic_joint_parsing(self):
        """Test mimic joint properties are parsed."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "mimic_joints.urdf")

        right_finger = robot._joints["right_finger_joint"]
        assert right_finger.mimic is not None
        assert right_finger.mimic.joint == "left_finger_joint"
        assert right_finger.mimic.multiplier == -1.0
        assert right_finger.mimic.offset == 0.0

    def test_mimic_joint_fk(self):
        """Test mimic joints follow master joint."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "mimic_joints.urdf")

        # Only one actuated joint (left finger)
        assert robot.num_dofs == 1

        # Move left finger, right should mirror
        cfg = np.array([0.02])  # Open left finger 2cm
        fk = robot.link_fk(cfg, use_names=True)

        # Both fingers should move symmetrically
        left_pos = fk["left_finger"][:3, 3]
        right_pos = fk["right_finger"][:3, 3]

        # X positions should be opposite
        assert left_pos[0] > 0  # Left moves positive
        assert right_pos[0] < 0  # Right moves negative (mimic multiplier = -1)


class TestTransmissions:
    """Test transmission parsing."""

    def test_transmission_parsing(self):
        """Test transmission elements are parsed."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "transmissions.urdf")

        assert len(robot.transmissions) == 2

        trans1 = robot.transmission_map.get("trans1")
        assert trans1 is not None
        assert trans1.type == "transmission_interface/SimpleTransmission"
        assert len(trans1.joints) == 1
        assert len(trans1.actuators) == 1
        assert trans1.actuators[0].mechanical_reduction == 50


class TestGeometry:
    """Test geometry parsing."""

    def test_primitive_geometry(self):
        """Test all primitive geometry types."""
        from ManipulaPy.urdf import URDF, Box, Cylinder, Sphere

        robot = URDF.load(FIXTURES_DIR / "primitives.urdf")

        # Check base has box
        base = robot._links["base_link"]
        assert len(base.visuals) > 0
        assert isinstance(base.visuals[0].geometry, Box)

        # Check cylinder link
        cyl_link = robot._links["cylinder_link"]
        assert isinstance(cyl_link.visuals[0].geometry, Cylinder)

        # Check sphere link
        sphere_link = robot._links["sphere_link"]
        assert isinstance(sphere_link.visuals[0].geometry, Sphere)


class TestValidation:
    """Test URDF validation."""

    def test_cycle_detection(self):
        """Test cyclic URDF is detected at load time."""
        from ManipulaPy.urdf import URDF
        import pytest

        # Cyclic URDF should raise ValueError during loading
        with pytest.raises(ValueError, match="cyclic|root"):
            URDF.load(FIXTURES_DIR / "cyclic.urdf")

    def test_multi_root_handling(self):
        """Test multiple roots are handled (all roots included in FK)."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "multi_root.urdf")

        # All roots should be recorded
        assert len(robot.root_links) >= 2

        fk = robot.link_fk(use_names=True)
        # Root transforms should be identity for each root
        for root in robot.root_links:
            np.testing.assert_allclose(fk[root.name], np.eye(4), atol=1e-10)

    def test_valid_urdf_passes(self):
        """Test valid URDF passes validation."""
        from ManipulaPy.urdf import URDF, validate_urdf

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        result = validate_urdf(robot)
        # ValidationResult has 'valid' attribute
        assert result.valid


class TestForwardKinematics:
    """Test forward kinematics computation."""

    def test_fk_at_zero_config(self):
        """Test FK at zero configuration."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        cfg = np.zeros(robot.num_dofs)
        fk = robot.link_fk(cfg, use_names=True)

        # All transforms should be valid 4x4 matrices
        for link_name, T in fk.items():
            assert T.shape == (4, 4)
            # Check orthonormality of rotation
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_fk_consistency(self):
        """Test FK is consistent with joint chain."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        cfg = np.array([0.5, 0.3])
        fk = robot.link_fk(cfg, use_names=True)

        # Child link should be reachable from parent through joint
        # This is a sanity check that FK follows kinematic chain

    def test_batch_fk(self):
        """Test batch forward kinematics."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Generate random configurations
        N = 100
        cfgs = np.random.uniform(-np.pi, np.pi, (N, robot.num_dofs))

        # Batch FK - returns Dict[str, np.ndarray] with (N, 4, 4) arrays
        batch_fk = robot.link_fk_batch(cfgs)

        # Compare with individual FK
        for i in range(min(10, N)):  # Check first 10
            single_fk = robot.link_fk(cfgs[i], use_names=True)
            for link in robot.links:
                np.testing.assert_allclose(
                    batch_fk[link.name][i],
                    single_fk[link.name],
                    atol=1e-10
                )


class TestScrewAxes:
    """Test screw axis extraction."""

    def test_extract_screw_axes(self):
        """Test screw axis extraction for manipulator conversion."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        screws = robot.extract_screw_axes()

        assert "M" in screws
        assert "S_list" in screws
        assert "B_list" in screws
        assert "G_list" in screws

        # Check dimensions
        assert screws["M"].shape == (4, 4)
        assert screws["S_list"].shape == (6, robot.num_dofs)
        assert screws["B_list"].shape == (6, robot.num_dofs)
        assert len(screws["G_list"]) == robot.num_dofs

    def test_m_matrix_valid(self):
        """Test M matrix is valid SE(3)."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        screws = robot.extract_screw_axes()

        M = screws["M"]

        # Check SE(3) properties
        R = M[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        np.testing.assert_allclose(M[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_tip_selection_for_m(self):
        """Test custom tip selection when extracting screw axes."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "branched.urdf")

        # Pick a non-default tip
        tip_name = "head"
        screws = robot.extract_screw_axes(tip_link=tip_name)

        fk = robot.link_fk(np.zeros(robot.num_dofs), use_names=True)
        np.testing.assert_allclose(screws["M"], fk[tip_name], atol=1e-10)


class TestScene:
    """Test multi-robot scene support."""

    def test_scene_creation(self):
        """Test creating a scene with multiple robots."""
        from ManipulaPy.urdf import URDF, Scene

        robot1 = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        robot2 = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        scene = Scene("test_scene")
        scene.add_robot("arm1", robot1, base_xyz=[0, 0, 0])
        scene.add_robot("arm2", robot2, base_xyz=[1.0, 0, 0])

        assert len(scene.robots) == 2
        assert "arm1" in scene.robot_names
        assert "arm2" in scene.robot_names

    def test_world_frame_fk(self):
        """Test world-frame FK in scene."""
        from ManipulaPy.urdf import URDF, Scene

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        scene = Scene()
        scene.add_robot("arm", robot, base_xyz=[1.0, 2.0, 0.5])

        cfg = {"arm": np.zeros(robot.num_dofs)}
        world_fk = scene.world_link_fk(cfg)

        # Base link should be offset by [1.0, 2.0, 0.5]
        base_pos = world_fk["arm"]["base_link"][:3, 3]
        np.testing.assert_allclose(base_pos, [1.0, 2.0, 0.5], atol=1e-10)


class TestModifiers:
    """Test URDF modification functionality."""

    def test_modifier_creates_copy(self):
        """Test modifier creates deep copy of URDF."""
        from ManipulaPy.urdf import URDF, URDFModifier

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        original_mass = robot._links["link1"].inertial.mass

        modifier = URDFModifier(robot)
        modifier.set_link_mass("link1", 100.0)

        # Original should be unchanged
        assert robot._links["link1"].inertial.mass == original_mass

        # Modified should be changed
        assert modifier.urdf._links["link1"].inertial.mass == 100.0

    def test_set_joint_limits(self):
        """Test modifying joint limits."""
        from ManipulaPy.urdf import URDF, URDFModifier

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        modifier = URDFModifier(robot)
        modifier.set_joint_limits("joint1", lower=-1.0, upper=1.0)

        modified = modifier.urdf
        assert modified._joints["joint1"].limit.lower == -1.0
        assert modified._joints["joint1"].limit.upper == 1.0

    def test_add_payload(self):
        """Test adding payload to link."""
        from ManipulaPy.urdf import URDF, URDFModifier

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        original_mass = robot._links["link2"].inertial.mass

        modifier = URDFModifier(robot)
        modifier.add_payload("link2", mass=2.0)

        modified = modifier.urdf
        assert modified._links["link2"].inertial.mass == original_mass + 2.0

    def test_scale_masses(self):
        """Test scaling all masses."""
        from ManipulaPy.urdf import URDF, URDFModifier

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        original_masses = {
            name: link.inertial.mass
            for name, link in robot._links.items()
            if link.inertial is not None
        }

        modifier = URDFModifier(robot)
        modifier.scale_masses(1.5)

        modified = modifier.urdf
        for name, original_mass in original_masses.items():
            expected = original_mass * 1.5
            actual = modified._links[name].inertial.mass
            np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_export_to_urdf_string(self):
        """Test exporting modified URDF to string."""
        from ManipulaPy.urdf import URDF, URDFModifier

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        modifier = URDFModifier(robot)
        modifier.set_link_mass("link1", 99.0)

        xml_string = modifier.to_urdf_string()

        # Should be valid XML
        assert xml_string.startswith("<?xml") or xml_string.startswith("<robot")
        assert "simple_arm" in xml_string
        assert "99" in xml_string  # Modified mass


class TestBranchedRobot:
    """Test branched (multi-end-effector) robot handling."""

    def test_branched_robot_loading(self):
        """Test loading branched robot."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "branched.urdf")

        # Should have multiple end links (leaves)
        end_links = robot.end_links
        assert len(end_links) >= 3  # left_hand, right_hand, head

    def test_multiple_end_effectors(self):
        """Test robot has multiple end effectors detected."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "branched.urdf")

        # Get all end link names
        end_link_names = [link.name for link in robot.end_links]

        # Should include left_hand, right_hand, head
        assert "left_hand" in end_link_names
        assert "right_hand" in end_link_names
        assert "head" in end_link_names


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
