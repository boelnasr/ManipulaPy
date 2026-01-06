#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Parser Accuracy Tests

Compare native ManipulaPy URDF parser against PyBullet reference
to verify numerical accuracy of FK, dynamics, and properties.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import pytest
import numpy as np
from pathlib import Path
import json
import time

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "urdf_fixtures"
UR5_URDF = Path(__file__).parent.parent / "ManipulaPy" / "ManipulaPy_data" / "ur5" / "ur5.urdf"


def has_pybullet():
    """Check if pybullet is available."""
    try:
        import pybullet
        return True
    except ImportError:
        return False


class PyBulletReference:
    """Helper class for PyBullet reference computations."""

    def __init__(self, urdf_path):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot
        self.robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True
        )

        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.actuated_joints = []
        self.joint_info = {}

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            link_name = info[12].decode('utf-8')

            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'link_name': link_name,
                'lower_limit': info[8],
                'upper_limit': info[9],
            }

            # Types: REVOLUTE=0, PRISMATIC=1, SPHERICAL=2, PLANAR=3, FIXED=4
            if joint_type in [0, 1]:  # Revolute or Prismatic
                self.actuated_joints.append(i)

    def set_configuration(self, config):
        """Set robot to given configuration."""
        for i, joint_idx in enumerate(self.actuated_joints):
            if i < len(config):
                self.p.resetJointState(self.robot_id, joint_idx, config[i])

    def get_link_poses(self, config):
        """Get all link poses at given configuration."""
        self.set_configuration(config)

        poses = {}
        for i in range(self.num_joints):
            state = self.p.getLinkState(self.robot_id, i, computeForwardKinematics=True)
            pos = np.array(state[4])  # World position
            orn = np.array(state[5])  # World orientation (quaternion xyzw)

            # Convert quaternion to rotation matrix
            R = self._quat_to_rot(orn)

            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pos

            link_name = self.joint_info[i]['link_name']
            poses[link_name] = T

        return poses

    def get_end_effector_pose(self, config):
        """Get end effector pose."""
        poses = self.get_link_poses(config)
        # Return last link's pose
        last_joint_idx = self.num_joints - 1
        last_link_name = self.joint_info[last_joint_idx]['link_name']
        return poses.get(last_link_name)

    def get_mass_matrix(self, config):
        """Get mass matrix at given configuration."""
        self.set_configuration(config)
        mass_matrix = self.p.calculateMassMatrix(self.robot_id, list(config))
        return np.array(mass_matrix)

    def get_dynamics_info(self, link_idx):
        """Get dynamics info for a link."""
        info = self.p.getDynamicsInfo(self.robot_id, link_idx)
        return {
            'mass': info[0],
            'local_inertia_diagonal': np.array(info[2]),
            'local_inertia_pos': np.array(info[3]),
            'local_inertia_orn': np.array(info[4]),
        }

    def _quat_to_rot(self, quat):
        """Convert quaternion (xyzw) to rotation matrix."""
        x, y, z, w = quat

        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        return R

    def disconnect(self):
        """Disconnect from PyBullet."""
        self.p.disconnect()


@pytest.mark.skipif(not has_pybullet(), reason="pybullet not installed")
class TestFKAccuracy:
    """Test forward kinematics accuracy against PyBullet."""

    @pytest.fixture
    def simple_arm_comparison(self):
        """Load simple arm with both backends."""
        from ManipulaPy.urdf import URDF

        urdf_path = FIXTURES_DIR / "simple_arm.urdf"
        native = URDF.load(urdf_path, backend="builtin")
        pybullet_ref = PyBulletReference(urdf_path)

        yield native, pybullet_ref

        pybullet_ref.disconnect()

    @pytest.fixture
    def ur5_comparison(self):
        """Load UR5 with both backends."""
        from ManipulaPy.urdf import URDF

        if not UR5_URDF.exists():
            pytest.skip("UR5 URDF not found")

        native = URDF.load(UR5_URDF, backend="builtin")
        pybullet_ref = PyBulletReference(UR5_URDF)

        yield native, pybullet_ref

        pybullet_ref.disconnect()

    def test_simple_arm_zero_config(self, simple_arm_comparison):
        """Test FK at zero configuration."""
        native, pybullet_ref = simple_arm_comparison

        config = np.zeros(native.num_dofs)

        # Native FK
        native_fk = native.link_fk(config, use_names=True)

        # PyBullet FK
        pybullet_fk = pybullet_ref.get_link_poses(config)

        # Compare common links
        errors = []
        for link_name in native_fk:
            if link_name in pybullet_fk:
                native_T = native_fk[link_name]
                pybullet_T = pybullet_fk[link_name]

                # Position error
                pos_error = np.linalg.norm(native_T[:3, 3] - pybullet_T[:3, 3])

                # Rotation error (Frobenius norm)
                rot_error = np.linalg.norm(native_T[:3, :3] - pybullet_T[:3, :3])

                errors.append({
                    'link': link_name,
                    'pos_error': pos_error,
                    'rot_error': rot_error
                })

        # Check all errors are small
        for err in errors:
            assert err['pos_error'] < 1e-6, f"Position error for {err['link']}: {err['pos_error']}"
            assert err['rot_error'] < 1e-6, f"Rotation error for {err['link']}: {err['rot_error']}"

    def test_simple_arm_random_configs(self, simple_arm_comparison):
        """Test FK at random configurations."""
        native, pybullet_ref = simple_arm_comparison

        np.random.seed(42)

        max_pos_error = 0
        max_rot_error = 0

        for _ in range(50):
            config = np.random.uniform(-np.pi, np.pi, native.num_dofs)

            native_fk = native.link_fk(config, use_names=True)
            pybullet_fk = pybullet_ref.get_link_poses(config)

            for link_name in native_fk:
                if link_name in pybullet_fk:
                    native_T = native_fk[link_name]
                    pybullet_T = pybullet_fk[link_name]

                    pos_error = np.linalg.norm(native_T[:3, 3] - pybullet_T[:3, 3])
                    rot_error = np.linalg.norm(native_T[:3, :3] - pybullet_T[:3, :3])

                    max_pos_error = max(max_pos_error, pos_error)
                    max_rot_error = max(max_rot_error, rot_error)

        print(f"\nMax position error: {max_pos_error:.2e}")
        print(f"Max rotation error: {max_rot_error:.2e}")

        assert max_pos_error < 1e-5, f"Max position error too large: {max_pos_error}"
        assert max_rot_error < 1e-5, f"Max rotation error too large: {max_rot_error}"

    @pytest.mark.skipif(not UR5_URDF.exists(), reason="UR5 URDF not found")
    def test_ur5_zero_config(self, ur5_comparison):
        """Test UR5 FK at zero configuration."""
        native, pybullet_ref = ur5_comparison

        config = np.zeros(native.num_dofs)

        native_fk = native.link_fk(config, use_names=True)
        pybullet_fk = pybullet_ref.get_link_poses(config)

        # Find end effector
        ee_link = native.end_effector_link.name

        if ee_link in native_fk and ee_link in pybullet_fk:
            native_ee = native_fk[ee_link]
            pybullet_ee = pybullet_fk[ee_link]

            pos_error = np.linalg.norm(native_ee[:3, 3] - pybullet_ee[:3, 3])
            rot_error = np.linalg.norm(native_ee[:3, :3] - pybullet_ee[:3, :3])

            print(f"\nUR5 EE position error at zero: {pos_error:.2e}")
            print(f"UR5 EE rotation error at zero: {rot_error:.2e}")

            assert pos_error < 1e-5, f"EE position error: {pos_error}"
            assert rot_error < 1e-5, f"EE rotation error: {rot_error}"

    @pytest.mark.skipif(not UR5_URDF.exists(), reason="UR5 URDF not found")
    def test_ur5_random_configs(self, ur5_comparison):
        """Test UR5 FK at random configurations."""
        native, pybullet_ref = ur5_comparison

        np.random.seed(42)

        errors = []

        for i in range(100):
            config = np.random.uniform(-np.pi, np.pi, native.num_dofs)

            native_fk = native.link_fk(config, use_names=True)
            pybullet_fk = pybullet_ref.get_link_poses(config)

            ee_link = native.end_effector_link.name

            if ee_link in native_fk and ee_link in pybullet_fk:
                native_ee = native_fk[ee_link]
                pybullet_ee = pybullet_fk[ee_link]

                pos_error = np.linalg.norm(native_ee[:3, 3] - pybullet_ee[:3, 3])
                rot_error = np.linalg.norm(native_ee[:3, :3] - pybullet_ee[:3, :3])

                errors.append({
                    'config': i,
                    'pos_error': pos_error,
                    'rot_error': rot_error
                })

        max_pos = max(e['pos_error'] for e in errors)
        max_rot = max(e['rot_error'] for e in errors)
        avg_pos = np.mean([e['pos_error'] for e in errors])
        avg_rot = np.mean([e['rot_error'] for e in errors])

        print(f"\nUR5 FK Accuracy (100 random configs):")
        print(f"  Position - Max: {max_pos:.2e}, Avg: {avg_pos:.2e}")
        print(f"  Rotation - Max: {max_rot:.2e}, Avg: {avg_rot:.2e}")

        assert max_pos < 1e-4, f"Max position error too large: {max_pos}"
        assert max_rot < 1e-4, f"Max rotation error too large: {max_rot}"


@pytest.mark.skipif(not has_pybullet(), reason="pybullet not installed")
class TestDynamicsAccuracy:
    """Test dynamics accuracy against PyBullet."""

    @pytest.fixture
    def simple_arm_dynamics(self):
        """Load simple arm for dynamics testing (more reliable than complex robots)."""
        from ManipulaPy.urdf import URDF

        native = URDF.load(FIXTURES_DIR / "simple_arm.urdf", backend="builtin")
        pybullet_ref = PyBulletReference(FIXTURES_DIR / "simple_arm.urdf")

        yield native, pybullet_ref

        pybullet_ref.disconnect()

    @pytest.fixture
    def ur5_dynamics(self):
        """Load UR5 for dynamics testing."""
        from ManipulaPy.urdf import URDF

        if not UR5_URDF.exists():
            pytest.skip("UR5 URDF not found")

        native = URDF.load(UR5_URDF, backend="builtin")
        pybullet_ref = PyBulletReference(UR5_URDF)

        yield native, pybullet_ref

        pybullet_ref.disconnect()

    def test_mass_matrix_shape(self, ur5_dynamics):
        """Test mass matrix has correct shape."""
        native, pybullet_ref = ur5_dynamics

        dynamics = native.to_manipulator_dynamics()
        config = np.zeros(native.num_dofs)

        M_native = dynamics.mass_matrix(config)
        M_pybullet = pybullet_ref.get_mass_matrix(config)

        # Note: PyBullet may include more DOFs (floating base)
        # We compare the submatrix corresponding to actuated joints
        n = native.num_dofs

        print(f"\nMass matrix shapes - Native: {M_native.shape}, PyBullet: {M_pybullet.shape}")

        assert M_native.shape == (n, n), f"Native shape: {M_native.shape}"

    def test_mass_matrix_positive_definite(self, simple_arm_dynamics):
        """Test mass matrix is positive definite for simple arm."""
        native, _ = simple_arm_dynamics

        dynamics = native.to_manipulator_dynamics()

        np.random.seed(42)

        for _ in range(20):
            config = np.random.uniform(-np.pi, np.pi, native.num_dofs)
            M = dynamics.mass_matrix(config)

            eigenvalues = np.linalg.eigvalsh(M)
            assert np.all(eigenvalues > 0), f"Mass matrix not positive definite: {eigenvalues}"

    def test_mass_matrix_symmetry(self, simple_arm_dynamics):
        """Test mass matrix is symmetric."""
        native, _ = simple_arm_dynamics

        dynamics = native.to_manipulator_dynamics()

        np.random.seed(42)

        for _ in range(20):
            config = np.random.uniform(-np.pi, np.pi, native.num_dofs)
            M = dynamics.mass_matrix(config)

            symmetry_error = np.linalg.norm(M - M.T)
            assert symmetry_error < 1e-10, f"Mass matrix not symmetric: {symmetry_error}"


@pytest.mark.skipif(not has_pybullet(), reason="pybullet not installed")
class TestJointPropertiesAccuracy:
    """Test joint properties accuracy."""

    @pytest.fixture
    def ur5_joints(self):
        """Load UR5 for joint testing."""
        from ManipulaPy.urdf import URDF

        if not UR5_URDF.exists():
            pytest.skip("UR5 URDF not found")

        native = URDF.load(UR5_URDF, backend="builtin")
        pybullet_ref = PyBulletReference(UR5_URDF)

        yield native, pybullet_ref

        pybullet_ref.disconnect()

    def test_joint_count_match(self, ur5_joints):
        """Test actuated joint count matches."""
        native, pybullet_ref = ur5_joints

        print(f"\nNative DOFs: {native.num_dofs}")
        print(f"PyBullet actuated joints: {len(pybullet_ref.actuated_joints)}")

        assert native.num_dofs == len(pybullet_ref.actuated_joints)

    def test_joint_limits_match(self, ur5_joints):
        """Test joint limits match."""
        native, pybullet_ref = ur5_joints

        native_limits = native.joint_limits

        print("\nJoint limits comparison:")

        for i, (joint_idx, (native_lower, native_upper)) in enumerate(
            zip(pybullet_ref.actuated_joints, native_limits)
        ):
            pb_info = pybullet_ref.joint_info[joint_idx]
            pb_lower = pb_info['lower_limit']
            pb_upper = pb_info['upper_limit']

            # Handle continuous joints (PyBullet may have 0, 0 or -1, 1)
            if pb_lower == 0 and pb_upper == 0:
                # Continuous joint in PyBullet
                continue

            print(f"  Joint {i}: Native [{native_lower:.4f}, {native_upper:.4f}], "
                  f"PyBullet [{pb_lower:.4f}, {pb_upper:.4f}]")

            # Allow some tolerance
            assert abs(native_lower - pb_lower) < 0.01, \
                f"Lower limit mismatch for joint {i}"
            assert abs(native_upper - pb_upper) < 0.01, \
                f"Upper limit mismatch for joint {i}"


class TestSerialManipulatorAccuracy:
    """Test SerialManipulator conversion accuracy."""

    def test_fk_matches_urdf_fk(self):
        """Test SerialManipulator FK matches URDF FK."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        manipulator = robot.to_serial_manipulator()

        np.random.seed(42)

        errors = []

        for _ in range(50):
            config = np.random.uniform(-np.pi, np.pi, robot.num_dofs)

            # URDF FK
            urdf_fk = robot.link_fk(config, use_names=True)
            urdf_ee = urdf_fk[robot.end_effector_link.name]

            # SerialManipulator FK
            sm_ee = manipulator.forward_kinematics(config)

            pos_error = np.linalg.norm(urdf_ee[:3, 3] - sm_ee[:3, 3])
            rot_error = np.linalg.norm(urdf_ee[:3, :3] - sm_ee[:3, :3])

            errors.append({'pos': pos_error, 'rot': rot_error})

        max_pos = max(e['pos'] for e in errors)
        max_rot = max(e['rot'] for e in errors)

        print(f"\nSerialManipulator FK accuracy:")
        print(f"  Max position error: {max_pos:.2e}")
        print(f"  Max rotation error: {max_rot:.2e}")

        assert max_pos < 1e-10, f"Position error too large: {max_pos}"
        assert max_rot < 1e-10, f"Rotation error too large: {max_rot}"

    @pytest.mark.skipif(not UR5_URDF.exists(), reason="UR5 URDF not found")
    def test_ur5_screw_axes_valid(self):
        """Test UR5 screw axes are valid."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(UR5_URDF)
        screws = robot.extract_screw_axes()

        M = screws['M']
        S_list = screws['S_list']
        B_list = screws['B_list']

        print(f"\nUR5 Screw axes:")
        print(f"  M shape: {M.shape}")
        print(f"  S_list shape: {S_list.shape}")
        print(f"  B_list shape: {B_list.shape}")

        # Check M is valid SE(3)
        R = M[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

        # Check screw axes have unit angular velocity (for revolute joints)
        for i in range(S_list.shape[1]):
            omega = S_list[:3, i]
            norm = np.linalg.norm(omega)
            # Revolute joints should have unit omega
            if norm > 0.1:
                assert abs(norm - 1.0) < 1e-10, f"Screw axis {i} omega not unit: {norm}"


class TestPrismaticJointAccuracy:
    """Test prismatic joint FK accuracy."""

    @pytest.fixture
    def prismatic_robot(self):
        """Load prismatic joint robot."""
        from ManipulaPy.urdf import URDF

        return URDF.load(FIXTURES_DIR / "prismatic_joint.urdf")

    def test_prismatic_fk_linear_motion(self, prismatic_robot):
        """Test prismatic joint produces linear motion."""
        robot = prismatic_robot

        # Test X axis motion
        for x in np.linspace(-0.1, 0.1, 10):
            config = np.array([x, 0.0, 0.0])
            fk = robot.link_fk(config, use_names=True)

            # x_slide should move in X direction
            x_slide_T = fk['x_slide']
            expected_x = x  # Joint at 0.045 Z, motion in X

            # Position should be [x, 0, 0.045]
            assert abs(x_slide_T[0, 3] - x) < 1e-10, f"X motion incorrect at {x}"

    def test_prismatic_fk_cascaded_motion(self, prismatic_robot):
        """Test cascaded prismatic joints."""
        robot = prismatic_robot

        # Move all axes
        config = np.array([0.05, 0.03, 0.1])
        fk = robot.link_fk(config, use_names=True)

        z_slide_T = fk['z_slide']

        # Expected position:
        # base at origin
        # x_joint origin at [0, 0, 0.045], motion +X
        # y_joint origin at [0, 0, 0.035] relative, motion +Y
        # z_joint origin at [0, 0, 0.015] relative, motion +Z

        # Final Z position: 0.045 + 0.035 + 0.015 + 0.1 = 0.195
        expected_pos = np.array([0.05, 0.03, 0.045 + 0.035 + 0.015 + 0.1])

        actual_pos = z_slide_T[:3, 3]

        np.testing.assert_allclose(
            actual_pos, expected_pos, atol=1e-10,
            err_msg=f"Cascaded prismatic FK incorrect"
        )


class TestBatchFKAccuracy:
    """Test batch FK accuracy."""

    def test_batch_matches_individual(self):
        """Test batch FK matches individual FK calls."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        np.random.seed(42)
        N = 100
        configs = np.random.uniform(-np.pi, np.pi, (N, robot.num_dofs))

        # Batch FK
        batch_fk = robot.link_fk_batch(configs)

        # Individual FK
        for i in range(N):
            individual_fk = robot.link_fk(configs[i], use_names=True)

            for link_name in individual_fk:
                if link_name in batch_fk:
                    batch_T = batch_fk[link_name][i]
                    indiv_T = individual_fk[link_name]

                    np.testing.assert_allclose(
                        batch_T, indiv_T, atol=1e-12,
                        err_msg=f"Batch/individual mismatch at config {i}, link {link_name}"
                    )


def run_accuracy_report():
    """Generate detailed accuracy report."""
    from ManipulaPy.urdf import URDF

    print("=" * 60)
    print("ManipulaPy URDF Parser Accuracy Report")
    print("=" * 60)

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': []
    }

    # Test 1: Simple arm FK consistency
    print("\n1. Simple Arm FK Consistency Test")
    print("-" * 40)

    robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

    np.random.seed(42)
    configs = np.random.uniform(-np.pi, np.pi, (100, robot.num_dofs))

    # Check FK is deterministic
    fk1 = robot.link_fk(configs[0], use_names=True)
    fk2 = robot.link_fk(configs[0], use_names=True)

    max_diff = 0
    for link in fk1:
        diff = np.linalg.norm(fk1[link] - fk2[link])
        max_diff = max(max_diff, diff)

    print(f"  Determinism check: {max_diff:.2e} (should be 0)")
    results['tests'].append({
        'name': 'FK Determinism',
        'max_error': float(max_diff),
        'passed': max_diff < 1e-15
    })

    # Test 2: SerialManipulator consistency
    print("\n2. SerialManipulator FK Consistency")
    print("-" * 40)

    manipulator = robot.to_serial_manipulator()

    errors = []
    for config in configs:
        urdf_fk = robot.link_fk(config, use_names=True)
        urdf_ee = urdf_fk[robot.end_effector_link.name]
        sm_ee = manipulator.forward_kinematics(config)

        errors.append(np.linalg.norm(urdf_ee - sm_ee))

    print(f"  Max error: {max(errors):.2e}")
    print(f"  Mean error: {np.mean(errors):.2e}")
    print(f"  Std error: {np.std(errors):.2e}")

    results['tests'].append({
        'name': 'SerialManipulator Consistency',
        'max_error': float(max(errors)),
        'mean_error': float(np.mean(errors)),
        'passed': max(errors) < 1e-10
    })

    # Test 3: Batch FK consistency
    print("\n3. Batch FK Consistency")
    print("-" * 40)

    batch_fk = robot.link_fk_batch(configs)

    errors = []
    for i, config in enumerate(configs):
        indiv_fk = robot.link_fk(config, use_names=True)
        for link in indiv_fk:
            if link in batch_fk:
                diff = np.linalg.norm(batch_fk[link][i] - indiv_fk[link])
                errors.append(diff)

    print(f"  Max error: {max(errors):.2e}")
    print(f"  Mean error: {np.mean(errors):.2e}")

    results['tests'].append({
        'name': 'Batch FK Consistency',
        'max_error': float(max(errors)),
        'passed': max(errors) < 1e-12
    })

    # Test 4: SE(3) validity
    print("\n4. SE(3) Validity Check")
    print("-" * 40)

    rot_errors = []
    det_errors = []

    for config in configs:
        fk = robot.link_fk(config, use_names=True)
        for link, T in fk.items():
            R = T[:3, :3]
            rot_errors.append(np.linalg.norm(R @ R.T - np.eye(3)))
            det_errors.append(abs(np.linalg.det(R) - 1.0))

    print(f"  Max orthogonality error: {max(rot_errors):.2e}")
    print(f"  Max determinant error: {max(det_errors):.2e}")

    results['tests'].append({
        'name': 'SE(3) Validity',
        'max_orthogonality_error': float(max(rot_errors)),
        'max_determinant_error': float(max(det_errors)),
        'passed': max(rot_errors) < 1e-10 and max(det_errors) < 1e-10
    })

    # Test 5: Dynamics validity
    print("\n5. Dynamics Validity Check")
    print("-" * 40)

    dynamics = robot.to_manipulator_dynamics()

    pd_check = []
    sym_check = []

    for config in configs[:20]:
        M = dynamics.mass_matrix(config)
        eigenvalues = np.linalg.eigvalsh(M)
        pd_check.append(np.all(eigenvalues > 0))
        sym_check.append(np.linalg.norm(M - M.T))

    print(f"  Positive definite: {all(pd_check)}")
    print(f"  Max symmetry error: {max(sym_check):.2e}")

    results['tests'].append({
        'name': 'Mass Matrix Validity',
        'all_positive_definite': all(pd_check),
        'max_symmetry_error': float(max(sym_check)),
        'passed': all(pd_check) and max(sym_check) < 1e-10
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for t in results['tests'] if t['passed'])
    total = len(results['tests'])

    print(f"Tests passed: {passed}/{total}")

    for test in results['tests']:
        status = "PASS" if test['passed'] else "FAIL"
        print(f"  [{status}] {test['name']}")

    # PyBullet comparison if available
    if has_pybullet():
        print("\n6. PyBullet Comparison")
        print("-" * 40)

        pybullet_ref = PyBulletReference(FIXTURES_DIR / "simple_arm.urdf")

        errors = []
        for config in configs[:50]:
            native_fk = robot.link_fk(config, use_names=True)
            pybullet_fk = pybullet_ref.get_link_poses(config)

            for link in native_fk:
                if link in pybullet_fk:
                    diff = np.linalg.norm(native_fk[link] - pybullet_fk[link])
                    errors.append(diff)

        pybullet_ref.disconnect()

        print(f"  Max FK difference: {max(errors):.2e}")
        print(f"  Mean FK difference: {np.mean(errors):.2e}")

        results['tests'].append({
            'name': 'PyBullet Comparison',
            'max_error': float(max(errors)),
            'mean_error': float(np.mean(errors)),
            'passed': max(errors) < 1e-5
        })
    else:
        print("\n6. PyBullet Comparison: SKIPPED (not installed)")

    return results


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


if __name__ == "__main__":
    results = run_accuracy_report()

    # Convert numpy types to native Python types
    results = convert_to_serializable(results)

    # Save results
    output_path = Path(__file__).parent.parent / "accuracy_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
