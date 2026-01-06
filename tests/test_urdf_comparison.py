#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
URDF Parser Comparison Tests

Compare native ManipulaPy URDF parser against urchin backend
to verify identical results.

Phase 7: Final Integration Testing

Copyright (c) 2025 Mohamed Aboelnasr
"""

import pytest
import numpy as np
from pathlib import Path
import warnings

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "urdf_fixtures"


def has_urchin():
    """Check if urchin is available."""
    try:
        import urchin
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_urchin(), reason="urchin not installed")
class TestURDFComparison:
    """Compare native parser with urchin for identical results."""

    @pytest.fixture
    def load_both(self):
        """Factory to load URDF with both backends."""
        from ManipulaPy.urdf import URDF

        def _load(urdf_path):
            native = URDF.load(urdf_path, backend="builtin")
            try:
                urchin_robot = URDF.load(urdf_path, backend="urchin")
            except Exception as e:
                pytest.skip(f"urchin failed to load: {e}")
            return native, urchin_robot

        return _load

    def test_link_count_match(self, load_both):
        """Verify link count is identical."""
        native, urchin = load_both(FIXTURES_DIR / "simple_arm.urdf")
        assert len(native.links) == len(urchin.links)

    def test_joint_count_match(self, load_both):
        """Verify joint count is identical."""
        native, urchin = load_both(FIXTURES_DIR / "simple_arm.urdf")
        assert len(native.joints) == len(urchin.joints)

    def test_num_dofs_match(self, load_both):
        """Verify DOF count is identical."""
        native, urchin = load_both(FIXTURES_DIR / "simple_arm.urdf")
        assert native.num_dofs == urchin.num_dofs

    def test_fk_matches_at_zero(self, load_both):
        """Verify FK at zero configuration matches."""
        native, urchin = load_both(FIXTURES_DIR / "simple_arm.urdf")

        cfg = np.zeros(native.num_dofs)

        fk_native = native.link_fk(cfg, use_names=True)
        fk_urchin = urchin.link_fk(cfg, use_names=True)

        for link in native.links:
            if link.name in fk_native and link.name in fk_urchin:
                np.testing.assert_allclose(
                    fk_native[link.name],
                    fk_urchin[link.name],
                    atol=1e-10,
                    err_msg=f"FK mismatch for {link.name} at zero config"
                )

    def test_fk_matches_random_configs(self, load_both):
        """Verify FK matches at random configurations."""
        native, urchin = load_both(FIXTURES_DIR / "simple_arm.urdf")

        for _ in range(20):
            cfg = np.random.uniform(-np.pi, np.pi, native.num_dofs)

            fk_native = native.link_fk(cfg, use_names=True)
            fk_urchin = urchin.link_fk(cfg, use_names=True)

            for link in native.links:
                if link.name in fk_native and link.name in fk_urchin:
                    np.testing.assert_allclose(
                        fk_native[link.name],
                        fk_urchin[link.name],
                        atol=1e-8,
                        err_msg=f"FK mismatch for {link.name} at random config"
                    )


class TestNativeParserStandalone:
    """Test native parser functionality without comparison."""

    def test_simple_arm_fk_consistency(self):
        """Test FK is internally consistent."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # FK at zero should give same result twice
        cfg = np.zeros(robot.num_dofs)
        fk1 = robot.link_fk(cfg, use_names=True)
        fk2 = robot.link_fk(cfg, use_names=True)

        for link_name in fk1:
            np.testing.assert_allclose(fk1[link_name], fk2[link_name])

    def test_serial_manipulator_conversion(self):
        """Test conversion to SerialManipulator."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Should not raise
        manipulator = robot.to_serial_manipulator()

        # Basic checks
        assert manipulator is not None
        # Check screw axes have correct shape
        assert manipulator.S_list.shape[1] == robot.num_dofs

    def test_dynamics_conversion(self):
        """Test conversion to ManipulatorDynamics."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Should not raise
        dynamics = robot.to_manipulator_dynamics()

        # Basic checks
        assert dynamics is not None

        # Mass matrix should be positive definite
        cfg = np.zeros(robot.num_dofs)
        M = dynamics.mass_matrix(cfg)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0), "Mass matrix should be positive definite"

    def test_extract_screw_axes(self):
        """Test screw axis extraction."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        screws = robot.extract_screw_axes()

        # Should have all required keys
        assert "M" in screws
        assert "S_list" in screws
        assert "B_list" in screws
        assert "G_list" in screws

        # M should be valid SE(3)
        M = screws["M"]
        assert M.shape == (4, 4)
        R = M[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_fk_matches_serial_manipulator(self):
        """Test FK results match SerialManipulator FK."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        manipulator = robot.to_serial_manipulator()

        for _ in range(10):
            cfg = np.random.uniform(-1, 1, robot.num_dofs)

            # URDF FK
            urdf_fk = robot.link_fk(cfg, use_names=True)
            urdf_ee = urdf_fk[robot.end_effector_link.name]

            # SerialManipulator FK
            sm_ee = manipulator.forward_kinematics(cfg)

            np.testing.assert_allclose(
                urdf_ee, sm_ee, atol=1e-10,
                err_msg="URDF FK doesn't match SerialManipulator FK"
            )


class TestPerformanceBenchmarks:
    """Performance benchmarks for the native parser."""

    def test_load_time_reasonable(self):
        """Test URDF loading time is reasonable."""
        import time
        from ManipulaPy.urdf import URDF

        times = []
        for _ in range(10):
            start = time.perf_counter()
            robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        # Should load in under 100ms for simple URDF
        assert avg_time < 0.1, f"Loading too slow: {avg_time*1000:.2f}ms"

    def test_fk_time_reasonable(self):
        """Test FK computation time is reasonable."""
        import time
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        cfg = np.zeros(robot.num_dofs)

        # Warmup
        for _ in range(10):
            robot.link_fk(cfg)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            robot.link_fk(cfg)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        # Should compute FK in under 1ms
        assert avg_time < 0.001, f"FK too slow: {avg_time*1e6:.2f}us"

    def test_batch_fk_faster_than_loop(self):
        """Test batch FK is faster than individual FK calls."""
        import time
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        N = 100
        cfgs = np.random.uniform(-np.pi, np.pi, (N, robot.num_dofs))

        # Individual FK
        start = time.perf_counter()
        for i in range(N):
            robot.link_fk(cfgs[i])
        individual_time = time.perf_counter() - start

        # Batch FK
        start = time.perf_counter()
        robot.link_fk_batch(cfgs)
        batch_time = time.perf_counter() - start

        # Batch should be at least as fast (ideally faster)
        # Allow some tolerance since batch has overhead
        assert batch_time < individual_time * 2, \
            f"Batch FK ({batch_time*1000:.2f}ms) should be faster than " \
            f"individual ({individual_time*1000:.2f}ms)"


class TestRobustness:
    """Test robustness to various input scenarios."""

    def test_empty_configuration(self):
        """Test handling of empty/zero configuration."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Zero config
        cfg = np.zeros(robot.num_dofs)
        fk = robot.link_fk(cfg)
        assert len(fk) > 0

    def test_large_configuration_values(self):
        """Test handling of large joint values."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")

        # Large values (beyond limits)
        cfg = np.ones(robot.num_dofs) * 100.0
        fk = robot.link_fk(cfg)  # Should not crash

        # Results should still be valid SE(3)
        for link, T in fk.items():
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_repeated_fk_calls(self):
        """Test FK is consistent across repeated calls."""
        from ManipulaPy.urdf import URDF

        robot = URDF.load(FIXTURES_DIR / "simple_arm.urdf")
        cfg = np.array([0.5, 0.3])

        # Call FK many times
        results = [robot.link_fk(cfg, use_names=True) for _ in range(10)]

        # All should be identical
        for result in results[1:]:
            for link_name in results[0]:
                np.testing.assert_allclose(
                    results[0][link_name],
                    result[link_name],
                    err_msg="FK results inconsistent across calls"
                )


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
