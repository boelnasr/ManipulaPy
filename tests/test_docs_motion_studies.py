import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "robotics_motion_studies.py"


def load_studies():
    spec = importlib.util.spec_from_file_location("robotics_motion_studies", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_panda_fixture_is_deterministic_and_cpu_only():
    studies = load_studies()
    first = studies.load_panda_fixture()
    second = studies.load_panda_fixture()

    assert first.joint_names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert first.serial is not None and first.dynamics is not None
    assert first.urdf_path == second.urdf_path
    assert len(first.joint_limits) == 7
    assert np.array_equal(studies.STUDY_TIME, np.linspace(0.0, 4.0, 61))
    assert studies.GRAVITY.tolist() == [0.0, 0.0, -9.81]
    assert studies.TOOL_WRENCH.tolist() == [0.0] * 6
    assert studies.JOINT_CLEARANCE == 0.20

    for pose in (
        studies.START,
        studies.MID,
        studies.GOAL,
        studies.NEAR_SINGULAR,
    ):
        assert pose.shape == (7,)
        assert pose.dtype == np.float64
        assert np.isfinite(pose).all()
        assert all(
            low <= q <= high
            for q, (low, high) in zip(pose, first.joint_limits)
        )

    sigma_min = np.linalg.svd(
        np.asarray(first.serial.jacobian(studies.NEAR_SINGULAR)),
        compute_uv=False,
    )[-1]
    assert sigma_min < studies.SINGULARITY_THRESHOLD == 1e-4


def test_assert_finite_names_invalid_results():
    studies = load_studies()
    with np.testing.assert_raises_regex(
        RuntimeError, "mass matrix contains non-finite values"
    ):
        studies.assert_finite("mass matrix", np.array([0.0, np.nan]))
