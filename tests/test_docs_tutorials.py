import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "kinematics_tutorial.py"


def load_example():
    spec = importlib.util.spec_from_file_location("kinematics_tutorial", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_panda_arm_contract_and_tutorial_results():
    example = load_example()
    robot, names, limits, full_dof = example.load_panda()
    result = example.compute_tutorial_results()

    assert names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert len(limits) == example.ARM_DOF == 7
    assert full_dof == 8
    assert result.pose.shape == (4, 4)
    assert result.jacobian.shape == (6, 7)
    assert result.twist.shape == (6,)
    assert np.isfinite(result.twist).all()
    assert result.ik_success
    assert result.ik_iterations <= 20
    assert result.translation_residual < 1e-5
    assert result.rotation_residual < 1e-5
    assert all(low <= q <= high for q, (low, high) in zip(result.ik_solution, limits))
    assert np.allclose(result.pose[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(
        result.pose[:3, :3].T @ result.pose[:3, :3], np.eye(3), atol=1e-8
    )


def test_ik_trace_is_real_finite_solver_data():
    example = load_example()
    budgets, translation, rotation = example.compute_ik_trace()
    assert np.array_equal(budgets, np.arange(1, 9))
    assert translation.shape == rotation.shape == budgets.shape
    assert np.isfinite(translation).all()
    assert np.isfinite(rotation).all()
    assert translation[-1] < 1e-5
    assert rotation[-1] < 1e-5
