import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "kinematics_tutorial.py"
TUTORIALS = ROOT / "docs" / "source" / "tutorials"
MANIM = ROOT / "docs" / "manim"
ASSETS = ROOT / "docs" / "source" / "_static" / "tutorials" / "kinematics"


def read(path):
    return path.read_text(encoding="utf-8")


def marker_body(source, marker):
    start = f"# [{marker}-start]"
    end = f"# [{marker}-end]"
    return source.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


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


def test_one_canonical_kinematics_tutorial_route():
    index = read(TUTORIALS / "index.rst")
    canonical = read(TUTORIALS / "kinematics_guide.rst")
    legacy = read(TUTORIALS / "Kinematics.rst")
    assert "   kinematics_guide\n" in index
    assert "   Kinematics\n" not in index
    assert "Kinematics with the Franka Panda" in canonical
    assert "This tutorial has moved" in legacy
    assert ":doc:`kinematics_guide`" in legacy
    assert "code-block:: python" not in legacy


def test_tutorial_uses_tested_regions_and_current_api():
    source = read(TUTORIALS / "kinematics_guide.rst")
    for marker in (
        "load-panda",
        "forward-kinematics",
        "velocity-kinematics",
        "inverse-kinematics",
        "validation",
    ):
        assert f":start-after: # [{marker}-start]" in source
        assert f":end-before: # [{marker}-end]" in source
    for forbidden in (
        "batch_forward_kinematics",
        "sample_workspace",
        "lm_inverse_kinematics",
        "position_inverse_kinematics",
        "success, sol",
        "Python ≥\u00a03.8",
    ):
        assert forbidden not in source
    assert "\u00a0" not in source


def test_literalinclude_regions_are_unindented_executable_tutorial_units():
    source = read(EXAMPLE)
    regions = {
        marker: marker_body(source, marker)
        for marker in (
            "load-panda",
            "forward-kinematics",
            "velocity-kinematics",
            "inverse-kinematics",
            "validation",
        )
    }

    for marker, body in regions.items():
        assert body.lstrip("\n") == body.lstrip("\n ")
        compile(body, f"{EXAMPLE}:{marker}", "exec")

    assert "import numpy as np" in regions["load-panda"]
    assert "HOME = np.array" in regions["load-panda"]
    assert "TARGET = np.array" in regions["load-panda"]
    assert "JOINT_RATES = np.array" in regions["load-panda"]
    assert "def load_panda" in regions["load-panda"]
    assert "def forward_kinematics_step(robot, home, target)" in regions[
        "forward-kinematics"
    ]
    assert "def velocity_kinematics_step(robot, configuration, joint_rates)" in regions[
        "velocity-kinematics"
    ]
    assert "def inverse_kinematics_step(robot, target_pose, initial_guess)" in regions[
        "inverse-kinematics"
    ]
    assert "def validation_step(robot, solution, target_pose)" in regions["validation"]


def test_literalinclude_regions_execute_as_the_displayed_tutorial():
    source = read(EXAMPLE)
    namespace = {"__name__": "kinematics_tutorial_snippets"}
    for marker in (
        "load-panda",
        "forward-kinematics",
        "velocity-kinematics",
        "inverse-kinematics",
        "validation",
    ):
        body = marker_body(source, marker)
        exec(compile(body, f"{EXAMPLE}:{marker}", "exec"), namespace)

    robot, _names, _limits, _full_dof = namespace["load_panda"]()
    pose, target_pose = namespace["forward_kinematics_step"](
        robot, namespace["HOME"], namespace["TARGET"]
    )
    jacobian, twist, _singular_values = namespace["velocity_kinematics_step"](
        robot, namespace["HOME"], namespace["JOINT_RATES"]
    )
    solution, success, iterations = namespace["inverse_kinematics_step"](
        robot, target_pose, namespace["HOME"]
    )
    translation, rotation = namespace["validation_step"](
        robot, solution, target_pose
    )

    assert pose.shape == (4, 4)
    assert jacobian.shape == (6, 7)
    assert twist.shape == (6,)
    assert success
    assert iterations <= 20
    assert translation < 1e-5
    assert rotation < 1e-5


def test_manim_pipeline_is_render_only_and_pinned():
    scenes = read(MANIM / "kinematics_scenes.py")
    renderer = read(MANIM / "render_kinematics.py")
    requirements = read(MANIM / "requirements.txt")
    assert requirements.strip() == "manim==0.20.1"
    for scene in (
        "PandaForwardKinematics",
        "PandaJacobianVelocity",
        "PandaIKConvergence",
    ):
        assert f"class {scene}" in scenes
        assert scene in renderer
    assert "compute_tutorial_results" in scenes
    assert "compute_ik_trace" in scenes
    assert "docs/requirements.txt" not in renderer


def test_manim_config_has_stable_scientific_output():
    config = read(MANIM / "manim.cfg")
    for contract in ("pixel_width = 960", "pixel_height = 540", "frame_rate = 30"):
        assert contract in config
