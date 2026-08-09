import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "robotics_motion_studies.py"
MANIM = ROOT / "docs" / "manim"
EXPECTED_SCENE_KEYS = {
    "dynamics-mass",
    "dynamics-torque",
    "dynamics-roundtrip",
    "singularity-ellipsoid",
    "singularity-monitor",
    "planning-scaling",
    "planning-domains",
    "planning-collision",
    "control-comparison",
    "control-metrics",
}


def load_studies():
    spec = importlib.util.spec_from_file_location("robotics_motion_studies", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_motion_renderer():
    path = MANIM / "render_motion_studies.py"
    spec = importlib.util.spec_from_file_location("render_motion_studies", path)
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


def test_motion_registry_has_ten_unique_domain_assets():
    renderer = load_motion_renderer()
    assert set(renderer.SCENES) == EXPECTED_SCENE_KEYS
    specs = tuple(renderer.SCENES.values())
    assert len({spec.filename for spec in specs}) == 10
    assert Counter(spec.domain for spec in specs) == {
        "dynamics": 3,
        "singularity": 2,
        "path_planning": 3,
        "control": 2,
    }
    assert renderer._selected_scenes("all") == specs
    for key, scene in renderer.SCENES.items():
        assert renderer._selected_scenes(key) == (scene,)
        assert scene.scene_source.parent == MANIM


def test_motion_renderer_builds_isolated_manim_and_ffmpeg_commands(tmp_path):
    renderer = load_motion_renderer()
    scene = renderer.SCENES["dynamics-mass"]
    command = renderer._render_command(scene, tmp_path, ".mp4")
    assert command[command.index("--output_file") + 1] == scene.scene_class
    assert command[command.index("--renderer") + 1] == "cairo"
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--resolution") + 1] == "960,540"
    assert command[-2:] == [str(scene.scene_source), scene.scene_class]

    executable = Path("/usr/bin/ffmpeg")
    video = tmp_path / "scene.mp4"
    palette = tmp_path / "palette.png"
    gif = tmp_path / "scene.gif"
    assert renderer._palette_command(executable, video, palette) == [
        "/usr/bin/ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", "fps=15,palettegen=max_colors=256:stats_mode=diff", str(palette),
    ]
    assert renderer._gif_command(executable, video, palette, gif) == [
        "/usr/bin/ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-i", str(palette), "-filter_complex",
        "fps=15[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
        "-gifflags", "+transdiff", "-loop", "-1", str(gif),
    ]


def test_motion_asset_validation_rejects_invalid_pillow_images(tmp_path):
    renderer = load_motion_renderer()
    png = tmp_path / "valid.png"
    Image.new("RGB", (960, 540), (23, 33, 38)).save(png, format="PNG")
    renderer._validate_asset(png, ".png")

    first = Image.new("RGB", (960, 540), (23, 33, 38))
    second = Image.new("RGB", (960, 540), (24, 34, 39))
    gif = tmp_path / "valid.gif"
    first.save(gif, format="GIF", save_all=True, append_images=[second], duration=67)
    renderer._validate_asset(gif, ".gif")

    looping = tmp_path / "looping.gif"
    first.save(
        looping,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=67,
        loop=0,
    )
    with pytest.raises(renderer.RenderOutputError, match="loops infinitely"):
        renderer._validate_asset(looping, ".gif")

    too_long = tmp_path / "too-long.gif"
    first.save(
        too_long,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=2_500,
    )
    with pytest.raises(renderer.RenderOutputError, match="duration must be under"):
        renderer._validate_asset(too_long, ".gif")

    wrong_size = tmp_path / "wrong-size.png"
    Image.new("RGB", (320, 180)).save(wrong_size, format="PNG")
    with pytest.raises(renderer.RenderOutputError, match="invalid dimensions"):
        renderer._validate_asset(wrong_size, ".png")


def test_shared_scene_primitives_define_accessible_visual_tokens():
    source = (MANIM / "scientific_scene.py").read_text(encoding="utf-8")
    for symbol in (
        "PANEL",
        "TEAL",
        "AMBER",
        "VIOLATION",
        "panda_chain",
        "time_cursor",
        "metric_badge",
        "study_title",
        "scientific_legend",
    ):
        assert symbol in source
    assert "rate_func=linear" in source
    for forbidden in ("Flash(", "Wiggle(", "there_and_back", "random"):
        assert forbidden not in source


def test_dynamics_study_reconstructs_torque_and_round_trip():
    result = load_studies().compute_dynamics_results()
    assert result.time.shape == (61,)
    assert result.theta.shape == result.velocity.shape == result.acceleration.shape == (
        61,
        7,
    )
    assert result.mass_matrices.shape == (61, 7, 7)
    assert np.isfinite(result.mass_matrices).all()
    symmetry_error = np.max(
        np.abs(result.mass_matrices - result.mass_matrices.swapaxes(1, 2))
    )
    assert symmetry_error < 1e-8
    reconstructed = result.inertia + result.velocity_force + result.gravity + result.tool
    assert np.allclose(
        reconstructed, result.total_torque, atol=1e-8, rtol=1e-8
    )
    round_trip_error = np.max(
        np.abs(result.recovered_acceleration - result.acceleration)
    )
    assert round_trip_error < 1e-8


def test_dynamics_example_region_is_top_level_and_executable():
    source = EXAMPLE.read_text(encoding="utf-8")
    start = "# [dynamics-study-start]"
    end = "# [dynamics-study-end]"
    assert source.count(start) == source.count(end) == 1
    body = source.split(start, 1)[1].split(end, 1)[0]
    assert body.lstrip("\n") == body.lstrip("\n ")
    compile(body, f"{EXAMPLE}:dynamics-study", "exec")
    assert "mass_matrix" in body
    assert "inverse_dynamics" in body
    assert "forward_dynamics" in body


def test_dynamics_scenes_and_guide_embed_three_accessible_studies():
    scenes = (MANIM / "dynamics_scenes.py").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "source" / "user_guide" / "Dynamics.rst").read_text(
        encoding="utf-8"
    )
    classes = (
        "PandaMassMatrixEvolution",
        "PandaTorqueDecomposition",
        "PandaDynamicsRoundTrip",
    )
    stems = (
        "panda_mass_matrix_evolution",
        "panda_torque_decomposition",
        "panda_dynamics_round_trip",
    )
    for scene in classes:
        assert f"class {scene}(Scene):" in scenes
    assert "compute_dynamics_results" in scenes
    assert "rate_func=linear" in scenes
    for stem in stems:
        assert guide.count(f"{stem}.gif") == 1
        assert guide.count(f"{stem}.png") == 3
    assert guide.count("What to notice") >= 3
    assert ":start-after: # [dynamics-study-start]" in guide
    assert ":end-before: # [dynamics-study-end]" in guide
    assert guide.count('width="960" height="540"') >= 3
    assert guide.count(".. only:: html and not epub") >= 3
