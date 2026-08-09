import configparser
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageSequence, ImageStat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "kinematics_tutorial.py"
TUTORIALS = ROOT / "docs" / "source" / "tutorials"
MANIM = ROOT / "docs" / "manim"
ASSETS = ROOT / "docs" / "source" / "_static" / "tutorials" / "kinematics"
MAX_GIF_BYTES = 2_000_000
MAX_TOTAL_GIF_BYTES = 4_000_000


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


def load_renderer():
    path = MANIM / "render_kinematics.py"
    spec = importlib.util.spec_from_file_location("render_kinematics", path)
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
    assert "_solve_to_target" not in regions["validation"]
    assert re.search(r"# \[[^\]]+-start\]", regions["validation"]) is None


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


def test_committed_kinematics_media_pairs_are_valid():
    stems = (
        "panda_forward_kinematics",
        "panda_jacobian_velocity",
        "panda_ik_convergence",
    )
    total_gif_bytes = 0
    for stem in stems:
        gif = ASSETS / f"{stem}.gif"
        png = ASSETS / f"{stem}.png"
        assert gif.stat().st_size > 10_000
        assert gif.stat().st_size < MAX_GIF_BYTES
        total_gif_bytes += gif.stat().st_size
        assert png.stat().st_size > 10_000
        with Image.open(gif) as animated:
            assert animated.size == (960, 540)
            assert getattr(animated, "n_frames", 1) > 1
        with Image.open(png) as still:
            assert still.size == (960, 540)
            assert still.format == "PNG"
    assert total_gif_bytes < MAX_TOTAL_GIF_BYTES


def test_committed_kinematics_gifs_play_once_within_five_seconds():
    for gif in sorted(ASSETS.glob("*.gif")):
        with Image.open(gif) as animated:
            assert "loop" not in animated.info, f"{gif.name} contains a loop extension"
            duration_ms = sum(
                int(frame.info.get("duration", 0))
                for frame in ImageSequence.Iterator(animated)
            )
        assert duration_ms < 5_000, f"{gif.name} lasts {duration_ms / 1_000:.2f}s"


def test_kinematics_tutorial_media_embedding_is_accessible():
    source = read(TUTORIALS / "kinematics_guide.rst")
    studies = (
        (
            "panda_forward_kinematics",
            "Seven Panda arm joints build a base-to-tool transform and reveal "
            "the resulting tool frame.",
            "Base-to-tool transform and resulting tool frame.",
        ),
        (
            "panda_jacobian_velocity",
            "Seven joint rates pass through a six-by-seven Jacobian into angular "
            "and linear tool velocity.",
            "Joint rates mapped to angular and linear tool velocity.",
        ),
        (
            "panda_ik_convergence",
            "Translation and rotation residuals converge as inverse kinematics "
            "approaches a reachable Panda pose.",
            "Translation and rotation residuals converge to a reachable pose.",
        ),
    )

    for stem, alt_text, caption in studies:
        assert (
            f'<source media="(prefers-reduced-motion: reduce)" '
            f'srcset="../_static/tutorials/kinematics/{stem}.png">' in source
        )
        assert (
            f'<img src="../_static/tutorials/kinematics/{stem}.gif" '
            'width="960" height="540"' in source
        )
        assert (
            source.count(f".. figure:: ../_static/tutorials/kinematics/{stem}.png")
            == 2
        )
        assert source.count(alt_text) == 3
        assert source.count(caption) == 3

    assert ".. only:: html and not epub" in source
    assert ".. only:: epub" in source
    assert ".. only:: latex" in source
    for image in re.findall(r"<img\b[^>]*>", source):
        assert re.search(r'\balt="[^"\n]+"', image)
        assert 'loading="lazy"' in image


def test_committed_kinematics_gifs_preserve_palette_and_final_frame_fidelity():
    stems = (
        "panda_forward_kinematics",
        "panda_jacobian_velocity",
        "panda_ik_convergence",
    )
    for stem in stems:
        with (
            Image.open(ASSETS / f"{stem}.gif") as animated,
            Image.open(ASSETS / f"{stem}.png") as still,
        ):
            animated.seek(animated.n_frames - 1)
            gif_final = animated.convert("RGB")
            png_final = still.convert("RGB")

        colors = gif_final.getcolors(maxcolors=960 * 540)
        assert colors is not None
        palette = [color for _count, color in colors]
        channel_levels = min(
            len({color[channel] for color in palette}) for channel in range(3)
        )
        assert channel_levels >= 32

        full_difference = ImageChops.difference(gif_final, png_final)
        assert max(ImageStat.Stat(full_difference).rms) < 8.0

        background_difference = ImageChops.difference(
            gif_final.crop((800, 460, 950, 530)),
            png_final.crop((800, 460, 950, 530)),
        )
        assert max(ImageStat.Stat(background_difference).rms) < 3.0

        title_difference = ImageChops.difference(
            gif_final.crop((20, 20, 940, 120)),
            png_final.crop((20, 20, 940, 120)),
        )
        assert max(ImageStat.Stat(title_difference).rms) < 5.0


def test_manim_tool_frame_marks_all_three_axes():
    scenes = read(MANIM / "kinematics_scenes.py")
    triad = scenes.split("triad = VGroup(", maxsplit=1)[1].split(
        "return VGroup(links", maxsplit=1
    )[0]
    assert triad.count("Arrow(") == 3
    for axis in ("x", "y", "z"):
        assert f'MathTex("{axis}"' in triad


def test_manim_rules_and_major_content_are_frame_centered():
    scenes = read(MANIM / "kinematics_scenes.py")
    assert scenes.count("rule = _rule_below(title)") == 3
    assert "rule.set_x(0.0)" in scenes
    assert "equation.set_x(0.0)" in scenes
    assert "charts.set_x(0.0)" in scenes


def test_manim_ik_plot_uses_explicit_log10_scientific_scale():
    scenes = read(MANIM / "kinematics_scenes.py")
    assert "RESIDUAL_DISPLAY_FLOOR = 1e-9" in scenes
    assert "RESIDUAL_TOLERANCE = 1e-5" in scenes
    assert "TOLERANCE_LOG10 = -5.0" in scenes
    assert "np.log10(np.maximum(residuals, RESIDUAL_DISPLAY_FLOOR))" in scenes
    assert "RESIDUAL_DECADES = (0, -3, -5, -7, -9)" in scenes
    assert 'rf"10^{{{decade}}}"' in scenes
    assert "y_range=[-9.0, 0.0, 1.0]" in scenes


def test_manim_render_command_and_config_isolate_user_settings(tmp_path):
    renderer = load_renderer()
    command = renderer._render_command(renderer.SCENES["fk"], tmp_path, ".mp4")
    assert command[command.index("--output_file") + 1] == "PandaForwardKinematics"
    assert command[command.index("--renderer") + 1] == "cairo"
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--resolution") + 1] == "960,540"

    config = configparser.ConfigParser()
    config.read(MANIM / "manim.cfg")
    assert config["CLI"].getboolean("transparent") is False
    assert config["CLI"].getfloat("background_opacity") == 1.0
    assert config["CLI"].getfloat("frame_width") == pytest.approx(128.0 / 9.0)
    assert config["CLI"].getfloat("frame_height") == 8.0


def test_manim_renderer_builds_argument_list_ffmpeg_palette_pipeline(tmp_path):
    renderer = load_renderer()
    executable = Path("/usr/bin/ffmpeg")
    video = tmp_path / "scene.mp4"
    palette = tmp_path / "palette.png"
    gif = tmp_path / "scene.gif"

    palette_command = renderer._palette_command(executable, video, palette)
    assert palette_command == [
        "/usr/bin/ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=15,palettegen=max_colors=256:stats_mode=diff",
        str(palette),
    ]

    gif_command = renderer._gif_command(executable, video, palette, gif)
    assert gif_command == [
        "/usr/bin/ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(palette),
        "-filter_complex",
        "fps=15[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
        "-gifflags",
        "+transdiff",
        "-loop",
        "-1",
        str(gif),
    ]


def test_manim_asset_validation_uses_real_pillow_images(tmp_path):
    renderer = load_renderer()
    png = tmp_path / "valid.png"
    Image.new("RGB", (960, 540), (23, 33, 38)).save(png, format="PNG")
    renderer._validate_asset(png, ".png")

    gif = tmp_path / "valid.gif"
    first = Image.new("RGB", (960, 540), (23, 33, 38))
    second = Image.new("RGB", (960, 540), (24, 34, 39))
    first.save(
        gif,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=67,
    )
    renderer._validate_asset(gif, ".gif")

    looping_gif = tmp_path / "infinite.gif"
    first.save(
        looping_gif,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=67,
        loop=0,
    )
    with pytest.raises(renderer.RenderOutputError, match="loops infinitely"):
        renderer._validate_asset(looping_gif, ".gif")

    wrong_size = tmp_path / "wrong-size.png"
    Image.new("RGB", (320, 180)).save(wrong_size)
    with pytest.raises(renderer.RenderOutputError, match="invalid dimensions"):
        renderer._validate_asset(wrong_size, ".png")
