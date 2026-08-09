#!/usr/bin/env python3
"""Render or validate normalized Manim assets for the kinematics tutorial."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageSequence, ImageStat, UnidentifiedImageError

MANIM_DIR = Path(__file__).resolve().parent
REPOSITORY = MANIM_DIR.parents[1]
SCENE_SOURCE = MANIM_DIR / "kinematics_scenes.py"
CONFIG = MANIM_DIR / "manim.cfg"
ASSET_DIR = REPOSITORY / "docs" / "source" / "_static" / "tutorials" / "kinematics"
EXPECTED_DIMENSIONS = (960, 540)
GIF_FRAME_RATE = 15
MAX_GIF_BYTES = 2_000_000
MAX_TOTAL_GIF_BYTES = 4_000_000
MAX_GIF_DURATION_MS = 5_000
MIN_CHANNEL_LEVELS = 32
MAX_FULL_FRAME_RMS = 8.0
MAX_BACKGROUND_RMS = 3.0
MAX_TITLE_RMS = 5.0
BACKGROUND_CROP = (800, 460, 950, 530)
TITLE_CROP = (20, 20, 940, 120)


@dataclass(frozen=True)
class SceneSpec:
    scene_class: str
    filename: str


SCENES = {
    "fk": SceneSpec("PandaForwardKinematics", "panda_forward_kinematics"),
    "jacobian": SceneSpec("PandaJacobianVelocity", "panda_jacobian_velocity"),
    "ik": SceneSpec("PandaIKConvergence", "panda_ik_convergence"),
}


class RenderOutputError(RuntimeError):
    """Raised when a render is missing, ambiguous, or invalid."""


def _selected_scenes(selection: str) -> tuple[SceneSpec, ...]:
    if selection == "all":
        return tuple(SCENES.values())
    return (SCENES[selection],)


def _find_render(root: Path, expected_name: str) -> Path:
    matches = sorted(path for path in root.rglob(expected_name) if path.is_file())
    if not matches:
        raise RenderOutputError(
            f"missing render output {expected_name!r} beneath {root}"
        )
    if len(matches) != 1:
        locations = ", ".join(str(path) for path in matches)
        raise RenderOutputError(
            f"ambiguous render output {expected_name!r}: found {len(matches)} "
            f"files ({locations})"
        )
    return matches[0]


def _validate_asset(path: Path, expected_suffix: str) -> None:
    if path.suffix != expected_suffix:
        raise RenderOutputError(
            f"invalid asset suffix for {path}: expected {expected_suffix!r}"
        )
    if not path.is_file():
        raise RenderOutputError(f"missing committed asset: {path}")
    if path.stat().st_size == 0:
        raise RenderOutputError(f"empty asset: {path}")
    try:
        with Image.open(path) as image:
            dimensions = image.size
            image_format = image.format
            frame_count = getattr(image, "n_frames", 1)
            loop = image.info.get("loop")
            duration_ms = (
                sum(
                    int(frame.info.get("duration", 0))
                    for frame in ImageSequence.Iterator(image)
                )
                if expected_suffix == ".gif"
                else 0
            )
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise RenderOutputError(f"unreadable image asset {path}: {error}") from error
    if dimensions != EXPECTED_DIMENSIONS:
        raise RenderOutputError(
            f"invalid dimensions for {path}: expected {EXPECTED_DIMENSIONS}, "
            f"found {dimensions}"
        )
    expected_format = {".gif": "GIF", ".png": "PNG"}[expected_suffix]
    if image_format != expected_format:
        raise RenderOutputError(
            f"invalid format for {path}: expected {expected_format}, "
            f"found {image_format}"
        )
    if expected_suffix == ".gif":
        if frame_count <= 1:
            raise RenderOutputError(f"GIF is not animated: {path}")
        if loop == 0:
            raise RenderOutputError(
                f"GIF loops infinitely instead of playing once: {path}"
            )
        if loop is not None:
            raise RenderOutputError(f"GIF repeats instead of playing once: {path}")
        if duration_ms >= MAX_GIF_DURATION_MS:
            raise RenderOutputError(
                f"GIF duration must be under {MAX_GIF_DURATION_MS} ms: {path} "
                f"lasts {duration_ms} ms"
            )
        if path.stat().st_size >= MAX_GIF_BYTES:
            raise RenderOutputError(
                f"GIF exceeds {MAX_GIF_BYTES}-byte delivery budget: {path} "
                f"is {path.stat().st_size} bytes"
            )


def _maximum_rms_difference(
    left: Image.Image,
    right: Image.Image,
    crop: tuple[int, int, int, int] | None = None,
) -> float:
    if crop is not None:
        left = left.crop(crop)
        right = right.crop(crop)
    difference = ImageChops.difference(left, right)
    return max(ImageStat.Stat(difference).rms)


def _validate_media_pair(gif: Path, png: Path) -> None:
    _validate_asset(gif, ".gif")
    _validate_asset(png, ".png")
    try:
        with Image.open(gif) as animated, Image.open(png) as still:
            animated.seek(animated.n_frames - 1)
            gif_final = animated.convert("RGB")
            png_final = still.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise RenderOutputError(
            f"unable to compare final GIF and PNG frames: {error}"
        ) from error

    colors = gif_final.getcolors(
        maxcolors=EXPECTED_DIMENSIONS[0] * EXPECTED_DIMENSIONS[1]
    )
    if colors is None:
        raise RenderOutputError(f"unable to inspect GIF palette fidelity: {gif}")
    palette = [color for _count, color in colors]
    channel_levels = min(
        len({color[channel] for color in palette}) for channel in range(3)
    )
    if channel_levels < MIN_CHANNEL_LEVELS:
        raise RenderOutputError(
            f"coarse GIF palette for {gif}: only {channel_levels} levels in "
            "the least-detailed channel"
        )

    fidelity_checks = (
        ("full-frame", None, MAX_FULL_FRAME_RMS),
        ("background", BACKGROUND_CROP, MAX_BACKGROUND_RMS),
        ("title/text", TITLE_CROP, MAX_TITLE_RMS),
    )
    for label, crop, maximum in fidelity_checks:
        rms = _maximum_rms_difference(gif_final, png_final, crop)
        if rms >= maximum:
            raise RenderOutputError(
                f"poor {label} GIF fidelity for {gif}: RMS {rms:.3f} "
                f"exceeds {maximum:.3f}"
            )


def _validate_total_gif_budget() -> None:
    gifs = [ASSET_DIR / f"{spec.filename}.gif" for spec in SCENES.values()]
    total = sum(path.stat().st_size for path in gifs)
    if total >= MAX_TOTAL_GIF_BYTES:
        raise RenderOutputError(
            f"kinematics GIFs exceed {MAX_TOTAL_GIF_BYTES}-byte total delivery "
            f"budget: {total} bytes"
        )


def _render_command(spec: SceneSpec, media_dir: Path, suffix: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "manim",
        "--config_file",
        str(CONFIG),
        "--media_dir",
        str(media_dir),
        "--output_file",
        spec.scene_class,
        "--renderer",
        "cairo",
        "--seed",
        "0",
        "--resolution",
        f"{EXPECTED_DIMENSIONS[0]},{EXPECTED_DIMENSIONS[1]}",
        f"--format={suffix.removeprefix('.')}",
    ]
    if suffix == ".png":
        command.append("--save_last_frame")
    command.extend((str(SCENE_SOURCE), spec.scene_class))
    return command


def _palette_command(ffmpeg: Path, video: Path, palette: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={GIF_FRAME_RATE},palettegen=max_colors=256:stats_mode=diff",
        str(palette),
    ]


def _gif_command(ffmpeg: Path, video: Path, palette: Path, gif: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(palette),
        "-filter_complex",
        f"fps={GIF_FRAME_RATE}[x];[x][1:v]paletteuse="
        "dither=sierra2_4a:diff_mode=rectangle",
        "-gifflags",
        "+transdiff",
        "-loop",
        "-1",
        str(gif),
    ]


def _ffmpeg_executable() -> Path:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RenderOutputError(
            "ffmpeg executable not found on PATH; install FFmpeg in the "
            "render-only environment"
        )
    return Path(executable)


def render_scene(spec: SceneSpec) -> None:
    with TemporaryDirectory(prefix=f"manim-{spec.filename}-") as temporary:
        media_dir = Path(temporary)
        rendered: dict[str, Path] = {}
        for suffix in (".mp4", ".png"):
            subprocess.run(
                _render_command(spec, media_dir, suffix),
                check=True,
                cwd=REPOSITORY,
            )
            source = _find_render(media_dir, f"{spec.scene_class}{suffix}")
            rendered[suffix] = source

        palette = media_dir / f"{spec.scene_class}-palette.png"
        gif = media_dir / f"{spec.scene_class}.gif"
        ffmpeg = _ffmpeg_executable()
        subprocess.run(
            _palette_command(ffmpeg, rendered[".mp4"], palette),
            check=True,
            cwd=REPOSITORY,
        )
        subprocess.run(
            _gif_command(ffmpeg, rendered[".mp4"], palette, gif),
            check=True,
            cwd=REPOSITORY,
        )
        rendered[".gif"] = gif
        _validate_media_pair(rendered[".gif"], rendered[".png"])

        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        for suffix in (".gif", ".png"):
            source = rendered[suffix]
            destination = ASSET_DIR / f"{spec.filename}{suffix}"
            shutil.copy2(source, destination)
            print(f"wrote {destination.relative_to(REPOSITORY)}")
        _validate_media_pair(
            ASSET_DIR / f"{spec.filename}.gif",
            ASSET_DIR / f"{spec.filename}.png",
        )


def check_scene(spec: SceneSpec) -> None:
    gif = ASSET_DIR / f"{spec.filename}.gif"
    png = ASSET_DIR / f"{spec.filename}.png"
    _validate_media_pair(gif, png)
    print(f"valid {gif.relative_to(REPOSITORY)}")
    print(f"valid {png.relative_to(REPOSITORY)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("all", "fk", "jacobian", "ik"),
        default="all",
        help="scene to render or validate (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate committed outputs without importing Manim",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        for spec in _selected_scenes(arguments.scene):
            if arguments.check:
                check_scene(spec)
            else:
                render_scene(spec)
        if arguments.scene == "all":
            _validate_total_gif_budget()
    except (RenderOutputError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
