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

from PIL import Image, UnidentifiedImageError

MANIM_DIR = Path(__file__).resolve().parent
REPOSITORY = MANIM_DIR.parents[1]
SCENE_SOURCE = MANIM_DIR / "kinematics_scenes.py"
CONFIG = MANIM_DIR / "manim.cfg"
ASSET_DIR = REPOSITORY / "docs" / "source" / "_static" / "tutorials" / "kinematics"
EXPECTED_DIMENSIONS = (960, 540)


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
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise RenderOutputError(f"unreadable image asset {path}: {error}") from error
    if dimensions != EXPECTED_DIMENSIONS:
        raise RenderOutputError(
            f"invalid dimensions for {path}: expected {EXPECTED_DIMENSIONS}, "
            f"found {dimensions}"
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


def render_scene(spec: SceneSpec) -> None:
    with TemporaryDirectory(prefix=f"manim-{spec.filename}-") as temporary:
        media_dir = Path(temporary)
        rendered: dict[str, Path] = {}
        for suffix in (".gif", ".png"):
            subprocess.run(
                _render_command(spec, media_dir, suffix),
                check=True,
                cwd=REPOSITORY,
            )
            source = _find_render(media_dir, f"{spec.scene_class}{suffix}")
            _validate_asset(source, suffix)
            rendered[suffix] = source

        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        for suffix, source in rendered.items():
            destination = ASSET_DIR / f"{spec.filename}{suffix}"
            shutil.copy2(source, destination)
            _validate_asset(destination, suffix)
            print(f"wrote {destination.relative_to(REPOSITORY)}")


def check_scene(spec: SceneSpec) -> None:
    for suffix in (".gif", ".png"):
        asset = ASSET_DIR / f"{spec.filename}{suffix}"
        _validate_asset(asset, suffix)
        print(f"valid {asset.relative_to(REPOSITORY)}")


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
    except (RenderOutputError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
