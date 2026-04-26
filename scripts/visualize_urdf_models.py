#!/usr/bin/env python3
"""
Visual inspection helper for URDF models in ManipulaPy_data.

Uses the URDF trimesh viewer (if available) to show each model and
lets you step through them interactively.
"""

import argparse
from pathlib import Path
import sys

from ManipulaPy.urdf import URDF


def _collect_urdfs(data_root: Path, name_filter: str) -> list[Path]:
    urdfs = sorted(data_root.rglob("crx20ia_l.urdf"))  # Example specific model
    if name_filter:
        name_filter = name_filter.lower()
        urdfs = [p for p in urdfs if name_filter in str(p).lower()]
    return urdfs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step through URDF models and open a viewer for each."
    )
    parser.add_argument(
        "--data-root",
        default="ManipulaPy/ManipulaPy_data",
        help="Root directory containing URDF models",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Substring filter for URDF paths (case-insensitive)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) for resuming",
    )
    parser.add_argument(
        "--collision",
        action="store_true",
        help="Show collision geometry instead of visuals",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching URDFs and exit",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not pause between models",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        return 1

    # Ensure trimesh backend is available
    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("trimesh is required for visualization.")
        print("Install with: pip install trimesh pyglet")
        return 1

    urdfs = _collect_urdfs(data_root, args.filter)
    if not urdfs:
        print("No URDFs found.")
        return 0

    if args.list:
        for i, p in enumerate(urdfs):
            print(f"[{i}] {p}")
        return 0

    total = len(urdfs)
    for idx, path in enumerate(urdfs):
        if idx < args.start:
            continue

        print(f"[{idx + 1}/{total}] {path}")
        try:
            robot = URDF.load(path, load_meshes=True)
            robot.show(use_collision=args.collision)
        except Exception as exc:
            print(f"  Error: {exc}")

        if args.no_wait:
            continue

        try:
            resp = input("Press Enter to continue, 's' to skip, 'q' to quit: ").strip().lower()
        except EOFError:
            resp = "q"
        if resp == "q":
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
