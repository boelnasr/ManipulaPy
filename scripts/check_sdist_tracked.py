#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail if a built sdist contains files that git does not track.

The sdist is assembled from the WORKING TREE, not from git. Any untracked or
gitignored file that happens to match a MANIFEST.in include is therefore
collected and published, and a .gitignore entry at the repository root does
not protect paths nested under the package directory. That is not
hypothetical: a local tool configuration file was shipped this way in a
previous release.

Narrowing MANIFEST.in fixes the leak that happened. This check is what stops
the next one, by asserting the invariant directly rather than trusting a
pattern list to stay correct: *every file in the sdist must be tracked by
git*.

Usage:
    python scripts/check_sdist_tracked.py dist/manipulapy-1.4.0.tar.gz

Exits non-zero and lists the offenders if any untracked file is present.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

# Files setuptools synthesises during the build. They are legitimately absent
# from git and are the only untracked content an sdist may contain.
GENERATED_EXACT = {"PKG-INFO", "setup.cfg"}
GENERATED_PREFIXES = ("ManipulaPy.egg-info/",)


def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {p for p in out.split("\0") if p}


def sdist_members(archive: Path) -> list[str]:
    names = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Strip the "<name>-<version>/" root directory the sdist adds.
            _, _, relative = member.name.partition("/")
            if relative:
                names.append(relative)
    return names


def is_generated(path: str) -> bool:
    return path in GENERATED_EXACT or path.startswith(GENERATED_PREFIXES)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    archive = Path(argv[1])
    if not archive.is_file():
        print(f"error: no such sdist: {archive}", file=sys.stderr)
        return 2

    tracked = tracked_files()
    offenders = sorted(
        path
        for path in sdist_members(archive)
        if path not in tracked and not is_generated(path)
    )

    if offenders:
        print(f"FAIL: {archive.name} contains {len(offenders)} untracked file(s):")
        for path in offenders:
            print(f"  {path}")
        print(
            "\nThese are in the working tree but not in git, so they would be "
            "published to PyPI.\nEither track them, or exclude them in "
            "MANIFEST.in. Do not widen an include to make this pass."
        )
        return 1

    print(f"OK: every file in {archive.name} is tracked by git.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
