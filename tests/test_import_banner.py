#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
``import ManipulaPy`` keeps stdout clean and its greeting ASCII.

Before 1.4.2 an emoji banner was printed to stdout at import, which raised
UnicodeEncodeError on a Windows cp1252 console and crashed the import. The
greeting now goes to stderr as plain ASCII, unless MANIPULAPY_QUIET=1.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_in_subprocess(**env_overrides: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "MANIPULAPY_QUIET"}
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import ManipulaPy"],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=300,
        check=False,
    )


def test_import_succeeds_on_cp1252_console() -> None:
    result = _import_in_subprocess(PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"Traceback" not in result.stderr


def test_import_prints_nothing_to_stdout() -> None:
    result = _import_in_subprocess()
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""


def test_greeting_is_ascii_on_stderr() -> None:
    from ManipulaPy import __version__

    result = _import_in_subprocess(PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    greeting = [
        line
        for line in result.stderr.splitlines()
        if line.startswith(f"ManipulaPy {__version__} loaded".encode())
    ]
    assert len(greeting) == 1, result.stderr
    greeting[0].decode("ascii")  # raises on any non-ASCII byte


def test_quiet_flag_silences_the_greeting() -> None:
    result = _import_in_subprocess(MANIPULAPY_QUIET="1")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""
    assert b"ManipulaPy" not in result.stderr
