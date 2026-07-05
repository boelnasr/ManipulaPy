#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Atheris fuzz target for the native URDF parser (URDF.from_xml_string)."""

import logging
import sys
import xml.etree.ElementTree as ET

import atheris

# include filters on the top-level package name, so this instruments
# ManipulaPy modules only (not numpy/scipy).
with atheris.instrument_imports(include=["ManipulaPy"]):
    from ManipulaPy.urdf import URDF

# Malformed inputs trigger a recovery-path warning per exec; keep output quiet.
logging.disable(logging.CRITICAL)


def TestOneInput(data: bytes) -> None:
    xml_string = data.decode("utf-8", errors="ignore")
    try:
        URDF.from_xml_string(xml_string)  # load_meshes defaults to False
    except (ValueError, ET.ParseError):
        # The parser deliberately raises ValueError on invalid/unrecoverable
        # URDF content; ParseError is its documented XML failure mode.
        pass


if __name__ == "__main__":
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()
