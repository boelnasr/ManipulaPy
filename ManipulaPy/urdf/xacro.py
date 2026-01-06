#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Xacro Macro Processor

Processes Xacro files (XML macros) to generate standard URDF XML.

Copyright (c) 2025 Mohamed Aboelnasr
"""

from pathlib import Path
from typing import Optional, Dict
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


class XacroProcessor:
    """
    Xacro macro processor.

    Supports:
    - Using system xacro command (ROS)
    - Basic inline macro expansion (fallback)
    """

    @classmethod
    def process(
        cls,
        filename: Path,
        args: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Process Xacro file to URDF XML string.

        Args:
            filename: Path to .xacro file
            args: Xacro arguments {name: value}

        Returns:
            Processed URDF XML string
        """
        filename = Path(filename).resolve()

        if not filename.exists():
            raise FileNotFoundError(f"Xacro file not found: {filename}")

        # Try system xacro command first (ROS)
        try:
            return cls._process_with_command(filename, args)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try xacro Python package
        try:
            return cls._process_with_package(filename, args)
        except ImportError:
            pass

        # Fallback: basic processing (no macro expansion)
        logger.warning(
            "Xacro command and package not available. "
            "Attempting basic processing without macro expansion."
        )
        return cls._process_basic(filename)

    @classmethod
    def _process_with_command(
        cls,
        filename: Path,
        args: Optional[Dict[str, str]] = None,
    ) -> str:
        """Process using system xacro command."""
        cmd = ["xacro", str(filename)]

        if args:
            for key, value in args.items():
                cmd.append(f"{key}:={value}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout

    @classmethod
    def _process_with_package(
        cls,
        filename: Path,
        args: Optional[Dict[str, str]] = None,
    ) -> str:
        """Process using xacro Python package."""
        import xacro

        # Build argument list
        xacro_args = [str(filename)]
        if args:
            for key, value in args.items():
                xacro_args.append(f"{key}:={value}")

        # Process xacro
        doc = xacro.process_file(str(filename), mappings=args or {})
        return doc.toprettyxml(indent="  ")

    @classmethod
    def _process_basic(cls, filename: Path) -> str:
        """
        Basic xacro processing without macro expansion.

        Handles:
        - xacro:include directives
        - Removes xacro namespace declarations

        Does NOT handle:
        - Macro definitions and calls
        - Property substitutions
        - Conditionals
        """
        import xml.etree.ElementTree as ET
        import re

        content = filename.read_text(encoding="utf-8")

        # Remove xacro namespace prefix from tags
        content = re.sub(r"<xacro:", "<", content)
        content = re.sub(r"</xacro:", "</", content)

        # Remove xacro namespace declaration
        content = re.sub(r'xmlns:xacro="[^"]*"', "", content)

        # Handle includes
        include_pattern = re.compile(r'<include\s+filename="([^"]+)"\s*/>')

        def replace_include(match):
            include_file = match.group(1)

            # Resolve relative path
            if not Path(include_file).is_absolute():
                include_file = filename.parent / include_file

            include_path = Path(include_file)
            if include_path.exists():
                include_content = include_path.read_text(encoding="utf-8")
                # Strip XML declaration from included file
                include_content = re.sub(r"<\?xml[^?]*\?>", "", include_content)
                # Strip root robot tags
                include_content = re.sub(
                    r"<robot[^>]*>|</robot>", "", include_content
                )
                return include_content
            else:
                logger.warning(f"Include file not found: {include_file}")
                return ""

        content = include_pattern.sub(replace_include, content)

        # Remove remaining xacro-specific elements (macros, properties, etc.)
        content = re.sub(r"<macro[^>]*>.*?</macro>", "", content, flags=re.DOTALL)
        content = re.sub(r"<property[^>]*/>", "", content)
        content = re.sub(r"<arg[^>]*/>", "", content)

        return content

    @classmethod
    def is_xacro_file(cls, filename: Path) -> bool:
        """Check if file is a xacro file."""
        filename = Path(filename)
        return (
            filename.suffix.lower() == ".xacro"
            or ".xacro" in filename.name.lower()
        )
