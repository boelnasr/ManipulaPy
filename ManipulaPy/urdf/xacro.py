#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Xacro Macro Processor

Processes Xacro files (XML macros) to generate standard URDF XML.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Xacro arg names must be plain Python-identifier-shaped tokens. Values must
# not contain shell metacharacters or look like CLI flags (so a malicious
# arg can't smuggle a flag past xacro). Negative numbers (``-1.5``) are
# allowed because the flag check requires an alphabetic char after the dash.
_XACRO_ARG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_VALUE_CHARS = set(";|&`$<>\n\r\x00")
_LOOKS_LIKE_FLAG_RE = re.compile(r"^-[^0-9.]")
_MAX_VALUE_LEN = 4096


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
        """Process a xacro file via the system ``xacro`` command.

        Builds an ``xacro <filename> key:=value ...`` command line, validating
        every argument name and value to prevent shell injection before passing
        them to :func:`subprocess.run`.

        Args:
            filename: Path to the ``.xacro`` file to process.
            args: Optional mapping of xacro argument names to values. Names must
                match ``[A-Za-z_][A-Za-z0-9_]*``; values must not contain shell
                metacharacters, look like CLI flags, or exceed the maximum
                allowed length.

        Returns:
            str: The expanded URDF XML emitted on the command's standard output.

        Raises:
            ValueError: If an argument name is invalid, or a value contains a
                forbidden shell metacharacter, resembles a CLI flag, or exceeds
                the maximum value length.
        """
        cmd = ["xacro", str(filename)]

        if args:
            for key, value in args.items():
                key_str = str(key)
                if not _XACRO_ARG_NAME_RE.match(key_str):
                    raise ValueError(
                        f"Invalid xacro arg name {key!r}: must match "
                        "[A-Za-z_][A-Za-z0-9_]*"
                    )
                value_str = str(value)
                if any(c in value_str for c in _FORBIDDEN_VALUE_CHARS):
                    raise ValueError(
                        f"Invalid xacro arg value for {key!r}: contains "
                        "shell metacharacter"
                    )
                if _LOOKS_LIKE_FLAG_RE.match(value_str):
                    raise ValueError(
                        f"Invalid xacro arg value for {key!r}: value looks "
                        f"like a CLI flag ({value_str!r})"
                    )
                if len(value_str) > _MAX_VALUE_LEN:
                    raise ValueError(
                        f"Invalid xacro arg value for {key!r}: exceeds "
                        f"{_MAX_VALUE_LEN} chars"
                    )
                cmd.append(f"{key_str}:={value_str}")

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
        """Process a xacro file via the ``xacro`` Python package.

        Validates argument names, then expands the file in-process using
        :func:`xacro.process_file` and serializes the result to pretty-printed
        XML.

        Args:
            filename: Path to the ``.xacro`` file to process.
            args: Optional mapping of xacro argument names to values, passed as
                xacro ``mappings``. Names must match
                ``[A-Za-z_][A-Za-z0-9_]*``.

        Returns:
            str: The expanded URDF as pretty-printed XML (two-space indentation).

        Raises:
            ImportError: If the ``xacro`` Python package is not installed.
            ValueError: If an argument name does not match the allowed pattern.
        """
        import xacro

        if args:
            for key in args:
                if not _XACRO_ARG_NAME_RE.match(str(key)):
                    raise ValueError(
                        f"Invalid xacro arg name {key!r}: must match "
                        "[A-Za-z_][A-Za-z0-9_]*"
                    )

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
        """Basic xacro processing without macro expansion.

        A fallback used when neither the system ``xacro`` command nor the
        ``xacro`` Python package is available. It strips the ``xacro``
        namespace, inlines ``<include>`` directives, and removes unhandled
        xacro elements via regular expressions.

        Handles:
            - ``xacro:include`` directives (inlined into the document)
            - Removal of xacro namespace declarations

        Does NOT handle:
            - Macro definitions and calls
            - Property substitutions
            - Conditionals

        Args:
            filename: Path to the ``.xacro`` file to process.

        Returns:
            str: The processed URDF XML with the xacro namespace removed and
            includes inlined. Macros, properties, and ``<arg>`` elements are
            stripped rather than expanded.
        """
        import re
        import xml.etree.ElementTree as ET

        content = filename.read_text(encoding="utf-8")

        # Remove xacro namespace prefix from tags
        content = re.sub(r"<xacro:", "<", content)
        content = re.sub(r"</xacro:", "</", content)

        # Remove xacro namespace declaration
        content = re.sub(r'xmlns:xacro="[^"]*"', "", content)

        # Handle includes
        include_pattern = re.compile(r'<include\s+filename="([^"]+)"\s*/>')

        def replace_include(match: re.Match) -> str:
            """Inline the contents of an ``<include>`` directive.

            Resolves the included file path (relative paths are resolved
            against the parent file's directory), reads its contents, and
            strips the XML declaration and wrapping ``<robot>`` tags so the
            fragment can be embedded.

            Args:
                match: Regex match for ``<include filename="..."/>``; group 1
                    holds the include filename.

            Returns:
                str: The cleaned XML contents of the included file, or an empty
                string if the file does not exist.
            """
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
                include_content = re.sub(r"<robot[^>]*>|</robot>", "", include_content)
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
        """Check whether a file is a xacro file.

        Args:
            filename: Path (or path-like) to inspect.

        Returns:
            bool: ``True`` if the file has a ``.xacro`` suffix or contains
            ``.xacro`` in its name (case-insensitive), otherwise ``False``.
        """
        filename = Path(filename)
        return filename.suffix.lower() == ".xacro" or ".xacro" in filename.name.lower()
