#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exit successfully only when a JUnit XML report contains zero skips."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _count(value: str, *, element: str, path: Path) -> int:
    """Parse one non-negative JUnit count with a useful failure message."""
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{path}: {element} has invalid skipped count {value!r}"
        ) from exc
    if count < 0:
        raise ValueError(f"{path}: {element} has negative skipped count {count}")
    return count


def skipped_count(root: ET.Element, path: Path) -> int:
    """Return a non-double-counted skipped total for common JUnit layouts.

    Pytest writes a ``testsuites`` root with one or more ``testsuite``
    children, while other producers use a single ``testsuite`` root or nested
    suites. Aggregate attributes can repeat child totals, so each outer suite's
    declared count is authoritative for that subtree. Concrete skipped
    testcases are also counted to catch inconsistent producer metadata.
    """
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError(
            f"{path}: expected JUnit root <testsuite> or <testsuites>, "
            f"found <{root.tag}>"
        )

    declared = {}
    for element in (root, *root.findall(".//testsuite")):
        if "skipped" in element.attrib:
            declared[id(element)] = _count(
                element.attrib["skipped"],
                element=element.tag,
                path=path,
            )

    def suite_total(suite: ET.Element) -> int:
        if id(suite) in declared:
            return declared[id(suite)]
        direct = len(suite.findall("./testcase/skipped"))
        return direct + sum(
            suite_total(child) for child in suite.findall("./testsuite")
        )

    if id(root) in declared:
        aggregate = declared[id(root)]
    elif root.tag == "testsuite":
        aggregate = suite_total(root)
    else:
        aggregate = sum(suite_total(suite) for suite in root.findall("./testsuite"))
    concrete = len(root.findall(".//testcase/skipped"))
    return max([aggregate, concrete, *declared.values()], default=0)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} PATH", file=sys.stderr)
        return 2

    path = Path(argv[1])
    try:
        root = ET.parse(path).getroot()
        skipped = skipped_count(root, path)
    except (OSError, ET.ParseError, ValueError) as exc:
        print(f"JUnit validation failed: {exc}", file=sys.stderr)
        return 2

    if skipped:
        print(
            f"JUnit validation failed: {path} reports {skipped} skipped test(s)",
            file=sys.stderr,
        )
        return 1
    print(f"JUnit validation passed: {path} reports zero skipped tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
