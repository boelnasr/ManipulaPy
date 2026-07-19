#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""URDF topology helpers for potential-field collision checking."""

from typing import Any, Set


def build_link_adjacency(
    urdf: Any, exclude_grandparents: bool = True
) -> Set[frozenset]:
    """Pairs of link names excluded from self-collision checks.

    Always includes parent<->child pairs joined by any URDF joint. Optionally
    extends to grandparent<->grandchild, matching SRDF convention for arms whose
    successive-link geometry overlaps slightly even when not coincident.

    Returns a set of frozenset({name_a, name_b}) so callers can do
    order-independent ``frozenset(pair) in acm`` checks.
    """
    excluded = set()
    parent_of = {j.child: j.parent for j in urdf.joints}
    for j in urdf.joints:
        excluded.add(frozenset({j.parent, j.child}))
    if exclude_grandparents:
        for child, parent in parent_of.items():
            grandparent = parent_of.get(parent)
            if grandparent:
                excluded.add(frozenset({grandparent, child}))
    return excluded
