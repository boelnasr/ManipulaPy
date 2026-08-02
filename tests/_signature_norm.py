#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canonicalise NumPy-version-variant spellings in rendered signatures.

``numpy.typing.NDArray[np.float64]`` is an alias whose repr embeds NumPy's own
shape type-parameter, and NumPy has respelled it three times across 2.x:

    numpy 2.0/2.1  ->  numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    numpy 2.2      ->  numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float64]]
    numpy >= 2.3   ->  numpy.ndarray[tuple[Any, ...], numpy.dtype[numpy.float64]]

``inspect.formatannotation`` adds a fourth axis: it strips the ``typing.``
prefix only when the outermost annotation lives in ``typing``, so the very same
alias renders ``tuple[typing.Any, ...]`` bare and ``tuple[Any, ...]`` nested
inside ``Union``/``Optional``/``Tuple``.

None of that is a ManipulaPy fact - ManipulaPy always writes ``NDArray[...]``
and never spells a shape parameter itself - so the shape slot is rewritten to
one canonical form before a signature is frozen or compared. Only the shape
slot of a ``numpy.ndarray[SHAPE, numpy.dtype[...]]`` alias is touched: the
dtype, the parameter names, their order, their defaults and every non-NumPy
annotation are left byte-for-byte intact.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import re

# Every shape spelling NumPy 2.x has shipped, times the typing.-prefix variants
# inspect.formatannotation can produce. Enumerated literally (never a wildcard)
# so the rewrite cannot silently swallow an unrecognised annotation.
_SHAPE_SPELLINGS = (
    "Any",
    "typing.Any",
    "tuple[int, ...]",
    "tuple[Any, ...]",
    "tuple[typing.Any, ...]",
)

# Canonical form is the numpy 2.2 spelling, which is what the checked-in
# goldens already contain - so normalising is a no-op on them and no frozen
# value has to be regenerated. The token asserts nothing about the shape; it is
# purely a canonicalisation placeholder.
_CANONICAL_PREFIX = "numpy.ndarray[tuple[int, ...], "

_SHAPE_RE = re.compile(
    r"numpy\.ndarray\[(?:"
    + "|".join(re.escape(spelling) for spelling in _SHAPE_SPELLINGS)
    + r"), (?=numpy\.dtype\[)"
)


def canonical_signature(text: str) -> str:
    """Return ``text`` with NumPy ndarray shape parameters canonicalised."""
    return _SHAPE_RE.sub(_CANONICAL_PREFIX, text)
