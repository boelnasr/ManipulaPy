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

The rewrite has a cost that the rendered signature alone cannot pay back: an
author who respells an annotation *on purpose* - say ``NDArray[np.float64]``
-> ``np.ndarray[Any, np.dtype[np.float64]]``, which drops the shape typing -
lands in exactly the same canonical string as a NumPy upgrade does. That is
real API drift, and at the rendered-signature layer the two causes are
genuinely indistinguishable. ``annotation_sources`` supplies the missing
facet: what the author literally *wrote*, read back from the AST. It changes
when the annotation is edited and does not change when NumPy changes its repr,
so freezing it alongside the canonical signature restores the lost signal
without reintroducing the cross-version false positives.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import ast
import inspect
import re
import textwrap
from typing import Any, Callable, Dict

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


# Recorded instead of an annotation map when the source text cannot be read
# back (C implementations, exec'd or generated code, wrappers that hide their
# definition). Spelled as a one-entry map with an obviously bracketed key so it
# can never be confused with "the function has no annotations" - an empty map -
# nor with a real parameter name.
UNAVAILABLE_ANNOTATION_SOURCES: Dict[str, str] = {
    "<annotation-source>": "<unavailable>"
}


def _annotation_text(node: ast.expr) -> str:
    """Layout-independent source text of one annotation expression.

    ``ast.unparse`` rather than the raw source slice: it preserves exactly the
    names the author wrote - ``NDArray[np.float64]`` stays ``NDArray[...]``,
    ``np.ndarray[Any, np.dtype[np.float64]]`` stays spelled out - while
    discarding line breaks, indentation and padding, so reflowing a long
    annotation is not frozen as API drift.
    """
    return ast.unparse(node)


def annotation_sources(fn: Callable[..., Any]) -> Dict[str, str]:
    """Return ``{parameter: annotation source text}`` as the author wrote it.

    Keyed by parameter name plus ``"return"`` for the return annotation;
    unannotated parameters are omitted (their disappearance is already visible
    in the rendered signature). Returns
    :data:`UNAVAILABLE_ANNOTATION_SOURCES` when the definition's source cannot
    be recovered.
    """
    target = inspect.unwrap(fn)
    target = getattr(target, "__func__", target)
    try:
        source = textwrap.dedent(inspect.getsource(target))
        definition = ast.parse(source).body[0]
    except (OSError, TypeError, SyntaxError, ValueError, IndexError):
        return dict(UNAVAILABLE_ANNOTATION_SOURCES)
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return dict(UNAVAILABLE_ANNOTATION_SOURCES)

    args = definition.args
    parameters = [
        *args.posonlyargs,
        *args.args,
        *([args.vararg] if args.vararg else []),
        *args.kwonlyargs,
        *([args.kwarg] if args.kwarg else []),
    ]
    sources = {
        arg.arg: _annotation_text(arg.annotation)
        for arg in parameters
        if arg.annotation is not None
    }
    if definition.returns is not None:
        sources["return"] = _annotation_text(definition.returns)
    return sources
