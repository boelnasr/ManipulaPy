#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Byte-exact source freeze for the ``cuda.jit`` kernels.

Both CUDA kernel suites assert that the raw kernel bodies never move, so the
digest lives here once instead of being mirrored into each of them: two copies
of a freeze are two places for a blind spot to survive a fix.

The AST is used only to *locate* the kernels; the digest is taken over
repo-owned bytes. ``ast.dump`` cannot be hashed here because it renders
whatever fields the running CPython declares in ``_fields``, so its text moves
with the interpreter: 3.9 dropped ``annotation=None``/``kind=None``, 3.12
appended PEP 695 ``type_params=[]`` to every ``FunctionDef``, and 3.13 flips
the ``show_empty`` default. The kernel source does not move.

``split("\\n")`` rather than ``splitlines()`` is deliberate: ``splitlines``
also breaks on form feed and \\x0b, which the tokenizer does not count as line
breaks, so it could desynchronise from ``lineno``.

Lines are hashed exactly as written -- no ``rstrip``. Stripping was originally
chosen so a trailing-whitespace-only edit would not move the digest, but
trailing whitespace is not always insignificant: inside a docstring or any
other multiline string it is part of the value, so a stripped digest reports
"unchanged" for a kernel whose ``__doc__`` and code object genuinely differ.
A freeze that can be edited around is not a freeze, so the exact bytes win.
The cost -- a whitespace-only touch now moves the digest -- is bounded by the
lint gate, which rejects W291/W293 in the kernel sources anyway.

Digests are keyed ``"<module stem>.<function name>"``. A bare function name is
not unique across the package, so a kernel added to one module under a name
already used in another would overwrite the earlier entry and hide a real
edit to it. The stem comes from the file path rather than ``__name__`` so the
key does not depend on how the caller imported the module.
"""

import ast
import hashlib
from pathlib import Path


def cuda_kernel_digests(module) -> dict:
    """Hash each ``cuda.jit`` kernel's own source lines, keyed by module.

    Args:
        module: An imported module whose ``__file__`` is readable source.

    Returns:
        ``{"<module stem>.<kernel name>": sha256 hexdigest}`` covering every
        function decorated with ``cuda.jit`` in that module.

    Raises:
        ValueError: Two ``cuda.jit`` kernels in the module share a name, which
            would otherwise silently collapse into one digest.
    """
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")  # read_text() already normalised newlines
    digests = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_cuda_kernel = any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "cuda"
            and decorator.func.attr == "jit"
            for decorator in node.decorator_list
        )
        if not is_cuda_kernel:
            continue
        first = node.decorator_list[0].lineno  # include the decorator itself
        payload = "\n".join(lines[first - 1 : node.end_lineno])
        key = f"{path.stem}.{node.name}"
        if key in digests:
            raise ValueError(f"Duplicate cuda.jit kernel name: {key}")
        digests[key] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digests
