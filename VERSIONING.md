# Version Declaration Reference

This document lists **every place the ManipulaPy version is declared** so a
release bump can be applied consistently. Current version: **`1.3.2`**.

> Audited 2026-05-30 — all declaration sites below read `1.3.2`.

## Authoritative declarations

These are the source-of-truth version strings. A version bump **must** update
every row in this table.

| # | File | Line | Declaration | Notes |
|---|------|------|-------------|-------|
| 1 | `pyproject.toml` | 12 | `version = "1.3.2"` | Primary packaging metadata (PEP 621). |
| 2 | `setup.py` | 11 | `version="1.3.2"` | Legacy/setuptools shim; kept in sync with pyproject. |
| 3 | `ManipulaPy/__init__.py` | 23 | `__version__ = "1.3.2"` | Runtime `ManipulaPy.__version__` / `get_version()`. |
| 4 | `ManipulaPy/urdf/__init__.py` | 98 | `__version__ = "1.3.2"` | URDF subpackage version (was `1.3.0`; aligned to package version). |

## Derived / dynamic

These follow the authoritative sources automatically — normally **no manual
edit needed** — but each carries a hardcoded fallback worth checking on a bump.

| # | File | Line | Declaration | Notes |
|---|------|------|-------------|-------|
| 5 | `docs/source/conf.py` | 59 | `version = release = ManipulaPy.__version__` | Primary: read dynamically from the installed package. |
| 6 | `docs/source/conf.py` | 61 | `version = release = "1.3.2"` | Fallback used only if the import in the `try` block fails — bump alongside #3. |
| 7 | `docs/source/index.rst` | 497 | `version={1.3.2}` | BibTeX citation snippet; hardcoded, update on a bump. |

## Enforcement

`tests/test_v132_regressions.py` (the `test_*version*` cases) asserts that
`ManipulaPy.__version__` matches `pyproject.toml [project].version`, so rows
**#1** and **#3** can never silently drift. Run:

```bash
python3 -m pytest tests/test_v132_regressions.py -k version -q
```

## Intentionally NOT version declarations

These contain version-like strings but must **not** be bumped — they are
historical facts, examples, or unrelated version numbers.

| File | What it is | Why leave it |
|------|------------|--------------|
| `CHANGELOG.md` | `## [1.3.2]` heading + per-item notes | Records the 1.3.2 release; history, not a declaration. |
| Source/docs `.. versionchanged:: 1.3.2`, `"fixed in v1.3.2"`, `"upgrading from 1.3.1"` | Sphinx directives & prose | Describe *when* something changed; rewriting them would falsify history. |
| `Benchmark/README.MD` (`"manipulapy_version": "1.1.0"`) | Example benchmark JSON, timestamped `2025-07-18` | A captured result snapshot; bumping it would create an incoherent "1.3.2 @ 2025-07-18". |
| `Benchmark/performance_benchmark.py` (`'benchmark_version': '1.0.0'`) | The benchmark tool's own version | Separate from the package version. |
| `docs/CHANGELOG_GUIDE.md` (`1.2.0  # ← Update this`) | Instructional placeholder in a how-to | Illustrative example, not a live value. |

## Bump checklist

When releasing a new version, update rows **#1–#4**, **#6**, and **#7**, then:

```bash
# confirm package + subpackage report the new version
python3 -c "import ManipulaPy, ManipulaPy.urdf as u; print(ManipulaPy.__version__, u.__version__)"
# confirm no declaration drifted
python3 -m pytest tests/test_v132_regressions.py -k version -q
# confirm nothing stale remains (replace X.Y.Z with the OLD version)
git grep -nE "__version__\s*=\s*[\"']X\.Y\.Z[\"']" -- '*.py'
```
