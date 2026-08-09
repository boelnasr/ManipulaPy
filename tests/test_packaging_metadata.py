"""Regression tests for published optional-dependency metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _project_metadata() -> dict:
    with PYPROJECT.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)["project"]


def test_accelerators_remain_optional_and_no_extra_installs_tpu_jax() -> None:
    project = _project_metadata()
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]

    assert all(
        accelerator not in dependency.lower()
        for dependency in dependencies
        for accelerator in ("torch", "jax", "cupy")
    )
    # No TPU extra: XLA:TPU implements neither float64 LU decomposition nor
    # int64 dot, and this backend enables jax_enable_x64 unconditionally, so
    # inv/solve and the dynamics built on them raise UNIMPLEMENTED on real
    # hardware. Advertising an installable path to that is worse than omitting
    # it; restoring one needs a per-platform precision domain first.
    assert "jax-tpu" not in extras
    assert all(
        "jax[tpu]" not in dependency.lower()
        for requirements in extras.values()
        for dependency in requirements
    )


def test_every_jax_requirement_is_gated_off_unsupported_pythons() -> None:
    """No extra may become unresolvable on a supported Python.

    ``requires-python`` admits 3.9, but JAX 0.6 declares ``>=3.10``. An
    unmarked JAX requirement does not merely leave the backend unavailable
    there -- it makes ``pip install "ManipulaPy[all]"`` fail to resolve on an
    interpreter the project claims to support.
    """
    project = _project_metadata()
    assert project["requires-python"] == ">=3.9"

    jax_requirements = [
        dependency
        for requirements in project["optional-dependencies"].values()
        for dependency in requirements
        if dependency.lower().startswith("jax")
    ]
    assert jax_requirements, "expected JAX to appear in at least one extra"
    for dependency in jax_requirements:
        assert "python_version >= '3.10'" in dependency, dependency
