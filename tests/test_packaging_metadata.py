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


def test_accelerators_remain_optional_and_tpu_extra_is_linux_only() -> None:
    project = _project_metadata()
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]

    assert all(
        accelerator not in dependency.lower()
        for dependency in dependencies
        for accelerator in ("torch", "jax", "cupy")
    )
    assert extras["jax-tpu"] == [
        "jax[tpu]>=0.6.0; sys_platform == 'linux'",
    ]
    assert all("jax[tpu]" not in dependency.lower() for dependency in extras["all"])
