"""Stable scene registry for the advanced robotics motion studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MANIM_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SceneSpec:
    """One renderable scene and its committed documentation destination."""

    key: str
    domain: str
    scene_source: Path
    scene_class: str
    filename: str


SCENES = {
    spec.key: spec
    for spec in (
        SceneSpec(
            "dynamics-mass",
            "dynamics",
            MANIM_DIR / "dynamics_scenes.py",
            "PandaMassMatrixEvolution",
            "panda_mass_matrix_evolution",
        ),
        SceneSpec(
            "dynamics-torque",
            "dynamics",
            MANIM_DIR / "dynamics_scenes.py",
            "PandaTorqueDecomposition",
            "panda_torque_decomposition",
        ),
        SceneSpec(
            "dynamics-roundtrip",
            "dynamics",
            MANIM_DIR / "dynamics_scenes.py",
            "PandaDynamicsRoundTrip",
            "panda_dynamics_round_trip",
        ),
        SceneSpec(
            "singularity-ellipsoid",
            "singularity",
            MANIM_DIR / "singularity_scenes.py",
            "PandaManipulabilityCollapse",
            "panda_manipulability_collapse",
        ),
        SceneSpec(
            "singularity-monitor",
            "singularity",
            MANIM_DIR / "singularity_scenes.py",
            "PandaSingularityMonitor",
            "panda_singularity_monitor",
        ),
        SceneSpec(
            "planning-scaling",
            "path_planning",
            MANIM_DIR / "path_planning_scenes.py",
            "PandaTimeScalingComparison",
            "panda_time_scaling_comparison",
        ),
        SceneSpec(
            "planning-domains",
            "path_planning",
            MANIM_DIR / "path_planning_scenes.py",
            "PandaInterpolationDomains",
            "panda_interpolation_domains",
        ),
        SceneSpec(
            "planning-collision",
            "path_planning",
            MANIM_DIR / "path_planning_scenes.py",
            "PandaCollisionCorrection",
            "panda_collision_correction",
        ),
        SceneSpec(
            "control-comparison",
            "control",
            MANIM_DIR / "control_scenes.py",
            "PandaControllerComparison",
            "panda_controller_comparison",
        ),
        SceneSpec(
            "control-metrics",
            "control",
            MANIM_DIR / "control_scenes.py",
            "PandaControlMetrics",
            "panda_control_metrics",
        ),
    )
}
