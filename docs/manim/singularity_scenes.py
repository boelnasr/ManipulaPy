"""Deterministic Manim studies for Panda singularity analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedLine,
    Ellipse,
    FadeIn,
    MathTex,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    linear,
)

MANIM_DIR = Path(__file__).resolve().parent
EXAMPLES = MANIM_DIR.parents[0] / "examples"
for location in (MANIM_DIR, EXAMPLES):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from robotics_motion_studies import compute_singularity_results  # noqa: E402
from scientific_scene import (  # noqa: E402
    AMBER,
    INK,
    MUTED,
    RULE,
    TEAL,
    VIOLATION,
    metric_badge,
    sampled_panda_chain,
    scientific_legend,
    study_title,
)


SINGULARITY_THRESHOLD = 1e-4


def _ellipsoid(radii: np.ndarray, scale: float) -> VGroup:
    """Show an abstract singular-value cross-section of the 6D twist ellipsoid."""
    width = max(0.06, 3.8 * float(radii[0]) / scale)
    height = max(0.04, 3.8 * float(radii[-1]) / scale)
    ellipse = Ellipse(
        width=width,
        height=height,
        color=TEAL,
        stroke_width=3,
        fill_color=TEAL,
        fill_opacity=0.12,
    ).shift(RIGHT * 3.1 + DOWN * 0.2)
    strong = MathTex(r"\sigma_{J,\max}", color=TEAL, font_size=20)
    weak = MathTex(r"\sigma_{J,\min}", color=AMBER, font_size=20)
    strong.next_to(ellipse, RIGHT, buff=0.12)
    weak.next_to(ellipse, DOWN, buff=0.12)
    return VGroup(ellipse, strong, weak)


class PandaManipulabilityCollapse(Scene):
    """Show the full-Jacobian weakest twist mode collapsing near singularity."""

    def construct(self) -> None:
        result = compute_singularity_results()
        scale = float(np.max(result.singular_values))
        title = study_title(
            "A full-Jacobian twist mode collapses",
            "Abstract singular-value cross-section · not a workspace ellipse",
        ).to_edge(UP, buff=0.28)
        progress = ValueTracker(0.0)
        chain = sampled_panda_chain(result.theta, progress)
        ellipsoid = _ellipsoid(result.singular_values[0], scale)
        final_ellipsoid = _ellipsoid(result.singular_values[-1], scale)
        ratio = result.singular_values[-1, -1] / result.singular_values[0, -1]
        badge = metric_badge("weak-axis retention", f"{100.0 * ratio:.1f}", "%", "warning")
        badge.next_to(final_ellipsoid, DOWN, buff=0.16)

        self.add(title, chain, ellipsoid)
        self.play(
            progress.animate.set_value(1.0),
            ellipsoid.animate.become(final_ellipsoid),
            run_time=3.6,
            rate_func=linear,
        )
        self.play(FadeIn(badge), run_time=0.35, rate_func=linear)
        self.wait(0.35)


class PandaSingularityMonitor(Scene):
    """Track minimum singular value and condition number along one path."""

    def construct(self) -> None:
        result = compute_singularity_results()
        log_sigma = np.log10(np.maximum(result.minimum_sigma, 1e-6))
        log_condition = np.log10(np.maximum(result.condition_number, 1.0))
        sigma_axes = Axes(
            x_range=[0.0, 4.0, 1.0],
            y_range=[-6.0, 0.0, 2.0],
            x_length=6.7,
            y_length=1.45,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 14},
        ).shift(RIGHT * 2.45 + UP * 0.45)
        condition_axes = Axes(
            x_range=[0.0, 4.0, 1.0],
            y_range=[0.0, 5.0, 1.0],
            x_length=6.7,
            y_length=1.45,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 14},
        ).shift(RIGHT * 2.45 + DOWN * 1.45)
        sigma_curve = sigma_axes.plot_line_graph(
            result.time,
            log_sigma,
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3,
        )
        condition_curve = condition_axes.plot_line_graph(
            result.time,
            log_condition,
            add_vertex_dots=False,
            line_color=AMBER,
            stroke_width=3,
        )
        threshold = DashedLine(
            sigma_axes.coords_to_point(0.0, -4.0),
            sigma_axes.coords_to_point(4.0, -4.0),
            color=VIOLATION,
            stroke_width=2,
        )
        threshold_label = MathTex(r"10^{-4}", color=VIOLATION, font_size=18)
        threshold_label.next_to(threshold, UP, buff=0.04).align_to(threshold, RIGHT)
        labels = VGroup(
            Text("log10 minimum singular value", color=MUTED, font_size=14).next_to(
                sigma_axes, UP, buff=0.08
            ),
            Text("log10 condition number", color=MUTED, font_size=14).next_to(
                condition_axes, UP, buff=0.08
            ),
            Text("time [s]", color=MUTED, font_size=14).next_to(
                condition_axes, DOWN, buff=0.10
            ),
        )
        legend = scientific_legend(
            (("σ min", TEAL, False), ("condition", AMBER, True))
        ).scale(0.72).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.70)
        title = study_title(
            "One path, two conditioning signals",
            "The public 1e-4 threshold marks the near-singular final pose",
        ).to_edge(UP, buff=0.28)
        progress = ValueTracker(0.0)
        chain = sampled_panda_chain(result.theta, progress)
        final_badge = metric_badge(
            "status",
            "near singular",
            status="violation",
        ).next_to(condition_axes, DOWN, buff=0.12).align_to(condition_axes, RIGHT)

        self.add(
            title,
            chain,
            sigma_axes,
            condition_axes,
            threshold,
            threshold_label,
            labels,
            legend,
        )
        self.play(
            Create(sigma_curve),
            Create(condition_curve),
            progress.animate.set_value(1.0),
            run_time=3.55,
            rate_func=linear,
        )
        self.play(FadeIn(final_badge), run_time=0.35, rate_func=linear)
        self.wait(0.35)
