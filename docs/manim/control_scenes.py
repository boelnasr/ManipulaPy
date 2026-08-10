"""Deterministic Manim studies for Panda manipulator control."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedVMobject,
    FadeIn,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    linear,
)

MANIM_DIR = Path(__file__).resolve().parent
EXAMPLES = MANIM_DIR.parents[0] / "examples"
for location in (MANIM_DIR, EXAMPLES):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from robotics_motion_studies import compute_control_results  # noqa: E402
from scientific_scene import (  # noqa: E402
    AMBER,
    INK,
    MUTED,
    PANEL,
    RULE,
    TEAL,
    VIOLATION,
    metric_badge,
    panda_chain,
    scientific_legend,
    study_title,
)


REFERENCE = "#FFFFFF"


def _metric_display(value: float, unit: str) -> tuple[str, str]:
    if not np.isfinite(value):
        return "not reached", "warning"
    return f"{value:.3g}", "normal"


class PandaControllerComparison(Scene):
    """Compare equal-condition open-loop, PID, and computed-torque tracking."""

    def construct(self) -> None:
        result = compute_control_results()
        joint = result.target_joint
        histories = [run.theta[:, joint] for run in result.runs.values()]
        y_min = float(np.floor(min(np.min(values) for values in histories) * 2) / 2)
        y_max = float(np.ceil(max(np.max(values) for values in histories) * 2) / 2)
        y_max = max(y_max, 0.5)
        axes = Axes(
            x_range=[0.0, 4.0, 1.0],
            y_range=[y_min, y_max, 0.5],
            x_length=7.0,
            y_length=3.6,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 14},
        ).shift(RIGHT * 2.35 + DOWN * 0.25)
        reference = axes.plot_line_graph(
            result.time,
            result.reference[:, joint],
            add_vertex_dots=False,
            line_color=REFERENCE,
            stroke_width=2.0,
        )
        open_loop = DashedVMobject(
            axes.plot_line_graph(
                result.time,
                result.runs["open_loop"].theta[:, joint],
                add_vertex_dots=False,
                line_color=VIOLATION,
                stroke_width=2.2,
            ),
            num_dashes=34,
        )
        pid = DashedVMobject(
            axes.plot_line_graph(
                result.time,
                result.runs["pid"].theta[:, joint],
                add_vertex_dots=False,
                line_color=AMBER,
                stroke_width=2.5,
            ),
            num_dashes=42,
        )
        computed = axes.plot_line_graph(
            result.time,
            result.runs["computed_torque"].theta[:, joint],
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3.0,
        )
        title = study_title(
            "Feedback changes the same Panda step response",
            "Identical target, disturbance, plant, integration, and torque limits",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.runs["computed_torque"].theta[0])
        final_chain = panda_chain(result.runs["computed_torque"].theta[-1])
        legend = scientific_legend(
            (
                ("reference", REFERENCE, False),
                ("open-loop feedforward", VIOLATION, True),
                ("PID", AMBER, True),
                ("computed torque", TEAL, False),
            ),
            font_size=13,
        ).next_to(axes, UP, buff=0.10).align_to(axes, RIGHT)
        labels = VGroup(
            Text("time [s]", color=MUTED, font_size=14).next_to(
                axes.x_axis, DOWN, buff=0.12
            ),
            Text("joint 1 [rad]", color=MUTED, font_size=14)
            .rotate(np.pi / 2)
            .next_to(axes.y_axis, LEFT, buff=0.12),
        )

        self.add(title, chain, axes, labels, legend)
        self.play(
            Create(reference),
            Create(open_loop),
            Create(pid),
            Create(computed),
            Transform(chain, final_chain),
            run_time=3.65,
            rate_func=linear,
        )
        self.wait(0.4)


class PandaControlMetrics(Scene):
    """Synchronize tracking, effort, error, and public response metrics."""

    def construct(self) -> None:
        result = compute_control_results()
        run = result.runs["computed_torque"]
        joint = result.target_joint
        response = run.theta[:, joint]
        reference = result.reference[:, joint]
        error = reference - response
        effort = np.abs(run.torque[:, joint])
        response_axes = Axes(
            x_range=[0.0, 4.0, 1.0],
            y_range=[-0.02, 0.20, 0.05],
            x_length=6.8,
            y_length=2.25,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 13},
        ).shift(RIGHT * 2.4 + UP * 0.35)
        signal_axes = Axes(
            x_range=[0.0, 4.0, 1.0],
            y_range=[0.0, 1.0, 0.25],
            x_length=6.8,
            y_length=1.25,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.2, "font_size": 12},
        ).shift(RIGHT * 2.4 + DOWN * 1.70)
        tolerance = 0.02 * abs(float(reference[-1]))
        band = Rectangle(
            width=4.0 * response_axes.x_axis.unit_size,
            height=2.0 * tolerance * response_axes.y_axis.unit_size,
            color=TEAL,
            fill_color=TEAL,
            fill_opacity=0.10,
            stroke_width=1,
        ).move_to(response_axes.coords_to_point(2.0, float(reference[-1])))
        response_curve = response_axes.plot_line_graph(
            result.time,
            response,
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3,
        )
        reference_curve = DashedVMobject(
            response_axes.plot_line_graph(
                result.time,
                reference,
                add_vertex_dots=False,
                line_color=REFERENCE,
                stroke_width=2,
            ),
            num_dashes=36,
        )
        error_scale = max(float(np.max(np.abs(error))), 1e-12)
        effort_scale = max(float(np.max(effort)), 1e-12)
        error_curve = signal_axes.plot_line_graph(
            result.time,
            np.abs(error) / error_scale,
            add_vertex_dots=False,
            line_color=AMBER,
            stroke_width=2.4,
        )
        effort_curve = DashedVMobject(
            signal_axes.plot_line_graph(
                result.time,
                effort / effort_scale,
                add_vertex_dots=False,
                line_color=VIOLATION,
                stroke_width=2.2,
            ),
            num_dashes=36,
        )
        badges = VGroup()
        badge_specs = (
            ("rise time", result.metrics["rise_time"], "s"),
            ("overshoot", result.metrics["percent_overshoot"], "%"),
            ("settling time", result.metrics["settling_time"], "s"),
            ("steady error", result.metrics["steady_state_error"], "rad"),
        )
        for label, value, unit in badge_specs:
            displayed, status = _metric_display(value, unit)
            badges.add(metric_badge(label, displayed, unit, status))
        badges.arrange_in_grid(rows=2, cols=2, buff=(0.15, 0.12))
        badges.scale(0.82).next_to(signal_axes, DOWN, buff=0.13).align_to(
            signal_axes, RIGHT
        )
        title = study_title(
            "Tracking quality needs more than one number",
            "Computed torque · public step metrics · ±2% settling tolerance",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(run.theta[0])
        final_chain = panda_chain(run.theta[-1])
        labels = VGroup(
            Text("response + tolerance", color=MUTED, font_size=14).next_to(
                response_axes, UP, buff=0.07
            ),
            Text("normalized |error| / |effort|", color=MUTED, font_size=13).next_to(
                signal_axes, UP, buff=0.06
            ),
        )

        self.add(title, chain, response_axes, signal_axes, band, labels)
        self.play(
            Create(reference_curve),
            Create(response_curve),
            Create(error_curve),
            Create(effort_curve),
            Transform(chain, final_chain),
            run_time=3.55,
            rate_func=linear,
        )
        self.play(FadeIn(badges), run_time=0.35, rate_func=linear)
        self.wait(0.35)
