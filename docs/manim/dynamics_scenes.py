"""Deterministic Manim studies for Panda manipulator dynamics."""

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
    FadeIn,
    MathTex,
    Scene,
    Square,
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

from robotics_motion_studies import compute_dynamics_results  # noqa: E402
from scientific_scene import (  # noqa: E402
    AMBER,
    INK,
    MUTED,
    PANEL,
    RULE,
    TEAL,
    metric_badge,
    panda_chain,
    scientific_legend,
    study_title,
    time_cursor,
)


def _heatmap(matrix: np.ndarray, absolute_scale: float) -> VGroup:
    """Return a fixed-scale signed heatmap for one seven-by-seven matrix."""
    cells = VGroup()
    for row in range(7):
        for column in range(7):
            value = float(matrix[row, column])
            color = TEAL if value >= 0.0 else AMBER
            opacity = 0.10 + 0.82 * min(abs(value) / absolute_scale, 1.0)
            cell = Square(side_length=0.42, color=RULE, stroke_width=0.7)
            cell.set_fill(color, opacity=opacity)
            cell.move_to([column * 0.42, -row * 0.42, 0.0])
            cells.add(cell)
    cells.center().shift(RIGHT * 3.15 + DOWN * 0.25)
    label = MathTex(r"M(q)\;[\mathrm{kg\,m^2}]", color=INK, font_size=24)
    label.next_to(cells, UP, buff=0.18)
    return VGroup(cells, label)


def _trace_axes(y_min: float, y_max: float, y_step: float, label: str) -> VGroup:
    axes = Axes(
        x_range=[0.0, 4.0, 1.0],
        y_range=[y_min, y_max, y_step],
        x_length=7.0,
        y_length=3.25,
        axis_config={"color": RULE, "stroke_width": 1.4, "font_size": 16},
        tips=False,
    )
    axes.shift(RIGHT * 2.35 + DOWN * 0.35)
    labels = VGroup(
        Text("time [s]", color=MUTED, font_size=15).next_to(
            axes.x_axis, DOWN, buff=0.18
        ),
        Text(label, color=MUTED, font_size=15).rotate(np.pi / 2).next_to(
            axes.y_axis, LEFT, buff=0.18
        ),
    )
    return VGroup(axes, labels)


class PandaMassMatrixEvolution(Scene):
    """Show configuration motion beside a fixed-scale mass-matrix heatmap."""

    def construct(self) -> None:
        result = compute_dynamics_results()
        title = study_title(
            "Mass matrix follows configuration",
            "The same Panda carries different joint coupling across its motion",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.theta[0])
        final_chain = panda_chain(result.theta[-1])
        scale = float(np.max(np.abs(result.mass_matrices)))
        heatmap = _heatmap(result.mass_matrices[0], scale)
        final_heatmap = _heatmap(result.mass_matrices[-1], scale)
        symmetry = np.max(
            np.abs(result.mass_matrices - result.mass_matrices.swapaxes(1, 2))
        )
        badge = metric_badge("symmetry residual", f"{symmetry:.1e}")
        badge.next_to(heatmap, DOWN, buff=0.18)

        self.add(title, chain, heatmap, badge)
        self.play(
            Transform(chain, final_chain),
            Transform(heatmap, final_heatmap),
            run_time=3.7,
            rate_func=linear,
        )
        self.wait(0.4)


class PandaTorqueDecomposition(Scene):
    """Separate inertia, velocity, gravity, and total torque over one motion."""

    def construct(self) -> None:
        result = compute_dynamics_results()
        joint = int(np.ptp(result.total_torque, axis=0).argmax())
        traces = (
            (result.inertia[:, joint], TEAL, False, "inertia"),
            (result.velocity_force[:, joint], AMBER, True, "velocity"),
            (result.gravity[:, joint], INK, False, "gravity"),
            (result.total_torque[:, joint], "#FFFFFF", True, "total"),
        )
        bound = max(float(np.max(np.abs(values))) for values, *_rest in traces)
        bound = max(1.0, float(np.ceil(bound)))
        plot = _trace_axes(-bound, bound, max(1.0, bound / 2.0), "torque [N m]")
        axes = plot[0]
        curves = VGroup(
            *(
                axes.plot_line_graph(
                    result.time,
                    values,
                    add_vertex_dots=False,
                    line_color=color,
                    stroke_width=3.0 if label == "total" else 2.2,
                )
                for values, color, _dashed, label in traces
            )
        )
        legend = scientific_legend(
            tuple((label, color, dashed) for _v, color, dashed, label in traces)
        ).scale(0.9)
        legend.next_to(axes, UP, buff=0.16).align_to(axes, RIGHT)
        title = study_title(
            "Torque is a sum of physical effects",
            f"Joint {joint + 1} · one model, one state, one shared time cursor",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.theta[0])
        final_chain = panda_chain(result.theta[-1])
        cursor = time_cursor(axes, 0.0)
        final_cursor = time_cursor(axes, float(result.time[-1]))

        self.add(title, chain, plot, legend, cursor)
        self.play(
            Create(curves),
            Transform(chain, final_chain),
            Transform(cursor, final_cursor),
            run_time=3.7,
            rate_func=linear,
        )
        self.wait(0.4)


class PandaDynamicsRoundTrip(Scene):
    """Verify inverse-dynamics torque by recovering the requested acceleration."""

    def construct(self) -> None:
        result = compute_dynamics_results()
        joint = int(np.ptp(result.acceleration, axis=0).argmax())
        desired = result.acceleration[:, joint]
        recovered = result.recovered_acceleration[:, joint]
        bound = max(0.25, float(np.ceil(np.max(np.abs(desired)) * 2.0) / 2.0))
        plot = _trace_axes(
            -bound, bound, max(0.25, bound / 2.0), r"acceleration [rad/s^2]"
        )
        axes = plot[0]
        desired_curve = axes.plot_line_graph(
            result.time,
            desired,
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3,
        )
        recovered_curve = axes.plot_line_graph(
            result.time,
            recovered,
            add_vertex_dots=False,
            line_color=AMBER,
            stroke_width=2,
        )
        legend = scientific_legend(
            (("desired", TEAL, False), ("recovered", AMBER, True))
        ).next_to(axes, UP, buff=0.15).align_to(axes, RIGHT)
        error = float(np.max(np.abs(result.acceleration - result.recovered_acceleration)))
        badge = metric_badge("max round-trip error", f"{error:.1e}", "rad/s²")
        badge.next_to(axes, DOWN, buff=0.14).align_to(axes, RIGHT)
        title = study_title(
            "Inverse and forward dynamics close the loop",
            f"Joint {joint + 1} · recovered acceleration overlays the request",
        ).to_edge(UP, buff=0.28)
        flow = Text(
            "acceleration → inverse dynamics → torque → forward dynamics",
            color=MUTED,
            font_size=16,
        ).next_to(title, DOWN, buff=0.16)

        self.add(title, flow, plot, legend)
        self.play(
            Create(desired_curve),
            Create(recovered_curve),
            FadeIn(badge),
            run_time=3.7,
            rate_func=linear,
        )
        self.wait(0.4)
