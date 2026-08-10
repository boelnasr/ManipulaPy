"""Deterministic Manim studies for Panda trajectory and path planning."""

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
    Circle,
    Create,
    DashedVMobject,
    Dot,
    Ellipse,
    FadeIn,
    Scene,
    Text,
    Transform,
    VGroup,
    VMobject,
    linear,
)

MANIM_DIR = Path(__file__).resolve().parent
EXAMPLES = MANIM_DIR.parents[0] / "examples"
for location in (MANIM_DIR, EXAMPLES):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from robotics_motion_studies import compute_planning_results  # noqa: E402
from scientific_scene import (  # noqa: E402
    AMBER,
    INK,
    MUTED,
    RULE,
    TEAL,
    VIOLATION,
    metric_badge,
    panda_chain,
    scientific_legend,
    study_title,
)


def _polyline(
    axes: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    stroke_width: float,
) -> VMobject:
    """Build one continuous vector path through scientific samples."""
    points = [
        axes.coords_to_point(float(x), float(y))
        for x, y in zip(x_values, y_values)
    ]
    return VMobject(color=color, stroke_width=stroke_width).set_points_as_corners(
        points
    )


def _mini_plot(
    time: np.ndarray,
    cubic: np.ndarray,
    quintic: np.ndarray,
    label: str,
) -> VGroup:
    bound = max(1e-4, float(np.max(np.abs(np.concatenate((cubic, quintic))))))
    axes = Axes(
        x_range=[0.0, 4.0, 2.0],
        y_range=[-bound, bound, bound],
        x_length=3.0,
        y_length=1.25,
        tips=False,
        axis_config={"color": RULE, "stroke_width": 1.1, "font_size": 11},
    )
    cubic_line = DashedVMobject(
        _polyline(axes, time, cubic, AMBER, 2.0), num_dashes=30
    )
    quintic_line = _polyline(axes, time, quintic, TEAL, 2.4)
    name = Text(label, color=MUTED, font_size=13).next_to(axes, UP, buff=0.05)
    return VGroup(axes, cubic_line, quintic_line, name)


class PandaTimeScalingComparison(Scene):
    """Compare cubic and quintic endpoint behavior on the same joint motion."""

    def construct(self) -> None:
        result = compute_planning_results()
        joint = int(np.ptp(result.quintic.positions, axis=0).argmax())
        plots = VGroup(
            _mini_plot(
                result.time,
                result.cubic.positions[:, joint],
                result.quintic.positions[:, joint],
                "position [rad]",
            ),
            _mini_plot(
                result.time,
                result.cubic.velocities[:, joint],
                result.quintic.velocities[:, joint],
                "velocity [rad/s]",
            ),
            _mini_plot(
                result.time,
                result.cubic.accelerations[:, joint],
                result.quintic.accelerations[:, joint],
                "acceleration [rad/s²]",
            ),
            _mini_plot(
                result.time,
                result.cubic.jerk[:, joint],
                result.quintic.jerk[:, joint],
                "jerk [rad/s³]",
            ),
        )
        top = VGroup(plots[0], plots[1]).arrange(RIGHT, buff=0.30)
        bottom = VGroup(plots[2], plots[3]).arrange(RIGHT, buff=0.30)
        grid = VGroup(top, bottom).arrange(DOWN, buff=0.30)
        grid.shift(RIGHT * 2.5 + DOWN * 0.45)
        title = study_title(
            "Time scaling changes endpoint smoothness",
            f"Joint {joint + 1} · identical Panda endpoints and four-second duration",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.quintic.positions[0])
        final_chain = panda_chain(result.quintic.positions[-1])
        legend = scientific_legend(
            (("cubic", AMBER, True), ("quintic", TEAL, False))
        ).scale(0.9).next_to(grid, DOWN, buff=0.10).align_to(grid, RIGHT)
        curves = VGroup(*(plot[1:] for plot in plots))

        self.add(title, chain, *(plot[0] for plot in plots), legend)
        self.play(
            Create(curves),
            Transform(chain, final_chain),
            run_time=3.7,
            rate_func=linear,
        )
        self.wait(0.4)


class PandaInterpolationDomains(Scene):
    """Contrast a joint-space tool path with a Cartesian straight line."""

    def construct(self) -> None:
        result = compute_planning_results()
        axes = Axes(
            x_range=[0.40, 0.60, 0.05],
            y_range=[-0.15, 0.10, 0.05],
            x_length=6.5,
            y_length=3.5,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 14},
        ).shift(RIGHT * 2.35 + DOWN * 0.35)
        joint_curve = DashedVMobject(
            _polyline(
                axes,
                result.joint_tool_path[:, 0],
                result.joint_tool_path[:, 1],
                AMBER,
                3,
            ),
            num_dashes=36,
        )
        cartesian_curve = _polyline(
            axes,
            result.cartesian_tool_path[:, 0],
            result.cartesian_tool_path[:, 1],
            TEAL,
            3,
        )
        endpoints = VGroup(
            Dot(axes.coords_to_point(*result.joint_tool_path[0, :2]), color=INK),
            Dot(axes.coords_to_point(*result.joint_tool_path[-1, :2]), color=INK),
        )
        title = study_title(
            "The interpolation domain shapes the tool path",
            "Same Panda start and goal · XY projection · orientation interpolated",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.quintic.positions[0])
        final_chain = panda_chain(result.quintic.positions[-1])
        legend = scientific_legend(
            (("joint space", AMBER, True), ("Cartesian", TEAL, False))
        ).next_to(axes, UP, buff=0.12).align_to(axes, RIGHT)
        labels = VGroup(
            Text("tool x [m]", color=MUTED, font_size=14).next_to(
                axes.x_axis, DOWN, buff=0.12
            ),
            Text("tool y [m]", color=MUTED, font_size=14).rotate(np.pi / 2).next_to(
                axes.y_axis, LEFT, buff=0.12
            ),
        )

        self.add(title, chain, axes, labels, legend, endpoints)
        self.play(
            Create(joint_curve),
            Create(cartesian_curve),
            Transform(chain, final_chain),
            run_time=3.7,
            rate_func=linear,
        )
        self.wait(0.4)


class PandaCollisionCorrection(Scene):
    """Show public potential-field correction around a joint-space obstacle."""

    def construct(self) -> None:
        result = compute_planning_results()
        projection = (1, 6)
        axes = Axes(
            x_range=[-0.4, 0.4, 0.2],
            y_range=[-0.6, 1.0, 0.4],
            x_length=6.0,
            y_length=4.0,
            tips=False,
            axis_config={"color": RULE, "stroke_width": 1.3, "font_size": 14},
        ).shift(RIGHT * 2.4 + DOWN * 0.35)
        nominal = DashedVMobject(
            _polyline(
                axes,
                result.nominal_path[:, projection[0]],
                result.nominal_path[:, projection[1]],
                MUTED,
                2.2,
            ),
            num_dashes=24,
        )
        corrected_dots = VGroup(
            *(
                Dot(
                    axes.coords_to_point(*point[list(projection)]),
                    color=TEAL,
                    radius=0.035,
                )
                for point in result.corrected_path
            )
        )
        obstacle = Ellipse(
            width=2.0 * 0.20 * axes.x_axis.unit_size,
            height=2.0 * 0.20 * axes.y_axis.unit_size,
            color=VIOLATION,
            fill_color=VIOLATION,
            fill_opacity=0.12,
            stroke_width=2,
        ).move_to(axes.coords_to_point(*result.obstacle_q[list(projection)]))
        obstacle_label = Text(
            "waypoint exclusion · 0.20 rad",
            color=VIOLATION,
            font_size=14,
        ).next_to(obstacle, UP, buff=0.08)
        title = study_title(
            "Potential fields shift colliding waypoints",
            "q2–q7 projection · sampled configurations, not workspace geometry",
        ).to_edge(UP, buff=0.28)
        chain = panda_chain(result.nominal_path[0])
        final_chain = panda_chain(result.corrected_path[-1])
        badge = metric_badge(
            "minimum waypoint clearance",
            f"{result.minimum_joint_clearance:.3f}",
            "rad",
        ).next_to(axes, DOWN, buff=0.10).align_to(axes, RIGHT)
        legend = scientific_legend(
            (("nominal interpolation", MUTED, True), ("corrected samples", TEAL, False))
        ).next_to(axes, UP, buff=0.12).align_to(axes, RIGHT)
        labels = VGroup(
            Text("joint 2 [rad]", color=MUTED, font_size=14).next_to(
                axes.x_axis, DOWN, buff=0.12
            ),
            Text("joint 7 [rad]", color=MUTED, font_size=14)
            .rotate(np.pi / 2)
            .next_to(axes.y_axis, LEFT, buff=0.12),
        )

        self.add(
            title,
            chain,
            axes,
            labels,
            nominal,
            obstacle,
            obstacle_label,
            legend,
        )
        self.play(
            FadeIn(corrected_dots),
            Transform(chain, final_chain),
            run_time=3.6,
            rate_func=linear,
        )
        self.play(
            FadeIn(badge),
            run_time=0.35,
            rate_func=linear,
        )
        self.wait(0.35)
