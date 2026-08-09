"""Deterministic Manim scenes for the Panda kinematics tutorial.

This module belongs to the render-only documentation environment described by
``docs/manim/requirements.txt``.  It is deliberately not imported by Sphinx or
the regular test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    LaggedStart,
    Line,
    MathTex,
    Matrix,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
    smooth,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from kinematics_tutorial import (  # noqa: E402
    HOME,
    JOINT_RATES,
    TARGET,
    compute_ik_trace,
    compute_tutorial_results,
)

INK = "#E6ECEE"
MUTED = "#94A3A8"
TEAL = "#63C9BC"
RULE = "#405158"
PANEL = "#172126"
RESIDUAL_DISPLAY_FLOOR = 1e-9
RESIDUAL_TOLERANCE = 1e-5
TOLERANCE_LOG10 = -5.0
RESIDUAL_DECADES = (0, -3, -5, -7, -9)


def _title(text: str, subtitle: str) -> VGroup:
    """Return the common restrained title lockup."""
    heading = Text(text, color=INK, font_size=31, weight="MEDIUM")
    detail = Text(subtitle, color=MUTED, font_size=17)
    return VGroup(heading, detail).arrange(DOWN, aligned_edge=LEFT, buff=0.08)


def _rule_below(title: VGroup) -> Line:
    """Place a full-width rule below a title while keeping the frame centered."""
    rule = Line(LEFT * 6.7, RIGHT * 6.7, color=RULE, stroke_width=1)
    rule.next_to(title, DOWN, buff=0.18)
    rule.set_x(0.0)
    return rule


def _normalized_chain(configuration: np.ndarray) -> np.ndarray:
    """Map seven measured joint angles to a normalized schematic chain."""
    headings = np.cumsum(np.pi / 2.0 + np.asarray(configuration) * 0.34)
    steps = np.column_stack((np.cos(headings), np.sin(headings)))
    points = np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))
    span = np.ptp(points, axis=0)
    scale = 3.35 / max(float(span.max()), 1.0)
    return points * scale


def _chain(configuration: np.ndarray) -> VGroup:
    """Draw a base-to-tool schematic whose bends encode the joint values."""
    points = _normalized_chain(configuration)
    points -= (points.min(axis=0) + points.max(axis=0)) / 2.0
    points[:, 0] -= 3.15
    links = VGroup(
        *(
            Line(
                [*points[index], 0.0],
                [*points[index + 1], 0.0],
                color=RULE,
                stroke_width=5,
            )
            for index in range(7)
        )
    )
    joints = VGroup()
    for index, point in enumerate(points[1:], start=1):
        marker = Circle(radius=0.13, color=INK, stroke_width=2).set_fill(
            PANEL, opacity=1
        )
        marker.move_to([*point, 0.0])
        number = MathTex(str(index), color=INK, font_size=17).move_to(marker)
        joints.add(VGroup(marker, number))
    base = VGroup(
        Rectangle(width=0.65, height=0.16, color=RULE, stroke_width=2),
        Text("base", color=MUTED, font_size=14),
    ).arrange(DOWN, buff=0.08)
    base.move_to([*points[0], 0.0]).shift(DOWN * 0.2)

    endpoint = np.array([*points[-1], 0.0])
    y_projection = LEFT * 0.38 + DOWN * 0.26
    triad = VGroup(
        Arrow(endpoint, endpoint + RIGHT * 0.55, buff=0, color=TEAL, stroke_width=3),
        Arrow(
            endpoint,
            endpoint + y_projection,
            buff=0,
            color=MUTED,
            stroke_width=2.5,
        ),
        Arrow(endpoint, endpoint + UP * 0.55, buff=0, color=INK, stroke_width=3),
        MathTex("x", color=TEAL, font_size=18).next_to(
            endpoint + RIGHT * 0.55, RIGHT, buff=0.04
        ),
        MathTex("y", color=MUTED, font_size=18).next_to(
            endpoint + y_projection, LEFT, buff=0.04
        ),
        MathTex("z", color=INK, font_size=18).next_to(
            endpoint + UP * 0.55, UP, buff=0.04
        ),
        Text("tool frame", color=MUTED, font_size=14).next_to(
            endpoint, DOWN + RIGHT, buff=0.2
        ),
    )
    return VGroup(links, joints, base, triad)


def _transform_matrix(entries: list[list[str]]) -> VGroup:
    """Build a legible 4 by 4 transform with rotation/translation blocks."""
    matrix = Matrix(
        entries,
        element_to_mobject_config={"color": INK, "font_size": 19},
        h_buff=0.62,
        v_buff=0.48,
        bracket_h_buff=0.10,
        bracket_v_buff=0.10,
    )
    matrix.set_column_colors(INK, INK, INK, TEAL)
    rotation = Rectangle(width=2.03, height=1.56, color=RULE, stroke_width=1)
    translation = Rectangle(width=0.64, height=1.56, color=TEAL, stroke_width=1)
    homogeneous = Rectangle(width=2.68, height=0.52, color=RULE, stroke_width=1)
    rotation.move_to(matrix.get_entries()[5])
    translation.move_to(matrix.get_entries()[7])
    homogeneous.move_to(matrix.get_entries()[14])
    return VGroup(matrix, rotation, translation, homogeneous)


class PandaForwardKinematics(Scene):
    """Reveal the seven-joint chain and evaluate its tool transform."""

    def construct(self) -> None:
        result = compute_tutorial_results()
        title = _title(
            "Forward kinematics",
            "Franka Panda  ·  base to tool  ·  seven revolute joints",
        ).to_corner(UP + LEFT, buff=0.38)
        rule = _rule_below(title)
        chain = _chain(HOME)
        joint_change = VGroup(
            Text("home to target joint change", color=MUTED, font_size=14),
            MathTex(
                f"{np.linalg.norm(TARGET - HOME):.3f}\\,\\mathrm{{rad}}",
                color=INK,
                font_size=18,
            ),
        ).arrange(RIGHT, buff=0.12)
        joint_change.to_corner(DOWN + LEFT, buff=0.38)

        label = MathTex(r"{}^{0}T_{7}", color=TEAL, font_size=28)
        symbolic = _transform_matrix(
            [
                [r"r_{11}", r"r_{12}", r"r_{13}", "p_x"],
                [r"r_{21}", r"r_{22}", r"r_{23}", "p_y"],
                [r"r_{31}", r"r_{32}", r"r_{33}", "p_z"],
                ["0", "0", "0", "1"],
            ]
        )
        numeric = _transform_matrix(
            [[f"{value:+.3f}" for value in row] for row in result.pose]
        )
        transform_panel = VGroup(label, symbolic).arrange(DOWN, buff=0.2)
        transform_panel.scale(0.92).to_edge(RIGHT, buff=0.48).shift(DOWN * 0.1)
        numeric.scale(0.92).move_to(symbolic)
        units = Text("translation column in metres", color=MUTED, font_size=15)
        units.next_to(transform_panel, DOWN, buff=0.18)

        self.play(FadeIn(title), Create(rule), run_time=0.7, rate_func=smooth)
        self.play(
            LaggedStart(
                Create(chain[0]),
                FadeIn(chain[1]),
                FadeIn(chain[2]),
                FadeIn(chain[3]),
                FadeIn(joint_change),
                lag_ratio=0.14,
            ),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(Write(label), FadeIn(symbolic), FadeIn(units), run_time=0.8)
        self.play(Transform(symbolic, numeric), run_time=1.2, rate_func=smooth)
        self.wait(0.7)


def _value_grid(values: np.ndarray, width: float, height: float) -> VGroup:
    """Return a heat/value grid using magnitudes and the shared teal channel."""
    array = np.asarray(values, dtype=float)
    magnitude = np.abs(array)
    peak = max(float(magnitude.max()), np.finfo(float).eps)
    rows, columns = array.shape
    grid = VGroup()
    for row in range(rows):
        for column in range(columns):
            opacity = 0.05 + 0.32 * float(magnitude[row, column] / peak)
            cell = Rectangle(
                width=width,
                height=height,
                color=RULE,
                stroke_width=0.7,
                fill_color=TEAL,
                fill_opacity=opacity,
            )
            value = MathTex(
                f"{array[row, column]:+.2f}", color=INK, font_size=15
            )
            entry = VGroup(cell, value)
            entry.move_to(
                [
                    (column - (columns - 1) / 2.0) * width,
                    ((rows - 1) / 2.0 - row) * height,
                    0.0,
                ]
            )
            grid.add(entry)
    return grid


def _value_column(values: np.ndarray, font_size: int = 18) -> VGroup:
    entries = VGroup(
        *(MathTex(f"{value:+.3f}", color=INK, font_size=font_size) for value in values)
    ).arrange(DOWN, buff=0.12)
    bracket = Rectangle(
        width=entries.width + 0.3,
        height=entries.height + 0.18,
        color=RULE,
        stroke_width=1,
    )
    return VGroup(bracket, entries)


class PandaJacobianVelocity(Scene):
    """Show measured joint rates mapped through the computed Jacobian."""

    def construct(self) -> None:
        result = compute_tutorial_results()
        title = _title(
            "Jacobian velocity map",
            "Seven joint rates map to one spatial tool twist",
        ).to_corner(UP + LEFT, buff=0.38)
        rule = _rule_below(title)

        rates = _value_column(JOINT_RATES)
        rate_label = VGroup(
            MathTex(r"\dot{q}", color=TEAL, font_size=28),
            Text("joint rates [rad/s]", color=MUTED, font_size=15),
        ).arrange(DOWN, buff=0.08)
        rate_group = VGroup(rate_label, rates).arrange(DOWN, buff=0.18)

        grid = _value_grid(result.jacobian, width=0.58, height=0.42)
        matrix_label = VGroup(
            MathTex(r"J_s(q)", color=TEAL, font_size=28),
            Text("6 × 7 space Jacobian", color=MUTED, font_size=15),
        ).arrange(DOWN, buff=0.08)
        matrix_group = VGroup(matrix_label, grid).arrange(DOWN, buff=0.18)

        angular = _value_column(result.twist[:3])
        linear_values = _value_column(result.twist[3:])
        output_label = MathTex(r"V_s=J_s(q)\dot{q}", color=TEAL, font_size=25)
        angular_label = Text("angular [rad/s]", color=MUTED, font_size=15)
        linear_label = Text("linear [m/s]", color=MUTED, font_size=15)
        outputs = VGroup(
            output_label,
            angular_label,
            angular,
            linear_label,
            linear_values,
        ).arrange(DOWN, buff=0.12)

        equation = VGroup(rate_group, matrix_group, outputs).arrange(
            RIGHT, buff=0.58, aligned_edge=DOWN
        )
        equation.scale(0.91).next_to(rule, DOWN, buff=0.34)
        equation.set_x(0.0)

        self.play(FadeIn(title), Create(rule), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(rate_group), run_time=0.6, rate_func=smooth)
        self.play(
            LaggedStart(*(FadeIn(cell) for cell in grid), lag_ratio=0.015),
            FadeIn(matrix_label),
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(FadeIn(outputs), run_time=0.8, rate_func=smooth)
        self.wait(0.7)


def _log10_residuals_for_display(residuals: np.ndarray) -> np.ndarray:
    """Map real residuals to log10, clamping only below the display floor."""
    return np.log10(np.maximum(residuals, RESIDUAL_DISPLAY_FLOOR))


def _residual_axis(
    budgets: np.ndarray,
    residuals: np.ndarray,
    label: str,
    units: str,
    line_color: str,
) -> tuple[VGroup, Axes, VGroup]:
    display_values = _log10_residuals_for_display(residuals)
    axes = Axes(
        x_range=[float(budgets[0]), float(budgets[-1]), 1],
        y_range=[-9.0, 0.0, 1.0],
        x_length=8.7,
        y_length=1.65,
        axis_config={"color": RULE, "stroke_width": 1},
        x_axis_config={"include_numbers": True, "font_size": 16},
        y_axis_config={"include_numbers": False},
        tips=False,
    )
    decade_labels = VGroup()
    for decade in RESIDUAL_DECADES:
        tick = Line(
            axes.c2p(float(budgets[0]), decade),
            axes.c2p(float(budgets[0]) + 0.08, decade),
            color=RULE,
            stroke_width=1,
        )
        decade_label = MathTex(
            rf"10^{{{decade}}}", color=MUTED, font_size=13
        ).next_to(tick, LEFT, buff=0.06)
        decade_labels.add(tick, decade_label)
    curve = axes.plot_line_graph(
        x_values=budgets,
        y_values=display_values,
        line_color=line_color,
        stroke_width=3,
        add_vertex_dots=True,
        vertex_dot_radius=0.04,
        vertex_dot_style={"fill_color": line_color, "stroke_color": line_color},
    )
    heading = VGroup(
        Text(label, color=INK, font_size=17),
        Text(units, color=MUTED, font_size=14),
    ).arrange(RIGHT, buff=0.12)
    heading.next_to(axes, UP, aligned_edge=LEFT, buff=0.06)
    tolerance = DashedLine(
        axes.c2p(float(budgets[0]), TOLERANCE_LOG10),
        axes.c2p(float(budgets[-1]), TOLERANCE_LOG10),
        color=TEAL,
        stroke_width=1.5,
        dash_length=0.08,
    )
    tolerance_label = MathTex(
        r"10^{-5}\ \mathrm{tolerance}", color=TEAL, font_size=14
    ).next_to(tolerance, RIGHT, buff=0.08)
    return (
        VGroup(
            axes,
            decade_labels,
            heading,
            tolerance,
            tolerance_label,
            curve,
        ),
        axes,
        curve,
    )


class PandaIKConvergence(Scene):
    """Plot real solver residuals in explicit log10 display coordinates."""

    def construct(self) -> None:
        budgets, translation, rotation = compute_ik_trace()
        title = _title(
            "Inverse kinematics convergence",
            "Recorded residuals  ·  log10 scale  ·  display floor 10⁻⁹",
        ).to_corner(UP + LEFT, buff=0.38)
        rule = _rule_below(title)

        translation_group, translation_axes, translation_curve = _residual_axis(
            budgets, translation, "translation residual", "[m]", TEAL
        )
        rotation_group, rotation_axes, rotation_curve = _residual_axis(
            budgets, rotation, "rotation residual", "[rad]", INK
        )
        charts = VGroup(translation_group, rotation_group).arrange(
            DOWN, buff=0.38, aligned_edge=LEFT
        )
        charts.scale(0.88).next_to(rule, DOWN, buff=0.28)
        charts.set_x(0.0)
        x_label = Text("iteration budget", color=MUTED, font_size=15).next_to(
            charts, DOWN, buff=0.08
        )

        translation_log = _log10_residuals_for_display(translation)
        rotation_log = _log10_residuals_for_display(rotation)
        solved = np.flatnonzero(
            (translation <= RESIDUAL_TOLERANCE)
            & (rotation <= RESIDUAL_TOLERANCE)
        )
        solved_marks = VGroup()
        if solved.size:
            index = int(solved[0])
            solved_marks.add(
                Dot(
                    translation_axes.c2p(budgets[index], translation_log[index]),
                    radius=0.08,
                    color=TEAL,
                ),
                Dot(
                    rotation_axes.c2p(budgets[index], rotation_log[index]),
                    radius=0.08,
                    color=TEAL,
                ),
            )
            solved_label = VGroup(
                Text("solved tolerance", color=MUTED, font_size=14),
                MathTex(r"\leq 10^{-5}", color=TEAL, font_size=18),
            ).arrange(RIGHT, buff=0.08)
            solved_label.next_to(rule, DOWN, aligned_edge=RIGHT, buff=0.12)
            solved_marks.add(solved_label)

        self.play(FadeIn(title), Create(rule), run_time=0.7, rate_func=smooth)
        self.play(
            Create(translation_axes),
            Create(rotation_axes),
            FadeIn(translation_group[1:5]),
            FadeIn(rotation_group[1:5]),
            FadeIn(x_label),
            run_time=0.9,
            rate_func=smooth,
        )
        self.play(
            Create(translation_curve),
            Create(rotation_curve),
            run_time=1.5,
            rate_func=smooth,
        )
        if solved_marks:
            self.play(FadeIn(solved_marks), run_time=0.6, rate_func=smooth)
        self.wait(0.7)
