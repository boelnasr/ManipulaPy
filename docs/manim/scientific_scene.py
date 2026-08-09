"""Shared Manim primitives for ManipulaPy's scientific motion studies."""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    DashedLine,
    Dot,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Text,
    VGroup,
    linear,
)


INK = "#E6ECEE"
MUTED = "#94A3A8"
TEAL = "#63C9BC"
AMBER = "#E6B566"
VIOLATION = "#E27676"
RULE = "#405158"
PANEL = "#172126"


def study_title(title: str, subtitle: str) -> VGroup:
    """Return the shared restrained title lockup and centered divider."""
    heading = Text(title, color=INK, font_size=31, weight="MEDIUM")
    detail = Text(subtitle, color=MUTED, font_size=17)
    lockup = VGroup(heading, detail).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    rule = Line(LEFT * 6.7, RIGHT * 6.7, color=RULE, stroke_width=1)
    rule.next_to(lockup, DOWN, buff=0.18).set_x(0.0)
    return VGroup(lockup, rule)


def panel_frame(width: float, height: float) -> RoundedRectangle:
    """Return a stable plot surface compatible with both documentation themes."""
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        color=RULE,
        stroke_width=1.2,
        fill_color=PANEL,
        fill_opacity=1.0,
    )


def _normalized_chain(configuration: np.ndarray) -> np.ndarray:
    headings = np.cumsum(np.pi / 2.0 + np.asarray(configuration) * 0.34)
    steps = np.column_stack((np.cos(headings), np.sin(headings)))
    points = np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))
    span = np.ptp(points, axis=0)
    return points * (3.35 / max(float(span.max()), 1.0))


def panda_chain(configuration: np.ndarray, center_x: float = -3.15) -> VGroup:
    """Draw a seven-joint Panda schematic whose bends encode joint values."""
    points = _normalized_chain(np.asarray(configuration, dtype=np.float64))
    points -= (points.min(axis=0) + points.max(axis=0)) / 2.0
    points[:, 0] += center_x
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
    endpoint = np.array([*points[-1], 0.0])
    tool = VGroup(
        Arrow(endpoint, endpoint + RIGHT * 0.48, buff=0, color=TEAL, stroke_width=3),
        Arrow(endpoint, endpoint + UP * 0.48, buff=0, color=INK, stroke_width=3),
        Text("tool", color=MUTED, font_size=14).next_to(endpoint, DOWN, buff=0.15),
    )
    return VGroup(links, joints, tool)


def time_cursor(axes, time_value: float, color: str = TEAL) -> DashedLine:
    """Return a vertical cursor spanning an Axes object's declared y-range."""
    y_min, y_max = float(axes.y_range[0]), float(axes.y_range[1])
    return DashedLine(
        axes.coords_to_point(time_value, y_min),
        axes.coords_to_point(time_value, y_max),
        color=color,
        stroke_width=2,
        dash_length=0.08,
    )


def scientific_legend(
    items: tuple[tuple[str, str, bool], ...], font_size: int = 15
) -> VGroup:
    """Build a line-style-and-color legend so color is never the sole signal."""
    rows = VGroup()
    for label, color, dashed in items:
        line_type = DashedLine if dashed else Line
        sample = line_type(
            LEFT * 0.18, RIGHT * 0.18, color=color, stroke_width=3
        )
        row = VGroup(sample, Text(label, color=INK, font_size=font_size))
        rows.add(row.arrange(RIGHT, buff=0.12))
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.10)


def metric_badge(
    label: str,
    value: str,
    unit: str = "",
    status: str = "normal",
) -> VGroup:
    """Return a compact metric value with a semantic status marker."""
    colors = {"normal": TEAL, "warning": AMBER, "violation": VIOLATION}
    if status not in colors:
        raise ValueError(f"unknown metric status: {status}")
    marker = Dot(radius=0.055, color=colors[status])
    label_text = Text(label, color=MUTED, font_size=14)
    value_text = Text(
        f"{value}{(' ' + unit) if unit else ''}", color=INK, font_size=18
    )
    text = VGroup(label_text, value_text).arrange(
        DOWN, aligned_edge=LEFT, buff=0.03
    )
    content = VGroup(marker, text).arrange(RIGHT, buff=0.12)
    box = RoundedRectangle(
        width=max(1.65, content.width + 0.30),
        height=max(0.62, content.height + 0.22),
        corner_radius=0.08,
        color=RULE,
        fill_color=PANEL,
        fill_opacity=1.0,
        stroke_width=1,
    ).move_to(content)
    return VGroup(box, content)


def play_linear(scene, *animations, run_time: float = 1.0) -> None:
    """Play scientific time evolution without decorative easing."""
    scene.play(*animations, run_time=run_time, rate_func=linear)
