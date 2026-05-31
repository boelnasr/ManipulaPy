#!/usr/bin/env python3
"""
Unit tests for the potential_field module (attractive/repulsive potentials and gradients).

These tests are CPU-only and avoid any GPU/CUDA dependencies.
"""

import numpy as np
import pytest

from ManipulaPy.potential_field import PotentialField


def finite_difference_gradient(
    pf: PotentialField, q: np.ndarray, q_goal: np.ndarray, obstacles
) -> np.ndarray:
    """Numerical gradient of total potential (attractive + repulsive) via central difference."""
    eps = 1e-6
    grad = np.zeros_like(q, dtype=float)

    def total_potential(x: np.ndarray) -> float:
        """Sum of attractive and repulsive potentials at configuration x."""
        return pf.compute_attractive_potential(
            x, q_goal
        ) + pf.compute_repulsive_potential(x, obstacles)

    for i in range(q.size):
        e_i = np.zeros_like(q)
        e_i[i] = 1.0
        grad[i] = (total_potential(q + eps * e_i) - total_potential(q - eps * e_i)) / (
            2 * eps
        )
    return grad


def test_attractive_potential_zero_at_goal() -> None:
    """Attractive potential is zero when the configuration equals the goal."""
    pf = PotentialField(attractive_gain=2.0)
    q = np.array([1.0, -1.0, 0.5])
    q_goal = q.copy()
    assert pf.compute_attractive_potential(q, q_goal) == pytest.approx(0.0)


def test_attractive_potential_matches_expected_value() -> None:
    """Attractive potential equals 0.5 * gain * squared distance to goal."""
    pf = PotentialField(attractive_gain=1.5)
    q = np.array([1.0, 2.0])
    q_goal = np.array([0.5, -1.0])
    # 0.5 * k * ||q - q_goal||^2
    expected = 0.5 * 1.5 * np.sum((q - q_goal) ** 2)
    assert pf.compute_attractive_potential(q, q_goal) == pytest.approx(expected)


def test_repulsive_potential_zero_outside_influence() -> None:
    """Repulsive potential is zero for obstacles beyond the influence distance."""
    pf = PotentialField(repulsive_gain=10.0, influence_distance=0.5)
    q = np.array([0.0, 0.0])
    obstacles = [np.array([2.0, 0.0])]
    assert pf.compute_repulsive_potential(q, obstacles) == pytest.approx(0.0)


def test_repulsive_potential_positive_inside_influence() -> None:
    """Repulsive potential matches the expected value for an obstacle inside influence."""
    pf = PotentialField(repulsive_gain=2.0, influence_distance=1.0)
    q = np.array([0.0, 0.0])
    obstacles = [np.array([0.5, 0.0])]
    d = np.linalg.norm(q - obstacles[0])
    # From compute_repulsive_potential: 10 * (2 * gain * (1/d - 1/d0)^2)
    expected = 10 * (
        2 * pf.repulsive_gain * (1.0 / d - 1.0 / pf.influence_distance) ** 2
    )
    assert pf.compute_repulsive_potential(q, obstacles) == pytest.approx(expected)


def test_repulsive_potential_multiple_obstacles_accumulates() -> None:
    """Repulsive potential sums contributions across multiple obstacles."""
    pf = PotentialField(repulsive_gain=1.0, influence_distance=1.0)
    q = np.array([0.0, 0.0])
    obstacles = [np.array([0.5, 0.0]), np.array([-0.5, 0.0])]
    total = pf.compute_repulsive_potential(q, obstacles)
    single = pf.compute_repulsive_potential(q, [obstacles[0]])
    assert total == pytest.approx(2 * single)


def test_gradient_matches_finite_difference() -> None:
    """Analytic gradient matches the numerical finite-difference gradient."""
    pf = PotentialField(attractive_gain=1.2, repulsive_gain=0.8, influence_distance=1.0)
    q = np.array([0.2, 0.1])
    q_goal = np.array([0.5, -0.1])
    obstacles = [np.array([0.6, 0.1])]  # within influence distance

    analytic = pf.compute_gradient(q, q_goal, obstacles)
    numeric = finite_difference_gradient(pf, q, q_goal, obstacles)

    assert analytic == pytest.approx(numeric, rel=1e-3, abs=1e-5)


def test_gradient_without_obstacles_is_purely_attractive() -> None:
    """With no obstacles the gradient reduces to the attractive term."""
    pf = PotentialField(attractive_gain=2.0, repulsive_gain=5.0, influence_distance=0.5)
    q = np.array([0.25, -0.5, 0.75])
    q_goal = np.zeros_like(q)
    obstacles = []

    grad = pf.compute_gradient(q, q_goal, obstacles)
    expected = pf.attractive_gain * (q - q_goal)

    assert grad == pytest.approx(expected)


def test_gradient_repulsive_term_zero_outside_influence() -> None:
    """Repulsive gradient term is zero for obstacles outside the influence distance."""
    pf = PotentialField(attractive_gain=0.0, repulsive_gain=5.0, influence_distance=0.5)
    q = np.array([0.0, 0.0])
    obstacles = [np.array([1.0, 0.0])]  # outside influence
    grad = pf.compute_gradient(q, np.zeros_like(q), obstacles)
    assert grad == pytest.approx(np.zeros_like(q))
