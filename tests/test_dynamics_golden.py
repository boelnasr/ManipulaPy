#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Golden numeric oracle for ManipulaPy dynamics - ManipulaPy.

Task 0.F of the v1.4 swappable-backend release. Before ``mass_matrix`` /
``inverse_dynamics`` / ``gravity_forces`` / ``velocity_quadratic_forces`` get
refactored onto the new backend abstraction, this test freezes their exact
float64 outputs for two bundled robots (Franka Panda, Universal Robots UR5)
across ~25 joint configurations each - all-zeros, the parsed joint-limit
poses, a near-extended pose, and seed-fixed interior samples. The refactor is
behavior-preserving only if it reproduces these goldens bit-for-bit (to the
stated tolerance).

The goldens live in:
    tests/data/dynamics_golden_panda.npz
    tests/data/dynamics_golden_ur5.npz

Each ``.npz`` stores BOTH the replayed inputs (thetas, dthetas, ddthetas, g,
Ftips) AND the recorded outputs, so the test never re-samples: there is no RNG
at test time. To regenerate (only when a change to the dynamics is intentional
and reviewed), run this module as a script:

    env -u PYTHONPATH -u ROS_DISTRO -u AMENT_PREFIX_PATH NUMBA_DISABLE_CUDA=1 \
        MPLBACKEND=Agg .venv/bin/python tests/test_dynamics_golden.py --regen

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from ManipulaPy.ManipulaPy_data import get_robot_urdf
from ManipulaPy.urdf_processor import URDFToSerialManipulator

DATA_DIR = Path(__file__).parent / "data"

# Robots frozen by this oracle.
ROBOTS: List[str] = ["panda", "ur5"]

# Gravity vector used for every configuration.
G_VEC = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# Deterministic generation seed. Only consumed at --regen time; the resulting
# samples are stored in the .npz and replayed verbatim by the test.
SEED = 20260705

# Quantities frozen per configuration. Names double as the .npz output keys.
QUANTITIES: Tuple[str, ...] = (
    "mass_matrix",
    "inverse_dynamics",
    "gravity_forces",
    "velocity_quadratic_forces",
)

# Reproducibility-safe across BLAS builds, with per-quantity absolute floors.
# RTOL=1e-7 is global. The absolute floor differs by how each quantity is
# produced:
#   - mass_matrix and gravity_forces are computed directly (J_k^T G_k J_k /
#     J_k^T F_k), so they hold to a genuinely tight ATOL=1e-9.
#   - velocity_quadratic_forces builds Christoffel terms from a central
#     finite-difference of the mass matrix at eps=1e-6 (dynamics.py:186-188).
#     The 1/(2*eps) division amplifies drift, and multithreaded BLAS reorders
#     the J_k^T G_k J_k reductions run-to-run; on UR5's small Coriolis terms
#     this reaches ~1e-9 and varies even between repeat .venv runs. A quantity
#     derived from an eps=1e-6 central difference cannot be asserted
#     reproducible to 1e-9, so it (and inverse_dynamics, which adds it in) get
#     ATOL=1e-8. RTOL=1e-7 still catches any real drift the backend refactor
#     might introduce.
RTOL = 1e-7
ATOL = {
    "mass_matrix": 1e-9,
    "gravity_forces": 1e-9,
    "velocity_quadratic_forces": 1e-8,
    "inverse_dynamics": 1e-8,
}


def _golden_path(robot: str) -> Path:
    return DATA_DIR / f"dynamics_golden_{robot}.npz"


def _build_dynamics(robot: str):
    """Construct dynamics via the public consumer path (URDF -> objects)."""
    processor = URDFToSerialManipulator(get_robot_urdf(robot), load_meshes=False)
    return processor.serial_manipulator, processor.dynamics


def _finite_limits(sm, n: int) -> np.ndarray:
    """Parsed joint limits as an ``(n, 2)`` array, falling back to +/-pi."""
    limits = getattr(sm, "joint_limits", None) or [(None, None)] * n
    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        lo, hi = limits[i] if i < len(limits) else (None, None)
        out[i, 0] = -np.pi if lo is None else float(lo)
        out[i, 1] = np.pi if hi is None else float(hi)
    return out


def _sample_configs(robot: str) -> Dict[str, np.ndarray]:
    """Sample ~25 deliberate + seeded-interior configs and their inputs.

    Called ONLY at regeneration time. Returns a dict of stored inputs; outputs
    are computed separately so the replay path and the regen path share the
    exact same input arrays.
    """
    sm, _ = _build_dynamics(robot)
    n = sm.S_list.shape[1]
    lims = _finite_limits(sm, n)
    lo, hi = lims[:, 0], lims[:, 1]

    thetas: List[np.ndarray] = []
    # Deliberate edge poses.
    thetas.append(np.zeros(n))          # home / rest
    thetas.append(lo.copy())            # lower joint-limit pose
    thetas.append(hi.copy())            # upper joint-limit pose
    thetas.append(np.full(n, 0.02))     # near-fully-extended (tiny offsets)

    # Seeded interior samples, uniform within the parsed limits, generated once.
    rng = np.random.default_rng(SEED)
    n_random = 25 - len(thetas)
    for _ in range(n_random):
        thetas.append(rng.uniform(lo, hi))

    theta_arr = np.asarray(thetas, dtype=np.float64)
    c = theta_arr.shape[0]

    # Velocities / accelerations: config 0 stays at rest (exercises the
    # zero-velocity Coriolis branch and makes ID==gravity a stored invariant);
    # the rest get seeded nonzero motion.
    dthetas = rng.uniform(-1.0, 1.0, size=(c, n))
    ddthetas = rng.uniform(-1.0, 1.0, size=(c, n))
    dthetas[0] = 0.0
    ddthetas[0] = 0.0

    # End-effector wrench: zero everywhere except one deliberate nonzero case.
    ftips = np.zeros((c, 6), dtype=np.float64)
    ftips[5] = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 0.75], dtype=np.float64)

    return {
        "thetas": theta_arr,
        "dthetas": np.asarray(dthetas, dtype=np.float64),
        "ddthetas": np.asarray(ddthetas, dtype=np.float64),
        "g": G_VEC.copy(),
        "ftips": ftips,
    }


def _compute_outputs(robot: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Evaluate the four frozen quantities for each stored input row."""
    _, dyn = _build_dynamics(robot)
    thetas = inputs["thetas"]
    dthetas = inputs["dthetas"]
    ddthetas = inputs["ddthetas"]
    g = inputs["g"]
    ftips = inputs["ftips"]
    c, n = thetas.shape

    mm = np.empty((c, n, n), dtype=np.float64)
    idn = np.empty((c, n), dtype=np.float64)
    gf = np.empty((c, n), dtype=np.float64)
    vqf = np.empty((c, n), dtype=np.float64)

    for i in range(c):
        mm[i] = dyn.mass_matrix(thetas[i])
        idn[i] = dyn.inverse_dynamics(thetas[i], dthetas[i], ddthetas[i], g, ftips[i])
        gf[i] = dyn.gravity_forces(thetas[i], g)
        vqf[i] = dyn.velocity_quadratic_forces(thetas[i], dthetas[i])

    return {
        "mass_matrix": mm,
        "inverse_dynamics": idn,
        "gravity_forces": gf,
        "velocity_quadratic_forces": vqf,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden():
    """Load both robots' golden .npz files once (inputs + outputs)."""
    out = {}
    for robot in ROBOTS:
        path = _golden_path(robot)
        assert path.exists(), (
            f"Golden fixture missing at {path}. Regenerate with:\n"
            f"    env -u PYTHONPATH -u ROS_DISTRO -u AMENT_PREFIX_PATH "
            f"NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg .venv/bin/python "
            f"{Path(__file__).name} --regen"
        )
        out[robot] = dict(np.load(path))
    return out


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_dynamics_matches_golden(golden, robot, quantity):
    """Replay stored inputs and assert live dynamics match the frozen output."""
    data = golden[robot]
    inputs = {
        "thetas": data["thetas"],
        "dthetas": data["dthetas"],
        "ddthetas": data["ddthetas"],
        "g": data["g"],
        "ftips": data["ftips"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        live = _compute_outputs(robot, inputs)[quantity]

    expected = data[quantity]
    atol = ATOL[quantity]
    assert live.shape == expected.shape, (
        f"{robot}.{quantity}: shape {live.shape} != golden {expected.shape}"
    )
    mism = ~np.isclose(live, expected, rtol=RTOL, atol=atol)
    assert not mism.any(), (
        f"{robot}.{quantity}: {int(mism.sum())} element(s) drifted beyond "
        f"rtol={RTOL}, atol={atol}. Max abs diff "
        f"{np.max(np.abs(live - expected)):.3e} at config rows "
        f"{sorted(set(np.argwhere(mism)[:, 0].tolist()))}."
    )


@pytest.mark.parametrize("robot", ROBOTS)
def test_mass_matrix_symmetric_and_pd(golden, robot):
    """Cross-check: stored mass matrices are symmetric and positive-definite."""
    mm = golden[robot]["mass_matrix"]
    # Three representative configs: rest, an interior sample, and the last row.
    for idx in (0, mm.shape[0] // 2, mm.shape[0] - 1):
        M = mm[idx]
        assert np.allclose(M, M.T, rtol=0, atol=1e-12), (
            f"{robot}: mass matrix at config {idx} is not symmetric "
            f"(max asymmetry {np.max(np.abs(M - M.T)):.3e})."
        )
        eig = np.linalg.eigvalsh(M)
        assert eig.min() > 0.0, (
            f"{robot}: mass matrix at config {idx} is not positive-definite "
            f"(min eigenvalue {eig.min():.3e})."
        )


@pytest.mark.parametrize("robot", ROBOTS)
def test_inverse_dynamics_at_rest_equals_gravity(golden, robot):
    """Cross-check: ID(theta, 0, 0, g, 0) == gravity_forces(theta, g)."""
    data = golden[robot]
    _, dyn = _build_dynamics(robot)
    thetas = data["thetas"]
    g = data["g"]
    n = thetas.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in (0, thetas.shape[0] // 2, thetas.shape[0] - 1):
            tau_rest = dyn.inverse_dynamics(
                thetas[idx], np.zeros(n), np.zeros(n), g, np.zeros(6)
            )
            grav = dyn.gravity_forces(thetas[idx], g)
            assert np.allclose(tau_rest, grav, rtol=RTOL, atol=ATOL["gravity_forces"]), (
                f"{robot}: inverse_dynamics at rest != gravity_forces at "
                f"config {idx} (max diff {np.max(np.abs(tau_rest - grav)):.3e})."
            )


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def _regen() -> None:
    """Sample inputs, compute outputs, and write both robots' golden .npz."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for robot in ROBOTS:
            inputs = _sample_configs(robot)
            outputs = _compute_outputs(robot, inputs)
            payload = {**inputs, **outputs}
            path = _golden_path(robot)
            np.savez(path, **payload)
            size_kb = path.stat().st_size / 1024.0
            n = inputs["thetas"].shape[1]
            c = inputs["thetas"].shape[0]
            print(f"Wrote {path.name}: n={n}, configs={c}, {size_kb:.1f} KB")


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "--regen" in sys.argv:
        _regen()
    else:
        print("Pass --regen to (re)write the golden fixtures.", file=sys.stderr)
        sys.exit(2)
