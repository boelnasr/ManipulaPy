#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Public-API return-contract freeze test - ManipulaPy.

This is a HARD INVARIANT for the v1.4 swappable-backend refactor. Under the
DEFAULT (NumPy) backend, every public entry point below must keep both its
signature AND its return type/shape/dtype. The real failure mode the refactor
must not introduce is return-type/dtype drift (e.g. a torch.Tensor or a
float32 array leaking out where consumers - notably the downstream ROS2
wrapper - expect a float64 np.ndarray). Signatures alone are not enough.

The golden contract lives in tests/data/api_contract_golden.json. To regenerate
it (only when a change is intentional), run this module as a script:

    NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg .venv/bin/python tests/test_public_api_freeze.py --regen

The test rebuilds the live contract with the same deterministic fixtures and
asserts equality per symbol, printing a readable unified diff naming the symbol
and which facet (signature / return type / shape / dtype) drifted.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import difflib
import inspect
import json
import os
import sys
import warnings
from math import pi
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
from ManipulaPy.singularity import Singularity
from ManipulaPy.trac_ik import TracIKSolver, trac_ik_solve

GOLDEN_PATH = Path(__file__).parent / "data" / "api_contract_golden.json"


# ---------------------------------------------------------------------------
# Contract description helpers
# ---------------------------------------------------------------------------


def _fqcn(value: Any) -> str:
    """Fully-qualified class name of a value (e.g. ``numpy.ndarray``)."""
    tp = type(value)
    module = tp.__module__
    if module == "builtins":
        return tp.__qualname__
    return f"{module}.{tp.__qualname__}"


def describe(value: Any) -> Dict[str, Any]:
    """Describe a return value's type and (for arrays) shape/dtype.

    Recurses into tuples, lists and dicts so a dtype drift buried inside a
    trajectory dict or an IK result tuple is still captured.
    """
    entry: Dict[str, Any] = {"type": _fqcn(value)}
    if isinstance(value, np.ndarray):
        entry["shape"] = list(value.shape)
        entry["dtype"] = str(value.dtype)
    elif isinstance(value, np.generic):
        # NumPy scalar (e.g. np.float64, np.bool_) - dtype is the contract.
        entry["dtype"] = str(value.dtype)
    elif isinstance(value, (tuple, list)):
        entry["elements"] = [describe(v) for v in value]
    elif isinstance(value, dict):
        entry["items"] = {str(k): describe(v) for k, v in sorted(value.items())}
    return entry


def _signature_str(fn: Callable[..., Any]) -> str:
    """Stable string form of a callable's signature."""
    return str(inspect.signature(fn))


# ---------------------------------------------------------------------------
# Deterministic robot fixtures (mirror test_kinematics.py / test_dynamics.py)
# ---------------------------------------------------------------------------


def _screw_axes() -> np.ndarray:
    """Space-frame screw axes for the shared 6-DOF test robot, shape (6, 6)."""
    return np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, -0.089, 0, 0],
            [0, -1, 0, -0.089, 0, 0.425],
            [0, -1, 0, -0.089, 0, 0.817],
            [1, 0, 0, 0, 0.109, 0],
            [0, -1, 0, -0.089, 0, 0.817],
        ],
        dtype=float,
    ).T


def _home_pose() -> np.ndarray:
    """Home configuration transform (4, 4) for the shared test robot."""
    return np.array(
        [[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]], dtype=float
    )


def build_serial_manipulator() -> SerialManipulator:
    """Construct the shared 6-DOF SerialManipulator fixture."""
    s_list = _screw_axes()
    return SerialManipulator(
        M_list=_home_pose(),
        omega_list=s_list[:3, :],
        S_list=s_list,
        B_list=s_list.copy(),
        joint_limits=[(-pi, pi)] * 6,
    )


def build_dynamics() -> ManipulatorDynamics:
    """Construct the shared 6-DOF ManipulatorDynamics fixture."""
    s_list = _screw_axes()
    glist = np.stack(
        [np.eye(6, dtype=float) * (1.0 + i * 0.1) for i in range(6)], axis=0
    )
    return ManipulatorDynamics(
        M_list=_home_pose(),
        omega_list=None,
        r_list=None,
        b_list=None,
        S_list=s_list,
        B_list=s_list.copy(),
        Glist=glist,
    )


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def build_contract() -> Dict[str, Dict[str, Any]]:
    """Call every scoped public symbol once and record its frozen facets.

    All inputs are deterministic (fixed joint configurations, no RNG dependence
    in the recorded facets) and every call is routed through the default NumPy
    CPU path.
    """
    theta = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=float)
    theta2 = theta + 0.2
    dtheta = np.full(6, 0.05, dtype=float)
    ddtheta = np.full(6, 0.01, dtype=float)
    g_vec = np.array([0.0, 0.0, -9.81], dtype=float)
    ftip = np.zeros(6, dtype=float)

    robot = build_serial_manipulator()
    dyn = build_dynamics()
    sing = Singularity(robot)

    T_desired = robot.forward_kinematics(theta)

    trac_solver = TracIKSolver(
        fk_func=lambda th: robot.forward_kinematics(th, frame="space"),
        jacobian_func=lambda th: robot.jacobian(th, frame="space"),
        joint_limits=robot.joint_limits,
        n_joints=6,
    )

    planner = OptimizedTrajectoryPlanning(
        robot,
        "nonexistent.urdf",  # CollisionChecker fails -> None; CPU path unaffected
        dyn,
        [(-pi, pi)] * 6,
        use_cuda=False,
    )

    # A tiny CPU joint trajectory reused as input for the derivative/ID methods.
    joint_traj = planner.joint_trajectory(theta, theta2, 1.0, 8, 5)
    pos = joint_traj["positions"]
    vel = joint_traj["velocities"]
    acc = joint_traj["accelerations"]
    num_steps = pos.shape[0]
    taumat = np.zeros((num_steps, 6), dtype=float)
    ftipmat = np.zeros((num_steps, 6), dtype=float)

    # (symbol_name, bound_callable, call) — call is a zero-arg thunk so heavy or
    # RNG-driven solvers run exactly once and only their type facets are frozen.
    specs: List[Tuple[str, Callable[..., Any], Callable[[], Any]]] = [
        (
            "SerialManipulator.forward_kinematics",
            robot.forward_kinematics,
            lambda: robot.forward_kinematics(theta),
        ),
        (
            "SerialManipulator.jacobian",
            robot.jacobian,
            lambda: robot.jacobian(theta),
        ),
        (
            "SerialManipulator.end_effector_velocity",
            robot.end_effector_velocity,
            lambda: robot.end_effector_velocity(theta, dtheta),
        ),
        (
            "SerialManipulator.iterative_inverse_kinematics",
            robot.iterative_inverse_kinematics,
            lambda: robot.iterative_inverse_kinematics(
                T_desired, theta.copy(), max_iterations=50
            ),
        ),
        (
            "SerialManipulator.robust_inverse_kinematics",
            robot.robust_inverse_kinematics,
            lambda: robot.robust_inverse_kinematics(
                T_desired, max_attempts=1, max_iterations=100
            ),
        ),
        (
            "SerialManipulator.smart_inverse_kinematics",
            robot.smart_inverse_kinematics,
            lambda: robot.smart_inverse_kinematics(
                T_desired, max_iterations=100, auto_fallback=False
            ),
        ),
        (
            "SerialManipulator.trac_ik",
            robot.trac_ik,
            lambda: robot.trac_ik(T_desired, theta0=theta.copy(), timeout=0.05),
        ),
        (
            "ManipulatorDynamics.mass_matrix",
            dyn.mass_matrix,
            lambda: dyn.mass_matrix(theta),
        ),
        (
            "ManipulatorDynamics.velocity_quadratic_forces",
            dyn.velocity_quadratic_forces,
            lambda: dyn.velocity_quadratic_forces(theta, dtheta),
        ),
        (
            "ManipulatorDynamics.gravity_forces",
            dyn.gravity_forces,
            lambda: dyn.gravity_forces(theta, g_vec),
        ),
        (
            "ManipulatorDynamics.inverse_dynamics",
            dyn.inverse_dynamics,
            lambda: dyn.inverse_dynamics(theta, dtheta, ddtheta, g_vec, ftip),
        ),
        (
            "ManipulatorDynamics.forward_dynamics",
            dyn.forward_dynamics,
            lambda: dyn.forward_dynamics(theta, dtheta, ddtheta, g_vec, ftip),
        ),
        (
            "Singularity.singularity_analysis",
            sing.singularity_analysis,
            lambda: sing.singularity_analysis(theta),
        ),
        (
            "Singularity.near_singularity_detection",
            sing.near_singularity_detection,
            lambda: sing.near_singularity_detection(theta),
        ),
        (
            "Singularity.condition_number",
            sing.condition_number,
            lambda: sing.condition_number(theta),
        ),
        (
            "trac_ik.TracIKSolver.solve",
            trac_solver.solve,
            lambda: trac_solver.solve(T_desired, theta0=theta.copy(), timeout=0.05),
        ),
        (
            "trac_ik.trac_ik_solve",
            trac_ik_solve,
            lambda: trac_ik_solve(robot, T_desired, theta0=theta.copy(), timeout=0.05),
        ),
        (
            "OptimizedTrajectoryPlanning.joint_trajectory",
            planner.joint_trajectory,
            lambda: planner.joint_trajectory(theta, theta2, 1.0, 8, 5),
        ),
        (
            "OptimizedTrajectoryPlanning.cartesian_trajectory",
            planner.cartesian_trajectory,
            lambda: planner.cartesian_trajectory(
                robot.forward_kinematics(theta),
                robot.forward_kinematics(theta2),
                1.0,
                8,
                5,
            ),
        ),
        (
            "OptimizedTrajectoryPlanning.calculate_derivatives",
            planner.calculate_derivatives,
            lambda: planner.calculate_derivatives(pos, 0.1),
        ),
        (
            "OptimizedTrajectoryPlanning.inverse_dynamics_trajectory",
            planner.inverse_dynamics_trajectory,
            lambda: planner.inverse_dynamics_trajectory(pos, vel, acc),
        ),
        (
            "OptimizedTrajectoryPlanning.forward_dynamics_trajectory",
            planner.forward_dynamics_trajectory,
            lambda: planner.forward_dynamics_trajectory(
                theta, dtheta, taumat, g_vec, ftipmat, 0.1, 1
            ),
        ),
    ]

    contract: Dict[str, Dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn, call in specs:
            result = call()
            contract[name] = {
                "signature": _signature_str(fn),
                "return": describe(result),
            }
    return contract


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def _leaf_diffs(path: str, golden: Any, live: Any, out: List[str]) -> None:
    """Collect ``path: golden != live`` leaf differences between two facets."""
    if isinstance(golden, dict) and isinstance(live, dict):
        for key in sorted(set(golden) | set(live)):
            child = f"{path}.{key}" if path else key
            if key not in golden:
                out.append(f"{child}: <missing> != {live[key]!r}")
            elif key not in live:
                out.append(f"{child}: {golden[key]!r} != <missing>")
            else:
                _leaf_diffs(child, golden[key], live[key], out)
    elif isinstance(golden, list) and isinstance(live, list):
        if len(golden) != len(live):
            out.append(f"{path}.length: {len(golden)} != {len(live)}")
        for i, (g, l) in enumerate(zip(golden, live)):
            _leaf_diffs(f"{path}[{i}]", g, l, out)
    elif golden != live:
        out.append(f"{path}: {golden!r} != {live!r}")


def _symbol_report(name: str, golden: Dict[str, Any], live: Dict[str, Any]) -> str:
    """Human-readable report for one drifted symbol: facets + unified diff."""
    facets: List[str] = []
    _leaf_diffs("", golden, live, facets)

    golden_json = json.dumps(golden, indent=2, sort_keys=True).splitlines()
    live_json = json.dumps(live, indent=2, sort_keys=True).splitlines()
    udiff = "\n".join(
        difflib.unified_diff(
            golden_json,
            live_json,
            fromfile=f"golden/{name}",
            tofile=f"live/{name}",
            lineterm="",
        )
    )
    facet_lines = "\n".join(f"    - {f}" for f in facets)
    return f"\nSYMBOL DRIFT: {name}\n  drifted facets:\n{facet_lines}\n{udiff}"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_public_api_return_contract() -> None:
    """Every scoped public symbol must match its frozen signature + return facets."""
    assert GOLDEN_PATH.exists(), (
        f"Golden contract missing at {GOLDEN_PATH}. Regenerate with:\n"
        f"    NUMBA_DISABLE_CUDA=1 MPLBACKEND=Agg .venv/bin/python "
        f"{Path(__file__).name} --regen"
    )

    golden = json.loads(GOLDEN_PATH.read_text())
    live = build_contract()

    problems: List[str] = []

    missing = sorted(set(golden) - set(live))
    for name in missing:
        problems.append(f"\nSYMBOL MISSING FROM LIVE CONTRACT: {name}")

    extra = sorted(set(live) - set(golden))
    for name in extra:
        problems.append(f"\nSYMBOL NOT IN GOLDEN (add via --regen): {name}")

    for name in sorted(set(golden) & set(live)):
        if golden[name] != live[name]:
            problems.append(_symbol_report(name, golden[name], live[name]))

    assert not problems, (
        f"Public API contract drift detected in {len(problems)} symbol(s). "
        f"If intentional, regenerate the golden file.\n" + "\n".join(problems)
    )


def _regen() -> None:
    """Write the golden contract from the current live API."""
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    GOLDEN_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(contract)} symbols to {GOLDEN_PATH}")


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "--regen" in sys.argv:
        _regen()
    else:
        print("Pass --regen to (re)write the golden contract.", file=sys.stderr)
        sys.exit(2)
