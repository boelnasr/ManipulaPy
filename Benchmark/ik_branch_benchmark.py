#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
IK Branch Benchmark — Compares every IK solver across 3 codebases:
  1. current   (ManipulaPy/)
  2. branch1   (ik_new_branch/ManipulaPy/)
  3. branch2   (ik_test_branch_2/ManipulaPy/)

Metrics per solver:
  - Success rate (% of targets solved within tolerance)
  - Mean / median / p95 solve time
  - Mean position error (m) and orientation error (rad) for successes
  - Mean iterations (where applicable)

Usage:
  python Benchmark/ik_branch_benchmark.py [--num-targets 50] [--seed 42]
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Suppress noisy imports
# ---------------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MANIPULAPY_FORCE_CPU", "1")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    success: bool
    theta: np.ndarray
    time_s: float
    iterations: int  # -1 if not reported
    pos_error: float
    rot_error: float


@dataclass
class SolverStats:
    name: str
    branch: str
    results: List[SolveResult] = field(default_factory=list)

    @property
    def n(self):
        return len(self.results)

    @property
    def successes(self):
        return [r for r in self.results if r.success]

    @property
    def success_rate(self):
        return len(self.successes) / self.n * 100 if self.n else 0.0

    @property
    def times(self):
        return [r.time_s for r in self.results]

    @property
    def success_times(self):
        return [r.time_s for r in self.successes]

    @property
    def mean_time(self):
        t = self.times
        return np.mean(t) if t else 0.0

    @property
    def median_time(self):
        t = self.times
        return np.median(t) if t else 0.0

    @property
    def p95_time(self):
        t = self.times
        return np.percentile(t, 95) if t else 0.0

    @property
    def mean_pos_err(self):
        errs = [r.pos_error for r in self.successes]
        return np.mean(errs) if errs else float("nan")

    @property
    def mean_rot_err(self):
        errs = [r.rot_error for r in self.successes]
        return np.mean(errs) if errs else float("nan")

    @property
    def mean_iters(self):
        iters = [r.iterations for r in self.results if r.iterations >= 0]
        return np.mean(iters) if iters else float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_branch_module(branch_path: Path, module_name: str):
    """Import a module from a specific branch directory."""
    pkg_path = branch_path / "ManipulaPy"
    if not pkg_path.exists():
        raise FileNotFoundError(f"No ManipulaPy package at {pkg_path}")

    # Temporarily prepend branch path to sys.path
    branch_str = str(branch_path)
    sys.path.insert(0, branch_str)
    try:
        # Force reimport
        for key in list(sys.modules.keys()):
            if key.startswith("ManipulaPy"):
                del sys.modules[key]
        mod = importlib.import_module(f"ManipulaPy.{module_name}")
        return mod
    finally:
        sys.path.remove(branch_str)


def _load_robot_from_branch(branch_path: Path):
    """Load a SerialManipulator from a branch using the xArm URDF."""
    branch_str = str(branch_path)
    sys.path.insert(0, branch_str)
    try:
        for key in list(sys.modules.keys()):
            if key.startswith("ManipulaPy"):
                del sys.modules[key]

        from ManipulaPy.kinematics import SerialManipulator
        from ManipulaPy.urdf_processor import URDFToSerialManipulator

        # Use the xArm URDF from the MAIN project data (shared across branches)
        xarm_urdf = str(
            PROJECT_ROOT / "ManipulaPy" / "ManipulaPy_data" / "xarm" / "xarm6_robot.urdf"
        )
        processor = URDFToSerialManipulator(xarm_urdf)
        robot = processor.serial_manipulator
        return robot, SerialManipulator
    finally:
        sys.path.remove(branch_str)


def _has_method(robot, method_name: str) -> bool:
    return hasattr(robot, method_name) and callable(getattr(robot, method_name))


def _compute_errors(
    robot, theta: np.ndarray, T_target: np.ndarray
) -> Tuple[float, float]:
    """Compute position and orientation errors."""
    try:
        T_actual = robot.forward_kinematics(theta, frame="space")
    except Exception:
        try:
            T_actual = robot.forward_kinematics(theta)
        except Exception:
            return float("nan"), float("nan")

    pos_err = np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3])

    R_err = T_actual[:3, :3].T @ T_target[:3, :3]
    trace_val = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    rot_err = abs(np.arccos(trace_val))

    return pos_err, rot_err


def _generate_targets(robot, num_targets: int, rng: np.random.RandomState):
    """Generate reachable IK targets by sampling random joint angles and computing FK."""
    targets = []
    n_joints = robot.S_list.shape[1] if robot.S_list.ndim == 2 else len(robot.S_list)

    for _ in range(num_targets):
        if robot.joint_limits:
            theta = np.array(
                [rng.uniform(lo, hi) for lo, hi in robot.joint_limits[:n_joints]]
            )
        else:
            theta = rng.uniform(-np.pi, np.pi, n_joints)

        try:
            T = robot.forward_kinematics(theta, frame="space")
        except Exception:
            try:
                T = robot.forward_kinematics(theta)
            except Exception:
                continue

        targets.append((T, theta))

    return targets


# ---------------------------------------------------------------------------
# Solver wrappers — uniform interface returning SolveResult
# ---------------------------------------------------------------------------

def _bench_iterative(robot, T_target, theta_ref, n_joints):
    """Benchmark iterative_inverse_kinematics."""
    theta0 = np.zeros(n_joints)
    t0 = time.perf_counter()
    try:
        theta, success, iters = robot.iterative_inverse_kinematics(
            T_target, theta0, eomg=1e-4, ev=1e-4, max_iterations=5000,
            damping=2e-2, step_cap=0.3
        )
    except Exception:
        return SolveResult(False, np.zeros(n_joints), time.perf_counter() - t0, -1, float("nan"), float("nan"))
    elapsed = time.perf_counter() - t0
    pos_err, rot_err = _compute_errors(robot, theta, T_target)
    return SolveResult(success, theta, elapsed, iters, pos_err, rot_err)


def _bench_smart(robot, T_target, theta_ref, n_joints):
    """Benchmark smart_inverse_kinematics."""
    t0 = time.perf_counter()
    try:
        theta, success, iters = robot.smart_inverse_kinematics(
            T_target, strategy="workspace_heuristic",
            eomg=1e-4, ev=1e-4, max_iterations=5000,
            damping=2e-2, step_cap=0.3
        )
    except Exception:
        return SolveResult(False, np.zeros(n_joints), time.perf_counter() - t0, -1, float("nan"), float("nan"))
    elapsed = time.perf_counter() - t0
    pos_err, rot_err = _compute_errors(robot, theta, T_target)
    return SolveResult(success, theta, elapsed, iters, pos_err, rot_err)


def _bench_robust(robot, T_target, theta_ref, n_joints):
    """Benchmark robust_inverse_kinematics."""
    t0 = time.perf_counter()
    try:
        result = robot.robust_inverse_kinematics(
            T_target, max_attempts=10, eomg=2e-3, ev=2e-3,
            max_iterations=1500
        )
        theta, success = result[0], result[1]
        iters = result[2] if len(result) > 2 else -1
    except Exception:
        return SolveResult(False, np.zeros(n_joints), time.perf_counter() - t0, -1, float("nan"), float("nan"))
    elapsed = time.perf_counter() - t0
    pos_err, rot_err = _compute_errors(robot, theta, T_target)
    return SolveResult(success, theta, elapsed, iters, pos_err, rot_err)


def _bench_trac_ik(robot, T_target, theta_ref, n_joints):
    """Benchmark trac_ik."""
    t0 = time.perf_counter()
    try:
        theta, success, solve_time = robot.trac_ik(
            T_target, timeout=0.2, eomg=1e-4, ev=1e-4, num_restarts=5
        )
    except Exception:
        return SolveResult(False, np.zeros(n_joints), time.perf_counter() - t0, -1, float("nan"), float("nan"))
    elapsed = time.perf_counter() - t0
    pos_err, rot_err = _compute_errors(robot, theta, T_target)
    return SolveResult(success, theta, elapsed, -1, pos_err, rot_err)


# ---------------------------------------------------------------------------
# Branch runner
# ---------------------------------------------------------------------------

SOLVER_DEFS = [
    ("iterative_inverse_kinematics", _bench_iterative),
    ("smart_inverse_kinematics", _bench_smart),
    ("robust_inverse_kinematics", _bench_robust),
    ("trac_ik", _bench_trac_ik),
]


def run_branch(
    branch_name: str,
    branch_path: Path,
    targets: List[Tuple[np.ndarray, np.ndarray]],
) -> List[SolverStats]:
    """Run all available solvers for one branch and return stats."""
    print(f"\n{'='*60}")
    print(f"  Branch: {branch_name}")
    print(f"  Path:   {branch_path}")
    print(f"{'='*60}")

    robot, _ = _load_robot_from_branch(branch_path)
    n_joints = robot.S_list.shape[1] if robot.S_list.ndim == 2 else len(robot.S_list)

    all_stats = []
    for method_name, bench_fn in SOLVER_DEFS:
        if not _has_method(robot, method_name):
            print(f"  [{method_name}] — not available, skipping")
            continue

        stats = SolverStats(name=method_name, branch=branch_name)
        print(f"  [{method_name}] running {len(targets)} targets ... ", end="", flush=True)

        for T_target, theta_ref in targets:
            result = bench_fn(robot, T_target, theta_ref, n_joints)
            stats.results.append(result)

        print(
            f"{stats.success_rate:5.1f}% success, "
            f"mean {stats.mean_time*1000:7.1f} ms"
        )
        all_stats.append(stats)

    return all_stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_branch_stats: Dict[str, List[SolverStats]], num_targets: int):
    """Print a formatted comparison report."""
    print("\n")
    print("=" * 100)
    print(f"  IK BENCHMARK REPORT — {num_targets} random reachable targets (xArm6)")
    print("=" * 100)

    # Collect all solver names across branches
    solver_names = []
    for stats_list in all_branch_stats.values():
        for s in stats_list:
            if s.name not in solver_names:
                solver_names.append(s.name)

    header = (
        f"{'Branch':<16} {'Solver':<30} {'Success':>8} "
        f"{'Mean ms':>9} {'Med ms':>9} {'P95 ms':>9} "
        f"{'Pos Err':>10} {'Rot Err':>10} {'Iters':>8}"
    )
    print(header)
    print("-" * len(header))

    for branch_name, stats_list in all_branch_stats.items():
        for solver_name in solver_names:
            matching = [s for s in stats_list if s.name == solver_name]
            if not matching:
                print(
                    f"{branch_name:<16} {solver_name:<30} {'N/A':>8} "
                    f"{'—':>9} {'—':>9} {'—':>9} "
                    f"{'—':>10} {'—':>10} {'—':>8}"
                )
                continue

            s = matching[0]
            iters_str = f"{s.mean_iters:.0f}" if not np.isnan(s.mean_iters) else "—"
            pos_str = f"{s.mean_pos_err:.2e}" if not np.isnan(s.mean_pos_err) else "—"
            rot_str = f"{s.mean_rot_err:.2e}" if not np.isnan(s.mean_rot_err) else "—"

            print(
                f"{branch_name:<16} {solver_name:<30} {s.success_rate:>7.1f}% "
                f"{s.mean_time*1000:>9.1f} {s.median_time*1000:>9.1f} {s.p95_time*1000:>9.1f} "
                f"{pos_str:>10} {rot_str:>10} {iters_str:>8}"
            )
        print()

    # --- Per-solver comparison (side-by-side) ---
    print("=" * 100)
    print("  PER-SOLVER COMPARISON (best branch highlighted)")
    print("=" * 100)

    branch_names = list(all_branch_stats.keys())
    for solver_name in solver_names:
        entries = {}
        for bn in branch_names:
            matching = [s for s in all_branch_stats[bn] if s.name == solver_name]
            if matching:
                entries[bn] = matching[0]

        if not entries:
            continue

        print(f"\n  {solver_name}")
        print(f"  {'':.<50}")

        # Find best success rate
        best_sr = max((e.success_rate for e in entries.values()), default=0)
        best_time = min((e.mean_time for e in entries.values()), default=float("inf"))

        for bn, e in entries.items():
            sr_marker = " << BEST" if e.success_rate == best_sr and len(entries) > 1 else ""
            tm_marker = " << FASTEST" if e.mean_time == best_time and len(entries) > 1 else ""
            marker = sr_marker or tm_marker
            print(
                f"    {bn:<14}  success={e.success_rate:5.1f}%  "
                f"mean={e.mean_time*1000:7.1f}ms  "
                f"median={e.median_time*1000:7.1f}ms  "
                f"p95={e.p95_time*1000:7.1f}ms"
                f"{marker}"
            )

    print("\n" + "=" * 100)


def save_results_json(
    all_branch_stats: Dict[str, List[SolverStats]], filepath: Path
):
    """Save raw results to JSON for later analysis."""
    data = {}
    for branch_name, stats_list in all_branch_stats.items():
        data[branch_name] = {}
        for s in stats_list:
            data[branch_name][s.name] = {
                "success_rate": round(s.success_rate, 2),
                "n_targets": s.n,
                "n_successes": len(s.successes),
                "mean_time_ms": round(s.mean_time * 1000, 3),
                "median_time_ms": round(s.median_time * 1000, 3),
                "p95_time_ms": round(s.p95_time * 1000, 3),
                "mean_pos_error": round(s.mean_pos_err, 8) if not np.isnan(s.mean_pos_err) else None,
                "mean_rot_error": round(s.mean_rot_err, 8) if not np.isnan(s.mean_rot_err) else None,
                "mean_iterations": round(s.mean_iters, 1) if not np.isnan(s.mean_iters) else None,
            }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IK Branch Benchmark")
    parser.add_argument("--num-targets", type=int, default=50,
                        help="Number of random reachable targets (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # --- Define branches ---
    branches = {
        "current": PROJECT_ROOT,
        "branch1": PROJECT_ROOT / "ik_new_branch",
        "branch2": PROJECT_ROOT / "ik_test_branch_2",
    }

    # Verify all branches exist
    for name, path in branches.items():
        pkg = path / "ManipulaPy"
        if not pkg.exists():
            print(f"WARNING: {name} at {path} not found, skipping")
    branches = {n: p for n, p in branches.items() if (p / "ManipulaPy").exists()}

    if not branches:
        print("ERROR: No valid branches found.")
        sys.exit(1)

    # --- Generate targets from current branch robot ---
    print(f"Generating {args.num_targets} reachable targets (seed={args.seed}) ...")
    robot_main, _ = _load_robot_from_branch(PROJECT_ROOT)
    targets = _generate_targets(robot_main, args.num_targets, rng)
    print(f"  Generated {len(targets)} valid targets")

    if not targets:
        print("ERROR: Could not generate any targets.")
        sys.exit(1)

    # --- Run benchmarks ---
    all_branch_stats = {}
    for branch_name, branch_path in branches.items():
        try:
            stats = run_branch(branch_name, branch_path, targets)
            all_branch_stats[branch_name] = stats
        except Exception as e:
            print(f"\n  ERROR running {branch_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Report ---
    print_report(all_branch_stats, len(targets))

    # --- Save ---
    save_path = args.save or str(PROJECT_ROOT / "Benchmark" / "ik_branch_benchmark_results.json")
    save_results_json(all_branch_stats, Path(save_path))


if __name__ == "__main__":
    main()
