#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced GPU Acceleration Demo - ManipulaPy

This demo showcases ManipulaPy's GPU/CPU auto-dispatch strategy for
high-throughput trajectory generation. It:

- Probes the CUDA runtime through ManipulaPy.cuda_kernels and reports,
  honestly, whether a usable GPU is present on this machine.
- Benchmarks single-trajectory generation with the public CPU fallback
  kernel (``trajectory_cpu_fallback``) across a range of trajectory lengths.
- Benchmarks *batch* trajectory generation through the high-level
  ``OptimizedTrajectoryPlanning`` planner (which auto-selects GPU when
  available and degrades gracefully to CPU otherwise).
- Exercises the profiling helpers (``profile_start`` / ``profile_stop``,
  ``CUDAPerformanceMonitor``, ``get_memory_pool_stats``) that are safe to
  call without a GPU.
- Plots a CPU-vs-(GPU-if-present) throughput / scaling chart.

The demo is GPU-aware but never GPU-dependent: when CUDA is missing it runs
the CPU path only, labels every result as CPU, and exits cleanly. No raw
``@cuda.jit`` kernel is ever invoked directly.

Usage:
    python gpu_acceleration_advanced_demo.py

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

try:
    from ManipulaPy.cuda_kernels import (
        CUDA_AVAILABLE,
        CUDAPerformanceMonitor,
        check_cuda_availability,
        check_cupy_availability,
        get_gpu_properties,
        get_memory_pool_stats,
        print_performance_recommendations,
        profile_start,
        profile_stop,
        trajectory_cpu_fallback,
    )
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
except ImportError as exc:  # pragma: no cover - install guard
    print(f"Error importing ManipulaPy modules: {exc}")
    print("Please ensure ManipulaPy is properly installed.")
    raise SystemExit(1)

# Save plots next to this script regardless of the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))

# Quintic time-scaling (method=5) is the most representative trajectory profile.
TRAJECTORY_METHOD = 5


def detect_backend() -> Tuple[bool, str]:
    """Probe the CUDA runtime and return ``(gpu_active, human_summary)``.

    All probing goes through ManipulaPy's public availability helpers, so this
    is safe on a CPU-only box (it never touches a ``@cuda.jit`` kernel).
    """
    cuda_ok = bool(check_cuda_availability())
    cupy_ok = bool(check_cupy_availability())
    gpu_active = cuda_ok and CUDA_AVAILABLE

    if gpu_active:
        props = get_gpu_properties() or {}
        name = props.get("name", "unknown GPU")
        summary = f"GPU active: {name} (CuPy={'yes' if cupy_ok else 'no'})"
    else:
        summary = (
            "GPU not available - running CPU-only "
            f"(numba CUDA={'yes' if cuda_ok else 'no'}, CuPy={'yes' if cupy_ok else 'no'})"
        )
    return gpu_active, summary


def build_planner() -> Tuple[OptimizedTrajectoryPlanning, int]:
    """Load the bundled xArm6 model and build an auto-dispatching planner.

    Returns the planner and the robot's joint count. ``use_cuda=None`` lets the
    planner pick GPU when present and fall back to CPU otherwise.
    """
    proc = URDFToSerialManipulator(urdf_file)
    robot = proc.serial_manipulator
    dynamics = proc.dynamics
    num_joints = robot.S_list.shape[1]
    joint_limits = [(-np.pi, np.pi)] * num_joints

    planner = OptimizedTrajectoryPlanning(
        robot,
        urdf_file,
        dynamics,
        joint_limits,
        use_cuda=None,          # auto-detect; safe CPU fallback
        cuda_threshold=10,
        enable_profiling=False,
    )
    return planner, num_joints


def _timeit(fn, repeats: int) -> float:
    """Return the best wall-clock time (seconds) over ``repeats`` calls."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def benchmark_single_cpu(
    num_joints: int,
    lengths: List[int],
    repeats: int = 3,
) -> Dict[int, float]:
    """Benchmark the public ``trajectory_cpu_fallback`` kernel.

    Generates one quintic joint trajectory of each requested length and records
    the best run time. This is the always-available CPU reference path.
    """
    thetastart = np.zeros(num_joints)
    thetaend = np.linspace(0.4, 1.2, num_joints)
    times: Dict[int, float] = {}
    for n in lengths:
        t = _timeit(
            lambda n=n: trajectory_cpu_fallback(
                thetastart, thetaend, 2.0, n, TRAJECTORY_METHOD
            ),
            repeats,
        )
        times[n] = t
        print(f"    CPU single  N={n:5d}  ->  {t * 1e3:8.2f} ms")
    return times


def benchmark_batch(
    planner: OptimizedTrajectoryPlanning,
    num_joints: int,
    batch_sizes: List[int],
    traj_length: int = 100,
    repeats: int = 2,
) -> Dict[int, float]:
    """Benchmark batch trajectory generation via the auto-dispatch planner.

    With no GPU the planner logs a fallback notice and runs the batch
    sequentially on the CPU; with a GPU it uses the batched kernel. Either way
    the call returns the same dict shape, so the demo stays portable.
    """
    rng = np.random.default_rng(0)
    times: Dict[int, float] = {}
    for b in batch_sizes:
        starts = np.zeros((b, num_joints))
        ends = rng.uniform(-1.0, 1.0, size=(b, num_joints))
        t = _timeit(
            lambda starts=starts, ends=ends: planner.batch_joint_trajectory(
                starts, ends, 2.0, traj_length, TRAJECTORY_METHOD
            ),
            repeats,
        )
        times[b] = t
        total = b * traj_length
        rate = total / t if t > 0 else 0.0
        print(
            f"    Batch  size={b:4d}  ({total:6d} waypoints)  ->  "
            f"{t * 1e3:8.2f} ms  ({rate / 1e3:7.1f} k waypoints/s)"
        )
    return times


def profiling_snapshot(num_joints: int, traj_length: int) -> None:
    """Exercise the CPU-safe profiling helpers and print a short report."""
    monitor = CUDAPerformanceMonitor()
    profile_start()
    # A representative kernel launch we can attribute timing to.
    start = time.perf_counter()
    trajectory_cpu_fallback(
        np.zeros(num_joints), np.ones(num_joints) * 0.5, 2.0, traj_length, TRAJECTORY_METHOD
    )
    elapsed = time.perf_counter() - start
    monitor.record_kernel_launch("trajectory_cpu_fallback", elapsed, traj_length, num_joints)
    prof = profile_stop() or {}

    print("    profile_stop() keys :", sorted(prof.keys()) or "(empty on CPU)")
    print("    memory pool stats   :", get_memory_pool_stats() or "(empty on CPU)")
    stats = monitor.get_stats()
    launches = stats.get("total_launches", stats.get("kernel_launches", "n/a"))
    print(f"    monitor launches    : {launches}")
    print("    library recommendation:")
    print_performance_recommendations(traj_length, num_joints)


def plot_scaling(
    single_times: Dict[int, float],
    batch_times: Dict[int, float],
    traj_length: int,
    gpu_active: bool,
    out_dir: str,
) -> str:
    """Plot single-trajectory scaling and batch throughput; return the path."""
    label = "GPU" if gpu_active else "CPU"
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: single-trajectory latency vs trajectory length.
    lengths = sorted(single_times)
    ax_left.plot(
        lengths,
        [single_times[n] * 1e3 for n in lengths],
        "o-",
        color="#1f77b4",
        label=f"single trajectory ({label})",
    )
    ax_left.set_xlabel("trajectory length N (waypoints)")
    ax_left.set_ylabel("time per trajectory (ms)")
    ax_left.set_title("Single-trajectory generation scaling")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend()

    # Right: batch throughput (waypoints/second) vs batch size.
    sizes = sorted(batch_times)
    throughput = [
        (b * traj_length) / batch_times[b] / 1e3 if batch_times[b] > 0 else 0.0
        for b in sizes
    ]
    ax_right.plot(sizes, throughput, "s-", color="#d62728", label=f"batch ({label})")
    ax_right.set_xlabel("batch size (number of trajectories)")
    ax_right.set_ylabel("throughput (k waypoints/s)")
    ax_right.set_title(f"Batch throughput  (N={traj_length} each)")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend()

    backend_note = "GPU detected" if gpu_active else "CPU fallback (no GPU detected)"
    fig.suptitle(f"ManipulaPy trajectory throughput - {backend_note}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, "gpu_acceleration_scaling.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    """Run the GPU/CPU auto-dispatch acceleration demo end to end."""
    print("=== ManipulaPy: Advanced GPU Acceleration Demo ===\n")

    gpu_active, summary = detect_backend()
    print(f"[backend] {summary}\n")

    print("[setup] Loading bundled xArm6 model and building auto-dispatch planner...")
    planner, num_joints = build_planner()
    print(f"[setup] Robot has {num_joints} joints.\n")

    # Modest sizes keep the CPU run well under the time budget while staying
    # representative of the scaling behavior.
    single_lengths = [50, 100, 200, 400, 800]
    batch_sizes = [2, 4, 8, 16, 32]
    traj_length = 100

    print("[benchmark] Single-trajectory CPU fallback kernel:")
    single_times = benchmark_single_cpu(num_joints, single_lengths)
    print()

    print("[benchmark] Batch trajectory generation (auto-dispatch):")
    batch_times = benchmark_batch(planner, num_joints, batch_sizes, traj_length)
    print()

    print("[profiling] CPU-safe profiling helpers:")
    profiling_snapshot(num_joints, traj_length)
    print()

    plot_path = plot_scaling(single_times, batch_times, traj_length, gpu_active, HERE)
    print(f"[output] Saved scaling chart -> {plot_path}")

    backend_word = "GPU" if gpu_active else "CPU-only"
    largest = max(batch_sizes)
    total_wp = largest * traj_length
    rate = total_wp / batch_times[largest] / 1e3 if batch_times[largest] > 0 else 0.0
    print(
        f"\n[summary] Backend: {backend_word}. "
        f"Peak batch throughput ~{rate:.1f} k waypoints/s "
        f"(batch={largest}, N={traj_length})."
    )
    if not gpu_active:
        print(
            "[summary] No usable CUDA device found; all results above are CPU "
            "fallback. On a CUDA-capable machine the same code path uses the GPU "
            "kernels automatically."
        )
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
