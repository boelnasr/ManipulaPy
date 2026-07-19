#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Planner factory and benchmarking utilities - ManipulaPy

Advanced-user entry points that construct or compare
``OptimizedTrajectoryPlanning`` instances. Kept out of ``trajectory_planning``
so that module holds only the planner class and its runtime-compatibility
machinery.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

from . import _kernels as _runtime
from ._kernels import Any, Dict, np, time
from .trajectory_planning import OptimizedTrajectoryPlanning


# Enhanced utility functions for advanced users
def create_optimized_planner(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    torque_limits=None,
    target_speedup=40.0,
    gpu_memory_mb=None,
    enable_profiling=False,
    kernel_type="auto",
) -> OptimizedTrajectoryPlanning:
    """
    Enhanced factory function to create an optimized trajectory planner.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        torque_limits: Torque limits (optional)
        target_speedup: Target speedup over CPU (default: 40x)
        gpu_memory_mb: GPU memory pool size in MB (optional)
        enable_profiling: Enable CUDA profiling (optional)
        kernel_type: Kernel selection strategy (optional)

    Returns:
        OptimizedTrajectoryPlanning: Configured planner instance
    """
    # Auto-detect optimal settings
    cuda_available = _runtime.check_cuda_availability()

    # Adaptive threshold based on target speedup and problem size
    num_joints = len(joint_limits)

    if cuda_available:
        gpu_props = _runtime.get_gpu_properties()
        if gpu_props:
            sm_count = gpu_props["multiprocessor_count"]
            if target_speedup >= 40:
                threshold = max(50, int(sm_count * 10000 / num_joints))
            elif target_speedup >= 20:
                threshold = max(50, int(sm_count * 5000 / num_joints))
            else:
                threshold = max(50, int(sm_count * 1000 / num_joints))
        else:
            threshold = 1000
    else:
        threshold = float("inf")  # Never use GPU if not available

    # Create planner with optimized settings
    planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        torque_limits=torque_limits,
        use_cuda=None,  # Auto-detect
        cuda_threshold=threshold,
        memory_pool_size_mb=gpu_memory_mb,
        enable_profiling=enable_profiling,
        auto_optimize=True,
        kernel_type=kernel_type,
        target_speedup=target_speedup,
    )

    _runtime.logger.info(
        f"Created optimized planner for {num_joints} joints, "
        f"target: {target_speedup}x speedup, CUDA: {cuda_available}"
    )

    return planner


def compare_implementations(
    serial_manipulator,
    urdf_path,
    dynamics,
    joint_limits,
    test_params=None,
    detailed_analysis=True,
) -> Dict[str, Any]:
    """
    Enhanced implementation comparison with detailed kernel analysis.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_params: Test parameters (optional)
        detailed_analysis: Whether to perform detailed kernel comparison

    Returns:
        dict: Comprehensive comparison results
    """
    if test_params is None:
        test_params = {"N": 5000, "Tf": 2.0, "method": 5, "num_runs": 5}

    # Create CPU-only planner
    cpu_planner = OptimizedTrajectoryPlanning(
        serial_manipulator=serial_manipulator,
        urdf_path=urdf_path,
        dynamics=dynamics,
        joint_limits=joint_limits,
        use_cuda=False,
    )

    # Create GPU planner (if available)
    gpu_planner = None
    if _runtime.check_cuda_availability():
        gpu_planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits,
            use_cuda=True,
            cuda_threshold=0,  # Force GPU usage
            kernel_type="auto_tune",
        )

    # Generate test data
    num_joints = len(joint_limits)
    thetastart = np.random.uniform(-1, 1, num_joints).astype(np.float32)
    thetaend = np.random.uniform(-1, 1, num_joints).astype(np.float32)

    results = {"cpu": {}, "gpu": {}}

    # Test CPU implementation
    _runtime.logger.info("Testing CPU implementation...")
    cpu_times = []
    for run in range(test_params.get("num_runs", 3)):
        start_time = time.time()
        cpu_result = cpu_planner.joint_trajectory(
            thetastart,
            thetaend,
            test_params["Tf"],
            test_params["N"],
            test_params["method"],
        )
        cpu_times.append(time.time() - start_time)

    cpu_mean_time = np.mean(cpu_times)
    results["cpu"] = {
        "mean_time": cpu_mean_time,
        "std_time": np.std(cpu_times),
        "min_time": np.min(cpu_times),
        "max_time": np.max(cpu_times),
        "result_shape": cpu_result["positions"].shape,
        "stats": cpu_planner.get_performance_stats(),
    }

    # Test GPU implementation (if available)
    if gpu_planner is not None:
        _runtime.logger.info("Testing GPU implementation...")

        # Test different kernels if detailed analysis requested
        if detailed_analysis:
            kernel_results = gpu_planner.benchmark_all_kernels(
                N=test_params["N"],
                num_joints=num_joints,
                num_runs=test_params.get("num_runs", 3),
            )
            results["kernel_comparison"] = kernel_results

        # Test best configuration
        gpu_times = []
        for run in range(test_params.get("num_runs", 3)):
            start_time = time.time()
            gpu_result = gpu_planner.joint_trajectory(
                thetastart,
                thetaend,
                test_params["Tf"],
                test_params["N"],
                test_params["method"],
            )
            gpu_times.append(time.time() - start_time)

        gpu_mean_time = np.mean(gpu_times)
        speedup = cpu_mean_time / gpu_mean_time if gpu_mean_time > 0 else 0

        results["gpu"] = {
            "mean_time": gpu_mean_time,
            "std_time": np.std(gpu_times),
            "min_time": np.min(gpu_times),
            "max_time": np.max(gpu_times),
            "result_shape": gpu_result["positions"].shape,
            "stats": gpu_planner.get_performance_stats(),
            "speedup": speedup,
        }

        # Compare accuracy
        pos_diff = np.abs(cpu_result["positions"] - gpu_result["positions"])
        vel_diff = np.abs(cpu_result["velocities"] - gpu_result["velocities"])
        acc_diff = np.abs(cpu_result["accelerations"] - gpu_result["accelerations"])

        results["accuracy"] = {
            "max_pos_diff": np.max(pos_diff),
            "max_vel_diff": np.max(vel_diff),
            "max_acc_diff": np.max(acc_diff),
            "mean_pos_diff": np.mean(pos_diff),
            "mean_vel_diff": np.mean(vel_diff),
            "mean_acc_diff": np.mean(acc_diff),
        }

        # Print comprehensive results
        print("\n🚀 Implementation Comparison Results:")
        print("=" * 50)
        print(
            f"CPU Time: {cpu_mean_time*1000:.2f} ± "
            f"{results['cpu']['std_time']*1000:.2f} ms"
        )
        print(
            f"GPU Time: {gpu_mean_time*1000:.2f} ± "
            f"{results['gpu']['std_time']*1000:.2f} ms"
        )
        print(f"Speedup: {speedup:.1f}x")
        print(f"Max Position Error: {results['accuracy']['max_pos_diff']:.2e}")
        print(f"Mean Position Error: {results['accuracy']['mean_pos_diff']:.2e}")

        if speedup >= 40:
            print("🎯 Achieved 40x+ speedup target!")
        elif speedup >= 20:
            print("⚡ Good speedup achieved!")
        elif speedup >= 5:
            print("✅ Moderate speedup achieved")
        else:
            print("⚠️  Limited speedup - consider larger problem sizes")

        _runtime.logger.info(f"GPU speedup: {speedup:.2f}x")
    else:
        results["gpu"] = {"available": False}
        _runtime.logger.info("GPU not available for comparison")

    return results


def benchmark_kernel_performance_comprehensive(
    serial_manipulator, urdf_path, dynamics, joint_limits, test_sizes=None, num_runs=5
) -> Dict[str, Dict[str, Any]]:
    """
    Comprehensive kernel performance benchmarking across multiple problem sizes.

    Args:
        serial_manipulator: SerialManipulator instance
        urdf_path: Path to URDF file
        dynamics: ManipulatorDynamics instance
        joint_limits: Joint limits
        test_sizes: List of (N, joints) tuples to test
        num_runs: Number of runs per test

    Returns:
        dict: Comprehensive benchmark results
    """
    if not _runtime.check_cuda_availability():
        _runtime.logger.warning("CUDA not available for comprehensive benchmarking")
        return {}

    if test_sizes is None:
        test_sizes = [
            (1000, 6),
            (5000, 6),
            (10000, 6),
            (20000, 6),
            (1000, 12),
            (5000, 12),
            (10000, 12),
        ]

    print("\n🔬 Comprehensive Kernel Performance Benchmarking")
    print("=" * 60)

    all_results = {}

    for N, joints in test_sizes:
        _runtime.logger.info(f"Testing N={N}, joints={joints}")

        # Create optimized planner
        planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_path,
            dynamics=dynamics,
            joint_limits=joint_limits[:joints],  # Use subset of joints
            use_cuda=True,
            cuda_threshold=0,
            kernel_type="auto_tune",
        )

        # Benchmark all kernels for this problem size
        kernel_results = planner.benchmark_all_kernels(
            N=N, num_joints=joints, num_runs=num_runs
        )

        all_results[f"N{N}_J{joints}"] = {
            "N": N,
            "joints": joints,
            "total_elements": N * joints,
            "kernel_results": kernel_results,
        }

        # Find best kernel for this size
        if kernel_results:
            best_kernel = min(
                kernel_results.keys(), key=lambda k: kernel_results[k]["mean_time"]
            )
            best_time = kernel_results[best_kernel]["mean_time"]
            throughput = (N * joints) / best_time / 1e6

            print(
                f"N={N:5d}, J={joints:2d}: Best={best_kernel:<15} "
                f"Time={best_time*1000:6.2f}ms Throughput={throughput:6.1f}M/s"
            )

    return all_results
