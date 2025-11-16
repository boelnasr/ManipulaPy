#!/usr/bin/env python3
"""
IK Initial Guess Comparison Benchmark

Demonstrates the impact of different initial guess strategies on
IK convergence speed and success rate.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ManipulaPy.kinematics import SerialManipulator
from ik_initial_guess_strategies import IKInitialGuessGenerator


def create_example_6dof_manipulator() -> SerialManipulator:
    """Create a 6-DOF manipulator for testing."""
    # Example: Simple 6-DOF arm with revolute joints
    M = np.eye(4)
    M[:3, 3] = [0, 0, 0.8]  # End-effector at home config

    # Screw axes (space frame)
    omega_list = np.array([
        [0, 0, 1],   # J1: Z-axis rotation
        [0, 1, 0],   # J2: Y-axis rotation
        [0, 1, 0],   # J3: Y-axis rotation
        [1, 0, 0],   # J4: X-axis rotation (wrist)
        [0, 1, 0],   # J5: Y-axis rotation (wrist)
        [1, 0, 0],   # J6: X-axis rotation (wrist)
    ]).T

    # Points on screw axes
    r_list = np.array([
        [0, 0, 0],      # J1 at base
        [0, 0, 0.15],   # J2 elevated
        [0, 0, 0.40],   # J3 (elbow)
        [0, 0, 0.65],   # J4 (wrist 1)
        [0, 0, 0.75],   # J5 (wrist 2)
        [0, 0, 0.80],   # J6 (wrist 3)
    ]).T

    # Joint limits (typical industrial robot)
    joint_limits = [
        (-np.pi, np.pi),           # J1: ±180°
        (-2*np.pi/3, 2*np.pi/3),  # J2: ±120°
        (-2*np.pi/3, 2*np.pi/3),  # J3: ±120°
        (-np.pi, np.pi),           # J4: ±180°
        (-np.pi/2, np.pi/2),      # J5: ±90°
        (-np.pi, np.pi),           # J6: ±180°
    ]

    manipulator = SerialManipulator(
        M_list=M,
        omega_list=omega_list,
        r_list=r_list,
        joint_limits=joint_limits
    )

    return manipulator


def generate_test_poses(
    manipulator: SerialManipulator,
    n_poses: int = 20
) -> List[Tuple[NDArray, NDArray]]:
    """
    Generate random reachable test poses.

    Returns:
        List of (T_desired, theta_actual) pairs
    """
    test_poses = []

    for _ in range(n_poses):
        # Generate random valid joint configuration
        theta_actual = []
        for mn, mx in manipulator.joint_limits:
            if mn is not None and mx is not None:
                theta_actual.append(np.random.uniform(mn, mx))
            else:
                theta_actual.append(np.random.uniform(-np.pi, np.pi))

        theta_actual = np.array(theta_actual)

        # Compute forward kinematics to get reachable pose
        T_desired = manipulator.forward_kinematics(theta_actual)

        test_poses.append((T_desired, theta_actual))

    return test_poses


def benchmark_initial_guess_strategy(
    manipulator: SerialManipulator,
    test_poses: List[Tuple],
    strategy_name: str,
    guess_func,
    max_iterations: int = 500
) -> Dict:
    """
    Benchmark a specific initial guess strategy.

    Returns:
        Dictionary with performance metrics
    """
    results = {
        'strategy': strategy_name,
        'success_rate': 0,
        'avg_iterations': 0,
        'avg_time': 0,
        'iterations_list': [],
        'times_list': [],
        'errors_list': [],
    }

    successes = 0
    total_iterations = 0
    total_time = 0

    for T_desired, theta_actual in test_poses:
        # Generate initial guess using the strategy
        theta0 = guess_func(T_desired)

        # Run IK
        start_time = time()
        theta_solution, success, iterations = manipulator.iterative_inverse_kinematics(
            T_desired,
            theta0,
            max_iterations=max_iterations,
            eomg=1e-4,
            ev=1e-4
        )
        elapsed_time = time() - start_time

        # Compute final error
        T_result = manipulator.forward_kinematics(theta_solution)
        pos_error = np.linalg.norm(T_desired[:3, 3] - T_result[:3, 3])

        # Record results
        if success:
            successes += 1

        total_iterations += iterations
        total_time += elapsed_time

        results['iterations_list'].append(iterations)
        results['times_list'].append(elapsed_time)
        results['errors_list'].append(pos_error)

    # Compute averages
    n_tests = len(test_poses)
    results['success_rate'] = successes / n_tests * 100
    results['avg_iterations'] = total_iterations / n_tests
    results['avg_time'] = total_time / n_tests

    return results


def run_benchmark():
    """Run comprehensive benchmark comparing all strategies."""
    print("=" * 60)
    print("IK Initial Guess Strategy Benchmark")
    print("=" * 60)

    # Create manipulator
    print("\n1. Creating 6-DOF manipulator...")
    manipulator = create_example_6dof_manipulator()

    # Generate test poses
    print("2. Generating test poses...")
    n_test_poses = 50
    test_poses = generate_test_poses(manipulator, n_test_poses)
    print(f"   Generated {n_test_poses} random reachable poses")

    # Create initial guess generator
    ik_guesser = IKInitialGuessGenerator(manipulator)

    # Define strategies to test
    print("\n3. Testing initial guess strategies...")
    strategies = [
        {
            'name': 'Zero Configuration',
            'func': lambda T: np.zeros(6),
        },
        {
            'name': 'Random Guess',
            'func': lambda T: ik_guesser._random_configuration(),
        },
        {
            'name': 'Workspace Heuristic',
            'func': lambda T: ik_guesser.workspace_heuristic(T),
        },
        {
            'name': 'Midpoint of Joint Limits',
            'func': lambda T: np.array([
                (mn + mx) / 2 if mn is not None and mx is not None else 0
                for mn, mx in manipulator.joint_limits
            ]),
        },
    ]

    # Run benchmarks
    all_results = []
    for strategy in strategies:
        print(f"\n   Testing: {strategy['name']}...")
        results = benchmark_initial_guess_strategy(
            manipulator,
            test_poses,
            strategy['name'],
            strategy['func']
        )
        all_results.append(results)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n{'Strategy':<30} {'Success Rate':<15} {'Avg Iter':<12} {'Avg Time (ms)'}")
    print("-" * 75)

    for result in all_results:
        print(f"{result['strategy']:<30} "
              f"{result['success_rate']:>6.1f}%        "
              f"{result['avg_iterations']:>8.1f}     "
              f"{result['avg_time']*1000:>8.2f}")

    # Visualization
    plot_benchmark_results(all_results)


def plot_benchmark_results(all_results: List[Dict]):
    """Create visualization of benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategy_names = [r['strategy'] for r in all_results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))

    # 1. Success Rate
    ax = axes[0, 0]
    success_rates = [r['success_rate'] for r in all_results]
    bars = ax.bar(strategy_names, success_rates, color=colors)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('IK Success Rate by Strategy')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    # 2. Average Iterations
    ax = axes[0, 1]
    avg_iters = [r['avg_iterations'] for r in all_results]
    bars = ax.bar(strategy_names, avg_iters, color=colors)
    ax.set_ylabel('Average Iterations')
    ax.set_title('Convergence Speed (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Iteration Distribution (Box Plot)
    ax = axes[1, 0]
    iterations_data = [r['iterations_list'] for r in all_results]
    bp = ax.boxplot(iterations_data, labels=strategy_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Iterations')
    ax.set_title('Iteration Count Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Time Distribution (Box Plot)
    ax = axes[1, 1]
    times_data = [np.array(r['times_list']) * 1000 for r in all_results]  # Convert to ms
    bp = ax.boxplot(times_data, labels=strategy_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computation Time Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('ik_initial_guess_benchmark.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Benchmark plot saved: ik_initial_guess_benchmark.png")
    plt.show()


if __name__ == "__main__":
    run_benchmark()
