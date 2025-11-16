#!/usr/bin/env python3
"""
Quick Start: Smart IK Initial Guess

Minimal working example showing immediate improvement with smart initial guesses.

Copyright (c) 2025 Mohamed Aboelnasr
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ManipulaPy.kinematics import SerialManipulator
from ik_initial_guess_strategies import IKInitialGuessGenerator


def create_simple_6dof_robot():
    """Create a simple 6-DOF robot for demonstration."""
    # Home configuration end-effector pose
    M = np.eye(4)
    M[:3, 3] = [0, 0, 0.8]  # 80cm height at home

    # Screw axes (space frame)
    omega_list = np.array([
        [0, 0, 1],   # J1: Base rotation
        [0, 1, 0],   # J2: Shoulder
        [0, 1, 0],   # J3: Elbow
        [1, 0, 0],   # J4: Wrist 1
        [0, 1, 0],   # J5: Wrist 2
        [1, 0, 0],   # J6: Wrist 3
    ]).T

    # Points on screw axes
    r_list = np.array([
        [0, 0, 0],      # J1
        [0, 0, 0.15],   # J2
        [0, 0, 0.40],   # J3
        [0, 0, 0.65],   # J4
        [0, 0, 0.75],   # J5
        [0, 0, 0.80],   # J6
    ]).T

    # Joint limits (typical)
    joint_limits = [
        (-np.pi, np.pi),
        (-2*np.pi/3, 2*np.pi/3),
        (-2*np.pi/3, 2*np.pi/3),
        (-np.pi, np.pi),
        (-np.pi/2, np.pi/2),
        (-np.pi, np.pi),
    ]

    return SerialManipulator(
        M_list=M,
        omega_list=omega_list,
        r_list=r_list,
        joint_limits=joint_limits
    )


def demo_basic_improvement():
    """Demonstrate basic improvement with workspace heuristic."""
    print("=" * 70)
    print("DEMO 1: Basic Improvement with Workspace Heuristic")
    print("=" * 70)

    # Create robot
    robot = create_simple_6dof_robot()
    ik_gen = IKInitialGuessGenerator(robot)

    # Target pose
    T_target = np.eye(4)
    T_target[:3, 3] = [0.35, 0.15, 0.45]  # Reachable position

    print(f"\nTarget position: {T_target[:3, 3]}")

    # Method 1: Zero initial guess (baseline)
    print("\n--- Method 1: Zero Initial Guess ---")
    theta0_zeros = np.zeros(6)
    theta1, success1, iters1 = robot.iterative_inverse_kinematics(
        T_target, theta0_zeros, max_iterations=500
    )
    print(f"  Success: {success1}")
    print(f"  Iterations: {iters1}")
    if success1:
        T_result = robot.forward_kinematics(theta1)
        error = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
        print(f"  Position error: {error:.6f}m")

    # Method 2: Workspace heuristic (improved)
    print("\n--- Method 2: Workspace Heuristic ---")
    theta0_smart = ik_gen.workspace_heuristic(T_target)
    print(f"  Initial guess: {np.round(theta0_smart, 3)}")
    theta2, success2, iters2 = robot.iterative_inverse_kinematics(
        T_target, theta0_smart, max_iterations=500
    )
    print(f"  Success: {success2}")
    print(f"  Iterations: {iters2}")
    if success2:
        T_result = robot.forward_kinematics(theta2)
        error = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
        print(f"  Position error: {error:.6f}m")

    # Show improvement
    if success1 and success2:
        improvement = (iters1 - iters2) / iters1 * 100
        print(f"\n✨ Improvement: {improvement:.1f}% fewer iterations!")
        print(f"   ({iters1} → {iters2} iterations)")


def demo_trajectory_tracking():
    """Demonstrate trajectory tracking with extrapolation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Trajectory Tracking with Extrapolation")
    print("=" * 70)

    # Create robot
    robot = create_simple_6dof_robot()
    ik_gen = IKInitialGuessGenerator(robot)

    # Create a simple linear trajectory
    waypoints = []
    for t in np.linspace(0, 1, 10):
        T = np.eye(4)
        T[:3, 3] = [0.3 + 0.1*t, 0.1*np.sin(2*np.pi*t), 0.4 + 0.05*t]
        waypoints.append(T)

    print(f"\nTracking {len(waypoints)} waypoints...")

    # Track with extrapolation
    theta_current = np.zeros(6)
    total_iterations = 0

    print("\nWaypoint tracking:")
    for i, T_waypoint in enumerate(waypoints):
        # Use extrapolation for smooth tracking
        theta0 = ik_gen.current_configuration_extrapolation(
            theta_current,
            robot.forward_kinematics(theta_current),
            T_waypoint,
            alpha=0.5
        )

        theta_new, success, iters = robot.iterative_inverse_kinematics(
            T_waypoint, theta0, max_iterations=100
        )

        total_iterations += iters
        status = "✓" if success else "✗"
        print(f"  Waypoint {i:2d}: {status} {iters:3d} iterations")

        if success:
            theta_current = theta_new
        else:
            print(f"    Warning: Failed at waypoint {i}")

    avg_iters = total_iterations / len(waypoints)
    print(f"\nAverage: {avg_iters:.1f} iterations per waypoint")
    print(f"💡 Tip: With good initial guess, typically < 10 iterations!")


def demo_multiple_restarts():
    """Demonstrate multiple random restarts for difficult poses."""
    print("\n" + "=" * 70)
    print("DEMO 3: Multiple Random Restarts for Difficult Poses")
    print("=" * 70)

    # Create robot
    robot = create_simple_6dof_robot()
    ik_gen = IKInitialGuessGenerator(robot)

    # Challenging pose (near singularity or workspace boundary)
    T_difficult = np.eye(4)
    T_difficult[:3, 3] = [0.1, 0.05, 0.75]  # Close to vertical

    print(f"\nTarget position (challenging): {T_difficult[:3, 3]}")

    # Try multiple restarts
    print("\n--- Multiple Random Restarts (5 attempts) ---")
    theta_best, success, all_results = ik_gen.multiple_random_restarts(
        T_difficult,
        n_attempts=5,
        max_iterations=200
    )

    print(f"\nResults from {len(all_results)} attempts:")
    for i, (theta, success, iters, error) in enumerate(all_results):
        status = "✓" if success else "✗"
        print(f"  Attempt {i+1}: {status} {iters:4d} iters, error={error:.6f}")

    print(f"\n✨ Best solution: error={min(r[3] for r in all_results):.6f}")


def demo_caching():
    """Demonstrate solution caching for repeated similar tasks."""
    print("\n" + "=" * 70)
    print("DEMO 4: Solution Caching for Repeated Tasks")
    print("=" * 70)

    # Create robot
    robot = create_simple_6dof_robot()
    ik_gen = IKInitialGuessGenerator(robot)

    # Simulate pick-and-place: 3 pickup locations, 2 place locations
    pickup_positions = [
        [0.35, 0.10, 0.20],
        [0.35, 0.15, 0.20],
        [0.35, 0.20, 0.20],
    ]
    place_positions = [
        [0.30, -0.15, 0.40],
        [0.30, -0.20, 0.40],
    ]

    print("\nPhase 1: Building cache (warmup)")
    print("-" * 40)

    # Warmup: Solve IK for all positions and cache
    for i, pos in enumerate(pickup_positions + place_positions):
        T = np.eye(4)
        T[:3, 3] = pos

        theta0 = ik_gen.workspace_heuristic(T)
        theta, success, iters = robot.iterative_inverse_kinematics(T, theta0)

        if success:
            ik_gen.add_to_cache(T, theta)
            print(f"  Position {i+1}: Cached ({iters} iterations)")

    print(f"\nCache size: {len(ik_gen.solution_cache)} solutions")

    print("\nPhase 2: Using cached solutions")
    print("-" * 40)

    # Now solve similar poses using cache
    total_iters_cached = 0
    n_tests = 5

    for i in range(n_tests):
        # Random pose near cached solutions
        base_pos = pickup_positions[i % len(pickup_positions)]
        noise = np.random.randn(3) * 0.02  # 2cm noise
        T_test = np.eye(4)
        T_test[:3, 3] = base_pos + noise

        # Use cached nearest neighbor
        theta0 = ik_gen.cached_nearest_neighbor(T_test, k=3)
        theta, success, iters = robot.iterative_inverse_kinematics(
            T_test, theta0, max_iterations=100
        )

        total_iters_cached += iters
        status = "✓" if success else "✗"
        print(f"  Test {i+1}: {status} {iters:3d} iterations")

    avg_cached = total_iters_cached / n_tests
    print(f"\n✨ Average with cache: {avg_cached:.1f} iterations")
    print(f"💡 Cache provides excellent initial guesses for similar poses!")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Smart IK Initial Guess - Quick Start" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_basic_improvement()
        demo_trajectory_tracking()
        demo_multiple_restarts()
        demo_caching()

        print("\n" + "=" * 70)
        print("🎉 All demos completed successfully!")
        print("=" * 70)

        print("\n📚 Next Steps:")
        print("  1. Read IK_INITIAL_GUESS_GUIDE.md for detailed strategies")
        print("  2. Run benchmark: python examples/ik_initial_guess_comparison.py")
        print("  3. Integrate into your robot code (see INTEGRATION_EXAMPLE.md)")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
