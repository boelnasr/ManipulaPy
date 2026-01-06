#!/usr/bin/env python3
"""
Quick demonstration of robust_inverse_kinematics success rate improvement.

Compares:
1. iterative_inverse_kinematics (baseline)
2. smart_inverse_kinematics with workspace_heuristic
3. robust_inverse_kinematics (new adaptive multi-start)
"""

import numpy as np
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file
from ManipulaPy import ik_helpers

# Load robot
print("="*70)
print("ROBUST IK DEMONSTRATION")
print("="*70)
print("Loading xArm 6-DOF robot...")
processor = URDFToSerialManipulator(urdf_file)
robot = processor.serial_manipulator

# Generate 10 test targets
print("\nGenerating 10 test targets...")
num_tests = 10
targets = []
for _ in range(num_tests):
    config = [np.random.uniform(low, high) for low, high in robot.joint_limits]
    T = robot.forward_kinematics(config)
    targets.append(T)

print(f"Generated {len(targets)} targets\n")

# Test parameters
eomg = 1e-3
ev = 1e-3
max_iterations = 2000

# Method 1: Iterative IK (baseline)
print("="*70)
print("METHOD 1: iterative_inverse_kinematics (baseline)")
print("="*70)
successes_iterative = 0
for i, T_target in enumerate(targets):
    # Use workspace heuristic for fair comparison
    theta0 = ik_helpers.workspace_heuristic_guess(T_target, 6, robot.joint_limits)

    solution, success, iters = robot.iterative_inverse_kinematics(
        T_target, theta0, eomg=eomg, ev=ev, max_iterations=max_iterations
    )
    if success:
        successes_iterative += 1
        print(f"  Test {i+1}: ✓ SUCCESS ({iters} iters)")
    else:
        print(f"  Test {i+1}: ✗ FAILED")

rate_iterative = successes_iterative / num_tests * 100
print(f"\nSuccess rate: {successes_iterative}/{num_tests} ({rate_iterative:.1f}%)\n")

# Method 2: Smart IK with workspace heuristic
print("="*70)
print("METHOD 2: smart_inverse_kinematics(workspace_heuristic)")
print("="*70)
successes_smart = 0
for i, T_target in enumerate(targets):
    solution, success, iters = robot.smart_inverse_kinematics(
        T_target,
        strategy='workspace_heuristic',
        eomg=eomg, ev=ev, max_iterations=max_iterations
    )
    if success:
        successes_smart += 1
        print(f"  Test {i+1}: ✓ SUCCESS ({iters} iters)")
    else:
        print(f"  Test {i+1}: ✗ FAILED")

rate_smart = successes_smart / num_tests * 100
print(f"\nSuccess rate: {successes_smart}/{num_tests} ({rate_smart:.1f}%)\n")

# Method 3: Robust IK (adaptive multi-start)
print("="*70)
print("METHOD 3: robust_inverse_kinematics (adaptive multi-start)")
print("="*70)
successes_robust = 0
strategies_used = []
for i, T_target in enumerate(targets):
    solution, success, total_iters, strategy = robot.robust_inverse_kinematics(
        T_target,
        max_attempts=10,
        eomg=eomg, ev=ev, max_iterations=max_iterations,
        verbose=False
    )
    if success:
        successes_robust += 1
        strategies_used.append(strategy)
        print(f"  Test {i+1}: ✓ SUCCESS ({total_iters} total iters, strategy={strategy})")
    else:
        print(f"  Test {i+1}: ✗ FAILED")

rate_robust = successes_robust / num_tests * 100
print(f"\nSuccess rate: {successes_robust}/{num_tests} ({rate_robust:.1f}%)")

# Strategy statistics
if strategies_used:
    from collections import Counter
    strategy_counts = Counter(strategies_used)
    print(f"Winning strategies: {dict(strategy_counts)}\n")

# Summary comparison
print("="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"{'Method':<50} {'Success Rate'}")
print("-"*70)
print(f"{'iterative_inverse_kinematics':<50} {rate_iterative:>5.1f}%")
print(f"{'smart_inverse_kinematics(workspace_heuristic)':<50} {rate_smart:>5.1f}%")
print(f"{'robust_inverse_kinematics (RECOMMENDED)':<50} {rate_robust:>5.1f}%")
print("="*70)

improvement = rate_robust - rate_iterative
print(f"\n🎯 Improvement: +{improvement:.1f}% ({successes_robust - successes_iterative} more successes)")
print(f"📊 Relative improvement: {improvement/max(rate_iterative, 0.1):.1f}x better")
print("\n✅ Recommendation: Use robust_inverse_kinematics for production!")
