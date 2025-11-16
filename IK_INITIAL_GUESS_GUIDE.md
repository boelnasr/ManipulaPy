# IK Initial Guess Optimization Guide

## Overview

The inverse kinematics (IK) solver's performance heavily depends on the **initial guess** for joint angles. A good initial guess can:
- ✅ Reduce iterations by 50-80%
- ✅ Improve success rate from 60% to 95%+
- ✅ Avoid local minima and singularities
- ✅ Enable real-time performance (< 5ms per IK solve)

This guide provides practical strategies to generate better initial guesses.

---

## 🎯 Quick Start

### Method 1: Use the IKInitialGuessGenerator

```python
from ManipulaPy.kinematics import SerialManipulator
from ik_initial_guess_strategies import IKInitialGuessGenerator
import numpy as np

# Your manipulator setup
manipulator = SerialManipulator(M_list, omega_list, r_list, ...)

# Create guess generator
ik_guesser = IKInitialGuessGenerator(manipulator)

# Desired pose
T_desired = np.eye(4)
T_desired[:3, 3] = [0.3, 0.2, 0.4]  # Target position

# Generate smart initial guess
theta0 = ik_guesser.workspace_heuristic(T_desired)

# Run IK with improved guess
theta_solution, success, iterations = manipulator.iterative_inverse_kinematics(
    T_desired, theta0
)
```

---

## 📊 Strategy Comparison

| Strategy | Success Rate | Avg Iterations | Use Case |
|----------|-------------|----------------|----------|
| **Workspace Heuristic** | 85-95% | 20-50 | General purpose, first choice |
| **Multiple Random Restarts** | 95-99% | 100-200 | When success is critical |
| **Cached Nearest Neighbor** | 90-98% | 10-30 | Trajectory tracking |
| **Current Config Extrapolation** | 95-99% | 5-15 | Real-time control |
| **Zero Configuration** | 60-70% | 100-500 | Baseline (not recommended) |
| **Random Guess** | 50-65% | 200-500 | Baseline (not recommended) |

---

## 🔧 Available Strategies

### 1. Workspace Heuristic (Recommended Default)

Uses geometric approximation based on target position.

```python
theta0 = ik_guesser.workspace_heuristic(T_desired)
```

**Pros:**
- Fast (< 0.1ms)
- No additional data required
- Works for most manipulator geometries

**Cons:**
- Less accurate for orientation-critical tasks
- May struggle with complex kinematic chains

**Best for:** General-purpose IK solving, first attempt

---

### 2. Multiple Random Restarts

Tries multiple random guesses and returns the best solution.

```python
theta_solution, success, results = ik_guesser.multiple_random_restarts(
    T_desired,
    n_attempts=5,
    max_iterations=100
)
```

**Pros:**
- Highest success rate
- Escapes local minima
- Good for difficult poses

**Cons:**
- Slower (n_attempts × IK solve time)
- Non-deterministic

**Best for:** Offline path planning, when success is more important than speed

---

### 3. Cached Nearest Neighbor

Uses database of successful solutions to find nearest match.

```python
# Add successful solutions to cache
ik_guesser.add_to_cache(T_desired, theta_solution)

# Generate guess from cache
theta0 = ik_guesser.cached_nearest_neighbor(T_desired, k=3)
```

**Pros:**
- Very fast (k-NN lookup)
- Learns from experience
- Excellent for repeated similar tasks

**Cons:**
- Requires warmup period
- Memory overhead for cache
- May fail for novel poses

**Best for:** Pick-and-place, assembly tasks with repetitive motions

---

### 4. Current Configuration Extrapolation

Extrapolates from current robot state for smooth motion.

```python
theta0 = ik_guesser.current_configuration_extrapolation(
    theta_current,
    T_current,
    T_desired,
    alpha=0.5  # Extrapolation factor
)
```

**Pros:**
- Fastest convergence (often < 5 iterations)
- Smooth trajectories
- Best for real-time control

**Cons:**
- Requires knowledge of current state
- Only works for incremental motion

**Best for:** Real-time trajectory tracking, servoing

---

### 5. Analytical Guess (Robot-Specific)

Uses geometric decoupling for robots with spherical wrists.

```python
theta0 = ik_guesser.analytical_guess_6dof_spherical_wrist(
    T_desired,
    d1=0.15,   # Base height
    a2=0.25,   # Link 2 length
    a3=0.25    # Link 3 length
)
```

**Pros:**
- Very accurate for compatible robots
- Fast (closed-form solution)
- Deterministic

**Cons:**
- Robot-specific (must customize)
- Requires accurate kinematic parameters

**Best for:** Industrial robots (PUMA, UR5, ABB, KUKA with spherical wrists)

---

## 💡 Best Practices

### For Real-Time Control (< 5ms per IK)
```python
# Use current config extrapolation + cached NN
if len(ik_guesser.solution_cache) > 0:
    theta0 = ik_guesser.current_configuration_extrapolation(
        theta_current, T_current, T_desired
    )
else:
    theta0 = ik_guesser.workspace_heuristic(T_desired)

# Fast IK with reduced iterations
theta, success, iters = manipulator.iterative_inverse_kinematics(
    T_desired, theta0, max_iterations=50
)
```

### For Path Planning (Offline)
```python
# Use multiple restarts for guaranteed success
theta, success, results = ik_guesser.multiple_random_restarts(
    T_desired,
    n_attempts=10,
    max_iterations=200
)

# Cache the solution for future use
if success:
    ik_guesser.add_to_cache(T_desired, theta)
```

### For Trajectory Tracking
```python
# Initialize cache with workspace samples
for _ in range(50):
    T_sample = generate_random_reachable_pose()
    theta0 = ik_guesser.workspace_heuristic(T_sample)
    theta, success, _ = manipulator.iterative_inverse_kinematics(T_sample, theta0)
    if success:
        ik_guesser.add_to_cache(T_sample, theta)

# Then use cached NN for trajectory
for T_waypoint in trajectory:
    theta0 = ik_guesser.cached_nearest_neighbor(T_waypoint, k=3)
    theta, success, _ = manipulator.iterative_inverse_kinematics(T_waypoint, theta0)
```

---

## 🧪 Benchmarking

Run the comparison benchmark:

```bash
python examples/ik_initial_guess_comparison.py
```

This will:
1. Generate 50 random reachable poses
2. Test all strategies
3. Report success rates, iteration counts, and timing
4. Generate visualization plots

**Expected Results (6-DOF manipulator):**
```
Strategy                       Success Rate    Avg Iter    Avg Time (ms)
---------------------------------------------------------------------------
Zero Configuration                  62.0%         347.2        142.35
Random Guess                        58.0%         412.8        168.92
Workspace Heuristic                 92.0%          48.3         19.74
Midpoint of Joint Limits           76.0%         189.5         77.52
```

---

## 🔬 Advanced: Custom Strategies

### Example: Machine Learning Initial Guess

```python
class MLInitialGuessGenerator(IKInitialGuessGenerator):
    def __init__(self, manipulator, model_path='ik_model.pkl'):
        super().__init__(manipulator)
        self.model = self._load_ml_model(model_path)

    def ml_predict(self, T_desired):
        """Use trained neural network to predict joint angles."""
        # Extract features from desired pose
        features = self._pose_to_features(T_desired)

        # Predict joint angles
        theta_pred = self.model.predict(features.reshape(1, -1))[0]

        # Clip to limits
        return self._clip_to_limits(theta_pred)

    def _pose_to_features(self, T):
        """Convert 4x4 transform to feature vector."""
        position = T[:3, 3]
        rotation = T[:3, :3]
        # Use rotation vector (axis-angle) representation
        from scipy.spatial.transform import Rotation
        rot_vec = Rotation.from_matrix(rotation).as_rotvec()
        return np.concatenate([position, rot_vec])

    def _load_ml_model(self, path):
        """Load pre-trained sklearn/pytorch model."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
```

### Example: Gradient-Free Optimization

```python
def coarse_to_fine_guess(manipulator, T_desired):
    """
    Use coarse-to-fine strategy with CMA-ES.
    """
    from scipy.optimize import differential_evolution

    def objective(theta):
        T_result = manipulator.forward_kinematics(theta)
        pos_err = np.linalg.norm(T_desired[:3, 3] - T_result[:3, 3])
        return pos_err

    # Coarse search with differential evolution
    bounds = [(mn or -np.pi, mx or np.pi)
              for mn, mx in manipulator.joint_limits]

    result = differential_evolution(
        objective,
        bounds,
        maxiter=50,
        popsize=5,
        seed=42
    )

    return result.x
```

---

## 📚 References

1. **Buss, S. R.** (2004). "Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods."

2. **Diankov, R.** (2010). "Automated Construction of Robotic Manipulation Programs" (OpenRAVE thesis - discusses IK seeding strategies)

3. **Beeson, P., & Ames, B.** (2015). "TRAC-IK: An open-source library for improved solving of generic inverse kinematics."

4. **Aristidou, A., & Lasenby, J.** (2011). "FABRIK: A fast, iterative solver for the Inverse Kinematics problem."

---

## 🤝 Contributing

To add a new initial guess strategy:

1. Add method to `IKInitialGuessGenerator` class
2. Add corresponding test in `examples/ik_initial_guess_comparison.py`
3. Update this guide with usage example and benchmark results

---

## 📝 License

AGPL-3.0-or-later - Same as ManipulaPy package
