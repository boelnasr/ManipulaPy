# Integration Example: Adding Smart IK Initial Guess to SerialManipulator

This document shows how to integrate the IK initial guess strategies directly into the `SerialManipulator` class.

## Option 1: Add as Helper Method (Recommended)

Add this method to `ManipulaPy/kinematics.py` in the `SerialManipulator` class:

```python
def smart_inverse_kinematics(
    self,
    T_desired: NDArray[np.float64],
    theta_current: Optional[Union[NDArray[np.float64], List[float]]] = None,
    strategy: str = "auto",
    **ik_kwargs
) -> Tuple[NDArray[np.float64], bool, int]:
    """
    Inverse kinematics with smart initial guess selection.

    Automatically chooses the best initial guess strategy based on
    available information and desired performance.

    Args:
        T_desired: Desired 4x4 transformation matrix
        theta_current: Current joint angles (optional, for extrapolation)
        strategy: Initial guess strategy:
            - "auto": Automatically select best strategy (default)
            - "workspace": Geometric workspace heuristic
            - "extrapolate": Extrapolate from current config (requires theta_current)
            - "zeros": Start from zero configuration
            - "midpoint": Start from midpoint of joint limits
        **ik_kwargs: Additional arguments passed to iterative_inverse_kinematics

    Returns:
        (theta, success, iterations)

    Example:
        >>> T_target = np.eye(4)
        >>> T_target[:3, 3] = [0.3, 0.2, 0.4]
        >>>
        >>> # Simple usage (auto strategy)
        >>> theta, success, iters = robot.smart_inverse_kinematics(T_target)
        >>>
        >>> # With current configuration (for smooth trajectories)
        >>> theta, success, iters = robot.smart_inverse_kinematics(
        ...     T_target,
        ...     theta_current=robot_current_angles,
        ...     strategy="extrapolate"
        ... )
    """
    # Generate initial guess based on strategy
    if strategy == "auto":
        if theta_current is not None:
            # Use extrapolation if current config available
            theta0 = self._extrapolate_initial_guess(theta_current, T_desired)
        else:
            # Otherwise use workspace heuristic
            theta0 = self._workspace_heuristic_guess(T_desired)

    elif strategy == "workspace":
        theta0 = self._workspace_heuristic_guess(T_desired)

    elif strategy == "extrapolate":
        if theta_current is None:
            raise ValueError("strategy='extrapolate' requires theta_current")
        theta0 = self._extrapolate_initial_guess(theta_current, T_desired)

    elif strategy == "zeros":
        n_joints = len(self.joint_limits)
        theta0 = np.zeros(n_joints)

    elif strategy == "midpoint":
        theta0 = np.array([
            (mn + mx) / 2 if mn is not None and mx is not None else 0
            for mn, mx in self.joint_limits
        ])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clip to joint limits
    theta0 = self._clip_to_limits(theta0)

    # Run IK with the smart initial guess
    return self.iterative_inverse_kinematics(T_desired, theta0, **ik_kwargs)


def _workspace_heuristic_guess(self, T_desired: NDArray[np.float64]) -> NDArray[np.float64]:
    """Generate initial guess using workspace heuristic."""
    n_joints = len(self.joint_limits)
    theta = np.zeros(n_joints)

    # Extract desired position
    p = T_desired[:3, 3]

    # Joint 1: rotation in XY plane
    if n_joints >= 1:
        theta[0] = np.arctan2(p[1], p[0])

    # Joint 2: elevation angle
    if n_joints >= 2:
        r_xy = np.sqrt(p[0]**2 + p[1]**2)
        theta[1] = np.arctan2(p[2], r_xy)

    # Joint 3: elbow configuration (heuristic)
    if n_joints >= 3:
        theta[2] = np.pi / 4  # 45° as neutral

    # Joints 4-6: wrist orientation (simplified)
    if n_joints > 3:
        R = T_desired[:3, :3]
        # ZYZ Euler angles
        if np.abs(R[2, 2]) < 0.9999:
            theta[4] = np.arccos(np.clip(R[2, 2], -1, 1))
            theta[3] = np.arctan2(R[1, 2], R[0, 2])
            theta[5] = np.arctan2(R[2, 1], -R[2, 0]) if n_joints > 5 else 0
        else:
            theta[3] = np.arctan2(R[1, 0], R[0, 0])
            theta[4] = 0
            theta[5] = 0

    return theta


def _extrapolate_initial_guess(
    self,
    theta_current: Union[NDArray[np.float64], List[float]],
    T_desired: NDArray[np.float64],
    alpha: float = 0.5
) -> NDArray[np.float64]:
    """Extrapolate initial guess from current configuration."""
    theta_current = np.array(theta_current, dtype=float)

    # Current pose
    T_current = self.forward_kinematics(theta_current)

    # Pose error
    T_err = T_desired @ np.linalg.inv(T_current)

    # Extract twist
    V_err = utils.se3ToVec(utils.MatrixLog6(T_err))

    # Estimate joint velocity
    J = self.jacobian(theta_current)
    dtheta = np.linalg.pinv(J) @ V_err

    # Extrapolate
    theta_guess = theta_current + alpha * dtheta

    return theta_guess


def _clip_to_limits(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip joint angles to limits."""
    theta_clipped = theta.copy()
    for i, (mn, mx) in enumerate(self.joint_limits):
        if mn is not None:
            theta_clipped[i] = max(theta_clipped[i], mn)
        if mx is not None:
            theta_clipped[i] = min(theta_clipped[i], mx)
    return theta_clipped
```

---

## Option 2: Use as Standalone Module (No Code Changes)

Keep the code in `ik_initial_guess_strategies.py` and import as needed:

```python
from ManipulaPy.kinematics import SerialManipulator
from ik_initial_guess_strategies import IKInitialGuessGenerator

# Your robot setup
robot = SerialManipulator(...)

# Create guess generator
ik_gen = IKInitialGuessGenerator(robot)

# Use it
T_target = np.eye(4)
T_target[:3, 3] = [0.3, 0.2, 0.4]

theta0 = ik_gen.workspace_heuristic(T_target)
theta, success, iters = robot.iterative_inverse_kinematics(T_target, theta0)
```

---

## Usage Examples

### Example 1: Simple Single IK Solve

```python
import numpy as np
from ManipulaPy.kinematics import SerialManipulator

# Create robot
robot = SerialManipulator(M_list, omega_list, r_list, joint_limits=limits)

# Target pose
T_target = np.eye(4)
T_target[:3, 3] = [0.4, 0.1, 0.3]

# Solve IK with smart initial guess (using Option 1)
theta, success, iterations = robot.smart_inverse_kinematics(T_target)

print(f"Success: {success}")
print(f"Solution: {theta}")
print(f"Iterations: {iterations}")
```

### Example 2: Trajectory Tracking

```python
# Waypoints for trajectory
waypoints = [
    np.array([0.3, 0.0, 0.4]),
    np.array([0.3, 0.1, 0.4]),
    np.array([0.3, 0.2, 0.4]),
    # ... more waypoints
]

# Track trajectory
theta_current = np.zeros(6)  # Start from home

trajectory_solution = []
for i, point in enumerate(waypoints):
    # Create target pose
    T_target = np.eye(4)
    T_target[:3, 3] = point

    # Solve using extrapolation from current config
    theta, success, iters = robot.smart_inverse_kinematics(
        T_target,
        theta_current=theta_current,
        strategy="extrapolate",
        max_iterations=50  # Fast convergence expected
    )

    if success:
        trajectory_solution.append(theta)
        theta_current = theta  # Update for next waypoint
        print(f"Waypoint {i}: {iters} iterations")
    else:
        print(f"Waypoint {i}: FAILED")
        break
```

### Example 3: Multiple Attempts with Fallback

```python
def robust_ik_solve(robot, T_target, max_attempts=3):
    """
    Try multiple strategies with fallback.
    """
    strategies = ["auto", "workspace", "midpoint"]

    for i, strategy in enumerate(strategies):
        theta, success, iters = robot.smart_inverse_kinematics(
            T_target,
            strategy=strategy,
            max_iterations=200
        )

        if success:
            print(f"Success with strategy '{strategy}' ({iters} iters)")
            return theta, True

        print(f"Attempt {i+1} failed (strategy: {strategy})")

    print("All attempts failed")
    return None, False


# Use it
theta, success = robust_ik_solve(robot, T_target)
```

### Example 4: Benchmark Improvement

```python
import time

# Old way (zero initial guess)
theta0_old = np.zeros(6)
start = time.time()
theta_old, success_old, iters_old = robot.iterative_inverse_kinematics(
    T_target, theta0_old
)
time_old = time.time() - start

# New way (smart initial guess)
start = time.time()
theta_new, success_new, iters_new = robot.smart_inverse_kinematics(T_target)
time_new = time.time() - start

print("Comparison:")
print(f"  Old: {iters_old} iterations, {time_old*1000:.2f}ms, success={success_old}")
print(f"  New: {iters_new} iterations, {time_new*1000:.2f}ms, success={success_new}")
print(f"  Improvement: {(iters_old-iters_new)/iters_old*100:.1f}% fewer iterations")
```

---

## Migration Path

### Step 1: Test with Standalone Module
1. Use `ik_initial_guess_strategies.py` as-is
2. Benchmark on your specific robot
3. Identify best strategies for your use case

### Step 2: Integrate into Codebase (Optional)
1. Add methods to `SerialManipulator` class
2. Update existing code to use `smart_inverse_kinematics()`
3. Keep old method for backward compatibility

### Step 3: Optimize Further
1. Train ML model on your specific robot
2. Build solution cache for common tasks
3. Profile and tune damping/step size parameters

---

## Performance Tips

1. **For real-time control** (< 5ms):
   - Use `strategy="extrapolate"` with current config
   - Set `max_iterations=50` or less
   - Pre-warm solution cache

2. **For offline planning** (success rate critical):
   - Use multiple random restarts
   - Increase `max_iterations=1000`
   - Verify solutions meet constraints

3. **For pick-and-place** (repetitive tasks):
   - Build solution cache during warmup
   - Use `cached_nearest_neighbor`
   - Periodically clear stale cache entries

---

## Testing

Add unit tests to verify integration:

```python
def test_smart_ik_strategies():
    """Test all smart IK strategies."""
    robot = create_test_robot()

    T_target = np.eye(4)
    T_target[:3, 3] = [0.3, 0.2, 0.4]

    strategies = ["auto", "workspace", "zeros", "midpoint"]

    for strategy in strategies:
        theta, success, iters = robot.smart_inverse_kinematics(
            T_target,
            strategy=strategy,
            max_iterations=500
        )

        # Verify solution if successful
        if success:
            T_result = robot.forward_kinematics(theta)
            error = np.linalg.norm(T_target[:3, 3] - T_result[:3, 3])
            assert error < 1e-3, f"Strategy {strategy} failed accuracy test"

        print(f"Strategy '{strategy}': success={success}, iters={iters}")
```

---

## Troubleshooting

**Q: IK still fails frequently**
- Check joint limits are correctly specified
- Verify target pose is reachable
- Increase `max_iterations`
- Try multiple random restarts

**Q: Too slow for real-time**
- Use extrapolation strategy
- Reduce `max_iterations` to 20-50
- Profile to find bottlenecks (likely in FK/Jacobian)

**Q: Solutions are jerky**
- Use extrapolation with smaller `alpha` (0.3-0.5)
- Add joint velocity limits
- Smooth trajectory with interpolation

---

## Next Steps

1. Read `IK_INITIAL_GUESS_GUIDE.md` for detailed strategy descriptions
2. Run `examples/ik_initial_guess_comparison.py` to benchmark
3. Integrate the method that works best for your application
4. Profile and optimize further if needed

Happy coding! 🚀
