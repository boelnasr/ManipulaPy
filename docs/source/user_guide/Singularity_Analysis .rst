Singularity Analysis User Guide
================================

The Singularity Analysis module provides comprehensive tools for analyzing and visualizing
robot manipulator singularities, workspace boundaries, and manipulability characteristics.
Understanding singularities is crucial for robust robot control, trajectory planning,
and workspace optimization.

.. note::
   This guide assumes familiarity with robotics fundamentals, linear algebra, and 
   Python 3.10+ with NumPy, SciPy, Matplotlib, and optionally CUDA/Numba for acceleration.

.. contents:: Table of Contents
   :depth: 2
   :local:

Theoretical Background
----------------------

What are Singularities?
~~~~~~~~~~~~~~~~~~~~~~~~

A **singularity** occurs when a robot manipulator loses one or more degrees of freedom,
meaning the end-effector cannot move in certain directions regardless of joint velocities.
Mathematically, this happens when the Jacobian matrix becomes rank-deficient (determinant ≈ 0).

**Types of Singularities:**

1. **Boundary Singularities**: Occur at workspace boundaries when the arm is fully extended
2. **Interior Singularities**: Occur within the workspace when joints align in specific configurations
3. **Wrist Singularities**: Occur when wrist axes align, losing rotational degrees of freedom

**Physical Consequences:**

- Infinite joint velocities required for certain end-effector motions
- Loss of controllability in specific directions
- Numerical instability in inverse kinematics
- Reduced manipulability and dexterity

Mathematical Foundations
~~~~~~~~~~~~~~~~~~~~~~~~

**Jacobian Matrix**

The velocity relationship between joint space and Cartesian space:

.. math::

   \dot{\mathbf{x}} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}

where:
- :math:`\dot{\mathbf{x}} \in \mathbb{R}^6` is the end-effector twist (linear + angular velocity)
- :math:`\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6 \times n}` is the Jacobian matrix
- :math:`\dot{\mathbf{q}} \in \mathbb{R}^n` is the joint velocity vector

**Singularity Detection**

A configuration is singular when:

.. math::

   \det(\mathbf{J}) = 0 \quad \text{or} \quad \sigma_{\min}(\mathbf{J}) < \epsilon

where :math:`\sigma_{\min}` is the smallest singular value and :math:`\epsilon` is a small threshold.

**Manipulability Measure**

The manipulability ellipsoid describes the robot's velocity capabilities:

.. math::

   w(\mathbf{q}) = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}

**Condition Number**

Measures how close a configuration is to singularity:

.. math::

   \kappa(\mathbf{J}) = \frac{\sigma_{\max}(\mathbf{J})}{\sigma_{\min}(\mathbf{J})}

Installation and Setup
----------------------

Prerequisites
~~~~~~~~~~~~~

.. code-block:: bash

    # Core dependencies
    pip install ManipulaPy numpy scipy matplotlib

    # Optional GPU acceleration
    pip install numba cupy-cuda11x  # or cupy-cuda12x

    # 3D visualization enhancements
    pip install plotly vtk mayavi

Verification
~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from ManipulaPy.singularity import Singularity
    from ManipulaPy.kinematics import SerialManipulator
    
    # Test CUDA availability
    try:
        from numba import cuda
        print(f"CUDA available: {cuda.is_available()}")
        if cuda.is_available():
            print(f"GPU devices: {cuda.gpus}")
    except ImportError:
        print("Numba/CUDA not available - using CPU only")

Quick Start
-----------

Basic Singularity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ManipulaPy.singularity import Singularity
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    import numpy as np

    # Load robot model
    urdf_processor = URDFToSerialManipulator("robot.urdf")
    robot = urdf_processor.serial_manipulator

    # Create singularity analyzer
    singularity_analyzer = Singularity(robot)

    # Test configuration
    joint_angles = np.array([0.5, -0.3, 0.8, 0, -0.5, 0])

    # Check for singularity
    is_singular = singularity_analyzer.singularity_analysis(joint_angles)
    print(f"Configuration is singular: {is_singular}")

    # Calculate condition number
    condition_num = singularity_analyzer.condition_number(joint_angles)
    print(f"Condition number: {condition_num:.2f}")

    # Check proximity to singularity
    near_singular = singularity_analyzer.near_singularity_detection(
        joint_angles, threshold=100
    )
    print(f"Near singularity: {near_singular}")

Manipulability Ellipsoid Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    # Create 3D visualization
    fig = plt.figure(figsize=(15, 6))

    # Compare different configurations
    configs = [
        np.array([0, 0, 0, 0, 0, 0]),           # Home position
        np.array([1.5, 0, 0, 0, 0, 0]),        # Extended arm
        np.array([0, 1.57, -1.57, 0, 0, 0])    # Folded configuration
    ]

    for i, config in enumerate(configs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        singularity_analyzer.manipulability_ellipsoid(config, ax=ax)
        ax.set_title(f"Configuration {i+1}")

    plt.tight_layout()
    plt.show()

Module API Reference
--------------------

Singularity Class
~~~~~~~~~~~~~~~~~~

.. autoclass:: ManipulaPy.singularity.Singularity
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Core Analysis Methods
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: __init__(serial_manipulator)

   Initialize singularity analysis for a given robot.

   :param SerialManipulator serial_manipulator: Robot model from ManipulaPy.kinematics
   :raises TypeError: If serial_manipulator is not a valid SerialManipulator instance

.. py:method:: singularity_analysis(thetalist) -> bool

   Detect exact singularities using Jacobian determinant.

   :param np.ndarray thetalist: Joint angles in radians (shape: n_joints,)
   :return: True if configuration is singular
   :rtype: bool

   Uses threshold of 1e-4 for determinant comparison. For more nuanced analysis,
   use :py:meth:`condition_number` or :py:meth:`near_singularity_detection`.

   .. code-block:: python

       # Example: Check multiple configurations
       test_configs = [
           np.zeros(6),                    # Home
           np.array([0, np.pi/2, 0, 0, 0, 0]),  # Shoulder singularity
           np.array([0, 0, 0, 0, np.pi/2, 0])   # Wrist singularity
       ]
       
       for i, config in enumerate(test_configs):
           singular = analyzer.singularity_analysis(config)
           print(f"Config {i+1} singular: {singular}")

.. py:method:: condition_number(thetalist) -> float

   Calculate Jacobian condition number for singularity proximity assessment.

   :param np.ndarray thetalist: Joint angles in radians
   :return: Condition number (≥1, higher values indicate closer to singularity)
   :rtype: float

   Condition numbers interpretation:
   - κ ≈ 1: Well-conditioned, far from singularity
   - κ = 10-100: Moderately conditioned
   - κ > 100: Ill-conditioned, approaching singularity
   - κ → ∞: At singularity

.. py:method:: near_singularity_detection(thetalist, threshold=100) -> bool

   Detect proximity to singularities using condition number threshold.

   :param np.ndarray thetalist: Joint angles in radians
   :param float threshold: Condition number threshold (default: 100)
   :return: True if near singularity
   :rtype: bool

   .. note::
      Typical thresholds:
      - 100: Close to singularity
      - 1000: Very close to singularity
      - 10000: Extremely close to singularity

Visualization Methods
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: manipulability_ellipsoid(thetalist, ax=None)

   Visualize manipulability ellipsoids for linear and angular velocities.

   :param np.ndarray thetalist: Joint angles in radians
   :param matplotlib.axes.Axes3D ax: Optional 3D axis for plotting
   
   Creates two ellipsoids:
   - **Blue ellipsoid**: Linear velocity capabilities
   - **Red ellipsoid**: Angular velocity capabilities

   Ellipsoid properties:
   - **Volume**: Overall manipulability measure
   - **Shape**: Directional velocity capabilities
   - **Orientation**: Principal motion directions

.. py:method:: plot_workspace_monte_carlo(joint_limits, num_samples=10000)

   Generate workspace boundary using Monte Carlo sampling with optional GPU acceleration.

   :param list joint_limits: List of (min, max) tuples for each joint
   :param int num_samples: Number of random samples (default: 10000)

   Creates 3D convex hull visualization of reachable workspace. Uses CUDA acceleration
   when available for faster sampling of large point clouds.

   .. code-block:: python

       # High-resolution workspace analysis
       joint_limits = [(-np.pi, np.pi)] * 6
       analyzer.plot_workspace_monte_carlo(
           joint_limits, 
           num_samples=100000  # High resolution
       )

Advanced Analysis Examples
--------------------------

Comprehensive Singularity Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create detailed singularity maps across the workspace:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def create_singularity_map(analyzer, joint_ranges, resolution=20):
        """Create 2D singularity map varying two joints."""
        
        # Define sweep ranges for first two joints
        q1_range = np.linspace(joint_ranges[0][0], joint_ranges[0][1], resolution)
        q2_range = np.linspace(joint_ranges[1][0], joint_ranges[1][1], resolution)
        
        # Initialize condition number grid
        condition_map = np.zeros((resolution, resolution))
        
        # Fixed values for other joints
        q_fixed = np.zeros(6)
        
        for i, q1 in enumerate(q1_range):
            for j, q2 in enumerate(q2_range):
                q_test = q_fixed.copy()
                q_test[0] = q1
                q_test[1] = q2
                
                try:
                    condition_map[j, i] = analyzer.condition_number(q_test)
                except np.linalg.LinAlgError:
                    condition_map[j, i] = 1e6  # Singular configuration
        
        return q1_range, q2_range, condition_map

    # Create and plot singularity map
    q1_vals, q2_vals, cond_map = create_singularity_map(
        singularity_analyzer, 
        [(-np.pi, np.pi), (-np.pi, np.pi)],
        resolution=50
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(q1_vals, q2_vals, cond_map, 
                     levels=50, norm=LogNorm(vmin=1, vmax=1e4))
    ax.contour(q1_vals, q2_vals, cond_map, levels=[100, 1000], 
               colors=['red', 'darkred'], linewidths=2)
    
    ax.set_xlabel('Joint 1 Angle (rad)')
    ax.set_ylabel('Joint 2 Angle (rad)')
    ax.set_title('Singularity Map (Condition Number)')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Condition Number')
    
    # Add singularity contour labels
    ax.text(0.02, 0.95, 'Red: κ=100\nDark Red: κ=1000', 
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.show()

Trajectory Singularity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze singularities along planned trajectories:

.. code-block:: python

    def analyze_trajectory_singularities(analyzer, trajectory, time_steps=None):
        """Analyze singularities along a trajectory."""
        
        n_points = len(trajectory)
        if time_steps is None:
            time_steps = np.linspace(0, 1, n_points)
        
        # Compute metrics along trajectory
        condition_numbers = []
        singular_points = []
        manipulability_measures = []
        
        for i, config in enumerate(trajectory):
            # Condition number
            cond_num = analyzer.condition_number(config)
            condition_numbers.append(cond_num)
            
            # Singularity detection
            is_singular = analyzer.singularity_analysis(config)
            singular_points.append(is_singular)
            
            # Manipulability measure
            J = analyzer.serial_manipulator.jacobian(config)
            manipulability = np.sqrt(np.linalg.det(J @ J.T))
            manipulability_measures.append(manipulability)
        
        return {
            'time': time_steps,
            'condition_numbers': np.array(condition_numbers),
            'singular_points': np.array(singular_points),
            'manipulability': np.array(manipulability_measures)
        }

    def plot_trajectory_analysis(analysis_results):
        """Plot trajectory singularity analysis results."""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        time = analysis_results['time']
        
        # Condition number plot
        axes[0].semilogy(time, analysis_results['condition_numbers'], 'b-', linewidth=2)
        axes[0].axhline(y=100, color='orange', linestyle='--', label='Warning (κ=100)')
        axes[0].axhline(y=1000, color='red', linestyle='--', label='Critical (κ=1000)')
        axes[0].set_ylabel('Condition Number')
        axes[0].set_title('Trajectory Singularity Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Singular points
        singular_indices = np.where(analysis_results['singular_points'])[0]
        if len(singular_indices) > 0:
            axes[0].scatter(time[singular_indices], 
                           analysis_results['condition_numbers'][singular_indices],
                           color='red', s=100, marker='x', label='Singular Points')
        
        # Manipulability measure
        axes[1].plot(time, analysis_results['manipulability'], 'g-', linewidth=2)
        axes[1].set_ylabel('Manipulability Measure')
        axes[1].grid(True, alpha=0.3)
        
        # Velocity scaling factor (inverse condition number)
        velocity_scaling = 1.0 / analysis_results['condition_numbers']
        axes[2].plot(time, velocity_scaling, 'm-', linewidth=2)
        axes[2].set_ylabel('Max Velocity Scaling')
        axes[2].set_xlabel('Trajectory Parameter')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Real-Time Singularity Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement real-time singularity monitoring for robot control:

.. code-block:: python

    import time
    import threading
    from collections import deque

    class SingularityMonitor:
        """Real-time singularity monitoring system."""
        
        def __init__(self, analyzer, warning_threshold=100, critical_threshold=1000):
            self.analyzer = analyzer
            self.warning_threshold = warning_threshold
            self.critical_threshold = critical_threshold
            
            # Monitoring state
            self.is_monitoring = False
            self.monitor_thread = None
            
            # Data storage
            self.condition_history = deque(maxlen=1000)
            self.time_history = deque(maxlen=1000)
            self.alerts = deque(maxlen=100)
            
        def start_monitoring(self, update_rate=10):
            """Start real-time monitoring."""
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(update_rate,)
            )
            self.monitor_thread.start()
            print(f"Singularity monitoring started at {update_rate} Hz")
            
        def stop_monitoring(self):
            """Stop monitoring."""
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
            print("Singularity monitoring stopped")
            
        def _monitor_loop(self, update_rate):
            """Main monitoring loop."""
            dt = 1.0 / update_rate
            
            while self.is_monitoring:
                start_time = time.time()
                
                # Get current robot configuration (placeholder)
                current_config = self._get_current_configuration()
                
                # Analyze singularity
                condition_num = self.analyzer.condition_number(current_config)
                
                # Store data
                self.condition_history.append(condition_num)
                self.time_history.append(time.time())
                
                # Check thresholds
                if condition_num > self.critical_threshold:
                    alert = {
                        'time': time.time(),
                        'level': 'CRITICAL',
                        'condition_number': condition_num,
                        'config': current_config.copy()
                    }
                    self.alerts.append(alert)
                    print(f"CRITICAL: Condition number = {condition_num:.1f}")
                    
                elif condition_num > self.warning_threshold:
                    alert = {
                        'time': time.time(),
                        'level': 'WARNING', 
                        'condition_number': condition_num,
                        'config': current_config.copy()
                    }
                    self.alerts.append(alert)
                    print(f"WARNING: Condition number = {condition_num:.1f}")
                
                # Maintain update rate
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        def _get_current_configuration(self):
            """Get current robot configuration (placeholder)."""
            # This would interface with actual robot hardware
            t = time.time()
            return np.array([
                0.5 * np.sin(0.1 * t),
                0.3 * np.cos(0.15 * t),
                0.4 * np.sin(0.08 * t),
                0.2 * np.cos(0.12 * t),
                0.1 * np.sin(0.2 * t),
                0.05 * np.cos(0.25 * t)
            ])

Performance Optimization
------------------------

Computational Efficiency Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Jacobian Caching**

Cache Jacobian computations for repeated configurations:

.. code-block:: python

    class CachedSingularityAnalyzer:
        """Singularity analyzer with Jacobian caching."""
        
        def __init__(self, serial_manipulator, cache_size=1000):
            self.analyzer = Singularity(serial_manipulator)
            self.jacobian_cache = {}
            self.cache_size = cache_size
            self.cache_hits = 0
            self.cache_misses = 0
            
        def _config_key(self, thetalist, precision=1e-6):
            """Create hashable key for configuration."""
            return tuple(np.round(thetalist / precision) * precision)
            
        def _get_jacobian(self, thetalist):
            """Get Jacobian with caching."""
            key = self._config_key(thetalist)
            
            if key in self.jacobian_cache:
                self.cache_hits += 1
                return self.jacobian_cache[key]
            
            # Compute Jacobian
            J = self.analyzer.serial_manipulator.jacobian(thetalist)
            
            # Manage cache size
            if len(self.jacobian_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.jacobian_cache))
                del self.jacobian_cache[oldest_key]
            
            self.jacobian_cache[key] = J
            self.cache_misses += 1
            return J

**2. Batch Processing**

Process multiple configurations efficiently:

.. code-block:: python

    def batch_singularity_analysis(analyzer, configurations, batch_size=100):
        """Efficiently analyze multiple configurations."""
        
        results = {
            'condition_numbers': [],
            'singular_flags': [],
            'manipulability': []
        }
        
        for i in range(0, len(configurations), batch_size):
            batch = configurations[i:i+batch_size]
            
            # Process batch
            for config in batch:
                # Condition number
                cond_num = analyzer.condition_number(config)
                results['condition_numbers'].append(cond_num)
                
                # Singularity flag
                singular = cond_num > 1000  # Threshold
                results['singular_flags'].append(singular)
                
                # Manipulability
                J = analyzer.serial_manipulator.jacobian(config)
                manip = np.sqrt(np.linalg.det(J @ J.T))
                results['manipulability'].append(manip)
            
            # Progress update
            print(f"Processed {min(i+batch_size, len(configurations))}/{len(configurations)} configurations")
        
        return {k: np.array(v) for k, v in results.items()}

Troubleshooting Guide
---------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Numerical Instability**

Problem: Condition numbers become infinite or NaN

.. code-block:: python

    def robust_condition_number(analyzer, thetalist, max_condition=1e12):
        """Robust condition number computation with error handling."""
        try:
            J = analyzer.serial_manipulator.jacobian(thetalist)
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(J)):
                return max_condition
            
            # SVD-based condition number
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
            
            # Filter out very small singular values
            s_filtered = s[s > 1e-15]
            if len(s_filtered) < len(s):
                return max_condition
            
            condition = s_filtered[0] / s_filtered[-1]
            return min(condition, max_condition)
            
        except (np.linalg.LinAlgError, ValueError):
            return max_condition

**2. Performance Issues**

Problem: Slow computation for large workspaces

Solutions:
- Use smaller sample sizes for initial exploration
- Enable CUDA acceleration if available
- Cache Jacobian computations for repeated configurations
- Use parallel processing for batch analysis

**3. Memory Issues**

Problem: Out of memory for large datasets

Solutions:
- Process data in smaller chunks
- Use streaming analysis to disk
- Reduce precision of stored data
- Clear intermediate variables regularly

**4. Visualization Issues**

Problem: Poor visualization performance or cluttered plots

Solutions:
- Reduce number of plotted points
- Use adaptive sampling strategies
- Implement level-of-detail rendering
- Save plots to files instead of displaying interactively

Validation and Testing
----------------------

Unit Testing Framework
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import unittest

    class TestSingularityAnalysis(unittest.TestCase):
        """Unit tests for singularity analysis module."""
        
        def setUp(self):
            """Set up test fixtures."""
            from ManipulaPy.urdf_processor import URDFToSerialManipulator
            
            self.urdf_processor = URDFToSerialManipulator("test_robot.urdf")
            self.robot = self.urdf_processor.serial_manipulator
            self.analyzer = Singularity(self.robot)
            
        def test_condition_number_positive(self):
            """Test that condition numbers are always positive."""
            test_configs = [
                np.zeros(6),
                np.array([0.5, -0.3, 0.8, 0.2, -0.1, 0.4]),
                np.array([1.5, 0.8, -0.5, 0.3, -0.2, 0.6])
            ]
            
            for config in test_configs:
                condition_num = self.analyzer.condition_number(config)
                self.assertGreater(condition_num, 0, "Condition number must be positive")
                self.assertTrue(np.isfinite(condition_num), "Condition number must be finite")

Best Practices
--------------

Singularity-Aware Robot Programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Always check condition numbers** before executing trajectories
2. **Use manipulability measures** for trajectory optimization
3. **Implement singularity avoidance** in motion planning
4. **Monitor real-time singularity metrics** during operation
5. **Validate analysis results** with known test cases

Integration Guidelines
~~~~~~~~~~~~~~~~~~~~~~

- **Trajectory Planning**: Integrate singularity analysis into path optimization
- **Control Systems**: Use condition numbers for adaptive control gains
- **Safety Systems**: Implement singularity-based motion limits
- **Performance Optimization**: Cache computations for repeated analyses

See Also
--------

- :doc:`/user_guide/Kinematics` — Forward and Inverse Kinematics
- :doc:`/user_guide/Trajectory_Planning` — Path Planning with Singularity Avoidance
- :doc:`/user_guide/Control` — Singularity-Robust Control Strategies
- :doc:`/api/singularity` — Complete API Reference
- `NumPy Linear Algebra Documentation <https://numpy.org/doc/stable/reference/routines.linalg.html>`_
- `SciPy Spatial Documentation <https://docs.scipy.org/doc/scipy/reference/spatial.html>`_