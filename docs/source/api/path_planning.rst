.. _api-path-planning:

===============================
Path Planning API Reference
===============================

This page documents **ManipulaPy.path_planning**, the module for optimized trajectory generation with adaptive GPU/CPU execution, CUDA acceleration, and collision avoidance.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Trajectory_Planning`.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

------------------------
OptimizedTrajectoryPlanning Class
------------------------

.. currentmodule:: ManipulaPy.path_planning

.. autoclass:: OptimizedTrajectoryPlanning
   :members:
   :show-inheritance:

   Highly optimized trajectory planning class with adaptive GPU/CPU execution, memory pooling, and batch processing capabilities.

   .. rubric:: Constructor

   .. automethod:: __init__

      **Parameters:**
        - **serial_manipulator** (*SerialManipulator*) -- Robot kinematics object
        - **urdf_path** (*str*) -- Path to URDF file for collision checking
        - **dynamics** (*ManipulatorDynamics*) -- Robot dynamics object
        - **joint_limits** (*list*) -- Joint limits as [(min, max), ...] tuples
        - **torque_limits** (*list*, optional) -- Torque limits as [(min, max), ...] tuples
        - **use_cuda** (*bool*, optional) -- Force GPU (True), CPU (False), or auto-detect (None)
        - **cuda_threshold** (*int*) -- Minimum problem size before using GPU (default: 50)
        - **memory_pool_size_mb** (*int*, optional) -- GPU memory pool size in MB
        - **enable_profiling** (*bool*) -- Enable CUDA profiling for performance analysis

      **Attributes Created:**
        - **collision_checker** (*CollisionChecker*) -- URDF-based collision detection
        - **potential_field** (*PotentialField*) -- Artificial potential field for obstacle avoidance
        - **cuda_available** (*bool*) -- Whether CUDA acceleration is available
        - **gpu_properties** (*dict*) -- GPU device properties for optimization
        - **performance_stats** (*dict*) -- Performance tracking for GPU/CPU usage

   .. rubric:: Core Trajectory Generation

   .. automethod:: joint_trajectory

      **Parameters:**
        - **thetastart** (*array_like*) -- Starting joint angles in radians
        - **thetaend** (*array_like*) -- Target joint angles in radians
        - **Tf** (*float*) -- Total trajectory time in seconds
        - **N** (*int*) -- Number of trajectory points
        - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

      **Returns:**
        - **trajectory** (*dict*) -- Dictionary containing:
          
          - **positions** (*numpy.ndarray*) -- Joint positions (N×n)
          - **velocities** (*numpy.ndarray*) -- Joint velocities (N×n)
          - **accelerations** (*numpy.ndarray*) -- Joint accelerations (N×n)

      **Adaptive Execution:** Automatically chooses GPU or CPU based on problem size and hardware availability.

      **CUDA Acceleration:** Uses optimized 2D parallelized kernels with shared memory optimization.

      **Collision Avoidance:** Automatically applies potential field-based collision avoidance.

   .. automethod:: batch_joint_trajectory

      **Parameters:**
        - **thetastart_batch** (*numpy.ndarray*) -- Starting angles for multiple trajectories (batch_size, num_joints)
        - **thetaend_batch** (*numpy.ndarray*) -- Ending angles for multiple trajectories (batch_size, num_joints)
        - **Tf** (*float*) -- Total trajectory time in seconds
        - **N** (*int*) -- Number of trajectory points
        - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

      **Returns:**
        - **trajectory** (*dict*) -- Batch trajectory data with shape (batch_size, N, num_joints)

      **3D Parallelization:** Uses CUDA 3D grids for (batch, time, joint) parallel computation.

      **Memory Optimization:** Efficient memory pooling for large batch processing.

   .. automethod:: cartesian_trajectory

      **Parameters:**
        - **Xstart** (*numpy.ndarray*) -- Initial SE(3) transformation matrix (4×4)
        - **Xend** (*numpy.ndarray*) -- Target SE(3) transformation matrix (4×4)
        - **Tf** (*float*) -- Total trajectory time in seconds
        - **N** (*int*) -- Number of trajectory points
        - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

      **Returns:**
        - **trajectory** (*dict*) -- Dictionary containing:
          
          - **positions** (*numpy.ndarray*) -- Cartesian positions (N×3)
          - **velocities** (*numpy.ndarray*) -- Linear velocities (N×3)
          - **accelerations** (*numpy.ndarray*) -- Linear accelerations (N×3)
          - **orientations** (*numpy.ndarray*) -- Rotation matrices (N×3×3)

      **Hybrid Computation:** Uses GPU for position derivatives, CPU for complex orientation interpolation.

      **SE(3) Interpolation:** Matrix logarithms and exponentials for smooth orientation changes.

   .. rubric:: Dynamics Integration

   .. automethod:: inverse_dynamics_trajectory

      **Parameters:**
        - **thetalist_trajectory** (*numpy.ndarray*) -- Joint angle trajectory (N×n)
        - **dthetalist_trajectory** (*numpy.ndarray*) -- Joint velocity trajectory (N×n)
        - **ddthetalist_trajectory** (*numpy.ndarray*) -- Joint acceleration trajectory (N×n)
        - **gravity_vector** (*array_like*, optional) -- Gravity vector [gx, gy, gz] (default: [0, 0, -9.81])
        - **Ftip** (*array_like*, optional) -- End-effector wrench [fx, fy, fz, mx, my, mz] (default: zeros)

      **Returns:**
        - **torques** (*numpy.ndarray*) -- Required joint torques (N×n)

      **Adaptive Execution:** GPU with 2D parallelization for large trajectories, CPU for small ones.

      **Memory Management:** Automatic memory pooling and pinned memory transfers.

      **Robust Fallback:** Automatic CPU fallback on GPU errors.

   .. automethod:: forward_dynamics_trajectory

      **Parameters:**
        - **thetalist** (*numpy.ndarray*) -- Initial joint angles
        - **dthetalist** (*numpy.ndarray*) -- Initial joint velocities
        - **taumat** (*numpy.ndarray*) -- Applied torque trajectory (N×n)
        - **g** (*numpy.ndarray*) -- Gravity vector [gx, gy, gz]
        - **Ftipmat** (*numpy.ndarray*) -- External force trajectory (N×6)
        - **dt** (*float*) -- Integration time step
        - **intRes** (*int*) -- Integration resolution (sub-steps per dt)

      **Returns:**
        - **simulation** (*dict*) -- Dictionary containing:
          
          - **positions** (*numpy.ndarray*) -- Simulated joint positions (N×n)
          - **velocities** (*numpy.ndarray*) -- Simulated joint velocities (N×n)
          - **accelerations** (*numpy.ndarray*) -- Simulated joint accelerations (N×n)

      **Parallel Integration:** GPU-accelerated numerical integration with automatic joint limit enforcement.

   .. rubric:: Private GPU/CPU Methods

   .. automethod:: _joint_trajectory_gpu

      GPU-accelerated joint trajectory generation with optimized memory management.

   .. automethod:: _joint_trajectory_cpu

      CPU-based joint trajectory generation with Numba optimization.

   .. automethod:: _apply_collision_avoidance_gpu

      Apply GPU-accelerated potential field-based collision avoidance.

   .. automethod:: _apply_collision_avoidance_cpu

      Apply CPU-based potential field collision avoidance.

   .. automethod:: _inverse_dynamics_gpu

      GPU-accelerated inverse dynamics computation with optimized memory management.

   .. automethod:: _inverse_dynamics_cpu

      CPU-based inverse dynamics computation.

   .. automethod:: _forward_dynamics_gpu

      GPU-accelerated forward dynamics computation with optimized memory management.

   .. automethod:: _forward_dynamics_cpu

      CPU-based forward dynamics computation.

   .. automethod:: _cartesian_trajectory_gpu

      GPU-accelerated Cartesian trajectory computation.

   .. automethod:: _cartesian_trajectory_cpu

      CPU-based Cartesian trajectory computation.

   .. automethod:: _batch_joint_trajectory_cpu

      CPU fallback for batch trajectory generation.

   .. automethod:: _should_use_gpu

      Determine if GPU should be used based on problem size and availability.

      **Parameters:**
        - **N** (*int*) -- Number of trajectory points
        - **num_joints** (*int*) -- Number of joints

      **Returns:**
        - **bool** -- True if GPU should be used

   .. automethod:: _get_or_resize_gpu_array

      Return a pooled CUDA array with the requested shape/dtype.

      **Parameters:**
        - **array_name** (*str*) -- Key for array cache
        - **shape** (*tuple*) -- Desired array shape
        - **dtype** (*np.dtype*) -- Desired data type

      **Returns:**
        - **cuda.device_array or None** -- GPU array or None if CUDA unavailable

   .. rubric:: Performance Monitoring

   .. automethod:: get_performance_stats

      Get comprehensive performance statistics for GPU vs CPU usage.

      **Returns:**
        - **dict** -- Performance statistics including:
          
          - **gpu_calls** (*int*) -- Number of GPU function calls
          - **cpu_calls** (*int*) -- Number of CPU function calls
          - **total_gpu_time** (*float*) -- Total GPU execution time
          - **total_cpu_time** (*float*) -- Total CPU execution time
          - **avg_gpu_time** (*float*) -- Average GPU execution time
          - **avg_cpu_time** (*float*) -- Average CPU execution time
          - **gpu_usage_percent** (*float*) -- Percentage of operations using GPU
          - **memory_transfers** (*int*) -- Number of memory transfers
          - **kernel_launches** (*int*) -- Number of CUDA kernel launches

   .. automethod:: reset_performance_stats

      Reset all performance statistics to zero.

   .. automethod:: benchmark_performance

      Benchmark the performance of GPU vs CPU implementations.

      **Parameters:**
        - **test_cases** (*list*, optional) -- List of test cases to benchmark

      **Returns:**
        - **dict** -- Benchmark results for each test case

   .. rubric:: Memory Management

   .. automethod:: cleanup_gpu_memory

      Clean up GPU memory pools and cached arrays.

   .. rubric:: Visualization Methods

   .. automethod:: plot_trajectory
      :staticmethod:

      **Parameters:**
        - **trajectory_data** (*dict*) -- Trajectory data with positions, velocities, accelerations
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "Joint Trajectory")
        - **labels** (*list*, optional) -- Joint labels for legend

   .. automethod:: plot_tcp_trajectory

      **Parameters:**
        - **trajectory** (*list*) -- List of joint angle configurations
        - **dt** (*float*) -- Time step between trajectory points

   .. automethod:: plot_cartesian_trajectory

      **Parameters:**
        - **trajectory_data** (*dict*) -- Cartesian trajectory data
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "Cartesian Trajectory")

   .. automethod:: plot_ee_trajectory

      **Parameters:**
        - **trajectory_data** (*dict*) -- End-effector trajectory data
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "End-Effector Trajectory")

   .. rubric:: Utility Methods

   .. automethod:: calculate_derivatives

      **Parameters:**
        - **positions** (*array_like*) -- Position trajectory
        - **dt** (*float*) -- Time step between positions

      **Returns:**
        - **velocity** (*numpy.ndarray*) -- Finite difference velocities
        - **acceleration** (*numpy.ndarray*) -- Finite difference accelerations
        - **jerk** (*numpy.ndarray*) -- Finite difference jerk

   .. automethod:: plan_trajectory

      **Parameters:**
        - **start_position** (*list*) -- Initial joint configuration
        - **target_position** (*list*) -- Desired joint configuration
        - **obstacle_points** (*list*) -- Environment obstacle points

      **Returns:**
        - **trajectory** (*list*) -- Planned joint trajectory waypoints

------------------------
TrajectoryPlanning Class
------------------------

.. autoclass:: TrajectoryPlanning
   :members:
   :show-inheritance:

   Backward compatibility alias for OptimizedTrajectoryPlanning.

   This ensures existing code continues to work while providing access to all optimizations.

-------------------
Utility Functions
-------------------

.. autofunction:: create_optimized_planner

   Factory function to create an optimized trajectory planner with recommended settings.

   **Parameters:**
     - **serial_manipulator** (*SerialManipulator*) -- SerialManipulator instance
     - **urdf_path** (*str*) -- Path to URDF file
     - **dynamics** (*ManipulatorDynamics*) -- ManipulatorDynamics instance
     - **joint_limits** (*list*) -- Joint limits
     - **torque_limits** (*list*, optional) -- Torque limits
     - **gpu_memory_mb** (*int*, optional) -- GPU memory pool size in MB
     - **enable_profiling** (*bool*) -- Enable CUDA profiling

   **Returns:**
     - **OptimizedTrajectoryPlanning** -- Configured planner instance

.. autofunction:: compare_implementations

   Compare performance between CPU and GPU implementations.

   **Parameters:**
     - **serial_manipulator** (*SerialManipulator*) -- SerialManipulator instance
     - **urdf_path** (*str*) -- Path to URDF file
     - **dynamics** (*ManipulatorDynamics*) -- ManipulatorDynamics instance
     - **joint_limits** (*list*) -- Joint limits
     - **test_params** (*dict*, optional) -- Test parameters

   **Returns:**
     - **dict** -- Comparison results including timing and accuracy metrics

-------------------
CPU Optimization Functions
-------------------

.. autofunction:: _trajectory_cpu_fallback

   Numba-optimized CPU trajectory generation with parallel execution.

   **Parameters:**
     - **thetastart** (*numpy.ndarray*) -- Starting joint angles
     - **thetaend** (*numpy.ndarray*) -- Ending joint angles
     - **Tf** (*float*) -- Final time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method

   **Returns:**
     - **tuple** -- (positions, velocities, accelerations) arrays

   **Optimization:** Uses Numba's parallel prange for CPU parallelization.

.. autofunction:: _traj_cpu_njit

   Thin wrapper for Numba-optimized trajectory computation.

   **Parameters:**
     - **thetastart** (*numpy.ndarray*) -- Starting joint angles
     - **thetaend** (*numpy.ndarray*) -- Ending joint angles
     - **Tf** (*float*) -- Final time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method

   **Returns:**
     - **tuple** -- (positions, velocities, accelerations) arrays

-------------
Usage Examples
-------------

**Optimized Trajectory Planning**::

   from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   
   # Setup with optimization options
   processor = URDFToSerialManipulator("robot.urdf")
   planner = OptimizedTrajectoryPlanning(
       processor.serial_manipulator,
       "robot.urdf",
       processor.dynamics,
       joint_limits=[(-np.pi, np.pi)] * 6,
       use_cuda=None,  # Auto-detect
       cuda_threshold=100,  # Use GPU for problems > 100*6 = 600 elements
       memory_pool_size_mb=256,  # 256MB GPU memory pool
       enable_profiling=True
   )
   
   # Generate trajectory with automatic GPU/CPU selection
   trajectory = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[0.5, -0.3, 0.8, 0.1, -0.2, 0.4],
       Tf=3.0,
       N=2000,  # Large enough to trigger GPU
       method=5
   )
   
   # Check performance stats
   stats = planner.get_performance_stats()
   print(f"GPU usage: {stats['gpu_usage_percent']:.1f}%")
   print(f"Average GPU time: {stats['avg_gpu_time']*1000:.2f}ms")

**Factory Function Usage**::

   from ManipulaPy.path_planning import create_optimized_planner
   
   # Create planner with auto-optimized settings
   planner = create_optimized_planner(
       serial_manipulator=processor.serial_manipulator,
       urdf_path="robot.urdf",
       dynamics=processor.dynamics,
       joint_limits=[(-np.pi, np.pi)] * 6,
       gpu_memory_mb=512,
       enable_profiling=False
   )

**Batch Trajectory Processing**::

   # Generate 10 trajectories simultaneously
   batch_size = 10
   num_joints = 6
   
   starts = np.random.uniform(-np.pi, np.pi, (batch_size, num_joints))
   ends = np.random.uniform(-np.pi, np.pi, (batch_size, num_joints))
   
   # Process all trajectories in parallel on GPU
   batch_trajectories = planner.batch_joint_trajectory(
       thetastart_batch=starts,
       thetaend_batch=ends,
       Tf=2.0,
       N=500,
       method=3
   )
   
   print(f"Batch shape: {batch_trajectories['positions'].shape}")  # (10, 500, 6)

**Performance Comparison**::

   from ManipulaPy.path_planning import compare_implementations
   
   # Compare CPU vs GPU performance
   results = compare_implementations(
       serial_manipulator=processor.serial_manipulator,
       urdf_path="robot.urdf",
       dynamics=processor.dynamics,
       joint_limits=[(-np.pi, np.pi)] * 6,
       test_params={"N": 5000, "Tf": 3.0, "method": 5}
   )
   
   print(f"CPU time: {results['cpu']['time']:.4f}s")
   if results['gpu']['available']:
       print(f"GPU time: {results['gpu']['time']:.4f}s")
       print(f"Speedup: {results['gpu']['speedup']:.2f}x")
       print(f"Max position difference: {results['accuracy']['max_pos_diff']:.2e}")

**Advanced Dynamics Integration**::

   # Generate trajectory
   trajectory = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[1.0, -0.5, 0.8, 0.2, -0.3, 0.6],
       Tf=4.0,
       N=1000,
       method=5
   )
   
   # Compute required torques with GPU acceleration
   torques = planner.inverse_dynamics_trajectory(
       trajectory["positions"],
       trajectory["velocities"],
       trajectory["accelerations"],
       gravity_vector=[0, 0, -9.81],
       Ftip=[0, 0, -10, 0, 0, 0]  # 10N downward force
   )
   
   # Forward dynamics simulation
   simulation = planner.forward_dynamics_trajectory(
       thetalist=[0, 0, 0, 0, 0, 0],
       dthetalist=[0, 0, 0, 0, 0, 0],
       taumat=torques,
       g=[0, 0, -9.81],
       Ftipmat=np.zeros((1000, 6)),
       dt=0.004,
       intRes=5
   )

**Performance Monitoring**::

   # Run multiple operations
   for i in range(5):
       trajectory = planner.joint_trajectory(
           thetastart=np.random.uniform(-1, 1, 6),
           thetaend=np.random.uniform(-1, 1, 6),
           Tf=2.0,
           N=1000,
           method=3
       )
   
   # Analyze performance
   stats = planner.get_performance_stats()
   print(f"Total operations: {stats['gpu_calls'] + stats['cpu_calls']}")
   print(f"GPU efficiency: {stats['gpu_usage_percent']:.1f}%")
   print(f"Memory transfers: {stats['memory_transfers']}")
   
   # Benchmark different problem sizes
   benchmark_results = planner.benchmark_performance([
       {"N": 100, "joints": 6, "name": "Small"},
       {"N": 2000, "joints": 6, "name": "Large"},
       {"N": 1000, "joints": 12, "name": "Many joints"},
   ])
   
   for name, result in benchmark_results.items():
       print(f"{name}: {result['total_time']:.4f}s (GPU: {result['used_gpu']})")

**Memory Management**::

   # Clean up GPU resources explicitly
   planner.cleanup_gpu_memory()
   
   # Reset performance stats
   planner.reset_performance_stats()

**Cartesian Space Planning with Hybrid Computation**::

   # Define start and end poses
   T_start = np.eye(4)
   T_start[:3, 3] = [0.5, 0.2, 0.3]
   
   T_end = np.eye(4)
   T_end[:3, 3] = [0.3, 0.4, 0.5]
   # Add rotation
   from ManipulaPy.utils import rotation_matrix_z
   T_end[:3, :3] = rotation_matrix_z(np.pi/3)
   
   # Generate Cartesian trajectory (GPU for derivatives, CPU for orientations)
   cart_traj = planner.cartesian_trajectory(
       Xstart=T_start,
       Xend=T_end,
       Tf=3.0,
       N=1000,
       method=5
   )
   
   # Visualize
   planner.plot_cartesian_trajectory(cart_traj, Tf=3.0)

**Adaptive Threshold Tuning**::

   # The planner automatically adapts thresholds based on performance
   initial_threshold = planner.cpu_threshold
   
   # Run several operations
   for _ in range(10):
       trajectory = planner.joint_trajectory(
           thetastart=np.random.uniform(-1, 1, 6),
           thetaend=np.random.uniform(-1, 1, 6),
           Tf=2.0,
           N=800,  # Around threshold
           method=3
       )
   
   # Check if threshold adapted
   final_threshold = planner.cpu_threshold
   print(f"Threshold adapted: {initial_threshold} -> {final_threshold}")

-------------
Key Features
-------------

**Adaptive Execution:**
  - Automatic GPU/CPU selection based on problem size
  - Intelligent threshold adaptation based on performance history
  - Graceful fallback on GPU errors or memory constraints

**CUDA Acceleration:**
  - 2D parallelized kernels for optimal GPU utilization
  - Shared memory optimization for time-scaling computations
  - Memory pooling to reduce allocation overhead
  - Pinned memory transfers for maximum PCIe bandwidth

**Memory Management:**
  - Per-instance GPU array caching
  - Global memory pool with automatic cleanup
  - Configurable memory pool sizes
  - Explicit memory management controls

**Performance Monitoring:**
  - Detailed timing statistics for GPU vs CPU usage
  - Automatic performance tracking and analysis
  - Built-in benchmarking capabilities
  - Adaptive threshold tuning based on empirical performance

**Batch Processing:**
  - 3D parallelized batch trajectory generation
  - Efficient memory management for large batches
  - Automatic load balancing across GPU cores

**Robust Operation:**
  - Comprehensive error handling with fallbacks
  - Automatic recovery from GPU memory issues
  - Extensive logging for debugging and optimization

-----------------
Performance Characteristics
-----------------

**GPU Acceleration Thresholds:**
  - **Small problems** (N × joints < 200): CPU (lower overhead)
  - **Medium problems** (200 ≤ N × joints < 5000): Adaptive selection
  - **Large problems** (N × joints ≥ 5000): GPU preferred

**Memory Usage:**
  - **Joint trajectory**: ~12 bytes per (time, joint) element
  - **Cartesian trajectory**: ~36 bytes per time step
  - **Dynamics computation**: ~20 bytes per (time, joint) element

**Typical Speedups (on modern GPU):**
  - **Joint trajectories**: 5-20× for N > 1000
  - **Batch processing**: 10-50× for large batches
  - **Dynamics computation**: 3-15× for long trajectories

-----------------
Configuration Guidelines
-----------------

**GPU Memory Pool Sizing:**
  - Small robots (≤6 DOF): 128-256 MB
  - Medium robots (7-12 DOF): 256-512 MB  
  - Large robots (>12 DOF): 512+ MB

**CUDA Threshold Tuning:**
  - High-end GPUs: Lower threshold (50-100)
  - Mid-range GPUs: Medium threshold (100-200)
  - Integrated GPUs: Higher threshold (200-500)

**Batch Size Recommendations:**
  - Memory-limited: batch_size ≤ 50
  - Compute-limited: batch_size ≤ 200
  - High-memory systems: batch_size ≤ 1000

-----------------
See Also
-----------------

- :doc:`cuda_kernels` -- Low-level CUDA acceleration functions
- :doc:`kinematics` -- Forward and inverse kinematics for trajectory execution
- :doc:`dynamics` -- Dynamics computations for torque calculation
- :doc:`control` -- Controllers for trajectory following
- :doc:`potential_field` -- Collision avoidance and path optimization
- :doc:`../user_guide/Path_Planning` -- Conceptual overview and planning strategies