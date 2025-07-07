.. _api-cuda-kernels:

==========================
CUDA Kernels API Reference
==========================

This page documents **ManipulaPy.cuda_kernels**, the module providing CUDA-accelerated functions for trajectory planning and dynamics computation.

.. tip::
   All CUDA functionality is optional and gracefully degrades to CPU implementations when GPU support is unavailable.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

----------------
Module Overview
----------------

.. currentmodule:: ManipulaPy.cuda_kernels

The CUDA kernels module provides high-performance GPU-accelerated functions for:

- **Trajectory Generation**: Cubic and quintic time scaling with parallel computation
- **Dynamics Computation**: Inverse and forward dynamics for trajectory batches
- **Potential Fields**: Attractive and repulsive potential field calculations
- **Automatic Fallback**: CPU implementations when CUDA is unavailable

.. note::
   **GPU Requirements**: CUDA kernels require:
   
   - NVIDIA GPU with compute capability 3.5+
   - CUDA toolkit 11.0+ or 12.0+
   - Installation: ``pip install ManipulaPy[gpu-cuda11]`` or ``pip install ManipulaPy[gpu-cuda12]``

-----------------
Availability Check
-----------------

.. autofunction:: check_cuda_availability

   Check if CUDA is available and provide helpful error messages.

   **Returns:**
     - **bool** -- True if CUDA is available, False otherwise

   **Example:**
     >>> from ManipulaPy.cuda_kernels import check_cuda_availability
     >>> if check_cuda_availability():
     ...     print("CUDA acceleration available")
     ... else:
     ...     print("Falling back to CPU implementation")

.. autofunction:: check_cupy_availability

   Check if CuPy is available for GPU array operations.

   **Returns:**
     - **bool** -- True if CuPy is available, False otherwise

-----------------
CUDA Kernels
-----------------

Trajectory Kernels
------------------

.. autofunction:: trajectory_kernel

   CUDA kernel to compute joint trajectories with cubic or quintic time scaling.

   **Parameters:**
     - **thetastart** (*cuda.device_array*) -- Starting joint angles (n,)
     - **thetaend** (*cuda.device_array*) -- Ending joint angles (n,)
     - **traj_pos** (*cuda.device_array*) -- Output positions (N, n)
     - **traj_vel** (*cuda.device_array*) -- Output velocities (N, n)
     - **traj_acc** (*cuda.device_array*) -- Output accelerations (N, n)
     - **Tf** (*float*) -- Total trajectory time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

   **Grid Configuration:**
     - **Threads per block**: 256
     - **Blocks per grid**: ``(N + 255) // 256``

   **Mathematical Formulation:**

   For cubic time scaling (method=3):

   .. math::

      s(t) &= 3\left(\frac{t}{T_f}\right)^2 - 2\left(\frac{t}{T_f}\right)^3 \\
      \dot{s}(t) &= \frac{6t(T_f - t)}{T_f^3} \\
      \ddot{s}(t) &= \frac{6(T_f - 2t)}{T_f^2}

   For quintic time scaling (method=5):

   .. math::

      s(t) &= 10\left(\frac{t}{T_f}\right)^3 - 15\left(\frac{t}{T_f}\right)^4 + 6\left(\frac{t}{T_f}\right)^5

.. autofunction:: cartesian_trajectory_kernel

   CUDA kernel for Cartesian space trajectory generation.

   **Parameters:**
     - **pstart** (*cuda.device_array*) -- Starting position [x, y, z]
     - **pend** (*cuda.device_array*) -- Ending position [x, y, z]
     - **traj_pos** (*cuda.device_array*) -- Output positions (N, 3)
     - **traj_vel** (*cuda.device_array*) -- Output velocities (N, 3)
     - **traj_acc** (*cuda.device_array*) -- Output accelerations (N, 3)
     - **Tf** (*float*) -- Total trajectory time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

Dynamics Kernels
-----------------

.. autofunction:: inverse_dynamics_kernel

   CUDA kernel for parallel inverse dynamics computation.

   **Parameters:**
     - **thetalist_trajectory** (*cuda.device_array*) -- Joint angles (N, n)
     - **dthetalist_trajectory** (*cuda.device_array*) -- Joint velocities (N, n)
     - **ddthetalist_trajectory** (*cuda.device_array*) -- Joint accelerations (N, n)
     - **gravity_vector** (*cuda.device_array*) -- Gravity vector [0, 0, -9.81]
     - **Ftip** (*cuda.device_array*) -- External forces at end-effector
     - **Glist** (*cuda.device_array*) -- Spatial inertia matrices (n, 6, 6)
     - **Slist** (*cuda.device_array*) -- Screw axes (6, n)
     - **M** (*cuda.device_array*) -- Home configuration matrix (4, 4)
     - **torques_trajectory** (*cuda.device_array*) -- Output torques (N, n)
     - **torque_limits** (*cuda.device_array*) -- Torque limits (n, 2)

   **Mathematical Formulation:**

   .. math::

      \boldsymbol{\tau} = M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + G(\mathbf{q}) + J^T(\mathbf{q})\mathbf{F}_{tip}

.. autofunction:: forward_dynamics_kernel

   CUDA kernel for parallel forward dynamics computation.

   **Parameters:**
     - **thetalist** (*cuda.device_array*) -- Initial joint angles
     - **dthetalist** (*cuda.device_array*) -- Initial joint velocities
     - **taumat** (*cuda.device_array*) -- Applied torques (N, n)
     - **g** (*cuda.device_array*) -- Gravity vector
     - **Ftipmat** (*cuda.device_array*) -- External forces (N, 6)
     - **dt** (*float*) -- Integration time step
     - **intRes** (*int*) -- Integration resolution
     - **Glist** (*cuda.device_array*) -- Spatial inertia matrices
     - **Slist** (*cuda.device_array*) -- Screw axes
     - **M** (*cuda.device_array*) -- Home configuration
     - **thetamat** (*cuda.device_array*) -- Output positions (N, n)
     - **dthetamat** (*cuda.device_array*) -- Output velocities (N, n)
     - **ddthetamat** (*cuda.device_array*) -- Output accelerations (N, n)
     - **joint_limits** (*cuda.device_array*) -- Joint limits (n, 2)

Potential Field Kernels
------------------------

.. autofunction:: attractive_potential_kernel

   CUDA kernel for attractive potential field computation.

   **Parameters:**
     - **positions** (*cuda.device_array*) -- Query positions (N, 3)
     - **goal** (*cuda.device_array*) -- Goal position [x, y, z]
     - **potential** (*cuda.device_array*) -- Output potential values (N,)

   **Mathematical Formulation:**

   .. math::

      U_{att}(\mathbf{q}) = \frac{1}{2}\|\mathbf{q} - \mathbf{q}_{goal}\|^2

.. autofunction:: repulsive_potential_kernel

   CUDA kernel for repulsive potential field computation.

   **Parameters:**
     - **positions** (*cuda.device_array*) -- Query positions (N, 3)
     - **obstacles** (*cuda.device_array*) -- Obstacle positions (M, 3)
     - **potential** (*cuda.device_array*) -- Output potential values (N,)
     - **influence_distance** (*float*) -- Maximum influence distance

   **Mathematical Formulation:**

   .. math::

      U_{rep}(\mathbf{q}) = \begin{cases}
      \frac{1}{2}\left(\frac{1}{\rho(\mathbf{q})} - \frac{1}{\rho_0}\right)^2 & \text{if } \rho(\mathbf{q}) \leq \rho_0 \\
      0 & \text{if } \rho(\mathbf{q}) > \rho_0
      \end{cases}

   where :math:`\rho(\mathbf{q})` is the distance to nearest obstacle and :math:`\rho_0` is the influence distance.

.. autofunction:: gradient_kernel

   CUDA kernel for numerical gradient computation.

   **Parameters:**
     - **potential** (*cuda.device_array*) -- Potential field values (N,)
     - **gradient** (*cuda.device_array*) -- Output gradient (N-1,)

-------------------
CPU Fallback Functions
-------------------

.. autofunction:: trajectory_cpu_fallback

   CPU implementation for trajectory generation when CUDA is unavailable.

   **Parameters:**
     - **thetastart** (*np.ndarray*) -- Starting joint angles
     - **thetaend** (*np.ndarray*) -- Ending joint angles
     - **Tf** (*float*) -- Final time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

   **Returns:**
     - **tuple** -- (positions, velocities, accelerations) arrays

   **Example:**
     >>> import numpy as np
     >>> from ManipulaPy.cuda_kernels import trajectory_cpu_fallback
     >>> start = np.array([0.0, 0.0, 0.0])
     >>> end = np.array([1.0, 0.5, -0.3])
     >>> pos, vel, acc = trajectory_cpu_fallback(start, end, Tf=2.0, N=100, method=3)

--------------
Usage Examples
--------------

Basic CUDA Trajectory Generation
---------------------------------

::

   import numpy as np
   from numba import cuda
   from ManipulaPy.cuda_kernels import trajectory_kernel, check_cuda_availability

   if check_cuda_availability():
       # Setup trajectory parameters
       thetastart = np.array([0.0, 0.0, 0.0], dtype=np.float32)
       thetaend = np.array([1.0, 0.5, -0.3], dtype=np.float32)
       N = 1000
       Tf = 2.0
       method = 3  # Cubic scaling

       # Allocate GPU memory
       d_start = cuda.to_device(thetastart)
       d_end = cuda.to_device(thetaend)
       d_pos = cuda.device_array((N, 3), dtype=np.float32)
       d_vel = cuda.device_array((N, 3), dtype=np.float32)
       d_acc = cuda.device_array((N, 3), dtype=np.float32)

       # Configure kernel launch
       threads_per_block = 256
       blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

       # Launch kernel
       trajectory_kernel[blocks_per_grid, threads_per_block](
           d_start, d_end, d_pos, d_vel, d_acc, Tf, N, method
       )

       # Copy results back to host
       positions = d_pos.copy_to_host()
       velocities = d_vel.copy_to_host()
       accelerations = d_acc.copy_to_host()

Batch Dynamics Computation
---------------------------

::

   from ManipulaPy.cuda_kernels import inverse_dynamics_kernel

   # Batch inverse dynamics for 1000 trajectory points
   N_points = 1000
   n_joints = 6

   # Setup trajectory data (on GPU)
   d_theta_traj = cuda.device_array((N_points, n_joints), dtype=np.float32)
   d_dtheta_traj = cuda.device_array((N_points, n_joints), dtype=np.float32)
   d_ddtheta_traj = cuda.device_array((N_points, n_joints), dtype=np.float32)
   d_torques = cuda.device_array((N_points, n_joints), dtype=np.float32)

   # Robot parameters (simplified)
   gravity = np.array([0, 0, -9.81], dtype=np.float32)
   d_gravity = cuda.to_device(gravity)
   
   # Configure and launch kernel
   threads_per_block = 256
   blocks_per_grid = (N_points + threads_per_block - 1) // threads_per_block

   inverse_dynamics_kernel[blocks_per_grid, threads_per_block](
       d_theta_traj, d_dtheta_traj, d_ddtheta_traj,
       d_gravity, d_Ftip, d_Glist, d_Slist, d_M,
       d_torques, d_torque_limits
   )

   # Results available in d_torques

Potential Field Path Planning
-----------------------------

::

   from ManipulaPy.cuda_kernels import attractive_potential_kernel, repulsive_potential_kernel

   # Setup workspace grid
   N_points = 10000
   positions = np.random.uniform(-5, 5, (N_points, 3)).astype(np.float32)
   goal = np.array([4.0, 4.0, 1.0], dtype=np.float32)
   obstacles = np.array([
       [1.0, 1.0, 0.5],
       [2.0, 3.0, 1.0],
       [-1.0, 2.0, 0.8]
   ], dtype=np.float32)

   # GPU memory allocation
   d_positions = cuda.to_device(positions)
   d_goal = cuda.to_device(goal)
   d_obstacles = cuda.to_device(obstacles)
   d_potential = cuda.device_array(N_points, dtype=np.float32)

   threads_per_block = 256
   blocks_per_grid = (N_points + threads_per_block - 1) // threads_per_block

   # Compute attractive potential
   attractive_potential_kernel[blocks_per_grid, threads_per_block](
       d_positions, d_goal, d_potential
   )

   # Add repulsive potential
   influence_distance = 1.0
   repulsive_potential_kernel[blocks_per_grid, threads_per_block](
       d_positions, d_obstacles, d_potential, influence_distance
   )

   # Copy results
   total_potential = d_potential.copy_to_host()

Error Handling and Fallbacks
-----------------------------

::

   from ManipulaPy.cuda_kernels import check_cuda_availability, trajectory_cpu_fallback

   def generate_trajectory(start, end, Tf, N, method=3):
       """Generate trajectory with automatic GPU/CPU fallback"""
       
       if check_cuda_availability():
           try:
               # GPU implementation
               from numba import cuda
               from ManipulaPy.cuda_kernels import trajectory_kernel
               
               # Convert to appropriate types
               start = np.asarray(start, dtype=np.float32)
               end = np.asarray(end, dtype=np.float32)
               
               # GPU computation
               d_start = cuda.to_device(start)
               d_end = cuda.to_device(end)
               d_pos = cuda.device_array((N, len(start)), dtype=np.float32)
               d_vel = cuda.device_array((N, len(start)), dtype=np.float32)
               d_acc = cuda.device_array((N, len(start)), dtype=np.float32)
               
               threads_per_block = 256
               blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
               
               trajectory_kernel[blocks_per_grid, threads_per_block](
                   d_start, d_end, d_pos, d_vel, d_acc, Tf, N, method
               )
               
               return d_pos.copy_to_host(), d_vel.copy_to_host(), d_acc.copy_to_host()
               
           except Exception as e:
               print(f"GPU computation failed: {e}")
               print("Falling back to CPU implementation")
       
       # CPU fallback
       return trajectory_cpu_fallback(start, end, Tf, N, method)

Memory Management
-----------------

::

   import gc
   from numba import cuda

   def cleanup_gpu_memory():
       """Clean up GPU memory after kernel execution"""
       # Force garbage collection
       gc.collect()
       
       # Synchronize CUDA context
       cuda.synchronize()
       
       # Optional: Reset CUDA context (use with caution)
       # cuda.close()

   # Example usage in a loop
   for i in range(num_iterations):
       # GPU computation
       result = compute_on_gpu(data)
       
       # Process results
       process_results(result)
       
       # Clean up every few iterations
       if i % 10 == 0:
           cleanup_gpu_memory()

-------------
Performance Tips
-------------

**Memory Optimization:**
  - Reuse device arrays when possible
  - Use appropriate data types (float32 for most kernels)
  - Minimize host-device transfers

**Kernel Configuration:**
  - Use 256 or 512 threads per block for optimal occupancy
  - Ensure sufficient work per thread to hide latency
  - Profile with ``nvprof`` or ``nsight`` for optimization

**Batch Processing:**
  - Process multiple trajectory points simultaneously
  - Use shared memory for frequently accessed data
  - Consider memory coalescing patterns

**Error Handling:**
  - Always check CUDA availability before kernel calls
  - Implement CPU fallbacks for robustness
  - Monitor GPU memory usage to avoid out-of-memory errors

-------------
Known Limitations
-------------

.. warning::
   **Current Limitations:**
   
   - Dynamics kernels use simplified models for demonstration
   - No support for parallel manipulators
   - Limited error checking within kernels
   - Memory management is manual

**Planned Improvements:**
  - Complete dynamics model implementation
  - Automatic memory management
  - Support for variable DOF robots
  - Enhanced error reporting

---------
See Also
---------

- :doc:`path_planning` -- High-level trajectory planning using CUDA kernels
- :doc:`dynamics` -- Robot dynamics models for kernel computations
- :doc:`potential_field` -- Potential field methods using CUDA acceleration
- :doc:`control` -- GPU-accelerated control algorithms