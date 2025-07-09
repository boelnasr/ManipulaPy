.. _api-cuda-kernels:

==========================
CUDA Kernels API Reference
==========================

.. currentmodule:: ManipulaPy.cuda_kernels

This module provides CUDA-accelerated functions for trajectory planning and dynamics computation with automatic CPU fallback.

--------------------
Availability Check
--------------------

.. autofunction:: check_cuda_availability

   Check if CUDA is available and provide helpful diagnostic information.

   **Returns:**
     - **bool** -- True if CUDA is available, False otherwise

.. autofunction:: check_cupy_availability

   Check if CuPy is available for GPU array operations.

   **Returns:**
     - **bool** -- True if CuPy is available, False otherwise

.. autofunction:: get_gpu_properties

   Retrieve current CUDA device properties for kernel optimization and resource allocation.

   **Returns:**
     - **dict or None** -- GPU device properties including multiprocessor count, memory limits, etc.

-------------------
Core CUDA Kernels
-------------------

Trajectory Kernels
--------------------

.. autofunction:: trajectory_kernel

   CUDA kernel for generating joint trajectory points with time-scaling.

   **Parameters:**
     - **thetastart** (*cuda.device_array*) -- Starting joint angles
     - **thetaend** (*cuda.device_array*) -- Target joint angles  
     - **traj_pos** (*cuda.device_array*) -- Output trajectory positions
     - **traj_vel** (*cuda.device_array*) -- Output trajectory velocities
     - **traj_acc** (*cuda.device_array*) -- Output trajectory accelerations
     - **Tf** (*float*) -- Total trajectory time
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)
     - **stream** (*int*) -- CUDA stream for kernel execution

.. autofunction:: cartesian_trajectory_kernel

   CUDA kernel for generating Cartesian trajectory with time-scaling.

   **Parameters:**
     - **pstart** (*cuda.device_array*) -- Starting point coordinates [x, y, z]
     - **pend** (*cuda.device_array*) -- Ending point coordinates [x, y, z]
     - **traj_pos** (*cuda.device_array*) -- Output trajectory positions
     - **traj_vel** (*cuda.device_array*) -- Output trajectory velocities
     - **traj_acc** (*cuda.device_array*) -- Output trajectory accelerations
     - **Tf** (*float*) -- Total trajectory duration
     - **N** (*int*) -- Number of trajectory points
     - **method** (*int*) -- Time-scaling method (3=cubic, 5=quintic)
     - **stream** (*int*) -- CUDA stream for kernel execution

.. autofunction:: batch_trajectory_kernel

   Optimized CUDA kernel for batch trajectory generation with time-scaling.

   **Parameters:**
     - **thetastart_batch** (*cuda.device_array*) -- Starting joint positions for each batch
     - **thetaend_batch** (*cuda.device_array*) -- Ending joint positions for each batch
     - **traj_pos_batch** (*cuda.device_array*) -- Output trajectory positions
     - **traj_vel_batch** (*cuda.device_array*) -- Output trajectory velocities
     - **traj_acc_batch** (*cuda.device_array*) -- Output trajectory accelerations
     - **Tf** (*float*) -- Total trajectory duration
     - **N** (*int*) -- Number of trajectory timesteps
     - **method** (*int*) -- Time-scaling method (3=cubic, 5=quintic)
     - **batch_size** (*int*) -- Number of trajectory batches
     - **stream** (*int*) -- CUDA stream for kernel execution

Dynamics Kernels
-------------------

.. autofunction:: inverse_dynamics_kernel

   Optimized CUDA kernel for computing inverse dynamics using 2D parallelization.

   **Parameters:**
     - **thetalist_trajectory** (*cuda.device_array*) -- Joint position trajectory
     - **dthetalist_trajectory** (*cuda.device_array*) -- Joint velocity trajectory
     - **ddthetalist_trajectory** (*cuda.device_array*) -- Joint acceleration trajectory
     - **gravity_vector** (*cuda.device_array*) -- Gravity vector
     - **Ftip** (*cuda.device_array*) -- End-effector wrench
     - **Glist** (*cuda.device_array*) -- Mass matrix diagonal elements
     - **Slist** (*cuda.device_array*) -- Velocity quadratic force coefficients
     - **M** (*cuda.device_array*) -- Full mass matrix
     - **torques_trajectory** (*cuda.device_array*) -- Output joint torque trajectory
     - **torque_limits** (*cuda.device_array*) -- Joint torque limits
     - **stream** (*int*) -- CUDA stream for kernel execution

.. autofunction:: forward_dynamics_kernel

   Compute forward dynamics for a robotic system using a CUDA kernel.

   **Parameters:**
     - **thetalist** (*cuda.device_array*) -- Initial joint positions
     - **dthetalist** (*cuda.device_array*) -- Initial joint velocities
     - **taumat** (*cuda.device_array*) -- Applied joint torques trajectory
     - **g** (*cuda.device_array*) -- Gravity vector
     - **Ftipmat** (*cuda.device_array*) -- End-effector wrenches
     - **dt** (*float*) -- Total time step
     - **intRes** (*int*) -- Integration resolution/substeps
     - **Glist** (*cuda.device_array*) -- Mass matrix diagonal elements
     - **Slist** (*cuda.device_array*) -- Velocity quadratic force coefficients
     - **M** (*cuda.device_array*) -- Full mass matrix
     - **thetamat** (*cuda.device_array*) -- Output joint position trajectory
     - **dthetamat** (*cuda.device_array*) -- Output joint velocity trajectory
     - **ddthetamat** (*cuda.device_array*) -- Output joint acceleration trajectory
     - **joint_limits** (*cuda.device_array*) -- Joint position limits
     - **stream** (*int*) -- CUDA stream for kernel execution

Potential Field Kernels
--------------------------

.. autofunction:: fused_potential_gradient_kernel

   CUDA kernel for computing potential and gradient for path planning.

   **Parameters:**
     - **positions** (*cuda.device_array*) -- Input positions to evaluate
     - **goal** (*cuda.device_array*) -- Target goal point coordinates
     - **obstacles** (*cuda.device_array*) -- Array of obstacle point coordinates
     - **potential** (*cuda.device_array*) -- Output array for computed potential values
     - **gradient** (*cuda.device_array*) -- Output array for computed gradient vectors
     - **influence_distance** (*float*) -- Distance threshold for obstacle influence
     - **stream** (*int*) -- CUDA stream for kernel execution

.. autofunction:: attractive_potential_kernel

   Legacy CUDA kernel for attractive potential field computation.

   **Parameters:**
     - **positions** (*cuda.device_array*) -- Query positions (N, 3)
     - **goal** (*cuda.device_array*) -- Goal position [x, y, z]
     - **potential** (*cuda.device_array*) -- Output potential values (N,)

.. autofunction:: repulsive_potential_kernel

   Legacy CUDA kernel for repulsive potential field computation.

   **Parameters:**
     - **positions** (*cuda.device_array*) -- Query positions (N, 3)
     - **obstacles** (*cuda.device_array*) -- Obstacle positions (M, 3)
     - **potential** (*cuda.device_array*) -- Output potential values (N,)
     - **influence_distance** (*float*) -- Maximum influence distance

.. autofunction:: gradient_kernel

   Legacy CUDA kernel for numerical gradient computation.

   **Parameters:**
     - **potential** (*cuda.device_array*) -- Potential field values (N,)
     - **gradient** (*cuda.device_array*) -- Output gradient (N-1,)



-------------------
High-Level Wrappers
-------------------

.. autofunction:: optimized_trajectory_generation

   Generates an optimized trajectory using CUDA acceleration with automatic memory management.

   **Parameters:**
     - **thetastart** (*np.ndarray*) -- Initial joint configuration
     - **thetaend** (*np.ndarray*) -- Final joint configuration
     - **Tf** (*float*) -- Total trajectory duration
     - **N** (*int*) -- Number of trajectory timesteps
     - **method** (*int*) -- Trajectory generation method
     - **use_pinned** (*bool*) -- Use pinned memory for faster GPU transfers

   **Returns:**
     - **tuple** -- (trajectory positions, trajectory velocities, trajectory accelerations)

.. autofunction:: optimized_potential_field

   Compute potential field and gradient for a set of positions using a CUDA-accelerated kernel.

   **Parameters:**
     - **positions** (*np.ndarray*) -- Input positions to compute potential field for
     - **goal** (*np.ndarray*) -- Target goal position
     - **obstacles** (*np.ndarray*) -- Array of obstacle positions
     - **influence_distance** (*float*) -- Distance within which obstacles influence the potential field
     - **use_pinned** (*bool*) -- Use pinned memory for faster GPU transfers

   **Returns:**
     - **tuple** -- (potential values, gradient vectors) for each input position

.. autofunction:: optimized_batch_trajectory_generation

   Efficiently generate batch trajectories using CUDA acceleration.

   **Parameters:**
     - **thetastart_batch** (*np.ndarray*) -- Batch of initial joint configurations
     - **thetaend_batch** (*np.ndarray*) -- Batch of final joint configurations
     - **Tf** (*float*) -- Total trajectory duration
     - **N** (*int*) -- Number of trajectory timesteps
     - **method** (*int*) -- Trajectory generation method identifier
     - **use_pinned** (*bool*) -- Use pinned memory for faster GPU transfers

   **Returns:**
     - **tuple** -- Batch of trajectory positions, velocities, and accelerations

----------------------
CPU Fallback Functions
----------------------

.. autofunction:: trajectory_cpu_fallback

   Compute trajectory positions, velocities, and accelerations on the CPU when CUDA is unavailable.

   **Parameters:**
     - **thetastart** (*np.ndarray*) -- Initial joint configurations
     - **thetaend** (*np.ndarray*) -- Target joint configurations
     - **Tf** (*float*) -- Total trajectory duration
     - **N** (*int*) -- Number of trajectory points to generate
     - **method** (*int*) -- Time scaling method (3=cubic, 5=quintic)

   **Returns:**
     - **tuple** -- (positions, velocities, accelerations) arrays

-------------------
Memory Management
-------------------

.. autofunction:: get_cuda_array

   Get a CUDA array from the memory pool.

   **Parameters:**
     - **shape** (*tuple*) -- Array dimensions
     - **dtype** (*np.dtype*) -- Data type

   **Returns:**
     - **cuda.device_array** -- GPU array from memory pool

.. autofunction:: return_cuda_array

   Return a CUDA array to the memory pool.

   **Parameters:**
     - **array** (*cuda.device_array*) -- GPU array to return

.. autofunction:: _h2d_pinned

   Helper function for pinned memory H2D transfers.

   **Parameters:**
     - **arr** (*np.ndarray*) -- Array to transfer to device

   **Returns:**
     - **cuda.device_array** -- Device array with data transferred

Memory Pool Class
-------------------

.. autoclass:: _GlobalCudaMemoryPool

   A memory pool for managing CUDA device arrays to improve memory allocation efficiency.

   .. automethod:: get_array

      Get a GPU array from the pool or allocate new one.

      **Parameters:**
        - **shape** (*tuple*) -- Array shape
        - **dtype** (*np.dtype*) -- Data type

      **Returns:**
        - **cuda.device_array** -- GPU array

   .. automethod:: return_array

      Return a GPU array to the memory pool for potential future reuse.

      **Parameters:**
        - **array** (*cuda.device_array*) -- The CUDA device array to return

   .. automethod:: clear

      Clear the memory pool.

-------------------
Grid Configuration
-------------------

.. autofunction:: make_1d_grid

   Create a 1D grid configuration for CUDA kernel launch with optimal thread and block sizing.

   **Parameters:**
     - **size** (*int*) -- Total number of elements or work items to process
     - **threads** (*int*) -- Desired number of threads per block

   **Returns:**
     - **tuple** -- ((blocks,), (threads,)) for kernel launch configuration

.. autofunction:: make_2d_grid

   Compute optimal 2D grid configuration for CUDA kernel launch.

   **Parameters:**
     - **N** (*int*) -- First dimension of problem space
     - **num_joints** (*int*) -- Second dimension of problem space
     - **block_size** (*tuple*) -- Initial suggested block dimensions

   **Returns:**
     - **tuple** -- ((blocks_x, blocks_y), (threads_x, threads_y))

-------------------
Performance Tools
-------------------

.. autofunction:: benchmark_kernel_performance

   Benchmark the performance of a specific CUDA kernel by executing it multiple times.

   **Parameters:**
     - **kernel_name** (*str*) -- Name of the kernel to benchmark
     - ***args** -- Arguments to pass to the kernel function
     - **num_runs** (*int*) -- Number of times to run the kernel

   **Returns:**
     - **dict or None** -- Performance metrics including average, std, min/max times

.. autofunction:: profile_start

   Start CUDA profiling.

.. autofunction:: profile_stop

   Stop CUDA profiling.

.. autofunction:: _best_2d_config

   Auto-tune 2D CUDA kernel launch configuration for optimal performance.

   **Parameters:**
     - **N** (*int*) -- Number of time steps or trajectory points
     - **J** (*int*) -- Number of joints or degrees of freedom

   **Returns:**
     - **tuple** -- ((grid_x, grid_y), (block_x, block_y))

-------------------
Constant Memory
-------------------

.. autofunction:: setup_constant_array

   Set up a constant memory array for frequently accessed data.

   **Parameters:**
     - **name** (*str*) -- Unique identifier for the constant memory array
     - **data** (*array-like*) -- Data to be stored in the constant memory array

   **Returns:**
     - **cuda.const.array** -- A CUDA constant memory array

.. autofunction:: get_constant_array

   Retrieve a constant memory array by its name.

   **Parameters:**
     - **name** (*str*) -- The unique identifier of the constant memory array

   **Returns:**
     - **cuda.const.array or None** -- The constant memory array if it exists

-------------------
Module Constants
-------------------

.. autodata:: CUDA_AVAILABLE

   **bool** -- True if CUDA is available, False otherwise

.. autodata:: CUPY_AVAILABLE

   **bool** -- True if CuPy is available, False otherwise

.. autodata:: FAST_MATH

   **bool** -- Whether fast math optimizations are enabled

.. autodata:: float_t

   **type** -- Float precision type (float32 or float16)

----------------------
Environment Variables
----------------------

**MANIPULAPY_FASTMATH**
   Set to "1" to enable fast math optimizations (~2x speedup with relaxed IEEE 754 compliance)

**MANIPULAPY_USE_FP16**
   Set to "1" to use 16-bit floating point precision for memory-bound kernels