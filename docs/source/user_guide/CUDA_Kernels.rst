CUDA Kernels User Guide
==========================

This guide covers the **CUDA acceleration** features in ManipulaPy, which provide GPU-accelerated functions for high-performance robotic computations including trajectory planning, dynamics computation, and potential field calculations.

.. contents:: **Quick Navigation**
   :local:
   :depth: 2

What is GPU Acceleration?
----------------------------

**GPU acceleration** uses NVIDIA graphics cards to perform parallel computations much faster than traditional CPU processing. ManipulaPy leverages CUDA through:

- **Numba CUDA kernels**: Low-level parallel computation
- **CuPy arrays**: NumPy-compatible GPU arrays  
- **Automatic fallback**: Graceful degradation to CPU when GPU unavailable
- **Memory optimization**: Efficient GPU memory management

Key benefits include:

- **10-100x Performance Improvement**: Significant speedup for large-scale computations
- **Real-time Capable**: Suitable for real-time control applications
- **Seamless Integration**: Works directly with existing ManipulaPy classes

Mathematical Foundation
--------------------------

CUDA Kernel Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Trajectory Generation Kernel:**

The GPU kernel computes smooth trajectories using parallel processing:

.. math::
   s(t) = \begin{cases}
     3(t/T_f)^2 - 2(t/T_f)^3 & \text{cubic} \\
     10(t/T_f)^3 - 15(t/T_f)^4 + 6(t/T_f)^5 & \text{quintic}
   \end{cases}

where each thread computes one trajectory point in parallel:

.. code-block:: python

   @cuda.jit
   def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):
       idx = cuda.grid(1)
       if idx < N:
           t = idx * (Tf / (N - 1))
           # Compute time scaling function...

**Dynamics Computation:**

Parallel inverse dynamics using the Newton-Euler formulation:

.. math::
   \boldsymbol\tau = M(\boldsymbol\theta)\ddot{\boldsymbol\theta} + C(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + G(\boldsymbol\theta) + J^T(\boldsymbol\theta)F_{ext}

Each GPU thread processes one trajectory point simultaneously.

Installation and Setup
-------------------------

System Requirements
~~~~~~~~~~~~~~~~~~~~~~

**Hardware:**
- NVIDIA GPU with Compute Capability 3.0 or higher
- 4GB+ GPU memory recommended
- PCIe 3.0 x16 slot

**Software:**
- NVIDIA GPU drivers (latest recommended)
- CUDA Toolkit 11.0+ or 12.0+
- Python 3.8+

Installation Options
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Full GPU Support (Recommended)**

.. code-block:: bash

   # For CUDA 11.x systems
   pip install cupy-cuda11x numba ManipulaPy

   # For CUDA 12.x systems  
   pip install cupy-cuda12x numba ManipulaPy

**Option 2: CPU-Only Installation**

.. code-block:: bash

   # Basic installation (no GPU acceleration)
   pip install ManipulaPy

Verification
~~~~~~~~~~~~~~

.. code-block:: python

   from numba import cuda
   import numpy as np

   # Check CUDA availability
   try:
       cuda.detect()
       print("✅ CUDA acceleration available")
       CUDA_AVAILABLE = True
   except:
       print("❌ CUDA not available - using CPU fallback")
       CUDA_AVAILABLE = False

   # Check CuPy availability
   try:
       import cupy as cp
       cp.cuda.Device(0).compute_capability
       print("✅ CuPy GPU arrays available")
       CUPY_AVAILABLE = True
   except:
       print("❌ CuPy not available")
       CUPY_AVAILABLE = False

Available CUDA Kernels
-------------------------

Trajectory Generation Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**trajectory_kernel**

Generates smooth joint trajectories with cubic or quintic time scaling.

.. code-block:: python

   @cuda.jit
   def trajectory_kernel(thetastart, thetaend, traj_pos, traj_vel, traj_acc, Tf, N, method):

**Parameters:**
- **thetastart**: Starting joint angles
- **thetaend**: Ending joint angles  
- **traj_pos**: Output trajectory positions
- **traj_vel**: Output trajectory velocities
- **traj_acc**: Output trajectory accelerations
- **Tf**: Total trajectory time
- **N**: Number of trajectory points
- **method**: Time scaling method (3=cubic, 5=quintic)

**cartesian_trajectory_kernel**

Generates Cartesian space trajectories for end-effector motion.

.. code-block:: python

   @cuda.jit
   def cartesian_trajectory_kernel(pstart, pend, traj_pos, traj_vel, traj_acc, Tf, N, method):

Dynamics Computation Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**inverse_dynamics_kernel**

Computes required joint torques for given motion trajectories.

.. code-block:: python

   @cuda.jit
   def inverse_dynamics_kernel(
       thetalist_trajectory, dthetalist_trajectory, ddthetalist_trajectory, 
       gravity_vector, Ftip, Glist, Slist, M, torques_trajectory, torque_limits):

**Features:**
- Parallel computation across trajectory points
- Includes mass matrix, Coriolis, and gravity effects
- Automatic torque limit enforcement

**forward_dynamics_kernel**

Simulates robot motion given applied torques.

.. code-block:: python

   @cuda.jit
   def forward_dynamics_kernel(
       thetalist, dthetalist, taumat, g, Ftipmat, dt, intRes,
       Glist, Slist, M, thetamat, dthetamat, ddthetamat, joint_limits):

Potential Field Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

**attractive_potential_kernel**

Computes attractive potential fields for path planning.

.. code-block:: python

   @cuda.jit
   def attractive_potential_kernel(positions, goal, potential):

**repulsive_potential_kernel**

Computes repulsive potential fields around obstacles.

.. code-block:: python

   @cuda.jit
   def repulsive_potential_kernel(positions, obstacles, potential, influence_distance):

**gradient_kernel**

Computes gradients of potential fields for navigation.

.. code-block:: python

   @cuda.jit
   def gradient_kernel(potential, gradient):

Using GPU Acceleration
-------------------------

Basic Trajectory Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from ManipulaPy.path_planning import TrajectoryPlanning
   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   def generate_gpu_trajectory_example():
       """Example using ManipulaPy's TrajectoryPlanning with GPU acceleration."""
       
       # Load robot model
       urdf_path = "path/to/robot.urdf"
       urdf_processor = URDFToSerialManipulator(urdf_path)
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       
       # Create trajectory planner
       joint_limits = [(-np.pi, np.pi)] * 6  # Example 6-DOF robot
       trajectory_planner = TrajectoryPlanning(
           robot, urdf_path, dynamics, joint_limits
       )
       
       # Generate trajectory using GPU acceleration (automatic)
       thetastart = np.zeros(6, dtype=np.float32)
       thetaend = np.array([1.5, 0.8, -0.5, 0.3, 1.2, -0.7], dtype=np.float32)
       Tf = 2.0  # 2 seconds
       N = 1000  # 1000 points
       method = 5  # Quintic time scaling
       
       trajectory = trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf, N, method
       )
       
       print(f"Generated trajectory with {trajectory['positions'].shape[0]} points")
       print(f"Trajectory shape: {trajectory['positions'].shape}")
       
       return trajectory

   # Example usage
   trajectory = generate_gpu_trajectory_example()

TrajectoryPlanning with GPU Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning
   from ManipulaPy.urdf_processor import URDFToSerialManipulator

   # Initialize robot and dynamics
   urdf_processor = URDFToSerialManipulator("robot.urdf")
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics

   # Set up joint and torque limits
   num_joints = len(dynamics.Glist)
   joint_limits = [(-np.pi, np.pi)] * num_joints
   torque_limits = [(-50, 50)] * num_joints

   # Create trajectory planner (automatically uses GPU when available)
   trajectory_planner = TrajectoryPlanning(
       robot, "robot.urdf", dynamics, joint_limits, torque_limits
   )

   # Generate trajectory - GPU acceleration happens automatically
   thetastart = np.zeros(num_joints)
   thetaend = np.ones(num_joints) * 0.5
   trajectory = trajectory_planner.joint_trajectory(
       thetastart, thetaend, Tf=2.0, N=1000, method=3
   )

   # Compute inverse dynamics - also GPU accelerated
   torques = trajectory_planner.inverse_dynamics_trajectory(
       trajectory["positions"],
       trajectory["velocities"], 
       trajectory["accelerations"]
   )

   print(f"Computed torques shape: {torques.shape}")

Batch Dynamics Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_trajectory_dynamics_example():
       """Compute dynamics for entire trajectory using GPU acceleration."""
       
       # Initialize robot and trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       
       num_joints = len(dynamics.Glist)
       joint_limits = [(-np.pi, np.pi)] * num_joints
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # Generate trajectory first
       thetastart = np.zeros(num_joints)
       thetaend = np.ones(num_joints) * 0.5
       
       trajectory = trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=2.0, N=2000, method=5
       )
       
       # Compute inverse dynamics (GPU accelerated automatically)
       torques = trajectory_planner.inverse_dynamics_trajectory(
           trajectory["positions"],
           trajectory["velocities"],
           trajectory["accelerations"],
           gravity_vector=[0, 0, -9.81],
           Ftip=[0, 0, 0, 0, 0, 0]
       )
       
       print(f"Computed torques shape: {torques.shape}")
       return torques

   # Example usage
   trajectory_torques = compute_trajectory_dynamics_example()

Potential Field Path Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.potential_field import PotentialField

   def potential_field_example():
       """Example using potential fields with GPU acceleration."""
       
       # Create potential field
       potential_field = PotentialField(
           attractive_gain=1.0,
           repulsive_gain=100.0,
           influence_distance=0.5
       )
       
       # Define workspace points
       workspace_points = np.random.rand(10000, 3).astype(np.float32)
       goal_position = np.array([5.0, 5.0, 2.0])
       obstacles = np.array([[2.0, 2.0, 1.0], [3.0, 4.0, 1.5]])
       
       # Compute potential field (uses GPU kernels internally when available)
       total_potential = np.zeros(len(workspace_points))
       
       for i, point in enumerate(workspace_points):
           attractive = potential_field.compute_attractive_potential(point, goal_position)
           repulsive = potential_field.compute_repulsive_potential(point, obstacles)
           total_potential[i] = attractive + repulsive
       
       return total_potential

   # Example usage
   potential_values = potential_field_example()

Performance Optimization
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimal_memory_usage():
       """Example of optimal GPU memory usage with ManipulaPy."""
       
       # Pre-allocate trajectories for efficiency
       max_points = 5000
       num_joints = 6
       
       # Use float32 for GPU efficiency
       thetastart = np.zeros(num_joints, dtype=np.float32)
       thetaend = np.ones(num_joints, dtype=np.float32)
       
       # Initialize trajectory planner once
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * num_joints
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # Process multiple trajectories efficiently
       trajectories = []
       for i in range(10):
           # Vary end positions
           end_pos = thetaend * (i + 1) * 0.1
           
           trajectory = trajectory_planner.joint_trajectory(
               thetastart, end_pos, Tf=2.0, N=max_points, method=3
           )
           trajectories.append(trajectory)
       
       return trajectories

   def cleanup_gpu_memory():
       """Clean up GPU memory after computations."""
       try:
           import cupy as cp
           mempool = cp.get_default_memory_pool()
           mempool.free_all_blocks()
           print("GPU memory cleaned up")
       except ImportError:
           pass

   trajectories = optimal_memory_usage()
   cleanup_gpu_memory()

GPU Configuration Check
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def check_gpu_configuration():
       """Check optimal GPU configuration for kernels."""
       
       try:
           from numba import cuda
           
           if cuda.is_available():
               device = cuda.get_current_device()
               print(f"GPU: {device.name}")
               print(f"Compute Capability: {device.compute_capability}")
               print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
               print(f"Max block dimensions: {device.MAX_BLOCK_DIM_X}")
               print(f"Multiprocessor count: {device.MULTIPROCESSOR_COUNT}")
           else:
               print("No CUDA device available")
       except ImportError:
           print("Numba CUDA not available")

   check_gpu_configuration()

Memory Pool Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def setup_gpu_memory_pool(max_memory_gb=4):
       """Configure GPU memory pool for optimal performance."""
       
       try:
           import cupy as cp
           mempool = cp.get_default_memory_pool()
           
           # Set memory limit to prevent OOM
           max_bytes = int(max_memory_gb * 1024**3)
           mempool.set_limit(size=max_bytes)
           
           print(f"GPU memory pool configured with {max_memory_gb} GB limit")
           
           return mempool
           
       except ImportError:
           print("CuPy not available - cannot configure memory pool")
           return None
       except Exception as e:
           print(f"Error configuring memory pool: {e}")
           return None

   def cleanup_gpu_memory():
       """Clean up GPU memory."""
       
       try:
           import cupy as cp
           mempool = cp.get_default_memory_pool()
           mempool.free_all_blocks()
           print("GPU memory cleaned up")
       except ImportError:
           pass
       except Exception as e:
           print(f"Error cleaning up GPU memory: {e}")

   # Example usage
   mempool = setup_gpu_memory_pool(max_memory_gb=6)
   
   # ... perform GPU computations ...
   
   cleanup_gpu_memory()

CPU Fallback
--------------

Automatic Fallback System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManipulaPy modules automatically fall back to CPU computation when GPU acceleration is not available:

.. code-block:: python

   def test_cpu_fallback():
       """Test CPU fallback functionality."""
       
       # Initialize components
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       # TrajectoryPlanning automatically handles CPU fallback
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # This will use GPU if available, CPU otherwise
       thetastart = np.zeros(6)
       thetaend = np.ones(6)
       
       trajectory = trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=1.0, N=1000, method=3
       )
       
       # Check if result is valid regardless of GPU/CPU
       assert trajectory["positions"].shape == (1000, 6)
       assert trajectory["velocities"].shape == (1000, 6)
       assert trajectory["accelerations"].shape == (1000, 6)
       
       print("Trajectory generation successful (GPU or CPU)")
       
       return trajectory

   test_trajectory = test_cpu_fallback()

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   def compare_cpu_gpu_performance():
       """Compare CPU vs GPU performance across different problem sizes."""
       
       try:
           from numba import cuda
           cuda_available = cuda.is_available()
       except:
           cuda_available = False
       
       # Initialize trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       problem_sizes = [100, 500, 1000, 2000]
       results = []
       
       for N in problem_sizes:
           thetastart = np.zeros(6)
           thetaend = np.ones(6)
           
           # Time the trajectory generation
           start_time = time.perf_counter()
           trajectory = trajectory_planner.joint_trajectory(
               thetastart, thetaend, Tf=2.0, N=N, method=3
           )
           elapsed_time = time.perf_counter() - start_time
           
           results.append({
               'problem_size': N,
               'time_ms': elapsed_time * 1000,
               'cuda_available': cuda_available
           })
           
           status = "GPU" if cuda_available else "CPU"
           print(f"N={N:4d}: {elapsed_time*1000:6.2f}ms ({status})")
       
       return results

   # Run performance comparison
   import time
   perf_results = compare_cpu_gpu_performance()

Troubleshooting
------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CUDA Installation Issues**

Problem: ``ImportError: No module named 'numba.cuda'``

Solutions:

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi

   # Install appropriate CuPy version
   pip install cupy-cuda11x  # For CUDA 11.x
   pip install cupy-cuda12x  # For CUDA 12.x

   # Install Numba with CUDA support
   pip install numba

   # Verify installation
   python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"

**GPU Memory Errors**

Problem: ``CudaAPIError: CUDA_ERROR_OUT_OF_MEMORY``

Solutions:

.. code-block:: python

   # Reduce trajectory size
   N = 1000  # Instead of 10000

   # Process in smaller chunks
   def process_large_trajectory(thetastart, thetaend, total_points=10000):
       """Process large trajectory in chunks."""
       chunk_size = 1000
       all_positions = []
       all_velocities = []
       all_accelerations = []
       
       # Initialize trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       for i in range(0, total_points, chunk_size):
           chunk_N = min(chunk_size, total_points - i)
           chunk_Tf = 2.0 * chunk_N / total_points
           
           chunk_traj = trajectory_planner.joint_trajectory(
               thetastart, thetaend, chunk_Tf, chunk_N, method=3
           )
           
           all_positions.append(chunk_traj["positions"])
           all_velocities.append(chunk_traj["velocities"])
           all_accelerations.append(chunk_traj["accelerations"])
       
       return {
           "positions": np.vstack(all_positions),
           "velocities": np.vstack(all_velocities),
           "accelerations": np.vstack(all_accelerations)
       }

**Performance Issues**

Problem: GPU slower than expected

Debugging:

.. code-block:: python

   import time

   def benchmark_trajectory_generation():
       """Benchmark trajectory generation performance."""
       
       # Initialize trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       problem_sizes = [100, 500, 1000, 2000, 5000]
       
       for N in problem_sizes:
           thetastart = np.zeros(6)
           thetaend = np.ones(6)
           
           # Time the trajectory generation
           start_time = time.perf_counter()
           
           trajectory = trajectory_planner.joint_trajectory(
               thetastart, thetaend, Tf=2.0, N=N, method=3
           )
           
           end_time = time.perf_counter()
           elapsed_ms = (end_time - start_time) * 1000
           
           print(f"N={N:4d}: {elapsed_ms:6.2f} ms")

   benchmark_trajectory_generation()

Best Practices
----------------

Data Types
~~~~~~~~~~~~

.. code-block:: python

   # ✅ Good: Use float32 for GPU efficiency
   thetastart = np.zeros(6, dtype=np.float32)
   thetaend = np.ones(6, dtype=np.float32)

   # ❌ Bad: Using float64 (unnecessary precision, slower)
   thetastart = np.zeros(6, dtype=np.float64)

Problem Sizing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✅ Good: Use appropriate problem sizes for GPU acceleration
   def optimal_problem_sizing():
       """Choose problem sizes that benefit from GPU acceleration."""
       
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # Check if CUDA is available
       try:
           from numba import cuda
           cuda_available = cuda.is_available()
       except:
           cuda_available = False
       
       # Adjust problem size based on available acceleration
       if cuda_available:
           N = 5000  # Large problem size for GPU
           print("Using GPU acceleration with large problem size")
       else:
           N = 1000  # Smaller problem size for CPU
           print("Using CPU with moderate problem size")
       
       trajectory = trajectory_planner.joint_trajectory(
           np.zeros(6), np.ones(6), Tf=2.0, N=N, method=3
       )
       
       return trajectory

Error Handling
~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✅ Good: Comprehensive error handling
   def safe_trajectory_operation(trajectory_params):
       """Safe trajectory operation with comprehensive error handling."""
       
       try:
           # Initialize components
           urdf_processor = URDFToSerialManipulator("robot.urdf")
           robot = urdf_processor.serial_manipulator
           dynamics = urdf_processor.dynamics
           joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
           
           trajectory_planner = TrajectoryPlanning(
               robot, "robot.urdf", dynamics, joint_limits
           )
           
           # Perform trajectory generation
           result = trajectory_planner.joint_trajectory(**trajectory_params)
           
           # Validate result
           if result is None or result["positions"].shape[0] == 0:
               print("Invalid result, using fallback")
               return create_fallback_trajectory(**trajectory_params)
           
           return result
           
       except FileNotFoundError:
           print("URDF file not found, using default robot model")
           return create_default_trajectory(**trajectory_params)
       except MemoryError:
           print("Insufficient memory, reducing problem size")
           reduced_params = trajectory_params.copy()
           reduced_params['N'] = min(reduced_params.get('N', 1000), 500)
           return safe_trajectory_operation(reduced_params)
       except Exception as e:
           print(f"Unexpected error: {e}. Using basic fallback.")
           return create_fallback_trajectory(**trajectory_params)

   def create_fallback_trajectory(**params):
       """Create simple linear trajectory as fallback."""
       thetastart = params.get('thetastart', np.zeros(6))
       thetaend = params.get('thetaend', np.ones(6))
       N = params.get('N', 100)
       
       positions = np.linspace(thetastart, thetaend, N)
       velocities = np.zeros_like(positions)
       accelerations = np.zeros_like(positions)
       
       return {
           'positions': positions,
           'velocities': velocities,
           'accelerations': accelerations
       }

Memory Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✅ Good: Clean up GPU memory after large computations
   def efficient_trajectory_processing():
       """Process trajectories efficiently with memory management."""
       
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       results = []
       
       try:
           for i in range(10):  # Process multiple trajectories
               thetastart = np.zeros(6)
               thetaend = np.random.uniform(-1, 1, 6)
               
               trajectory = trajectory_planner.joint_trajectory(
                   thetastart, thetaend, Tf=2.0, N=1000, method=3
               )
               results.append(trajectory)
           
           return results
           
       finally:
           # Clean up GPU memory
           try:
               import cupy as cp
               mempool = cp.get_default_memory_pool()
               mempool.free_all_blocks()
           except ImportError:
               pass

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✅ Good: Minimize data transfers and reuse components
   class EfficientTrajectoryProcessor:
       def __init__(self, urdf_path):
           """Initialize processor with reusable components."""
           
           # Initialize once and reuse
           self.urdf_processor = URDFToSerialManipulator(urdf_path)
           self.robot = self.urdf_processor.serial_manipulator
           self.dynamics = self.urdf_processor.dynamics
           self.joint_limits = [(-np.pi, np.pi)] * len(self.dynamics.Glist)
           
           self.trajectory_planner = TrajectoryPlanning(
               self.robot, urdf_path, self.dynamics, self.joint_limits
           )
               
       def process_trajectory(self, thetastart, thetaend, Tf=2.0, N=1000, method=3):
           """Process single trajectory efficiently."""
           
           return self.trajectory_planner.joint_trajectory(
               thetastart, thetaend, Tf, N, method
           )
       
       def process_batch(self, trajectory_configs):
           """Process multiple trajectories efficiently."""
           
           results = []
           for config in trajectory_configs:
               result = self.trajectory_planner.joint_trajectory(**config)
               results.append(result)
           
           # Clean up after batch
           try:
               import cupy as cp
               mempool = cp.get_default_memory_pool()
               mempool.free_all_blocks()
           except ImportError:
               pass
               
           return results

   # Usage example
   processor = EfficientTrajectoryProcessor("robot.urdf")

   configs = [
       {"thetastart": np.zeros(6), "thetaend": np.ones(6)*0.5, "Tf": 2.0, "N": 1000, "method": 3},
       {"thetastart": np.ones(6)*0.5, "thetaend": np.zeros(6), "Tf": 1.5, "N": 800, "method": 5},
   ]

   results = processor.process_batch(configs)

Integration with ManipulaPy
------------------------------

Integration with Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.sim import Simulation

   def gpu_accelerated_simulation():
       """Example of GPU-accelerated simulation."""
       
       # Create simulation
       joint_limits = [(-np.pi, np.pi)] * 6
       torque_limits = [(-50, 50)] * 6
       
       sim = Simulation(
           urdf_file_path="robot.urdf",
           joint_limits=joint_limits,
           torque_limits=torque_limits
       )
       
       # Initialize robot and planner
       sim.initialize_robot()
       sim.initialize_planner_and_controller()
       
       # Generate GPU-accelerated trajectory
       thetastart = np.zeros(6)
       thetaend = np.array([0.5, -0.3, 0.8, -0.2, 0.4, -0.6])
       
       trajectory = sim.trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=3.0, N=3000, method=5
       )
       
       # Run simulation with GPU-generated trajectory
       end_pos = sim.run_trajectory(trajectory["positions"])
       
       return end_pos

   final_position = gpu_accelerated_simulation()

Integration with Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.control import ManipulatorController
   import cupy as cp

   def gpu_accelerated_control():
       """Example of using GPU acceleration with control."""
       
       # Initialize robot components
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # Create controller
       controller = ManipulatorController(dynamics)
       
       # Generate reference trajectory using GPU
       thetastart = np.zeros(6)
       thetaend = np.ones(6) * 0.5
       
       trajectory = trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=2.0, N=2000, method=5
       )
       
       # Use CuPy for control computations if available
       try:
           import cupy as cp
           
           # Convert to CuPy arrays for GPU computation
           thetalistd = cp.asarray(trajectory["positions"][0])
           dthetalistd = cp.asarray(trajectory["velocities"][0])
           ddthetalistd = cp.asarray(trajectory["accelerations"][0])
           
           thetalist = cp.zeros(6)
           dthetalist = cp.zeros(6)
           g = cp.array([0, 0, -9.81])
           
           # Control gains
           Kp = cp.array([10.0] * 6)
           Ki = cp.array([0.1] * 6)
           Kd = cp.array([1.0] * 6)
           
           # Compute control signal
           tau = controller.computed_torque_control(
               thetalistd, dthetalistd, ddthetalistd,
               thetalist, dthetalist, g, dt=0.01,
               Kp=Kp, Ki=Ki, Kd=Kd
           )
           
           return cp.asnumpy(tau)
           
       except ImportError:
           # Fall back to NumPy
           return controller.computed_torque_control(
               trajectory["positions"][0],
               trajectory["velocities"][0], 
               trajectory["accelerations"][0],
               np.zeros(6), np.zeros(6),
               np.array([0, 0, -9.81]),
               dt=0.01,
               Kp=np.array([10.0] * 6),
               Ki=np.array([0.1] * 6),
               Kd=np.array([1.0] * 6)
           )

   control_torque = gpu_accelerated_control()

Working Examples
------------------

Complete GPU Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def complete_gpu_workflow_example():
       """Complete example showing GPU acceleration workflow."""
       
       print("=== Complete GPU Acceleration Example ===")
       
       # 1. Check GPU availability
       try:
           from numba import cuda
           cuda_available = cuda.is_available()
           print(f"1. CUDA Available: {cuda_available}")
       except:
           cuda_available = False
           print("1. CUDA Not Available - using CPU fallback")
       
       try:
           import cupy as cp
           cupy_available = True
           print("   CuPy Available: True")
       except:
           cupy_available = False
           print("   CuPy Available: False")
       
       # 2. Initialize robot and trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       print(f"2. Initialized {len(dynamics.Glist)}-DOF robot")
       
       # 3. Generate trajectory (GPU accelerated if available)
       thetastart = np.zeros(len(dynamics.Glist))
       thetaend = np.ones(len(dynamics.Glist)) * 0.5
       N = 5000 if cuda_available else 1000  # Larger problem for GPU
       
       start_time = time.perf_counter()
       trajectory = trajectory_planner.joint_trajectory(
           thetastart, thetaend, Tf=2.0, N=N, method=5
       )
       trajectory_time = time.perf_counter() - start_time
       
       acceleration_type = "GPU" if cuda_available else "CPU"
       print(f"3. Generated {N}-point trajectory using {acceleration_type}")
       print(f"   Time: {trajectory_time*1000:.2f} ms")
       
       # 4. Compute dynamics (also GPU accelerated)
       start_time = time.perf_counter()
       torques = trajectory_planner.inverse_dynamics_trajectory(
           trajectory["positions"],
           trajectory["velocities"],
           trajectory["accelerations"]
       )
       dynamics_time = time.perf_counter() - start_time
       
       print(f"4. Computed inverse dynamics")
       print(f"   Time: {dynamics_time*1000:.2f} ms")
       print(f"   Torques shape: {torques.shape}")
       
       # 5. Memory cleanup
       if cupy_available:
           try:
               import cupy as cp
               mempool = cp.get_default_memory_pool()
               mempool.free_all_blocks()
               print("5. GPU memory cleaned up")
           except:
               print("5. Memory cleanup skipped")
       else:
           print("5. No GPU memory to clean up")
       
       # 6. Performance summary
       total_time = trajectory_time + dynamics_time
       print(f"\n=== Performance Summary ===")
       print(f"Total computation time: {total_time*1000:.2f} ms")
       print(f"Trajectory generation: {trajectory_time*1000:.2f} ms")
       print(f"Dynamics computation: {dynamics_time*1000:.2f} ms")
       print(f"Acceleration: {acceleration_type}")
       
       return {
           'trajectory': trajectory,
           'torques': torques,
           'performance': {
               'total_time_ms': total_time * 1000,
               'trajectory_time_ms': trajectory_time * 1000,
               'dynamics_time_ms': dynamics_time * 1000,
               'acceleration_type': acceleration_type,
               'problem_size': N
           }
       }

   # Run the complete example
   import time
   results = complete_gpu_workflow_example()

Batch Processing Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_processing_example():
       """Example of efficient batch processing with GPU acceleration."""
       
       print("=== Batch Processing Example ===")
       
       # Initialize trajectory processor
       processor = EfficientTrajectoryProcessor("robot.urdf")
       
       # Define multiple trajectory configurations
       trajectory_configs = [
           {
               "thetastart": np.zeros(6), 
               "thetaend": np.array([0.5, -0.3, 0.2, -0.4, 0.6, -0.1]),
               "Tf": 2.0, "N": 1000, "method": 3
           },
           {
               "thetastart": np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4]), 
               "thetaend": np.zeros(6),
               "Tf": 1.5, "N": 800, "method": 5
           },
           {
               "thetastart": np.ones(6) * 0.2, 
               "thetaend": np.ones(6) * -0.3,
               "Tf": 3.0, "N": 1500, "method": 3
           },
           {
               "thetastart": np.random.uniform(-0.5, 0.5, 6), 
               "thetaend": np.random.uniform(-0.5, 0.5, 6),
               "Tf": 2.5, "N": 1200, "method": 5
           }
       ]
       
       # Process batch
       start_time = time.perf_counter()
       results = processor.process_batch(trajectory_configs)
       batch_time = time.perf_counter() - start_time
       
       # Analyze results
       total_points = sum(result["positions"].shape[0] for result in results)
       
       print(f"Processed {len(trajectory_configs)} trajectories")
       print(f"Total trajectory points: {total_points}")
       print(f"Batch processing time: {batch_time*1000:.2f} ms")
       print(f"Average time per trajectory: {batch_time*1000/len(trajectory_configs):.2f} ms")
       
       # Validate results
       for i, result in enumerate(results):
           config = trajectory_configs[i]
           expected_shape = (config["N"], 6)
           actual_shape = result["positions"].shape
           
           print(f"Trajectory {i+1}: {actual_shape} ({'✅' if actual_shape == expected_shape else '❌'})")
       
       return results

   batch_results = batch_processing_example()

Performance Profiling Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def performance_profiling_example():
       """Comprehensive performance profiling example."""
       
       print("=== Performance Profiling Example ===")
       
       # Check available acceleration
       try:
           from numba import cuda
           cuda_available = cuda.is_available()
       except:
           cuda_available = False
       
       # Initialize trajectory planner
       urdf_processor = URDFToSerialManipulator("robot.urdf")
       robot = urdf_processor.serial_manipulator
       dynamics = urdf_processor.dynamics
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
       
       trajectory_planner = TrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits
       )
       
       # Test different problem sizes
       problem_sizes = [100, 500, 1000, 2000, 5000]
       
       print(f"Testing trajectory generation performance:")
       print(f"Acceleration: {'GPU (CUDA)' if cuda_available else 'CPU'}")
       print("Problem Size | Time (ms) | Points/sec")
       print("-" * 40)
       
       results = []
       
       for N in problem_sizes:
           thetastart = np.zeros(len(dynamics.Glist))
           thetaend = np.random.uniform(-1, 1, len(dynamics.Glist))
           
           # Warm up (first run may be slower due to initialization)
           trajectory_planner.joint_trajectory(
               thetastart, thetaend, Tf=2.0, N=min(N, 100), method=3
           )
           
           # Measure actual performance
           start_time = time.perf_counter()
           trajectory = trajectory_planner.joint_trajectory(
               thetastart, thetaend, Tf=2.0, N=N, method=3
           )
           elapsed_time = time.perf_counter() - start_time
           
           points_per_sec = N / elapsed_time
           
           print(f"{N:11d} | {elapsed_time*1000:8.2f} | {points_per_sec:9.0f}")
           
           results.append({
               'problem_size': N,
               'time_ms': elapsed_time * 1000,
               'points_per_sec': points_per_sec,
               'cuda_available': cuda_available
           })
           
           # Clean up GPU memory between tests
           try:
               import cupy as cp
               mempool = cp.get_default_memory_pool()
               mempool.free_all_blocks()
           except ImportError:
               pass
       
       # Performance analysis
       if len(results) >= 2:
           small_size_time = results[0]['time_ms']
           large_size_time = results[-1]['time_ms']
           
           print(f"\n=== Performance Analysis ===")
           print(f"Smallest problem: {results[0]['problem_size']} points in {small_size_time:.2f} ms")
           print(f"Largest problem: {results[-1]['problem_size']} points in {large_size_time:.2f} ms")
           
           if large_size_time > 0:
               efficiency = (results[-1]['problem_size'] / results[0]['problem_size']) / (large_size_time / small_size_time)
               print(f"Scaling efficiency: {efficiency:.2f} (1.0 = linear scaling)")
       
       return results

   profile_results = performance_profiling_example()

When to Use GPU Acceleration
-------------------------------

**Recommended for:**

- **Large trajectory generation** (N > 1000 points)
- **Batch dynamics computation** for multiple trajectories
- **Dense potential field calculations** with many sample points
- **Real-time path planning** with many obstacles
- **Iterative optimization algorithms** with repeated computations

**Not recommended for:**

- **Small computations** (N < 100 points)
- **One-off calculations** or prototype development
- **Memory-constrained systems** with limited GPU memory
- **Development/debugging phases** where CPU debugging is easier

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Problem Size Recommendations
   :header-rows: 1
   :widths: 20 30 25 25

   * - Problem Size
     - Trajectory Points
     - Recommended Hardware
     - Expected Speedup
   * - Small
     - < 100
     - CPU sufficient
     - No benefit
   * - Medium  
     - 100-1000
     - GPU optional
     - 2-5x speedup
   * - Large
     - 1000-10000
     - GPU recommended
     - 10-50x speedup
   * - Very Large
     - > 10000
     - GPU required
     - 50-100x speedup

Getting Started Checklist
----------------------------

Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **✅ Install CUDA support**: ``pip install cupy-cuda11x numba ManipulaPy``
2. **✅ Verify installation**: Check ``cuda.is_available()``
3. **✅ Start with TrajectoryPlanning**: Use existing ManipulaPy classes
4. **✅ Test performance**: Compare GPU vs CPU for your use case
5. **✅ Optimize gradually**: Apply memory management best practices
6. **✅ Monitor system**: Watch for memory usage and thermal limits

Quick Start Guide
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick start: GPU-accelerated trajectory generation
   
   # 1. Import required modules
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.path_planning import TrajectoryPlanning
   import numpy as np
   
   # 2. Load robot
   urdf_processor = URDFToSerialManipulator("your_robot.urdf")
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics
   
   # 3. Create trajectory planner (GPU acceleration automatic)
   joint_limits = [(-np.pi, np.pi)] * len(dynamics.Glist)
   planner = TrajectoryPlanning(robot, "your_robot.urdf", dynamics, joint_limits)
   
   # 4. Generate trajectory
   start = np.zeros(len(dynamics.Glist))
   end = np.ones(len(dynamics.Glist)) * 0.5
   
   trajectory = planner.joint_trajectory(
       start, end, Tf=2.0, N=2000, method=5  # Large N benefits from GPU
   )
   
   # 5. Compute dynamics
   torques = planner.inverse_dynamics_trajectory(
       trajectory["positions"],
       trajectory["velocities"],
       trajectory["accelerations"]
   )
   
   print(f"Generated {trajectory['positions'].shape[0]} trajectory points")
   print(f"Computed {torques.shape[0]} torque values")

Common Patterns
~~~~~~~~~~~~~~~~~~~~

**Pattern 1: Batch Processing**

.. code-block:: python

   # Process multiple trajectories efficiently
   configs = [
       {"thetastart": start1, "thetaend": end1, "Tf": 2.0, "N": 1000, "method": 3},
       {"thetastart": start2, "thetaend": end2, "Tf": 1.5, "N": 800, "method": 5},
       # ... more configurations
   ]
   
   results = []
   for config in configs:
       trajectory = planner.joint_trajectory(**config)
       results.append(trajectory)

**Pattern 2: Progressive Problem Sizing**

.. code-block:: python

   # Start small, scale up based on performance
   base_N = 500
   
   # Test performance with base size
   start_time = time.perf_counter()
   test_trajectory = planner.joint_trajectory(start, end, Tf=2.0, N=base_N, method=3)
   test_time = time.perf_counter() - start_time
   
   # Scale up if performance is good
   if test_time < 0.1:  # Less than 100ms
       N = base_N * 10  # Scale up 10x
   else:
       N = base_N
   
   # Generate final trajectory
   trajectory = planner.joint_trajectory(start, end, Tf=2.0, N=N, method=3)

**Pattern 3: Memory-Aware Processing**

.. code-block:: python

   # Process with automatic memory cleanup
   try:
       # Large computation
       trajectory = planner.joint_trajectory(start, end, Tf=2.0, N=10000, method=5)
       torques = planner.inverse_dynamics_trajectory(
           trajectory["positions"], trajectory["velocities"], trajectory["accelerations"]
       )
       
   finally:
       # Always clean up
       try:
           import cupy as cp
           cp.get_default_memory_pool().free_all_blocks()
       except ImportError:
           pass



Conclusion
-------------

The ManipulaPy CUDA Kernels module provides significant performance improvements for robotics applications through:

- **High-Performance Computing**: 10-100x speedup for suitable workloads
- **Seamless Integration**: Works directly with TrajectoryPlanning and other ManipulaPy modules
- **Automatic Fallback**: Graceful degradation to CPU when GPU unavailable
- **Memory Efficiency**: Optimized GPU memory management

The GPU acceleration is built into ManipulaPy's core modules like ``TrajectoryPlanning``, so you can benefit from it without changing your existing code - just install the CUDA dependencies and ManipulaPy will automatically use GPU acceleration when available.

For additional support and advanced usage patterns, refer to the `ManipulaPy documentation <https://manipulapy.readthedocs.io>`_ and `GitHub repository <https://github.com/manipulapy/ManipulaPy>`_.

API Reference
~~~~~~~~~~~~~~~~~

For complete function documentation: :doc:`../api/cuda_kernels`