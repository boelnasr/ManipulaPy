Simulation Module User Guide
============================

The Simulation Module provides a comprehensive PyBullet-based environment for visualizing,
testing, and interacting with your ManipulaPy-powered robot models. It supports
real-time control, trajectory playback, collision checking, data logging, and GPU-accelerated
computations through CuPy integration.

.. note::
   This guide assumes Python 3.10+, PyBullet 3.2.5+, CuPy 12.0+, and Matplotlib 3.5+.
   For optimal performance, CUDA-capable GPU is recommended but not required.

.. contents:: Table of Contents
   :depth: 3
   :local:

Installation and Setup
----------------------

Prerequisites
~~~~~~~~~~~~~

Before using the Simulation module, ensure you have the following dependencies:

**Core Dependencies**

.. code-block:: bash

   pip install ManipulaPy[simulation] pybullet>=3.2.5 matplotlib>=3.5 numpy>=1.21

**GPU Acceleration (Optional but Recommended)**

.. code-block:: bash

   pip install cupy-cuda11x  # For CUDA 11.x
   # or
   pip install cupy-cuda12x  # For CUDA 12.x

**URDF Processing**

.. code-block:: bash

   pip install urchin>=0.0.27

Verification
~~~~~~~~~~~~

Verify your installation with this simple test:

.. code-block:: python

   import pybullet as p
   import ManipulaPy.sim as sim
   import cupy as cp

   print("PyBullet version:", p.__version__)
   print("CuPy version:", cp.__version__)
   print("CUDA available:", cp.cuda.is_available())
   print("Simulation module loaded:", hasattr(sim, "Simulation"))

.. tip::
   If CUDA is not available, the simulation will automatically fall back to CPU computation
   with slightly reduced performance.

Quick Start
-----------

Basic Trajectory Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch a simple trajectory playback with your robot:

.. code-block:: python

   from ManipulaPy.sim import Simulation
   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf
   import numpy as np

   # Define joint limits for a 6-DOF manipulator
   joint_limits = [(-3.14, 3.14)] * 6
   torque_limits = [(-50, 50)] * 6  # Newton-meters

   # Create Simulation instance
   sim = Simulation(
       urdf_file_path=xarm_urdf,
       joint_limits=joint_limits,
       torque_limits=torque_limits,
       time_step=0.01,           # 100 Hz simulation
       real_time_factor=1.0      # Real-time playback
   )

   # Initialize robot dynamics and planning
   sim.initialize_robot()
   sim.initialize_planner_and_controller()

   # Generate a smooth sinusoidal trajectory
   t = np.linspace(0, 4*np.pi, 300)
   trajectory = []
   for t_i in t:
       # Create smooth joint motion
       joint_angles = 0.5 * np.sin(0.5 * t_i) * np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1])
       trajectory.append(joint_angles)

   # Execute trajectory with visualization
   final_ee_position = sim.run_trajectory(trajectory)
   print(f"Final end-effector position: {final_ee_position}")

Interactive Manual Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter manual control mode for real-time robot manipulation:

.. code-block:: python

   # Enter interactive mode with GUI sliders
   sim.manual_control()
   
   # This opens PyBullet GUI with:
   # - Joint sliders for each degree of freedom
   # - Gravity control (-20 to 20 m/s²)
   # - Time step adjustment (0.001 to 0.1 s)
   # - Reset button to return to home position

Module API Reference
--------------------

Simulation Class
~~~~~~~~~~~~~~~~

.. autoclass:: ManipulaPy.sim.Simulation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Constructor Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: __init__(urdf_file_path, joint_limits, torque_limits=None, time_step=0.01, real_time_factor=1.0, physics_client=None)

   Initialize the simulation environment.

   :param str urdf_file_path: Path to the robot's URDF file
   :param list joint_limits: List of (min, max) tuples for each joint in radians
   :param list torque_limits: Optional list of (min, max) torques in N⋅m. If None, uses unlimited torques
   :param float time_step: Physics simulation time step in seconds (default: 0.01)
   :param float real_time_factor: Playback speed multiplier (1.0 = real-time, 0.5 = half-speed)
   :param int physics_client: Existing PyBullet client ID, or None to create new GUI client

   :raises FileNotFoundError: If URDF file doesn't exist
   :raises ValueError: If joint_limits and URDF DOF don't match

Core Simulation Methods
^^^^^^^^^^^^^^^^^^^^^^^

**Trajectory Execution**

.. py:method:: run_trajectory(joint_trajectory) -> np.ndarray

   Execute a sequence of joint configurations with physics simulation and visualization.

   :param list joint_trajectory: List of joint angle arrays, each matching robot DOF
   :return: Final end-effector position [x, y, z]
   :rtype: np.ndarray

   .. code-block:: python

      # Example: Linear interpolation between two poses
      start = np.zeros(6)
      end = np.array([0.5, -0.3, 0.8, 0, -0.5, 0])
      
      trajectory = []
      for i in range(100):
          alpha = i / 99.0
          pose = start + alpha * (end - start)
          trajectory.append(pose)
      
      final_pos = sim.run_trajectory(trajectory)

**Controller Integration**

.. py:method:: run_controller(controller, desired_positions, desired_velocities, desired_accelerations, g, Ftip, Kp, Ki, Kd) -> np.ndarray

   Execute closed-loop control with real-time feedback and visualization.

   :param ManipulatorController controller: Controller instance from ManipulaPy.control
   :param np.ndarray desired_positions: Desired joint positions (N_steps × DOF)
   :param np.ndarray desired_velocities: Desired joint velocities (N_steps × DOF)
   :param np.ndarray desired_accelerations: Desired joint accelerations (N_steps × DOF)
   :param list g: Gravity vector [gx, gy, gz] in m/s²
   :param list Ftip: External force/torque at end-effector [fx, fy, fz, τx, τy, τz]
   :param list Kp: Proportional gains for each joint
   :param list Ki: Integral gains for each joint
   :param list Kd: Derivative gains for each joint
   :return: Final end-effector position
   :rtype: np.ndarray

**Manual Control and Interaction**

.. py:method:: manual_control()

   Enter interactive manual control mode with GUI sliders.

   Creates real-time sliders for:
   - Each robot joint (within specified limits)
   - Gravity magnitude and direction
   - Physics time step
   - Reset button for returning to home position

   :raises KeyboardInterrupt: Exit manual mode with Ctrl+C

Initialization and Setup Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: initialize_robot()

   Process URDF file and create robot dynamics model.

   This method:
   - Loads and parses the URDF file
   - Extracts kinematic and dynamic parameters
   - Creates SerialManipulator and ManipulatorDynamics instances
   - Identifies non-fixed joints for control

.. py:method:: initialize_planner_and_controller()

   Initialize trajectory planning and control modules.

   Creates:
   - TrajectoryPlanning instance for path generation
   - ManipulatorController for closed-loop control
   - Collision checking and potential field modules

State Management and Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: get_joint_positions() -> np.ndarray

   Get current joint positions from simulation.

   :return: Joint angles in radians
   :rtype: np.ndarray

.. py:method:: set_joint_positions(joint_positions)

   Set target joint positions for position control.

   :param array_like joint_positions: Target angles in radians

.. py:method:: check_collisions()

   Check for self-collisions and log contact points.

   Queries PyBullet contact detection and logs warnings for any detected collisions
   between robot links.

Data Logging and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: save_joint_states(filename="joint_states.csv")

   Save current joint states to CSV file.

   :param str filename: Output CSV filename
   
   Creates CSV with columns: Position, Velocity for each joint.

.. py:method:: plot_trajectory_in_scene(joint_trajectory, end_effector_trajectory)

   Create 3D visualization of end-effector trajectory.

   :param list joint_trajectory: Joint angle trajectory
   :param list end_effector_trajectory: End-effector position trajectory

Advanced Examples
-----------------

Closed-Loop Control Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate advanced controller integration with GPU acceleration:

.. code-block:: python

   from ManipulaPy.sim import Simulation
   from ManipulaPy.control import ManipulatorController
   from ManipulaPy.path_planning import TrajectoryPlanning
   import cupy as cp
   import numpy as np

   # Setup simulation
   sim = Simulation(urdf_file_path="robot.urdf", 
                   joint_limits=[(-np.pi, np.pi)]*6,
                   torque_limits=[(-100, 100)]*6)
   sim.initialize_robot()
   sim.initialize_planner_and_controller()

   # Generate reference trajectory
   planner = TrajectoryPlanning(sim.robot, "robot.urdf", sim.dynamics, 
                               sim.joint_limits, sim.torque_limits)
   
   traj_data = planner.joint_trajectory(
       thetastart=np.zeros(6),
       thetaend=np.array([0.5, -0.3, 0.8, 0, -0.5, 0]),
       Tf=5.0,  # 5 second trajectory
       N=500,   # 500 waypoints
       method=5 # Quintic time scaling
   )

   # Controller parameters
   controller = ManipulatorController(sim.dynamics)
   Kp = np.array([50, 40, 30, 20, 15, 10])  # Position gains
   Ki = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # Integral gains  
   Kd = np.array([5, 4, 3, 2, 1.5, 1])     # Derivative gains

   # Execute closed-loop control
   final_pos = sim.run_controller(
       controller=controller,
       desired_positions=traj_data["positions"],
       desired_velocities=traj_data["velocities"], 
       desired_accelerations=traj_data["accelerations"],
       g=[0, 0, -9.81],
       Ftip=[0, 0, 0, 0, 0, 0],
       Kp=Kp, Ki=Ki, Kd=Kd
   )

Multi-Phase Simulation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine trajectory execution, manual control, and data logging:

.. code-block:: python

   import time
   from pathlib import Path

   # Phase 1: Automated trajectory execution
   print("Phase 1: Running automated trajectory...")
   
   # Create complex multi-segment trajectory
   segments = []
   waypoints = [
       np.array([0, 0, 0, 0, 0, 0]),           # Home
       np.array([0.8, -0.5, 0.3, 0.2, -0.1, 0]), # Intermediate
       np.array([0.2, 0.7, -0.4, -0.3, 0.6, 0.8]) # Target
   ]
   
   for i in range(len(waypoints)-1):
       segment = []
       for j in range(50):
           alpha = j / 49.0
           pose = waypoints[i] + alpha * (waypoints[i+1] - waypoints[i])
           segment.append(pose)
       segments.extend(segment)
   
   sim.run_trajectory(segments)
   
   # Phase 2: Manual inspection and adjustment
   print("Phase 2: Manual control mode...")
   print("Use GUI sliders to inspect robot configuration")
   print("Press Reset button when done")
   
   try:
       sim.manual_control()
   except KeyboardInterrupt:
       print("Manual control ended")
   
   # Phase 3: Data logging and analysis
   print("Phase 3: Saving simulation data...")
   
   # Save joint states
   timestamp = int(time.time())
   sim.save_joint_states(f"joint_states_{timestamp}.csv")
   
   # Additional trajectory analysis
   ee_positions = []
   for joint_config in segments[-10:]:  # Last 10 poses
       sim.set_joint_positions(joint_config)
       time.sleep(0.1)
       ee_pos = sim.get_joint_positions()  # Current state
       ee_positions.append(ee_pos)
   
   print(f"Trajectory analysis complete. Final EE positions saved.")

GPU-Accelerated Batch Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leverage CuPy for high-performance batch processing:

.. code-block:: python

   import cupy as cp
   import time

   # Generate large batch of test trajectories
   n_trajectories = 100
   trajectory_length = 200
   
   print(f"Generating {n_trajectories} test trajectories...")
   
   # Create random but smooth trajectories
   np.random.seed(42)
   trajectories = []
   
   for i in range(n_trajectories):
       # Random waypoints
       waypoints = np.random.uniform(-1, 1, (5, 6))  # 5 waypoints, 6 DOF
       
       # Smooth interpolation
       trajectory = []
       for j in range(4):  # 4 segments
           for k in range(trajectory_length // 4):
               alpha = k / (trajectory_length // 4 - 1)
               pose = waypoints[j] + alpha * (waypoints[j+1] - waypoints[j])
               trajectory.append(pose)
       
       trajectories.append(trajectory)
   
   # Batch execution with timing
   execution_times = []
   final_positions = []
   
   print("Executing batch simulation...")
   for i, traj in enumerate(trajectories):
       start_time = time.time()
       
       # Clip to joint limits
       traj_clipped = []
       for pose in traj:
           clipped = np.clip(pose, 
                           [limits[0] for limits in sim.joint_limits],
                           [limits[1] for limits in sim.joint_limits])
           traj_clipped.append(clipped)
       
       final_pos = sim.run_trajectory(traj_clipped)
       execution_time = time.time() - start_time
       
       execution_times.append(execution_time)
       final_positions.append(final_pos)
       
       if (i + 1) % 20 == 0:
           print(f"Completed {i+1}/{n_trajectories} trajectories")
   
   # Performance analysis
   avg_time = np.mean(execution_times)
   std_time = np.std(execution_times)
   
   print(f"\nBatch Simulation Results:")
   print(f"Average execution time: {avg_time:.3f} ± {std_time:.3f} seconds")
   print(f"Total simulation time: {sum(execution_times):.1f} seconds")
   print(f"Trajectories per minute: {60 * n_trajectories / sum(execution_times):.1f}")

Advanced Configuration
----------------------

Physics Parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune simulation physics for accuracy vs. performance:

.. code-block:: python

   # High-precision configuration
   sim_precise = Simulation(
       urdf_file_path="robot.urdf",
       joint_limits=joint_limits,
       time_step=0.001,      # 1000 Hz for high precision
       real_time_factor=0.1  # Slow motion for detailed analysis
   )

   # Real-time configuration  
   sim_realtime = Simulation(
       urdf_file_path="robot.urdf", 
       joint_limits=joint_limits,
       time_step=0.02,       # 50 Hz for real-time performance
       real_time_factor=1.0  # Real-time execution
   )

   # After initialization, adjust physics parameters
   import pybullet as p
   
   # Set solver iterations for accuracy
   p.setPhysicsEngineParameter(numSolverIterations=100)
   
   # Enable additional collision margin
   p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
   
   # Configure constraint solving
   p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS)

Custom Visualization and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add custom visual elements and debugging information:

.. code-block:: python

   import pybullet as p

   def add_coordinate_frame(position, orientation, size=0.1):
       """Add coordinate frame visualization."""
       # X-axis (red)
       p.addUserDebugLine(position, 
                         [position[0]+size, position[1], position[2]], 
                         [1,0,0], lineWidth=3)
       # Y-axis (green)  
       p.addUserDebugLine(position,
                         [position[0], position[1]+size, position[2]],
                         [0,1,0], lineWidth=3)
       # Z-axis (blue)
       p.addUserDebugLine(position,
                         [position[0], position[1], position[2]+size], 
                         [0,0,1], lineWidth=3)

   def add_workspace_visualization(sim, resolution=20):
       """Visualize robot workspace boundary."""
       workspace_points = []
       
       # Sample random joint configurations
       for _ in range(1000):
           joints = []
           for limit in sim.joint_limits:
               joints.append(np.random.uniform(limit[0], limit[1]))
           
           # Get end-effector position
           T = sim.robot.forward_kinematics(joints)
           workspace_points.append(T[:3, 3])
       
       # Draw workspace boundary points
       for point in workspace_points[::10]:  # Subsample for performance
           p.addUserDebugLine(point, point, [0.5, 0.5, 0.5], lineWidth=1)

   # Usage in simulation
   sim.initialize_robot()
   
   # Add visualization elements
   add_coordinate_frame([0, 0, 0], [0, 0, 0, 1], 0.2)  # World frame
   add_workspace_visualization(sim)

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

Optimize memory usage for long-running simulations:

.. code-block:: python

   import gc
   import psutil
   import cupy as cp

   def monitor_memory():
       """Monitor CPU and GPU memory usage."""
       # CPU memory
       cpu_memory = psutil.virtual_memory()
       print(f"CPU Memory: {cpu_memory.percent:.1f}% used")
       
       # GPU memory (if available)
       if cp.cuda.is_available():
           mempool = cp.get_default_memory_pool()
           print(f"GPU Memory: {mempool.used_bytes() / 1024**3:.2f} GB used")

   def cleanup_simulation():
       """Clean up memory after simulation."""
       # Force garbage collection
       gc.collect()
       
       # Clear GPU memory pool
       if cp.cuda.is_available():
           mempool = cp.get_default_memory_pool()
           mempool.free_all_blocks()
       
       print("Memory cleanup completed")

   # Example usage in long simulation loop
   for i in range(100):
       # Run trajectory
       sim.run_trajectory(trajectory)
       
       # Periodic cleanup
       if i % 10 == 0:
           monitor_memory()
           cleanup_simulation()

Parallel Simulation Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

Configure multiple simulation instances for parallel processing:

.. code-block:: python

   import multiprocessing as mp
   import pybullet as p

   def run_simulation_worker(worker_id, urdf_path, joint_limits, trajectory_queue, result_queue):
       """Worker function for parallel simulation."""
       # Create simulation in DIRECT mode (no GUI)
       client = p.connect(p.DIRECT)
       
       sim = Simulation(urdf_file_path=urdf_path,
                       joint_limits=joint_limits, 
                       physics_client=client)
       sim.initialize_robot()
       
       while True:
           try:
               trajectory = trajectory_queue.get(timeout=1)
               if trajectory is None:  # Shutdown signal
                   break
                   
               final_pos = sim.run_trajectory(trajectory)
               result_queue.put((worker_id, final_pos))
               
           except:
               break
       
       p.disconnect(client)

   # Usage example
   def parallel_simulation_example():
       n_workers = 4
       trajectory_queue = mp.Queue()
       result_queue = mp.Queue()
       
       # Start worker processes
       workers = []
       for i in range(n_workers):
           worker = mp.Process(target=run_simulation_worker,
                             args=(i, "robot.urdf", joint_limits, 
                                   trajectory_queue, result_queue))
           worker.start()
           workers.append(worker)
       
       # Submit trajectories
       for traj in trajectories:
           trajectory_queue.put(traj)
       
       # Collect results
       results = []
       for _ in trajectories:
           results.append(result_queue.get())
       
       # Shutdown workers
       for _ in workers:
           trajectory_queue.put(None)
       for worker in workers:
           worker.join()
       
       return results

Troubleshooting Guide
--------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Black screen or no GUI**

.. code-block:: python

   # Check PyBullet GUI availability
   try:
       client = p.connect(p.GUI)
       print("GUI mode available")
   except:
       print("GUI mode not available, using DIRECT mode")
       client = p.connect(p.DIRECT)
   finally:
       p.disconnect(client)

**Solutions:**
- Ensure X11 forwarding for remote systems: ``ssh -X username@hostname``
- Install GUI libraries: ``sudo apt-get install python3-tk``
- Use DIRECT mode for headless servers

**2. Joint limit violations**

.. code-block:: python

   def validate_trajectory(trajectory, joint_limits):
       """Validate trajectory against joint limits."""
       violations = []
       for i, pose in enumerate(trajectory):
           for j, (angle, (min_val, max_val)) in enumerate(zip(pose, joint_limits)):
               if angle < min_val or angle > max_val:
                   violations.append((i, j, angle, min_val, max_val))
       return violations

   # Usage
   violations = validate_trajectory(trajectory, sim.joint_limits)
   if violations:
       print(f"Found {len(violations)} joint limit violations")
       for step, joint, angle, min_val, max_val in violations[:5]:
           print(f"Step {step}, Joint {joint}: {angle:.3f} not in [{min_val:.3f}, {max_val:.3f}]")

**3. Simulation instability**

Reduce time step and increase solver iterations:

.. code-block:: python

   # More stable configuration
   sim = Simulation(urdf_file_path="robot.urdf",
                   joint_limits=joint_limits,
                   time_step=0.001)  # Smaller time step
   
   # Increase solver accuracy
   p.setPhysicsEngineParameter(numSolverIterations=200)
   p.setPhysicsEngineParameter(useSplitImpulse=True)

**4. Performance issues**

.. code-block:: python

   # Profile simulation performance
   import time
   import cProfile

   def profile_trajectory(sim, trajectory):
       """Profile trajectory execution."""
       profiler = cProfile.Profile()
       
       start_time = time.time()
       profiler.enable()
       
       sim.run_trajectory(trajectory)
       
       profiler.disable()
       execution_time = time.time() - start_time
       
       print(f"Execution time: {execution_time:.3f} seconds")
       print(f"FPS: {len(trajectory) / execution_time:.1f}")
       
       # Print top time consumers
       profiler.print_stats(sort='cumtime', lines=10)

**5. CUDA/CuPy errors**

.. code-block:: python

   # Graceful fallback to CPU
   try:
       import cupy as cp
       device = cp.cuda.Device()
       print(f"Using GPU: {device.id}")
   except ImportError:
       print("CuPy not available, using NumPy")
       import numpy as cp
   except Exception as e:
       print(f"CUDA error: {e}, falling back to NumPy")
       import numpy as cp

Debugging Tools
~~~~~~~~~~~~~~

Enable detailed logging and visualization:

.. code-block:: python

   import logging

   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

   # Add trajectory debugging
   def debug_trajectory(sim, trajectory, step_interval=10):
       """Debug trajectory execution with detailed logging."""
       for i, pose in enumerate(trajectory):
           if i % step_interval == 0:
               # Log joint positions
               print(f"Step {i}: joints = {pose}")
               
               # Check joint limits
               for j, (angle, (min_val, max_val)) in enumerate(zip(pose, sim.joint_limits)):
                   if angle < min_val or angle > max_val:
                       print(f"  WARNING: Joint {j} out of limits: {angle:.3f}")
               
               # Visualize current pose
               sim.set_joint_positions(pose)
               time.sleep(0.1)

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~

Typical performance metrics for reference:

.. list-table:: Performance Benchmarks
   :header-rows: 1
   :widths: 30 20 20 30

   * - Configuration
     - FPS (GUI)
     - FPS (Headless)
     - Notes
   * - Default (dt=0.01)
     - 60-100
     - 200-500
     - Standard real-time
   * - High precision (dt=0.001)
     - 10-20
     - 50-100
     - Detailed physics
   * - Fast preview (dt=0.02)
     - 100-200
     - 500-1000
     - Quick visualization
   * - GPU accelerated
     - +20-50%
     - +30-70%
     - With CuPy controller

Best Practices
--------------

1. **Always validate trajectories** before execution
2. **Use appropriate time steps** for your application
3. **Monitor memory usage** in long simulations  
4. **Enable collision checking** for safety-critical applications
5. **Save simulation data** for post-analysis
6. **Use GPU acceleration** for compute-intensive controllers
7. **Profile performance** to identify bottlenecks

Integration with Other Modules
-----------------------------

The Simulation module integrates seamlessly with other ManipulaPy components:

- **Kinematics**: Forward/inverse kinematics for trajectory generation
- **Dynamics**: Physics-based simulation and control
- **Path Planning**: Trajectory optimization and collision avoidance  
- **Control**: Closed-loop feedback control implementation
- **Vision**: Sensor simulation and perception testing

See Also
--------

- :doc:`/user_guide/Control` — Controller Implementation Guide
- :doc:`/user_guide/Trajectory_Planning` — Path Planning and Optimization
- :doc:`/user_guide/Dynamics` — Robot Dynamics and Physics
- :doc:`/api/simulation` — Complete API Reference
- `PyBullet Documentation <https://pybullet.org/>`_ — Physics Engine Reference
- `CuPy Documentation <https://cupy.dev/>`_ — GPU Acceleration Reference