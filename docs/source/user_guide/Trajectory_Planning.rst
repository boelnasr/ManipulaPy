Trajectory Planning User Guide
===============================

This guide covers the trajectory planning capabilities in ManipulaPy, including joint-space and Cartesian-space trajectory generation, dynamics integration, and collision avoidance using CUDA acceleration.

Introduction
----------------

The TrajectoryPlanning class provides comprehensive trajectory generation and execution capabilities for robotic manipulators. It combines kinematic trajectory planning with dynamic analysis and collision avoidance to generate feasible, smooth robot motions.

**Key Features:**
- Joint-space trajectory generation with cubic/quintic time scaling
- Cartesian-space trajectory planning for end-effector paths
- CUDA-accelerated computations for real-time performance
- Integrated dynamics analysis (forward/inverse dynamics)
- Collision detection and avoidance using potential fields
- Support for various trajectory optimization objectives

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~~~~

This section summarizes the key mathematical constructs behind joint-space and Cartesian trajectory generation, as well as obstacle avoidance via potential fields.

Joint-Space Time Scaling
----------------------------

Given start :math:`\boldsymbol\theta_{0}` and end :math:`\boldsymbol\theta_{f}`, total duration :math:`T`, define a scalar time-scaling function

.. math::
   s = \frac{t}{T}, \quad s \in [0,1]

Common choices:

- **Cubic** (zero end-velocity):

  .. math::
     \sigma_{3}(s) = 3s^{2} - 2s^{3}

- **Quintic** (zero end-velocity & zero end-acceleration):

  .. math::
     \sigma_{5}(s) = 10s^{3} - 15s^{4} + 6s^{5}

.. math::
   \mathbf{q}(t) = \mathbf{q}_{\text{start}} + \sigma(s) \cdot (\mathbf{q}_{\text{end}} - \mathbf{q}_{\text{start}})

   \boldsymbol\theta(t)
     = \boldsymbol\theta_{0}
       + \bigl(\boldsymbol\theta_{f} - \boldsymbol\theta_{0}\bigr)\,\sigma_{m}(s),

with

- velocity:  
  :math:`\dot{\boldsymbol\theta}(t)
     = \frac{1}{T}\bigl(\boldsymbol\theta_{f}-\boldsymbol\theta_{0}\bigr)\,\sigma_{m}'(s)`

- acceleration:  
  :math:`\ddot{\boldsymbol\theta}(t)
     = \frac{1}{T^{2}}\bigl(\boldsymbol\theta_{f}-\boldsymbol\theta_{0}\bigr)\,\sigma_{m}''(s)`

where :math:`m=3` or :math:`5` for cubic/quintic.


Cartesian-Space Interpolation
--------------------------------

Given start/end end-effector poses :math:`X_{0},X_{f}\in SE(3)`, define the relative transform

.. math::

   \Delta X = X_{0}^{-1}X_{f}
   = \exp\bigl(\,[\Xi]\,\bigr),

with twist :math:`\Xi\in\mathfrak{se}(3)` given by the matrix logarithm.  Then

.. math::

   X(t)
     = X_{0}\,\exp\!\bigl([\Xi]\;\sigma_{m}(s)\bigr).

Extract :

- position:  the translational part of :math:`X(t)`  
- orientation:  the rotational part via Rodrigues’ formula on :math:`[\Xi]`  

Dynamics-Aware Torque Computation
---------------------------------

Once joint positions, velocities and accelerations are known, the required torques along the trajectory follow by inverse dynamics:

.. math::

   \boldsymbol\tau(t)
     = M\bigl(\boldsymbol\theta(t)\bigr)\,\ddot{\boldsymbol\theta}(t)
       + C\bigl(\boldsymbol\theta(t),\dot{\boldsymbol\theta}(t)\bigr)\,\dot{\boldsymbol\theta}(t)
       + G\bigl(\boldsymbol\theta(t)\bigr)
       \;+\; J(\boldsymbol\theta(t))^{T}\,\mathbf F_{\mathrm{tip}}.

Potential-Field Collision Avoidance
-----------------------------------

Obstacles at positions :math:`\mathbf p_{i}` generate repulsive potentials

.. math::

   U_{\mathrm{rep}}(\mathbf q)
     = \sum_{i}
       \begin{cases}
         \tfrac12\,\eta\Bigl(\tfrac{1}{\lVert \mathbf p(\mathbf q)-\mathbf p_{i}\rVert}
         - \tfrac{1}{d_{0}}\Bigr)^{2},
         & \lVert \mathbf p(\mathbf q)-\mathbf p_{i}\rVert < d_{0},\\
         0, & \text{otherwise},
       \end{cases}

and an attractive potential toward the goal :math:`U_{\mathrm{att}}(\mathbf q)
=\tfrac12\,\zeta\,\lVert \mathbf p(\mathbf q)-\mathbf p_{f}\rVert^{2}`.

The total artificial potential

.. math::

   U(\mathbf q) = U_{\mathrm{att}} + U_{\mathrm{rep}},

yields a force in joint space via the Jacobian transpose:

.. math::

   \boldsymbol\tau_{\mathrm{obs}}
     = -J(\mathbf q)^{T}\,\nabla_{\mathbf p}U\bigl(\mathbf p(\mathbf q)\bigr).

Trajectory generation incorporates these collision-avoidance torques into an optimization loop to adjust :math:`\boldsymbol\theta(t)` so that obstacles are circumvented while preserving smoothness.

Putting It All Together
~~~~~~~~~~~~~~~~~~~~~~~

1. **Time-scale** with :math:`\sigma_{3}` or :math:`\sigma_{5}` for smooth joint profiles.  
2. **Interpolate** Cartesian end-effector motion on SE(3).  
3. **Compute** velocities/accelerations and feed into inverse dynamics for torque evaluation.  
4. **Inject** obstacle gradients from potential fields to reshape the path.  

This mathematical framework underlies all high-level methods in the `TrajectoryPlanning` class.



Basic Usage
---------------

Setting Up Trajectory Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   
   # Load robot model
   processor = URDFToSerialManipulator("robot.urdf")
   robot = processor.serial_manipulator
   dynamics = processor.dynamics
   
   # Define joint and torque limits
   joint_limits = [(-np.pi, np.pi)] * 6  # 6-DOF robot
   torque_limits = [(-50, 50)] * 6       # ±50 N⋅m per joint
   
   # Create trajectory planner
   planner = TrajectoryPlanning(
       serial_manipulator=robot,
       urdf_path="robot.urdf",
       dynamics=dynamics,
       joint_limits=joint_limits,
       torque_limits=torque_limits
   )
   
   print("Trajectory planner initialized successfully")

Simple Joint Trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Define start and end configurations
   theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
   theta_end = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
   
   # Trajectory parameters
   Tf = 3.0      # Duration: 3 seconds
   N = 100       # Number of points
   method = 3    # Cubic time scaling
   
   # Generate trajectory
   trajectory = planner.joint_trajectory(theta_start, theta_end, Tf, N, method)
   
   print(f"Generated trajectory with {N} points")
   print(f"Position shape: {trajectory['positions'].shape}")
   print(f"Velocity shape: {trajectory['velocities'].shape}")
   print(f"Acceleration shape: {trajectory['accelerations'].shape}")
   
   # Verify start and end points
   np.testing.assert_allclose(trajectory['positions'][0], theta_start, rtol=1e-3)
   np.testing.assert_allclose(trajectory['positions'][-1], theta_end, rtol=1e-3)

TrajectoryPlanning Class
---------------------------

Class Constructor
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   TrajectoryPlanning(serial_manipulator, urdf_path, dynamics, joint_limits, torque_limits=None)

**Parameters:**
- ``serial_manipulator``: SerialManipulator instance for kinematics
- ``urdf_path``: Path to robot URDF file for collision checking
- ``dynamics``: ManipulatorDynamics instance for dynamics computations
- ``joint_limits``: List of (min, max) tuples for each joint
- ``torque_limits``: Optional list of (min, max) torque limits

**Attributes:**
- ``serial_manipulator``: Robot kinematics model
- ``dynamics``: Robot dynamics model
- ``joint_limits``: Joint position constraints
- ``torque_limits``: Joint torque constraints
- ``collision_checker``: Collision detection system
- ``potential_field``: Potential field for obstacle avoidance

Core Methods
----------------

joint_trajectory()
~~~~~~~~~~~~~~~~~~~~~

Generates smooth joint-space trajectories with CUDA acceleration:

.. code-block:: python

   def joint_trajectory_example():
       """Demonstrate joint trajectory generation options."""
       
       # Setup
       theta_start = np.zeros(6)
       theta_end = np.array([0.8, -0.5, 0.3, -0.2, 0.6, -0.4])
       
       # Method 1: Cubic time scaling (smooth velocity)
       traj_cubic = planner.joint_trajectory(
           theta_start, theta_end, Tf=2.0, N=50, method=3
       )
       
       # Method 2: Quintic time scaling (smooth acceleration)
       traj_quintic = planner.joint_trajectory(
           theta_start, theta_end, Tf=2.0, N=50, method=5
       )
       
       # Compare velocity profiles
       import matplotlib.pyplot as plt
       
       time_steps = np.linspace(0, 2.0, 50)
       
       plt.figure(figsize=(12, 4))
       
       plt.subplot(1, 2, 1)
       plt.plot(time_steps, traj_cubic['velocities'][:, 0], 'b-', label='Cubic')
       plt.plot(time_steps, traj_quintic['velocities'][:, 0], 'r-', label='Quintic')
       plt.title('Joint 1 Velocity')
       plt.xlabel('Time (s)')
       plt.ylabel('Velocity (rad/s)')
       plt.legend()
       plt.grid(True)
       
       plt.subplot(1, 2, 2)
       plt.plot(time_steps, traj_cubic['accelerations'][:, 0], 'b-', label='Cubic')
       plt.plot(time_steps, traj_quintic['accelerations'][:, 0], 'r-', label='Quintic')
       plt.title('Joint 1 Acceleration')
       plt.xlabel('Time (s)')
       plt.ylabel('Acceleration (rad/s²)')
       plt.legend()
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return traj_cubic, traj_quintic
   
   # Generate and compare trajectories
   cubic_traj, quintic_traj = joint_trajectory_example()

cartesian_trajectory()
~~~~~~~~~~~~~~~~~~~~~~~~~

Generates Cartesian-space trajectories for end-effector motion:

.. code-block:: python

   def cartesian_trajectory_example():
       """Demonstrate Cartesian trajectory generation."""
       
       # Define start and end poses
       X_start = np.eye(4)
       X_start[:3, 3] = [0.3, 0.2, 0.5]  # Start position
       
       X_end = np.eye(4) 
       X_end[:3, 3] = [0.5, -0.1, 0.4]   # End position
       # Add rotation (45° about Z-axis)
       angle = np.pi/4
       X_end[:3, :3] = np.array([
           [np.cos(angle), -np.sin(angle), 0],
           [np.sin(angle),  np.cos(angle), 0],
           [0,              0,             1]
       ])
       
       # Generate Cartesian trajectory
       cart_traj = planner.cartesian_trajectory(
           X_start, X_end, Tf=3.0, N=75, method=5
       )
       
       print("Cartesian trajectory generated:")
       print(f"- Positions: {cart_traj['positions'].shape}")
       print(f"- Velocities: {cart_traj['velocities'].shape}")
       print(f"- Accelerations: {cart_traj['accelerations'].shape}")
       print(f"- Orientations: {cart_traj['orientations'].shape}")
       
       # Visualize path
       positions = cart_traj['positions']
       
       plt.figure(figsize=(10, 8))
       
       # 3D path
       ax = plt.subplot(2, 2, 1, projection='3d')
       ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
       ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                 c='green', s=100, label='Start')
       ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                 c='red', s=100, label='End')
       ax.set_xlabel('X (m)')
       ax.set_ylabel('Y (m)')
       ax.set_zlabel('Z (m)')
       ax.set_title('3D Path')
       ax.legend()
       
       # X-Y projection
       plt.subplot(2, 2, 2)
       plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
       plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100)
       plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100)
       plt.xlabel('X (m)')
       plt.ylabel('Y (m)')
       plt.title('X-Y Projection')
       plt.grid(True)
       plt.axis('equal')
       
       # Velocity profile
       time_steps = np.linspace(0, 3.0, 75)
       velocities = cart_traj['velocities']
       velocity_magnitude = np.linalg.norm(velocities, axis=1)
       
       plt.subplot(2, 2, 3)
       plt.plot(time_steps, velocity_magnitude, 'r-', linewidth=2)
       plt.xlabel('Time (s)')
       plt.ylabel('Speed (m/s)')
       plt.title('End-Effector Speed')
       plt.grid(True)
       
       # Acceleration profile
       accelerations = cart_traj['accelerations']
       acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
       
       plt.subplot(2, 2, 4)
       plt.plot(time_steps, acceleration_magnitude, 'g-', linewidth=2)
       plt.xlabel('Time (s)')
       plt.ylabel('Acceleration (m/s²)')
       plt.title('End-Effector Acceleration')
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return cart_traj
   
   # Generate Cartesian trajectory
   cartesian_traj = cartesian_trajectory_example()

Dynamics Integration
-----------------------

inverse_dynamics_trajectory()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes required joint torques along a trajectory:

.. code-block:: python

   def dynamics_analysis_example():
       """Analyze dynamics along a trajectory."""
       
       # Generate joint trajectory
       theta_start = np.zeros(6)
       theta_end = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
       
       trajectory = planner.joint_trajectory(
           theta_start, theta_end, Tf=2.0, N=50, method=5
       )
       
       # Compute required torques
       torques = planner.inverse_dynamics_trajectory(
           trajectory['positions'],
           trajectory['velocities'], 
           trajectory['accelerations'],
           gravity_vector=[0, 0, -9.81],
           Ftip=[0, 0, 0, 0, 0, 0]  # No external forces
       )
       
       print(f"Torque trajectory shape: {torques.shape}")
       
       # Analyze torque requirements
       time_steps = np.linspace(0, 2.0, 50)
       
       plt.figure(figsize=(15, 10))
       
       # Plot joint torques
       for i in range(6):
           plt.subplot(2, 3, i+1)
           plt.plot(time_steps, torques[:, i], 'b-', linewidth=2)
           plt.axhline(y=planner.torque_limits[i][1], color='r', linestyle='--', 
                      label=f'Limit: ±{planner.torque_limits[i][1]} N⋅m')
           plt.axhline(y=planner.torque_limits[i][0], color='r', linestyle='--')
           plt.xlabel('Time (s)')
           plt.ylabel('Torque (N⋅m)')
           plt.title(f'Joint {i+1} Torque')
           plt.grid(True)
           plt.legend()
       
       plt.tight_layout()
       plt.show()
       
       # Check if torques exceed limits
       max_torques = np.max(np.abs(torques), axis=0)
       torque_limits_array = np.array([limit[1] for limit in planner.torque_limits])
       
       safety_factors = max_torques / torque_limits_array
       
       print("\nTorque Analysis:")
       for i, (max_torque, limit, safety) in enumerate(zip(max_torques, torque_limits_array, safety_factors)):
           status = "⚠️ EXCEEDED" if safety > 1.0 else "✓ OK"
           print(f"Joint {i+1}: Max {max_torque:.1f} N⋅m / Limit {limit:.1f} N⋅m ({safety:.1%}) {status}")
       
       return torques
   
   # Analyze dynamics
   trajectory_torques = dynamics_analysis_example()

forward_dynamics_trajectory()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates robot motion given applied torques:

.. code-block:: python

   def forward_dynamics_simulation():
       """Simulate robot motion using forward dynamics."""
       
       # Initial conditions
       theta_initial = np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.0])
       theta_dot_initial = np.zeros(6)
       
       # Define control torques (simple step input)
       N_steps = 100
       dt = 0.01
       
       tau_matrix = np.zeros((N_steps, 6))
       tau_matrix[:, 0] = 5.0   # 5 N⋅m on joint 1
       tau_matrix[:, 2] = -3.0  # -3 N⋅m on joint 3
       
       # External forces (none)
       Ftip_matrix = np.zeros((N_steps, 6))
       
       # Simulate forward dynamics
       sim_result = planner.forward_dynamics_trajectory(
           thetalist=theta_initial,
           dthetalist=theta_dot_initial,
           taumat=tau_matrix,
           g=[0, 0, -9.81],
           Ftipmat=Ftip_matrix,
           dt=dt,
           intRes=1
       )
       
       print("Forward dynamics simulation completed:")
       print(f"- Position trajectory: {sim_result['positions'].shape}")
       print(f"- Velocity trajectory: {sim_result['velocities'].shape}")
       print(f"- Acceleration trajectory: {sim_result['accelerations'].shape}")
       
       # Plot results
       time_steps = np.arange(N_steps) * dt
       
       plt.figure(figsize=(15, 8))
       
       # Joint positions
       plt.subplot(2, 3, 1)
       for i in range(6):
           plt.plot(time_steps, np.degrees(sim_result['positions'][:, i]), 
                   label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Position (degrees)')
       plt.title('Joint Positions')
       plt.legend()
       plt.grid(True)
       
       # Joint velocities  
       plt.subplot(2, 3, 2)
       for i in range(6):
           plt.plot(time_steps, sim_result['velocities'][:, i], 
                   label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Velocity (rad/s)')
       plt.title('Joint Velocities')
       plt.legend()
       plt.grid(True)
       
       # Applied torques
       plt.subplot(2, 3, 3)
       for i in range(6):
           plt.plot(time_steps, tau_matrix[:, i], label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Torque (N⋅m)')
       plt.title('Applied Torques')
       plt.legend()
       plt.grid(True)
       
       # End-effector trajectory
       ee_positions = []
       for pos in sim_result['positions']:
           T = planner.serial_manipulator.forward_kinematics(pos)
           ee_positions.append(T[:3, 3])
       ee_positions = np.array(ee_positions)
       
       ax = plt.subplot(2, 3, 4, projection='3d')
       ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2)
       ax.set_xlabel('X (m)')
       ax.set_ylabel('Y (m)')
       ax.set_zlabel('Z (m)')
       ax.set_title('End-Effector Path')
       
       # Energy analysis
       kinetic_energies = []
       for i, (pos, vel) in enumerate(zip(sim_result['positions'], sim_result['velocities'])):
           M = planner.dynamics.mass_matrix(pos)
           kinetic_energy = 0.5 * vel.T @ M @ vel
           kinetic_energies.append(kinetic_energy)
       
       plt.subplot(2, 3, 5)
       plt.plot(time_steps, kinetic_energies, 'r-', linewidth=2)
       plt.xlabel('Time (s)')
       plt.ylabel('Kinetic Energy (J)')
       plt.title('System Kinetic Energy')
       plt.grid(True)
       
       # Phase plot (position vs velocity for joint 1)
       plt.subplot(2, 3, 6)
       plt.plot(np.degrees(sim_result['positions'][:, 0]), 
               sim_result['velocities'][:, 0], 'g-', linewidth=2)
       plt.xlabel('Joint 1 Position (degrees)')
       plt.ylabel('Joint 1 Velocity (rad/s)')
       plt.title('Phase Plot (Joint 1)')
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return sim_result
   
   # Run forward dynamics simulation
   simulation_result = forward_dynamics_simulation()

Trajectory Visualization
---------------------------

plot_trajectory()
~~~~~~~~~~~~~~~~~~~~

Static plotting of trajectory data:

.. code-block:: python

   def trajectory_visualization_example():
       """Comprehensive trajectory visualization."""
       
       # Generate sample trajectory
       theta_start = np.array([0.0, 0.5, -0.3, 0.0, 0.2, 0.0])
       theta_end = np.array([0.8, -0.2, 0.4, -0.5, 0.6, -0.3])
       
       trajectory = planner.joint_trajectory(
           theta_start, theta_end, Tf=3.0, N=100, method=5
       )
       
       # Use built-in plotting method
       TrajectoryPlanning.plot_trajectory(
           trajectory, 
           Tf=3.0, 
           title="6-DOF Robot Joint Trajectory",
           labels=[f"Joint {i+1}" for i in range(6)]
       )
       
       return trajectory
   
   # Visualize trajectory
   sample_trajectory = trajectory_visualization_example()

plot_cartesian_trajectory()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualization for Cartesian trajectories:

.. code-block:: python

   def cartesian_visualization_example():
       """Visualize Cartesian trajectory."""
       
       # Generate Cartesian trajectory
       X_start = np.eye(4)
       X_start[:3, 3] = [0.4, 0.3, 0.5]
       
       X_end = np.eye(4)
       X_end[:3, 3] = [0.6, -0.2, 0.3]
       
       cart_traj = planner.cartesian_trajectory(
           X_start, X_end, Tf=2.5, N=80, method=3
       )
       
       # Use built-in Cartesian plotting
       planner.plot_cartesian_trajectory(
           cart_traj,
           Tf=2.5,
           title="End-Effector Cartesian Trajectory"
       )
       
       return cart_traj
   
   # Visualize Cartesian trajectory
   cartesian_viz = cartesian_visualization_example()

Advanced Features
---------------------

Collision Avoidance
~~~~~~~~~~~~~~~~~~~~~~

The trajectory planner includes collision detection and avoidance:

.. code-block:: python

   def collision_avoidance_example():
       """Demonstrate collision avoidance in trajectory planning."""
       
       # Generate trajectory that might have collisions
       theta_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
       theta_end = np.array([np.pi/2, np.pi/3, -np.pi/4, 0.0, np.pi/6, 0.0])
       
       trajectory = planner.joint_trajectory(
           theta_start, theta_end, Tf=3.0, N=150, method=5
       )
       
       print("Trajectory generated with collision avoidance:")
       print(f"- Points: {trajectory['positions'].shape[0]}")
       print(f"- Collision checks: Integrated via potential fields")
       
       # The trajectory planner automatically applies potential field
       # modifications to avoid collisions during generation
       
       # Analyze trajectory smoothness
       positions = trajectory['positions']
       velocities = trajectory['velocities']
       accelerations = trajectory['accelerations']
       
       # Compute smoothness metrics
       velocity_changes = np.diff(velocities, axis=0)
       acceleration_changes = np.diff(accelerations, axis=0)
       
       smoothness_metric = np.mean(np.linalg.norm(acceleration_changes, axis=1))
       print(f"- Trajectory smoothness metric: {smoothness_metric:.6f}")
       
       return trajectory
   
   # Generate collision-aware trajectory
   safe_trajectory = collision_avoidance_example()

Multi-Point Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating trajectories through multiple waypoints:

.. code-block:: python

   def multi_waypoint_trajectory():
       """Generate trajectory through multiple waypoints."""
       
       # Define waypoints
       waypoints = [
           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),           # Start
           np.array([0.3, 0.5, -0.2, 0.1, 0.3, -0.1]),         # Waypoint 1
           np.array([0.6, -0.3, 0.4, -0.2, 0.6, 0.2]),         # Waypoint 2
           np.array([0.8, 0.2, -0.1, 0.3, -0.2, -0.3])         # End
       ]
       
       # Generate trajectory segments
       segment_duration = 2.0
       points_per_segment = 50
       
       full_trajectory = {
           'positions': [],
           'velocities': [],
           'accelerations': []
       }
       
       for i in range(len(waypoints) - 1):
           segment = planner.joint_trajectory(
               waypoints[i], waypoints[i+1], 
               Tf=segment_duration, N=points_per_segment, method=5
           )
           
           # Append to full trajectory (avoid duplicate points)
           if i == 0:
               full_trajectory['positions'].extend(segment['positions'])
               full_trajectory['velocities'].extend(segment['velocities'])
               full_trajectory['accelerations'].extend(segment['accelerations'])
           else:
               # Skip first point to avoid duplication
               full_trajectory['positions'].extend(segment['positions'][1:])
               full_trajectory['velocities'].extend(segment['velocities'][1:])
               full_trajectory['accelerations'].extend(segment['accelerations'][1:])
       
       # Convert to numpy arrays
       for key in full_trajectory:
           full_trajectory[key] = np.array(full_trajectory[key])
       
       total_time = segment_duration * (len(waypoints) - 1)
       total_points = full_trajectory['positions'].shape[0]
       
       print(f"Multi-waypoint trajectory generated:")
       print(f"- Waypoints: {len(waypoints)}")
       print(f"- Total duration: {total_time} seconds")
       print(f"- Total points: {total_points}")
       
       # Plot the full trajectory
       time_steps = np.linspace(0, total_time, total_points)
       
       plt.figure(figsize=(15, 5))
       
       # Joint positions
       plt.subplot(1, 3, 1)
       for i in range(6):
           plt.plot(time_steps, np.degrees(full_trajectory['positions'][:, i]), 
                   label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Position (degrees)')
       plt.title('Multi-Waypoint Joint Positions')
       plt.legend()
       plt.grid(True)
       
       # Mark waypoints
       waypoint_times = [i * segment_duration for i in range(len(waypoints))]
       for wpt_time in waypoint_times:
           plt.axvline(x=wpt_time, color='red', linestyle='--', alpha=0.7)
       
       # Joint velocities
       plt.subplot(1, 3, 2)
       for i in range(6):
           plt.plot(time_steps, full_trajectory['velocities'][:, i], 
                   label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Velocity (rad/s)')
       plt.title('Joint Velocities')
       plt.legend()
       plt.grid(True)
       
       # End-effector path
       ee_positions = []
       for pos in full_trajectory['positions']:
           T = planner.serial_manipulator.forward_kinematics(pos)
           ee_positions.append(T[:3, 3])
       ee_positions = np.array(ee_positions)
       
       ax = plt.subplot(1, 3, 3, projection='3d')
       ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
              'b-', linewidth=2, label='Path')
       
       # Mark waypoint positions
       for i, waypoint in enumerate(waypoints):
           T = planner.serial_manipulator.forward_kinematics(waypoint)
           pos = T[:3, 3]
           ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, 
                     label=f'Waypoint {i+1}' if i == 0 else "")
       
       ax.set_xlabel('X (m)')
       ax.set_ylabel('Y (m)')
       ax.set_zlabel('Z (m)')
       ax.set_title('End-Effector Path')
       ax.legend()
       
       plt.tight_layout()
       plt.show()
       
       return full_trajectory, waypoints
   
   # Generate multi-waypoint trajectory
   multi_traj, waypoints = multi_waypoint_trajectory()

Performance Optimization
---------------------------

CUDA Acceleration
~~~~~~~~~~~~~~~~~~~

The trajectory planner uses CUDA for high-performance computations:

.. code-block:: python

   def performance_comparison():
       """Compare CPU vs CUDA performance for trajectory generation."""
       
       import time
       
       # Large trajectory for performance testing
       theta_start = np.zeros(6)
       theta_end = np.array([1.0, -0.8, 0.6, -0.4, 1.2, -0.6])
       
       N_large = 1000  # Many points for performance test
       Tf = 5.0
       
       print("Performance Comparison: CPU vs CUDA")
       print("=" * 40)
       
       # Time the trajectory generation
       start_time = time.time()
       
       trajectory_cuda = planner.joint_trajectory(
           theta_start, theta_end, Tf, N_large, method=5
       )
       
       cuda_time = time.time() - start_time
       
       print(f"CUDA trajectory generation:")
       print(f"- Points: {N_large}")
       print(f"- Time: {cuda_time:.3f} seconds")
       print(f"- Rate: {N_large/cuda_time:.1f} points/second")
       
       # Memory usage estimation
       memory_per_point = 6 * 4 * 3  # 6 joints * 4 bytes * 3 arrays (pos, vel, acc)
       total_memory = N_large * memory_per_point / 1024 / 1024  # MB
       
       print(f"- Memory usage: ~{total_memory:.1f} MB")
       
       # Test dynamics integration performance
       start_time = time.time()
       
       torques = planner.inverse_dynamics_trajectory(
           trajectory_cuda['positions'],
           trajectory_cuda['velocities'],
           trajectory_cuda['accelerations']
       )
       
       dynamics_time = time.time() - start_time
       
       print(f"\nDynamics computation:")
       print(f"- Time: {dynamics_time:.3f} seconds")
       print(f"- Rate: {N_large/dynamics_time:.1f} points/second")
       
       return trajectory_cuda, cuda_time, dynamics_time
   
   # Run performance comparison
   perf_traj, traj_time, dyn_time = performance_comparison()

Batch Processing
~~~~~~~~~~~~~~~~~~~

Processing multiple trajectories efficiently:

.. code-block:: python

   def batch_trajectory_processing():
       """Process multiple trajectories in batch for efficiency."""
       
       # Generate multiple start/end configurations
       n_trajectories = 10
       
       start_configs = []
       end_configs = []
       
       for i in range(n_trajectories):
           start = np.random.uniform(-0.5, 0.5, 6)
           end = np.random.uniform(-0.8, 0.8, 6)
           start_configs.append(start)
           end_configs.append(end)
       
       print(f"Batch processing {n_trajectories} trajectories:")
       
       # Process all trajectories
       trajectories = []
       torque_profiles = []
       
       start_time = time.time()
       
       for i, (start, end) in enumerate(zip(start_configs, end_configs)):
           # Generate trajectory
           traj = planner.joint_trajectory(start, end, Tf=2.0, N=50, method=5)
           
           # Compute dynamics
           torques = planner.inverse_dynamics_trajectory(
               traj['positions'], traj['velocities'], traj['accelerations']
           )
           
           trajectories.append(traj)
           torque_profiles.append(torques)
           
           if (i + 1) % 5 == 0:
               print(f"  Processed {i + 1}/{n_trajectories} trajectories")
       
       total_time = time.time() - start_time
       
       print(f"Batch processing completed:")
       print(f"- Total time: {total_time:.3f} seconds")
       print(f"- Average per trajectory: {total_time/n_trajectories:.3f} seconds")
       
       # Analyze batch results
       max_torques = []
       for torques in torque_profiles:
           max_torque = np.max(np.abs(torques))
           max_torques.append(max_torque)
       
       print(f"\nBatch analysis:")
       print(f"- Average max torque: {np.mean(max_torques):.2f} N⋅m")
       print(f"- Max torque range: {np.min(max_torques):.2f} - {np.max(max_torques):.2f} N⋅m")
       
       return trajectories, torque_profiles
   
   # Run batch processing
   batch_trajs, batch_torques = batch_trajectory_processing()

Real-Time Applications
-------------------------

Trajectory Execution
~~~~~~~~~~~~~~~~~~~~~~

Real-time trajectory following for robot control:

.. code-block:: python

   def real_time_trajectory_execution():
       """Simulate real-time trajectory execution."""
       
       # Generate reference trajectory
       theta_start = np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.0])
       theta_end = np.array([0.8, -0.3, 0.5, -0.2, 0.6, -0.4])
       
       ref_trajectory = planner.joint_trajectory(
           theta_start, theta_end, Tf=4.0, N=400, method=5  # 100 Hz
       )
       
       # Simulation parameters
       dt = 0.01  # 100 Hz control rate
       n_steps = ref_trajectory['positions'].shape[0]
       
       # Control parameters
       Kp = np.diag([100, 80, 60, 40, 30, 20])
       Kd = np.diag([10, 8, 6, 4, 3, 2])
       
       # Initialize simulation state
       current_pos = theta_start.copy()
       current_vel = np.zeros(6)
       
       # Storage for results
       actual_positions = []
       actual_velocities = []
       control_torques = []
       tracking_errors = []
       
       print("Simulating real-time trajectory execution...")
       
       for i in range(n_steps):
           # Get reference at current time
           ref_pos = ref_trajectory['positions'][i]
           ref_vel = ref_trajectory['velocities'][i]
           ref_acc = ref_trajectory['accelerations'][i]
           
           # Compute tracking error
           pos_error = ref_pos - current_pos
           vel_error = ref_vel - current_vel
           
           # PD control with feedforward
           tau_pd = Kp @ pos_error + Kd @ vel_error
           
           # Feedforward compensation
           tau_ff = planner.dynamics.inverse_dynamics(
               ref_pos, ref_vel, ref_acc, [0, 0, -9.81], np.zeros(6)
           )
           
           # Total control torque
           tau_total = tau_pd + tau_ff
           
           # Apply torque limits
           for j in range(6):
               tau_total[j] = np.clip(tau_total[j], 
                                    planner.torque_limits[j][0], 
                                    planner.torque_limits[j][1])
           
           # Simulate robot dynamics
           acceleration = planner.dynamics.forward_dynamics(
               current_pos, current_vel, tau_total, [0, 0, -9.81], np.zeros(6)
           )
           
           # Integrate (simple Euler integration)
           current_vel += acceleration * dt
           current_pos += current_vel * dt
           
           # Apply joint limits
           for j in range(6):
               if current_pos[j] < planner.joint_limits[j][0]:
                   current_pos[j] = planner.joint_limits[j][0]
                   current_vel[j] = 0
               elif current_pos[j] > planner.joint_limits[j][1]:
                   current_pos[j] = planner.joint_limits[j][1]
                   current_vel[j] = 0
           
           # Store results
           actual_positions.append(current_pos.copy())
           actual_velocities.append(current_vel.copy())
           control_torques.append(tau_total.copy())
           tracking_errors.append(np.linalg.norm(pos_error))
       
       # Convert to arrays
       actual_positions = np.array(actual_positions)
       actual_velocities = np.array(actual_velocities)
       control_torques = np.array(control_torques)
       tracking_errors = np.array(tracking_errors)
       
       # Analysis
       time_steps = np.arange(n_steps) * dt
       
       print("Trajectory execution completed:")
       print(f"- Duration: {time_steps[-1]:.1f} seconds")
       print(f"- Final tracking error: {tracking_errors[-1]:.4f} rad")
       print(f"- RMS tracking error: {np.sqrt(np.mean(tracking_errors**2)):.4f} rad")
       print(f"- Max tracking error: {np.max(tracking_errors):.4f} rad")
       
       # Plot results
       plt.figure(figsize=(15, 12))
       
       # Position tracking
       plt.subplot(3, 2, 1)
       for i in range(6):
           plt.plot(time_steps, np.degrees(ref_trajectory['positions'][:, i]), 
                   '--', alpha=0.7, label=f'Ref Joint {i+1}')
           plt.plot(time_steps, np.degrees(actual_positions[:, i]), 
                   '-', linewidth=2, label=f'Act Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Position (degrees)')
       plt.title('Position Tracking')
       plt.legend()
       plt.grid(True)
       
       # Velocity tracking
       plt.subplot(3, 2, 2)
       for i in range(6):
           plt.plot(time_steps, ref_trajectory['velocities'][:, i], 
                   '--', alpha=0.7, label=f'Ref Joint {i+1}')
           plt.plot(time_steps, actual_velocities[:, i], 
                   '-', linewidth=2, label=f'Act Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Velocity (rad/s)')
       plt.title('Velocity Tracking')
       plt.legend()
       plt.grid(True)
       
       # Control torques
       plt.subplot(3, 2, 3)
       for i in range(6):
           plt.plot(time_steps, control_torques[:, i], label=f'Joint {i+1}')
       plt.xlabel('Time (s)')
       plt.ylabel('Torque (N⋅m)')
       plt.title('Control Torques')
       plt.legend()
       plt.grid(True)
       
       # Tracking error
       plt.subplot(3, 2, 4)
       plt.plot(time_steps, np.degrees(tracking_errors), 'r-', linewidth=2)
       plt.xlabel('Time (s)')
       plt.ylabel('Tracking Error (degrees)')
       plt.title('Position Tracking Error')
       plt.grid(True)
       
       # End-effector tracking
       ref_ee_positions = []
       actual_ee_positions = []
       
       for ref_pos, act_pos in zip(ref_trajectory['positions'], actual_positions):
           T_ref = planner.serial_manipulator.forward_kinematics(ref_pos)
           T_act = planner.serial_manipulator.forward_kinematics(act_pos)
           ref_ee_positions.append(T_ref[:3, 3])
           actual_ee_positions.append(T_act[:3, 3])
       
       ref_ee_positions = np.array(ref_ee_positions)
       actual_ee_positions = np.array(actual_ee_positions)
       
       ax = plt.subplot(3, 2, 5, projection='3d')
       ax.plot(ref_ee_positions[:, 0], ref_ee_positions[:, 1], ref_ee_positions[:, 2], 
              'b--', alpha=0.7, linewidth=2, label='Reference')
       ax.plot(actual_ee_positions[:, 0], actual_ee_positions[:, 1], actual_ee_positions[:, 2], 
              'r-', linewidth=2, label='Actual')
       ax.set_xlabel('X (m)')
       ax.set_ylabel('Y (m)')
       ax.set_zlabel('Z (m)')
       ax.set_title('End-Effector Tracking')
       ax.legend()
       
       # Control effort
       plt.subplot(3, 2, 6)
       control_effort = np.linalg.norm(control_torques, axis=1)
       plt.plot(time_steps, control_effort, 'g-', linewidth=2)
       plt.xlabel('Time (s)')
       plt.ylabel('Total Control Effort (N⋅m)')
       plt.title('Control Effort')
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return {
           'reference': ref_trajectory,
           'actual_positions': actual_positions,
           'actual_velocities': actual_velocities,
           'control_torques': control_torques,
           'tracking_errors': tracking_errors
       }
   
   # Run real-time simulation
   execution_results = real_time_trajectory_execution()

Practical Applications
-------------------------

Pick and Place Operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete pick-and-place trajectory planning:

.. code-block:: python

   def pick_and_place_trajectory():
       """Generate trajectory for pick-and-place operation."""
       
       # Define task waypoints
       home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
       
       # Approach position (above object)
       approach_pos = np.array([0.3, 0.2, 0.4])
       approach_joints = planner.serial_manipulator.iterative_inverse_kinematics(
           np.array([[1, 0, 0, approach_pos[0]],
                     [0, 1, 0, approach_pos[1]],
                     [0, 0, 1, approach_pos[2]],
                     [0, 0, 0, 1]]),
           home_joints
       )[0]
       
       # Pick position (at object)
       pick_pos = approach_pos - np.array([0, 0, 0.1])
       pick_joints = planner.serial_manipulator.iterative_inverse_kinematics(
           np.array([[1, 0, 0, pick_pos[0]],
                     [0, 1, 0, pick_pos[1]],
                     [0, 0, 1, pick_pos[2]],
                     [0, 0, 0, 1]]),
           approach_joints
       )[0]
       
       # Place position
       place_pos = np.array([0.5, -0.1, 0.3])
       place_joints = planner.serial_manipulator.iterative_inverse_kinematics(
           np.array([[1, 0, 0, place_pos[0]],
                     [0, 1, 0, place_pos[1]],
                     [0, 0, 1, place_pos[2]],
                     [0, 0, 0, 1]]),
           pick_joints
       )[0]
       
       # Define trajectory segments
       segments = [
           ("Move to approach", home_joints, approach_joints, 2.0),
           ("Approach object", approach_joints, pick_joints, 1.0),
           ("Pick up", pick_joints, approach_joints, 1.0),  # Lift
           ("Move to place", approach_joints, place_joints, 3.0),
           ("Place object", place_joints, pick_joints, 1.0),  # Lower
           ("Return home", pick_joints, home_joints, 2.0)
       ]
       
       # Generate complete trajectory
       complete_trajectory = {
           'positions': [],
           'velocities': [],
           'accelerations': [],
           'segments': []
       }
       
       print("Generating pick-and-place trajectory:")
       
       for i, (name, start, end, duration) in enumerate(segments):
           print(f"  {i+1}. {name} ({duration}s)")
           
           # Generate segment
           segment = planner.joint_trajectory(
               start, end, Tf=duration, N=int(duration*50), method=5  # 50 Hz
           )
           
           # Add to complete trajectory
           if i == 0:
               complete_trajectory['positions'].extend(segment['positions'])
               complete_trajectory['velocities'].extend(segment['velocities'])
               complete_trajectory['accelerations'].extend(segment['accelerations'])
           else:
               # Skip first point to avoid duplication
               complete_trajectory['positions'].extend(segment['positions'][1:])
               complete_trajectory['velocities'].extend(segment['velocities'][1:])
               complete_trajectory['accelerations'].extend(segment['accelerations'][1:])
           
           complete_trajectory['segments'].append({
               'name': name,
               'start_index': len(complete_trajectory['positions']) - len(segment['positions']),
               'end_index': len(complete_trajectory['positions']) - 1,
               'duration': duration
           })
       
       # Convert to arrays
       for key in ['positions', 'velocities', 'accelerations']:
           complete_trajectory[key] = np.array(complete_trajectory[key])
       
       total_duration = sum(seg[3] for seg in segments)
       total_points = complete_trajectory['positions'].shape[0]
       
       print(f"\nTrajectory generated:")
       print(f"- Total duration: {total_duration} seconds")
       print(f"- Total points: {total_points}")
       
       # Compute dynamics for entire trajectory
       torques = planner.inverse_dynamics_trajectory(
           complete_trajectory['positions'],
           complete_trajectory['velocities'],
           complete_trajectory['accelerations']
       )
       
       # Visualize complete operation
       time_steps = np.linspace(0, total_duration, total_points)
       
       plt.figure(figsize=(15, 10))
       
       # Joint trajectories with segment markers
       plt.subplot(2, 2, 1)
       for i in range(6):
           plt.plot(time_steps, np.degrees(complete_trajectory['positions'][:, i]), 
                   label=f'Joint {i+1}')
       
       # Mark segment boundaries
       current_time = 0
       for segment in segments:
           current_time += segment[3]
           plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.5)
       
       plt.xlabel('Time (s)')
       plt.ylabel('Position (degrees)')
       plt.title('Pick-and-Place Joint Trajectories')
       plt.legend()
       plt.grid(True)
       
       # End-effector path
       ee_positions = []
       for pos in complete_trajectory['positions']:
           T = planner.serial_manipulator.forward_kinematics(pos)
           ee_positions.append(T[:3, 3])
       ee_positions = np.array(ee_positions)
       
       ax = plt.subplot(2, 2, 2, projection='3d')
       ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
              'b-', linewidth=2, label='End-effector path')
       
       # Mark key positions
       key_positions = [approach_pos, pick_pos, place_pos]
       key_labels = ['Approach', 'Pick', 'Place']
       colors = ['green', 'red', 'blue']
       
       for pos, label, color in zip(key_positions, key_labels, colors):
           ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, label=label)
       
       ax.set_xlabel('X (m)')
       ax.set_ylabel('Y (m)')
       ax.set_zlabel('Z (m)')
       ax.set_title('End-Effector Path')
       ax.legend()
       
       # Torque requirements
       plt.subplot(2, 2, 3)
       for i in range(6):
           plt.plot(time_steps, torques[:, i], label=f'Joint {i+1}')
       
       # Mark segment boundaries
       current_time = 0
       for segment in segments:
           current_time += segment[3]
           plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.5)
       
       plt.xlabel('Time (s)')
       plt.ylabel('Torque (N⋅m)')
       plt.title('Required Torques')
       plt.legend()
       plt.grid(True)
       
       # Velocity profile
       plt.subplot(2, 2, 4)
       velocity_magnitude = np.linalg.norm(complete_trajectory['velocities'], axis=1)
       plt.plot(time_steps, velocity_magnitude, 'g-', linewidth=2)
       
       # Mark segment boundaries  
       current_time = 0
       for i, segment in enumerate(segments):
           current_time += segment[3]
           plt.axvline(x=current_time, color='red', linestyle='--', alpha=0.5)
           if i < len(segments) - 1:
               plt.text(current_time - segment[3]/2, plt.ylim()[1]*0.8, 
                       segment[0], rotation=90, ha='center', fontsize=8)
       
       plt.xlabel('Time (s)')
       plt.ylabel('Joint Velocity Magnitude (rad/s)')
       plt.title('Velocity Profile')
       plt.grid(True)
       
       plt.tight_layout()
       plt.show()
       
       return complete_trajectory, torques
   
   # Generate pick-and-place trajectory
   pick_place_traj, pick_place_torques = pick_and_place_trajectory()

Best Practices
-----------------

Trajectory Design Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def trajectory_design_guidelines():
       """Best practices for trajectory design."""
       
       guidelines = {
           "Time Scaling": {
               "description": "Choose appropriate time scaling method",
               "recommendations": [
                   "Use cubic (method=3) for smooth velocity profiles",
                   "Use quintic (method=5) for smooth acceleration profiles", 
                   "Quintic is preferred for high-speed operations",
                   "Consider jerk constraints for smooth robot motion"
               ]
           },
           
           "Duration Selection": {
               "description": "Set appropriate trajectory duration",
               "recommendations": [
                   "Longer durations reduce peak velocities and accelerations",
                   "Consider robot dynamics limits when setting duration",
                   "Balance between speed and smoothness requirements",
                   "Account for payload and operational constraints"
               ]
           },
           
           "Sampling Rate": {
               "description": "Choose appropriate number of trajectory points",
               "recommendations": [
                   "Use 50-100 Hz for typical robot control",
                   "Higher rates for high-speed or precision operations",
                   "Consider computational resources for real-time execution",
                   "Ensure sufficient resolution for smooth motion"
               ]
           },
           
           "Joint Limits": {
               "description": "Respect robot physical constraints",
               "recommendations": [
                   "Always check joint position limits",
                   "Consider velocity and acceleration limits",
                   "Include safety margins in limit checking",
                   "Use inverse kinematics to verify reachability"
               ]
           },
           
           "Dynamics Considerations": {
               "description": "Account for robot dynamics",
               "recommendations": [
                   "Verify torque requirements don't exceed limits", 
                   "Consider payload effects on dynamics",
                   "Account for gravity compensation needs",
                   "Plan for energy-efficient trajectories"
               ]
           }
       }
       
       print("Trajectory Design Best Practices")
       print("=" * 50)
       
       for category, info in guidelines.items():
           print(f"\n{category}:")
           print(f"  {info['description']}")
           for rec in info['recommendations']:
               print(f"  • {rec}")
       
       return guidelines
   
   # Display guidelines
   design_guidelines = trajectory_design_guidelines()

Error Handling and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def trajectory_debugging_tools():
       """Tools for debugging trajectory planning issues."""
       
       def validate_trajectory(trajectory):
           """Validate trajectory properties."""
           
           print("Trajectory Validation:")
           print("-" * 25)
           
           positions = trajectory['positions']
           velocities = trajectory['velocities']
           accelerations = trajectory['accelerations']
           
           # Check shapes
           assert positions.shape[0] == velocities.shape[0] == accelerations.shape[0]
           print(f"✓ Consistent trajectory length: {positions.shape[0]} points")
           
           # Check for NaN or infinite values
           if np.any(~np.isfinite(positions)):
               print("❌ Invalid positions detected")
               return False
           print("✓ All positions are finite")
           
           if np.any(~np.isfinite(velocities)):
               print("❌ Invalid velocities detected")
               return False
           print("✓ All velocities are finite")
           
           if np.any(~np.isfinite(accelerations)):
               print("❌ Invalid accelerations detected")
               return False
           print("✓ All accelerations are finite")
           
           # Check boundary conditions
           start_vel = np.linalg.norm(velocities[0])
           end_vel = np.linalg.norm(velocities[-1])
           
           if start_vel > 1e-3:
               print(f"⚠️ Non-zero start velocity: {start_vel:.6f}")
           else:
               print("✓ Zero start velocity")
           
           if end_vel > 1e-3:
               print(f"⚠️ Non-zero end velocity: {end_vel:.6f}")
           else:
               print("✓ Zero end velocity")
           
           # Check smoothness
           vel_changes = np.diff(velocities, axis=0)
           max_vel_change = np.max(np.linalg.norm(vel_changes, axis=1))
           print(f"✓ Max velocity change: {max_vel_change:.6f} rad/s")
           
           return True
       
       def check_dynamics_feasibility(trajectory, planner):
           """Check if trajectory is dynamically feasible."""
           
           print("\nDynamics Feasibility Check:")
           print("-" * 30)
           
           try:
               torques = planner.inverse_dynamics_trajectory(
                   trajectory['positions'],
                   trajectory['velocities'],
                   trajectory['accelerations']
               )
               
               # Check torque limits
               max_torques = np.max(np.abs(torques), axis=0)
               torque_limits = np.array([limit[1] for limit in planner.torque_limits])
               
               violations = max_torques > torque_limits
               
               if np.any(violations):
                   print("❌ Torque limit violations detected:")
                   for i, violation in enumerate(violations):
                       if violation:
                           print(f"   Joint {i+1}: {max_torques[i]:.1f} > {torque_limits[i]:.1f} N⋅m")
                   return False
               else:
                   print("✓ All torques within limits")
                   max_usage = np.max(max_torques / torque_limits)
                   print(f"✓ Max torque usage: {max_usage:.1%}")
                   return True
                   
           except Exception as e:
               print(f"❌ Dynamics computation failed: {e}")
               return False
       
       # Example usage
       print("Trajectory Debugging Tools")
       print("=" * 40)
       
       # Generate test trajectory
       theta_start = np.zeros(6)
       theta_end = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
       
       test_trajectory = planner.joint_trajectory(
           theta_start, theta_end, Tf=2.0, N=50, method=5
       )
       
       # Run validation
       is_valid = validate_trajectory(test_trajectory)
       is_feasible = check_dynamics_feasibility(test_trajectory, planner)
       
       overall_status = "✓ PASSED" if (is_valid and is_feasible) else "❌ FAILED"
       print(f"\nOverall Status: {overall_status}")
       
       return is_valid and is_feasible
   
   # Run debugging tools
   debug_result = trajectory_debugging_tools()

Summary
-----------

The ManipulaPy Trajectory Planning module provides comprehensive trajectory generation capabilities for robotic manipulators:

**Core Features:**
- **Joint-space trajectories** with cubic/quintic time scaling
- **Cartesian-space trajectories** for end-effector motion
- **CUDA acceleration** for high-performance computation
- **Dynamics integration** for torque analysis and simulation
- **Collision avoidance** using potential field methods

**Key Classes and Methods:**
- ``TrajectoryPlanning``: Main class for trajectory generation
- ``joint_trajectory()``: Generate smooth joint-space paths
- ``cartesian_trajectory()``: Create end-effector trajectories  
- ``inverse_dynamics_trajectory()``: Compute required torques
- ``forward_dynamics_trajectory()``: Simulate robot motion

**Advanced Capabilities:**
- Multi-waypoint trajectory generation
- Real-time trajectory execution simulation
- Batch processing for multiple trajectories
- Pick-and-place operation planning
- Performance optimization with CUDA

**Best Practices:**
- Use quintic scaling for smooth acceleration profiles
- Validate trajectories for dynamics feasibility
- Check joint and torque limit compliance
- Consider collision avoidance requirements
- Optimize for computational performance

The trajectory planning module enables users to generate smooth, dynamically feasible robot motions for a wide range of applications from simple point-to-point movements to complex multi-segment operations.
