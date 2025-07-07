.. _api-path-planning:

===============================
Path Planning API Reference
===============================

This page documents **ManipulaPy.path_planning**, the module for trajectory generation with CUDA acceleration and collision avoidance.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Trajectory_Planning`.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

------------
TrajectoryPlanning Class
------------

.. currentmodule:: ManipulaPy.path_planning

.. autoclass:: TrajectoryPlanning
   :members:
   :show-inheritance:

   High-performance trajectory planning with CUDA acceleration, collision detection, and potential field navigation.

   .. rubric:: Constructor

   .. automethod:: __init__

      **Parameters:**
        - **serial_manipulator** (*SerialManipulator*) -- Robot kinematics object
        - **urdf_path** (*str*) -- Path to URDF file for collision checking
        - **dynamics** (*ManipulatorDynamics*) -- Robot dynamics object
        - **joint_limits** (*list*) -- Joint limits as [(min, max), ...] tuples
        - **torque_limits** (*list*, optional) -- Torque limits as [(min, max), ...] tuples

      **Attributes Created:**
        - **collision_checker** (*CollisionChecker*) -- URDF-based collision detection
        - **potential_field** (*PotentialField*) -- Artificial potential field for obstacle avoidance

   .. rubric:: Joint Space Trajectory Generation

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

      **CUDA Acceleration:** Uses GPU kernels for high-performance trajectory computation.

      **Collision Avoidance:** Automatically adjusts waypoints using potential field gradients.

      **Time Scaling Methods:**
        - **method=3:** Cubic (3rd order) polynomial with zero boundary velocities
        - **method=5:** Quintic (5th order) polynomial with zero boundary velocities and accelerations

   .. rubric:: Cartesian Space Trajectory Generation

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

      **SE(3) Interpolation:** Uses matrix logarithms and exponentials for smooth orientation changes.

      **CUDA Acceleration:** GPU-accelerated computation for position derivatives.

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

      **Formula:** τ = M(θ)θ̈ + C(θ,θ̇) + G(θ) + J^T F_ext

      **CUDA Acceleration:** Parallel computation across all trajectory points.

      **Torque Limiting:** Automatic clipping to specified torque limits.

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

      **Integration:** Multi-step numerical integration with joint limit enforcement.

      **CUDA Acceleration:** Parallel simulation across trajectory points.

   .. rubric:: Visualization Methods

   .. automethod:: plot_trajectory
      :staticmethod:

      **Parameters:**
        - **trajectory_data** (*dict*) -- Trajectory data with positions, velocities, accelerations
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "Joint Trajectory")
        - **labels** (*list*, optional) -- Joint labels for legend

      **Features:**
        - Multi-subplot layout showing position, velocity, and acceleration
        - Individual plots for each joint
        - Time-synchronized x-axes
        - Customizable labels and titles

   .. automethod:: plot_tcp_trajectory

      **Parameters:**
        - **trajectory** (*list*) -- List of joint angle configurations
        - **dt** (*float*) -- Time step between trajectory points

      **Features:**
        - Tool Center Point (TCP) position tracking
        - Velocity, acceleration, and jerk analysis
        - 4-subplot layout for comprehensive motion analysis

   .. automethod:: plot_cartesian_trajectory

      **Parameters:**
        - **trajectory_data** (*dict*) -- Cartesian trajectory data
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "Cartesian Trajectory")

      **Features:**
        - X, Y, Z position, velocity, and acceleration plots
        - Time-synchronized visualization
        - Color-coded coordinate axes

   .. automethod:: plot_ee_trajectory

      **Parameters:**
        - **trajectory_data** (*dict*) -- End-effector trajectory data
        - **Tf** (*float*) -- Total trajectory time
        - **title** (*str*, optional) -- Plot title (default: "End-Effector Trajectory")

      **Features:**
        - 3D trajectory visualization
        - Orientation vectors at sampled points
        - RGB color coding for orientation axes (R-G-B = X-Y-Z)

   .. rubric:: Utility Methods

   .. automethod:: calculate_derivatives

      **Parameters:**
        - **positions** (*array_like*) -- Position trajectory
        - **dt** (*float*) -- Time step between positions

      **Returns:**
        - **velocity** (*numpy.ndarray*) -- Finite difference velocities
        - **acceleration** (*numpy.ndarray*) -- Finite difference accelerations
        - **jerk** (*numpy.ndarray*) -- Finite difference jerk

      **Method:** Central finite differences for smooth derivative estimation.

   .. automethod:: plan_trajectory

      **Parameters:**
        - **start_position** (*list*) -- Initial joint configuration
        - **target_position** (*list*) -- Desired joint configuration
        - **obstacle_points** (*list*) -- Environment obstacle points

      **Returns:**
        - **trajectory** (*list*) -- Planned joint trajectory waypoints

      **Note:** Placeholder for advanced planning algorithms (RRT, RRT*, PRM).

-------------
Usage Examples
-------------

**Basic Joint Trajectory**::

   from ManipulaPy.path_planning import TrajectoryPlanning
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   
   # Setup
   processor = URDFToSerialManipulator("robot.urdf")
   planner = TrajectoryPlanning(
       processor.serial_manipulator,
       "robot.urdf",
       processor.dynamics,
       joint_limits=[(-np.pi, np.pi)] * 6
   )
   
   # Generate trajectory
   trajectory = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[0.5, -0.3, 0.8, 0.1, -0.2, 0.4],
       Tf=3.0,
       N=100,
       method=5  # Quintic time scaling
   )
   
   # Visualize
   planner.plot_trajectory(trajectory, Tf=3.0, title="Robot Motion")

**Cartesian Space Planning**::

   import numpy as np
   
   # Define start and end poses
   T_start = np.eye(4)
   T_start[:3, 3] = [0.5, 0.0, 0.3]  # Position
   
   T_end = np.eye(4)
   T_end[:3, 3] = [0.3, 0.2, 0.4]    # Target position
   T_end[:3, :3] = rotation_matrix_z(np.pi/4)  # 45° rotation
   
   # Generate Cartesian trajectory
   cart_traj = planner.cartesian_trajectory(
       Xstart=T_start,
       Xend=T_end,
       Tf=2.0,
       N=50,
       method=3  # Cubic time scaling
   )
   
   # Visualize in 3D
   planner.plot_ee_trajectory(cart_traj, Tf=2.0)

**Dynamics-Aware Planning**::

   # Generate joint trajectory
   joint_traj = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[0.8, -0.5, 0.6, 0.2, -0.3, 0.1],
       Tf=4.0,
       N=200,
       method=5
   )
   
   # Compute required torques
   torques = planner.inverse_dynamics_trajectory(
       joint_traj["positions"],
       joint_traj["velocities"], 
       joint_traj["accelerations"],
       gravity_vector=[0, 0, -9.81],
       Ftip=[0, 0, -5, 0, 0, 0]  # 5N downward force
   )
   
   # Visualize torque requirements
   plt.figure()
   for i in range(torques.shape[1]):
       plt.plot(torques[:, i], label=f'Joint {i+1}')
   plt.xlabel('Time Step')
   plt.ylabel('Torque (Nm)')
   plt.legend()
   plt.title('Required Joint Torques')
   plt.show()

**Forward Dynamics Simulation**::

   # Initial conditions
   theta0 = [0.1, 0.2, 0.1, 0, 0, 0]
   dtheta0 = [0, 0, 0, 0, 0, 0]
   
   # Torque profile (constant torques)
   n_steps = 100
   torque_profile = np.ones((n_steps, 6)) * [1.0, 0.5, 0.2, 0.1, 0.1, 0.05]
   
   # External forces (none)
   Ftip_profile = np.zeros((n_steps, 6))
   
   # Simulate
   simulation = planner.forward_dynamics_trajectory(
       thetalist=theta0,
       dthetalist=dtheta0,
       taumat=torque_profile,
       g=[0, 0, -9.81],
       Ftipmat=Ftip_profile,
       dt=0.01,
       intRes=5
   )
   
   # Plot results
   planner.plot_trajectory(simulation, Tf=1.0, title="Forward Dynamics Simulation")

**TCP Trajectory Analysis**::

   # Convert joint trajectory to TCP motion
   joint_configs = joint_traj["positions"]
   
   # Analyze TCP motion
   planner.plot_tcp_trajectory(joint_configs, dt=0.02)
   
   # Manual TCP position extraction
   tcp_positions = []
   for config in joint_configs:
       T = planner.serial_manipulator.forward_kinematics(config)
       tcp_positions.append(T[:3, 3])
   
   # Calculate motion derivatives
   vel, acc, jerk = planner.calculate_derivatives(tcp_positions, dt=0.02)
   
   # Check motion smoothness
   max_jerk = np.max(np.abs(jerk))
   print(f"Maximum jerk: {max_jerk:.3f} m/s³")

**Collision-Aware Planning**::

   # Define obstacles in workspace
   obstacles = [
       [0.4, 0.1, 0.3],  # Obstacle position 1
       [0.2, 0.3, 0.2],  # Obstacle position 2
   ]
   
   # Plan collision-free trajectory
   safe_trajectory = planner.plan_trajectory(
       start_position=[0, 0, 0, 0, 0, 0],
       target_position=[0.8, -0.4, 0.6, 0.1, -0.2, 0.3],
       obstacle_points=obstacles
   )
   
   # The joint_trajectory method automatically applies potential field
   # for collision avoidance during trajectory generation

**Advanced Time Scaling Comparison**::

   # Compare cubic vs quintic time scaling
   traj_cubic = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[1.0, -0.5, 0.8, 0, 0, 0],
       Tf=2.0, N=100, method=3
   )
   
   traj_quintic = planner.joint_trajectory(
       thetastart=[0, 0, 0, 0, 0, 0],
       thetaend=[1.0, -0.5, 0.8, 0, 0, 0],
       Tf=2.0, N=100, method=5
   )
   
   # Plot comparison
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
   
   time = np.linspace(0, 2.0, 100)
   ax1.plot(time, traj_cubic["velocities"][:, 0], label="Cubic")
   ax1.plot(time, traj_quintic["velocities"][:, 0], label="Quintic")
   ax1.set_ylabel("Velocity (rad/s)")
   ax1.legend()
   ax1.set_title("Joint 1 Velocity Comparison")
   
   ax2.plot(time, traj_cubic["accelerations"][:, 0], label="Cubic")
   ax2.plot(time, traj_quintic["accelerations"][:, 0], label="Quintic")
   ax2.set_ylabel("Acceleration (rad/s²)")
   ax2.set_xlabel("Time (s)")
   ax2.legend()
   ax2.set_title("Joint 1 Acceleration Comparison")
   
   plt.tight_layout()
   plt.show()

-------------
Key Features
-------------

- **CUDA Acceleration** for high-performance trajectory computation
- **Dual-space planning** supporting both joint and Cartesian trajectories
- **Collision avoidance** using potential fields and convex hull checking
- **Dynamics integration** with inverse and forward dynamics support
- **Multiple time scaling** methods (cubic and quintic polynomials)
- **Comprehensive visualization** tools for trajectory analysis
- **Automatic limit enforcement** for joints and torques
- **SE(3) interpolation** using matrix logarithms for smooth orientation changes

-----------------
Mathematical Foundation
-----------------

**Time Scaling Functions:**
  - Cubic: s(t) = 3(t/T)² - 2(t/T)³
  - Quintic: s(t) = 10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵

**SE(3) Interpolation:**
  - Rotation: R(s) = R₀ exp(log(R₀ᵀR₁) · s)
  - Position: p(s) = (1-s)p₀ + s·p₁

**Potential Fields:**
  - Attractive: U_att = ½k_att ||q - q_goal||²
  - Repulsive: U_rep = ½k_rep (1/d - 1/d₀)² if d < d₀

**Dynamics Equations:**
  - Inverse: τ = M(θ)θ̈ + C(θ,θ̇) + G(θ) + J^T F_ext
  - Forward: θ̈ = M(θ)⁻¹[τ - C(θ,θ̇) - G(θ) - J^T F_ext]

-----------------
Performance Considerations
-----------------

**CUDA Optimization:**
  - Automatic memory management with explicit cleanup
  - Optimal thread block sizing (256-1024 threads)
  - Coalesced memory access patterns

**Trajectory Resolution:**
  - Higher N values provide smoother trajectories
  - Balance between smoothness and computation time
  - Typical range: N = 50-500 for most applications

**Collision Checking:**
  - Convex hull approximation for efficiency
  - Potential field gradient descent for trajectory adjustment
  - Configurable iteration limits and step sizes

-----------------
See Also
-----------------

- :doc:`kinematics` -- Forward and inverse kinematics for trajectory execution
- :doc:`dynamics` -- Dynamics computations for torque calculation
- :doc:`control` -- Controllers for trajectory following
- :doc:`potential_field` -- Collision avoidance and path optimization
- :doc:`../user_guide/Trajectory_Planning` -- Conceptual overview and planning strategies