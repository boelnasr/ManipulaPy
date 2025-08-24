Dynamics User Guide
===================

This comprehensive guide covers robot dynamics computations in ManipulaPy, optimized for Python 3.10.12.

.. note::
   This guide is written for Python 3.10.12 users and includes version-specific optimizations and performance improvements.

Introduction to Robot Dynamics
----------------------------------

Robot dynamics deals with the relationship between forces/torques and motion in robotic systems. Unlike kinematics, which only considers geometric relationships, dynamics incorporates:

- **Mass properties** of robot links
- **Inertial forces** due to acceleration
- **Gravitational forces** acting on the robot
- **External forces** applied to the robot
- **Joint torques** required for desired motion

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core equation of motion for an n-DOF serial manipulator is the **Newton–Euler** (or **Lagrange**) form:

.. math::
   \boldsymbol\tau
     = M(\boldsymbol\theta)\,\ddot{\boldsymbol\theta}
       + C(\boldsymbol\theta,\dot{\boldsymbol\theta})\,\dot{\boldsymbol\theta}
       + G(\boldsymbol\theta)
       + J(\boldsymbol\theta)^{T}\,\mathbf F_{\mathrm{ext}}

where:

- :math:`\boldsymbol\tau\in\mathbb R^{n}` is the vector of joint torques  
- :math:`\boldsymbol\theta,\dot{\boldsymbol\theta},\ddot{\boldsymbol\theta}\in\mathbb R^{n}` are joint positions, velocities, and accelerations  
- :math:`\mathbf F_{\mathrm{ext}}\in\mathbb R^{6}` is the spatial external wrench (force/torque) at the end‐effector  

**Mass Matrix**

The symmetric, positive‐definite inertia matrix

.. math::
   M(\boldsymbol\theta)
     = \sum_{i=1}^{n} \bigl(\mathrm{Ad}_{T_{0}^{\,i-1}}^{T}\bigr)\;G_{i}\;\bigl(\mathrm{Ad}_{T_{0}^{\,i-1}}\bigr)

where for each link *i*,  
:math:`G_{i}` is its 6×6 spatial inertia, and  
:math:`\mathrm{Ad}_{T}` denotes the SE(3) adjoint of the transform from the base to link *i*.

**Coriolis & Centrifugal**

Combined velocity‐dependent forces:

.. math::

   C(\boldsymbol{\theta}, \dot{\boldsymbol{\theta}}) \, \dot{\boldsymbol{\theta}} =
   \begin{bmatrix}
       \sum\limits_{j,k=1}^{n} \Gamma_{1jk}(\boldsymbol{\theta}) \, \dot{\theta}_{j} \, \dot{\theta}_{k} \\[6pt]
       \vdots \\[3pt]
       \sum\limits_{j,k=1}^{n} \Gamma_{njk}(\boldsymbol{\theta}) \, \dot{\theta}_{j} \, \dot{\theta}_{k}
   \end{bmatrix},
   \quad
   \Gamma_{ijk} =
   \frac{1}{2} \left(
       \frac{\partial M_{ij}}{\partial \theta_k} +
       \frac{\partial M_{ik}}{\partial \theta_j} -
       \frac{\partial M_{jk}}{\partial \theta_i}
   \right)



**Gravity**

Derived from the potential energy

.. math::
   U(\boldsymbol\theta)
     = \sum_{i=1}^{n} m_{i}\;g^{T}\,p_{i}(\boldsymbol\theta),

the gravity torque vector is

.. math::
   G(\boldsymbol\theta)
     = \frac{\partial U}{\partial\boldsymbol\theta}
     = \begin{bmatrix}
         \tfrac{\partial U}{\partial\theta_{1}}\\[3pt]
         \vdots\\[3pt]
         \tfrac{\partial U}{\partial\theta_{n}}
       \end{bmatrix}.

Here, :math:`p_{i}(\boldsymbol\theta)` is the world‐frame position of link *i*'s center of mass.

**External Wrench Mapping**

An end‐effector wrench :math:`\mathbf F_{\mathrm{ext}}\in\mathbb R^{6}`  
is pulled back to joint torques via the Jacobian transpose:

.. math::
   \tau_{\mathrm{ext}}
     = J(\boldsymbol\theta)^{T}\,\mathbf F_{\mathrm{ext}}.

Putting it all together:

.. math::
   \boxed{
     \boldsymbol\tau
       = M(\boldsymbol\theta)\,\ddot{\boldsymbol\theta}
         + C(\boldsymbol\theta,\dot{\boldsymbol\theta})\,\dot{\boldsymbol\theta}
         + G(\boldsymbol\theta)
         + J(\boldsymbol\theta)^{T}\,\mathbf F_{\mathrm{ext}}
   }

This formulation underlies both:

- **Inverse Dynamics:** compute :math:`\boldsymbol\tau` from given :math:`(\boldsymbol\theta,\dot{\boldsymbol\theta},\ddot{\boldsymbol\theta})`  
- **Forward Dynamics:** compute :math:`\ddot{\boldsymbol\theta}` via :math:`\ddot{\boldsymbol\theta} = M(\theta)^{-1}\bigl(\boldsymbol\tau - C\,\dot{\boldsymbol\theta} - G - J^{T}\mathbf F_{\mathrm{ext}}\bigr)`

Key Concepts
~~~~~~~~~~~~~~~~

**Forward Dynamics**
   Given joint torques, compute joint accelerations: :math:`\ddot{\boldsymbol\theta} = f(\boldsymbol\tau, \boldsymbol\theta, \dot{\boldsymbol\theta})`

**Inverse Dynamics**
   Given desired motion, compute required torques: :math:`\boldsymbol\tau = f(\boldsymbol\theta, \dot{\boldsymbol\theta}, \ddot{\boldsymbol\theta})`

**Mass Matrix**
   Represents the robot's inertial properties and coupling between joints

**Velocity-Dependent Forces**
   Coriolis and centrifugal forces that arise from robot motion

Setting Up Robot Dynamics
--------------------------

Basic Setup from URDF
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.dynamics import ManipulatorDynamics

   # Load robot from URDF (automatically extracts inertial properties)
   urdf_processor = URDFToSerialManipulator("robot.urdf")
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics

   print(f"Robot has {len(dynamics.Glist)} links with inertial properties")

Manual Setup
~~~~~~~~~~~~~~~

For custom robots or when URDF is not available:

.. code-block:: python

   from ManipulaPy.dynamics import ManipulatorDynamics
   import numpy as np

   # Define robot parameters
   M_list = np.eye(4)  # Home configuration
   M_list[:3, 3] = [0.5, 0, 0.3]  # End-effector position

   # Screw axes in space frame
   S_list = np.array([
       [0, 0, 1, 0, 0, 0],      # Joint 1: rotation about z-axis
       [0, -1, 0, -0.1, 0, 0],  # Joint 2: rotation about -y-axis
       [0, -1, 0, -0.1, 0, 0.3], # Joint 3: rotation about -y-axis
   ]).T

   # Inertial properties for each link (6x6 spatial inertia matrices)
   Glist = []
   for i in range(3):  # 3 links
       G = np.zeros((6, 6))
       
       # Rotational inertia (upper-left 3x3)
       G[:3, :3] = np.diag([0.1, 0.1, 0.05])  # Ixx, Iyy, Izz
       
       # Mass (lower-right 3x3)
       mass = 2.0 - i * 0.5  # Decreasing mass towards end-effector
       G[3:, 3:] = mass * np.eye(3)
       
       Glist.append(G)

   # Create dynamics object
   dynamics = ManipulatorDynamics(
       M_list=M_list,
       omega_list=S_list[:3, :],  # Rotation axes
       r_list=None,  # Will be computed from S_list
       b_list=None,  # Body frame (optional)
       S_list=S_list,
       B_list=None,  # Will be computed
       Glist=Glist
   )

Mass Matrix Computation
---------------------------

The mass matrix represents the robot's inertial properties and varies with configuration.

Computing Mass Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define joint configuration
   theta = np.array([0.1, 0.3, -0.2])  # Joint angles in radians

   # Compute mass matrix
   M = dynamics.mass_matrix(theta)

   print(f"Mass matrix shape: {M.shape}")
   print(f"Mass matrix:\n{M}")

   # Check properties
   print(f"Matrix is symmetric: {np.allclose(M, M.T)}")
   print(f"Matrix is positive definite: {np.all(np.linalg.eigvals(M) > 0)}")

Configuration Dependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mass matrix changes with robot configuration:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Test different configurations
   configurations = np.linspace(-np.pi, np.pi, 50)
   condition_numbers = []
   determinants = []

   for angle in configurations:
       theta = np.array([angle, 0.0, 0.0])
       M = dynamics.mass_matrix(theta)
       
       condition_numbers.append(np.linalg.cond(M))
       determinants.append(np.linalg.det(M))

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   ax1.plot(configurations, condition_numbers)
   ax1.set_xlabel('Joint 1 Angle (rad)')
   ax1.set_ylabel('Condition Number')
   ax1.set_title('Mass Matrix Conditioning')
   ax1.grid(True)

   ax2.plot(configurations, determinants)
   ax2.set_xlabel('Joint 1 Angle (rad)')
   ax2.set_ylabel('Determinant')
   ax2.set_title('Mass Matrix Determinant')
   ax2.grid(True)

   plt.tight_layout()
   plt.show()

Caching for Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~

For real-time applications, cache mass matrix computations:

.. code-block:: python

   class CachedDynamics:
       def __init__(self, dynamics, tolerance=1e-3):
           self.dynamics = dynamics
           self.tolerance = tolerance
           self.cache = {}
       
       def mass_matrix_cached(self, theta):
           # Create cache key (rounded configuration)
           key = tuple(np.round(theta / self.tolerance) * self.tolerance)
           
           if key not in self.cache:
               self.cache[key] = self.dynamics.mass_matrix(theta)
           
           return self.cache[key]
       
       def clear_cache(self):
           self.cache.clear()

   # Usage
   cached_dynamics = CachedDynamics(dynamics)
   M = cached_dynamics.mass_matrix_cached(theta)

Velocity-Dependent Forces
----------------------------

Coriolis and centrifugal forces arise from robot motion and joint coupling.

Computing Velocity Forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define joint state
   theta = np.array([0.1, 0.3, -0.2])      # Joint positions
   theta_dot = np.array([0.5, -0.3, 0.8])  # Joint velocities

   # Compute velocity-dependent forces
   c = dynamics.velocity_quadratic_forces(theta, theta_dot)

   print(f"Velocity forces: {c}")
   print(f"Force magnitude: {np.linalg.norm(c)}")

Analyzing Velocity Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_velocity_effects(dynamics, theta, max_velocity=2.0):
       """Analyze how joint velocities affect Coriolis forces."""
       
       velocities = np.linspace(0, max_velocity, 20)
       force_magnitudes = []
       
       for vel in velocities:
           # Apply same velocity to all joints
           theta_dot = np.ones(len(theta)) * vel
           c = dynamics.velocity_quadratic_forces(theta, theta_dot)
           force_magnitudes.append(np.linalg.norm(c))
       
       # Plot results
       plt.figure(figsize=(8, 6))
       plt.plot(velocities, force_magnitudes, 'b-', linewidth=2)
       plt.xlabel('Joint Velocity (rad/s)')
       plt.ylabel('Coriolis Force Magnitude (N⋅m)')
       plt.title('Velocity-Dependent Forces')
       plt.grid(True)
       plt.show()
       
       return velocities, force_magnitudes

   # Analyze for current configuration
   analyze_velocity_effects(dynamics, theta)

Centrifugal vs Coriolis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate centrifugal (velocity²) and Coriolis (cross-coupling) effects:

.. code-block:: python

   def decompose_velocity_forces(dynamics, theta, theta_dot):
       """Decompose velocity forces into centrifugal and Coriolis components."""
       
       n = len(theta)
       centrifugal = np.zeros(n)
<<<<<<< HEAD
       coriolis = np.zeros(n)
       
       # Centrifugal forces (diagonal terms: i=j=k)
       for i in range(n):
           for j in range(n):
               if i == j:
                   christoffel = dynamics.partial_derivative(i, j, j, theta)
                   centrifugal[i] += christoffel * theta_dot[j] * theta_dot[j]
       
       # Coriolis forces (off-diagonal coupling: i≠j or j≠k)
       for i in range(n):
           for j in range(n):
               for k in range(n):
                   if not (i == j == k):
                       christoffel = dynamics.partial_derivative(i, j, k, theta)
                       coriolis[i] += christoffel * theta_dot[j] * theta_dot[k]
       
       return centrifugal, coriolis

   # Example usage
   theta = np.array([0.1, 0.3, -0.2])
   theta_dot = np.array([1.0, -0.5, 0.8])

   centrifugal, coriolis = decompose_velocity_forces(dynamics, theta, theta_dot)
   total_c = dynamics.velocity_quadratic_forces(theta, theta_dot)

   print(f"Centrifugal forces: {centrifugal}")
   print(f"Coriolis forces: {coriolis}")
   print(f"Total forces: {total_c}")
   print(f"Sum check: {np.allclose(centrifugal + coriolis, total_c)}")

Gravity Compensation
----------------------

Gravity forces act continuously on robot links and must be compensated for precise control.

Computing Gravity Forces
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard Earth gravity
   g_earth = [0, 0, -9.81]  # m/s²
   
   # Different orientations
   g_upright = [0, 0, -9.81]      # Robot upright
   g_inverted = [0, 0, 9.81]      # Robot inverted
   g_sideways = [9.81, 0, 0]      # Robot on its side

   # Compute gravity forces for different configurations
   configurations = [
       np.array([0, 0, 0]),           # Home position
       np.array([np.pi/2, 0, 0]),     # Joint 1 at 90°
       np.array([0, np.pi/2, 0]),     # Joint 2 at 90°
       np.array([0, 0, np.pi/2]),     # Joint 3 at 90°
   ]

   for i, theta in enumerate(configurations):
       g_torques = dynamics.gravity_forces(theta, g_earth)
       print(f"Config {i+1}: θ={theta}, G(θ)={g_torques}")

Gravity in Different Orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_gravity_effects(dynamics, gravity_vectors):
       """Analyze gravity effects for different robot orientations."""
       
       theta = np.array([np.pi/4, np.pi/6, -np.pi/3])  # Test configuration
       
       fig, axes = plt.subplots(1, len(gravity_vectors), figsize=(15, 4))
       if len(gravity_vectors) == 1:
           axes = [axes]
       
       for idx, (g_vec, label) in enumerate(gravity_vectors):
           g_forces = dynamics.gravity_forces(theta, g_vec)
           
           axes[idx].bar(range(len(g_forces)), g_forces)
           axes[idx].set_title(f'Gravity: {label}')
           axes[idx].set_xlabel('Joint')
           axes[idx].set_ylabel('Torque (N⋅m)')
           axes[idx].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

   # Test different orientations
   gravity_scenarios = [
       ([0, 0, -9.81], 'Upright'),
       ([0, 0, 9.81], 'Inverted'),
       ([9.81, 0, 0], 'On Side X'),
       ([0, 9.81, 0], 'On Side Y'),
   ]

   analyze_gravity_effects(dynamics, gravity_scenarios)

Configuration-Dependent Gravity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_gravity_variation(dynamics, joint_idx=0, g=[0, 0, -9.81]):
       """Plot how gravity torque varies with one joint angle."""
       
       # Vary one joint while keeping others fixed
       angles = np.linspace(-np.pi, np.pi, 100)
       base_config = np.zeros(len(dynamics.joint_limits))
       gravity_torques = []
       
       for angle in angles:
           theta = base_config.copy()
           theta[joint_idx] = angle
           
           g_forces = dynamics.gravity_forces(theta, g)
           gravity_torques.append(g_forces)
       
       gravity_torques = np.array(gravity_torques)
       
       # Plot all joints
       plt.figure(figsize=(10, 6))
       for j in range(gravity_torques.shape[1]):
           plt.plot(angles, gravity_torques[:, j], 
                   label=f'Joint {j+1}', linewidth=2)
       
       plt.xlabel(f'Joint {joint_idx+1} Angle (rad)')
       plt.ylabel('Gravity Torque (N⋅m)')
       plt.title('Gravity Torque vs Configuration')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()

   # Analyze gravity variation
   plot_gravity_variation(dynamics, joint_idx=1)

Inverse Dynamics
------------------

Inverse dynamics computes the joint torques required to achieve desired motion.

Basic Inverse Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define desired motion
   theta = np.array([0.1, 0.3, -0.2])           # Joint positions
   theta_dot = np.array([0.5, -0.3, 0.8])       # Joint velocities  
   theta_ddot = np.array([1.0, 0.5, -1.2])      # Joint accelerations

   # External conditions
   g = [0, 0, -9.81]                             # Gravity vector
   Ftip = [10, 0, 0, 0, 0, 0]                    # External wrench at end-effector

   # Compute required torques
   tau = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, g, Ftip)

   print(f"Joint positions: {theta}")
   print(f"Joint velocities: {theta_dot}")
   print(f"Joint accelerations: {theta_ddot}")
   print(f"Required torques: {tau}")

Trajectory Inverse Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_trajectory_torques(dynamics, trajectory_data, g=[0, 0, -9.81]):
       """Compute torques for an entire trajectory."""
       
       positions = trajectory_data['positions']
       velocities = trajectory_data['velocities'] 
       accelerations = trajectory_data['accelerations']
       
       num_points = positions.shape[0]
       num_joints = positions.shape[1]
       
       torques = np.zeros((num_points, num_joints))
       
       # Compute torques for each trajectory point
       for i in range(num_points):
           Ftip = [0, 0, 0, 0, 0, 0]  # No external forces
           
           tau = dynamics.inverse_dynamics(
               positions[i], velocities[i], accelerations[i], g, Ftip
           )
           torques[i] = tau
       
       return torques

   # Example with trajectory from path planning
   from ManipulaPy.path_planning import OptimizedTrajectoryPlanning

   # Create trajectory planner
   joint_limits = [(-np.pi, np.pi)] * len(dynamics.joint_limits)
   planner = OptimizedTrajectoryPlanning(
       robot, "robot.urdf", dynamics, joint_limits
   )

   # Generate trajectory
   start_config = np.zeros(3)
   end_config = np.array([np.pi/2, np.pi/4, -np.pi/6])
   
   trajectory = planner.joint_trajectory(
       start_config, end_config, Tf=2.0, N=100, method=5
   )

   # Compute required torques
   torques = compute_trajectory_torques(dynamics, trajectory)

   # Plot torques over time
   time_vector = np.linspace(0, 2.0, 100)
   
   plt.figure(figsize=(12, 8))
   for j in range(torques.shape[1]):
       plt.subplot(3, 1, j+1)
       plt.plot(time_vector, torques[:, j], linewidth=2)
       plt.ylabel(f'Joint {j+1} Torque (N⋅m)')
       plt.grid(True, alpha=0.3)
   
   plt.xlabel('Time (s)')
   plt.suptitle('Joint Torques for Trajectory')
   plt.tight_layout()
   plt.show()

Analyzing Torque Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_torque_components(dynamics, theta, theta_dot, theta_ddot, g, Ftip):
       """Break down inverse dynamics into individual components."""
       
       # Compute each component separately
       M = dynamics.mass_matrix(theta)
       inertial_torques = M @ theta_ddot
       
       coriolis_torques = dynamics.velocity_quadratic_forces(theta, theta_dot)
       gravity_torques = dynamics.gravity_forces(theta, g)
       
       J_transpose = dynamics.jacobian(theta).T
       external_torques = J_transpose @ Ftip
       
       total_torques = inertial_torques + coriolis_torques + gravity_torques + external_torques
       
       # Verify against direct computation
       tau_direct = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, g, Ftip)
       
       print("Torque Component Analysis:")
       print("-" * 40)
       print(f"Inertial:    {inertial_torques}")
       print(f"Coriolis:    {coriolis_torques}")
       print(f"Gravity:     {gravity_torques}")
       print(f"External:    {external_torques}")
       print(f"Total:       {total_torques}")
       print(f"Direct calc: {tau_direct}")
       print(f"Difference:  {np.abs(total_torques - tau_direct)}")
       
       # Create visualization
       components = [inertial_torques, coriolis_torques, gravity_torques, external_torques]
       labels = ['Inertial', 'Coriolis', 'Gravity', 'External']
       colors = ['red', 'blue', 'green', 'orange']
       
       fig, ax = plt.subplots(figsize=(10, 6))
       
       x = np.arange(len(theta))
       width = 0.2
       
       for i, (component, label, color) in enumerate(zip(components, labels, colors)):
           ax.bar(x + i*width, component, width, label=label, color=color, alpha=0.7)
       
       ax.set_xlabel('Joint')
       ax.set_ylabel('Torque (N⋅m)')
       ax.set_title('Inverse Dynamics Components')
       ax.set_xticks(x + width * 1.5)
       ax.set_xticklabels([f'Joint {i+1}' for i in range(len(theta))])
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       return {
           'inertial': inertial_torques,
           'coriolis': coriolis_torques,
           'gravity': gravity_torques,
           'external': external_torques,
           'total': total_torques
       }

   # Example analysis
   theta = np.array([np.pi/4, np.pi/6, -np.pi/3])
   theta_dot = np.array([1.0, -0.5, 0.8])
   theta_ddot = np.array([2.0, 1.0, -1.5])
   g = [0, 0, -9.81]
   Ftip = [5, 0, -10, 0, 0, 0]

   components = analyze_torque_components(dynamics, theta, theta_dot, theta_ddot, g, Ftip)

Forward Dynamics
------------------

Forward dynamics computes joint accelerations given applied torques.

Basic Forward Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define robot state and applied torques
   theta = np.array([0.1, 0.3, -0.2])           # Joint positions
   theta_dot = np.array([0.5, -0.3, 0.8])       # Joint velocities
   tau = np.array([10, 5, -8])                  # Applied torques

   # External conditions
   g = [0, 0, -9.81]                            # Gravity
   Ftip = [0, 0, 0, 0, 0, 0]                    # No external forces

   # Compute resulting accelerations
   theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, g, Ftip)

   print(f"Applied torques: {tau}")
   print(f"Resulting accelerations: {theta_ddot}")

   # Verify with inverse dynamics
   tau_verify = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, g, Ftip)
   print(f"Verification (should match applied): {tau_verify}")
   print(f"Error: {np.abs(tau - tau_verify)}")

Simulation with Forward Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def simulate_robot_motion(dynamics, initial_state, torque_function, dt, duration, g=[0, 0, -9.81]):
       """Simulate robot motion using forward dynamics integration."""
       
       # Initialize arrays
       num_steps = int(duration / dt)
       num_joints = len(initial_state['theta'])
       
       time_history = np.zeros(num_steps)
       theta_history = np.zeros((num_steps, num_joints))
       theta_dot_history = np.zeros((num_steps, num_joints))
       theta_ddot_history = np.zeros((num_steps, num_joints))
       tau_history = np.zeros((num_steps, num_joints))
       
       # Set initial conditions
       theta = initial_state['theta'].copy()
       theta_dot = initial_state['theta_dot'].copy()
       
       # Integration loop
       for i in range(num_steps):
           t = i * dt
           time_history[i] = t
           
           # Store current state
           theta_history[i] = theta
           theta_dot_history[i] = theta_dot
           
           # Compute applied torques
           tau = torque_function(t, theta, theta_dot)
           tau_history[i] = tau
           
           # Compute accelerations
           Ftip = [0, 0, 0, 0, 0, 0]  # No external forces
           theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, g, Ftip)
           theta_ddot_history[i] = theta_ddot
           
           # Integrate using Euler method (simple)
           theta_dot += theta_ddot * dt
           theta += theta_dot * dt
       
       return {
           'time': time_history,
           'theta': theta_history,
           'theta_dot': theta_dot_history,
           'theta_ddot': theta_ddot_history,
           'tau': tau_history
       }

   # Example: PD control to target position
   def pd_torque_controller(t, theta, theta_dot):
       """Simple PD controller."""
       target_theta = np.array([np.pi/2, np.pi/4, 0])
       target_theta_dot = np.zeros(3)
       
       Kp = np.array([100, 80, 60])  # Proportional gains
       Kd = np.array([10, 8, 6])     # Derivative gains
       
       error = target_theta - theta
       error_dot = target_theta_dot - theta_dot
       
       tau = Kp * error + Kd * error_dot
       return tau

   # Run simulation
   initial_state = {
       'theta': np.array([0, 0, 0]),
       'theta_dot': np.array([0, 0, 0])
   }

   simulation_results = simulate_robot_motion(
       dynamics, initial_state, pd_torque_controller, 
       dt=0.001, duration=2.0
   )

   # Plot results
   fig, axes = plt.subplots(4, 1, figsize=(12, 12))
   
   for j in range(3):
       # Position
       axes[0].plot(simulation_results['time'], simulation_results['theta'][:, j], 
                   label=f'Joint {j+1}', linewidth=2)
       
       # Velocity
       axes[1].plot(simulation_results['time'], simulation_results['theta_dot'][:, j], 
                   label=f'Joint {j+1}', linewidth=2)
       
       # Acceleration
       axes[2].plot(simulation_results['time'], simulation_results['theta_ddot'][:, j], 
                   label=f'Joint {j+1}', linewidth=2)
       
       # Torque
       axes[3].plot(simulation_results['time'], simulation_results['tau'][:, j], 
                   label=f'Joint {j+1}', linewidth=2)

   axes[0].set_ylabel('Position (rad)')
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)
   axes[0].set_title('Robot Motion Simulation')

   axes[1].set_ylabel('Velocity (rad/s)')
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)

   axes[2].set_ylabel('Acceleration (rad/s²)')
   axes[2].legend()
   axes[2].grid(True, alpha=0.3)

   axes[3].set_ylabel('Torque (N⋅m)')
   axes[3].set_xlabel('Time (s)')
   axes[3].legend()
   axes[3].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Advanced Dynamics Analysis
-----------------------------

Energy Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_robot_energy(dynamics, theta, theta_dot, g=[0, 0, -9.81]):
       """Compute kinetic and potential energy of the robot."""
       
       # Kinetic energy: T = 0.5 * θ̇ᵀ * M(θ) * θ̇
       M = dynamics.mass_matrix(theta)
       kinetic_energy = 0.5 * theta_dot.T @ M @ theta_dot
       
       # Potential energy (approximate using gravity forces)
       # PE = ∫ G(θ) dθ (numerical integration from zero configuration)
       potential_energy = 0.0
       
       # Numerical integration of gravity forces
       n_steps = 100
       for i in range(len(theta)):
           if theta[i] != 0:
               angles = np.linspace(0, theta[i], n_steps)
               for j in range(len(angles)):
                   test_theta = np.zeros_like(theta)
                   test_theta[i] = angles[j]
                   g_forces = dynamics.gravity_forces(test_theta, g)
                   potential_energy += g_forces[i] * (theta[i] / n_steps)
       
       total_energy = kinetic_energy + potential_energy
       
       return {
           'kinetic': kinetic_energy,
           'potential': potential_energy,
           'total': total_energy
       }

   # Energy analysis over trajectory
   def analyze_trajectory_energy(dynamics, trajectory_data, g=[0, 0, -9.81]):
       """Analyze energy throughout a trajectory."""
       
       positions = trajectory_data['positions']
       velocities = trajectory_data['velocities']
       
       num_points = positions.shape[0]
       energies = {
           'kinetic': np.zeros(num_points),
           'potential': np.zeros(num_points),
           'total': np.zeros(num_points)
       }
       
       for i in range(num_points):
           energy = compute_robot_energy(dynamics, positions[i], velocities[i], g)
           energies['kinetic'][i] = energy['kinetic']
           energies['potential'][i] = energy['potential']
           energies['total'][i] = energy['total']
       
       return energies

   # Plot energy analysis
   energies = analyze_trajectory_energy(dynamics, trajectory)
   time_vector = np.linspace(0, 2.0, len(energies['total']))

   plt.figure(figsize=(10, 6))
   plt.plot(time_vector, energies['kinetic'], label='Kinetic Energy', linewidth=2)
   plt.plot(time_vector, energies['potential'], label='Potential Energy', linewidth=2)
   plt.plot(time_vector, energies['total'], label='Total Energy', linewidth=2, linestyle='--')
   
   plt.xlabel('Time (s)')
   plt.ylabel('Energy (J)')
   plt.title('Robot Energy Analysis')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Manipulability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compute_dynamic_manipulability(dynamics, theta):
       """Compute dynamic manipulability ellipsoid."""
       
       # Get mass matrix and Jacobian
       M = dynamics.mass_matrix(theta)
       J = dynamics.jacobian(theta)
       
       # Kinetic energy manipulability ellipsoid
       # Λ = (J M⁻¹ Jᵀ)⁻¹ (operational space inertia)
       try:
           M_inv = np.linalg.inv(M)
           Lambda = np.linalg.inv(J @ M_inv @ J.T)
           
           # Singular value decomposition for ellipsoid
           U, S, Vt = np.linalg.svd(Lambda)
           
           return {
               'Lambda': Lambda,
               'singular_values': S,
               'condition_number': np.max(S) / np.min(S),
               'manipulability_measure': np.sqrt(np.linalg.det(Lambda))
           }
       except np.linalg.LinAlgError:
           return None

   def plot_manipulability_variation(dynamics, joint_idx=0):
       """Plot how manipulability varies with joint configuration."""
       
       angles = np.linspace(-np.pi, np.pi, 50)
       base_config = np.zeros(len(dynamics.joint_limits))
       
       manipulability_measures = []
       condition_numbers = []
       
       for angle in angles:
           theta = base_config.copy()
           theta[joint_idx] = angle
           
           manip_data = compute_dynamic_manipulability(dynamics, theta)
           if manip_data is not None:
               manipulability_measures.append(manip_data['manipulability_measure'])
               condition_numbers.append(manip_data['condition_number'])
           else:
               manipulability_measures.append(0)
               condition_numbers.append(np.inf)
       
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
       
       ax1.plot(angles, manipulability_measures, linewidth=2)
       ax1.set_ylabel('Manipulability Measure')
       ax1.set_title('Dynamic Manipulability Analysis')
       ax1.grid(True, alpha=0.3)
       
       ax2.plot(angles, condition_numbers, linewidth=2)
       ax2.set_xlabel(f'Joint {joint_idx+1} Angle (rad)')
       ax2.set_ylabel('Condition Number')
       ax2.set_yscale('log')
       ax2.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

   # Analyze manipulability
   plot_manipulability_variation(dynamics, joint_idx=1)

Performance Optimization
--------------------------

Mass Matrix Caching
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class OptimizedDynamics:
       """Optimized dynamics computation with intelligent caching."""
       
       def __init__(self, base_dynamics, cache_size=1000, tolerance=1e-4):
           self.base_dynamics = base_dynamics
           self.cache_size = cache_size
           self.tolerance = tolerance
           
           # LRU cache for mass matrices
           from collections import OrderedDict
           self.mass_cache = OrderedDict()
           
           # Gradient cache for Christoffel symbols
           self.christoffel_cache = {}
       
       def _config_key(self, theta):
           """Create cache key from configuration."""
           return tuple(np.round(theta / self.tolerance) * self.tolerance)
       
       def mass_matrix_cached(self, theta):
           """Cached mass matrix computation."""
           key = self._config_key(theta)
           
           if key in self.mass_cache:
               # Move to end (LRU)
               self.mass_cache.move_to_end(key)
               return self.mass_cache[key]
           
           # Compute and cache
           M = self.base_dynamics.mass_matrix(theta)
           
           # Maintain cache size
           if len(self.mass_cache) >= self.cache_size:
               self.mass_cache.popitem(last=False)
           
           self.mass_cache[key] = M
           return M
       
       def clear_cache(self):
           """Clear all caches."""
           self.mass_cache.clear()
           self.christoffel_cache.clear()
       
       def cache_stats(self):
           """Get cache statistics."""
           return {
               'mass_cache_size': len(self.mass_cache),
               'christoffel_cache_size': len(self.christoffel_cache),
               'cache_limit': self.cache_size
           }

   # Usage example
   optimized_dynamics = OptimizedDynamics(dynamics, cache_size=500)

   # Benchmark caching performance
   import time

   theta_test = np.array([0.1, 0.3, -0.2])
   
   # Uncached
   start_time = time.time()
   for _ in range(100):
       M = dynamics.mass_matrix(theta_test)
   uncached_time = time.time() - start_time
   
   # Cached (first call)
   start_time = time.time()
   for _ in range(100):
       M = optimized_dynamics.mass_matrix_cached(theta_test)
   cached_time = time.time() - start_time
   
   print(f"Uncached time: {uncached_time:.4f}s")
   print(f"Cached time: {cached_time:.4f}s")
   print(f"Speedup: {uncached_time/cached_time:.2f}x")
   print(f"Cache stats: {optimized_dynamics.cache_stats()}")

Parallel Computation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def parallel_trajectory_dynamics(dynamics, trajectory_data, num_processes=4):
       """Compute dynamics for trajectory using parallel processing."""
       
       import multiprocessing as mp
       from functools import partial
       
       def compute_point_dynamics(point_data):
           """Compute dynamics for a single trajectory point."""
           i, theta, theta_dot, theta_ddot = point_data
           
           try:
               # Compute all dynamic quantities
               M = dynamics.mass_matrix(theta)
               c = dynamics.velocity_quadratic_forces(theta, theta_dot)
               g_forces = dynamics.gravity_forces(theta, [0, 0, -9.81])
               
               # Inverse dynamics
               Ftip = [0, 0, 0, 0, 0, 0]
               tau = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, [0, 0, -9.81], Ftip)
               
               return i, {
                   'mass_matrix': M,
                   'coriolis_forces': c,
                   'gravity_forces': g_forces,
                   'required_torques': tau
               }
           except Exception as e:
               return i, {'error': str(e)}
       
       # Prepare data for parallel processing
       positions = trajectory_data['positions']
       velocities = trajectory_data['velocities']
       accelerations = trajectory_data['accelerations']
       
       point_data = [
           (i, positions[i], velocities[i], accelerations[i])
           for i in range(len(positions))
       ]
       
       # Process in parallel
       with mp.Pool(processes=num_processes) as pool:
           results = pool.map(compute_point_dynamics, point_data)
       
       # Reconstruct ordered results
       dynamics_data = {}
       for i, result in sorted(results):
           dynamics_data[i] = result
       
       return dynamics_data

   # Example usage (when trajectory is available)
   if 'trajectory' in locals():
       parallel_results = parallel_trajectory_dynamics(dynamics, trajectory)
       print(f"Computed dynamics for {len(parallel_results)} trajectory points")

GPU Acceleration Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def integrate_gpu_dynamics(dynamics, use_cuda=True):
       """Integrate with GPU-accelerated trajectory planning."""
       
       from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
       
       # Create GPU-optimized planner
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.joint_limits)
       
       planner = OptimizedTrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits,
           use_cuda=use_cuda,
           cuda_threshold=100,  # Use GPU for trajectories > 100 points
           enable_profiling=True,
           auto_optimize=True,
           target_speedup=40.0
       )
       
       # Generate high-resolution trajectory
       start_config = np.zeros(len(joint_limits))
       end_config = np.array([np.pi/2, np.pi/4, -np.pi/6])
       
       trajectory = planner.joint_trajectory(
           start_config, end_config, Tf=3.0, N=5000, method=5
       )
       
       # Compute dynamics using GPU-accelerated inverse dynamics
       torques = planner.inverse_dynamics_trajectory(
           trajectory['positions'],
           trajectory['velocities'], 
           trajectory['accelerations'])
       
       # Get performance statistics
       stats = planner.get_performance_stats()
       
       print(f"GPU acceleration used: {stats['gpu_calls'] > 0}")
       print(f"Total computation time: {stats['total_gpu_time'] + stats['total_cpu_time']:.4f}s")
       if stats['speedup_achieved'] > 0:
           print(f"Achieved speedup: {stats['speedup_achieved']:.2f}x")
       
       return trajectory, torques, stats

   # Example usage
   if CUDA_AVAILABLE:
       gpu_trajectory, gpu_torques, gpu_stats = integrate_gpu_dynamics(dynamics)
       print("GPU-accelerated dynamics computation completed")
   else:
       print("CUDA not available - using CPU computation")

Real-World Applications
-------------------------

Robot Control Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DynamicsBasedController:
       """Controller that uses full robot dynamics for high-performance control."""
       
       def __init__(self, dynamics, control_gains=None):
           self.dynamics = dynamics
           
           # Default PID gains
           if control_gains is None:
               n_joints = len(dynamics.joint_limits)
               self.Kp = np.full(n_joints, 100.0)
               self.Ki = np.full(n_joints, 10.0)
               self.Kd = np.full(n_joints, 20.0)
           else:
               self.Kp = control_gains['Kp']
               self.Ki = control_gains['Ki'] 
               self.Kd = control_gains['Kd']
           
           self.integral_error = None
       
       def computed_torque_control(self, theta_desired, theta_dot_desired, theta_ddot_desired,
                                  theta_current, theta_dot_current, dt, g=[0, 0, -9.81]):
           """Computed torque control using full dynamics model."""
           
           # Initialize integral error
           if self.integral_error is None:
               self.integral_error = np.zeros_like(theta_current)
           
           # Compute tracking errors
           position_error = theta_desired - theta_current
           velocity_error = theta_dot_desired - theta_dot_current
           self.integral_error += position_error * dt
           
           # PID feedback terms
           theta_ddot_feedback = (self.Kp * position_error + 
                                 self.Ki * self.integral_error +
                                 self.Kd * velocity_error)
           
           # Desired acceleration with feedback
           theta_ddot_command = theta_ddot_desired + theta_ddot_feedback
           
           # Compute feedforward torques using inverse dynamics
           Ftip = [0, 0, 0, 0, 0, 0]  # Assume no external forces
           tau_feedforward = self.dynamics.inverse_dynamics(
               theta_current, theta_dot_current, theta_ddot_command, g, Ftip
           )
           
           return tau_feedforward
       
       def adaptive_control(self, theta_desired, theta_dot_desired, theta_ddot_desired,
                           theta_current, theta_dot_current, adaptation_gain=0.1):
           """Adaptive control with parameter estimation."""
           
           # Simplified adaptive control - in practice this would be more sophisticated
           position_error = theta_desired - theta_current
           velocity_error = theta_dot_desired - theta_dot_current
           
           # Regression matrix (simplified)
           Y = np.outer(theta_ddot_desired, np.ones(len(theta_current)))
           
           # Parameter adaptation (simplified)
           if not hasattr(self, 'theta_hat'):
               self.theta_hat = np.ones(len(theta_current))
           
           # Update parameter estimates
           self.theta_hat += adaptation_gain * Y.T @ (position_error + velocity_error)
           
           # Compute control torques
           tau = Y @ self.theta_hat
           
           return tau
       
       def gravity_compensation_control(self, theta_current, theta_dot_desired, 
                                      theta_current_dot, g=[0, 0, -9.81]):
           """Simple gravity compensation with PD control."""
           
           # Gravity compensation
           gravity_torques = self.dynamics.gravity_forces(theta_current, g)
           
           # PD control for velocity tracking
           velocity_error = theta_dot_desired - theta_current_dot
           pd_torques = self.Kp * velocity_error + self.Kd * (-theta_current_dot)
           
           total_torques = gravity_torques + pd_torques
           return total_torques

   # Example usage with simulation
   def simulate_controlled_robot(dynamics, controller, target_trajectory, dt=0.001, duration=5.0):
       """Simulate robot under dynamics-based control."""
       
       num_steps = int(duration / dt)
       num_joints = len(dynamics.joint_limits)
       
       # Initialize state
       theta = np.zeros(num_joints)
       theta_dot = np.zeros(num_joints)
       
       # Storage for results
       results = {
           'time': np.zeros(num_steps),
           'theta_actual': np.zeros((num_steps, num_joints)),
           'theta_desired': np.zeros((num_steps, num_joints)),
           'theta_dot_actual': np.zeros((num_steps, num_joints)),
           'applied_torques': np.zeros((num_steps, num_joints)),
           'tracking_error': np.zeros((num_steps, num_joints))
       }
       
       # Simulation loop
       for i in range(num_steps):
           t = i * dt
           results['time'][i] = t
           
           # Get desired state from trajectory
           trajectory_index = min(int(t / duration * len(target_trajectory['positions'])), 
                                len(target_trajectory['positions']) - 1)
           
           theta_desired = target_trajectory['positions'][trajectory_index]
           theta_dot_desired = target_trajectory['velocities'][trajectory_index]
           theta_ddot_desired = target_trajectory['accelerations'][trajectory_index]
           
           # Store current state
           results['theta_actual'][i] = theta
           results['theta_desired'][i] = theta_desired
           results['theta_dot_actual'][i] = theta_dot
           results['tracking_error'][i] = theta_desired - theta
           
           # Compute control torques
           tau = controller.computed_torque_control(
               theta_desired, theta_dot_desired, theta_ddot_desired,
               theta, theta_dot, dt
           )
           results['applied_torques'][i] = tau
           
           # Apply torque limits (if available)
           if hasattr(dynamics, 'torque_limits'):
               tau = np.clip(tau, dynamics.torque_limits[:, 0], dynamics.torque_limits[:, 1])
           
           # Forward dynamics integration
           Ftip = [0, 0, 0, 0, 0, 0]
           theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, [0, 0, -9.81], Ftip)
           
           # Euler integration
           theta_dot += theta_ddot * dt
           theta += theta_dot * dt
       
       return results

   # Create controller and simulate
   controller = DynamicsBasedController(dynamics)
   
   if 'trajectory' in locals():
       simulation_results = simulate_controlled_robot(dynamics, controller, trajectory)
       
       # Plot simulation results
       fig, axes = plt.subplots(4, 1, figsize=(12, 12))
       
       for j in range(min(3, simulation_results['theta_actual'].shape[1])):
           # Position tracking
           axes[0].plot(simulation_results['time'], simulation_results['theta_desired'][:, j], 
                       '--', label=f'Joint {j+1} Desired', linewidth=2)
           axes[0].plot(simulation_results['time'], simulation_results['theta_actual'][:, j], 
                       label=f'Joint {j+1} Actual', linewidth=2)
           
           # Tracking error
           axes[1].plot(simulation_results['time'], simulation_results['tracking_error'][:, j], 
                       label=f'Joint {j+1}', linewidth=2)
           
           # Velocity
           axes[2].plot(simulation_results['time'], simulation_results['theta_dot_actual'][:, j], 
                       label=f'Joint {j+1}', linewidth=2)
           
           # Applied torques
           axes[3].plot(simulation_results['time'], simulation_results['applied_torques'][:, j], 
                       label=f'Joint {j+1}', linewidth=2)
       
       axes[0].set_ylabel('Position (rad)')
       axes[0].set_title('Position Tracking')
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)
       
       axes[1].set_ylabel('Error (rad)')
       axes[1].set_title('Tracking Error')
       axes[1].legend()
       axes[1].grid(True, alpha=0.3)
       
       axes[2].set_ylabel('Velocity (rad/s)')
       axes[2].set_title('Joint Velocities')
       axes[2].legend()
       axes[2].grid(True, alpha=0.3)
       
       axes[3].set_ylabel('Torque (N⋅m)')
       axes[3].set_xlabel('Time (s)')
       axes[3].set_title('Applied Torques')
       axes[3].legend()
       axes[3].grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()
       
       # Compute performance metrics
       final_error = np.mean(np.abs(simulation_results['tracking_error'][-100:, :]))
       max_error = np.max(np.abs(simulation_results['tracking_error']))
       print(f"Final tracking error: {final_error:.6f} rad")
       print(f"Maximum tracking error: {max_error:.6f} rad")

Collision and Contact Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ContactDynamics:
       """Handle contact and collision dynamics."""
       
       def __init__(self, base_dynamics, contact_stiffness=1e6, contact_damping=1e3):
           self.base_dynamics = base_dynamics
           self.contact_stiffness = contact_stiffness
           self.contact_damping = contact_damping
           
       def compute_contact_forces(self, theta, theta_dot, contact_points, contact_normals):
           """Compute contact forces using spring-damper model."""
           
           contact_forces = []
           
           for i, (point, normal) in enumerate(zip(contact_points, contact_normals)):
               # Get end-effector position and velocity
               T_ee = self.base_dynamics.forward_kinematics(theta)
               ee_position = T_ee[:3, 3]
               
               # Simple distance-based contact detection
               distance_to_contact = np.linalg.norm(ee_position - point)
               
               if distance_to_contact < 0.01:  # Contact threshold
                   # Penetration depth
                   penetration = 0.01 - distance_to_contact
                   
                   # Contact velocity (simplified)
                   J = self.base_dynamics.jacobian(theta)
                   ee_velocity = J[:3, :] @ theta_dot
                   contact_velocity = np.dot(ee_velocity, normal)
                   
                   # Spring-damper force
                   force_magnitude = (self.contact_stiffness * penetration + 
                                    self.contact_damping * contact_velocity)
                   
                   contact_force = force_magnitude * normal
                   contact_forces.append(contact_force)
               else:
                   contact_forces.append(np.zeros(3))
           
           return contact_forces
       
       def forward_dynamics_with_contact(self, theta, theta_dot, tau, g, contact_points, contact_normals):
           """Forward dynamics including contact forces."""
           
           # Compute contact forces
           contact_forces = self.compute_contact_forces(theta, theta_dot, contact_points, contact_normals)
           
           # Sum all contact forces into equivalent end-effector wrench
           total_contact_force = np.sum(contact_forces, axis=0) if contact_forces else np.zeros(3)
           Ftip_contact = np.concatenate([total_contact_force, np.zeros(3)])  # No contact moments for simplicity
           
           # Standard forward dynamics with contact forces
           theta_ddot = self.base_dynamics.forward_dynamics(theta, theta_dot, tau, g, Ftip_contact)
           
           return theta_ddot, contact_forces

   # Example: Robot interacting with environment
   def simulate_contact_interaction(dynamics, contact_points, contact_normals, duration=2.0, dt=0.001):
       """Simulate robot interaction with environment contacts."""
       
       contact_dynamics = ContactDynamics(dynamics)
       
       num_steps = int(duration / dt)
       num_joints = len(dynamics.joint_limits)
       
       # Initialize robot state
       theta = np.zeros(num_joints)
       theta_dot = np.zeros(num_joints)
       
       # Storage
       results = {
           'time': np.zeros(num_steps),
           'theta': np.zeros((num_steps, num_joints)),
           'theta_dot': np.zeros((num_steps, num_joints)),
           'contact_forces': [],
           'ee_position': np.zeros((num_steps, 3))
       }
       
       # Simple control: move towards contact
       target_position = contact_points[0] if contact_points else np.array([0.5, 0, 0.3])
       
       for i in range(num_steps):
           t = i * dt
           results['time'][i] = t
           results['theta'][i] = theta
           results['theta_dot'][i] = theta_dot
           
           # Get current end-effector position
           T_ee = dynamics.forward_kinematics(theta)
           ee_pos = T_ee[:3, 3]
           results['ee_position'][i] = ee_pos
           
           # Simple position control towards target
           position_error = target_position - ee_pos
           
           # Compute desired joint velocities (pseudo-inverse control)
           J = dynamics.jacobian(theta)
           J_pos = J[:3, :]  # Position part only
           
           try:
               theta_dot_desired = np.linalg.pinv(J_pos) @ (0.5 * position_error)  # Slow approach
           except:
               theta_dot_desired = np.zeros(num_joints)
           
           # PD control in joint space
           Kp = 50.0
           Kd = 10.0
           tau = Kp * (theta_dot_desired * dt) + Kd * (theta_dot_desired - theta_dot)
           
           # Forward dynamics with contact
           theta_ddot, contact_forces = contact_dynamics.forward_dynamics_with_contact(
               theta, theta_dot, tau, [0, 0, -9.81], contact_points, contact_normals
           )
           
           results['contact_forces'].append(contact_forces)
           
           # Integration
           theta_dot += theta_ddot * dt
           theta += theta_dot * dt
       
       return results

   # Example contact scenario
   contact_points = [np.array([0.4, 0, 0.2])]  # Contact point in workspace
   contact_normals = [np.array([1, 0, 0])]     # Contact normal (pointing away from surface)
   
   contact_results = simulate_contact_interaction(dynamics, contact_points, contact_normals)
   
   # Plot contact interaction
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   
   # End-effector trajectory
   axes[0, 0].plot(contact_results['ee_position'][:, 0], contact_results['ee_position'][:, 1])
   axes[0, 0].scatter(*contact_points[0][:2], c='red', s=100, marker='x', label='Contact Point')
   axes[0, 0].set_xlabel('X Position (m)')
   axes[0, 0].set_ylabel('Y Position (m)')
   axes[0, 0].set_title('End-Effector Trajectory')
   axes[0, 0].legend()
   axes[0, 0].grid(True, alpha=0.3)
   
   # Contact forces over time
   contact_force_magnitudes = [np.linalg.norm(cf[0]) if cf else 0 for cf in contact_results['contact_forces']]
   axes[0, 1].plot(contact_results['time'], contact_force_magnitudes)
   axes[0, 1].set_xlabel('Time (s)')
   axes[0, 1].set_ylabel('Contact Force Magnitude (N)')
   axes[0, 1].set_title('Contact Forces')
   axes[0, 1].grid(True, alpha=0.3)
   
   # Joint positions
   for j in range(min(3, contact_results['theta'].shape[1])):
       axes[1, 0].plot(contact_results['time'], contact_results['theta'][:, j], 
                      label=f'Joint {j+1}', linewidth=2)
   axes[1, 0].set_xlabel('Time (s)')
   axes[1, 0].set_ylabel('Joint Angle (rad)')
   axes[1, 0].set_title('Joint Positions')
   axes[1, 0].legend()
   axes[1, 0].grid(True, alpha=0.3)
   
   # Distance to contact
   distances = [np.linalg.norm(pos - contact_points[0]) for pos in contact_results['ee_position']]
   axes[1, 1].plot(contact_results['time'], distances)
   axes[1, 1].axhline(y=0.01, color='red', linestyle='--', label='Contact Threshold')
   axes[1, 1].set_xlabel('Time (s)')
   axes[1, 1].set_ylabel('Distance to Contact (m)')
   axes[1, 1].set_title('Approach to Contact')
   axes[1, 1].legend()
   axes[1, 1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Advanced Topics
-----------------

Linearized Dynamics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def linearize_dynamics(dynamics, theta_op, theta_dot_op, tau_op, g=[0, 0, -9.81]):
       """Linearize robot dynamics around an operating point."""
       
       epsilon = 1e-6
       n = len(theta_op)
       
       # Compute nominal forward dynamics
       Ftip = [0, 0, 0, 0, 0, 0]
       theta_ddot_nominal = dynamics.forward_dynamics(theta_op, theta_dot_op, tau_op, g, Ftip)
       
       # Linearization matrices
       A = np.zeros((2*n, 2*n))  # State matrix [theta; theta_dot]
       B = np.zeros((2*n, n))    # Input matrix (torque)
       
       # A matrix: ∂f/∂x where x = [theta; theta_dot]
       # Upper half: ∂theta_dot/∂theta = 0, ∂theta_dot/∂theta_dot = I
       A[:n, n:] = np.eye(n)
       
       # Lower half: ∂theta_ddot/∂theta and ∂theta_ddot/∂theta_dot
       for i in range(n):
           # ∂theta_ddot/∂theta_i
           theta_plus = theta_op.copy()
           theta_plus[i] += epsilon
           theta_ddot_plus = dynamics.forward_dynamics(theta_plus, theta_dot_op, tau_op, g, Ftip)
           
           theta_minus = theta_op.copy()
           theta_minus[i] -= epsilon
           theta_ddot_minus = dynamics.forward_dynamics(theta_minus, theta_dot_op, tau_op, g, Ftip)
           
           A[n:, i] = (theta_ddot_plus - theta_ddot_minus) / (2 * epsilon)
           
           # ∂theta_ddot/∂theta_dot_i
           theta_dot_plus = theta_dot_op.copy()
           theta_dot_plus[i] += epsilon
           theta_ddot_plus = dynamics.forward_dynamics(theta_op, theta_dot_plus, tau_op, g, Ftip)
           
           theta_dot_minus = theta_dot_op.copy()
           theta_dot_minus[i] -= epsilon
           theta_ddot_minus = dynamics.forward_dynamics(theta_op, theta_dot_minus, tau_op, g, Ftip)
           
           A[n:, n+i] = (theta_ddot_plus - theta_ddot_minus) / (2 * epsilon)
       
       # B matrix: ∂f/∂u where u = tau
       for i in range(n):
           tau_plus = tau_op.copy()
           tau_plus[i] += epsilon
           theta_ddot_plus = dynamics.forward_dynamics(theta_op, theta_dot_op, tau_plus, g, Ftip)
           
           tau_minus = tau_op.copy()
           tau_minus[i] -= epsilon
           theta_ddot_minus = dynamics.forward_dynamics(theta_op, theta_dot_op, tau_minus, g, Ftip)
           
           B[n:, i] = (theta_ddot_plus - theta_ddot_minus) / (2 * epsilon)
       
       return A, B, theta_ddot_nominal

   # Example: Linearize around equilibrium
   theta_equilibrium = np.array([0, np.pi/6, -np.pi/6])
   theta_dot_equilibrium = np.zeros(len(theta_equilibrium))
   
   # Find equilibrium torque (gravity compensation)
   tau_equilibrium = dynamics.gravity_forces(theta_equilibrium, [0, 0, -9.81])
   
   A, B, _ = linearize_dynamics(dynamics, theta_equilibrium, theta_dot_equilibrium, tau_equilibrium)
   
   print(f"Linearized A matrix shape: {A.shape}")
   print(f"Linearized B matrix shape: {B.shape}")
   
   # Analyze stability
   eigenvalues = np.linalg.eigvals(A)
   stable = np.all(np.real(eigenvalues) <= 0)
   
   print(f"System stable around equilibrium: {stable}")
   print(f"Eigenvalues: {eigenvalues}")

Model Identification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DynamicsIdentification:
       """Identify robot dynamic parameters from experimental data."""
       
       def __init__(self, nominal_dynamics):
           self.nominal_dynamics = nominal_dynamics
           
       def generate_identification_trajectory(self, duration=10.0, dt=0.01):
           """Generate exciting trajectory for parameter identification."""
           
           t = np.arange(0, duration, dt)
           n_joints = len(self.nominal_dynamics.joint_limits)
           
           # Multi-frequency excitation
           trajectory = np.zeros((len(t), n_joints))
           
           for j in range(n_joints):
               # Fundamental frequency
               f1 = 0.5 + j * 0.1  # Different frequency for each joint
               
               # Multiple harmonics for rich excitation
               for harmonic in range(1, 4):
                   amplitude = 0.3 / harmonic  # Decreasing amplitude
                   trajectory[:, j] += amplitude * np.sin(2 * np.pi * f1 * harmonic * t)
           
           # Compute velocities and accelerations
           velocities = np.gradient(trajectory, dt, axis=0)
           accelerations = np.gradient(velocities, dt, axis=0)
           
           return {
               'time': t,
               'positions': trajectory,
               'velocities': velocities,
               'accelerations': accelerations
           }
       
       def collect_identification_data(self, trajectory, noise_level=0.01):
           """Simulate data collection with sensor noise."""
           
           positions = trajectory['positions']
           velocities = trajectory['velocities']
           accelerations = trajectory['accelerations']
           
           # Generate "measured" torques using nominal model
           measured_torques = []
           
           for i in range(len(positions)):
               tau_nominal = self.nominal_dynamics.inverse_dynamics(
                   positions[i], velocities[i], accelerations[i],
                   [0, 0, -9.81], [0, 0, 0, 0, 0, 0]
               )
               
               # Add measurement noise
               noise = np.random.normal(0, noise_level, len(tau_nominal))
               tau_measured = tau_nominal + noise
               
               measured_torques.append(tau_measured)
           
           return np.array(measured_torques)
       
       def estimate_parameters(self, trajectory, measured_torques, regularization=1e-6):
           """Estimate dynamic parameters using least squares."""
           
           positions = trajectory['positions']
           velocities = trajectory['velocities']
           accelerations = trajectory['accelerations']
           
           n_samples = len(positions)
           n_joints = positions.shape[1]
           
           # Build regression matrix (simplified - just inertial parameters)
           # In practice, this would include all dynamic parameters
           Y = np.zeros((n_samples * n_joints, n_joints))  # Simplified regression matrix
           y = measured_torques.flatten()
           
           for i in range(n_samples):
               for j in range(n_joints):
                   row_idx = i * n_joints + j
                   # Simplified: assume diagonal mass matrix terms
                   Y[row_idx, j] = accelerations[i, j]
           
           # Least squares estimation with regularization
           theta_estimated = np.linalg.solve(
               Y.T @ Y + regularization * np.eye(Y.shape[1]),
               Y.T @ y
           )
           
           return theta_estimated
       
       def validate_identification(self, estimated_params, validation_trajectory, validation_torques):
           """Validate identified parameters on new data."""
           
           positions = validation_trajectory['positions']
           velocities = validation_trajectory['velocities']
           accelerations = validation_trajectory['accelerations']
           
           predicted_torques = []
           
           for i in range(len(positions)):
               # Use estimated parameters to predict torques
               # This is simplified - in practice would use full dynamic model
               tau_predicted = estimated_params * accelerations[i]
               predicted_torques.append(tau_predicted)
           
           predicted_torques = np.array(predicted_torques)
           
           # Compute validation metrics
           mse = np.mean((validation_torques - predicted_torques) ** 2)
           r_squared = 1 - np.sum((validation_torques - predicted_torques) ** 2) / \
                          np.sum((validation_torques - np.mean(validation_torques)) ** 2)
           
           return {
               'mse': mse,
               'r_squared': r_squared,
               'predicted_torques': predicted_torques
           }

   # Example identification procedure
   identifier = DynamicsIdentification(dynamics)
   
   # Generate identification trajectory
   id_trajectory = identifier.generate_identification_trajectory(duration=5.0)
   
   # Simulate data collection
   measured_torques = identifier.collect_identification_data(id_trajectory, noise_level=0.05)
   
   # Estimate parameters
   estimated_params = identifier.estimate_parameters(id_trajectory, measured_torques)
   print(f"Estimated parameters: {estimated_params}")
   
   # Generate validation data
   val_trajectory = identifier.generate_identification_trajectory(duration=2.0)
   val_torques = identifier.collect_identification_data(val_trajectory, noise_level=0.05)
   
   # Validate identification
   validation_results = identifier.validate_identification(
       estimated_params, val_trajectory, val_torques
   )
   
   print(f"Validation R²: {validation_results['r_squared']:.4f}")
   print(f"Validation MSE: {validation_results['mse']:.6f}")

Best Practices and Tips
-------------------------

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Cache mass matrices for repetitive computations
   cached_dynamics = OptimizedDynamics(dynamics, cache_size=1000)
   
   # 2. Use appropriate numerical tolerances
   def check_dynamics_properties(dynamics, theta):
       """Check important dynamic properties."""
       
       M = dynamics.mass_matrix(theta)
       
       # Positive definiteness
       eigenvals = np.linalg.eigvals(M)
       is_positive_definite = np.all(eigenvals > 1e-10)
       
       # Symmetry
       is_symmetric = np.allclose(M, M.T, atol=1e-12)
       
       # Condition number
       condition_number = np.linalg.cond(M)
       
       return {
           'positive_definite': is_positive_definite,
           'symmetric': is_symmetric,
           'condition_number': condition_number,
           'min_eigenvalue': np.min(eigenvals),
           'max_eigenvalue': np.max(eigenvals)
       }
   
   # 3. Monitor numerical stability
   theta_test = np.array([np.pi/4, np.pi/6, -np.pi/3])
   properties = check_dynamics_properties(dynamics, theta_test)
   
   print("Dynamics Properties Check:")
   for key, value in properties.items():
       print(f"  {key}: {value}")
   
   # 4. Use appropriate integration methods for simulation
   def runge_kutta_4_step(dynamics, theta, theta_dot, tau, dt, g=[0, 0, -9.81]):
       """4th-order Runge-Kutta integration step."""
       
       Ftip = [0, 0, 0, 0, 0, 0]
       
       # k1
       k1_theta_dot = theta_dot
       k1_theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, g, Ftip)
       
       # k2
       k2_theta_dot = theta_dot + 0.5 * dt * k1_theta_ddot
       k2_theta_ddot = dynamics.forward_dynamics(
           theta + 0.5 * dt * k1_theta_dot, k2_theta_dot, tau, g, Ftip
       )
       
       # k3
       k3_theta_dot = theta_dot + 0.5 * dt * k2_theta_ddot
       k3_theta_ddot = dynamics.forward_dynamics(
           theta + 0.5 * dt * k2_theta_dot, k3_theta_dot, tau, g, Ftip
       )
       
       # k4
       k4_theta_dot = theta_dot + dt * k3_theta_ddot
       k4_theta_ddot = dynamics.forward_dynamics(
           theta + dt * k3_theta_dot, k4_theta_dot, tau, g, Ftip
       )
       
       # Final update
       theta_new = theta + (dt/6) * (k1_theta_dot + 2*k2_theta_dot + 2*k3_theta_dot + k4_theta_dot)
theta_dot_new = theta_dot + (dt/6) * (k1_theta_ddot + 2*k2_theta_ddot + 2*k3_theta_ddot + k4_theta_ddot)
       
       return theta_new, theta_dot_new

Numerical Stability
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def ensure_numerical_stability(dynamics, theta, tolerance=1e-10):
       """Ensure numerical stability of dynamic computations."""
       
       try:
           # Check mass matrix properties
           M = dynamics.mass_matrix(theta)
           
           # 1. Check for NaN or infinite values
           if not np.all(np.isfinite(M)):
               raise ValueError("Mass matrix contains NaN or infinite values")
           
           # 2. Check positive definiteness
           eigenvals = np.linalg.eigvals(M)
           if np.any(eigenvals <= tolerance):
               print(f"Warning: Mass matrix near-singular (min eigenvalue: {np.min(eigenvals):.2e})")
               
               # Regularization for numerical stability
               M_regularized = M + tolerance * np.eye(M.shape[0])
               print("Applied regularization to mass matrix")
               return M_regularized
           
           # 3. Check symmetry
           if not np.allclose(M, M.T, atol=tolerance):
               print("Warning: Mass matrix not symmetric, enforcing symmetry")
               M = 0.5 * (M + M.T)
           
           return M
           
       except Exception as e:
           print(f"Error in dynamics computation: {e}")
           # Fallback to identity matrix scaled by average inertia
           n = len(theta)
           fallback_inertia = 1.0  # Default inertia value
           return fallback_inertia * np.eye(n)

   # Example usage with error handling
   def safe_inverse_dynamics(dynamics, theta, theta_dot, theta_ddot, g, Ftip):
       """Inverse dynamics with numerical safety checks."""
       
       try:
           # Ensure numerical stability
           M = ensure_numerical_stability(dynamics, theta)
           
           # Compute other terms with bounds checking
           c = dynamics.velocity_quadratic_forces(theta, theta_dot)
           g_forces = dynamics.gravity_forces(theta, g)
           J_transpose = dynamics.jacobian(theta).T
           
           # Check for unreasonable values
           if np.any(np.abs(c) > 1000):  # Reasonable torque limit
               print("Warning: Large Coriolis forces detected")
               c = np.clip(c, -1000, 1000)
           
           if np.any(np.abs(g_forces) > 500):  # Reasonable gravity limit
               print("Warning: Large gravity forces detected")
               g_forces = np.clip(g_forces, -500, 500)
           
           # Compute final torques
           tau = M @ theta_ddot + c + g_forces + J_transpose @ Ftip
           
           return tau
           
       except Exception as e:
           print(f"Error in inverse dynamics: {e}")
           # Return zero torques as fallback
           return np.zeros(len(theta))

Common Pitfalls and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def dynamics_debugging_guide():
       """Common issues and solutions in robot dynamics."""
       
       issues_and_solutions = {
           "Mass matrix not positive definite": {
               "causes": [
                   "Incorrect inertial parameters",
                   "Singular robot configuration", 
                   "Numerical precision issues"
               ],
               "solutions": [
                   "Check URDF inertial properties",
                   "Add regularization term",
                   "Avoid singular configurations",
                   "Use higher precision arithmetic"
               ],
               "code_example": """
   # Add regularization to mass matrix
   M = dynamics.mass_matrix(theta)
   eigenvals = np.linalg.eigvals(M)
   if np.min(eigenvals) < 1e-6:
       M += 1e-6 * np.eye(M.shape[0])
               """
           },
           
           "Large Coriolis forces": {
               "causes": [
                   "High joint velocities",
                   "Incorrect mass distribution",
                   "Numerical differentiation errors"
               ],
               "solutions": [
                   "Limit maximum joint velocities",
                   "Verify link mass properties",
                   "Use analytical derivatives when possible",
                   "Apply velocity-dependent damping"
               ],
               "code_example": """
   # Velocity limiting
   max_velocity = 5.0  # rad/s
   theta_dot_limited = np.clip(theta_dot, -max_velocity, max_velocity)
   c = dynamics.velocity_quadratic_forces(theta, theta_dot_limited)
               """
           },
           
           "Integration instability": {
               "causes": [
                   "Too large time step",
                   "Stiff dynamics",
                   "Discontinuous forces"
               ],
               "solutions": [
                   "Reduce integration time step",
                   "Use implicit integration methods",
                   "Add numerical damping",
                   "Smooth force transitions"
               ],
               "code_example": """
   # Adaptive time stepping
   def adaptive_timestep(error, dt, tolerance=1e-6):
       if error > tolerance:
           return dt * 0.5  # Reduce time step
       elif error < tolerance * 0.1:
           return dt * 1.1  # Increase time step
       return dt
               """
           }
       }
       
       return issues_and_solutions

   # Print debugging guide
   debugging_info = dynamics_debugging_guide()
   for issue, info in debugging_info.items():
       print(f"\n{issue.upper()}:")
       print(f"Causes: {', '.join(info['causes'])}")
       print(f"Solutions: {', '.join(info['solutions'])}")
       if 'code_example' in info:
           print("Code example:")
           print(info['code_example'])

Validation and Testing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def comprehensive_dynamics_test(dynamics, test_configurations=None):
       """Comprehensive test suite for robot dynamics."""
       
       if test_configurations is None:
           # Generate test configurations
           n_joints = len(dynamics.joint_limits)
           n_tests = 20
           
           test_configurations = []
           for _ in range(n_tests):
               # Random configurations within joint limits
               theta = np.random.uniform(
                   [limit[0] for limit in dynamics.joint_limits],
                   [limit[1] for limit in dynamics.joint_limits]
               )
               test_configurations.append(theta)
       
       test_results = {
           'mass_matrix_tests': [],
           'inverse_dynamics_tests': [],
           'forward_dynamics_tests': [],
           'energy_conservation_tests': []
       }
       
       print("Running comprehensive dynamics tests...")
       
       for i, theta in enumerate(test_configurations):
           print(f"Test {i+1}/{len(test_configurations)}", end=' ')
           
           try:
               # Test 1: Mass matrix properties
               M = dynamics.mass_matrix(theta)
               eigenvals = np.linalg.eigvals(M)
               
               mass_test = {
                   'configuration': theta,
                   'positive_definite': np.all(eigenvals > 1e-10),
                   'symmetric': np.allclose(M, M.T, atol=1e-12),
                   'condition_number': np.linalg.cond(M),
                   'passed': True
               }
               
               if not mass_test['positive_definite'] or not mass_test['symmetric']:
                   mass_test['passed'] = False
               
               test_results['mass_matrix_tests'].append(mass_test)
               
               # Test 2: Inverse-Forward dynamics consistency
               theta_dot = np.random.uniform(-1, 1, len(theta))
               theta_ddot = np.random.uniform(-2, 2, len(theta))
               g = [0, 0, -9.81]
               Ftip = [0, 0, 0, 0, 0, 0]
               
               # Inverse dynamics
               tau = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, g, Ftip)
               
               # Forward dynamics (should recover theta_ddot)
               theta_ddot_recovered = dynamics.forward_dynamics(theta, theta_dot, tau, g, Ftip)
               
               dynamics_test = {
                   'configuration': theta,
                   'acceleration_error': np.linalg.norm(theta_ddot - theta_ddot_recovered),
                   'passed': np.allclose(theta_ddot, theta_ddot_recovered, atol=1e-6)
               }
               
               test_results['inverse_dynamics_tests'].append(dynamics_test)
               test_results['forward_dynamics_tests'].append(dynamics_test)
               
               # Test 3: Energy conservation (simplified)
               # For a conservative system, energy should be conserved
               dt = 0.001
               duration = 0.1
               
               theta_sim = theta.copy()
               theta_dot_sim = theta_dot.copy()
               
               initial_energy = compute_robot_energy(dynamics, theta_sim, theta_dot_sim, g)
               
               # Simulate with zero torques (free motion)
               for _ in range(int(duration / dt)):
                   tau_zero = np.zeros(len(theta))
                   theta_ddot_sim = dynamics.forward_dynamics(theta_sim, theta_dot_sim, tau_zero, g, Ftip)
                   
                   # Simple Euler integration
                   theta_dot_sim += theta_ddot_sim * dt
                   theta_sim += theta_dot_sim * dt
               
               final_energy = compute_robot_energy(dynamics, theta_sim, theta_dot_sim, g)
               
               energy_test = {
                   'configuration': theta,
                   'initial_energy': initial_energy['total'],
                   'final_energy': final_energy['total'],
                   'energy_drift': abs(final_energy['total'] - initial_energy['total']),
                   'passed': abs(final_energy['total'] - initial_energy['total']) < 0.1  # Tolerance for numerical errors
               }
               
               test_results['energy_conservation_tests'].append(energy_test)
               
               print("✓")
               
           except Exception as e:
               print(f"✗ (Error: {e})")
               continue
       
       # Summarize results
       print("\nTest Results Summary:")
       print("=" * 50)
       
       mass_passed = sum(1 for test in test_results['mass_matrix_tests'] if test['passed'])
       print(f"Mass Matrix Tests: {mass_passed}/{len(test_results['mass_matrix_tests'])} passed")
       
       dynamics_passed = sum(1 for test in test_results['inverse_dynamics_tests'] if test['passed'])
       print(f"Dynamics Consistency Tests: {dynamics_passed}/{len(test_results['inverse_dynamics_tests'])} passed")
       
       energy_passed = sum(1 for test in test_results['energy_conservation_tests'] if test['passed'])
       print(f"Energy Conservation Tests: {energy_passed}/{len(test_results['energy_conservation_tests'])} passed")
       
       # Detailed analysis
       if test_results['mass_matrix_tests']:
           condition_numbers = [test['condition_number'] for test in test_results['mass_matrix_tests']]
           print(f"Mass Matrix Condition Numbers: mean={np.mean(condition_numbers):.2e}, max={np.max(condition_numbers):.2e}")
       
       if test_results['inverse_dynamics_tests']:
           acceleration_errors = [test['acceleration_error'] for test in test_results['inverse_dynamics_tests']]
           print(f"Acceleration Errors: mean={np.mean(acceleration_errors):.2e}, max={np.max(acceleration_errors):.2e}")
       
       return test_results

   # Run comprehensive test
   test_results = comprehensive_dynamics_test(dynamics)

Integration with Other Modules
---------------------------------

Control System Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Integrate dynamics with ManipulaPy control module
   from ManipulaPy.control import ManipulatorController
   
   def create_dynamics_aware_controller(dynamics, control_type="computed_torque"):
       """Create a controller that uses full dynamics model."""
       
       controller = ManipulatorController(dynamics)
       
       if control_type == "computed_torque":
           def control_function(theta_desired, theta_dot_desired, theta_ddot_desired,
                               theta_current, theta_dot_current, dt):
               
               # PID gains (should be tuned for specific robot)
               Kp = np.full(len(theta_current), 100.0)
               Ki = np.full(len(theta_current), 10.0)
               Kd = np.full(len(theta_current), 20.0)
               
               return controller.computed_torque_control(
                   thetalistd=theta_desired,
                   dthetalistd=theta_dot_desired,
                   ddthetalistd=theta_ddot_desired,
                   thetalist=theta_current,
                   dthetalist=theta_dot_current,
                   g=[0, 0, -9.81],
                   dt=dt,
                   Kp=Kp, Ki=Ki, Kd=Kd
               )
       
       elif control_type == "adaptive":
           def control_function(theta_desired, theta_dot_desired, theta_ddot_desired,
                               theta_current, theta_dot_current, dt):
               
               measurement_error = theta_desired - theta_current
               
               return controller.adaptive_control(
                   thetalist=theta_current,
                   dthetalist=theta_dot_current,
                   ddthetalist=theta_ddot_desired,
                   g=[0, 0, -9.81],
                   Ftip=[0, 0, 0, 0, 0, 0],
                   measurement_error=measurement_error,
                   adaptation_gain=0.1
               )
       
       else:
           raise ValueError(f"Unknown control type: {control_type}")
       
       return control_function

Path Planning Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Use dynamics in trajectory optimization
   from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
   
   def plan_dynamics_optimal_trajectory(dynamics, start_config, end_config, 
                                      duration=5.0, num_points=1000):
       """Plan trajectory considering dynamic constraints."""
       
       joint_limits = [(-np.pi, np.pi)] * len(start_config)
       
       # Create trajectory planner
       planner = OptimizedTrajectoryPlanning(
           robot, "robot.urdf", dynamics, joint_limits,
           use_cuda=True,  # Use GPU acceleration if available
           enable_profiling=True
       )
       
       # Generate initial trajectory
       trajectory = planner.joint_trajectory(
           start_config, end_config, duration, num_points, method=5
       )
       
       # Compute required torques
       torques = planner.inverse_dynamics_trajectory(
           trajectory['positions'],
           trajectory['velocities'],
           trajectory['accelerations']
       )
       
       # Check torque limits and feasibility
       max_torques = np.max(np.abs(torques), axis=0)
       torque_limits = np.array([100, 80, 60])  # Example limits
       
       if np.any(max_torques > torque_limits):
           print("Warning: Trajectory exceeds torque limits")
           print(f"Max torques: {max_torques}")
           print(f"Torque limits: {torque_limits}")
           
           # Could implement trajectory optimization here
           # to satisfy dynamic constraints
       
       return {
           'trajectory': trajectory,
           'torques': torques,
           'max_torques': max_torques,
           'feasible': np.all(max_torques <= torque_limits)
       }

Simulation Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Integrate with PyBullet simulation
   from ManipulaPy.sim import Simulation
   
   def compare_dynamics_models(dynamics, urdf_path, trajectory):
       """Compare ManipulaPy dynamics with PyBullet physics."""
       
       # Create PyBullet simulation
       joint_limits = [(-np.pi, np.pi)] * len(dynamics.joint_limits)
       sim = Simulation(urdf_path, joint_limits)
       sim.initialize_robot()
       
       # Run ManipulaPy dynamics
       manipulapy_results = []
       positions = trajectory['positions']
       velocities = trajectory['velocities']
       
       for i in range(len(positions) - 1):
           # Compute torques using ManipulaPy
           acceleration = (velocities[i+1] - velocities[i]) / 0.01  # Assume dt=0.01
           
           tau = dynamics.inverse_dynamics(
               positions[i], velocities[i], acceleration,
               [0, 0, -9.81], [0, 0, 0, 0, 0, 0]
           )
           
           manipulapy_results.append({
               'position': positions[i],
               'velocity': velocities[i],
               'acceleration': acceleration,
               'torque': tau
           })
       
       # Run PyBullet simulation
       pybullet_results = []
       sim.set_joint_positions(positions[0])
       
       for i in range(len(positions) - 1):
           # Apply computed torques in PyBullet
           current_pos = sim.get_joint_positions()
           
           pybullet_results.append({
               'position': current_pos,
               'velocity': sim.get_joint_velocities() if hasattr(sim, 'get_joint_velocities') else velocities[i],
               'torque': manipulapy_results[i]['torque']
           })
           
           # Step simulation
           sim.set_joint_positions(positions[i+1])
       
       # Compare results
       position_errors = []
       for mp_result, pb_result in zip(manipulapy_results, pybullet_results):
           error = np.linalg.norm(mp_result['position'] - pb_result['position'])
           position_errors.append(error)
       
       print(f"Average position error: {np.mean(position_errors):.6f} rad")
       print(f"Maximum position error: {np.max(position_errors):.6f} rad")
       
       return manipulapy_results, pybullet_results, position_errors

Conclusion
-----------

This comprehensive guide has covered the essential aspects of robot dynamics in ManipulaPy, from basic concepts to advanced applications. Key takeaways include:

**Fundamental Concepts**
- Robot dynamics describes the relationship between forces/torques and motion
- The Newton-Euler equation of motion is central to all dynamic computations
- Mass matrix, Coriolis forces, and gravity are the three main components

**Practical Implementation**
- ManipulaPy provides a complete dynamics framework with automatic URDF parsing
- Caching and optimization techniques are crucial for real-time applications
- Numerical stability requires careful handling of edge cases

**Advanced Topics**
- Contact dynamics and collision handling extend basic rigid-body dynamics
- Model identification enables adaptation to real-world systems
- Integration with control and planning modules enables complete robotic solutions

**Best Practices**
- Always validate dynamic properties (positive definiteness, symmetry)
- Use appropriate integration methods for simulation
- Implement comprehensive testing for reliability
- Consider GPU acceleration for computationally intensive applications

For further exploration, consider experimenting with:
- Custom dynamic models for specialized robots
- Real-time control implementation using the provided frameworks
- Integration with perception systems for adaptive behavior
- Advanced contact and interaction modeling

The ManipulaPy dynamics module provides a solid foundation for both research and practical robotics applications, with the flexibility to extend and customize for specific needs.

.. note::
   For the latest updates and additional examples, visit the ManipulaPy documentation at https://manipulapy.readthedocs.io/

.. seealso::
   **Related Documentation:**
      **Getting Started:**
   
   - :doc:`../getting_started/index` - Installation and quick start guide
   - :doc:`../Installation Guide` - Detailed installation instructions

   **API Reference:**
   
   - :doc:`../api/dynamics` - Dynamics module API documentation
   - :doc:`../api/kinematics` - Kinematics module API reference
   - :doc:`../api/control` - Control module API reference
   - :doc:`../api/path_planning` - Path planning module API reference
   - :doc:`../api/simulation` - Simulation module API reference

   **User Guides:**
   
   - :doc:`../user_guide/Kinematics` - Robot kinematics fundamentals and forward/inverse kinematics
   - :doc:`../user_guide/Control` - Robot control systems and advanced control algorithms
   - :doc:`../user_guide/Path_Planning` - Trajectory planning and path optimization
   - :doc:`../user_guide/Simulation` - PyBullet integration and real-time simulation
   - :doc:`../user_guide/CUDA_Kernels` - GPU acceleration and CUDA optimization
   - :doc:`../user_guide/URDF_Processor` - URDF parsing and robot model creation
   - :doc:`../user_guide/Singularity_Analysis` - Singularity detection and workspace analysis
   - :doc:`../user_guide/Potential_Field` - Potential field methods and collision avoidance
   

   
=======
       coriolis = np.zeros(n)
>>>>>>> cadojo/jcarpinelli/misc
