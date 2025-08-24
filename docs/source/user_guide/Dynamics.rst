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