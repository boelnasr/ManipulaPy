# Dynamics User Guide
===================

This comprehensive guide covers robot dynamics computations in ManipulaPy, optimized for Python 3.10.12.

.. note::
   This guide is written for Python 3.10.12 users and includes version-specific optimizations and performance improvements.

Introduction to Robot Dynamics
-------------------------------

Robot dynamics deals with the relationship between forces/torques and motion in robotic systems. Unlike kinematics, which only considers geometric relationships, dynamics incorporates:

- **Mass properties** of robot links

- **Inertial forces** due to acceleration

- **Gravitational forces** acting on the robot

- **External forces** applied to the robot

- **Joint torques** required for desired motion

Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~

The fundamental equation of robot dynamics is the **Newton-Euler equation**:

.. math::

   \tau = M(\theta)\ddot{\theta} + C(\theta,\dot{\theta})\dot{\theta} + G(\theta) + J^T(\theta)F_{ext}

Where:
   - :math:`\tau` = joint torques
   - :math:`M(\theta)` = mass/inertia matrix
   - :math:`C(\theta,\dot{\theta})` = Coriolis and centrifugal forces
   - :math:`G(\theta)` = gravitational forces
   - :math:`J^T(\theta)F_{ext}` = external forces mapped to joint space

Key Concepts
~~~~~~~~~~~~

**Forward Dynamics**
   Given joint torques, compute joint accelerations: :math:`\ddot{\theta} = f(\tau, \theta, \dot{\theta})`

**Inverse Dynamics**
   Given desired motion, compute required torques: :math:`\tau = f(\theta, \dot{\theta}, \ddot{\theta})`

**Mass Matrix**
   Represents the robot's inertial properties and coupling between joints

**Velocity-Dependent Forces**
   Coriolis and centrifugal forces that arise from robot motion

Setting Up Robot Dynamics
--------------------------

Basic Setup from URDF
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~

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
-----------------------

The mass matrix represents the robot's inertial properties and varies with configuration.

Computing Mass Matrix
~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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
-------------------------

Coriolis and centrifugal forces arise from robot motion and joint coupling.

Computing Velocity Forces
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define joint state
   theta = np.array([0.1, 0.3, -0.2])      # Joint positions
   theta_dot = np.array([0.5, -0.3, 0.8])  # Joint velocities

   # Compute velocity-dependent forces
   c = dynamics.velocity_quadratic_forces(theta, theta_dot)

   print(f"Velocity forces: {c}")
   print(f"Force magnitude: {np.linalg.norm(c)}")

Analyzing Velocity Effects
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

Separate centrifugal (velocity²) and Coriolis (cross-coupling) effects:

.. code-block:: python

   def decompose_velocity_forces(dynamics, theta, theta_dot):
       """Decompose velocity forces into centrifugal and Coriolis components."""
       
       n = len(theta)
       centrifugal = np.zeros(n)
       coriolis = np.zeros(n)
       
       # Centrifugal forces (diagonal terms)
       for i in range(n):
           theta_dot_i = np.zeros(n)
           theta_dot_i[i] = theta_dot[i]
           c_i = dynamics.velocity_quadratic_forces(theta, theta_dot_i)
           centrifugal += c_i
       
       # Total velocity forces
       c_total = dynamics.velocity_quadratic_forces(theta, theta_dot)
       
       # Coriolis forces (off-diagonal coupling)
       coriolis = c_total - centrifugal
       
       return centrifugal, coriolis

   # Example usage
   centrifugal, coriolis = decompose_velocity_forces(dynamics, theta, theta_dot)
   
   print(f"Centrifugal forces: {centrifugal}")
   print(f"Coriolis forces: {coriolis}")
   print(f"Total: {centrifugal + coriolis}")

Gravity Compensation
--------------------

Gravity forces must be overcome for the robot to maintain its position.

Computing Gravity Forces
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard Earth gravity
   g = [0, 0, -9.81]  # m/s²

   # Compute gravitational forces
   gravity_forces = dynamics.gravity_forces(theta, g)

   print(f"Gravity forces: {gravity_forces}")
   print(f"Total gravity torque: {np.linalg.norm(gravity_forces)} N⋅m")

Configuration Dependence
~~~~~~~~~~~~~~~~~~~~~~~~

Gravity forces vary significantly with robot pose:

.. code-block:: python

   def gravity_analysis(dynamics, g=[0, 0, -9.81]):
       """Analyze gravity forces across workspace."""
       
       # Test range of configurations
       angles = np.linspace(-np.pi/2, np.pi/2, 30)
       gravity_magnitudes = []
       configurations = []
       
       for angle1 in angles[::5]:  # Subsample for speed
           for angle2 in angles[::5]:
               theta = np.array([angle1, angle2, 0.0])
               g_forces = dynamics.gravity_forces(theta, g)
               
               gravity_magnitudes.append(np.linalg.norm(g_forces))
               configurations.append(theta.copy())
       
       configurations = np.array(configurations)
       gravity_magnitudes = np.array(gravity_magnitudes)
       
       # Find maximum gravity configuration
       max_idx = np.argmax(gravity_magnitudes)
       max_config = configurations[max_idx]
       max_gravity = gravity_magnitudes[max_idx]
       
       print(f"Maximum gravity torque: {max_gravity:.2f} N⋅m")
       print(f"At configuration: {np.degrees(max_config)} degrees")
       
       return configurations, gravity_magnitudes

   # Analyze gravity effects
   configs, g_mags = gravity_analysis(dynamics)

Gravity Compensation Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def gravity_compensation_demo():
       """Demonstrate gravity compensation for robot control."""
       
       # Simulation parameters
       dt = 0.01  # Time step
       duration = 5.0  # Simulation time
       time_steps = np.arange(0, duration, dt)
       
       # Initial conditions
       theta = np.array([0.2, 0.5, -0.3])
       theta_dot = np.zeros(3)
       
       # Storage for results
       positions = []
       velocities = []
       torques = []
       
       for t in time_steps:
           # Compute gravity compensation torque
           tau_gravity = dynamics.gravity_forces(theta, [0, 0, -9.81])
           
           # Apply gravity compensation (torque = gravity compensation)
           tau_applied = tau_gravity
           
           # Simulate dynamics (simplified)
           M = dynamics.mass_matrix(theta)
           c = dynamics.velocity_quadratic_forces(theta, theta_dot)
           
           # Forward dynamics: M*theta_ddot = tau - c - g
           tau_net = tau_applied - c - tau_gravity  # Should be ~0 for perfect compensation
           theta_ddot = np.linalg.solve(M, tau_net)
           
           # Integrate
           theta_dot += theta_ddot * dt
           theta += theta_dot * dt
           
           # Store results
           positions.append(theta.copy())
           velocities.append(theta_dot.copy())
           torques.append(tau_applied.copy())
       
       # Plot results
       positions = np.array(positions)
       velocities = np.array(velocities)
       torques = np.array(torques)
       
       fig, axes = plt.subplots(3, 1, figsize=(10, 8))
       
       # Positions
       for i in range(3):
           axes[0].plot(time_steps, np.degrees(positions[:, i]), label=f'Joint {i+1}')
       axes[0].set_ylabel('Position (degrees)')
       axes[0].set_title('Joint Positions with Gravity Compensation')
       axes[0].legend()
       axes[0].grid(True)
       
       # Velocities
       for i in range(3):
           axes[1].plot(time_steps, velocities[:, i], label=f'Joint {i+1}')
       axes[1].set_ylabel('Velocity (rad/s)')
       axes[1].set_title('Joint Velocities')
       axes[1].legend()
       axes[1].grid(True)
       
       # Torques
       for i in range(3):
           axes[2].plot(time_steps, torques[:, i], label=f'Joint {i+1}')
       axes[2].set_ylabel('Torque (N⋅m)')
       axes[2].set_xlabel('Time (s)')
       axes[2].set_title('Gravity Compensation Torques')
       axes[2].legend()
       axes[2].grid(True)
       
       plt.tight_layout()
       plt.show()

   # Run gravity compensation demo
   gravity_compensation_demo()

Forward and Inverse Dynamics
-----------------------------

Forward Dynamics
~~~~~~~~~~~~~~~~

Given torques, compute resulting accelerations:

.. code-block:: python

   # Define robot state and inputs
   theta = np.array([0.1, 0.3, -0.2])
   theta_dot = np.array([0.5, -0.3, 0.8])
   tau = np.array([10.0, 5.0, 2.0])  # Applied torques
   g = [0, 0, -9.81]  # Gravity
   F_ext = np.zeros(6)  # No external forces

   # Compute forward dynamics
   theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, g, F_ext)

   print(f"Applied torques: {tau}")
   print(f"Resulting accelerations: {theta_ddot}")

Inverse Dynamics
~~~~~~~~~~~~~~~~

Given desired motion, compute required torques:

.. code-block:: python

   # Define desired motion
   theta = np.array([0.1, 0.3, -0.2])
   theta_dot = np.array([0.5, -0.3, 0.8])
   theta_ddot_desired = np.array([1.0, -0.5, 0.3])  # Desired accelerations

   # Compute required torques
   tau_required = dynamics.inverse_dynamics(
       theta, theta_dot, theta_ddot_desired, g, F_ext
   )

   print(f"Desired accelerations: {theta_ddot_desired}")
   print(f"Required torques: {tau_required}")

   # Verify with forward dynamics
   theta_ddot_check = dynamics.forward_dynamics(
       theta, theta_dot, tau_required, g, F_ext
   )
   
   error = np.linalg.norm(theta_ddot_check - theta_ddot_desired)
   print(f"Verification error: {error:.6f}")

Trajectory Dynamics
~~~~~~~~~~~~~~~~~~~

Compute dynamics along a trajectory:

.. code-block:: python

   def trajectory_dynamics_analysis():
       """Analyze dynamics along a planned trajectory."""
       
       # Generate simple trajectory (sinusoidal motion)
       t_final = 5.0
       dt = 0.01
       time_steps = np.arange(0, t_final, dt)
       
       # Trajectory parameters
       amplitude = np.array([0.5, 0.3, 0.2])
       frequency = np.array([0.5, 0.8, 1.2])
       
       # Generate trajectory
       trajectory = []
       velocities = []
       accelerations = []
       
       for t in time_steps:
           # Position (sinusoidal)
           pos = amplitude * np.sin(2 * np.pi * frequency * t)
           
           # Velocity (derivative)
           vel = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
           
           # Acceleration (second derivative)
           acc = -amplitude * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * t)
           
           trajectory.append(pos)
           velocities.append(vel)
           accelerations.append(acc)
       
       trajectory = np.array(trajectory)
       velocities = np.array(velocities)
       accelerations = np.array(accelerations)
       
       # Compute required torques along trajectory
       torques = []
       for i, t in enumerate(time_steps):
           tau = dynamics.inverse_dynamics(
               trajectory[i], velocities[i], accelerations[i], 
               [0, 0, -9.81], np.zeros(6)
           )
           torques.append(tau)
       
       torques = np.array(torques)
       
       # Analyze results
       fig, axes = plt.subplots(2, 2, figsize=(12, 8))
       
       # Trajectory
       for j in range(3):
           axes[0, 0].plot(time_steps, np.degrees(trajectory[:, j]), label=f'Joint {j+1}')
       axes[0, 0].set_title('Joint Trajectories')
       axes[0, 0].set_ylabel('Position (degrees)')
       axes[0, 0].legend()
       axes[0, 0].grid(True)
       
       # Velocities
       for j in range(3):
           axes[0, 1].plot(time_steps, velocities[:, j], label=f'Joint {j+1}')
       axes[0, 1].set_title('Joint Velocities')
       axes[0, 1].set_ylabel('Velocity (rad/s)')
       axes[0, 1].legend()
       axes[0, 1].grid(True)
       
       # Accelerations
       for j in range(3):
           axes[1, 0].plot(time_steps, accelerations[:, j], label=f'Joint {j+1}')
       axes[1, 0].set_title('Joint Accelerations')
       axes[1, 0].set_ylabel('Acceleration (rad/s²)')
       axes[1, 0].set_xlabel('Time (s)')
       axes[1, 0].legend()
       axes[1, 0].grid(True)
       
       # Required torques
       for j in range(3):
           axes[1, 1].plot(time_steps, torques[:, j], label=f'Joint {j+1}')
       axes[1, 1].set_title('Required Torques')
       axes[1, 1].set_ylabel('Torque (N⋅m)')
       axes[1, 1].set_xlabel('Time (s)')
       axes[1, 1].legend()
       axes[1, 1].grid(True)
       
       plt.tight_layout()
       plt.show()
       
       # Print statistics
       max_torques = np.max(np.abs(torques), axis=0)
       print(f"Maximum torques required: {max_torques}")
       print(f"Peak total torque: {np.max(np.linalg.norm(torques, axis=1)):.2f} N⋅m")

   # Run trajectory analysis
   trajectory_dynamics_analysis()

Performance Optimization
-------------------------

GPU Acceleration
~~~~~~~~~~~~~~~~

For large-scale simulations, use GPU acceleration:

.. code-block:: python

   try:
       import cupy as cp
       
       def gpu_dynamics_demo():
           """Demonstrate GPU-accelerated dynamics computations."""
           
           # Generate batch of configurations
           n_configs = 1000
           configs = cp.random.uniform(-cp.pi, cp.pi, (n_configs, 3))
           
           # Measure performance
           import time
           
           # CPU computation
           start_time = time.time()
           cpu_results = []
           for i in range(100):  # Smaller sample for CPU
               config = cp.asnumpy(configs[i])
               M = dynamics.mass_matrix(config)
               cpu_results.append(M)
           cpu_time = time.time() - start_time
           
           print(f"CPU time for 100 configurations: {cpu_time:.3f} seconds")
           print(f"CPU rate: {100/cpu_time:.1f} computations/second")
           
           # For full GPU implementation, you would need CUDA kernels
           # This is a simplified example showing the concept
           
   except ImportError:
       print("CuPy not available - GPU acceleration not supported")

Parallel Processing
~~~~~~~~~~~~~~~~~~

Use multiprocessing for CPU parallelization:

.. code-block:: python

   from multiprocessing import Pool
   import functools

   def parallel_dynamics_computation(dynamics, configurations):
       """Compute dynamics for multiple configurations in parallel."""
       
       def compute_single_config(config):
           M = dynamics.mass_matrix(config)
           g_forces = dynamics.gravity_forces(config, [0, 0, -9.81])
           return {
               'config': config,
               'mass_matrix': M,
               'gravity_forces': g_forces,
               'condition_number': np.linalg.cond(M)
           }
       
       # Create partial function with dynamics object
       compute_func = functools.partial(compute_single_config)
       
       # Use multiprocessing
       with Pool() as pool:
           results = pool.map(compute_func, configurations)
       
       return results

   # Example usage
   test_configs = [
       np.array([0.1, 0.3, -0.2]),
       np.array([0.5, -0.2, 0.4]),
       np.array([-0.3, 0.6, 0.1]),
       np.array([0.8, -0.1, -0.3])
   ]

   # Parallel computation
   import time
   start_time = time.time()
   parallel_results = parallel_dynamics_computation(dynamics, test_configs)
   parallel_time = time.time() - start_time

   print(f"Parallel computation time: {parallel_time:.3f} seconds")
   print(f"Number of configurations processed: {len(parallel_results)}")

Advanced Topics
---------------

Energy Analysis
~~~~~~~~~~~~~~

Analyze kinetic and potential energy:

.. code-block:: python

   def energy_analysis(dynamics, theta, theta_dot, g=[0, 0, -9.81]):
       """Compute kinetic and potential energy of the robot."""
       
       # Kinetic energy: T = 0.5 * theta_dot^T * M * theta_dot
       M = dynamics.mass_matrix(theta)
       kinetic_energy = 0.5 * theta_dot.T @ M @ theta_dot
       
       # Potential energy (gravitational)
       # This is a simplified calculation - full implementation would
       # require link positions and masses
       potential_energy = 0.0  # Placeholder
       
       total_energy = kinetic_energy + potential_energy
       
       return {
           'kinetic': kinetic_energy,
           'potential': potential_energy,
           'total': total_energy
       }

   # Example usage
   energy = energy_analysis(dynamics, theta, theta_dot)
   print(f"Kinetic energy: {energy['kinetic']:.3f} J")
   print(f"Total energy: {energy['total']:.3f} J")

Linearization
~~~~~~~~~~~~

Linearize dynamics about an operating point:

.. code-block:: python

   def linearize_dynamics(dynamics, theta_0, theta_dot_0, epsilon=1e-6):
       """Linearize robot dynamics about operating point."""
       
       n = len(theta_0)
       
       # Compute nominal values
       M_0 = dynamics.mass_matrix(theta_0)
       c_0 = dynamics.velocity_quadratic_forces(theta_0, theta_dot_0)
       g_0 = dynamics.gravity_forces(theta_0, [0, 0, -9.81])
       
       # Compute Jacobians numerically
       dM_dtheta = np.zeros((n, n, n))
       dc_dtheta = np.zeros((n, n))
       dg_dtheta = np.zeros((n, n))
       
       for i in range(n):
           # Perturb theta
           theta_plus = theta_0.copy()
           theta_plus[i] += epsilon
           theta_minus = theta_0.copy()
           theta_minus[i] -= epsilon
           
           # Compute derivatives
           M_plus = dynamics.mass_matrix(theta_plus)
           M_minus = dynamics.mass_matrix(theta_minus)
           dM_dtheta[:, :, i] = (M_plus - M_minus) / (2 * epsilon)
           
           c_plus = dynamics.velocity_quadratic_forces(theta_plus, theta_dot_0)
           c_minus = dynamics.velocity_quadratic_forces(theta_minus, theta_dot_0)
           dc_dtheta[:, i] = (c_plus - c_minus) / (2 * epsilon)
           
           g_plus = dynamics.gravity_forces(theta_plus, [0, 0, -9.81])
           g_minus = dynamics.gravity_forces(theta_minus, [0, 0, -9.81])
           dg_dtheta[:, i] = (g_plus - g_minus) / (2 * epsilon)
       
       return {
           'M_0': M_0, 'c_0': c_0, 'g_0': g_0,
           'dM_dtheta': dM_dtheta, 'dc_dtheta': dc_dtheta, 'dg_dtheta': dg_dtheta
       }

   # Example usage
   operating_point = np.array([0.0, 0.0, 0.0])  # Home position
   operating_velocity = np.zeros(3)
   
   linearization = linearize_dynamics(dynamics, operating_point, operating_velocity)
   print(f"Mass matrix at operating point:\n{linearization['M_0']}")

Model Identification
~~~~~~~~~~~~~~~~~~~

Estimate robot parameters from experimental data:

.. code-block:: python

   def parameter_identification_demo():
       """Demonstrate robot parameter identification."""
       
       # Simulate noisy measurements
       np.random.seed(42)
       n_samples = 100
       
       # Generate test trajectories
       test_data = []
       for i in range(n_samples):
           theta = np.random.uniform(-0.5, 0.5, 3)
           theta_dot = np.random.uniform(-1.0, 1.0, 3)
           theta_ddot = np.random.uniform(-2.0, 2.0, 3)
           
           # Compute "measured" torques with noise
           tau_true = dynamics.inverse_dynamics(
               theta, theta_dot, theta_ddot, [0, 0, -9.81], np.zeros(6)
           )
           
           # Add measurement noise
           noise = np.random.normal(0, 0.1, 3)
           tau_measured = tau_true + noise
           
           test_data.append({
               'theta': theta,
               'theta_dot': theta_dot,
               'theta_ddot': theta_ddot,
               'tau_measured': tau_measured,
               'tau_true': tau_true
           })
       
       # Simple parameter identification (mass scaling factors)
       def objective_function(params):
           """Objective function for parameter identification."""
           mass_scale_factors = params
           
           total_error = 0.0
           for data in test_data:
               # Scale the inertia matrices
               scaled_Glist = []
               for i, G in enumerate(dynamics.Glist):
                   G_scaled = G.copy()
                   G_scaled[3:, 3:] *= mass_scale_factors[i]  # Scale mass components
                   scaled_Glist.append(G_scaled)
               
               # Create temporary dynamics with scaled parameters
               temp_dynamics = ManipulatorDynamics(
                   dynamics.M_list, dynamics.omega_list, dynamics.r_list,
                   dynamics.b_list, dynamics.S_list, dynamics.B_list,
                   scaled_Glist
               )
               
               # Compute predicted torques
               tau_predicted = temp_dynamics.inverse_dynamics(
                   data['theta'], data['theta_dot'], data['theta_ddot'],
                   [0, 0, -9.81], np.zeros(6)
               )
               
               # Accumulate error
               error = np.linalg.norm(tau_predicted - data['tau_measured'])
               total_error += error
           
           return total_error
       
       # Optimize parameters (simplified example)
       from scipy.optimize import minimize
       
       initial_guess = np.ones(len(dynamics.Glist))
       bounds = [(0.1, 10.0)] * len(dynamics.Glist)  # Mass can vary 10x
       
       result = minimize(objective_function, initial_guess, bounds=bounds)
       
       print(f"Identified mass scale factors: {result.x}")
       print(f"Optimization success: {result.success}")
       print(f"Final error: {result.fun:.3f}")

   # Run parameter identification
   parameter_identification_demo()

Manipulator Inertia Ellipsoid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize the manipulator's inertial characteristics:

.. code-block:: python

   def plot_inertia_ellipsoid(dynamics, theta):
       """Plot the manipulator inertia ellipsoid."""
       
       # Compute mass matrix
       M = dynamics.mass_matrix(theta)
       
       # Eigenvalue decomposition
       eigenvals, eigenvecs = np.linalg.eigh(M)
       
       # Create ellipsoid points
       u = np.linspace(0, 2 * np.pi, 50)
       v = np.linspace(0, np.pi, 25)
       
       # Unit sphere
       x_sphere = np.outer(np.cos(u), np.sin(v))
       y_sphere = np.outer(np.sin(u), np.sin(v))
       z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
       
       # Transform to ellipsoid
       points = np.stack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
       
       # Apply eigenvalue scaling and rotation
       if len(eigenvals) >= 3:
           scaling = np.diag(1.0 / np.sqrt(eigenvals[:3]))
           rotation = eigenvecs[:3, :3]
           transform = rotation @ scaling
           
           transformed_points = transform @ points
           
           x_ellipsoid = transformed_points[0].reshape(x_sphere.shape)
           y_ellipsoid = transformed_points[1].reshape(y_sphere.shape)
           z_ellipsoid = transformed_points[2].reshape(z_sphere.shape)
           
           # Plot
           fig = plt.figure(figsize=(10, 8))
           ax = fig.add_subplot(111, projection='3d')
           
           ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                          alpha=0.7, cmap='viridis')
           
           ax.set_xlabel('Inertia Direction 1')
           ax.set_ylabel('Inertia Direction 2')
           ax.set_zlabel('Inertia Direction 3')
           ax.set_title(f'Manipulator Inertia Ellipsoid at θ = {np.degrees(theta):.1f}°')
           
           plt.show()
           
           # Print eigenvalues
           print(f"Inertia eigenvalues: {eigenvals[:3]}")
           print(f"Condition number: {np.max(eigenvals[:3]) / np.min(eigenvals[:3]):.2f}")

   # Example usage
   plot_inertia_ellipsoid(dynamics, theta)

Dynamic Manipulability
~~~~~~~~~~~~~~~~~~~~~~

Analyze the robot's dynamic manipulability:

.. code-block:: python

   def dynamic_manipulability_analysis(dynamics, configurations):
       """Analyze dynamic manipulability across configurations."""
       
       manipulabilities = []
       condition_numbers = []
       
       for config in configurations:
           # Compute mass matrix
           M = dynamics.mass_matrix(config)
           
           # Compute Jacobian (assuming SerialManipulator interface)
           # J = robot.jacobian(config)  # This would need robot instance
           
           # For now, use mass matrix properties
           eigenvals = np.linalg.eigvals(M)
           
           # Dynamic manipulability (simplified)
           manip = np.prod(eigenvals) ** (1.0 / len(eigenvals))
           manipulabilities.append(manip)
           
           # Condition number
           cond_num = np.max(eigenvals) / np.min(eigenvals)
           condition_numbers.append(cond_num)
       
       return np.array(manipulabilities), np.array(condition_numbers)

   # Test across workspace
   test_configs = []
   for i in range(20):
       config = np.random.uniform(-np.pi/2, np.pi/2, 3)
       test_configs.append(config)

   manip_vals, cond_nums = dynamic_manipulability_analysis(dynamics, test_configs)
   
   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   ax1.plot(manip_vals, 'b-o')
   ax1.set_title('Dynamic Manipulability')
   ax1.set_ylabel('Manipulability Index')
   ax1.set_xlabel('Configuration Index')
   ax1.grid(True)
   
   ax2.plot(cond_nums, 'r-o')
   ax2.set_title('Mass Matrix Conditioning')
   ax2.set_ylabel('Condition Number')
   ax2.set_xlabel('Configuration Index')
   ax2.grid(True)
   
   plt.tight_layout()
   plt.show()

Real-World Applications
-----------------------

Robot Control Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate dynamics with control systems:

.. code-block:: python

   class DynamicsBasedController:
       """Example controller using robot dynamics."""
       
       def __init__(self, dynamics):
           self.dynamics = dynamics
           self.gravity = [0, 0, -9.81]
           self.integral_error = None
       
       def computed_torque_control(self, theta, theta_dot, theta_d, theta_dot_d, 
                                 theta_ddot_d, Kp, Kd):
           """Computed torque control using exact dynamics."""
           
           # Tracking errors
           e = theta_d - theta
           e_dot = theta_dot_d - theta_dot
           
           # Desired acceleration with feedback
           theta_ddot_cmd = theta_ddot_d + Kp @ e + Kd @ e_dot
           
           # Compute required torques using inverse dynamics
           tau = self.dynamics.inverse_dynamics(
               theta, theta_dot, theta_ddot_cmd, self.gravity, np.zeros(6)
           )
           
           return tau
       
       def pid_with_gravity_compensation(self, theta, theta_dot, theta_d, 
                                       Kp, Ki, Kd, dt):
           """PID control with gravity compensation."""
           
           # Initialize integral error if needed
           if self.integral_error is None:
               self.integral_error = np.zeros_like(theta)
           
           # Tracking errors
           e = theta_d - theta
           e_dot = -theta_dot  # Assuming zero desired velocity
           
           # Update integral error
           self.integral_error += e * dt
           
           # PID control law
           tau_pid = Kp @ e + Ki @ self.integral_error + Kd @ e_dot
           
           # Add gravity compensation
           tau_gravity = self.dynamics.gravity_forces(theta, self.gravity)
           
           return tau_pid + tau_gravity
       
       def adaptive_control(self, theta, theta_dot, theta_d, theta_dot_d,
                          theta_ddot_d, Kp, Kd, adaptation_gain=0.1):
           """Adaptive control with parameter uncertainty."""
           
           # This is a simplified adaptive controller
           # Real implementation would include parameter adaptation
           
           # Use nominal dynamics for feedforward
           tau_ff = self.dynamics.inverse_dynamics(
               theta_d, theta_dot_d, theta_ddot_d, self.gravity, np.zeros(6)
           )
           
           # Add feedback component
           e = theta_d - theta
           e_dot = theta_dot_d - theta_dot
           tau_fb = Kp @ e + Kd @ e_dot
           
           return tau_ff + tau_fb
       
       def impedance_control(self, theta, theta_dot, theta_d, F_ext, 
                           M_des, B_des, K_des):
           """Impedance control for interaction tasks."""
           
           # Desired impedance dynamics:
           # M_des * ddtheta + B_des * dtheta + K_des * (theta - theta_d) = F_ext
           
           # Compute desired acceleration
           e = theta - theta_d
           theta_ddot_d = np.linalg.solve(M_des, F_ext - B_des @ theta_dot - K_des @ e)
           
           # Use computed torque control to achieve desired acceleration
           tau = self.dynamics.inverse_dynamics(
               theta, theta_dot, theta_ddot_d, self.gravity, np.zeros(6)
           )
           
           return tau

   # Example usage
   controller = DynamicsBasedController(dynamics)

   # Control parameters
   Kp = np.diag([100, 80, 60])  # Proportional gains
   Kd = np.diag([20, 15, 10])   # Derivative gains
   Ki = np.diag([5, 4, 3])      # Integral gains

   # Desired trajectory point
   theta_desired = np.array([0.5, 0.3, -0.2])
   theta_dot_desired = np.array([0.1, -0.05, 0.08])
   theta_ddot_desired = np.array([0.0, 0.0, 0.0])

   # Current state
   theta_current = np.array([0.4, 0.25, -0.15])
   theta_dot_current = np.array([0.08, -0.03, 0.06])

   # Compute control torques
   tau_control = controller.computed_torque_control(
       theta_current, theta_dot_current,
       theta_desired, theta_dot_desired, theta_ddot_desired,
       Kp, Kd
   )

   print(f"Control torques: {tau_control}")

Simulation Integration
~~~~~~~~~~~~~~~~~~~~~

Integrate with physics simulators:

.. code-block:: python

   def complete_simulation_example():
       """Complete simulation example with dynamics and control."""
       
       # Simulation parameters
       dt = 0.001  # 1ms time step for stability
       t_final = 3.0
       time_steps = np.arange(0, t_final, dt)
       
       # Initial conditions
       theta = np.array([0.1, 0.2, -0.1])
       theta_dot = np.zeros(3)
       
       # Control parameters
       controller = DynamicsBasedController(dynamics)
       Kp = np.diag([100, 80, 60])
       Ki = np.diag([10, 8, 6])
       Kd = np.diag([20, 15, 10])
       
       # Target trajectory (figure-8 in joint space)
       def target_trajectory(t):
           amplitude = np.array([0.3, 0.2, 0.15])
           freq1 = 0.5
           freq2 = 1.0
           
           theta_d = amplitude * np.array([
               np.sin(2 * np.pi * freq1 * t),
               np.sin(2 * np.pi * freq2 * t),
               np.sin(2 * np.pi * freq1 * t) * np.cos(2 * np.pi * freq2 * t)
           ])
           
           theta_dot_d = amplitude * np.array([
               2 * np.pi * freq1 * np.cos(2 * np.pi * freq1 * t),
               2 * np.pi * freq2 * np.cos(2 * np.pi * freq2 * t),
               2 * np.pi * freq1 * np.cos(2 * np.pi * freq1 * t) * np.cos(2 * np.pi * freq2 * t) -
               2 * np.pi * freq2 * np.sin(2 * np.pi * freq1 * t) * np.sin(2 * np.pi * freq2 * t)
           ])
           
           return theta_d, theta_dot_d
       
       # Storage for results
       positions = []
       velocities = []
       accelerations = []
       torques = []
       errors = []
       desired_positions = []
       
       # Simulation loop
       for i, t in enumerate(time_steps):
           # Get desired trajectory
           theta_d, theta_dot_d = target_trajectory(t)
           
           # Compute control torques
           tau_control = controller.pid_with_gravity_compensation(
               theta, theta_dot, theta_d, Kp, Ki, Kd, dt
           )
           
           # Add small disturbances (realistic simulation)
           disturbance = np.random.normal(0, 0.01, 3) if i % 100 == 0 else np.zeros(3)
           tau_total = tau_control + disturbance
           
           # Apply actuator limits (realistic)
           tau_max = np.array([50, 40, 30])  # Maximum torques
           tau_total = np.clip(tau_total, -tau_max, tau_max)
           
           # Forward dynamics
           theta_ddot = dynamics.forward_dynamics(
               theta, theta_dot, tau_total, [0, 0, -9.81], np.zeros(6)
           )
           
           # Numerical integration (4th-order Runge-Kutta for accuracy)
           def dynamics_rhs(state, tau):
               th, th_dot = state[:3], state[3:]
               th_ddot = dynamics.forward_dynamics(th, th_dot, tau, [0, 0, -9.81], np.zeros(6))
               return np.concatenate([th_dot, th_ddot])
           
           # Current state
           state = np.concatenate([theta, theta_dot])
           
           # RK4 integration
           k1 = dynamics_rhs(state, tau_total)
           k2 = dynamics_rhs(state + 0.5 * dt * k1, tau_total)
           k3 = dynamics_rhs(state + 0.5 * dt * k2, tau_total)
           k4 = dynamics_rhs(state + dt * k3, tau_total)
           
           state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
           
           theta = state_new[:3]
           theta_dot = state_new[3:]
           
           # Apply joint limits
           theta_limits = np.array([[-np.pi, np.pi], [-np.pi/2, np.pi/2], [-np.pi, np.pi]])
           for j in range(3):
               if theta[j] < theta_limits[j, 0]:
                   theta[j] = theta_limits[j, 0]
                   theta_dot[j] = 0  # Stop at limit
               elif theta[j] > theta_limits[j, 1]:
                   theta[j] = theta_limits[j, 1]
                   theta_dot[j] = 0  # Stop at limit
           
           # Store results
           positions.append(theta.copy())
           velocities.append(theta_dot.copy())
           accelerations.append(theta_ddot.copy())
           torques.append(tau_total.copy())
           desired_positions.append(theta_d.copy())
           
           # Compute tracking error
           error = np.linalg.norm(theta - theta_d)
           errors.append(error)
       
       # Convert to arrays
       positions = np.array(positions)
       velocities = np.array(velocities)
       accelerations = np.array(accelerations)
       torques = np.array(torques)
       desired_positions = np.array(desired_positions)
       errors = np.array(errors)
       
       # Plot comprehensive results
       fig, axes = plt.subplots(2, 3, figsize=(15, 10))
       
       # Joint positions vs desired
       for i in range(3):
           axes[0, 0].plot(time_steps, np.degrees(positions[:, i]), 
                          label=f'Joint {i+1} Actual', linewidth=2)
           axes[0, 0].plot(time_steps, np.degrees(desired_positions[:, i]), 
                          '--', label=f'Joint {i+1} Desired', alpha=0.7)
       axes[0, 0].set_ylabel('Position (degrees)')
       axes[0, 0].set_title('Joint Positions')
       axes[0, 0].legend()
       axes[0, 0].grid(True)
       
       # Joint velocities
       for i in range(3):
           axes[0, 1].plot(time_steps, velocities[:, i], label=f'Joint {i+1}')
       axes[0, 1].set_ylabel('Velocity (rad/s)')
       axes[0, 1].set_title('Joint Velocities')
       axes[0, 1].legend()
       axes[0, 1].grid(True)
       
       # Control torques
       for i in range(3):
           axes[0, 2].plot(time_steps, torques[:, i], label=f'Joint {i+1}')
       axes[0, 2].set_ylabel('Torque (N⋅m)')
       axes[0, 2].set_title('Control Torques')
       axes[0, 2].legend()
       axes[0, 2].grid(True)
       
       # Tracking error
       axes[1, 0].plot(time_steps, np.degrees(errors), 'r-', linewidth=2)
       axes[1, 0].set_ylabel('Error (degrees)')
       axes[1, 0].set_xlabel('Time (s)')
       axes[1, 0].set_title('Tracking Error')
       axes[1, 0].grid(True)
       
       # Joint accelerations
       for i in range(3):
           axes[1, 1].plot(time_steps, accelerations[:, i], label=f'Joint {i+1}')
       axes[1, 1].set_ylabel('Acceleration (rad/s²)')
       axes[1, 1].set_xlabel('Time (s)')
       axes[1, 1].set_title('Joint Accelerations')
       axes[1, 1].legend()
       axes[1, 1].grid(True)
       
       # Phase plot (position vs velocity for first joint)
       axes[1, 2].plot(np.degrees(positions[:, 0]), velocities[:, 0], 'b-', alpha=0.7)
       axes[1, 2].set_xlabel('Joint 1 Position (degrees)')
       axes[1, 2].set_ylabel('Joint 1 Velocity (rad/s)')
       axes[1, 2].set_title('Phase Plot (Joint 1)')
       axes[1, 2].grid(True)
       
       plt.tight_layout()
       plt.show()
       
       # Print performance statistics
       final_error = np.degrees(errors[-1])
       max_error = np.degrees(np.max(errors))
       rms_error = np.degrees(np.sqrt(np.mean(errors**2)))
       
       print(f"\nSimulation Performance:")
       print(f"Final tracking error: {final_error:.2f} degrees")
       print(f"Maximum tracking error: {max_error:.2f} degrees")
       print(f"RMS tracking error: {rms_error:.2f} degrees")
       
       max_torques = np.max(np.abs(torques), axis=0)
       print(f"Maximum torques used: {max_torques} N⋅m")
       
       return {
           'time': time_steps,
           'positions': positions,
           'velocities': velocities,
           'torques': torques,
           'errors': errors
       }

   # Run complete simulation
   simulation_results = complete_simulation_example()

Force and Torque Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze required forces and torques for different tasks:

.. code-block:: python

   def force_torque_analysis():
       """Analyze force and torque requirements for different tasks."""
       
       # Define task scenarios
       scenarios = {
           'hover': {
               'theta': np.array([0.0, np.pi/4, -np.pi/4]),
               'theta_dot': np.zeros(3),
               'theta_ddot': np.zeros(3),
               'description': 'Static hovering against gravity'
           },
           'fast_motion': {
               'theta': np.array([0.2, 0.3, -0.1]),
               'theta_dot': np.array([2.0, -1.5, 1.0]),
               'theta_ddot': np.array([3.0, -2.0, 2.5]),
               'description': 'Fast dynamic motion'
           },
           'external_force': {
               'theta': np.array([0.1, 0.4, -0.2]),
               'theta_dot': np.array([0.5, -0.3, 0.2]),
               'theta_ddot': np.array([0.0, 0.0, 0.0]),
               'F_ext': np.array([10, 5, -8, 0, 0, 2]),
               'description': 'Interacting with environment'
           }
       }
       
       print("Force and Torque Analysis")
       print("=" * 50)
       
       for name, scenario in scenarios.items():
           print(f"\nScenario: {scenario['description']}")
           print("-" * 30)
           
           # Extract parameters
           theta = scenario['theta']
           theta_dot = scenario['theta_dot']
           theta_ddot = scenario['theta_ddot']
           F_ext = scenario.get('F_ext', np.zeros(6))
           
           # Compute required torques
           tau_total = dynamics.inverse_dynamics(
               theta, theta_dot, theta_ddot, [0, 0, -9.81], F_ext
           )
           
           # Break down torque components
           tau_inertial = dynamics.mass_matrix(theta) @ theta_ddot
           tau_coriolis = dynamics.velocity_quadratic_forces(theta, theta_dot)
           tau_gravity = dynamics.gravity_forces(theta, [0, 0, -9.81])
           
           # Compute external force contribution
           if hasattr(dynamics, 'jacobian'):
               # This would need robot instance - simplified here
               tau_external = np.zeros(3)  # Placeholder
           else:
               tau_external = np.zeros(3)
           
           print(f"Joint angles: {np.degrees(theta):.1f} degrees")
           print(f"Joint velocities: {theta_dot:.2f} rad/s")
           print(f"Joint accelerations: {theta_ddot:.2f} rad/s²")
           print(f"External forces: {F_ext[:3]:.1f} N, {F_ext[3:]:.2f} N⋅m")
           print()
           print("Torque breakdown:")
           print(f"  Inertial:   {tau_inertial:.2f} N⋅m")
           print(f"  Coriolis:   {tau_coriolis:.2f} N⋅m")
           print(f"  Gravity:    {tau_gravity:.2f} N⋅m")
           print(f"  External:   {tau_external:.2f} N⋅m")
           print(f"  Total:      {tau_total:.2f} N⋅m")
           print(f"  Magnitude:  {np.linalg.norm(tau_total):.2f} N⋅m")
           
           # Check against typical actuator limits
           actuator_limits = np.array([100, 80, 60])  # Typical limits
           safety_factor = tau_total / actuator_limits
           
           if np.any(np.abs(safety_factor) > 0.8):
               print(f"  ⚠️  High torque demand: {np.max(np.abs(safety_factor)):.1%} of limits")
           else:
               print(f"  ✅ Safe torque levels: {np.max(np.abs(safety_factor)):.1%} of limits")

   # Run force/torque analysis
   force_torque_analysis()

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~

**Numerical Instability**

.. code-block:: python

   def check_numerical_stability(dynamics, theta):
       """Check for potential numerical issues."""
       
       print("Numerical Stability Check")
       print("=" * 40)
       
       M = dynamics.mass_matrix(theta)
       
       # Check condition number
       cond_num = np.linalg.cond(M)
       print(f"Mass matrix condition number: {cond_num:.2e}")
       
       if cond_num > 1e12:
           print("⚠️ Warning: Poor conditioning - near singularity")
           print("   Recommendations:")
           print("   - Check robot configuration")
           print("   - Verify joint limits")
           print("   - Consider regularization")
       elif cond_num > 1e6:
           print("⚠️ Caution: Moderate conditioning issues")
       else:
           print("✅ Good conditioning")
       
       # Check positive definiteness
       eigenvals = np.linalg.eigvals(M)
       min_eigenval = np.min(eigenvals)
       print(f"Minimum eigenvalue: {min_eigenval:.6f}")
       
       if min_eigenval <= 0:
           print("❌ Error: Mass matrix not positive definite")
           print("   This indicates a fundamental problem with dynamics")
       elif min_eigenval < 1e-6:
           print("⚠️ Warning: Very small eigenvalue - check for singularities")
       else:
           print("✅ Matrix is positive definite")
       
       # Check symmetry
       symmetry_error = np.max(np.abs(M - M.T))
       print(f"Symmetry error: {symmetry_error:.2e}")
       
       if symmetry_error > 1e-10:
           print("⚠️ Warning: Mass matrix not symmetric")
           print("   This may indicate implementation errors")
       else:
           print("✅ Matrix is symmetric")
       
       # Check for NaN or infinite values
       if not np.all(np.isfinite(M)):
           print("❌ Error: Mass matrix contains NaN or infinite values")
           print("   Check input angles and robot parameters")
       else:
           print("✅ All matrix elements are finite")

   # Check current configuration
   check_numerical_stability(dynamics, theta)

**Performance Issues**

.. code-block:: python

   def performance_diagnostics():
       """Diagnose performance issues in dynamics computations."""
       
       import time
       
       # Test configurations
       test_configs = [
           np.array([0.0, 0.0, 0.0]),      # Home position
           np.array([0.5, 0.3, -0.2]),     # Random position
           np.array([1.2, -0.8, 0.9]),     # Near limits
           np.array([-0.7, 1.1, 0.4])      # Another configuration
       ]
       
       print("Performance Diagnostics")
       print("=" * 40)
       
       for i, config in enumerate(test_configs):
           print(f"\nConfiguration {i+1}: {np.degrees(config):.1f}°")
           
           # Mass matrix computation
           start = time.time()
           for _ in range(100):
               M = dynamics.mass_matrix(config)
           mass_time = (time.time() - start) / 100
           
           # Velocity forces computation
           theta_dot = np.array([0.1, -0.2, 0.3])
           start = time.time()
           for _ in range(100):
               c = dynamics.velocity_quadratic_forces(config, theta_dot)
           velocity_time = (time.time() - start) / 100
           
           # Gravity forces computation
           start = time.time()
           for _ in range(100):
               g = dynamics.gravity_forces(config, [0, 0, -9.81])
           gravity_time = (time.time() - start) / 100
           
           print(f"  Mass matrix:     {mass_time*1000:.2f} ms")
           print(f"  Velocity forces: {velocity_time*1000:.2f} ms")
           print(f"  Gravity forces:  {gravity_time*1000:.2f} ms")
           print(f"  Total per step:  {(mass_time + velocity_time + gravity_time)*1000:.2f} ms")
           
           # Check if performance is acceptable
           total_time = mass_time + velocity_time + gravity_time
           if total_time > 0.001:  # 1ms threshold
               print(f"  ⚠️ Slow computation: {total_time*1000:.1f} ms > 1 ms")
           else:
               print(f"  ✅ Good performance: {total_time*1000:.1f} ms")

   # Run performance diagnostics
   performance_diagnostics()

**Memory Issues**

.. code-block:: python

   def memory_usage_analysis():
       """Analyze memory usage in dynamics computations."""
       
       import tracemalloc
       
       print("Memory Usage Analysis")
       print("=" * 40)
       
       # Start memory tracing
       tracemalloc.start()
       
       # Baseline memory
       snapshot1 = tracemalloc.take_snapshot()
       
       # Compute dynamics for many configurations
       configs = [np.random.uniform(-np.pi, np.pi, 3) for _ in range(100)]
       
       for config in configs:
           M = dynamics.mass_matrix(config)
           c = dynamics.velocity_quadratic_forces(config, np.zeros(3))
           g = dynamics.gravity_forces(config, [0, 0, -9.81])
       
       # Check memory after computations
       snapshot2 = tracemalloc.take_snapshot()
       
       # Stop tracing
       tracemalloc.stop()
       
       # Analyze differences
       top_stats = snapshot2.compare_to(snapshot1, 'lineno')
       
       print("Top memory allocations:")
       for index, stat in enumerate(top_stats[:3], 1):
           print(f"{index}. {stat}")
       
       # Check for memory leaks
       total_size = sum(stat.size for stat in snapshot2.statistics('filename'))
       print(f"\nTotal memory used: {total_size / 1024 / 1024:.1f} MB")
       
       if total_size > 100 * 1024 * 1024:  # 100 MB threshold
           print("⚠️ High memory usage detected")
       else:
           print("✅ Memory usage is reasonable")

   # Run memory analysis
   memory_usage_analysis()

Best Practices
--------------

Code Organization
~~~~~~~~~~~~~~~~

.. code-block:: python

       """Best practice organization for dynamics computations."""
       
       def __init__(self, dynamics_instance):
           self.dynamics = dynamics_instance
           self.cache_enabled = True
           self.performance_monitor = True
           
       def compute_full_dynamics(self, theta, theta_dot, theta_ddot, g, F_ext):
           """Compute all dynamics components efficiently."""
           
           # Pre-compute commonly used quantities
           M = self.dynamics.mass_matrix(theta)
           c = self.dynamics.velocity_quadratic_forces(theta, theta_dot)
           g_forces = self.dynamics.gravity_forces(theta, g)
           J_T = self.dynamics.jacobian(theta).T
           
           # External force contribution
           tau_ext = J_T @ F_ext
           
           # Inverse dynamics
           tau_required = M @ theta_ddot + c + g_forces + tau_ext
           
           # Forward dynamics (for verification)
           tau_net = tau_required - c - g_forces - tau_ext
           theta_ddot_verify = np.linalg.solve(M, tau_net)
           
           return {
               'mass_matrix': M,
               'coriolis_forces': c,
               'gravity_forces': g_forces,
               'external_torques': tau_ext,
               'required_torques': tau_required,
               'computed_acceleration': theta_ddot_verify,
               'verification_error': np.linalg.norm(theta_ddot - theta_ddot_verify)
           }

   # Example usage
   dynamics_manager = RobotDynamicsManager(dynamics)
   
   # Test configuration
   theta = np.array([0.1, 0.3, -0.2])
   theta_dot = np.array([0.5, -0.3, 0.8])
   theta_ddot = np.array([1.0, -0.5, 0.3])
   g = [0, 0, -9.81]
   F_ext = np.array([2.0, 1.0, -5.0, 0.1, 0.2, 0.0])
   
   results = dynamics_manager.compute_full_dynamics(theta, theta_dot, theta_ddot, g, F_ext)
   print(f"Verification error: {results['verification_error']:.6f}")

Error Handling
~~~~~~~~~~~~~

.. code-block:: python

   def robust_dynamics_computation(dynamics, theta, theta_dot=None, theta_ddot=None):
       """Robust dynamics computation with error handling."""
       
       try:
           # Validate inputs
           theta = np.array(theta)
           if theta_dot is None:
               theta_dot = np.zeros_like(theta)
           if theta_ddot is None:
               theta_ddot = np.zeros_like(theta)
           
           theta_dot = np.array(theta_dot)
           theta_ddot = np.array(theta_ddot)
           
           # Check dimensions
           if not (len(theta) == len(theta_dot) == len(theta_ddot)):
               raise ValueError("Inconsistent vector dimensions")
           
           # Check for valid values
           if not np.all(np.isfinite(theta)):
               raise ValueError("Invalid joint angles (NaN or infinite)")
           
           # Compute dynamics safely
           M = dynamics.mass_matrix(theta)
           
           # Check mass matrix properties
           if np.linalg.cond(M) > 1e12:
               print("Warning: Mass matrix is poorly conditioned")
           
           c = dynamics.velocity_quadratic_forces(theta, theta_dot)
           g_forces = dynamics.gravity_forces(theta, [0, 0, -9.81])
           
           return {
               'success': True,
               'mass_matrix': M,
               'coriolis_forces': c,
               'gravity_forces': g_forces,
               'condition_number': np.linalg.cond(M)
           }
           
       except np.linalg.LinAlgError as e:
           return {
               'success': False,
               'error': f"Linear algebra error: {e}",
               'recommendation': "Check robot configuration for singularities"
           }
       except ValueError as e:
           return {
               'success': False,
               'error': f"Input validation error: {e}",
               'recommendation': "Verify input dimensions and values"
           }
       except Exception as e:
           return {
               'success': False,
               'error': f"Unexpected error: {e}",
               'recommendation': "Contact support with robot parameters"
           }

   # Example usage with error handling
   result = robust_dynamics_computation(dynamics, theta, theta_dot)
   if result['success']:
       print("Dynamics computation successful")
       print(f"Condition number: {result['condition_number']:.2e}")
   else:
       print(f"Error: {result['error']}")
       print(f"Recommendation: {result['recommendation']}")

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def example_dynamics_function(dynamics, theta, theta_dot, control_gains):
       """
       Example function demonstrating proper documentation for dynamics operations.
       
       This function computes robot dynamics and applies control law for trajectory tracking.
       It serves as a template for documenting dynamics-related functions in ManipulaPy.
       
       Parameters
       ----------
       dynamics : ManipulatorDynamics
           Instance of ManipulatorDynamics class containing robot parameters
       theta : np.ndarray, shape (n,)
           Current joint angles in radians
       theta_dot : np.ndarray, shape (n,)
           Current joint velocities in rad/s
       control_gains : dict
           Dictionary containing control gains:
           - 'Kp': np.ndarray, shape (n,n) - Proportional gain matrix
           - 'Kd': np.ndarray, shape (n,n) - Derivative gain matrix
           
       Returns
       -------
       dict
           Dictionary containing:
           - 'torques': np.ndarray, shape (n,) - Required joint torques in N⋅m
           - 'energy': float - Total kinetic energy in Joules
           - 'power': float - Instantaneous power in Watts
           
       Raises
       ------
       ValueError
           If input dimensions are inconsistent
       np.linalg.LinAlgError
           If mass matrix is singular
           
       Examples
       --------
       >>> # Setup robot dynamics
       >>> dynamics = ManipulatorDynamics(...)
       >>> theta = np.array([0.1, 0.3, -0.2])
       >>> theta_dot = np.array([0.5, -0.3, 0.8])
       >>> gains = {'Kp': np.diag([100, 80, 60]), 'Kd': np.diag([20, 15, 10])}
       >>> 
       >>> # Compute dynamics
       >>> result = example_dynamics_function(dynamics, theta, theta_dot, gains)
       >>> print(f"Required torques: {result['torques']}")
       >>> print(f"Kinetic energy: {result['energy']:.3f} J")
       
       Notes
       -----
       This function assumes:
       - Robot is operating in Earth gravity (9.81 m/s²)
       - No external forces applied to end-effector
       - Joint limits are respected in input configuration
       
       For high-frequency control applications, consider caching the mass matrix
       computation to improve performance.
       
       References
       ----------
       .. [1] Murray, R. M., Li, Z., & Sastry, S. S. (1994). A mathematical 
              introduction to robotic manipulation. CRC press.
       .. [2] Lynch, K. M., & Park, F. C. (2017). Modern robotics. Cambridge 
              University Press.
       """
       
       # Input validation
       theta = np.asarray(theta)
       theta_dot = np.asarray(theta_dot)
       
       if theta.shape != theta_dot.shape:
           raise ValueError("theta and theta_dot must have same shape")
       
       # Extract control gains
       Kp = control_gains['Kp']
       Kd = control_gains['Kd']
       
       # Compute dynamics
       M = dynamics.mass_matrix(theta)
       c = dynamics.velocity_quadratic_forces(theta, theta_dot)
       g_forces = dynamics.gravity_forces(theta, [0, 0, -9.81])
       
       # Simple PD control with gravity compensation
       theta_desired = np.zeros_like(theta)  # For this example
       error = theta_desired - theta
       error_dot = -theta_dot  # Assuming zero desired velocity
       
       tau_control = Kp @ error + Kd @ error_dot
       tau_total = tau_control + g_forces
       
       # Compute energy and power
       kinetic_energy = 0.5 * theta_dot.T @ M @ theta_dot
       power = tau_total.T @ theta_dot
       
       return {
           'torques': tau_total,
           'energy': kinetic_energy,
           'power': power
       }

Testing and Validation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_dynamics_implementation(dynamics):
       """
       Comprehensive validation of dynamics implementation.
       
       This function performs various tests to ensure the dynamics
       implementation is correct and numerically stable.
       """
       
       print("Dynamics Implementation Validation")
       print("=" * 50)
       
       # Test configurations
       test_configs = [
           np.zeros(3),                    # Home position
           np.array([0.1, 0.3, -0.2]),   # Random configuration
           np.array([np.pi/4, -np.pi/6, np.pi/3]),  # Larger angles
       ]
       
       validation_results = {}
       
       for i, theta in enumerate(test_configs):
           print(f"\nTest Configuration {i+1}: {np.degrees(theta)} degrees")
           print("-" * 30)
           
           try:
               # Test 1: Mass matrix properties
               M = dynamics.mass_matrix(theta)
               
               # Symmetry test
               symmetry_error = np.max(np.abs(M - M.T))
               is_symmetric = symmetry_error < 1e-10
               print(f"Mass matrix symmetric: {is_symmetric} (error: {symmetry_error:.2e})")
               
               # Positive definiteness test
               eigenvals = np.linalg.eigvals(M)
               is_positive_definite = np.all(eigenvals > 0)
               min_eigenval = np.min(eigenvals)
               print(f"Positive definite: {is_positive_definite} (min eigenval: {min_eigenval:.6f})")
               
               # Condition number test
               cond_num = np.linalg.cond(M)
               is_well_conditioned = cond_num < 1e6
               print(f"Well conditioned: {is_well_conditioned} (cond: {cond_num:.2e})")
               
               # Test 2: Velocity forces at zero velocity
               c_zero = dynamics.velocity_quadratic_forces(theta, np.zeros_like(theta))
               c_zero_norm = np.linalg.norm(c_zero)
               velocity_test_passed = c_zero_norm < 1e-10
               print(f"Zero velocity forces at rest: {velocity_test_passed} (norm: {c_zero_norm:.2e})")
               
               # Test 3: Consistency between forward and inverse dynamics
               theta_dot = np.array([0.1, -0.2, 0.15])
               theta_ddot_desired = np.array([0.5, -0.3, 0.2])
               
               # Compute required torques
               tau = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot_desired, 
                                             [0, 0, -9.81], np.zeros(6))
               
               # Compute resulting accelerations
               theta_ddot_computed = dynamics.forward_dynamics(theta, theta_dot, tau,
                                                             [0, 0, -9.81], np.zeros(6))
               
               consistency_error = np.linalg.norm(theta_ddot_desired - theta_ddot_computed)
               consistency_test_passed = consistency_error < 1e-6
               print(f"Forward/inverse consistency: {consistency_test_passed} (error: {consistency_error:.2e})")
               
               # Store results
               validation_results[f"config_{i+1}"] = {
                   'symmetric': is_symmetric,
                   'positive_definite': is_positive_definite,
                   'well_conditioned': is_well_conditioned,
                   'zero_velocity_test': velocity_test_passed,
                   'consistency_test': consistency_test_passed,
                   'condition_number': cond_num,
                   'symmetry_error': symmetry_error,
                   'consistency_error': consistency_error
               }
               
           except Exception as e:
               print(f"❌ Error in configuration {i+1}: {e}")
               validation_results[f"config_{i+1}"] = {'error': str(e)}
       
       # Overall assessment
       print(f"\nOverall Assessment")
       print("=" * 30)
       
       all_tests_passed = True
       for config_name, results in validation_results.items():
           if 'error' in results:
               all_tests_passed = False
               continue
               
           config_passed = all([
               results['symmetric'],
               results['positive_definite'],
               results['well_conditioned'],
               results['zero_velocity_test'],
               results['consistency_test']
           ])
           
           if not config_passed:
               all_tests_passed = False
       
       if all_tests_passed:
           print("✅ All validation tests passed!")
           print("   The dynamics implementation appears correct.")
       else:
           print("❌ Some validation tests failed.")
           print("   Please review the implementation and robot parameters.")
       
       return validation_results

   # Run validation
   validation_results = validate_dynamics_implementation(dynamics)

ManipulatorDynamics Class Reference
----------------------------------

The ManipulatorDynamics Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `ManipulatorDynamics` class is the core component for robot dynamics computations in ManipulaPy. It inherits from `SerialManipulator` and adds dynamics-specific functionality.

.. code-block:: python

   from ManipulaPy.dynamics import ManipulatorDynamics

Class Initialization
~~~~~~~~~~~~~~~~~~~~

The ManipulatorDynamics class is initialized with the robot's kinematic and dynamic parameters:

.. code-block:: python

   def __init__(self, M_list, omega_list, r_list, b_list, S_list, B_list, Glist):
       """
       Initialize ManipulatorDynamics instance.
       
       Parameters
       ----------
       M_list : np.ndarray, shape (4, 4)
           Home configuration matrix of the end-effector
       omega_list : np.ndarray, shape (3, n) 
           Joint rotation axes in space frame
       r_list : np.ndarray, shape (n, 3)
           Points on joint axes in space frame
       b_list : np.ndarray, shape (n, 3)
           Points on joint axes in body frame
       S_list : np.ndarray, shape (6, n)
           Screw axes in space frame
       B_list : np.ndarray, shape (6, n) 
           Screw axes in body frame
       Glist : list of np.ndarray, each shape (6, 6)
           Spatial inertia matrices for each link
       """

Key Methods
~~~~~~~~~~~

**mass_matrix(thetalist)**

Computes the robot's mass/inertia matrix for a given configuration:

.. code-block:: python

   M = dynamics.mass_matrix(theta)
   # Returns: np.ndarray, shape (n, n) - Mass matrix

**velocity_quadratic_forces(thetalist, dthetalist)**

Computes Coriolis and centrifugal forces:

.. code-block:: python

   c = dynamics.velocity_quadratic_forces(theta, theta_dot)
   # Returns: np.ndarray, shape (n,) - Velocity-dependent forces

**gravity_forces(thetalist, g)**

Computes gravitational forces acting on the robot:

.. code-block:: python

   g_forces = dynamics.gravity_forces(theta, [0, 0, -9.81])
   # Returns: np.ndarray, shape (n,) - Gravity torques

**inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip)**

Computes required joint torques for desired motion:

.. code-block:: python

   tau = dynamics.inverse_dynamics(theta, theta_dot, theta_ddot, g, F_ext)
   # Returns: np.ndarray, shape (n,) - Required joint torques

**forward_dynamics(thetalist, dthetalist, taulist, g, Ftip)**

Computes resulting accelerations from applied torques:

.. code-block:: python

   theta_ddot = dynamics.forward_dynamics(theta, theta_dot, tau, g, F_ext)
   # Returns: np.ndarray, shape (n,) - Joint accelerations

Implementation Details
~~~~~~~~~~~~~~~~~~~~~

**Caching System**

The class implements an intelligent caching system for mass matrix computations:

.. code-block:: python

   def mass_matrix(self, thetalist):
       thetalist_key = tuple(thetalist)
       if thetalist_key in self._mass_matrix_cache:
           return self._mass_matrix_cache[thetalist_key]
       
       # Compute mass matrix...
       M = self._compute_mass_matrix(thetalist)
       
       self._mass_matrix_cache[thetalist_key] = M
       return M

**Numerical Methods**

The velocity_quadratic_forces method uses numerical differentiation:

.. code-block:: python

   def partial_derivative(self, i, j, k, thetalist):
       epsilon = 1e-6
       thetalist_plus = np.array(thetalist)
       thetalist_plus[k] += epsilon
       M_plus = self.mass_matrix(thetalist_plus)

       thetalist_minus = np.array(thetalist)
       thetalist_minus[k] -= epsilon
       M_minus = self.mass_matrix(thetalist_minus)

       return (M_plus[i, j] - M_minus[i, j]) / (2 * epsilon)

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~

The implementation is based on the Newton-Euler formulation of robot dynamics:

**Equation of Motion**

.. math::

   \tau = M(\theta)\ddot{\theta} + C(\theta,\dot{\theta})\dot{\theta} + G(\theta) + J^T(\theta)F_{ext}

Where:
- :math:`M(\theta)` is computed using the articulated body algorithm
- :math:`C(\theta,\dot{\theta})` uses Christoffel symbols 
- :math:`G(\theta)` considers gravitational potential energy
- :math:`J^T(\theta)F_{ext}` maps external forces to joint space

**Mass Matrix Computation**

The mass matrix is computed using the articulated body method:

.. math::

   M_{ij} = \text{Tr}(Ad_{T_{0j}}^T G_j Ad_{T_{0j}} S_i S_j^T)

Where:
- :math:`Ad_{T_{0j}}` is the adjoint transformation from base to link j
- :math:`G_j` is the spatial inertia matrix of link j  
- :math:`S_i` is the screw axis of joint i

Summary
-------

The ManipulaPy dynamics module provides a comprehensive implementation of robot dynamics computations. Key features include:

- **Efficient mass matrix computation** with caching

- **Accurate Coriolis force calculation** using numerical methods

- **Gravity compensation** for various orientations

- **Forward and inverse dynamics** for control applications

- **Numerical stability** checks and error handling

- **Performance optimization** for real-time applications

The module is designed for both educational and practical applications, providing clear mathematical foundations while maintaining computational efficiency suitable for real-time control systems.

For advanced users, the module supports GPU acceleration, parallel processing, and custom optimization techniques for high-performance robotics applications.