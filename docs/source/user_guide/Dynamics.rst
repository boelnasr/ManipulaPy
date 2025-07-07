Dynamics User Guide
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

Combined velocity‐dependent forces

.. math::
   C(\boldsymbol\theta,\dot{\boldsymbol\theta})\,\dot{\boldsymbol\theta}
     = \begin{bmatrix}
         \sum_{j,k=1}^{n}\Gamma_{1jk}(\boldsymbol\theta)\,\dot\theta_{j}\,\dot\theta_{k}\\[6pt]
         \vdots\\[3pt]
         \sum_{j,k=1}^{n}\Gamma_{njk}(\boldsymbol\theta)\,\dot\theta_{j}\,\dot\theta_{k}
       \end{bmatrix},
\quad
\Gamma_{ijk}
  = \tfrac12\!\bigl(\tfrac{\partial M_{ij}}{\partial\theta_{k}}
                    +\tfrac{\partial M_{ik}}{\partial\theta_{j}}
                    -\tfrac{\partial M_{jk}}{\partial\theta_{i}}\bigr)

the **Christoffel symbols** of the first kind.

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
~~~~~~~~~~~~

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
       
       # Centrifugal forces (diagonal