Kinematics User Guide
=====================

This guide covers the **SerialManipulator** class in ManipulaPy, which provides forward kinematics, inverse kinematics, and Jacobian computations for serial robot manipulators using modern screw theory.

.. contents:: **Quick Navigation**
   :local:
   :depth: 2

What is Robot Kinematics?
-------------------------

**Robot kinematics** studies the geometry of manipulator motion without considering forces. The key problems are:

- **Forward Kinematics (FK):** Given joint angles → find end-effector pose
- **Inverse Kinematics (IK):** Given desired pose → find joint angles  
- **Jacobian Analysis:** Relationship between joint velocities and end-effector motion

ManipulaPy uses the **Product of Exponentials (PoE)** formulation with screw theory for numerically stable computations.

Mathematical Foundation
-----------------------

Product of Exponentials Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The forward kinematics is computed using:

**Space Frame:**

.. math::
   T(\boldsymbol\theta)
     = \exp\!\bigl([\mathcal S_{1}]\;\theta_{1}\bigr)\,
       \exp\!\bigl([\mathcal S_{2}]\;\theta_{2}\bigr)\,
       \cdots\,
       \exp\!\bigl([\mathcal S_{n}]\;\theta_{n}\bigr)\,M

**Body Frame:**

.. math::
   T(\boldsymbol\theta)
     = M\,
       \exp\!\bigl([\mathcal B_{1}]\;\theta_{1}\bigr)\,
       \exp\!\bigl([\mathcal B_{2}]\;\theta_{2}\bigr)\,
       \cdots\,
       \exp\!\bigl([\mathcal B_{n}]\;\theta_{n}\bigr)

where

.. math::
   [\mathcal S]
     = \begin{bmatrix}
         [\boldsymbol\omega]_\times & \mathbf v \\
         \mathbf0^T                & 0
       \end{bmatrix}
   \quad\text{and}\quad
   [\mathcal B]
     = \begin{bmatrix}
         [\boldsymbol\omega]_\times & \mathbf v \\
         \mathbf0^T                & 0
       \end{bmatrix}

with :math:`[\boldsymbol\omega]_\times` the usual 3×3 skew‐symmetric matrix of :math:`\boldsymbol\omega`.

Screw Axes
~~~~~~~~~~

Each joint's motion is encoded as a 6-vector

.. math::
   \mathcal S
     = \begin{bmatrix}\boldsymbol\omega\\\mathbf v\end{bmatrix}\,\in\mathbb R^6

with three canonical cases:

- **Revolute joint** about axis :math:`\boldsymbol\omega` through point :math:`\mathbf r`  

  .. math::
     \mathbf v = \mathbf r \times \boldsymbol\omega,\qquad
     \mathcal S
       = \begin{bmatrix}
           \boldsymbol\omega \\
           \mathbf r \times \boldsymbol\omega
         \end{bmatrix}

- **Prismatic joint** translating along unit vector :math:`\mathbf v`  

  .. math::
     \mathcal S
       = \begin{bmatrix}
           \mathbf0 \\
           \mathbf v
         \end{bmatrix},\quad \|\mathbf v\|=1

- **General (helical) screw** with pitch :math:`h`  

  .. math::
     \mathcal S
       = \begin{bmatrix}
           \boldsymbol\omega \\
           \mathbf r \times \boldsymbol\omega + h\,\boldsymbol\omega
         \end{bmatrix}

Jacobian Computation
~~~~~~~~~~~~~~~~~~~~

**Space Jacobian** for an n-DOF manipulator:

.. math::
   J_{s}(\boldsymbol\theta)
     = \bigl[\,\mathrm{Ad}_{T_{0}^{-1}(\boldsymbol\theta)}\,\mathcal S_{1}\;,\;
               \mathrm{Ad}_{T_{1}^{-1}(\boldsymbol\theta)}\,\mathcal S_{2}\;,\;
               \dots\;,\;
               \mathrm{Ad}_{T_{n-1}^{-1}(\boldsymbol\theta)}\,\mathcal S_{n}\bigr]

where

.. math::
   T_{i}(\boldsymbol\theta)
     = \exp\!\bigl([\mathcal S_{1}]\theta_{1}\bigr)\,
       \exp\!\bigl([\mathcal S_{2}]\theta_{2}\bigr)\,\cdots\,
       \exp\!\bigl([\mathcal S_{i}]\theta_{i}\bigr)

and

.. math::
   \mathrm{Ad}_{T}
     = \begin{bmatrix}
         R & \mathbf0 \\
         [\mathbf p]_\times R & R
       \end{bmatrix}
   \quad\text{for}\quad
   T = \begin{bmatrix}R & \mathbf p\\\mathbf0^T&1\end{bmatrix}.

**Body Jacobian** follows by

.. math::
   J_{b}(\boldsymbol\theta)
     = \mathrm{Ad}_{T(\boldsymbol\theta)^{-1}}\,J_{s}(\boldsymbol\theta)

Inverse Kinematics (Newton–Raphson / Damped Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error twist:**

.. math::
   \mathbf V_{\mathrm{error}}
     = \bigl(\log\!\bigl(T_{\mathrm{desired}}\,T(\boldsymbol\theta)^{-1}\bigr)\bigr)^\vee

**Update step:**

.. math::
   \boldsymbol\theta_{k+1}
     = \boldsymbol\theta_{k}
       + \alpha\,(J^{T}J + \lambda I)^{-1}J^{T}\,\mathbf V_{\mathrm{error}}

with damping :math:`\lambda` for numerical stability.

SerialManipulator Class
-----------------------

Constructor
~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.kinematics import SerialManipulator
   
   robot = SerialManipulator(
       M_list,           # Home configuration (4×4 matrix)
       omega_list,       # Joint axes (3×n matrix)  
       r_list=None,      # Points on joint axes (optional)
       b_list=None,      # Body frame points (optional)
       S_list=None,      # Space frame screw axes (6×n matrix)
       B_list=None,      # Body frame screw axes (6×n matrix)
       G_list=None,      # Inertia matrices (for dynamics)
       joint_limits=None # Joint limits [(min, max), ...]
   )

**Key Parameters:**
- **M_list**: 4×4 transformation matrix representing the home pose
- **omega_list**: 3×n matrix of joint rotation axes
- **S_list**: 6×n matrix of space frame screw axes (auto-computed if not provided)
- **joint_limits**: List of (min, max) tuples for each joint

Creating a Robot Model
----------------------

From URDF (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf
   
   # Load built-in xArm robot
   processor = URDFToSerialManipulator(xarm_urdf)
   robot = processor.serial_manipulator
   
   print(f"Robot has {len(robot.joint_limits)} joints")
   print(f"Home position: {robot.M_list[:3, 3]}")

Manual Definition
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Example: 2-DOF planar robot
   def create_2dof_planar_robot():
       """Create a simple 2-DOF planar RR robot."""
       
       # Link lengths
       L1, L2 = 0.5, 0.3
       
       # Home configuration (fully extended)
       M = np.array([
           [1, 0, 0, L1 + L2],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]
       ])
       
       # Space frame screw axes (both Z-axis rotations)
       S_list = np.array([
           # Joint 1 at origin
           [0, 0, 1, 0, 0, 0],
           # Joint 2 at (L1, 0, 0)
           [0, 0, 1, 0, -L1, 0]
       ]).T  # Shape: (6, 2)
       
       # Extract omega_list for constructor
       omega_list = S_list[:3, :]
       
       # Joint limits
       joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi)]
       
       robot = SerialManipulator(
           M_list=M,
           omega_list=omega_list,
           S_list=S_list,
           joint_limits=joint_limits
       )
       
       return robot
   
   # Create the robot
   planar_robot = create_2dof_planar_robot()

Forward Kinematics
------------------

Basic Forward Kinematics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define joint angles
   theta = np.array([0.5, -0.3])  # radians
   
   # Compute forward kinematics
   T = robot.forward_kinematics(theta, frame="space")
   
   # Extract position and orientation
   position = T[:3, 3]
   rotation_matrix = T[:3, :3]
   
   print(f"End-effector position: {position}")
   print(f"End-effector orientation:\n{rotation_matrix}")

Space vs Body Frames
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   theta = np.array([0.2, 0.3])
   
   # Both methods give the same result
   T_space = robot.forward_kinematics(theta, frame="space")
   T_body = robot.forward_kinematics(theta, frame="body")
   
   # Verify they're identical
   error = np.linalg.norm(T_space - T_body)
   print(f"Space vs Body frame error: {error:.2e}")  # Should be ~0

End-Effector Pose as Vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get pose as [x, y, z, roll, pitch, yaw]
   pose_vector = robot.end_effector_pose(theta)
   
   position = pose_vector[:3]
   euler_angles = pose_vector[3:]  # in radians
   
   print(f"Position: {position}")
   print(f"Orientation (degrees): {np.degrees(euler_angles)}")

Multiple Configurations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_multiple_configurations():
       """Test FK for multiple joint configurations."""
       
       # Test configurations
       test_configs = [
           np.array([0.0, 0.0]),      # Home position
           np.array([np.pi/4, -np.pi/4]),  # 45° configuration
           np.array([np.pi/2, 0.0]),      # Elbow up
           np.array([0.0, np.pi/2])       # Forearm up
       ]
       
       config_names = ["Home", "45° config", "Elbow up", "Forearm up"]
       
       for config, name in zip(test_configs, config_names):
           T = robot.forward_kinematics(config)
           pos = T[:3, 3]
           print(f"{name:12}: position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
   
   test_multiple_configurations()

Inverse Kinematics
------------------

Basic Inverse Kinematics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define target pose
   T_target = np.eye(4)
   T_target[:3, 3] = [0.6, 0.2, 0.0]  # desired position
   
   # Initial guess
   theta_initial = np.array([0.0, 0.0])
   
   # Solve inverse kinematics
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=theta_initial,
       eomg=1e-6,              # rotation error tolerance
       ev=1e-6,                # translation error tolerance
       max_iterations=1000
   )
   
   if success:
       print(f"✅ IK converged in {iterations} iterations")
       print(f"Solution: {np.degrees(solution)} degrees")
       
       # Verify solution
       T_achieved = robot.forward_kinematics(solution)
       error = np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3])
       print(f"Position error: {error:.6f} m")
   else:
       print("❌ IK failed to converge")

Advanced IK Options
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # More robust IK with damping
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=theta_initial,
       eomg=1e-6,
       ev=1e-6,
       max_iterations=1000,
       plot_residuals=True    # Save convergence plot
   )

Multiple IK Solutions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def find_multiple_solutions(robot, target_pos, n_attempts=5):
       """Find multiple IK solutions for the same target."""
       
       T_target = np.eye(4)
       T_target[:3, 3] = target_pos
       
       solutions = []
       
       for attempt in range(n_attempts):
           # Random initial guess
           joint_limits = np.array(robot.joint_limits)
           theta_init = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
           
           solution, success, _ = robot.iterative_inverse_kinematics(
               T_desired=T_target,
               thetalist0=theta_init,
               max_iterations=500
           )
           
           if success:
               # Check if this is a new solution
               is_new = True
               for existing_sol in solutions:
                   if np.linalg.norm(solution - existing_sol) < 0.1:
                       is_new = False
                       break
               
               if is_new:
                   solutions.append(solution)
                   print(f"Solution {len(solutions)}: {np.degrees(solution)}")
       
       return solutions
   
   # Test multiple solutions
   target = [0.5, 0.3, 0.0]
   multiple_solutions = find_multiple_solutions(robot, target)
   print(f"Found {len(multiple_solutions)} distinct solutions")

Jacobian Matrix
--------------------

Computing the Jacobian
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   theta = np.array([0.3, -0.2])
   
   # Compute Jacobian in space frame
   J_space = robot.jacobian(theta, frame="space")
   
   # Compute Jacobian in body frame  
   J_body = robot.jacobian(theta, frame="body")
   
   print(f"Space Jacobian shape: {J_space.shape}")  # (6, n_joints)
   print(f"Body Jacobian shape: {J_body.shape}")    # (6, n_joints)

Jacobian Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_jacobian(robot, theta):
       """Analyze Jacobian properties at a configuration."""
       
       J = robot.jacobian(theta)
       
       # Basic properties
       rank = np.linalg.matrix_rank(J)
       condition_number = np.linalg.cond(J)
       
       print(f"Jacobian Analysis:")
       print(f"  Shape: {J.shape}")
       print(f"  Rank: {rank} (full rank: {rank == min(J.shape)})")
       print(f"  Condition number: {condition_number:.2e}")
       
       # Singularity check
       if condition_number > 1e6:
           print("  ⚠️  Configuration is near singular!")
       else:
           print("  ✅ Configuration is well-conditioned")
       
       # Manipulability (for square Jacobians)
       if J.shape[0] == J.shape[1]:
           manipulability = abs(np.linalg.det(J))
           print(f"  Manipulability: {manipulability:.6f}")
       else:
           # For non-square Jacobians
           manipulability = np.sqrt(np.linalg.det(J @ J.T))
           print(f"  Manipulability: {manipulability:.6f}")
       
       return J, condition_number
   
   # Analyze current configuration
   J, cond_num = analyze_jacobian(robot, theta)

Velocity Kinematics
--------------------

End-Effector Velocity
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Current configuration and joint velocities
   theta = np.array([0.2, 0.3])
   theta_dot = np.array([0.1, -0.2])  # rad/s
   
   # Compute end-effector velocity
   V_ee = robot.end_effector_velocity(theta, theta_dot, frame="space")
   
   print(f"Joint velocities: {theta_dot} rad/s")
   print(f"End-effector velocity: {V_ee}")
   
   # Decompose spatial velocity
   linear_velocity = V_ee[:3]    # [vx, vy, vz]
   angular_velocity = V_ee[3:]   # [ωx, ωy, ωz]
   
   print(f"Linear velocity: {linear_velocity} m/s")
   print(f"Angular velocity: {angular_velocity} rad/s")

Joint Velocities from Desired Motion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Desired end-effector motion
   V_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2])  # Move +X, rotate +Z
   
   # Compute required joint velocities
   theta_dot_required = robot.joint_velocity(theta, V_desired, frame="space")
   
   print(f"Desired EE velocity: {V_desired}")
   print(f"Required joint velocities: {theta_dot_required} rad/s")
   
   # Verify by forward computation
   V_achieved = robot.end_effector_velocity(theta, theta_dot_required, frame="space")
   error = np.linalg.norm(V_achieved - V_desired)
   print(f"Velocity error: {error:.2e}")

State Management
-------------------

Robot State Tracking
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Update robot state
   robot.update_state(
       joint_positions=theta,
       joint_velocities=theta_dot
   )
   
   # Access current state
   print(f"Current positions: {robot.joint_positions}")
   print(f"Current velocities: {robot.joint_velocities}")
   
   # Compute current end-effector state
   current_pose = robot.end_effector_pose(robot.joint_positions)
   current_velocity = robot.end_effector_velocity(
       robot.joint_positions, 
       robot.joint_velocities
   )
   
   print(f"Current EE pose: {current_pose}")
   print(f"Current EE velocity: {current_velocity}")

Working Examples
----------------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def complete_kinematics_example():
       """Complete example showing all kinematic functions."""
       
       # Create a simple 3-DOF robot
       def create_3dof_robot():
           L1, L2, L3 = 0.3, 0.25, 0.15
           
           M = np.array([
               [1, 0, 0, L1 + L2 + L3],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]
           ])
           
           S_list = np.array([
               [0, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, -L1, 0],
               [0, 0, 1, 0, -(L1+L2), 0]
           ]).T
           
           omega_list = S_list[:3, :]
           joint_limits = [(-np.pi, np.pi)] * 3
           
           return SerialManipulator(
               M_list=M,
               omega_list=omega_list,
               S_list=S_list,
               joint_limits=joint_limits
           )
       
       robot = create_3dof_robot()
       print("=== Complete Kinematics Example ===")
       
       # 1. Forward Kinematics
       theta = np.array([0.5, -0.3, 0.8])
       T = robot.forward_kinematics(theta)
       print(f"1. Forward Kinematics:")
       print(f"   Joint angles: {np.degrees(theta)} deg")
       print(f"   End-effector position: {T[:3, 3]}")
       
       # 2. Inverse Kinematics
       T_target = np.eye(4)
       T_target[:3, 3] = [0.4, 0.2, 0.0]
       
       solution, success, iterations = robot.iterative_inverse_kinematics(
           T_desired=T_target,
           thetalist0=np.array([0.0, 0.0, 0.0])
       )
       
       print(f"\n2. Inverse Kinematics:")
       print(f"   Target position: {T_target[:3, 3]}")
       print(f"   Success: {success}")
       if success:
           print(f"   Solution: {np.degrees(solution)} deg")
           print(f"   Iterations: {iterations}")
       
       # 3. Jacobian Analysis
       J = robot.jacobian(theta)
       cond_num = np.linalg.cond(J)
       
       print(f"\n3. Jacobian Analysis:")
       print(f"   Shape: {J.shape}")
       print(f"   Condition number: {cond_num:.2e}")
       
       # 4. Velocity Kinematics
       theta_dot = np.array([0.1, -0.2, 0.3])
       V_ee = robot.end_effector_velocity(theta, theta_dot)
       
       print(f"\n4. Velocity Kinematics:")
       print(f"   Joint velocities: {theta_dot} rad/s")
       print(f"   EE linear velocity: {V_ee[:3]} m/s")
       print(f"   EE angular velocity: {V_ee[3:]} rad/s")
       
       return robot
   
   # Run the complete example
   example_robot = complete_kinematics_example()

Common Issues and Solutions
---------------------------

Troubleshooting IK Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def troubleshoot_ik(robot, T_target):
       """Helper function to troubleshoot IK issues."""
       
       print("🔍 IK Troubleshooting:")
       
       # Check if target is reasonable
       target_pos = T_target[:3, 3]
       target_distance = np.linalg.norm(target_pos)
       
       print(f"Target position: {target_pos}")
       print(f"Target distance from origin: {target_distance:.3f} m")
       
       # Try different initial guesses
       joint_limits = np.array(robot.joint_limits)
       
       attempts = [
           np.zeros(len(joint_limits)),                          # Zero guess
           np.mean(joint_limits, axis=1),                       # Middle of ranges
           np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])  # Random
       ]
       
       attempt_names = ["Zero", "Middle", "Random"]
       
       for i, (theta_init, name) in enumerate(zip(attempts, attempt_names)):
           solution, success, iterations = robot.iterative_inverse_kinematics(
               T_desired=T_target,
               thetalist0=theta_init,
               max_iterations=500
           )
           
           if success:
               T_achieved = robot.forward_kinematics(solution)
               error = np.linalg.norm(T_achieved[:3, 3] - target_pos)
               print(f"  {name} init: ✅ Success (error: {error:.2e}, iter: {iterations})")
               return solution
           else:
               print(f"  {name} init: ❌ Failed after {iterations} iterations")
       
       print("  All attempts failed. Target may be unreachable.")
       return None

Best Practices
------------------

1. **Robot Definition**
   - Use URDF files when possible for real robots
   - Validate screw axes are unit vectors for revolute joints
   - Set realistic joint limits

2. **Forward Kinematics**
   - Both space and body frames give identical results
   - Choose the frame that makes your calculations easier

3. **Inverse Kinematics**
   - Provide good initial guesses (avoid singularities)
   - Use damping for stability near singularities
   - Try multiple initial guesses for difficult targets

4. **Jacobian Analysis**
   - Monitor condition number to detect singularities
   - Use velocity kinematics for real-time control

5. **Performance**
   - Cache Jacobian computations when configuration doesn't change
   - Use appropriate tolerances (don't over-specify)

Next Steps
---------------

- **Dynamics**: Add forces and inertias → :doc:`Dynamics`
- **Trajectory Planning**: Plan smooth motions → :doc:`Trajectory_Planning`  
- **Control**: Implement feedback controllers → :doc:`Control`
- **Simulation**: Test in PyBullet → :doc:`Simulation`

API Reference
~~~~~~~~~~~~~
For complete function documentation: :doc:`../api/kinematics`