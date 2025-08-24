.. _api-kinematics:

=========================
Kinematics API Reference
=========================

This page documents **ManipulaPy.kinematics**, the module for manipulator kinematics computations.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/Kinematics`.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

------------
SerialManipulator Class
------------

.. currentmodule:: ManipulaPy.kinematics

.. autoclass:: SerialManipulator
   :members:
   :show-inheritance:

   Main class for serial manipulator kinematics using Product of Exponentials (PoE) formulation.

   .. rubric:: Constructor

   .. automethod:: __init__

      **Parameters:**

        - **M_list** (*numpy.ndarray*) -- Home configuration matrix (4×4)
        - **omega_list** (*numpy.ndarray*) -- Joint rotation axes (3×n)
        - **r_list** (*numpy.ndarray*, optional) -- Joint position vectors (3×n)
        - **b_list** (*numpy.ndarray*, optional) -- Body frame joint positions (3×n)
        - **S_list** (*numpy.ndarray*, optional) -- Space frame screw axes (6×n)
        - **B_list** (*numpy.ndarray*, optional) -- Body frame screw axes (6×n)
        - **G_list** (*numpy.ndarray*, optional) -- Spatial inertia matrices (6×6×n)
        - **joint_limits** (*list*, optional) -- Joint limits [(min, max), ...] for each joint

      **Auto-computation:** Missing S_list or B_list computed from omega_list and r_list/b_list.

   .. rubric:: State Management

   .. automethod:: update_state

      **Parameters:**

        - **joint_positions** (*array_like*) -- Current joint positions in radians
        - **joint_velocities** (*array_like*, optional) -- Current joint velocities in rad/s

      **Purpose:** Updates internal state for stateful operations.

   .. rubric:: Forward Kinematics

   .. automethod:: forward_kinematics

      **Parameters:**

        - **thetalist** (*array_like*) -- Joint angles in radians
        - **frame** (*str*) -- Reference frame: "space" or "body" (default: "space")

      **Returns:**

        - **T** (*numpy.ndarray*) -- 4×4 transformation matrix

      **Formula (Space):** T(θ) = e^{S₁θ₁} e^{S₂θ₂} ... e^{Sₙθₙ} M

      **Formula (Body):** T(θ) = M e^{B₁θ₁} e^{B₂θ₂} ... e^{Bₙθₙ}

      **Example:**
        >>> T = robot.forward_kinematics([0.1, 0.2, 0.3], frame="space")
        >>> position = T[:3, 3]  # Extract position
        >>> rotation = T[:3, :3]  # Extract rotation matrix

   .. automethod:: end_effector_pose

      **Parameters:**

        - **thetalist** (*array_like*) -- Joint angles in radians

      **Returns:**

        - **pose** (*numpy.ndarray*) -- 6-element vector [x, y, z, roll, pitch, yaw]

      **Components:** Position (meters) + Euler angles (radians)

   .. rubric:: Velocity Kinematics

   .. automethod:: jacobian

      **Parameters:**

        - **thetalist** (*array_like*) -- Joint angles in radians
        - **frame** (*str*) -- Reference frame: "space" or "body" (default: "space")

      **Returns:**

        - **J** (*numpy.ndarray*) -- 6×n Jacobian matrix

      **Structure:** J = [J_v; J_ω] where J_v is linear, J_ω is angular velocity Jacobian

      **Formula (Space):** J_s = [Ad_{T₁} S₁, Ad_{T₂} S₂, ..., Ad_{Tᵢ} Sᵢ, ...]

      **Formula (Body):** J_b = [Ad_{T⁻¹} B₁, Ad_{T⁻¹} B₂, ..., Ad_{T⁻¹} Bₙ]

   .. automethod:: end_effector_velocity

      **Parameters:**

        - **thetalist** (*array_like*) -- Joint angles in radians
        - **dthetalist** (*array_like*) -- Joint velocities in rad/s
        - **frame** (*str*) -- Reference frame: "space" or "body" (default: "space")

      **Returns:**

        - **V** (*numpy.ndarray*) -- 6-element twist vector [vx, vy, vz, ωx, ωy, ωz]

      **Formula:** V = J(θ) θ̇

   .. automethod:: joint_velocity

      **Parameters:**

        - **thetalist** (*array_like*) -- Joint angles in radians
        - **V_ee** (*array_like*) -- Desired end-effector velocity twist
        - **frame** (*str*) -- Reference frame: "space" or "body" (default: "space")

      **Returns:**

        - **dthetalist** (*numpy.ndarray*) -- Required joint velocities in rad/s

      **Formula:** θ̇ = J⁺(θ) V where J⁺ is Moore-Penrose pseudoinverse

   .. rubric:: Inverse Kinematics

   .. automethod:: iterative_inverse_kinematics

      **Parameters:**

        - **T_desired** (*numpy.ndarray*) -- Desired 4×4 transformation matrix
        - **thetalist0** (*array_like*) -- Initial guess for joint angles
        - **eomg** (*float*) -- Rotational error tolerance (default: 1e-9)
        - **ev** (*float*) -- Translational error tolerance (default: 1e-9)
        - **max_iterations** (*int*) -- Maximum iterations (default: 5000)
        - **plot_residuals** (*bool*) -- Plot convergence (default: False)

      **Returns:**

        - **thetalist** (*numpy.ndarray*) -- Solution joint angles
        - **success** (*bool*) -- Convergence status
        - **num_iterations** (*int*) -- Iterations used

      **Algorithm:** Newton-Raphson with Damped Least Squares

      **Update:** θₖ₊₁ = θₖ + α J⁺(θₖ) V_error

      **Convergence:** ||V_trans|| < ev AND ||V_rot|| < eomg

   .. automethod:: hybrid_inverse_kinematics

      **Parameters:**

        - **T_desired** (*numpy.ndarray*) -- Desired 4×4 transformation matrix
        - **neural_network** (*torch.nn.Module*) -- Trained neural network model
        - **scaler_X** (*sklearn.preprocessing.StandardScaler*) -- Input feature scaler
        - **scaler_y** (*sklearn.preprocessing.StandardScaler*) -- Output feature scaler
        - **device** (*torch.device*) -- PyTorch device for neural network
        - **thetalist0** (*array_like*, optional) -- Initial guess (if None, use neural network)
        - **eomg** (*float*) -- Rotational error tolerance (default: 1e-6)
        - **ev** (*float*) -- Translational error tolerance (default: 1e-6)
        - **max_iterations** (*int*) -- Maximum iterations (default: 500)

      **Returns:**

        - **thetalist** (*numpy.ndarray*) -- Solution joint angles
        - **success** (*bool*) -- Convergence status
        - **num_iterations** (*int*) -- Iterations used

      **Strategy:** Neural network initial guess + iterative refinement

-------------
Usage Examples
-------------

**Basic Setup**::

   from ManipulaPy.kinematics import SerialManipulator
   import numpy as np
   
   # Define robot parameters
   M = np.array([[1, 0, 0, 0.5],
                 [0, 1, 0, 0.0],
                 [0, 0, 1, 0.3],
                 [0, 0, 0, 1]])
   
   omega_list = np.array([[0, 0, 1],    # Joint 1: Z-axis rotation
                          [0, 1, 0],    # Joint 2: Y-axis rotation
                          [0, 1, 0]])   # Joint 3: Y-axis rotation
   
   robot = SerialManipulator(M_list=M, omega_list=omega_list)

**Forward Kinematics**::

   # Compute end-effector pose
   joint_angles = [0.5, -0.3, 0.8]
   T = robot.forward_kinematics(joint_angles, frame="space")
   
   print(f"Position: {T[:3, 3]}")
   print(f"Orientation:\n{T[:3, :3]}")
   
   # Get pose as [x, y, z, roll, pitch, yaw]
   pose = robot.end_effector_pose(joint_angles)

**Jacobian and Velocities**::

   # Compute Jacobian matrix
   J = robot.jacobian(joint_angles, frame="space")
   print(f"Jacobian shape: {J.shape}")
   
   # Forward velocity kinematics
   joint_velocities = [0.1, 0.2, -0.1]
   ee_velocity = robot.end_effector_velocity(joint_angles, joint_velocities)
   
   # Inverse velocity kinematics
   desired_velocity = [0.05, 0.0, 0.1, 0.0, 0.0, 0.2]
   required_joint_vel = robot.joint_velocity(joint_angles, desired_velocity)

**Inverse Kinematics**::

   # Define target pose
   T_target = np.array([[0, -1, 0, 0.3],
                        [1,  0, 0, 0.2],
                        [0,  0, 1, 0.4],
                        [0,  0, 0, 1]])
   
   # Solve inverse kinematics
   initial_guess = [0, 0, 0]
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=initial_guess,
       eomg=1e-6,
       ev=1e-6,
       plot_residuals=True
   )
   
   if success:
       print(f"IK converged in {iterations} iterations")
       print(f"Solution: {solution}")
   else:
       print("IK failed to converge")

**Hybrid Neural Network IK**::

   import torch
   from sklearn.preprocessing import StandardScaler
   
   # Load trained model and scalers
   model = torch.load("ik_model.pth")
   scaler_X = StandardScaler()  # Load fitted scaler
   scaler_y = StandardScaler()  # Load fitted scaler
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Solve with neural network initialization
   solution, success, iterations = robot.hybrid_inverse_kinematics(
       T_desired=T_target,
       neural_network=model,
       scaler_X=scaler_X,
       scaler_y=scaler_y,
       device=device
   )

**State Management**::

   # Update robot state
   current_positions = [0.1, 0.2, 0.3]
   current_velocities = [0.05, 0.1, 0.0]
   
   robot.update_state(current_positions, current_velocities)
   
   # Access current state
   print(f"Current positions: {robot.joint_positions}")
   print(f"Current velocities: {robot.joint_velocities}")

**Joint Limits**::

   # Define joint limits
   joint_limits = [(-np.pi, np.pi),      # Joint 1: ±180°
                   (-np.pi/2, np.pi/2),   # Joint 2: ±90°
                   (-np.pi/4, np.pi/4)]   # Joint 3: ±45°
   
   robot = SerialManipulator(
       M_list=M,
       omega_list=omega_list,
       joint_limits=joint_limits
   )
   
   # Limits are enforced during IK solving

-------------
Key Features
-------------

- **Product of Exponentials** formulation for robust kinematics

- **Dual frame support** (space and body frames) for flexibility

- **Automatic screw axis** computation from joint parameters

- **Damped least squares** IK with singularity handling

- **Neural network integration** for improved IK initialization

- **Joint limit enforcement** during inverse kinematics

- **Comprehensive velocity** kinematics with pseudoinverse

- **Convergence visualization** for debugging IK problems

-----------------
Mathematical Background
-----------------

**Screw Theory Foundation:**

  - Uses 6D twist vectors to represent joint motion
  - Exponential coordinates for robust orientation handling
  - Adjoint transformations for frame conversions

**Jacobian Computation:**

  - Space Jacobian: J_s = [Ad_{T₁} S₁, ..., Ad_{Tᵢ} Sᵢ, ...]
  - Body Jacobian: J_b computed via inverse transformations
  - Pseudoinverse used for redundant/singular configurations

**Inverse Kinematics:**

  - Newton-Raphson iteration in twist coordinates
  - Separate convergence criteria for translation and rotation
  - Adaptive step size for stability (α = 0.058)

-----------------
See Also
-----------------

- :doc:`dynamics` -- Dynamics computations building on kinematics
- :doc:`control` -- Controllers using kinematic models
- :doc:`path_planning` -- Trajectory generation with kinematic constraints
- :doc:`../user_guide/Kinematics` -- Conceptual overview and theory what to replace here 