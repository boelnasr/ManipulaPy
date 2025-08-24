# URDF Processor User Guide
==========================

This guide covers the URDF Processor module in ManipulaPy, which converts URDF (Unified Robot Description Format) files into SerialManipulator and ManipulatorDynamics objects for analytical robotics computations.

Introduction
-----------

The URDF Processor bridges the gap between URDF robot descriptions and ManipulaPy's analytical framework. It automatically extracts kinematic and dynamic parameters from URDF files and creates the necessary objects for robotics analysis.

**Key Features:**

- Automatic parameter extraction from URDF files
- Kinematic chain analysis and screw axis computation  
- Inertial property extraction for dynamics
- Joint limit handling with PyBullet integration
- Conversion to SerialManipulator and ManipulatorDynamics objects

Basic Usage
----------

Simple URDF Loading
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   
   # Load URDF and create objects
   processor = URDFToSerialManipulator("path/to/robot.urdf")
   
   # Access the created objects
   robot = processor.serial_manipulator      # SerialManipulator instance
   dynamics = processor.dynamics             # ManipulatorDynamics instance
   
   # Use the robot for computations
   import numpy as np
   theta = np.array([0.1, 0.3, -0.2])
   T = robot.forward_kinematics(theta)
   
   print(f"End-effector position: {T[:3, 3]}")

Using Built-in Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf
   
   # Load built-in xArm robot
   processor = URDFToSerialManipulator(xarm_urdf)
   robot = processor.serial_manipulator
   
   print(f"Robot has {len(robot.joint_limits)} joints")

URDFToSerialManipulator Class
----------------------------

Class Constructor
~~~~~~~~~~~~~~~~

.. code-block:: python

   URDFToSerialManipulator(urdf_name, use_pybullet_limits=True)

**Parameters:**

- ``urdf_name`` (str): Path to the URDF file
- ``use_pybullet_limits`` (bool): Extract joint limits from PyBullet simulation

**Attributes:**

- ``serial_manipulator``: SerialManipulator object for kinematics
- ``dynamics``: ManipulatorDynamics object for dynamics  
- ``robot_data``: Dictionary containing extracted parameters
- ``urdf_name``: Path to the loaded URDF file
- ``robot``: Loaded URDF object from urchin library

Extracted Parameters
~~~~~~~~~~~~~~~~~~~

The ``robot_data`` dictionary contains:

.. code-block:: python

   processor = URDFToSerialManipulator("robot.urdf")
   data = processor.robot_data
   
   print(f"Degrees of freedom: {data['actuated_joints_num']}")
   print(f"Home configuration shape: {data['M'].shape}")          # (4, 4)
   print(f"Space screw axes shape: {data['Slist'].shape}")        # (6, n)
   print(f"Body screw axes shape: {data['Blist'].shape}")         # (6, n)  
   print(f"Number of inertia matrices: {len(data['Glist'])}")     # n links

Core Methods
-----------

load_urdf()
~~~~~~~~~~

Extracts kinematic and dynamic parameters from the URDF file:

.. code-block:: python

   def parameter_extraction_example():
       processor = URDFToSerialManipulator("robot.urdf")
       data = processor.robot_data
       
       # Access screw axes
       Slist = data["Slist"]  # Shape: (6, n_joints)
       for i in range(Slist.shape[1]):
           omega = Slist[:3, i]  # Angular velocity part
           v = Slist[3:, i]      # Linear velocity part
           print(f"Joint {i+1}: ω={omega}, v={v}")
       
       # Access inertial properties  
       Glist = data["Glist"]  # List of (6, 6) spatial inertia matrices
       for i, G in enumerate(Glist):
           mass = G[3, 3]  # Mass (assuming diagonal)
           print(f"Link {i+1} mass: {mass:.3f} kg")
       
       # Home configuration
       M = data["M"]  # (4, 4) homogeneous transformation
       print(f"Home position: {M[:3, 3]}")

initialize_serial_manipulator()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates the SerialManipulator object:

.. code-block:: python

   # The processor automatically calls this during initialization
   processor = URDFToSerialManipulator("robot.urdf")
   robot = processor.serial_manipulator
   
   # Access SerialManipulator properties
   print(f"Joint limits: {robot.joint_limits}")
   print(f"Screw axes shape: {robot.S_list.shape}")
   print(f"Home configuration:\n{robot.M_list}")

initialize_manipulator_dynamics()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates the ManipulatorDynamics object:

.. code-block:: python

   processor = URDFToSerialManipulator("robot.urdf")
   dynamics = processor.dynamics
   
   # Use dynamics for computations
   theta = np.array([0.1, 0.3, -0.2])
   theta_dot = np.array([0.5, -0.3, 0.8])
   
   M = dynamics.mass_matrix(theta)
   c = dynamics.velocity_quadratic_forces(theta, theta_dot)
   g = dynamics.gravity_forces(theta, [0, 0, -9.81])
   
   print(f"Mass matrix shape: {M.shape}")
   print(f"Coriolis forces: {c}")
   print(f"Gravity forces: {g}")

Joint Limit Handling
-------------------

PyBullet Integration
~~~~~~~~~~~~~~~~~~~

When ``use_pybullet_limits=True``, the processor extracts joint limits from PyBullet:

.. code-block:: python

   # With PyBullet limits (default)
   processor_pyb = URDFToSerialManipulator("robot.urdf", use_pybullet_limits=True)
   
   # Without PyBullet limits (uses default ±π)
   processor_default = URDFToSerialManipulator("robot.urdf", use_pybullet_limits=False)
   
   # Compare limits
   pyb_limits = processor_pyb.serial_manipulator.joint_limits
   default_limits = processor_default.serial_manipulator.joint_limits
   
   for i, (pyb, default) in enumerate(zip(pyb_limits, default_limits)):
       print(f"Joint {i+1}:")
       print(f"  PyBullet: [{np.degrees(pyb[0]):6.1f}, {np.degrees(pyb[1]):6.1f}] deg")
       print(f"  Default:  [{np.degrees(default[0]):6.1f}, {np.degrees(default[1]):6.1f}] deg")

Custom Joint Limits
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   processor = URDFToSerialManipulator("robot.urdf")
   robot = processor.serial_manipulator
   
   # Set custom limits
   custom_limits = [
       (-np.pi, np.pi),        # Joint 1: full rotation
       (-np.pi/2, np.pi/2),    # Joint 2: ±90°
       (-np.pi/3, np.pi/3),    # Joint 3: ±60°
   ]
   
   robot.joint_limits = custom_limits[:len(robot.joint_limits)]

Utility Methods
--------------

Static Methods
~~~~~~~~~~~~~

.. code-block:: python

   # Extract position from transformation matrix
   T = np.eye(4)
   T[:3, 3] = [1, 2, 3]
   pos = URDFToSerialManipulator.transform_to_xyz(T)
   print(f"Position: {pos}")  # [1, 2, 3]
   
   # Find link by name
   processor = URDFToSerialManipulator("robot.urdf")
   link = URDFToSerialManipulator.get_link(processor.robot, "link_name")
   
   # Convert joint axes to screw axes
   joint_axes = np.array([[0, 0, 1], [0, 1, 0]]).T      # 2 joints
   joint_positions = np.array([[0, 0, 0], [0, 0, 0.5]]).T
   Slist = URDFToSerialManipulator.w_p_to_slist(joint_axes.T, joint_positions.T, 2)
   print(f"Screw axes shape: {Slist.shape}")  # (6, 2)

Visualization Methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   processor = URDFToSerialManipulator("robot.urdf")
   
   # Visualize robot using urchin (matplotlib)
   processor.visualize_robot()
   
   # Visualize trajectory animation
   n_joints = len(processor.serial_manipulator.joint_limits)
   trajectory = np.random.uniform(-0.5, 0.5, (50, n_joints))
   
   processor.visualize_trajectory(
       cfg_trajectory=trajectory,
       loop_time=3.0,
       use_collision=False
   )
   
   # Get joint information
   joint_info = processor.print_joint_info()
   print(f"Number of joints: {joint_info['num_joints']}")
   print(f"Joint names: {joint_info['joint_names']}")

Working Example
--------------

Complete Robot Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def complete_robot_setup():
       """Complete example of setting up a robot from URDF."""
       
       # Load URDF
       processor = URDFToSerialManipulator("robot.urdf")
       robot = processor.serial_manipulator
       dynamics = processor.dynamics
       
       print("Robot Setup Complete:")
       print(f"- DOF: {len(robot.joint_limits)}")
       print(f"- Joint limits: {robot.joint_limits}")
       
       # Test forward kinematics
       theta = np.zeros(len(robot.joint_limits))
       T_home = robot.forward_kinematics(theta)
       print(f"- Home position: {T_home[:3, 3]}")
       
       # Test inverse kinematics
       target = np.eye(4)
       target[:3, 3] = [0.3, 0.2, 0.4]
       
       solution, success, iterations = robot.iterative_inverse_kinematics(
           target, theta, max_iterations=500
       )
       
       print(f"- IK test: {'Success' if success else 'Failed'} ({iterations} iter)")
       
       # Test dynamics
       theta_test = np.array([0.1, 0.3, -0.2])[:len(robot.joint_limits)]
       M = dynamics.mass_matrix(theta_test)
       print(f"- Mass matrix condition: {np.linalg.cond(M):.2e}")
       
       return processor
   
   # Run complete setup
   processor = complete_robot_setup()

Kinematics and Dynamics Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def kinematics_dynamics_example():
       """Example using both kinematics and dynamics."""
       
       processor = URDFToSerialManipulator("robot.urdf")
       robot = processor.serial_manipulator
       dynamics = processor.dynamics
       
       # Define robot state
       n_joints = len(robot.joint_limits)
       theta = np.random.uniform(-0.5, 0.5, n_joints)
       theta_dot = np.random.uniform(-1.0, 1.0, n_joints)
       theta_ddot = np.random.uniform(-2.0, 2.0, n_joints)
       
       # Kinematics
       T = robot.forward_kinematics(theta)
       J = robot.jacobian(theta)
       V_ee = robot.end_effector_velocity(theta, theta_dot)
       
       print("Kinematics Results:")
       print(f"- End-effector position: {T[:3, 3]}")
       print(f"- Jacobian shape: {J.shape}")
       print(f"- End-effector velocity: {V_ee}")
       
       # Dynamics
       M = dynamics.mass_matrix(theta)
       c = dynamics.velocity_quadratic_forces(theta, theta_dot)
       g = dynamics.gravity_forces(theta, [0, 0, -9.81])
       
       # Inverse dynamics
       tau = dynamics.inverse_dynamics(
           theta, theta_dot, theta_ddot, [0, 0, -9.81], np.zeros(6)
       )
       
       # Forward dynamics
       theta_ddot_computed = dynamics.forward_dynamics(
           theta, theta_dot, tau, [0, 0, -9.81], np.zeros(6)
       )
       
       print("\nDynamics Results:")
       print(f"- Mass matrix determinant: {np.linalg.det(M):.6f}")
       print(f"- Required torques: {tau}")
       print(f"- Verification error: {np.linalg.norm(theta_ddot - theta_ddot_computed):.6f}")
       
       return robot, dynamics
   
   # Run example
   robot, dynamics = kinematics_dynamics_example()

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_urdf_loading(urdf_path):
       """Robust URDF loading with error handling."""
       
       try:
           # Attempt to load URDF
           processor = URDFToSerialManipulator(urdf_path)
           
           # Validate basic properties
           robot = processor.serial_manipulator
           dynamics = processor.dynamics
           
           # Check if robot has reasonable properties
           if len(robot.joint_limits) == 0:
               raise ValueError("No actuated joints found in URDF")
           
           # Test basic computation
           theta = np.zeros(len(robot.joint_limits))
           T = robot.forward_kinematics(theta)
           M = dynamics.mass_matrix(theta)
           
           # Check for numerical issues
           if not np.all(np.isfinite(T)):
               raise ValueError("Forward kinematics produces invalid results")
           
           if np.linalg.cond(M) > 1e12:
               print("Warning: Mass matrix is poorly conditioned")
           
           print(f"✅ Successfully loaded robot with {len(robot.joint_limits)} joints")
           return processor
           
       except FileNotFoundError:
           print(f"❌ URDF file not found: {urdf_path}")
           print("   Check file path and permissions")
           
       except Exception as e:
           print(f"❌ Error loading URDF: {e}")
           print("   Possible solutions:")
           print("   - Validate URDF syntax")
           print("   - Check for missing mesh files")
           print("   - Verify joint and link definitions")
           
       return None

   # Example usage
   processor = robust_urdf_loading("robot.urdf")

Best Practices
-------------

URDF File Requirements
~~~~~~~~~~~~~~~~~~~~~

For optimal results, ensure your URDF file has:

1. **Proper inertial properties** for all links

2. **Realistic joint limits** defined

3. **Consistent coordinate frames** throughout the chain

4. **Valid joint axis definitions** (unit vectors)

5. **Accessible mesh files** (if using complex geometries)

Performance Tips
~~~~~~~~~~~~~~~

.. code-block:: python

   # Cache the processor for repeated use
   _urdf_cache = {}
   
   def get_robot_processor(urdf_path):
       """Get cached processor or create new one."""
       if urdf_path not in _urdf_cache:
           _urdf_cache[urdf_path] = URDFToSerialManipulator(urdf_path)
       return _urdf_cache[urdf_path]
   
   # Use the cached version
   processor = get_robot_processor("robot.urdf")

Validation Checklist
~~~~~~~~~~~~~~~~~~~

Before using a processed URDF:

.. code-block:: python

   def validate_processor(processor):
       """Quick validation of URDF processor results."""
       
       robot = processor.serial_manipulator
       dynamics = processor.dynamics
       
       # Check 1: Forward kinematics at home
       theta_home = np.zeros(len(robot.joint_limits))
       T_home = robot.forward_kinematics(theta_home)
       print(f"✓ Home position: {T_home[:3, 3]}")
       
       # Check 2: Mass matrix properties
       M = dynamics.mass_matrix(theta_home)
       is_symmetric = np.allclose(M, M.T)
       is_positive_def = np.all(np.linalg.eigvals(M) > 0)
       print(f"✓ Mass matrix: symmetric={is_symmetric}, pos_def={is_positive_def}")
       
       # Check 3: Joint limits are reasonable
       reasonable_limits = all(
           abs(limit[1] - limit[0]) > 0.1 for limit in robot.joint_limits
       )
       print(f"✓ Joint limits: reasonable={reasonable_limits}")
       
       return is_symmetric and is_positive_def and reasonable_limits
   
   # Validate before use
   is_valid = validate_processor(processor)

Summary
-------

The URDF Processor provides seamless conversion from URDF robot descriptions to ManipulaPy's analytical framework:

**Key Components:**

- **URDFToSerialManipulator class**: Main interface for URDF processing

- **Automatic parameter extraction**: Kinematic and dynamic properties

- **Joint limit handling**: PyBullet integration for realistic limits

- **Object creation**: SerialManipulator and ManipulatorDynamics instances

**Typical Workflow:**

1. Load URDF file with ``URDFToSerialManipulator(urdf_path)``
2. Access ``serial_manipulator`` for kinematics computations
3. Access ``dynamics`` for dynamics computations  
4. Use standard ManipulaPy methods for analysis and control

**Best Practices:**

- Validate URDF files before processing
- Use PyBullet limits for realistic joint constraints
- Cache processors for repeated use
- Check extracted parameters for consistency

The URDF Processor enables you to leverage existing robot models while benefiting from ManipulaPy's analytical capabilities for advanced robotics applications.