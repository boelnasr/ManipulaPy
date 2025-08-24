.. _api-urdf-processor:

===============================
URDF Processor API Reference
===============================

This page documents **ManipulaPy.urdf_processor**, the module for URDF (Unified Robot Description Format) processing and conversion to ManipulaPy objects.

.. tip::
   For conceptual explanations, see :doc:`../user_guide/URDF_Processor`.

-----------------
Quick Navigation
-----------------

.. contents::
   :local:
   :depth: 2

------------
URDFToSerialManipulator Class
------------

.. currentmodule:: ManipulaPy.urdf_processor

.. autoclass:: URDFToSerialManipulator
   :members:
   :show-inheritance:

   Converts URDF files to SerialManipulator and ManipulatorDynamics objects with PyBullet integration.

   .. rubric:: Constructor

   .. automethod:: __init__

      **Parameters:**

        - **urdf_name** (*str*) -- Path to the URDF file
        - **use_pybullet_limits** (*bool*) -- Override URDF joint limits with PyBullet values (default: True)

      **Process:**

        1. Load URDF using urchin library
        2. Extract kinematic and dynamic parameters
        3. Optionally retrieve joint limits from PyBullet
        4. Create SerialManipulator and ManipulatorDynamics objects

      **Attributes Created:**

        - **urdf_name** (*str*) -- Path to URDF file
        - **robot** (*urchin.URDF*) -- Loaded URDF object
        - **robot_data** (*dict*) -- Extracted robot parameters
        - **serial_manipulator** (*SerialManipulator*) -- Kinematics object
        - **dynamics** (*ManipulatorDynamics*) -- Dynamics object

   .. rubric:: URDF Processing

   .. automethod:: load_urdf

      **Parameters:**

        - **urdf_name** (*str*) -- Path to the URDF file

      **Returns:**

        - **robot_data** (*dict*) -- Dictionary containing:
          
          - **M** (*numpy.ndarray*) -- Home configuration matrix (4×4)
          - **Slist** (*numpy.ndarray*) -- Space frame screw axes (6×n)
          - **Blist** (*numpy.ndarray*) -- Body frame screw axes (6×n)
          - **Glist** (*list*) -- Spatial inertia matrices (6×6 each)
          - **actuated_joints_num** (*int*) -- Number of actuated joints

      **Process:**

        1. Extract joint positions and rotation axes
        2. Build spatial inertia matrices from link properties
        3. Compute screw axes using Product of Exponentials
        4. Generate body frame screw axes via adjoint transformation

   .. automethod:: _get_joint_limits_from_pybullet

      **Returns:**

        - **joint_limits** (*list*) -- List of (lower, upper) tuples for each revolute joint

      **Process:**

        1. Connect to PyBullet in DIRECT mode (no GUI)
        2. Load URDF into PyBullet simulation
        3. Extract joint limits for all revolute joints
        4. Handle continuous joints with (-π, π) defaults
        5. Disconnect from PyBullet

      **Note:** Only processes revolute joints, ignoring fixed and other joint types.

   .. rubric:: Object Initialization

   .. automethod:: initialize_serial_manipulator

      **Returns:**

        - **manipulator** (*SerialManipulator*) -- Configured kinematics object

      **Configuration:**

        - Uses extracted URDF parameters (M, S_list, B_list)
        - Applies joint limits from PyBullet or defaults
        - Computes omega_list from screw axes

   .. automethod:: initialize_manipulator_dynamics

      **Returns:**

        - **dynamics** (*ManipulatorDynamics*) -- Configured dynamics object

      **Configuration:**

        - Inherits from SerialManipulator for kinematics
        - Uses spatial inertia matrices (Glist) from URDF
        - Extracts r_list from space frame screw axes

   .. rubric:: Visualization

   .. automethod:: visualize_robot

      **Purpose:** Display static robot model using matplotlib visualization from urchin library.

      **Usage:** Interactive 3D visualization with rotation and zoom capabilities.

   .. automethod:: visualize_trajectory

      **Parameters:**

        - **cfg_trajectory** (*numpy.ndarray* or *dict*, optional) -- Joint configurations over time
        - **loop_time** (*float*) -- Animation duration in seconds (default: 3.0)
        - **use_collision** (*bool*) -- Enable collision visualization (default: False)

      **Trajectory Formats:**

        - **NumPy array:** Shape (n_timesteps, n_joints) with joint angles

        - **Dictionary:** {joint_name: [angles...]} mapping joint names to angle sequences

        - **None:** Uses default motion from 0 to π/2 for all joints

      **Features:**

        - Smooth animation between waypoints
        - Collision geometry visualization
        - Configurable playback speed

   .. rubric:: Utilities

   .. automethod:: print_joint_info

      **Returns:**

        - **joint_info** (*dict*) -- Dictionary containing:
          
          - **num_joints** (*int*) -- Total number of joints
          - **joint_names** (*list*) -- Names of all joints in order

      **Purpose:** Inspect joint structure without console output.

   .. rubric:: Static Utility Methods

   .. automethod:: transform_to_xyz
      :staticmethod:

      **Parameters:**

        - **T** (*numpy.ndarray*) -- 4×4 transformation matrix

      **Returns:**

        - **position** (*numpy.ndarray*) -- 3-element position vector [x, y, z]

      **Purpose:** Extract position from homogeneous transformation matrix.

   .. automethod:: get_link
      :staticmethod:

      **Parameters:**

        - **robot** (*urchin.URDF*) -- Loaded URDF object
        - **link_name** (*str*) -- Name of link to find

      **Returns:**

        - **link** (*urchin.Link* or *None*) -- Link object if found, None otherwise

      **Purpose:** Find specific link by name in URDF structure.

   .. automethod:: w_p_to_slist
      :staticmethod:

      **Parameters:**

        - **w** (*numpy.ndarray*) -- Joint rotation axes (3×n)
        - **p** (*numpy.ndarray*) -- Joint positions (3×n)
        - **robot_dof** (*int*) -- Number of degrees of freedom

      **Returns:**

        - **Slist** (*numpy.ndarray*) -- Space frame screw axes (6×n)

      **Formula:** For each joint i: S_i = [ω_i; v_i] where v_i = -ω_i × p_i

      **Purpose:** Convert joint parameters to screw axis representation.

-------------
Usage Examples
-------------

**Basic URDF Loading**::

   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   
   # Load robot from URDF
   processor = URDFToSerialManipulator("robot.urdf")
   
   # Access created objects
   robot = processor.serial_manipulator
   dynamics = processor.dynamics
   
   print(f"Robot has {len(robot.joint_limits)} joints")

**Using Built-in Robot Models**::

   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf
   
   # Load xArm robot
   processor = URDFToSerialManipulator(xarm_urdf)
   
   # Perform forward kinematics
   joint_angles = [0.1, 0.2, 0.3, 0, 0, 0]
   T = processor.serial_manipulator.forward_kinematics(joint_angles)

**Joint Limits Configuration**::

   # Use PyBullet joint limits (default)
   processor_pb = URDFToSerialManipulator("robot.urdf", use_pybullet_limits=True)
   
   # Use default limits (-π, π)
   processor_default = URDFToSerialManipulator("robot.urdf", use_pybullet_limits=False)
   
   # Compare joint limits
   print("PyBullet limits:", processor_pb.serial_manipulator.joint_limits)
   print("Default limits:", processor_default.serial_manipulator.joint_limits)

**Robot Visualization**::

   # Static robot visualization
   processor.visualize_robot()
   
   # Trajectory animation with numpy array
   import numpy as np
   
   # Create trajectory: 100 timesteps, smooth motion
   n_steps = 100
   n_joints = len(processor.serial_manipulator.joint_limits)
   trajectory = np.zeros((n_steps, n_joints))
   
   for i in range(n_joints):
       trajectory[:, i] = np.linspace(0, np.pi/3, n_steps)
   
   processor.visualize_trajectory(
       cfg_trajectory=trajectory,
       loop_time=5.0,
       use_collision=True
   )

**Trajectory Animation with Dictionary**::

   # Get joint names
   joint_info = processor.print_joint_info()
   joint_names = joint_info["joint_names"]
   
   # Create trajectory dictionary
   trajectory_dict = {}
   for joint_name in joint_names:
       if "joint" in joint_name.lower():  # Filter actuated joints
           trajectory_dict[joint_name] = [0, np.pi/4, np.pi/2, np.pi/4, 0]
   
   processor.visualize_trajectory(trajectory_dict, loop_time=4.0)

**Extracting Robot Parameters**::

   # Access raw URDF data
   robot_data = processor.robot_data
   
   print(f"Home configuration:\n{robot_data['M']}")
   print(f"Number of DOF: {robot_data['actuated_joints_num']}")
   print(f"Screw axes shape: {robot_data['Slist'].shape}")
   
   # Access inertia matrices
   for i, G in enumerate(robot_data['Glist']):
       print(f"Link {i} inertia:\n{G}")

**Integration with Kinematics and Dynamics**::

   # Complete workflow example
   processor = URDFToSerialManipulator("robot.urdf")
   
   # Kinematics
   joint_angles = [0.5, -0.3, 0.8, 0.1, -0.2, 0.4]
   T = processor.serial_manipulator.forward_kinematics(joint_angles)
   J = processor.serial_manipulator.jacobian(joint_angles)
   
   # Dynamics
   joint_velocities = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
   joint_accelerations = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
   
   torques = processor.dynamics.inverse_dynamics(
       joint_angles, joint_velocities, joint_accelerations,
       g=[0, 0, -9.81], Ftip=[0, 0, 0, 0, 0, 0]
   )
   
   print(f"Required torques: {torques}")

**Custom URDF Processing**::

   # Access raw urchin URDF object
   urdf_obj = processor.robot
   
   # Inspect URDF structure
   print(f"Robot name: {urdf_obj.name}")
   print(f"Links: {[link.name for link in urdf_obj.links]}")
   print(f"Joints: {[joint.name for joint in urdf_obj.joints]}")
   
   # Get link-wise forward kinematics
   link_transforms = urdf_obj.link_fk()
   for link_name, transform in link_transforms.items():
       print(f"{link_name}: {transform[:3, 3]}")  # Position only

**Error Handling**::

   try:
       processor = URDFToSerialManipulator("nonexistent.urdf")
   except FileNotFoundError:
       print("URDF file not found")
   except Exception as e:
       print(f"URDF processing error: {e}")
   
   # Validate extracted parameters
   if processor.robot_data["actuated_joints_num"] == 0:
       print("Warning: No actuated joints found")
   
   if len(processor.dynamics.Glist) != processor.robot_data["actuated_joints_num"]:
       print("Warning: Inertia matrix count mismatch")

-------------
Key Features
-------------

- **Automatic parameter extraction** from URDF files using urchin library

- **PyBullet integration** for accurate joint limit retrieval

- **Dual object creation** (SerialManipulator + ManipulatorDynamics)

- **Flexible visualization** with static and animated display options

- **Robust trajectory handling** supporting both array and dictionary formats

- **Complete inertia processing** from URDF link properties

- **Screw theory conversion** from joint parameters to screw axes

- **Error handling** for malformed or missing URDF files


-----------------
Dependencies
-----------------

- **urchin** -- URDF loading and visualization

- **PyBullet** -- Joint limit extraction and simulation

- **NumPy** -- Numerical computations and array operations

- **ManipulaPy.kinematics** -- SerialManipulator class

- **ManipulaPy.dynamics** -- ManipulatorDynamics class

- **ManipulaPy.utils** -- Utility functions for transformations

-----------------
See Also
-----------------

- :doc:`kinematics` -- SerialManipulator class created by this processor
- :doc:`dynamics` -- ManipulatorDynamics class created by this processor
- :doc:`simulation` -- Simulation module using URDF models
- :doc:`../user_guide/URDF_Processor` -- Conceptual overview and URDF format details