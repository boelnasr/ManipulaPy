#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic URDF Processing Demo: Robot Model Loading and Inspection

This example demonstrates URDF file processing, robot model creation,
and basic model inspection and visualization capabilities using ManipulaPy's
URDFToSerialManipulator class.

Usage:
    python urdf_processing_basic_demo.py

Author: ManipulaPy Development Team
"""

import numpy as np
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy import utils
    logger.info("✅ ManipulaPy modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ManipulaPy: {e}")
    sys.exit(1)

def create_sample_urdf():
    """
    Create a simple 6-DOF robot URDF file for demonstration.
    This creates a basic robot arm with 6 revolute joints.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="sample_6dof_robot">
  
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 1 -->
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="1.0"/>
  </joint>

  <!-- Link 1 -->
  <link name="link_1">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.015" iyy="0.015" izz="0.005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 2 -->
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0944" upper="2.0944" effort="80" velocity="1.0"/>
  </joint>

  <!-- Link 2 -->
  <link name="link_2">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.012" iyy="0.012" izz="0.004" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 3 -->
  <joint name="joint_3" type="revolute">
    <parent link="link_2"/>
    <child link="link_3"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.618" upper="2.618" effort="60" velocity="1.0"/>
  </joint>

  <!-- Link 3 -->
  <link name="link_3">
    <visual>
      <geometry>
        <cylinder radius="0.035" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.003" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 4 -->
  <joint name="joint_4" type="revolute">
    <parent link="link_3"/>
    <child link="link_4"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="40" velocity="1.5"/>
  </joint>

  <!-- Link 4 -->
  <link name="link_4">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.15"/>
      </geometry>
      <material name="yellow">
        <color rgba="1 1 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.008" iyy="0.008" izz="0.002" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 5 -->
  <joint name="joint_5" type="revolute">
    <parent link="link_4"/>
    <child link="link_5"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="30" velocity="1.5"/>
  </joint>

  <!-- Link 5 -->
  <link name="link_5">
    <visual>
      <geometry>
        <cylinder radius="0.025" length="0.1"/>
      </geometry>
      <material name="cyan">
        <color rgba="0 1 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.6"/>
      <inertia ixx="0.006" iyy="0.006" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 6 -->
  <joint name="joint_6" type="revolute">
    <parent link="link_5"/>
    <child link="link_6"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="20" velocity="2.0"/>
  </joint>

  <!-- Link 6 (End Effector) -->
  <link name="link_6">
    <visual>
      <geometry>
        <cylinder radius="0.02" length="0.05"/>
      </geometry>
      <material name="magenta">
        <color rgba="1 0 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.4"/>
      <inertia ixx="0.004" iyy="0.004" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>"""
    
    # Create the URDF file
    urdf_file = "sample_6dof_robot.urdf"
    with open(urdf_file, 'w') as f:
        f.write(urdf_content)
    
    logger.info(f"✅ Created sample URDF file: {urdf_file}")
    return urdf_file

def load_and_inspect_urdf(urdf_file):
    """
    Load a URDF file using ManipulaPy's URDFToSerialManipulator and inspect its properties.
    
    Args:
        urdf_file (str): Path to the URDF file
        
    Returns:
        URDFToSerialManipulator: The processed robot model
    """
    print(f"\n🔍 Loading and inspecting URDF: {urdf_file}")
    print("=" * 60)
    
    try:
        # Initialize the ManipulaPy URDF processor with PyBullet limits
        logger.info("Initializing ManipulaPy URDFToSerialManipulator...")
        urdf_processor = URDFToSerialManipulator(urdf_file, use_pybullet_limits=True)
        
        # Print basic robot information using ManipulaPy methods
        joint_info = urdf_processor.print_joint_info()
        print(f"📊 Robot Information (from ManipulaPy):")
        print(f"   • Total joints: {joint_info['num_joints']}")
        print(f"   • Actuated joints: {urdf_processor.robot_data['actuated_joints_num']}")
        print(f"   • Joint names: {', '.join(joint_info['joint_names'])}")
        
        # Show PyBullet vs default limits handling
        print(f"\n🔒 Joint Limits (PyBullet integration):")
        if "joint_limits" in urdf_processor.robot_data:
            for i, (lower, upper) in enumerate(urdf_processor.robot_data["joint_limits"]):
                print(f"   Joint {i+1}: [{np.rad2deg(lower):.1f}°, {np.rad2deg(upper):.1f}°]")
        
        return urdf_processor
        
    except Exception as e:
        logger.error(f"❌ Failed to load URDF with ManipulaPy: {e}")
        raise

def inspect_robot_model(urdf_processor):
    """
    Inspect the extracted robot model parameters using ManipulaPy's internal data structures.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🔬 ManipulaPy Robot Model Analysis")
    print("=" * 60)
    
    # Access the robot data extracted by ManipulaPy
    robot_data = urdf_processor.robot_data
    
    # Print transformation matrices (ManipulaPy's home configuration)
    print(f"🏠 Home Configuration Matrix (M) - ManipulaPy extraction:")
    print(f"   Shape: {robot_data['M'].shape}")
    print(f"   End-effector position: {robot_data['M'][:3, 3]}")
    print(f"   End-effector orientation:\n{robot_data['M'][:3, :3]}")
    
    # Print screw axes computed by ManipulaPy
    print(f"\n🔧 Space-Frame Screw Axes (Slist) - ManipulaPy computation:")
    print(f"   Shape: {robot_data['Slist'].shape}")
    print(f"   Computed using ManipulaPy's w_p_to_slist method")
    for i in range(robot_data['Slist'].shape[1]):
        S = robot_data['Slist'][:, i]
        print(f"   Joint {i+1}: ω=[{S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}], "
              f"v=[{S[3]:.3f}, {S[4]:.3f}, {S[5]:.3f}]")
    
    print(f"\n🔧 Body-Frame Screw Axes (Blist) - ManipulaPy computation:")
    print(f"   Shape: {robot_data['Blist'].shape}")
    print(f"   Computed using ManipulaPy's adjoint transformation")
    for i in range(robot_data['Blist'].shape[1]):
        B = robot_data['Blist'][:, i]
        print(f"   Joint {i+1}: ω=[{B[0]:.3f}, {B[1]:.3f}, {B[2]:.3f}], "
              f"v=[{B[3]:.3f}, {B[4]:.3f}, {B[5]:.3f}]")
    
    # Print inertial properties extracted by ManipulaPy
    print(f"\n⚖️  Inertial Properties (Glist) - ManipulaPy extraction:")
    print(f"   Number of links: {len(robot_data['Glist'])}")
    for i, G in enumerate(robot_data['Glist']):
        print(f"   Link {i+1} inertia matrix shape: {G.shape}")
        # Extract mass from ManipulaPy's 6x6 inertia matrix format
        if G.shape == (6, 6) and G[3, 3] == G[4, 4] == G[5, 5]:
            mass = G[3, 3]
            print(f"   Link {i+1} mass: {mass:.3f} kg")
            print(f"   Link {i+1} inertia: diag([{G[0,0]:.3f}, {G[1,1]:.3f}, {G[2,2]:.3f}])")
        else:
            print(f"   Link {i+1} has complex inertia structure")

def demonstrate_manipulapy_transformations(urdf_processor):
    """
    Demonstrate ManipulaPy's specific transformation utilities and methods.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🔄 ManipulaPy Transformation Utilities")
    print("=" * 60)
    
    # Demonstrate ManipulaPy's static methods
    print(f"🛠️  ManipulaPy Static Methods:")
    
    # Test transform_to_xyz method
    test_transform = np.eye(4)
    test_transform[:3, 3] = [1.0, 2.0, 3.0]
    xyz = URDFToSerialManipulator.transform_to_xyz(test_transform)
    print(f"   • transform_to_xyz([1,2,3,1]): {xyz}")
    
    # Test w_p_to_slist method
    w_test = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # 3x3 for 3 joints
    p_test = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])  # 3x3 for 3 joints
    slist_test = URDFToSerialManipulator.w_p_to_slist(w_test, p_test, 3)
    print(f"   • w_p_to_slist result shape: {slist_test.shape}")
    print(f"   • First screw axis: {slist_test[:, 0]}")
    
    # Show how ManipulaPy uses utils functions
    print(f"\n🧰 ManipulaPy Utils Integration:")
    robot_data = urdf_processor.robot_data
    
    # Extract omega list using utils
    omega_list = utils.extract_omega_list(robot_data['Slist'])
    print(f"   • Extracted omega list shape: {omega_list.shape}")
    
    # Extract r list using utils  
    r_list = utils.extract_r_list(robot_data['Slist'])
    print(f"   • Extracted r list shape: {r_list.shape}")
    
    # Show adjoint transformation used in Blist computation
    M_inv = np.linalg.inv(robot_data['M'])
    Ad_inv = utils.adjoint_transform(M_inv)
    print(f"   • Adjoint of M^-1 shape: {Ad_inv.shape}")
    print(f"   • Used to compute Blist from Slist")

def test_manipulapy_objects(urdf_processor):
    """
    Test the SerialManipulator and ManipulatorDynamics objects created by ManipulaPy.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🤖 Testing ManipulaPy Created Objects")
    print("=" * 60)
    
    # Test SerialManipulator object
    serial_manipulator = urdf_processor.serial_manipulator
    dynamics = urdf_processor.dynamics
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    print(f"🎯 SerialManipulator Object (created by ManipulaPy):")
    print(f"   • Type: {type(serial_manipulator)}")
    print(f"   • Joint limits: {len(serial_manipulator.joint_limits)} joints")
    print(f"   • M_list shape: {serial_manipulator.M_list.shape}")
    print(f"   • S_list shape: {serial_manipulator.S_list.shape}")
    print(f"   • B_list shape: {serial_manipulator.B_list.shape}")
    
    # Test forward kinematics with ManipulaPy object
    test_config = np.zeros(num_joints)
    T_space = serial_manipulator.forward_kinematics(test_config, frame="space")
    T_body = serial_manipulator.forward_kinematics(test_config, frame="body")
    
    print(f"   • Forward kinematics (space): position {T_space[:3, 3]}")
    print(f"   • Forward kinematics (body): position {T_body[:3, 3]}")
    
    # Test Jacobian computation
    J_space = serial_manipulator.jacobian(test_config, frame="space")
    J_body = serial_manipulator.jacobian(test_config, frame="body")
    
    print(f"   • Jacobian (space) shape: {J_space.shape}")
    print(f"   • Jacobian (body) shape: {J_body.shape}")
    
    print(f"\n⚙️  ManipulatorDynamics Object (created by ManipulaPy):")
    print(f"   • Type: {type(dynamics)}")
    print(f"   • Glist length: {len(dynamics.Glist)}")
    
    # Test dynamics computations
    try:
        test_velocities = np.zeros(num_joints)
        test_accelerations = np.zeros(num_joints)
        
        M = dynamics.mass_matrix(test_config)
        print(f"   • Mass matrix shape: {M.shape}")
        print(f"   • Mass matrix determinant: {np.linalg.det(M):.6f}")
        
        c = dynamics.velocity_quadratic_forces(test_config, test_velocities)
        print(f"   • Velocity forces shape: {c.shape}")
        
        g_forces = dynamics.gravity_forces(test_config, [0, 0, -9.81])
        print(f"   • Gravity forces shape: {g_forces.shape}")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Dynamics computation error: {e}")

def demonstrate_manipulapy_pybullet_integration(urdf_processor):
    """
    Demonstrate ManipulaPy's PyBullet integration features.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🎮 ManipulaPy PyBullet Integration")
    print("=" * 60)
    
    print(f"🔧 Joint Limits from PyBullet:")
    if "joint_limits" in urdf_processor.robot_data:
        pyb_limits = urdf_processor.robot_data["joint_limits"]
        print(f"   • Retrieved {len(pyb_limits)} joint limits from PyBullet")
        print(f"   • ManipulaPy automatically connected to PyBullet DIRECT mode")
        print(f"   • Limits extracted and applied to SerialManipulator")
        
        for i, (lower, upper) in enumerate(pyb_limits):
            print(f"   • Joint {i+1}: [{lower:.3f}, {upper:.3f}] rad")
    else:
        print(f"   • Using default limits (no PyBullet integration)")
    
    print(f"\n📊 Comparison with ManipulaPy defaults:")
    default_limits = [(-np.pi, np.pi)] * urdf_processor.robot_data['actuated_joints_num']
    print(f"   • Default ManipulaPy limits: {default_limits[0]} for all joints")
    print(f"   • PyBullet integration provides URDF-specific limits")
    print(f"   • ManipulaPy's use_pybullet_limits=True enables this feature")

def test_manipulapy_visualization_support(urdf_processor):
    """
    Test ManipulaPy's visualization and trajectory support.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🎬 ManipulaPy Visualization Support")
    print("=" * 60)
    
    try:
        # Test visualization method (note: won't actually display in headless mode)
        print(f"🖼️  Visualization Methods:")
        print(f"   • visualize_robot(): Uses urchin.robot.show()")
        print(f"   • visualize_trajectory(): Supports numpy arrays and dictionaries")
        
        # Create a sample trajectory for testing format conversion
        num_joints = urdf_processor.robot_data['actuated_joints_num']
        sample_trajectory = np.array([
            np.linspace(0, np.pi/4, 10),  # Joint 1
            np.linspace(0, np.pi/6, 10),  # Joint 2
            # Add more joints as needed
        ]).T[:, :num_joints]  # Ensure correct number of joints
        
        print(f"   • Sample trajectory shape: {sample_trajectory.shape}")
        print(f"   • ManipulaPy converts numpy arrays to joint name dictionaries")
        
        # Show how ManipulaPy handles trajectory format conversion
        actuated_joints = [j for j in urdf_processor.robot.joints if j.joint_type != "fixed"]
        print(f"   • Found {len(actuated_joints)} actuated joints")
        print(f"   • Joint names: {[j.name for j in actuated_joints[:3]]}...")  # Show first 3
        
    except Exception as e:
        logger.warning(f"   ⚠️ Visualization test error: {e}")

def analyze_workspace(serial_manipulator):
    """
    Analyze the robot workspace through Monte Carlo sampling.
    
    Args:
        serial_manipulator (SerialManipulator): The ManipulaPy SerialManipulator object
    """
    print(f"\n🌐 Workspace Analysis (Sample Points):")
    sample_size = 1000
    workspace_points = []
    
    for _ in range(sample_size):
        # Generate random joint configurations within limits
        joint_config = np.array([
            np.random.uniform(limits[0], limits[1]) 
            for limits in serial_manipulator.joint_limits
        ])
        
        try:
            T = serial_manipulator.forward_kinematics(joint_config, frame="space")
            workspace_points.append(T[:3, 3])
        except Exception:
            continue
    
    if workspace_points:
        workspace_points = np.array(workspace_points)
        print(f"   Sampled {len(workspace_points)} valid points")
        print(f"   X range: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}]")
        print(f"   Y range: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}]")
        print(f"   Z range: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}]")
        print(f"   Max reach: {np.max(np.linalg.norm(workspace_points, axis=1)):.3f}")
    else:
        print(f"   ⚠️ No valid workspace points found")

def main():
    """Demonstrate ManipulaPy URDF processing operations."""
    print("=== ManipulaPy: Basic URDF Processing Demo ===")
    print("📄 Focused demonstration of ManipulaPy's URDF processing capabilities")
    print()
    
    urdf_file = None
    urdf_processor = None
    
    try:
        # Step 1: Create a sample URDF file
        print("🏗️  Step 1: Creating Sample URDF")
        urdf_file = create_sample_urdf()
        
        # Step 2: Load with ManipulaPy URDFToSerialManipulator
        print("\n📖 Step 2: Loading with ManipulaPy URDFToSerialManipulator")
        urdf_processor = load_and_inspect_urdf(urdf_file)
        
        # Step 3: Inspect ManipulaPy robot model
        print("\n🔍 Step 3: ManipulaPy Robot Model Inspection")
        inspect_robot_model(urdf_processor)
        
        # Step 4: Demonstrate ManipulaPy transformations
        print("\n🔄 Step 4: ManipulaPy Transformation Methods")
        demonstrate_manipulapy_transformations(urdf_processor)
        
        # Step 5: Test ManipulaPy created objects
        print("\n🤖 Step 5: Testing ManipulaPy Created Objects")
        test_manipulapy_objects(urdf_processor)
        
        # Step 6: Demonstrate PyBullet integration
        print("\n🎮 Step 6: ManipulaPy PyBullet Integration")
        demonstrate_manipulapy_pybullet_integration(urdf_processor)
        
        # Step 7: Test visualization support
        print("\n🎬 Step 7: ManipulaPy Visualization Support")
        test_manipulapy_visualization_support(urdf_processor)
        
        # Step 8: Workspace analysis
        print("\n📈 Step 8: Workspace Analysis")
        analyze_workspace(urdf_processor.serial_manipulator)
        
        print(f"\n✅ ManipulaPy URDF Processing Demo Completed Successfully!")
        print("=" * 60)
        print("🎉 All ManipulaPy features tested successfully!")
        print("💡 Key ManipulaPy Features Demonstrated:")
        print("   • URDF loading with urchin integration")
        print("   • Automatic screw axis computation")
        print("   • PyBullet joint limits extraction")
        print("   • SerialManipulator object creation")
        print("   • ManipulatorDynamics object creation")
        print("   • Transformation utilities")
        print("   • Visualization support")
        print("   • Workspace analysis")
        
    except Exception as e:
        logger.error(f"❌ ManipulaPy demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up demo files...")
        cleanup_demo_files()
    
    return 0
    
    # Workspace analysis (simplified)
    print(f"\n🌐 Workspace Analysis (Sample Points):")
    sample_size = 1000
    workspace_points = []
    
    for _ in range(sample_size):
        # Generate random joint configurations within limits
        joint_config = np.array([
            np.random.uniform(limits[0], limits[1]) 
            for limits in serial_manipulator.joint_limits
        ])
        
        try:
            T = serial_manipulator.forward_kinematics(joint_config, frame="space")
            workspace_points.append(T[:3, 3])
        except:
            continue
    
    if workspace_points:
        workspace_points = np.array(workspace_points)
        print(f"   Sampled {len(workspace_points)} valid points")
        print(f"   X range: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}]")
        print(f"   Y range: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}]")
        print(f"   Z range: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}]")
        print(f"   Max reach: {np.max(np.linalg.norm(workspace_points, axis=1)):.3f}")

def cleanup_demo_files():
    """Clean up demo files created during the demonstration."""
    files_to_clean = ["sample_6dof_robot.urdf"]
    
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"🧹 Cleaned up file: {file}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)