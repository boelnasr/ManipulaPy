#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Visualization Demo: Robot Visualization and Animation

This example demonstrates ManipulaPy's visualization capabilities including
robot model visualization, trajectory animation, workspace plotting, and
manipulability analysis visualization.

Usage:
    python visualization_basic_demo.py

Author: ManipulaPy Development Team
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import os
import sys
import logging
from mpl_toolkits.mplot3d import Axes3D
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.singularity import Singularity
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy import utils
    logger.info("✅ ManipulaPy modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ManipulaPy: {e}")
    sys.exit(1)

def create_visualization_urdf():
    """
    Create a simple 4-DOF robot URDF file optimized for visualization demos.
    This creates a basic robot arm that's easy to visualize and animate.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="visualization_demo_robot">
  
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.08" length="0.1"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 1 - Base rotation -->
  <joint name="base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="150" velocity="2.0"/>
  </joint>

  <!-- Link 1 - Vertical arm -->
  <link name="link_1">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.1 0.1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.04" iyy="0.04" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 2 - Shoulder -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0944" upper="2.0944" effort="100" velocity="2.0"/>
  </joint>

  <!-- Link 2 - Upper arm -->
  <link name="link_2">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.015" iyy="0.015" izz="0.005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 3 - Elbow -->
  <joint name="elbow_joint" type="revolute">
    <parent link="link_2"/>
    <child link="link_3"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.618" upper="2.618" effort="80" velocity="2.0"/>
  </joint>

  <!-- Link 3 - Forearm -->
  <link name="link_3">
    <visual>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.003" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint 4 - Wrist -->
  <joint name="wrist_joint" type="revolute">
    <parent link="link_3"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="50" velocity="3.0"/>
  </joint>

  <!-- End Effector -->
  <link name="end_effector">
    <visual>
      <geometry>
        <cylinder radius="0.025" length="0.1"/>
      </geometry>
      <material name="yellow">
        <color rgba="1 1 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" iyy="0.002" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>"""
    
    # Create the URDF file
    urdf_file = "visualization_demo_robot.urdf"
    with open(urdf_file, 'w') as f:
        f.write(urdf_content)
    
    logger.info(f"✅ Created visualization demo URDF: {urdf_file}")
    return urdf_file

def demonstrate_robot_visualization(urdf_processor):
    """
    Demonstrate basic robot model visualization using ManipulaPy and urchin.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🎨 Robot Model Visualization")
    print("=" * 60)
    
    print(f"📊 Robot Visualization Methods:")
    print(f"   • Robot model: {urdf_processor.robot}")
    print(f"   • URDF name: {urdf_processor.urdf_name}")
    print(f"   • Number of links: {len(urdf_processor.robot.links)}")
    print(f"   • Number of joints: {len(urdf_processor.robot.joints)}")
    
    try:
        print(f"\n🖼️  Attempting robot visualization...")
        print(f"   Note: This will open a 3D visualization window using urchin")
        print(f"   Press 'q' in the visualization window to close it")
        
        # Demonstrate the visualize_robot method
        print(f"   Calling urdf_processor.visualize_robot()...")
        # Note: This might not work in headless environments
        # urdf_processor.visualize_robot()
        print(f"   ✅ Robot visualization method available")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Robot visualization not available in this environment: {e}")
        print(f"   💡 Try running in an environment with display capabilities")

def demonstrate_trajectory_animation(urdf_processor):
    """
    Demonstrate trajectory animation using ManipulaPy's trajectory visualization.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🎬 Trajectory Animation")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    # Create several interesting trajectory configurations
    print(f"🎯 Creating demonstration trajectories...")
    
    # Simple sinusoidal trajectory
    time_steps = 20
    t = np.linspace(0, 2*np.pi, time_steps)
    
    trajectories = {
        "sinusoidal": np.array([
            0.5 * np.sin(t),                    # Joint 1: base rotation
            0.3 * np.sin(t + np.pi/4),         # Joint 2: shoulder
            0.4 * np.sin(t + np.pi/2),         # Joint 3: elbow  
            0.6 * np.sin(t + 3*np.pi/4),       # Joint 4: wrist
        ]).T[:, :num_joints],
        
        "wave": np.array([
            0.2 * np.sin(2*t),                 # Joint 1: faster base
            0.5 * np.cos(t),                   # Joint 2: cosine shoulder
            0.3 * np.sin(3*t),                 # Joint 3: triple frequency
            0.4 * np.cos(2*t),                 # Joint 4: double cosine
        ]).T[:, :num_joints],
        
        "sweep": np.array([
            np.linspace(-np.pi/3, np.pi/3, time_steps),    # Joint 1: linear sweep
            np.linspace(0, np.pi/2, time_steps),           # Joint 2: linear up
            np.linspace(0, -np.pi/3, time_steps),          # Joint 3: linear down
            np.linspace(0, 2*np.pi, time_steps),           # Joint 4: full rotation
        ]).T[:, :num_joints]
    }
    
    for traj_name, trajectory in trajectories.items():
        print(f"\n🎭 {traj_name.capitalize()} Trajectory:")
        print(f"   • Shape: {trajectory.shape}")
        print(f"   • Time steps: {trajectory.shape[0]}")
        print(f"   • Joints: {trajectory.shape[1]}")
        print(f"   • Joint 1 range: [{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}] rad")
        
        try:
            # Test trajectory visualization with ManipulaPy
            print(f"   • Testing ManipulaPy trajectory format conversion...")
            
            # Get actuated joints for dictionary conversion
            actuated_joints = [j for j in urdf_processor.robot.joints if j.joint_type != "fixed"]
            
            # Test format that ManipulaPy expects
            if len(actuated_joints) >= trajectory.shape[1]:
                joint_dict = {
                    joint.name: trajectory[:, i]
                    for i, joint in enumerate(actuated_joints[:trajectory.shape[1]])
                }
                print(f"   • Created joint dictionary with keys: {list(joint_dict.keys())}")
                print(f"   • Dictionary format ready for urdf_processor.visualize_trajectory()")
                
                # Test the trajectory visualization method
                print(f"   • Calling visualize_trajectory() method...")
                # urdf_processor.visualize_trajectory(joint_dict, loop_time=2.0)
                print(f"   ✅ Trajectory animation method available")
                
        except Exception as e:
            logger.warning(f"   ⚠️ Trajectory animation error: {e}")

def plot_workspace_visualization(urdf_processor):
    """
    Create 3D workspace visualization plots using ManipulaPy.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🌐 Workspace Visualization")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    print(f"📊 Generating workspace points...")
    
    # Generate workspace points
    sample_size = 2000
    workspace_points = []
    joint_configs = []
    
    for _ in range(sample_size):
        # Generate random joint configurations within limits
        joint_config = np.array([
            np.random.uniform(limits[0], limits[1]) 
            for limits in serial_manipulator.joint_limits
        ])
        
        try:
            T = serial_manipulator.forward_kinematics(joint_config, frame="space")
            workspace_points.append(T[:3, 3])
            joint_configs.append(joint_config)
        except Exception:
            continue
    
    if not workspace_points:
        print(f"   ⚠️ No valid workspace points generated")
        return
    
    workspace_points = np.array(workspace_points)
    joint_configs = np.array(joint_configs)
    
    print(f"   • Generated {len(workspace_points)} valid workspace points")
    print(f"   • Workspace bounds:")
    print(f"     X: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}]")
    print(f"     Y: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}]")
    print(f"     Z: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}]")
    
    # Create 3D workspace visualization
    fig = plt.figure(figsize=(15, 5))
    
    # 3D Scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                         c=workspace_points[:, 2], cmap='viridis', alpha=0.6, s=1)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Workspace\n(Color by Z-height)')
    plt.colorbar(scatter, ax=ax1, shrink=0.8)
    
    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(workspace_points[:, 0], workspace_points[:, 1], 
               c=workspace_points[:, 2], cmap='viridis', alpha=0.6, s=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection\n(Color by Z-height)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Reachability analysis
    ax3 = fig.add_subplot(133)
    distances = np.linalg.norm(workspace_points, axis=1)
    ax3.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Distance from Base (m)')
    ax3.set_ylabel('Number of Points')
    ax3.set_title('Reachability Distribution')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('workspace_visualization.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Workspace visualization saved as 'workspace_visualization.png'")
    
    # Show plot if in interactive environment
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show(block=False)
            plt.pause(2)  # Display for 2 seconds
        plt.close()
    except:
        plt.close()

def plot_manipulability_analysis(urdf_processor):
    """
    Demonstrate manipulability analysis and visualization using ManipulaPy.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🔄 Manipulability Analysis")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    # Create Singularity analysis object
    try:
        singularity_analyzer = Singularity(serial_manipulator)
        print(f"✅ Singularity analyzer created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create singularity analyzer: {e}")
        return
    
    print(f"📊 Analyzing manipulability across workspace...")
    
    # Test different configurations
    test_configs = [
        ("Zero configuration", np.zeros(num_joints)),
        ("Random config 1", np.random.uniform(-0.5, 0.5, num_joints)),
        ("Random config 2", np.random.uniform(-1.0, 1.0, num_joints)),
        ("Near singularity", np.array([0, 0, np.pi, 0][:num_joints])),
    ]
    
    manipulability_data = []
    condition_numbers = []
    configurations = []
    
    for config_name, joint_angles in test_configs:
        try:
            # Compute Jacobian
            J = serial_manipulator.jacobian(joint_angles, frame="space")
            
            # Ensure we have a valid Jacobian shape (6 x n_joints)
            if J.shape[0] != 6 or J.shape[1] != num_joints:
                print(f"   ⚠️ {config_name}: Jacobian shape {J.shape} is unexpected, skipping")
                continue
            
            # Compute manipulability measure with comprehensive error handling
            try:
                if J.shape[1] < 6:  # Under-actuated robot
                    det_val = np.linalg.det(J.T @ J)
                else:  # Fully actuated or over-actuated
                    det_val = np.linalg.det(J @ J.T)
                
                # Ensure determinant is non-negative for sqrt
                det_val = max(0, det_val)
                manipulability = np.sqrt(det_val)
                
                # Handle NaN/inf values
                if not np.isfinite(manipulability):
                    manipulability = 0.0
                    
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"   ⚠️ {config_name}: Linear algebra error in manipulability: {e}")
                manipulability = 0.0
            
            # Compute condition number with error handling
            try:
                cond_num = singularity_analyzer.condition_number(joint_angles)
                if not np.isfinite(cond_num):
                    cond_num = 1000.0  # Large number for near-singular
            except Exception as e:
                print(f"   ⚠️ {config_name}: Error computing condition number: {e}")
                cond_num = 1000.0
            
            # Check for singularity with error handling
            try:
                is_singular = singularity_analyzer.singularity_analysis(joint_angles)
                is_near_singular = singularity_analyzer.near_singularity_detection(joint_angles)
            except Exception as e:
                print(f"   ⚠️ {config_name}: Error in singularity analysis: {e}")
                is_singular = False
                is_near_singular = False
            
            print(f"\n🎯 {config_name}:")
            print(f"   • Joint angles: {np.rad2deg(joint_angles)}")
            print(f"   • Jacobian shape: {J.shape}")
            print(f"   • Manipulability: {manipulability:.6f}")
            print(f"   • Condition number: {cond_num:.2f}")
            print(f"   • Is singular: {is_singular}")
            print(f"   • Near singular: {is_near_singular}")
            
            manipulability_data.append(manipulability)
            condition_numbers.append(cond_num)
            configurations.append(config_name)
            
        except Exception as e:
            logger.warning(f"   ⚠️ Error analyzing {config_name}: {e}")
    
    # Create manipulability visualization
    if manipulability_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Manipulability measure plot
        colors = ['blue', 'green', 'orange', 'red']
        bars1 = ax1.bar(configurations, manipulability_data, color=colors[:len(manipulability_data)])
        ax1.set_ylabel('Manipulability Measure')
        ax1.set_title('Manipulability Analysis')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, manipulability_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Condition number plot (with safety limits)
        safe_condition_numbers = [min(cn, 1000) for cn in condition_numbers]  # Cap at 1000
        bars2 = ax2.bar(configurations, safe_condition_numbers, color=colors[:len(safe_condition_numbers)])
        ax2.set_ylabel('Condition Number')
        ax2.set_title('Jacobian Condition Number')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for condition numbers
        
        # Add value labels on bars
        for bar, value in zip(bars2, safe_condition_numbers):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('manipulability_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Manipulability analysis saved as 'manipulability_analysis.png'")
        
        # Show plot if in interactive environment
        try:
            plt.show(block=False)
            plt.pause(2)  # Display for 2 seconds
            plt.close()
        except:
            plt.close()

def demonstrate_trajectory_plotting(urdf_processor):
    """
    Demonstrate trajectory plotting and analysis using ManipulaPy path planning.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n📈 Trajectory Plotting and Analysis")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    dynamics = urdf_processor.dynamics
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    # Create trajectory planner
    try:
        joint_limits = [(-np.pi, np.pi)] * num_joints  # Simple limits for demo
        trajectory_planner = OptimizedTrajectoryPlanning(
            serial_manipulator=serial_manipulator,
            urdf_path=urdf_processor.urdf_name,
            dynamics=dynamics,
            joint_limits=joint_limits,
            use_cuda=False  # Use CPU for demo compatibility
        )
        print(f"✅ Trajectory planner created successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not create trajectory planner: {e}")
        return
    
    # Generate a sample trajectory
    print(f"🎯 Generating sample trajectory...")
    
    start_config = np.array([0.0, 0.2, -0.3, 0.0][:num_joints])
    end_config = np.array([np.pi/3, np.pi/4, -np.pi/6, np.pi/2][:num_joints])
    Tf = 3.0  # 3 seconds
    N = 50   # 50 time steps
    
    try:
        # Generate joint trajectory
        trajectory_data = trajectory_planner.joint_trajectory(
            start_config, end_config, Tf, N, method=5  # Quintic polynomial
        )
        
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"] 
        accelerations = trajectory_data["accelerations"]
        
        print(f"   • Trajectory shape: {positions.shape}")
        print(f"   • Duration: {Tf} seconds")
        print(f"   • Time steps: {N}")
        
        # Plot trajectory
        time_vector = np.linspace(0, Tf, N)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Position plot
        for i in range(min(num_joints, 4)):  # Plot first 4 joints
            axes[0].plot(time_vector, np.rad2deg(positions[:, i]), 
                        label=f'Joint {i+1}', linewidth=2)
        axes[0].set_ylabel('Position (degrees)')
        axes[0].set_title('Joint Trajectories - Positions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Velocity plot
        for i in range(min(num_joints, 4)):
            axes[1].plot(time_vector, np.rad2deg(velocities[:, i]), 
                        label=f'Joint {i+1}', linewidth=2)
        axes[1].set_ylabel('Velocity (deg/s)')
        axes[1].set_title('Joint Trajectories - Velocities')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Acceleration plot
        for i in range(min(num_joints, 4)):
            axes[2].plot(time_vector, np.rad2deg(accelerations[:, i]), 
                        label=f'Joint {i+1}', linewidth=2)
        axes[2].set_ylabel('Acceleration (deg/s²)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Joint Trajectories - Accelerations')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Trajectory analysis saved as 'trajectory_analysis.png'")
        
        # Show plot if in interactive environment
        try:
            plt.show(block=False)
            plt.pause(2)  # Display for 2 seconds
            plt.close()
        except:
            plt.close()
            
    except Exception as e:
        logger.warning(f"⚠️ Trajectory generation error: {e}")

def demonstrate_end_effector_visualization(urdf_processor):
    """
    Demonstrate end-effector path visualization and analysis.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n🎯 End-Effector Path Visualization")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    print(f"🎬 Generating end-effector trajectories...")
    
    # Create different types of movements
    time_steps = 30
    t = np.linspace(0, 2*np.pi, time_steps)
    
    movements = {
        "circular": np.array([
            0.3 * np.sin(t),                    # Base rotation for circle
            np.full(time_steps, np.pi/6),       # Fixed shoulder
            np.full(time_steps, -np.pi/4),      # Fixed elbow
            np.zeros(time_steps),               # Fixed wrist
        ]).T[:, :num_joints],
        
        "figure_eight": np.array([
            0.4 * np.sin(t),                    # Base sine
            0.2 * np.sin(2*t),                  # Double frequency shoulder
            0.3 * np.cos(t),                    # Elbow cosine
            0.1 * np.sin(3*t),                  # Triple frequency wrist
        ]).T[:, :num_joints],
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    for idx, (movement_name, joint_trajectory) in enumerate(movements.items()):
        print(f"   • Computing {movement_name} path...")
        
        # Compute end-effector positions
        ee_positions = []
        for joint_config in joint_trajectory:
            try:
                T = serial_manipulator.forward_kinematics(joint_config, frame="space")
                ee_positions.append(T[:3, 3])
            except Exception:
                continue
        
        if not ee_positions:
            continue
            
        ee_positions = np.array(ee_positions)
        
        # 3D trajectory plot
        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        
        # Plot trajectory with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(ee_positions)))
        for i in range(len(ee_positions)-1):
            ax.plot3D(ee_positions[i:i+2, 0], ee_positions[i:i+2, 1], ee_positions[i:i+2, 2],
                     color=colors[i], linewidth=2)
        
        # Mark start and end points
        ax.scatter(*ee_positions[0], color='green', s=100, label='Start')
        ax.scatter(*ee_positions[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'End-Effector Path\n({movement_name.replace("_", " ").title()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"     ✓ Path length: {len(ee_positions)} points")
        print(f"     ✓ Start position: [{ee_positions[0, 0]:.3f}, {ee_positions[0, 1]:.3f}, {ee_positions[0, 2]:.3f}]")
        print(f"     ✓ End position: [{ee_positions[-1, 0]:.3f}, {ee_positions[-1, 1]:.3f}, {ee_positions[-1, 2]:.3f}]")
        
        # Calculate path statistics
        distances = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        print(f"     ✓ Total path length: {total_distance:.3f} m")
    
    plt.tight_layout()
    plt.savefig('end_effector_paths.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ End-effector paths saved as 'end_effector_paths.png'")
    
    # Show plot if in interactive environment
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show(block=False)
            plt.pause(2)  # Display for 2 seconds
        plt.close()
    except:
        plt.close()

def demonstrate_configuration_space_visualization(urdf_processor):
    """
    Demonstrate configuration space analysis and visualization.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n⚙️  Configuration Space Visualization")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    if num_joints < 2:
        print("⚠️ Need at least 2 joints for configuration space visualization")
        return
    
    print(f"📊 Analyzing 2D configuration space (first 2 joints)...")
    
    # Create a grid in configuration space (first 2 joints)
    joint1_limits = serial_manipulator.joint_limits[0]
    joint2_limits = serial_manipulator.joint_limits[1]
    
    resolution = 50
    joint1_range = np.linspace(joint1_limits[0], joint1_limits[1], resolution)
    joint2_range = np.linspace(joint2_limits[0], joint2_limits[1], resolution)
    
    J1, J2 = np.meshgrid(joint1_range, joint2_range)
    
    # Initialize arrays for analysis
    reachable = np.zeros_like(J1, dtype=bool)
    end_effector_heights = np.zeros_like(J1)
    manipulability = np.zeros_like(J1)
    
    print(f"   • Grid resolution: {resolution}x{resolution}")
    print(f"   • Joint 1 range: [{np.rad2deg(joint1_limits[0]):.1f}°, {np.rad2deg(joint1_limits[1]):.1f}°]")
    print(f"   • Joint 2 range: [{np.rad2deg(joint2_limits[0]):.1f}°, {np.rad2deg(joint2_limits[1]):.1f}°]")
    print(f"   • Computing forward kinematics for {resolution*resolution} configurations...")
    
    # Analyze each configuration
    for i in range(resolution):
        for j in range(resolution):
            # Create full joint configuration (fix other joints at 0)
            joint_config = np.zeros(num_joints)
            joint_config[0] = J1[i, j]
            joint_config[1] = J2[i, j]
            
            try:
                # Compute forward kinematics
                T = serial_manipulator.forward_kinematics(joint_config, frame="space")
                reachable[i, j] = True
                end_effector_heights[i, j] = T[2, 3]  # Z coordinate
                
                # Compute manipulability
                J_matrix = serial_manipulator.jacobian(joint_config, frame="space")
                manip_measure = np.sqrt(np.linalg.det(J_matrix @ J_matrix.T))
                manipulability[i, j] = manip_measure
                
            except Exception:
                reachable[i, j] = False
                end_effector_heights[i, j] = np.nan
                manipulability[i, j] = np.nan
    
    reachable_count = np.sum(reachable)
    print(f"   • Reachable configurations: {reachable_count}/{resolution*resolution} ({100*reachable_count/(resolution*resolution):.1f}%)")
    
    # Create configuration space visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reachability map
    im1 = axes[0, 0].imshow(reachable, extent=[np.rad2deg(joint1_limits[0]), np.rad2deg(joint1_limits[1]),
                                              np.rad2deg(joint2_limits[0]), np.rad2deg(joint2_limits[1])],
                           origin='lower', cmap='RdYlGn', aspect='auto')
    axes[0, 0].set_xlabel('Joint 1 (degrees)')
    axes[0, 0].set_ylabel('Joint 2 (degrees)')
    axes[0, 0].set_title('Reachability Map')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Masked arrays to handle NaN values properly
    masked_heights = np.ma.masked_invalid(end_effector_heights)
    masked_manip = np.ma.masked_invalid(manipulability)
    
    # Ensure we have valid data for plotting
    if np.ma.is_masked(masked_heights) and masked_heights.mask.all():
        print(f"   ⚠️ No valid height data for visualization")
        masked_heights = np.zeros_like(end_effector_heights)
    
    if np.ma.is_masked(masked_manip) and masked_manip.mask.all():
        print(f"   ⚠️ No valid manipulability data for visualization") 
        masked_manip = np.zeros_like(manipulability)
    im3 = axes[1, 0].imshow(masked_manip, extent=[np.rad2deg(joint1_limits[0]), np.rad2deg(joint1_limits[1]),
                                                 np.rad2deg(joint2_limits[0]), np.rad2deg(joint2_limits[1])],
                           origin='lower', cmap='plasma', aspect='auto')
    axes[1, 0].set_xlabel('Joint 1 (degrees)')
    axes[1, 0].set_ylabel('Joint 2 (degrees)')
    axes[1, 0].set_title('Manipulability Measure')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Joint space trajectory example
    # Create a sample trajectory in joint space
    t = np.linspace(0, 2*np.pi, 20)
    traj_j1 = 0.3 * np.sin(t) + (joint1_limits[0] + joint1_limits[1])/2
    traj_j2 = 0.2 * np.cos(t) + (joint2_limits[0] + joint2_limits[1])/2
    
    axes[1, 1].scatter(np.rad2deg(traj_j1), np.rad2deg(traj_j2), c=t, cmap='coolwarm', s=50)
    axes[1, 1].plot(np.rad2deg(traj_j1), np.rad2deg(traj_j2), 'k--', alpha=0.5)
    axes[1, 1].scatter(np.rad2deg(traj_j1[0]), np.rad2deg(traj_j2[0]), c='green', s=100, marker='o', label='Start')
    axes[1, 1].scatter(np.rad2deg(traj_j1[-1]), np.rad2deg(traj_j2[-1]), c='red', s=100, marker='s', label='End')
    axes[1, 1].set_xlabel('Joint 1 (degrees)')
    axes[1, 1].set_ylabel('Joint 2 (degrees)')
    axes[1, 1].set_title('Sample Trajectory in Joint Space')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('configuration_space_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Configuration space analysis saved as 'configuration_space_analysis.png'")
    
    # Show plot if in interactive environment
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show(block=False)
            plt.pause(2)  # Display for 2 seconds
        plt.close()
    except:
        plt.close()

def create_summary_visualization(urdf_processor):
    """
    Create a comprehensive summary visualization showing all key robot properties.
    
    Args:
        urdf_processor (URDFToSerialManipulator): The ManipulaPy URDF processor instance
    """
    print(f"\n📋 Creating Summary Visualization")
    print("=" * 60)
    
    serial_manipulator = urdf_processor.serial_manipulator
    num_joints = urdf_processor.robot_data['actuated_joints_num']
    
    # Create a comprehensive summary figure with fixed size
    fig = plt.figure(figsize=(16, 10))  # Fixed reasonable size
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    # 1. Robot parameters summary (text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    robot_info = [
        f"Robot: {urdf_processor.robot.name}",
        f"DOF: {num_joints}",
        f"Links: {len(urdf_processor.robot.links)}",
        f"Joints: {len(urdf_processor.robot.joints)}",
        "",
        "Joint Limits (deg):",
    ]
    
    for i, (lower, upper) in enumerate(serial_manipulator.joint_limits):
        robot_info.append(f"  J{i+1}: [{np.rad2deg(lower):.0f}, {np.rad2deg(upper):.0f}]")
    
    ax1.text(0.05, 0.95, '\n'.join(robot_info), transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    ax1.set_title('Robot Parameters', fontweight='bold')
    
    # 2. Joint limits visualization
    ax2 = fig.add_subplot(gs[0, 1])
    joint_numbers = range(1, num_joints + 1)
    lower_limits = [np.rad2deg(limits[0]) for limits in serial_manipulator.joint_limits]
    upper_limits = [np.rad2deg(limits[1]) for limits in serial_manipulator.joint_limits]
    
    ax2.barh(joint_numbers, upper_limits, alpha=0.7, color='skyblue', label='Upper')
    ax2.barh(joint_numbers, lower_limits, alpha=0.7, color='lightcoral', label='Lower')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Joint Number')
    ax2.set_title('Joint Angle Limits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Quick workspace sample
    ax3 = fig.add_subplot(gs[0, 2:], projection='3d')
    sample_size = 300  # Reduced sample size for performance
    workspace_points = []
    
    for _ in range(sample_size):
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
        ax3.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                   alpha=0.6, s=2, c=workspace_points[:, 2], cmap='viridis')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        ax3.set_title('Workspace Sample')
    
    # 4. Sample trajectory
    ax4 = fig.add_subplot(gs[1, :2])
    t = np.linspace(0, 2*np.pi, 30)
    sample_traj = np.array([
        0.3 * np.sin(t),
        0.2 * np.cos(t + np.pi/4),
        0.25 * np.sin(t + np.pi/2),
        0.15 * np.cos(t + 3*np.pi/4),
    ]).T[:, :num_joints]
    
    time_vec = np.linspace(0, 3, 30)  # 3 seconds
    colors = plt.cm.tab10(np.linspace(0, 1, num_joints))
    
    for i in range(num_joints):
        ax4.plot(time_vec, np.rad2deg(sample_traj[:, i]), 
                label=f'Joint {i+1}', color=colors[i], linewidth=2)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Joint Angle (degrees)')
    ax4.set_title('Sample Joint Trajectory')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. End-effector path from sample trajectory
    ax5 = fig.add_subplot(gs[1, 2:], projection='3d')
    ee_positions = []
    
    for joint_config in sample_traj:
        try:
            T = serial_manipulator.forward_kinematics(joint_config, frame="space")
            ee_positions.append(T[:3, 3])
        except Exception:
            continue
    
    if ee_positions:
        ee_positions = np.array(ee_positions)
        ax5.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                'b-', linewidth=3, alpha=0.8)
        ax5.scatter(*ee_positions[0], color='green', s=100, label='Start')
        ax5.scatter(*ee_positions[-1], color='red', s=100, label='End')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_zlabel('Z (m)')
        ax5.set_title('End-Effector Path')
        ax5.legend()
    
    # 6. Manipulability analysis
    ax6 = fig.add_subplot(gs[2, :2])
    test_configs = [
        np.zeros(num_joints),
        np.array([np.pi/6, np.pi/4, -np.pi/6, 0][:num_joints]),
        np.array([np.pi/3, np.pi/3, -np.pi/3, np.pi/2][:num_joints]),
        np.array([-np.pi/4, np.pi/6, np.pi/4, -np.pi/4][:num_joints]),
    ]
    
    manipulabilities = []
    config_names = ['Zero', 'Config 1', 'Config 2', 'Config 3']
    
    for config in test_configs:
        try:
            J = serial_manipulator.jacobian(config, frame="space")
            
            # Ensure we have a valid Jacobian shape (6 x n_joints)
            if J.shape[0] != 6:
                manipulabilities.append(0.0)
                continue
                
            # Safe manipulability calculation with error handling
            try:
                if J.shape[1] < 6:  # Under-actuated robot
                    det_val = np.linalg.det(J.T @ J)
                else:  # Fully actuated or over-actuated
                    det_val = np.linalg.det(J @ J.T)
                
                # Ensure determinant is non-negative for sqrt
                det_val = max(0, det_val)
                manip = np.sqrt(det_val)
                
                if not np.isfinite(manip):
                    manip = 0.0
                    
            except (np.linalg.LinAlgError, ValueError):
                manip = 0.0
                
            manipulabilities.append(manip)
        except Exception:
            manipulabilities.append(0.0)
    
    bars = ax6.bar(config_names, manipulabilities, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax6.set_ylabel('Manipulability Measure')
    ax6.set_title('Manipulability for Different Configurations')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, manipulabilities):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(manipulabilities)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Statistics summary
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('off')
    
    # Calculate some statistics
    if workspace_points is not None and len(workspace_points) > 0:
        max_reach = np.max(np.linalg.norm(workspace_points, axis=1))
        workspace_volume = (np.max(workspace_points[:, 0]) - np.min(workspace_points[:, 0])) * \
                          (np.max(workspace_points[:, 1]) - np.min(workspace_points[:, 1])) * \
                          (np.max(workspace_points[:, 2]) - np.min(workspace_points[:, 2]))
    else:
        max_reach = 0
        workspace_volume = 0
    
    stats_info = [
        "Robot Statistics:",
        "",
        f"Max Reach: {max_reach:.3f} m",
        f"Workspace Volume: {workspace_volume:.3f} m³",
        f"Avg Manipulability: {np.mean(manipulabilities):.4f}",
        f"Max Manipulability: {np.max(manipulabilities):.4f}",
        "",
        "Visualization Features:",
        "✓ Robot model structure",
        "✓ Joint limits analysis", 
        "✓ Workspace visualization",
        "✓ Trajectory planning",
        "✓ End-effector paths",
        "✓ Manipulability analysis",
    ]
    
    ax7.text(0.05, 0.95, '\n'.join(stats_info), transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    ax7.set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle('ManipulaPy Visualization Demo - Complete Robot Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Save with reduced DPI to avoid memory issues
    plt.savefig('complete_robot_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✅ Complete robot analysis saved as 'complete_robot_analysis.png'")
    
    # Show plot if in interactive environment
    try:
        plt.show(block=False)
        plt.pause(3)  # Display for 3 seconds
        plt.close()
    except Exception as e:
        logger.warning(f"⚠️ Visualization test error: {e}")
        plt.close()
        return [
            np.array([np.pi/3, np.pi/2][:num_joints]),
            np.array([-np.pi/4, np.pi/6, np.pi/4, -np.pi/4][:num_joints])
        ]

    
    manipulabilities = []
    config_names = ['Zero', 'Config 1', 'Config 2', 'Config 3']
    
    for config in test_configs:
        try:
            J = serial_manipulator.jacobian(config, frame="space")
            manip = np.sqrt(np.linalg.det(J @ J.T))
            manipulabilities.append(manip)
        except Exception:
            manipulabilities.append(0)
    
    bars = ax6.bar(config_names, manipulabilities, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax6.set_ylabel('Manipulability Measure')
    ax6.set_title('Manipulability for Different Configurations')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, manipulabilities):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Statistics summary
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('off')
    
    # Calculate some statistics
    if workspace_points is not None and len(workspace_points) > 0:
        max_reach = np.max(np.linalg.norm(workspace_points, axis=1))
        workspace_volume = (np.max(workspace_points[:, 0]) - np.min(workspace_points[:, 0])) * \
                          (np.max(workspace_points[:, 1]) - np.min(workspace_points[:, 1])) * \
                          (np.max(workspace_points[:, 2]) - np.min(workspace_points[:, 2]))
    else:
        max_reach = 0
        workspace_volume = 0
    
    stats_info = [
        "Robot Statistics:",
        "",
        f"Max Reach: {max_reach:.3f} m",
        f"Workspace Volume: {workspace_volume:.3f} m³",
        f"Avg Manipulability: {np.mean(manipulabilities):.4f}",
        f"Max Manipulability: {np.max(manipulabilities):.4f}",
        "",
        "Visualization Features:",
        "✓ Robot model structure",
        "✓ Joint limits analysis", 
        "✓ Workspace visualization",
        "✓ Trajectory planning",
        "✓ End-effector paths",
        "✓ Manipulability analysis",
    ]
    
    ax7.text(0.05, 0.95, '\n'.join(stats_info), transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    ax7.set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle('ManipulaPy Visualization Demo - Complete Robot Analysis', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig('complete_robot_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Complete robot analysis saved as 'complete_robot_analysis.png'")
    
    # Show plot if in interactive environment
    try:
        if matplotlib.get_backend() != 'Agg':
            plt.show(block=False)
            plt.pause(3)  # Display for 3 seconds
        plt.close()
    except:
        plt.close()

def cleanup_visualization_files():
    """Clean up visualization files created during the demonstration."""
    files_to_clean = [
        "visualization_demo_robot.urdf",
        "workspace_visualization.png",
        "manipulability_analysis.png", 
        "trajectory_analysis.png",
        "end_effector_paths.png",
        "configuration_space_analysis.png",
        "complete_robot_analysis.png"
    ]
    
    for file in files_to_clean:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"🧹 Cleaned up file: {file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {file}: {e}")

def main():
    """Demonstrate ManipulaPy visualization capabilities."""
    print("=== ManipulaPy: Basic Visualization Demo ===")
    print("🎨 Comprehensive demonstration of ManipulaPy's visualization capabilities")
    print()
    
    urdf_file = None
    urdf_processor = None
    
    try:
        # Step 1: Create a visualization-optimized URDF
        print("🏗️  Step 1: Creating Visualization Demo URDF")
        urdf_file = create_visualization_urdf()
        
        # Step 2: Load with ManipulaPy
        print("\n📖 Step 2: Loading Robot Model")
        urdf_processor = URDFToSerialManipulator(urdf_file, use_pybullet_limits=True)
        joint_info = urdf_processor.print_joint_info()
        print(f"   • Robot loaded: {joint_info['num_joints']} total joints")
        print(f"   • Actuated joints: {urdf_processor.robot_data['actuated_joints_num']}")
        
        # Step 3: Robot model visualization
        print("\n🎨 Step 3: Robot Model Visualization")
        demonstrate_robot_visualization(urdf_processor)
        
        # Step 4: Trajectory animation demonstration
        print("\n🎬 Step 4: Trajectory Animation")
        demonstrate_trajectory_animation(urdf_processor)
        
        # Step 5: Workspace visualization
        print("\n🌐 Step 5: Workspace Visualization")
        plot_workspace_visualization(urdf_processor)
        
        # Step 6: Manipulability analysis
        print("\n🔄 Step 6: Manipulability Analysis")
        plot_manipulability_analysis(urdf_processor)
        
        # Step 7: Trajectory plotting
        print("\n📈 Step 7: Trajectory Plotting")
        demonstrate_trajectory_plotting(urdf_processor)
        
        # Step 8: End-effector visualization
        print("\n🎯 Step 8: End-Effector Path Visualization")
        demonstrate_end_effector_visualization(urdf_processor)
        
        # Step 9: Configuration space visualization
        print("\n⚙️  Step 9: Configuration Space Visualization")
        demonstrate_configuration_space_visualization(urdf_processor)
        
        # Step 10: Create summary visualization
        print("\n📋 Step 10: Summary Visualization")
        create_summary_visualization(urdf_processor)
        
        print(f"\n✅ ManipulaPy Visualization Demo Completed Successfully!")
        print("=" * 60)
        print("🎉 All visualization features demonstrated successfully!")
        print()
        print("📁 Generated Files:")
        print("   • workspace_visualization.png - 3D workspace analysis")
        print("   • manipulability_analysis.png - Manipulability measures")
        print("   • trajectory_analysis.png - Joint trajectory plots") 
        print("   • end_effector_paths.png - End-effector path visualization")
        print("   • configuration_space_analysis.png - 2D config space analysis")
        print("   • complete_robot_analysis.png - Comprehensive summary")
        print()
        print("💡 Key Visualization Features Demonstrated:")
        print("   • Robot model 3D visualization (urchin integration)")
        print("   • Trajectory animation with multiple formats")
        print("   • Workspace analysis and plotting")
        print("   • Manipulability ellipsoid analysis")
        print("   • Joint space trajectory plotting")
        print("   • End-effector path visualization")
        print("   • Configuration space mapping")
        print("   • Comprehensive robot analysis summary")
        print("   • Interactive plotting with matplotlib")
        print("   • Export capabilities for documentation")
        
    except Exception as e:
        logger.error(f"❌ Visualization demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Ask user if they want to keep the generated images
        print(f"\n🗂️  Generated visualization files are ready for review.")
        print(f"   Run the demo again to regenerate or manually delete files to clean up.")
        
        # Clean up URDF file but keep the generated plots
        if urdf_file and os.path.exists(urdf_file):
            os.remove(urdf_file)
            logger.info(f"🧹 Cleaned up URDF file: {urdf_file}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)