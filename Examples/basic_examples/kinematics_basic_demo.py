#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Kinematics Demo: Forward and Inverse Kinematics Fundamentals

This example demonstrates fundamental kinematic operations for robotic manipulators including
forward kinematics computation, inverse kinematics solving, Jacobian analysis, and basic
visualization of robot configurations and workspace.

Usage:
    python kinematics_basic_demo.py

Expected Output:
    - Console output showing joint angles and end-effector poses
    - Matplotlib plots of robot configurations and Jacobian analysis
    - Convergence information for inverse kinematics
    - Workspace visualization and joint limit analysis

Author: ManipulaPy Development Team
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add ManipulaPy to path if needed
try:
    import ManipulaPy
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    import ManipulaPy

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
import matplotlib
matplotlib.use('TkAgg')

class KinematicsBasicDemo:
    """
    Comprehensive demonstration of basic kinematics operations.
    """
    
    def __init__(self):
        """Initialize the demo with robot model loading."""
        self.robot = None
        self.joint_limits = None
        self.n_joints = 0
        
    def run_demo(self):
        """Run the complete kinematics demonstration."""
        print("=" * 70)
        print("   ManipulaPy: Basic Kinematics Demo")
        print("=" * 70)
        print()
        
        # Step 1: Load robot model
        if not self.load_robot_model():
            return False
            
        # Step 2: Demonstrate forward kinematics
        self.demonstrate_forward_kinematics()
        
        # Step 3: Demonstrate inverse kinematics
        self.demonstrate_inverse_kinematics()
        
        # Step 4: Demonstrate Jacobian analysis
        self.demonstrate_jacobian_analysis()
        
        # Step 5: Demonstrate workspace analysis
        self.demonstrate_workspace_analysis()
        
        # Step 6: Create comprehensive visualizations
        self.create_visualizations()
        
        print("\n" + "=" * 70)
        print("✅ Kinematics demo completed successfully!")
        print("📊 Check the generated plots for detailed analysis")
        print("=" * 70)
        
        return True
    
    def load_robot_model(self):
        """Load and initialize robot model."""
        print("🤖 Loading Robot Model")
        print("-" * 30)
        
        # Try to load built-in robot models
        urdf_file = None
        robot_name = "Unknown"
        
        try:
            from ManipulaPy.ManipulaPy_data.xarm import urdf_file
            robot_name = "xArm 6-DOF"
            print(f"📁 Using built-in {robot_name} model")
        except ImportError:
            try:
                from ManipulaPy.ManipulaPy_data.ur5 import urdf_file
                robot_name = "UR5"
                print(f"📁 Using built-in {robot_name} model")
            except ImportError:
                print("❌ No built-in robot models found!")
                print("💡 Please ensure ManipulaPy is properly installed with robot data.")
                print("💡 You can also provide your own URDF file path.")
                return False
        
        try:
            # Process URDF and create robot model
            print(f"⚙️ Processing URDF file...")
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator
            self.joint_limits = np.array(self.robot.joint_limits)
            self.n_joints = len(self.joint_limits)
            
            print(f"✅ {robot_name} loaded successfully!")
            print(f"   • Number of joints: {self.n_joints}")
            print(f"   • Joint limits:")
            
            for i, (min_val, max_val) in enumerate(self.joint_limits):
                range_deg = np.degrees(max_val - min_val)
                print(f"     Joint {i+1}: [{min_val:.2f}, {max_val:.2f}] rad "
                      f"(range: {range_deg:.1f}°)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading robot model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demonstrate_forward_kinematics(self):
        """Demonstrate forward kinematics with multiple configurations."""
        print(f"\n🔧 Forward Kinematics Demonstration")
        print("-" * 40)
        
        # Test configurations
        test_configs = {
            "Home (Zero)": np.zeros(self.n_joints),
            "Mid-range": np.array([np.mean(limits) for limits in self.joint_limits]),
            "Random": np.random.uniform(
                self.joint_limits[:, 0], 
                self.joint_limits[:, 1]
            ),
            "Test Pose": self._generate_safe_test_pose()
        }
        
        self.fk_results = {}
        
        for config_name, joint_angles in test_configs.items():
            print(f"\n📍 Configuration: {config_name}")
            print(f"   Joint angles: {joint_angles}")
            
            # Compute forward kinematics
            start_time = time.time()
            T_end = self.robot.forward_kinematics(joint_angles)
            fk_time = time.time() - start_time
            
            # Extract position and orientation
            position = T_end[:3, 3]
            rotation_matrix = T_end[:3, :3]
            
            # Convert rotation matrix to Euler angles (ZYX convention)
            euler_angles = self._rotation_matrix_to_euler_zyx(rotation_matrix)
            
            print(f"   End-effector position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
            print(f"   End-effector orientation (ZYX Euler): [{np.degrees(euler_angles[0]):.1f}°, "
                  f"{np.degrees(euler_angles[1]):.1f}°, {np.degrees(euler_angles[2]):.1f}°]")
            print(f"   Computation time: {fk_time*1000:.2f} ms")
            
            # Store results for visualization
            self.fk_results[config_name] = {
                'joint_angles': joint_angles,
                'position': position,
                'orientation': euler_angles,
                'transform': T_end,
                'computation_time': fk_time
            }
        
        # Performance analysis
        avg_time = np.mean([result['computation_time'] for result in self.fk_results.values()])
        print(f"\n📊 Forward Kinematics Performance:")
        print(f"   Average computation time: {avg_time*1000:.2f} ms")
        print(f"   Frequency capability: {1/avg_time:.0f} Hz")
    
    def demonstrate_inverse_kinematics(self):
        """Demonstrate inverse kinematics with different target poses."""
        print(f"\n🎯 Inverse Kinematics Demonstration")
        print("-" * 40)
        
        # Use forward kinematics results as IK targets
        target_configs = list(self.fk_results.keys())[1:3]  # Skip home, use mid-range and random
        
        self.ik_results = {}
        
        for config_name in target_configs:
            fk_result = self.fk_results[config_name]
            target_pose = fk_result['transform']
            target_position = fk_result['position']
            
            print(f"\n🎯 Target: {config_name}")
            print(f"   Target position: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}] m")
            
            # Use smart IK to pick an initial guess automatically
            start_time = time.time()
            solution, success, iterations = self.robot.smart_inverse_kinematics(
                T_desired=target_pose,
                strategy="workspace_heuristic",
                eomg=1e-6,      # Orientation tolerance (rad)
                ev=1e-6,        # Position tolerance (m)
                max_iterations=5000,
                plot_residuals=False,
                damping=1e-2,   # Damping for numerical stability (5e-2 is default)
                step_cap=0.3    # Maximum step size per iteration (rad)
            )
            ik_time = time.time() - start_time
            
            if success:
                print(f"   ✅ IK converged in {iterations} iterations ({ik_time*1000:.1f} ms) using smart IK")
                print(f"   Solution: {solution}")
                
                # Verify the solution
                verification_pose = self.robot.forward_kinematics(solution)
                position_error = np.linalg.norm(verification_pose[:3, 3] - target_position)
                orientation_error = self._compute_orientation_error(verification_pose[:3, :3], target_pose[:3, :3])
                
                print(f"   Verification:")
                print(f"     Position error: {position_error:.2e} m")
                print(f"     Orientation error: {orientation_error:.2e} rad")
                
                self.ik_results[config_name] = {
                    'target_pose': target_pose,
                    'solution': solution,
                    'iterations': iterations,
                    'computation_time': ik_time,
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'success': True
                }
                
            else:
                print(f"   ❌ IK failed to converge after {iterations} iterations")
                self.ik_results[config_name] = {
                    'success': False,
                    'iterations': iterations,
                    'computation_time': ik_time
                }
        
        # IK Performance Summary
        successful_iks = [result for result in self.ik_results.values() if result['success']]
        if successful_iks:
            avg_iterations = np.mean([result['iterations'] for result in successful_iks])
            avg_time = np.mean([result['computation_time'] for result in successful_iks])
            avg_position_error = np.mean([result['position_error'] for result in successful_iks])
            
            print(f"\n📊 Inverse Kinematics Performance:")
            print(f"   Success rate: {len(successful_iks)}/{len(self.ik_results)} ({100*len(successful_iks)/len(self.ik_results):.1f}%)")
            print(f"   Average iterations: {avg_iterations:.1f}")
            print(f"   Average computation time: {avg_time*1000:.1f} ms")
            print(f"   Average position accuracy: {avg_position_error:.2e} m")
    
    def demonstrate_jacobian_analysis(self):
        """Demonstrate Jacobian computation and analysis."""
        print(f"\n📐 Jacobian Analysis Demonstration")
        print("-" * 40)
        
        self.jacobian_results = {}
        
        # Analyze Jacobian at different configurations
        for config_name, fk_result in self.fk_results.items():
            joint_angles = fk_result['joint_angles']
            
            print(f"\n📐 Configuration: {config_name}")
            
            # Compute Jacobian matrices
            start_time = time.time()
            J_space = self.robot.jacobian(joint_angles, frame="space")
            J_body = self.robot.jacobian(joint_angles, frame="body")
            jacobian_time = time.time() - start_time
            
            # Analyze Jacobian properties
            det_J_space = np.linalg.det(J_space @ J_space.T)  # For non-square Jacobians
            cond_J_space = np.linalg.cond(J_space)
            
            # Compute manipulability
            manipulability = np.sqrt(det_J_space) if det_J_space >= 0 else 0
            
            # Singular value decomposition
            U, sigma, Vh = np.linalg.svd(J_space)
            min_singular_value = np.min(sigma)
            max_singular_value = np.max(sigma)
            
            print(f"   Jacobian shape: {J_space.shape}")
            print(f"   Condition number: {cond_J_space:.2f}")
            print(f"   Manipulability: {manipulability:.6f}")
            print(f"   Singular values: [{np.min(sigma):.4f}, {np.max(sigma):.4f}]")
            print(f"   Computation time: {jacobian_time*1000:.2f} ms")
            
            # Check for near-singularity
            is_near_singular = min_singular_value < 1e-3
            if is_near_singular:
                print(f"   ⚠️ Near singular configuration (min σ = {min_singular_value:.2e})")
            else:
                print(f"   ✅ Well-conditioned configuration")
            
            self.jacobian_results[config_name] = {
                'J_space': J_space,
                'J_body': J_body,
                'condition_number': cond_J_space,
                'manipulability': manipulability,
                'singular_values': sigma,
                'min_singular_value': min_singular_value,
                'max_singular_value': max_singular_value,
                'is_near_singular': is_near_singular,
                'computation_time': jacobian_time
            }
        
        # Jacobian Performance Summary
        avg_time = np.mean([result['computation_time'] for result in self.jacobian_results.values()])
        print(f"\n📊 Jacobian Analysis Performance:")
        print(f"   Average computation time: {avg_time*1000:.2f} ms")
        print(f"   Frequency capability: {1/avg_time:.0f} Hz")
    
    def demonstrate_workspace_analysis(self):
        """Demonstrate basic workspace analysis."""
        print(f"\n🌍 Workspace Analysis Demonstration")
        print("-" * 40)
        
        print("🔍 Sampling robot workspace...")
        
        # Generate random configurations within joint limits
        n_samples = 1000
        workspace_points = []
        
        start_time = time.time()
        for i in range(n_samples):
            # Generate random joint configuration
            joint_config = np.random.uniform(
                self.joint_limits[:, 0], 
                self.joint_limits[:, 1]
            )
            
            # Compute forward kinematics
            T = self.robot.forward_kinematics(joint_config)
            workspace_points.append(T[:3, 3])
        
        sampling_time = time.time() - start_time
        workspace_points = np.array(workspace_points)
        
        # Analyze workspace properties
        workspace_center = np.mean(workspace_points, axis=0)
        workspace_std = np.std(workspace_points, axis=0)
        workspace_range = np.max(workspace_points, axis=0) - np.min(workspace_points, axis=0)
        
        print(f"✅ Sampled {n_samples} workspace points in {sampling_time:.2f} seconds")
        print(f"   Workspace center: [{workspace_center[0]:.3f}, {workspace_center[1]:.3f}, {workspace_center[2]:.3f}] m")
        print(f"   Workspace range: [{workspace_range[0]:.3f}, {workspace_range[1]:.3f}, {workspace_range[2]:.3f}] m")
        print(f"   Workspace volume (approx): {np.prod(workspace_range):.3f} m³")
        
        self.workspace_results = {
            'points': workspace_points,
            'center': workspace_center,
            'range': workspace_range,
            'std': workspace_std,
            'sampling_time': sampling_time
        }
    
    def create_visualizations(self):
        """Create comprehensive visualization plots."""
        print(f"\n📊 Creating Visualization Plots")
        print("-" * 40)
        
        try:
            # Create comprehensive figure
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle('ManipulaPy: Basic Kinematics Demo - Comprehensive Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Joint Configurations Comparison
            ax1 = plt.subplot(3, 4, 1)
            self._plot_joint_configurations(ax1)
            
            # Plot 2: End-Effector Positions
            ax2 = plt.subplot(3, 4, 2)
            self._plot_end_effector_positions(ax2)
            
            # Plot 3: Jacobian Condition Numbers
            ax3 = plt.subplot(3, 4, 3)
            self._plot_jacobian_analysis(ax3)
            
            # Plot 4: Manipulability Analysis
            ax4 = plt.subplot(3, 4, 4)
            self._plot_manipulability_analysis(ax4)
            
            # Plot 5: Workspace 3D Scatter
            ax5 = plt.subplot(3, 4, 5, projection='3d')
            self._plot_workspace_3d(ax5)
            
            # Plot 6: Workspace Projections
            ax6 = plt.subplot(3, 4, 6)
            self._plot_workspace_projections(ax6)
            
            # Plot 7: IK Convergence Analysis
            ax7 = plt.subplot(3, 4, 7)
            self._plot_ik_convergence(ax7)
            
            # Plot 8: Jacobian Heatmap
            ax8 = plt.subplot(3, 4, 8)
            self._plot_jacobian_heatmap(ax8)
            
            # Plot 9: Joint Limits Visualization
            ax9 = plt.subplot(3, 4, 9)
            self._plot_joint_limits(ax9)
            
            # Plot 10: Performance Metrics
            ax10 = plt.subplot(3, 4, 10)
            self._plot_performance_metrics(ax10)
            
            # Plot 11: Singular Values Analysis
            ax11 = plt.subplot(3, 4, 11)
            self._plot_singular_values(ax11)
            
            # Plot 12: Error Analysis
            ax12 = plt.subplot(3, 4, 12)
            self._plot_error_analysis(ax12)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Visualization plots created successfully!")
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
            print("This might be due to missing display or matplotlib backend issues.")
    
    def _plot_joint_configurations(self, ax):
        """Plot joint configurations comparison."""
        configs = list(self.fk_results.keys())
        joint_angles_data = [self.fk_results[config]['joint_angles'] for config in configs]
        
        x = np.arange(self.n_joints)
        width = 0.2
        
        for i, (config, angles) in enumerate(zip(configs, joint_angles_data)):
            ax.bar(x + i*width, np.degrees(angles), width, label=config, alpha=0.8)
        
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Joint Angle (degrees)')
        ax.set_title('Joint Configurations Comparison')
        ax.set_xticks(x + width * (len(configs)-1)/2)
        ax.set_xticklabels([f'J{i+1}' for i in range(self.n_joints)])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_end_effector_positions(self, ax):
        """Plot end-effector positions."""
        configs = list(self.fk_results.keys())
        positions = [self.fk_results[config]['position'] for config in configs]
        
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions]
        z_pos = [pos[2] for pos in positions]
        
        ax.scatter(x_pos, y_pos, c=z_pos, cmap='viridis', s=100, alpha=0.8)
        
        for i, config in enumerate(configs):
            ax.annotate(config, (x_pos[i], y_pos[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('End-Effector Positions (Z as color)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add colorbar
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Z Position (m)')
    
    def _plot_jacobian_analysis(self, ax):
        """Plot Jacobian condition numbers."""
        configs = list(self.jacobian_results.keys())
        condition_numbers = [self.jacobian_results[config]['condition_number'] for config in configs]
        
        bars = ax.bar(configs, condition_numbers, alpha=0.8, color='skyblue')
        ax.set_ylabel('Condition Number')
        ax.set_title('Jacobian Condition Numbers')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, condition_numbers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_manipulability_analysis(self, ax):
        """Plot manipulability analysis."""
        configs = list(self.jacobian_results.keys())
        manipulability = [self.jacobian_results[config]['manipulability'] for config in configs]
        
        bars = ax.bar(configs, manipulability, alpha=0.8, color='lightcoral')
        ax.set_ylabel('Manipulability')
        ax.set_title('Manipulability Index')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, manipulability):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_workspace_3d(self, ax):
        """Plot 3D workspace."""
        points = self.workspace_results['points']
        
        # Subsample for better visualization
        n_plot = min(500, len(points))
        indices = np.random.choice(len(points), n_plot, replace=False)
        plot_points = points[indices]
        
        ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
                  alpha=0.6, s=20, c=plot_points[:, 2], cmap='plasma')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Workspace (3D)')
    
    def _plot_workspace_projections(self, ax):
        """Plot workspace projections."""
        points = self.workspace_results['points']
        
        ax.scatter(points[:, 0], points[:, 1], alpha=0.5, s=10, c='blue', label='XY Projection')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Workspace XY Projection')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
    
    def _plot_ik_convergence(self, ax):
        """Plot IK convergence analysis."""
        if not self.ik_results:
            ax.text(0.5, 0.5, 'No IK results\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('IK Convergence Analysis')
            return
        
        configs = [config for config, result in self.ik_results.items() if result['success']]
        iterations = [self.ik_results[config]['iterations'] for config in configs]
        times = [self.ik_results[config]['computation_time']*1000 for config in configs]
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar([i-0.2 for i in range(len(configs))], iterations, 0.4, 
                      label='Iterations', alpha=0.8, color='steelblue')
        bars2 = ax2.bar([i+0.2 for i in range(len(configs))], times, 0.4, 
                       label='Time (ms)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Iterations', color='steelblue')
        ax2.set_ylabel('Time (ms)', color='orange')
        ax.set_title('IK Convergence Performance')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_jacobian_heatmap(self, ax):
        """Plot Jacobian matrix heatmap."""
        # Use the first available Jacobian
        config = list(self.jacobian_results.keys())[0]
        J = self.jacobian_results[config]['J_space']
        
        im = ax.imshow(J, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Jacobian Matrix ({config})')
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('DOF Index')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add grid
        ax.set_xticks(range(J.shape[1]))
        ax.set_yticks(range(J.shape[0]))
        ax.grid(True, alpha=0.3)
    
    def _plot_joint_limits(self, ax):
        """Plot joint limits and current configurations."""
        joint_indices = range(self.n_joints)
        
        # Plot joint limits as filled area
        ax.fill_between(joint_indices, 
                       np.degrees(self.joint_limits[:, 0]), 
                       np.degrees(self.joint_limits[:, 1]), 
                       alpha=0.3, color='lightgray', label='Joint Limits')
        
        # Plot current configurations
        for config_name, fk_result in self.fk_results.items():
            angles_deg = np.degrees(fk_result['joint_angles'])
            ax.plot(joint_indices, angles_deg, 'o-', label=config_name, alpha=0.8)
        
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Joint Angle (degrees)')
        ax.set_title('Joint Angles vs Limits')
        ax.set_xticks(joint_indices)
        ax.set_xticklabels([f'J{i+1}' for i in joint_indices])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self, ax):
        """Plot performance metrics summary."""
        metrics = {
            'FK Time (ms)': np.mean([r['computation_time']*1000 for r in self.fk_results.values()]),
            'Jacobian Time (ms)': np.mean([r['computation_time']*1000 for r in self.jacobian_results.values()]),
        }
        
        if self.ik_results:
            successful_iks = [r for r in self.ik_results.values() if r['success']]
            if successful_iks:
                metrics['IK Time (ms)'] = np.mean([r['computation_time']*1000 for r in successful_iks])
        
        bars = ax.bar(metrics.keys(), metrics.values(), alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Time (ms)')
        ax.set_title('Performance Metrics')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_singular_values(self, ax):
        """Plot singular values analysis."""
        configs = list(self.jacobian_results.keys())
        
        for i, config in enumerate(configs):
            sigma = self.jacobian_results[config]['singular_values']
            ax.plot(sigma, 'o-', label=config, alpha=0.8)
        
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        ax.set_title('Jacobian Singular Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_error_analysis(self, ax):
        """Plot error analysis for IK solutions."""
        if not self.ik_results:
            ax.text(0.5, 0.5, 'No IK results\navailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Analysis')
            return
        
        successful_configs = [config for config, result in self.ik_results.items() if result['success']]
        
        if not successful_configs:
            ax.text(0.5, 0.5, 'No successful\nIK solutions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Analysis')
            return
        
        position_errors = [self.ik_results[config]['position_error'] for config in successful_configs]
        orientation_errors = [self.ik_results[config]['orientation_error'] for config in successful_configs]
        joint_errors = [self.ik_results[config]['joint_error'] for config in successful_configs]
        
        x = np.arange(len(successful_configs))
        width = 0.25
        
        ax.bar(x - width, position_errors, width, label='Position Error (m)', alpha=0.8)
        ax.bar(x, orientation_errors, width, label='Orientation Error (rad)', alpha=0.8)
        ax.bar(x + width, joint_errors, width, label='Joint Error (rad)', alpha=0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Error')
        ax.set_title('IK Solution Errors')
        ax.set_xticks(x)
        ax.set_xticklabels(successful_configs, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Helper methods
    def _generate_safe_test_pose(self):
        """Generate a safe test pose within joint limits."""
        # Generate pose that's 70% towards joint limits to avoid extreme configurations
        return 0.7 * np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
    
    def _rotation_matrix_to_euler_zyx(self, R):
        """Convert rotation matrix to ZYX Euler angles."""
        # ZYX Euler angles (roll, pitch, yaw)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # roll
            y = np.arctan2(-R[2, 0], sy)      # pitch
            z = np.arctan2(R[1, 0], R[0, 0])  # yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def _compute_orientation_error(self, R1, R2):
        """Compute orientation error between two rotation matrices."""
        R_error = R1.T @ R2
        # Convert to axis-angle representation and get angle
        trace = np.trace(R_error)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle


def main():
    """Main function to run the kinematics basic demo."""
    try:
        # Create and run demo
        demo = KinematicsBasicDemo()
        success = demo.run_demo()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("📋 Summary of demonstrated concepts:")
            print("   ✅ Robot model loading from URDF")
            print("   ✅ Forward kinematics computation")
            print("   ✅ Inverse kinematics solving")
            print("   ✅ Jacobian matrix analysis")
            print("   ✅ Workspace analysis")
            print("   ✅ Performance benchmarking")
            print("   ✅ Comprehensive visualization")
            
            print("\n📚 Key takeaways:")
            print("   • Forward kinematics: Given joint angles → end-effector pose")
            print("   • Inverse kinematics: Given end-effector pose → joint angles")
            print("   • Jacobian: Relates joint velocities to end-effector velocities")
            print("   • Manipulability: Measure of how well the robot can move")
            print("   • Condition number: Indicates numerical conditioning")
            
            print("\n🔗 Next steps:")
            print("   • Try intermediate_examples/trajectory_planning_intermediate_demo.py")
            print("   • Explore basic_examples/dynamics_basic_demo.py")
            print("   • Check out basic_examples/control_basic_demo.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
