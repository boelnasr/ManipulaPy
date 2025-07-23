#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Intermediate Simulation Demo - ManipulaPy

This script demonstrates the intermediate capabilities of the ManipulaPy library including:
- URDF loading and robot initialization
- Trajectory planning with GPU/CPU optimization
- Control system implementation
- Dynamics simulation
- Singularity analysis
- Performance benchmarking
- Visualization with plot saving

All generated plots are saved in the same folder as this script.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('TkAgg')
# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import ManipulaPy modules
try:
    # Add the parent directory to the path to find ManipulaPy
    current_dir = Path(__file__).parent.absolute()
    manipulapy_path = current_dir.parent.parent
    if str(manipulapy_path) not in sys.path:
        sys.path.insert(0, str(manipulapy_path))
    
    import ManipulaPy as mp
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning, create_optimized_planner
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.singularity import Singularity
    from ManipulaPy.cuda_kernels import check_cuda_availability
    
    print("✅ ManipulaPy modules imported successfully")
    
except ImportError as e:
    print(f"❌ Error importing ManipulaPy: {e}")
    print("Please ensure ManipulaPy is properly installed and accessible.")
    print("Current Python path:", sys.path)
    sys.exit(1)

# Configure matplotlib for non-interactive plotting
plt.ioff()  # Turn off interactive mode for automated saving
plt.rcParams['figure.max_open_warning'] = 50

class SimulationIntermediateDemo:
    """
    Comprehensive demonstration of ManipulaPy's intermediate features with
    automatic plot saving and performance analysis.
    """
    
    def __init__(self, urdf_path=None, save_plots=True):
        """
        Initialize the demonstration.
        
        Args:
            urdf_path (str, optional): Path to URDF file. If None, creates a default robot example.
            save_plots (bool): Whether to save generated plots to files.
        """
        self.save_plots = save_plots
        self.script_dir = Path(__file__).parent.absolute()
        self.plots_saved = []
        
        # Setup logging
        self.setup_logging()
        
        # URDF path handling
        if urdf_path is None:
            # Try to find a sample URDF or create a simple one
            self.urdf_path = self.create_sample_urdf()
        else:
            self.urdf_path = urdf_path
            
        # Verify URDF exists
        if not os.path.exists(self.urdf_path):
            self.logger.error(f"URDF file not found: {self.urdf_path}")
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
            
        self.logger.info(f"Using URDF: {self.urdf_path}")
        
        # Initialize robot components
        self.initialize_robot()
        
        # Performance tracking
        self.performance_results = {}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.script_dir / 'simulation_demo.log'
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode='w')
            ]
        )
        self.logger = logging.getLogger('SimulationDemo')
        
    def create_sample_urdf(self):
        """Create a simple sample URDF for demonstration if none is provided."""
        urdf_content = '''<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  
  <link name="link1">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  
  <link name="link2">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>'''
        
        urdf_path = self.script_dir / 'simple_arm.urdf'
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        self.logger.info(f"Created sample URDF: {urdf_path}")
        return str(urdf_path)
    
    def initialize_robot(self):
        """Initialize robot from URDF and setup components."""
        self.logger.info("Initializing robot components...")
        
        try:
            # Load URDF and create SerialManipulator
            self.urdf_processor = URDFToSerialManipulator(self.urdf_path, use_pybullet_limits=False)
            self.robot = self.urdf_processor.serial_manipulator
            self.dynamics = self.urdf_processor.dynamics
            
            # Setup joint limits
            self.joint_limits = self.robot.joint_limits
            self.num_joints = len(self.joint_limits)
            
            # Create torque limits (if not specified)
            self.torque_limits = [(-10.0, 10.0)] * self.num_joints
            
            # Initialize trajectory planner with safe settings
            self.planner = OptimizedTrajectoryPlanning(
                serial_manipulator=self.robot,
                urdf_path=self.urdf_path,
                dynamics=self.dynamics,
                joint_limits=self.joint_limits,
                torque_limits=self.torque_limits,
                use_cuda=None,  # Auto-detect
                cuda_threshold=1000,  # Conservative threshold
                enable_profiling=False  # Disable for stability
            )
            
            # Initialize controller
            self.controller = ManipulatorController(self.dynamics)
            
            # Initialize singularity analyzer
            self.singularity_analyzer = Singularity(self.robot)
            
            self.logger.info(f"Robot initialized successfully with {self.num_joints} joints")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize robot: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_plot(self, filename, title=None):
        """Save current plot with timestamp and title."""
        if not self.save_plots:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if title:
            filename = f"{timestamp}_{title}_{filename}"
        else:
            filename = f"{timestamp}_{filename}"
            
        filepath = self.script_dir / filename
        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            self.plots_saved.append(str(filepath))
            self.logger.info(f"Plot saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save plot {filepath}: {e}")
    
    def demonstrate_forward_kinematics(self):
        """Demonstrate forward kinematics with visualization."""
        self.logger.info("=== Forward Kinematics Demonstration ===")
        
        try:
            # Generate random joint configurations
            num_configs = 20
            joint_configs = []
            ee_positions = []
            
            for i in range(num_configs):
                # Random joint angles within limits
                angles = []
                for limit in self.joint_limits:
                    low, high = limit
                    angle = np.random.uniform(low, high)
                    angles.append(angle)
                joint_configs.append(angles)
                
                # Compute forward kinematics
                T = self.robot.forward_kinematics(angles)
                ee_pos = T[:3, 3]
                ee_positions.append(ee_pos)
                
                self.logger.info(f"Config {i+1}: joints={np.array(angles):.3f}, ee_pos={ee_pos:.3f}")
            
            # Plot workspace
            ee_positions = np.array(ee_positions)
            
            fig = plt.figure(figsize=(15, 6))
            
            # 3D workspace plot
            ax1 = fig.add_subplot(121, projection='3d')
            scatter = ax1.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                       c=range(num_configs), cmap='viridis', s=100)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('End-Effector Workspace (3D)')
            plt.colorbar(scatter, ax=ax1, shrink=0.5)
            
            # 2D projection
            ax2 = fig.add_subplot(122)
            scatter2 = ax2.scatter(ee_positions[:, 0], ee_positions[:, 1], 
                                 c=range(num_configs), cmap='viridis', s=100)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Workspace (Top View)')
            ax2.grid(True)
            ax2.axis('equal')
            plt.colorbar(scatter2, ax=ax2, label='Configuration')
            
            plt.tight_layout()
            self.save_plot('forward_kinematics_workspace.png', 'fk')
            plt.close()
            
            self.logger.info("Forward kinematics demonstration completed")
            
        except Exception as e:
            self.logger.error(f"Forward kinematics demo failed: {e}")
            plt.close()
    
    def demonstrate_inverse_kinematics(self):
        """Demonstrate inverse kinematics with convergence analysis."""
        self.logger.info("=== Inverse Kinematics Demonstration ===")
        
        try:
            # Define target poses (reachable positions)
            targets = [
                [0.2, 0.1, 0.2],   # Target 1
                [0.1, -0.2, 0.15], # Target 2
                [0.25, 0.0, 0.3],  # Target 3
            ]
            
            ik_results = []
            
            for i, target_pos in enumerate(targets):
                try:
                    # Create target transformation matrix
                    T_target = np.eye(4)
                    T_target[:3, 3] = target_pos
                    
                    # Initial guess
                    theta_init = np.zeros(self.num_joints)
                    
                    # Solve IK
                    start_time = time.time()
                    theta_solution, success, iterations = self.robot.iterative_inverse_kinematics(
                        T_target, theta_init, max_iterations=500, plot_residuals=False
                    )
                    solve_time = time.time() - start_time
                    
                    # Verify solution
                    T_result = self.robot.forward_kinematics(theta_solution)
                    error = np.linalg.norm(T_result[:3, 3] - target_pos)
                    
                    result = {
                        'target': target_pos,
                        'solution': theta_solution,
                        'success': success,
                        'iterations': iterations,
                        'solve_time': solve_time,
                        'error': error
                    }
                    ik_results.append(result)
                    
                    self.logger.info(f"Target {i+1}: {target_pos} -> Success: {success}, "
                                   f"Error: {error:.6f}m, Time: {solve_time:.3f}s, Iterations: {iterations}")
                    
                except Exception as e:
                    self.logger.warning(f"IK failed for target {i+1}: {e}")
                    ik_results.append({
                        'target': target_pos,
                        'success': False,
                        'error': float('inf'),
                        'solve_time': 0,
                        'iterations': 0
                    })
            
            # Plot IK results
            if ik_results:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Success rate
                successes = [r['success'] for r in ik_results]
                success_values = [1 if s else 0 for s in successes]
                ax1.bar(range(len(targets)), success_values, color=['green' if s else 'red' for s in successes])
                ax1.set_title('IK Success Rate')
                ax1.set_xlabel('Target')
                ax1.set_ylabel('Success (1) / Failure (0)')
                ax1.set_xticks(range(len(targets)))
                ax1.set_xticklabels([f'T{i+1}' for i in range(len(targets))])
                
                # Solution errors (only successful ones)
                successful_results = [r for r in ik_results if r['success']]
                if successful_results:
                    errors = [r['error'] for r in successful_results]
                    ax2.bar(range(len(successful_results)), errors)
                    ax2.set_title('IK Position Errors (Successful)')
                    ax2.set_xlabel('Successful Target')
                    ax2.set_ylabel('Error (m)')
                    ax2.set_yscale('log')
                
                # Solve times
                times = [r['solve_time'] for r in ik_results]
                ax3.bar(range(len(targets)), times)
                ax3.set_title('IK Solve Times')
                ax3.set_xlabel('Target')
                ax3.set_ylabel('Time (s)')
                ax3.set_xticks(range(len(targets)))
                ax3.set_xticklabels([f'T{i+1}' for i in range(len(targets))])
                
                # Joint solutions (only successful ones)
                if successful_results:
                    solutions = np.array([r['solution'] for r in successful_results])
                    for j in range(min(self.num_joints, 4)):  # Show max 4 joints
                        ax4.plot(range(len(successful_results)), solutions[:, j], 'o-', label=f'Joint {j+1}')
                    ax4.set_title('IK Joint Solutions')
                    ax4.set_xlabel('Successful Target')
                    ax4.set_ylabel('Joint Angle (rad)')
                    ax4.legend()
                    ax4.grid(True)
                
                plt.tight_layout()
                self.save_plot('inverse_kinematics_analysis.png', 'ik')
                plt.close()
            
            self.logger.info("Inverse kinematics demonstration completed")
            return ik_results
            
        except Exception as e:
            self.logger.error(f"Inverse kinematics demo failed: {e}")
            plt.close()
            return []
    
    def demonstrate_trajectory_planning(self):
        """Demonstrate trajectory planning with performance comparison."""
        self.logger.info("=== Trajectory Planning Demonstration ===")
        
        try:
            # Define trajectory parameters
            start_config = np.zeros(self.num_joints)
            
            # Create safe end configuration within joint limits
            end_config = []
            for limit in self.joint_limits:
                low, high = limit
                # Use 50% of the range
                angle = (high - low) * 0.25 + low
                end_config.append(angle)
            end_config = np.array(end_config)
            
            trajectory_params = [
                {'N': 50, 'Tf': 1.0, 'method': 3, 'name': 'Cubic_50'},
                {'N': 100, 'Tf': 2.0, 'method': 3, 'name': 'Cubic_100'},
                {'N': 200, 'Tf': 2.0, 'method': 5, 'name': 'Quintic_200'},
            ]
            
            trajectory_results = []
            
            for params in trajectory_params:
                self.logger.info(f"Planning trajectory: {params['name']}")
                
                try:
                    # Generate trajectory
                    start_time = time.time()
                    trajectory_data = self.planner.joint_trajectory(
                        start_config, end_config, params['Tf'], params['N'], params['method']
                    )
                    planning_time = time.time() - start_time
                    
                    result = {
                        'name': params['name'],
                        'params': params,
                        'trajectory': trajectory_data,
                        'planning_time': planning_time
                    }
                    trajectory_results.append(result)
                    
                    self.logger.info(f"Planned {params['name']} in {planning_time:.4f}s")
                    
                except Exception as e:
                    self.logger.warning(f"Trajectory planning failed for {params['name']}: {e}")
            
            # Plot trajectories
            if trajectory_results:
                fig, axes = plt.subplots(3, len(trajectory_results), figsize=(5*len(trajectory_results), 12))
                if len(trajectory_results) == 1:
                    axes = axes.reshape(-1, 1)
                
                for i, result in enumerate(trajectory_results):
                    traj = result['trajectory']
                    N = result['params']['N']
                    Tf = result['params']['Tf']
                    time_steps = np.linspace(0, Tf, N)
                    
                    # Plot positions
                    for j in range(min(self.num_joints, 4)):  # Limit to 4 joints for visibility
                        axes[0, i].plot(time_steps, traj['positions'][:, j], label=f'Joint {j+1}')
                    axes[0, i].set_title(f"{result['name']} - Positions")
                    axes[0, i].set_xlabel('Time (s)')
                    axes[0, i].set_ylabel('Position (rad)')
                    axes[0, i].legend(fontsize=8)
                    axes[0, i].grid(True)
                    
                    # Plot velocities
                    for j in range(min(self.num_joints, 4)):
                        axes[1, i].plot(time_steps, traj['velocities'][:, j], label=f'Joint {j+1}')
                    axes[1, i].set_title(f"{result['name']} - Velocities")
                    axes[1, i].set_xlabel('Time (s)')
                    axes[1, i].set_ylabel('Velocity (rad/s)')
                    axes[1, i].legend(fontsize=8)
                    axes[1, i].grid(True)
                    
                    # Plot accelerations
                    for j in range(min(self.num_joints, 4)):
                        axes[2, i].plot(time_steps, traj['accelerations'][:, j], label=f'Joint {j+1}')
                    axes[2, i].set_title(f"{result['name']} - Accelerations")
                    axes[2, i].set_xlabel('Time (s)')
                    axes[2, i].set_ylabel('Acceleration (rad/s²)')
                    axes[2, i].legend(fontsize=8)
                    axes[2, i].grid(True)
                
                plt.tight_layout()
                self.save_plot('trajectory_planning_comparison.png', 'traj')
                plt.close()
                
                # Performance comparison
                planning_times = [r['planning_time'] for r in trajectory_results]
                names = [r['name'] for r in trajectory_results]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(names, planning_times)
                ax.set_title('Trajectory Planning Performance')
                ax.set_ylabel('Planning Time (s)')
                ax.set_xlabel('Trajectory Type')
                
                # Add value labels on bars
                for bar, time_val in zip(bars, planning_times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{time_val:.4f}s', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                self.save_plot('trajectory_planning_performance.png', 'traj_perf')
                plt.close()
            
            self.logger.info("Trajectory planning demonstration completed")
            return trajectory_results
            
        except Exception as e:
            self.logger.error(f"Trajectory planning demo failed: {e}")
            plt.close()
            return []
    
    def demonstrate_singularity_analysis(self):
        """Demonstrate singularity analysis and workspace visualization."""
        self.logger.info("=== Singularity Analysis Demonstration ===")
        
        try:
            # Generate random configurations for analysis
            num_samples = 500  # Reduced for faster computation
            configurations = []
            condition_numbers = []
            is_singular_list = []
            
            for i in range(num_samples):
                # Random configuration
                config = []
                for limit in self.joint_limits:
                    low, high = limit
                    angle = np.random.uniform(low, high)
                    config.append(angle)
                configurations.append(config)
                
                # Compute condition number
                try:
                    cond_num = self.singularity_analyzer.condition_number(config)
                    condition_numbers.append(cond_num)
                    
                    # Check for singularity
                    is_singular = self.singularity_analyzer.singularity_analysis(config)
                    is_singular_list.append(is_singular)
                    
                except Exception as e:
                    self.logger.debug(f"Error computing condition number for config {i}: {e}")
                    condition_numbers.append(1000.0)  # Large but finite number
                    is_singular_list.append(False)
            
            configurations = np.array(configurations)
            condition_numbers = np.array(condition_numbers)
            is_singular_list = np.array(is_singular_list)
            
            # Filter out very large condition numbers for plotting
            reasonable_mask = condition_numbers < 1000
            reasonable_configs = configurations[reasonable_mask]
            reasonable_cond_nums = condition_numbers[reasonable_mask]
            
            # Compute end-effector positions for workspace visualization
            ee_positions = []
            for config in reasonable_configs[:200]:  # Limit for performance
                try:
                    T = self.robot.forward_kinematics(config)
                    ee_positions.append(T[:3, 3])
                except Exception:
                    ee_positions.append([0, 0, 0])
            ee_positions = np.array(ee_positions)
            
            # Plot singularity analysis
            fig = plt.figure(figsize=(15, 10))
            
            # Condition number histogram
            ax1 = fig.add_subplot(2, 3, 1)
            if len(reasonable_cond_nums) > 0:
                ax1.hist(reasonable_cond_nums, bins=30, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Condition Number')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Condition Number Distribution')
                ax1.grid(True)
            
            # Workspace colored by condition number
            if len(ee_positions) > 0 and len(reasonable_cond_nums[:len(ee_positions)]) > 0:
                ax2 = fig.add_subplot(2, 3, 2, projection='3d')
                scatter = ax2.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                                     c=reasonable_cond_nums[:len(ee_positions)], cmap='viridis', 
                                     s=20, alpha=0.6)
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title('Workspace (Condition Number)')
                try:
                    plt.colorbar(scatter, ax=ax2, shrink=0.5, aspect=5)
                except:
                    pass
            
            # Singularity locations
            singular_configs = configurations[is_singular_list]
            if len(singular_configs) > 0:
                singular_ee_pos = []
                for config in singular_configs[:50]:  # Limit for performance
                    try:
                        T = self.robot.forward_kinematics(config)
                        singular_ee_pos.append(T[:3, 3])
                    except Exception:
                        continue
                
                if len(singular_ee_pos) > 0 and len(ee_positions) > 0:
                    singular_ee_pos = np.array(singular_ee_pos)
                    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
                    ax3.scatter(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                               c='blue', s=10, alpha=0.3, label='Regular')
                    ax3.scatter(singular_ee_pos[:, 0], singular_ee_pos[:, 1], singular_ee_pos[:, 2],
                               c='red', s=30, alpha=0.8, label='Singular')
                    ax3.set_xlabel('X (m)')
                    ax3.set_ylabel('Y (m)')
                    ax3.set_zlabel('Z (m)')
                    ax3.set_title('Singularity Locations')
                    ax3.legend()
            
            # Performance metrics
            ax4 = fig.add_subplot(2, 3, 4)
            singular_percentage = len(is_singular_list[is_singular_list]) / len(is_singular_list) * 100
            mean_cond = np.mean(reasonable_cond_nums) if len(reasonable_cond_nums) > 0 else 0
            std_cond = np.std(reasonable_cond_nums) if len(reasonable_cond_nums) > 0 else 0
            
            metrics = [singular_percentage, mean_cond, std_cond]
            metric_names = ['Singularity %', 'Mean Cond.', 'Std Cond.']
            
            bars = ax4.bar(metric_names, metrics)
            ax4.set_title('Singularity Analysis Metrics')
            ax4.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.save_plot('singularity_analysis.png', 'singular')
            plt.close()
            
            self.logger.info(f"Singularity Analysis Results:")
            self.logger.info(f"  Total configurations analyzed: {num_samples}")
            self.logger.info(f"  Singular configurations: {len(is_singular_list[is_singular_list])} ({singular_percentage:.1f}%)")
            self.logger.info(f"  Mean condition number: {mean_cond:.2f}")
            
            return {
                'singularity_percentage': singular_percentage,
                'mean_condition_number': mean_cond,
                'configurations': configurations,
                'condition_numbers': condition_numbers
            }
            
        except Exception as e:
            self.logger.error(f"Singularity analysis demo failed: {e}")
            plt.close()
            return {}
    
    def demonstrate_performance_benchmarking(self):
        """Benchmark different computational approaches."""
        self.logger.info("=== Performance Benchmarking ===")
        
        try:
            # Check CUDA availability
            cuda_available = check_cuda_availability()
            self.logger.info(f"CUDA Available: {cuda_available}")
            
            # Benchmark trajectory planning performance
            benchmark_results = self.planner.benchmark_performance([
                {"N": 50, "joints": self.num_joints, "name": "Small"},
                {"N": 100, "joints": self.num_joints, "name": "Medium"},
                {"N": 200, "joints": self.num_joints, "name": "Large"},
            ])
            
            # Plot performance comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Total times
            names = list(benchmark_results.keys())
            times = [benchmark_results[name]['total_time'] for name in names]
            gpu_usage = [benchmark_results[name]['used_gpu'] for name in names]
            
            colors = ['green' if gpu else 'blue' for gpu in gpu_usage]
            bars1 = ax1.bar(names, times, color=colors)
            ax1.set_title('Trajectory Planning Performance')
            ax1.set_ylabel('Total Time (s)')
            ax1.set_xlabel('Problem Size')
            
            # Add value labels
            for bar, time_val in zip(bars1, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.4f}s', ha='center', va='bottom')
            
            # Add legend for colors
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='GPU'),
                              Patch(facecolor='blue', label='CPU')]
            ax1.legend(handles=legend_elements)
            
            # Problem size vs time
            sizes = [benchmark_results[name]['N'] * benchmark_results[name]['joints'] for name in names]
            ax2.loglog(sizes, times, 'o-')
            ax2.set_xlabel('Problem Size (N × joints)')
            ax2.set_ylabel('Time (s)')
            ax2.set_title('Scaling Performance')
            ax2.grid(True)
            
            # GPU vs CPU breakdown
            planner_stats = self.planner.get_performance_stats()
            if planner_stats['gpu_calls'] > 0 or planner_stats['cpu_calls'] > 0:
                call_counts = [planner_stats['cpu_calls'], planner_stats['gpu_calls']]
                call_times = [planner_stats['total_cpu_time'], planner_stats['total_gpu_time']]
                
                ax3.bar(['CPU Calls', 'GPU Calls'], call_counts)
                ax3.set_title('Method Usage Count')
                ax3.set_ylabel('Number of Calls')
                
                ax4.bar(['CPU Time', 'GPU Time'], call_times)
                ax4.set_title('Total Computation Time')
                ax4.set_ylabel('Time (s)')
            
            plt.tight_layout()
            self.save_plot('performance_benchmarking.png', 'perf')
            plt.close()
            
            self.performance_results['benchmarking'] = benchmark_results
            self.performance_results['planner_stats'] = planner_stats
            
            self.logger.info("Performance benchmarking completed")
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Performance benchmarking failed: {e}")
            plt.close()
            return {}
    
    def demonstrate_controller_features(self):
        """Demonstrate controller tuning and analysis."""
        self.logger.info("=== Controller Features Demonstration ===")
        
        try:
            # Test configuration
            test_config = np.zeros(self.num_joints)
            
            # Controller tuning demonstration
            Ku = 50.0  # Example ultimate gain
            Tu = 0.2   # Example ultimate period
            
            # Tune controller gains for different methods
            tuning_results = {}
            methods = ['P', 'PI', 'PID']
            
            for method in methods:
                try:
                    Kp, Ki, Kd = self.controller.tune_controller(Ku, Tu, kind=method)
                    tuning_results[method] = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
                    self.logger.info(f"Tuned {method} gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
                except Exception as e:
                    self.logger.warning(f"Controller tuning failed for {method}: {e}")
            
            # Plot gain recommendations
            if tuning_results:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                methods_list = list(tuning_results.keys())
                
                # Proportional gains
                kp_values = [tuning_results[m]['Kp'] for m in methods_list]
                if hasattr(kp_values[0], '__len__'):  # If arrays
                    kp_values = [np.mean(kp) if hasattr(kp, '__len__') else kp for kp in kp_values]
                ax1.bar(methods_list, kp_values)
                ax1.set_title('Proportional Gains (Kp)')
                ax1.set_ylabel('Gain')
                
                # Integral gains
                ki_values = [tuning_results[m]['Ki'] for m in methods_list]
                if hasattr(ki_values[0], '__len__'):  # If arrays
                    ki_values = [np.mean(ki) if hasattr(ki, '__len__') else ki for ki in ki_values]
                ax2.bar(methods_list, ki_values)
                ax2.set_title('Integral Gains (Ki)')
                ax2.set_ylabel('Gain')
                
                # Derivative gains
                kd_values = [tuning_results[m]['Kd'] for m in methods_list]
                if hasattr(kd_values[0], '__len__'):  # If arrays
                    kd_values = [np.mean(kd) if hasattr(kd, '__len__') else kd for kd in kd_values]
                ax3.bar(methods_list, kd_values)
                ax3.set_title('Derivative Gains (Kd)')
                ax3.set_ylabel('Gain')
                
                plt.tight_layout()
                self.save_plot('controller_tuning.png', 'control_tune')
                plt.close()
            
            # Demonstrate different control methods
            desired_pos = np.array([0.1] * self.num_joints)
            current_pos = np.zeros(self.num_joints)
            current_vel = np.zeros(self.num_joints)
            
            if 'PID' in tuning_results:
                gains = tuning_results['PID']
                
                # Convert gains to arrays if they're scalars
                Kp = gains['Kp'] * np.ones(self.num_joints) if np.isscalar(gains['Kp']) else gains['Kp']
                Ki = gains['Ki'] * np.ones(self.num_joints) if np.isscalar(gains['Ki']) else gains['Ki']
                Kd = gains['Kd'] * np.ones(self.num_joints) if np.isscalar(gains['Kd']) else gains['Kd']
                
                try:
                    # Test PID control
                    import cupy as cp
                    pid_output = self.controller.pid_control(
                        cp.asarray(desired_pos), 
                        cp.asarray(current_vel),
                        cp.asarray(current_pos), 
                        cp.asarray(current_vel),
                        dt=0.01,
                        Kp=cp.asarray(Kp),
                        Ki=cp.asarray(Ki),
                        Kd=cp.asarray(Kd)
                    )
                    
                    # Test PD control
                    pd_output = self.controller.pd_control(
                        cp.asarray(desired_pos),
                        cp.asarray(current_vel),
                        cp.asarray(current_pos),
                        cp.asarray(current_vel),
                        Kp=cp.asarray(Kp),
                        Kd=cp.asarray(Kd)
                    )
                    
                    self.logger.info(f"PID output: {cp.asnumpy(pid_output)}")
                    self.logger.info(f"PD output: {cp.asnumpy(pd_output)}")
                    
                except ImportError:
                    self.logger.warning("CuPy not available, skipping GPU control tests")
                except Exception as e:
                    self.logger.warning(f"Control test failed: {e}")
            
            self.logger.info("Controller features demonstration completed")
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"Controller features demo failed: {e}")
            plt.close()
            return {}
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        self.logger.info("=== Generating Summary Report ===")
        
        try:
            # Collect all performance data
            summary_data = {
                'robot_info': {
                    'num_joints': self.num_joints,
                    'joint_limits': self.joint_limits,
                    'urdf_path': self.urdf_path
                },
                'performance': self.performance_results,
                'plots_generated': len(self.plots_saved),
                'cuda_available': check_cuda_availability()
            }
            
            # Create summary plot
            fig = plt.figure(figsize=(16, 10))
            
            # Robot information
            ax1 = plt.subplot(2, 3, 1)
            ax1.text(0.5, 0.5, f"Robot Joints: {self.num_joints}\n"
                                f"CUDA Available: {summary_data['cuda_available']}\n"
                                f"Plots Generated: {summary_data['plots_generated']}\n"
                                f"URDF: {os.path.basename(self.urdf_path)}",
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax1.set_title('System Information')
            ax1.axis('off')
            
            # Performance summary (if available)
            if 'benchmarking' in self.performance_results:
                ax2 = plt.subplot(2, 3, 2)
                bench_data = self.performance_results['benchmarking']
                names = list(bench_data.keys())
                times = [bench_data[name]['total_time'] for name in names]
                
                ax2.bar(names, times)
                ax2.set_title('Performance Summary')
                ax2.set_ylabel('Time (s)')
                plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Feature coverage
            ax3 = plt.subplot(2, 3, 3)
            features = [
                'Forward Kinematics',
                'Inverse Kinematics', 
                'Trajectory Planning',
                'Singularity Analysis',
                'Performance Benchmarking',
                'Controller Features'
            ]
            coverage = [1, 1, 1, 1, 1, 1]  # All features demonstrated
            
            colors = ['green' if c else 'red' for c in coverage]
            ax3.barh(features, coverage, color=colors)
            ax3.set_title('Feature Coverage')
            ax3.set_xlabel('Demonstrated')
            ax3.set_xlim(0, 1.2)
            
            # Joint limits visualization
            ax4 = plt.subplot(2, 3, 4)
            joint_numbers = range(1, self.num_joints + 1)
            lower_limits = [limit[0] for limit in self.joint_limits]
            upper_limits = [limit[1] for limit in self.joint_limits]
            
            ax4.plot(joint_numbers, lower_limits, 'r-o', label='Lower Limit')
            ax4.plot(joint_numbers, upper_limits, 'g-o', label='Upper Limit')
            ax4.fill_between(joint_numbers, lower_limits, upper_limits, alpha=0.3)
            ax4.set_title('Joint Limits')
            ax4.set_xlabel('Joint Number')
            ax4.set_ylabel('Angle (rad)')
            ax4.legend()
            ax4.grid(True)
            
            # Execution timeline
            ax5 = plt.subplot(2, 3, 5)
            if self.plots_saved:
                timestamps = []
                for plot_path in self.plots_saved:
                    filename = os.path.basename(plot_path)
                    if '_' in filename:
                        timestamp_str = filename.split('_')[0]
                        try:
                            timestamp = time.strptime(timestamp_str, "%Y%m%d")
                            timestamps.append(timestamp)
                        except:
                            pass
                
                if timestamps:
                    ax5.hist([time.mktime(ts) for ts in timestamps], bins=10, alpha=0.7)
                    ax5.set_title('Plot Generation Timeline')
                    ax5.set_xlabel('Time')
                    ax5.set_ylabel('Plots Generated')
            
            # System resources (if available)
            ax6 = plt.subplot(2, 3, 6)
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                memory_data = [
                    memory_info.rss / 1024 / 1024,  # RSS in MB
                    memory_info.vms / 1024 / 1024,  # VMS in MB
                ]
                labels = ['Resident\nMemory (MB)', 'Virtual\nMemory (MB)']
                
                ax6.bar(labels, memory_data)
                ax6.set_title('Memory Usage')
                ax6.set_ylabel('Memory (MB)')
                
                for i, v in enumerate(memory_data):
                    ax6.text(i, v, f'{v:.1f}', ha='center', va='bottom')
                    
            except ImportError:
                ax6.text(0.5, 0.5, 'Memory info\nnot available\n(psutil not installed)', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('System Resources')
                ax6.axis('off')
            
            # Add large title
            fig.suptitle('ManipulaPy Simulation Demo - Comprehensive Report', 
                         fontsize=16, fontweight='bold')
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                    ha='right', va='bottom', fontsize=8)
            
            plt.tight_layout()
            self.save_plot('summary_report.png', 'summary')
            plt.close()
            
            # Save detailed report to text file
            report_file = self.script_dir / 'simulation_demo_report.txt'
            with open(report_file, 'w') as f:
                f.write("ManipulaPy Simulation Demo - Detailed Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {timestamp}\n\n")
                
                f.write("Robot Configuration:\n")
                f.write(f"  URDF Path: {self.urdf_path}\n")
                f.write(f"  Number of Joints: {self.num_joints}\n")
                f.write(f"  Joint Limits: {self.joint_limits}\n\n")
                
                f.write("System Information:\n")
                f.write(f"  CUDA Available: {summary_data['cuda_available']}\n")
                f.write(f"  Plots Generated: {summary_data['plots_generated']}\n")
                f.write(f"  Plots Saved To: {self.script_dir}\n\n")
                
                if self.plots_saved:
                    f.write("Generated Plots:\n")
                    for plot_path in self.plots_saved:
                        f.write(f"  - {os.path.basename(plot_path)}\n")
                    f.write("\n")
                
                f.write("Performance Results:\n")
                for key, value in self.performance_results.items():
                    f.write(f"  {key}: {str(value)[:100]}...\n")
            
            self.logger.info(f"Detailed report saved to: {report_file}")
            self.logger.info("Summary report generation completed")
            
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Summary report generation failed: {e}")
            plt.close()
            return {}
    
    def run_complete_demonstration(self):
        """Run the complete demonstration sequence."""
        self.logger.info("Starting ManipulaPy Intermediate Simulation Demo")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run all demonstrations
            self.logger.info("Running demonstration sequence...")
            
            self.demonstrate_forward_kinematics()
            self.demonstrate_inverse_kinematics()
            self.demonstrate_trajectory_planning()
            self.demonstrate_singularity_analysis()
            self.demonstrate_performance_benchmarking()
            self.demonstrate_controller_features()
            
            # Generate summary
            summary = self.generate_summary_report()
            
            total_time = time.time() - start_time
            self.logger.info(f"Demo completed successfully in {total_time:.2f} seconds")
            self.logger.info(f"Generated {len(self.plots_saved)} plots")
            
            if self.save_plots:
                self.logger.info("All plots saved to: " + str(self.script_dir))
                self.logger.info("Plot files:")
                for plot_file in self.plots_saved:
                    self.logger.info(f"  - {os.path.basename(plot_file)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup
            plt.close('all')
            if hasattr(self, 'planner'):
                try:
                    self.planner.cleanup_gpu_memory()
                except:
                    pass


def main():
    """Main function to run the demonstration."""
    print("ManipulaPy Intermediate Simulation Demo")
    print("======================================")
    print("This demo will generate plots and save them to the current directory.")
    print()
    
    # You can specify a custom URDF path here
    # urdf_path = "/path/to/your/robot.urdf"
    urdf_path = None  # Will create a sample URDF
    
    try:
        # Create and run demo
        demo = SimulationIntermediateDemo(urdf_path=urdf_path, save_plots=True)
        summary = demo.run_complete_demonstration()
        
        print("\n" + "="*50)
        print("DEMO SUMMARY")
        print("="*50)
        print(f"Robot joints: {summary['robot_info']['num_joints']}")
        print(f"CUDA available: {summary['cuda_available']}")
        print(f"Plots generated: {summary['plots_generated']}")
        print(f"Results saved to: {demo.script_dir}")
        print("\nGenerated files:")
        for plot_file in demo.plots_saved:
            print(f"  - {os.path.basename(plot_file)}")
        print(f"  - simulation_demo_report.txt")
        print(f"  - simulation_demo.log")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo finished. Check the generated plots and report files.")


if __name__ == "__main__":
    main()