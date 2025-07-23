#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Intermediate Trajectory Planning Demo - ManipulaPy

This demo showcases advanced trajectory planning capabilities including:
- GPU-accelerated trajectory generation
- Batch trajectory processing
- Collision-aware path planning
- Multi-segment trajectory generation
- Cartesian space trajectory planning
- Optimal trajectory timing
- Trajectory smoothing and filtering
- Performance benchmarking and optimization

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import matplotlib
import os
matplotlib.use('TkAgg')

# ManipulaPy imports
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.cuda_kernels import CUDA_AVAILABLE, check_cuda_availability
    from ManipulaPy import utils
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
except ImportError as e:
    print(f"Error importing ManipulaPy modules: {e}")
    print("Please ensure ManipulaPy is properly installed.")
    exit(1)

# Optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available, using CPU only")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntermediateTrajectoryDemo:
    """
    Demonstrates advanced trajectory planning techniques for robotic manipulators.
    """
    
    def __init__(self, use_simple_robot=False):
        """
        Initialize the trajectory planning demo.
        
        Args:
            use_simple_robot: If True, creates a simple 3-DOF robot. 
                             If False (default), uses the built-in XArm robot.
        """
        self.use_simple_robot = use_simple_robot
        # Get script directory for saving plots
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.setup_robot()
        self.setup_trajectory_planner()
        
    def setup_robot(self):
        """Set up the robot model (either XArm or simple)."""
        if self.use_simple_robot:
            self.setup_simple_robot()
        else:
            self.setup_xarm_robot()
            
    def setup_xarm_robot(self):
        """Load the built-in XArm robot from ManipulaPy data."""
        logger.info("Setting up XArm robot from built-in data...")
        
        try:
            # Load XArm robot using built-in URDF
            logger.info(f"Loading XArm URDF from: {urdf_file}")
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator
            self.dynamics = urdf_processor.dynamics
            
            # Get joint limits from the robot
            self.joint_limits = np.array(self.robot.joint_limits)
            num_joints = len(self.joint_limits)
            
            # XArm typical torque limits (approximate values for demonstration)
            xarm_torque_limits = {
                6: [(-50, 50), (-50, 50), (-30, 30), (-15, 15), (-15, 15), (-10, 10)],  # 6-DOF XArm
                7: [(-50, 50), (-50, 50), (-30, 30), (-15, 15), (-15, 15), (-10, 10), (-5, 5)]  # 7-DOF XArm
            }
            
            if num_joints in xarm_torque_limits:
                self.torque_limits = np.array(xarm_torque_limits[num_joints])
            else:
                # Default conservative limits
                self.torque_limits = np.array([(-30, 30)] * num_joints)
                logger.warning(f"Using default torque limits for {num_joints}-DOF robot")
            
            logger.info(f"✅ Loaded {num_joints}-DOF XArm robot successfully")
            logger.info(f"   Joint limits: {self.joint_limits.shape}")
            logger.info(f"   Torque limits: {self.torque_limits.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load XArm robot: {e}")
            logger.info("Falling back to simple robot...")
            self.use_simple_robot = True
            self.setup_simple_robot()
            
    def setup_simple_robot(self):
        """Create a simple 3-DOF planar robot for demonstration (fallback)."""
        logger.info("Setting up simple 3-DOF planar robot as fallback...")
        
        # Robot parameters (3-DOF planar robot)
        L1, L2, L3 = 1.0, 0.8, 0.6  # Link lengths
        
        # Home position (all joints at zero)
        M = np.array([
            [1, 0, 0, L1 + L2 + L3],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Screw axes in space frame
        S_list = np.array([
            [0, 0, 0],      # omega (rotation axes)
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],      # v (linear velocities)
            [0, 0, 0],
            [0, L1, L1+L2]
        ])
        
        # Body frame screw axes
        B_list = np.array([
            [0, 0, 0],
            [0, 0, 0], 
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [L2+L3, L3, 0]
        ])
        
        # Inertia matrices (simplified)
        G_list = []
        for i in range(3):
            # Simplified inertia matrix for each link
            mass = 1.0  # 1 kg per link
            Ixx = Iyy = Izz = 0.1  # Simplified inertia values
            G = np.array([
                [Ixx, 0, 0, 0, 0, 0],
                [0, Iyy, 0, 0, 0, 0],
                [0, 0, Izz, 0, 0, 0],
                [0, 0, 0, mass, 0, 0],
                [0, 0, 0, 0, mass, 0],
                [0, 0, 0, 0, 0, mass]
            ])
            G_list.append(G)
        
        # Joint limits (radians)
        joint_limits = [(-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi/3, np.pi/3)]
        
        # Extract omega and r lists
        omega_list = S_list[:3, :]
        r_list = utils.extract_r_list(S_list)
        
        # Create manipulator objects
        self.robot = SerialManipulator(
            M_list=M,
            omega_list=omega_list,
            r_list=r_list,
            S_list=S_list,
            B_list=B_list,
            G_list=G_list,
            joint_limits=joint_limits
        )
        
        self.dynamics = ManipulatorDynamics(
            M_list=M,
            omega_list=omega_list,
            r_list=r_list,
            b_list=None,
            S_list=S_list,
            B_list=B_list,
            Glist=G_list
        )
        
        # Control parameters
        self.joint_limits = np.array(joint_limits)
        self.torque_limits = np.array([(-10, 10), (-8, 8), (-5, 5)])  # Nm
        
        logger.info("✅ Simple robot setup complete")
        
    def setup_trajectory_planner(self):
        """Initialize the optimized trajectory planner."""
        logger.info("Setting up optimized trajectory planner...")
        
        try:
            self.planner = OptimizedTrajectoryPlanning(
                serial_manipulator=self.robot,
                urdf_path=urdf_file if not self.use_simple_robot else "simple_robot",
                dynamics=self.dynamics,
                joint_limits=self.joint_limits.tolist(),
                torque_limits=self.torque_limits.tolist(),
                use_cuda=None,  # Auto-detect
                cuda_threshold=100,  # Use GPU for problems with >100 elements
                enable_profiling=False
            )
            
            logger.info("✅ Optimized trajectory planner initialized")
            
            # Check GPU availability
            if check_cuda_availability():
                logger.info("🚀 GPU acceleration available for trajectory planning")
            else:
                logger.info("⚙️ Using CPU-only trajectory planning")
                
        except Exception as e:
            logger.error(f"Failed to initialize trajectory planner: {e}")
            raise
    
    def get_safe_waypoints(self, num_waypoints=5):
        """Generate safe waypoints within joint limits."""
        num_joints = len(self.joint_limits)
        waypoints = []
        
        for i in range(num_waypoints):
            # Generate waypoints that use a fraction of the joint range
            range_factor = 0.6  # Use 60% of available range
            waypoint = np.zeros(num_joints)
            
            for j in range(num_joints):
                min_pos = self.joint_limits[j, 0]
                max_pos = self.joint_limits[j, 1]
                center = (min_pos + max_pos) / 2
                range_size = (max_pos - min_pos) * range_factor
                
                # Create waypoints in a pattern
                angle = 2 * np.pi * i / num_waypoints
                waypoint[j] = center + (range_size / 2) * np.sin(angle + j * np.pi / 3)
            
            waypoints.append(waypoint)
        
        return np.array(waypoints)
    
    def demonstrate_basic_trajectory_generation(self):
        """Demonstrate basic joint space trajectory generation."""
        logger.info("\n🎯 Demonstrating Basic Trajectory Generation...")
        
        num_joints = len(self.joint_limits)
        
        # Define start and end configurations
        theta_start = np.zeros(num_joints)
        
        # Safe end configuration
        range_factor = 0.4
        theta_end = np.array([
            (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * range_factor * (-1)**i
            for i in range(num_joints)
        ])
        
        # Trajectory parameters
        T_final = 3.0  # 3 seconds
        N = 100        # 100 trajectory points
        
        logger.info(f"Generating trajectory from start to end configuration...")
        logger.info(f"  Start: {theta_start}")
        logger.info(f"  End: {theta_end}")
        
        # Test different time scaling methods
        methods = {3: "Cubic", 5: "Quintic"}
        results = {}
        
        for method_id, method_name in methods.items():
            logger.info(f"  Testing {method_name} time scaling...")
            
            start_time = time.time()
            trajectory = self.planner.joint_trajectory(
                thetastart=theta_start,
                thetaend=theta_end,
                Tf=T_final,
                N=N,
                method=method_id
            )
            elapsed_time = time.time() - start_time
            
            results[method_name] = {
                'trajectory': trajectory,
                'computation_time': elapsed_time
            }
            
            logger.info(f"    ✅ {method_name} completed in {elapsed_time:.4f}s")
        
        # Plot comparison
        self.plot_trajectory_comparison(results, T_final)
        
        # Analyze trajectory properties
        self.analyze_trajectory_properties(results, T_final)
        
        logger.info("✅ Basic trajectory generation demonstration complete")
        
    def plot_trajectory_comparison(self, results, T_final):
        """Plot comparison of different trajectory generation methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trajectory Generation Comparison', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red']
        methods = list(results.keys())
        
        # Get the first trajectory to determine number of joints and time steps
        first_traj = results[methods[0]]['trajectory']
        N = first_traj['positions'].shape[0]
        num_joints = first_traj['positions'].shape[1]
        time_history = np.linspace(0, T_final, N)
        
        # Position comparison (first 3 joints)
        ax = axes[0, 0]
        for i, (method, data) in enumerate(results.items()):
            positions = data['trajectory']['positions']
            for j in range(min(3, num_joints)):
                ax.plot(time_history, positions[:, j], 
                       color=colors[i], linestyle='-' if j == 0 else '--' if j == 1 else ':',
                       label=f'{method} Joint {j+1}', linewidth=2, alpha=0.8)
        
        ax.set_title('Position Profiles', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity comparison
        ax = axes[0, 1]
        for i, (method, data) in enumerate(results.items()):
            velocities = data['trajectory']['velocities']
            for j in range(min(3, num_joints)):
                ax.plot(time_history, velocities[:, j], 
                       color=colors[i], linestyle='-' if j == 0 else '--' if j == 1 else ':',
                       label=f'{method} Joint {j+1}', linewidth=2, alpha=0.8)
        
        ax.set_title('Velocity Profiles', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Acceleration comparison
        ax = axes[1, 0]
        for i, (method, data) in enumerate(results.items()):
            accelerations = data['trajectory']['accelerations']
            for j in range(min(3, num_joints)):
                ax.plot(time_history, accelerations[:, j], 
                       color=colors[i], linestyle='-' if j == 0 else '--' if j == 1 else ':',
                       label=f'{method} Joint {j+1}', linewidth=2, alpha=0.8)
        
        ax.set_title('Acceleration Profiles', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (rad/s²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Computation time comparison
        ax = axes[1, 1]
        methods_list = list(results.keys())
        times = [results[method]['computation_time'] for method in methods_list]
        
        bars = ax.bar(methods_list, times, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax.set_title('Computation Time Comparison', fontweight='bold')
        ax.set_ylabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # Add time values on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'trajectory_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Trajectory comparison plot saved as '{save_path}'")
        plt.close()
        
    def analyze_trajectory_properties(self, results, T_final):
        """Analyze trajectory properties like smoothness and continuity."""
        logger.info("\n📊 Trajectory Analysis:")
        
        for method, data in results.items():
            traj = data['trajectory']
            positions = traj['positions']
            velocities = traj['velocities']
            accelerations = traj['accelerations']
            
            # Check boundary conditions
            start_vel = np.linalg.norm(velocities[0])
            end_vel = np.linalg.norm(velocities[-1])
            start_acc = np.linalg.norm(accelerations[0])
            end_acc = np.linalg.norm(accelerations[-1])
            
            # Calculate smoothness metrics
            jerk = np.diff(accelerations, axis=0)
            max_jerk = np.max(np.abs(jerk))
            rms_jerk = np.sqrt(np.mean(jerk**2))
            
            # Peak values
            max_velocity = np.max(np.abs(velocities))
            max_acceleration = np.max(np.abs(accelerations))
            
            logger.info(f"  {method} Trajectory:")
            logger.info(f"    Start velocity: {start_vel:.6f} rad/s")
            logger.info(f"    End velocity: {end_vel:.6f} rad/s")
            logger.info(f"    Start acceleration: {start_acc:.6f} rad/s²")
            logger.info(f"    End acceleration: {end_acc:.6f} rad/s²")
            logger.info(f"    Max velocity: {max_velocity:.4f} rad/s")
            logger.info(f"    Max acceleration: {max_acceleration:.4f} rad/s²")
            logger.info(f"    Max jerk: {max_jerk:.4f} rad/s³")
            logger.info(f"    RMS jerk: {rms_jerk:.4f} rad/s³")
            logger.info(f"    Computation time: {data['computation_time']:.4f}s")
    
    def demonstrate_multi_segment_trajectories(self):
        """Demonstrate multi-segment trajectory generation through waypoints."""
        logger.info("\n🎯 Demonstrating Multi-Segment Trajectories...")
        
        # Generate waypoints
        waypoints = self.get_safe_waypoints(num_waypoints=4)
        num_joints = len(self.joint_limits)
        
        logger.info(f"Generated {len(waypoints)} waypoints:")
        for i, wp in enumerate(waypoints):
            logger.info(f"  Waypoint {i+1}: {wp}")
        
        # Trajectory parameters
        segment_time = 2.0  # 2 seconds per segment
        N_per_segment = 50  # 50 points per segment
        
        # Generate trajectories between consecutive waypoints
        segments = []
        total_computation_time = 0
        
        for i in range(len(waypoints) - 1):
            logger.info(f"  Generating segment {i+1} -> {i+2}...")
            
            start_time = time.time()
            segment = self.planner.joint_trajectory(
                thetastart=waypoints[i],
                thetaend=waypoints[i+1],
                Tf=segment_time,
                N=N_per_segment,
                method=5  # Quintic time scaling
            )
            elapsed_time = time.time() - start_time
            total_computation_time += elapsed_time
            
            segments.append(segment)
            logger.info(f"    ✅ Segment completed in {elapsed_time:.4f}s")
        
        # Combine segments
        combined_trajectory = self.combine_trajectory_segments(segments)
        
        logger.info(f"✅ Multi-segment trajectory generation complete")
        logger.info(f"  Total computation time: {total_computation_time:.4f}s")
        logger.info(f"  Total trajectory points: {combined_trajectory['positions'].shape[0]}")
        
        # Plot multi-segment trajectory
        self.plot_multi_segment_trajectory(combined_trajectory, waypoints, segment_time)
        
        # Analyze continuity at waypoints
        self.analyze_trajectory_continuity(segments, waypoints)
        
    def combine_trajectory_segments(self, segments):
        """Combine multiple trajectory segments into a single trajectory."""
        all_positions = []
        all_velocities = []
        all_accelerations = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                # Include all points from first segment
                all_positions.append(segment['positions'])
                all_velocities.append(segment['velocities'])
                all_accelerations.append(segment['accelerations'])
            else:
                # Skip first point to avoid duplication at waypoints
                all_positions.append(segment['positions'][1:])
                all_velocities.append(segment['velocities'][1:])
                all_accelerations.append(segment['accelerations'][1:])
        
        return {
            'positions': np.vstack(all_positions),
            'velocities': np.vstack(all_velocities),
            'accelerations': np.vstack(all_accelerations)
        }
    
    def plot_multi_segment_trajectory(self, trajectory, waypoints, segment_time):
        """Plot multi-segment trajectory with waypoint markers."""
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        
        N_total = positions.shape[0]
        num_joints = positions.shape[1]
        num_segments = len(waypoints) - 1
        
        # Create time vector
        total_time = num_segments * segment_time
        time_history = np.linspace(0, total_time, N_total)
        
        # Mark waypoint times
        waypoint_times = [i * segment_time for i in range(num_segments + 1)]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Multi-Segment Trajectory', fontsize=16, fontweight='bold')
        
        # Position plot
        ax = axes[0]
        for j in range(min(4, num_joints)):
            ax.plot(time_history, positions[:, j], label=f'Joint {j+1}', linewidth=2)
        
        # Mark waypoints
        for i, t in enumerate(waypoint_times):
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.7)
            ax.text(t, ax.get_ylim()[1]*0.9, f'WP{i+1}', ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_title('Joint Positions Through Waypoints', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity plot
        ax = axes[1]
        for j in range(min(4, num_joints)):
            ax.plot(time_history, velocities[:, j], label=f'Joint {j+1}', linewidth=2)
        
        # Mark waypoints
        for i, t in enumerate(waypoint_times):
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title('Joint Velocities Through Waypoints', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'multi_segment_trajectory.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Multi-segment trajectory plot saved as '{save_path}'")
        plt.close()
    
    def analyze_trajectory_continuity(self, segments, waypoints):
        """Analyze continuity at waypoints."""
        logger.info("\n📊 Trajectory Continuity Analysis:")
        
        for i in range(len(segments) - 1):
            # End of current segment
            end_pos = segments[i]['positions'][-1]
            end_vel = segments[i]['velocities'][-1]
            end_acc = segments[i]['accelerations'][-1]
            
            # Start of next segment
            start_pos = segments[i+1]['positions'][0]
            start_vel = segments[i+1]['velocities'][0]
            start_acc = segments[i+1]['accelerations'][0]
            
            # Continuity errors
            pos_error = np.linalg.norm(end_pos - start_pos)
            vel_error = np.linalg.norm(end_vel - start_vel)
            acc_error = np.linalg.norm(end_acc - start_acc)
            
            logger.info(f"  Waypoint {i+2} continuity:")
            logger.info(f"    Position error: {pos_error:.8f} rad")
            logger.info(f"    Velocity error: {vel_error:.8f} rad/s")
            logger.info(f"    Acceleration error: {acc_error:.8f} rad/s²")
            
            # Check if errors are within acceptable tolerance
            if pos_error < 1e-6:
                logger.info(f"    ✅ Position continuous")
            else:
                logger.warning(f"    ⚠️ Position discontinuity detected")
                
            if vel_error < 1e-6:
                logger.info(f"    ✅ Velocity continuous")
            else:
                logger.warning(f"    ⚠️ Velocity discontinuity detected")
    
    def demonstrate_batch_trajectory_generation(self):
        """Demonstrate batch processing of multiple trajectories."""
        logger.info("\n🎯 Demonstrating Batch Trajectory Generation...")
        
        if not check_cuda_availability():
            logger.warning("⚠️ GPU not available, batch processing will use CPU")
        
        num_joints = len(self.joint_limits)
        batch_size = 5
        
        # Generate random start and end configurations
        np.random.seed(42)  # For reproducible results
        
        theta_start_batch = np.zeros((batch_size, num_joints))
        theta_end_batch = np.zeros((batch_size, num_joints))
        
        for i in range(batch_size):
            for j in range(num_joints):
                # Random configurations within joint limits
                min_pos = self.joint_limits[j, 0]
                max_pos = self.joint_limits[j, 1]
                range_size = (max_pos - min_pos) * 0.6  # Use 60% of range
                center = (min_pos + max_pos) / 2
                
                theta_start_batch[i, j] = center + np.random.uniform(-range_size/4, range_size/4)
                theta_end_batch[i, j] = center + np.random.uniform(-range_size/2, range_size/2)
        
        logger.info(f"Generating {batch_size} trajectories simultaneously...")
        
        # Trajectory parameters
        T_final = 2.0
        N = 100
        method = 5  # Quintic
        
        # Batch generation
        start_time = time.time()
        batch_results = self.planner.batch_joint_trajectory(
            thetastart_batch=theta_start_batch,
            thetaend_batch=theta_end_batch,
            Tf=T_final,
            N=N,
            method=method
        )
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch generation completed in {batch_time:.4f}s")
        
        # Compare with sequential generation
        logger.info("Comparing with sequential generation...")
        
        sequential_results = []
        start_time = time.time()
        
        for i in range(batch_size):
            traj = self.planner.joint_trajectory(
                thetastart=theta_start_batch[i],
                thetaend=theta_end_batch[i],
                Tf=T_final,
                N=N,
                method=method
            )
            sequential_results.append(traj)
        
        sequential_time = time.time() - start_time
        
        logger.info(f"✅ Sequential generation completed in {sequential_time:.4f}s")
        
        # Performance analysis
        speedup = sequential_time / batch_time if batch_time > 0 else 0
        logger.info(f"📊 Batch Processing Performance:")
        logger.info(f"  Batch time: {batch_time:.4f}s")
        logger.info(f"  Sequential time: {sequential_time:.4f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Plot batch results
        self.plot_batch_trajectories(batch_results, theta_start_batch, theta_end_batch, T_final)
        
        # Get planner performance statistics
        stats = self.planner.get_performance_stats()
        logger.info(f"📊 Planner Statistics:")
        logger.info(f"  GPU calls: {stats['gpu_calls']}")
        logger.info(f"  CPU calls: {stats['cpu_calls']}")
        logger.info(f"  GPU usage: {stats['gpu_usage_percent']:.1f}%")
        
    def plot_batch_trajectories(self, batch_results, start_batch, end_batch, T_final):
        """Plot multiple trajectories from batch processing."""
        positions = batch_results['positions']
        batch_size, N, num_joints = positions.shape
        
        time_history = np.linspace(0, T_final, N)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Trajectory Generation Results', fontsize=16, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, batch_size))
        
        # Position trajectories for first joint
        ax = axes[0, 0]
        for i in range(batch_size):
            ax.plot(time_history, positions[i, :, 0], color=colors[i], 
                   linewidth=2, alpha=0.8, label=f'Traj {i+1}')
            ax.scatter([0, T_final], [start_batch[i, 0], end_batch[i, 0]], 
                      color=colors[i], s=50, zorder=5)
        
        ax.set_title('Joint 1 Position Trajectories', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity trajectories for first joint
        ax = axes[0, 1]
        velocities = batch_results['velocities']
        for i in range(batch_size):
            ax.plot(time_history, velocities[i, :, 0], color=colors[i], 
                   linewidth=2, alpha=0.8, label=f'Traj {i+1}')
        
        ax.set_title('Joint 1 Velocity Trajectories', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Start/End configuration comparison
        ax = axes[1, 0]
        joint_indices = np.arange(min(4, num_joints))
        width = 0.35
        
        mean_start = np.mean(start_batch, axis=0)[:len(joint_indices)]
        mean_end = np.mean(end_batch, axis=0)[:len(joint_indices)]
        
        ax.bar(joint_indices - width/2, mean_start, width, 
               label='Mean Start', alpha=0.8, color='skyblue')
        ax.bar(joint_indices + width/2, mean_end, width, 
               label='Mean End', alpha=0.8, color='lightcoral')
        
        ax.set_title('Average Start/End Configurations', fontweight='bold')
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Position (rad)')
        ax.set_xticks(joint_indices)
        ax.set_xticklabels([f'J{i+1}' for i in joint_indices])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Trajectory statistics
        ax = axes[1, 1]
        max_velocities = np.max(np.abs(velocities), axis=1)  # Max vel per trajectory
        max_accelerations = np.max(np.abs(batch_results['accelerations']), axis=1)
        
        trajectory_indices = np.arange(batch_size)
        
        ax.bar(trajectory_indices - width/2, np.max(max_velocities, axis=1), width, 
               label='Max Velocity', alpha=0.8, color='green')
        ax.bar(trajectory_indices + width/2, np.max(max_accelerations, axis=1), width, 
               label='Max Acceleration', alpha=0.8, color='orange')
        
        ax.set_title('Trajectory Statistics', fontweight='bold')
        ax.set_xlabel('Trajectory Index')
        ax.set_ylabel('Value')
        ax.set_xticks(trajectory_indices)
        ax.set_xticklabels([f'T{i+1}' for i in range(batch_size)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'batch_trajectories.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Batch trajectory plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_cartesian_trajectory_planning(self):
        """Demonstrate Cartesian space trajectory planning."""
        logger.info("\n🎯 Demonstrating Cartesian Trajectory Planning...")
        
        num_joints = len(self.joint_limits)
        
        # Choose safe joint configurations for start and end
        theta_start = np.zeros(num_joints)
        
        # Safe end configuration
        range_factor = 0.3
        theta_end = np.array([
            (self.joint_limits[i, 1] - self.joint_limits[i, 0]) * range_factor * np.sin(i)
            for i in range(num_joints)
        ])
        
        # Get end-effector poses
        T_start = self.robot.forward_kinematics(theta_start)
        T_end = self.robot.forward_kinematics(theta_end)
        
        logger.info("Forward kinematics computed:")
        logger.info(f"  Start EE position: {T_start[:3, 3]}")
        logger.info(f"  End EE position: {T_end[:3, 3]}")
        
        # Generate Cartesian trajectory
        T_final = 3.0
        N = 100
        method = 5  # Quintic
        
        logger.info("Generating Cartesian space trajectory...")
        
        start_time = time.time()
        cartesian_traj = self.planner.cartesian_trajectory(
            Xstart=T_start,
            Xend=T_end,
            Tf=T_final,
            N=N,
            method=method
        )
        cartesian_time = time.time() - start_time
        
        logger.info(f"✅ Cartesian trajectory generated in {cartesian_time:.4f}s")
        
        # Generate equivalent joint space trajectory
        logger.info("Generating equivalent joint space trajectory...")
        
        start_time = time.time()
        joint_traj = self.planner.joint_trajectory(
            thetastart=theta_start,
            thetaend=theta_end,
            Tf=T_final,
            N=N,
            method=method
        )
        joint_time = time.time() - start_time
        
        logger.info(f"✅ Joint trajectory generated in {joint_time:.4f}s")
        
        # Compare end-effector paths
        self.compare_cartesian_paths(cartesian_traj, joint_traj, theta_start, T_final)
        
        logger.info("✅ Cartesian trajectory planning demonstration complete")
    
    def compare_cartesian_paths(self, cartesian_traj, joint_traj, theta_start, T_final):
        """Compare Cartesian and joint space trajectories in end-effector space."""
        # Compute end-effector positions for joint trajectory
        joint_positions = joint_traj['positions']
        N = joint_positions.shape[0]
        
        ee_positions_joint = np.zeros((N, 3))
        for i in range(N):
            T = self.robot.forward_kinematics(joint_positions[i])
            ee_positions_joint[i] = T[:3, 3]
        
        # Cartesian trajectory positions
        cartesian_positions = cartesian_traj['positions']
        
        time_history = np.linspace(0, T_final, N)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cartesian vs Joint Space Trajectory Comparison', fontsize=16, fontweight='bold')
        
        # 3D path comparison
        ax = axes[0, 0]
        ax.plot(cartesian_positions[:, 0], cartesian_positions[:, 1], 
               'b-', linewidth=3, label='Cartesian Space', alpha=0.8)
        ax.plot(ee_positions_joint[:, 0], ee_positions_joint[:, 1], 
               'r--', linewidth=2, label='Joint Space', alpha=0.8)
        ax.scatter(cartesian_positions[0, 0], cartesian_positions[0, 1], 
                  color='green', s=100, label='Start', zorder=5)
        ax.scatter(cartesian_positions[-1, 0], cartesian_positions[-1, 1], 
                  color='red', s=100, label='End', zorder=5)
        
        ax.set_title('End-Effector Path (X-Y Plane)', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Position vs time
        ax = axes[0, 1]
        coords = ['X', 'Y', 'Z']
        for i in range(3):
            ax.plot(time_history, cartesian_positions[:, i], 
                   linewidth=2, label=f'Cartesian {coords[i]}', linestyle='-')
            ax.plot(time_history, ee_positions_joint[:, i], 
                   linewidth=2, label=f'Joint {coords[i]}', linestyle='--', alpha=0.8)
        
        ax.set_title('End-Effector Position vs Time', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity comparison
        ax = axes[1, 0]
        cartesian_velocities = cartesian_traj['velocities']
        joint_velocities = joint_traj['velocities']
        
        # Compute end-effector velocities for joint trajectory
        ee_velocities_joint = np.zeros((N, 3))
        for i in range(N):
            J = self.robot.jacobian(joint_positions[i])
            ee_vel_6d = J @ joint_velocities[i]
            ee_velocities_joint[i] = ee_vel_6d[:3]  # Linear velocities only
        
        for i in range(3):
            ax.plot(time_history, cartesian_velocities[:, i], 
                   linewidth=2, label=f'Cartesian {coords[i]}', linestyle='-')
            ax.plot(time_history, ee_velocities_joint[:, i], 
                   linewidth=2, label=f'Joint {coords[i]}', linestyle='--', alpha=0.8)
        
        ax.set_title('End-Effector Velocity vs Time', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Path error analysis
        ax = axes[1, 1]
        path_error = np.linalg.norm(cartesian_positions - ee_positions_joint, axis=1)
        max_error = np.max(path_error)
        mean_error = np.mean(path_error)
        
        ax.plot(time_history, path_error, 'purple', linewidth=2)
        ax.axhline(y=mean_error, color='orange', linestyle='--', 
                  label=f'Mean Error: {mean_error:.4f}m')
        ax.axhline(y=max_error, color='red', linestyle='--', 
                  label=f'Max Error: {max_error:.4f}m')
        
        ax.set_title('End-Effector Path Error', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'cartesian_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Cartesian comparison plot saved as '{save_path}'")
        plt.close()
        
        logger.info(f"📊 Path Error Analysis:")
        logger.info(f"  Maximum error: {max_error:.6f} m")
        logger.info(f"  Mean error: {mean_error:.6f} m")
        logger.info(f"  RMS error: {np.sqrt(np.mean(path_error**2)):.6f} m")
    
    def demonstrate_trajectory_optimization(self):
        """Demonstrate trajectory optimization for minimum time and smooth motion."""
        logger.info("\n🎯 Demonstrating Trajectory Optimization...")
        
        num_joints = len(self.joint_limits)
        
        # Define optimization problem
        theta_start = np.zeros(num_joints)
        theta_end = self.get_safe_waypoints(2)[1]  # Use second waypoint as target
        
        logger.info(f"Optimizing trajectory from {theta_start} to {theta_end}")
        
        # Test different time durations to find optimal
        time_candidates = [1.0, 2.0, 3.0, 4.0, 5.0]
        N = 100
        method = 5
        
        optimization_results = {}
        
        for T_final in time_candidates:
            logger.info(f"  Testing T = {T_final}s...")
            
            start_time = time.time()
            trajectory = self.planner.joint_trajectory(
                thetastart=theta_start,
                thetaend=theta_end,
                Tf=T_final,
                N=N,
                method=method
            )
            computation_time = time.time() - start_time
            
            # Calculate optimization metrics
            velocities = trajectory['velocities']
            accelerations = trajectory['accelerations']
            
            # Compute jerk (rate of change of acceleration)
            dt = T_final / (N - 1)
            jerk = np.diff(accelerations, axis=0) / dt
            
            # Optimization criteria
            max_velocity = np.max(np.abs(velocities))
            max_acceleration = np.max(np.abs(accelerations))
            max_jerk = np.max(np.abs(jerk))
            rms_jerk = np.sqrt(np.mean(jerk**2))
            
            # Energy-like criterion (integral of squared acceleration)
            energy = np.sum(accelerations**2) * dt
            
            # Smoothness criterion (integral of squared jerk)
            smoothness = np.sum(jerk**2) * dt
            
            optimization_results[T_final] = {
                'trajectory': trajectory,
                'computation_time': computation_time,
                'max_velocity': max_velocity,
                'max_acceleration': max_acceleration,
                'max_jerk': max_jerk,
                'rms_jerk': rms_jerk,
                'energy': energy,
                'smoothness': smoothness
            }
            
            logger.info(f"    Max vel: {max_velocity:.4f}, Max acc: {max_acceleration:.4f}")
            logger.info(f"    Energy: {energy:.4f}, Smoothness: {smoothness:.4f}")
        
        # Find optimal trajectories
        self.analyze_trajectory_optimization(optimization_results)
        
        # Plot optimization results
        self.plot_trajectory_optimization(optimization_results)
        
        logger.info("✅ Trajectory optimization demonstration complete")
    
    def analyze_trajectory_optimization(self, results):
        """Analyze trajectory optimization results."""
        logger.info("\n📊 Trajectory Optimization Analysis:")
        
        times = list(results.keys())
        
        # Find optimal trajectories for different criteria
        min_time_idx = 0  # Fastest trajectory
        min_energy_idx = min(range(len(times)), 
                            key=lambda i: results[times[i]]['energy'])
        min_smoothness_idx = min(range(len(times)), 
                               key=lambda i: results[times[i]]['smoothness'])
        min_jerk_idx = min(range(len(times)), 
                         key=lambda i: results[times[i]]['max_jerk'])
        
        criteria = {
            'Minimum Time': times[min_time_idx],
            'Minimum Energy': times[min_energy_idx],
            'Maximum Smoothness': times[min_smoothness_idx],
            'Minimum Jerk': times[min_jerk_idx]
        }
        
        for criterion, optimal_time in criteria.items():
            result = results[optimal_time]
            logger.info(f"  {criterion} (T = {optimal_time}s):")
            logger.info(f"    Max velocity: {result['max_velocity']:.4f} rad/s")
            logger.info(f"    Max acceleration: {result['max_acceleration']:.4f} rad/s²")
            logger.info(f"    Max jerk: {result['max_jerk']:.4f} rad/s³")
            logger.info(f"    Energy: {result['energy']:.4f}")
            logger.info(f"    Smoothness: {result['smoothness']:.4f}")
            logger.info(f"    Computation time: {result['computation_time']:.4f}s")
    
    def plot_trajectory_optimization(self, results):
        """Plot trajectory optimization results."""
        times = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trajectory Optimization Results', fontsize=16, fontweight='bold')
        
        # Optimization criteria vs time
        ax = axes[0, 0]
        energies = [results[t]['energy'] for t in times]
        smoothness = [results[t]['smoothness'] for t in times]
        
        ax2 = ax.twinx()
        line1 = ax.plot(times, energies, 'b-o', linewidth=2, label='Energy')
        line2 = ax2.plot(times, smoothness, 'r-s', linewidth=2, label='Smoothness')
        
        ax.set_xlabel('Trajectory Time (s)')
        ax.set_ylabel('Energy', color='b')
        ax2.set_ylabel('Smoothness', color='r')
        ax.set_title('Energy and Smoothness vs Trajectory Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        # Max values vs time
        ax = axes[0, 1]
        max_vels = [results[t]['max_velocity'] for t in times]
        max_accs = [results[t]['max_acceleration'] for t in times]
        max_jerks = [results[t]['max_jerk'] for t in times]
        
        ax.plot(times, max_vels, 'g-o', linewidth=2, label='Max Velocity')
        ax.plot(times, max_accs, 'b-s', linewidth=2, label='Max Acceleration')
        ax.plot(times, max_jerks, 'r-^', linewidth=2, label='Max Jerk')
        
        ax.set_xlabel('Trajectory Time (s)')
        ax.set_ylabel('Maximum Values')
        ax.set_title('Maximum Kinematic Values vs Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Sample trajectories comparison
        ax = axes[1, 0]
        sample_times = [times[0], times[len(times)//2], times[-1]]
        colors = ['red', 'blue', 'green']
        
        for i, (t, color) in enumerate(zip(sample_times, colors)):
            traj = results[t]['trajectory']
            positions = traj['positions']
            N = positions.shape[0]
            time_hist = np.linspace(0, t, N)
            
            # Plot first joint only for clarity
            ax.plot(time_hist, positions[:, 0], color=color, 
                   linewidth=2, label=f'T = {t}s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint 1 Position (rad)')
        ax.set_title('Sample Trajectory Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Computation time vs trajectory time
        ax = axes[1, 1]
        comp_times = [results[t]['computation_time'] for t in times]
        
        bars = ax.bar(range(len(times)), comp_times, color='skyblue', alpha=0.8)
        ax.set_xlabel('Trajectory Duration')
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Computation Time vs Trajectory Duration', fontweight='bold')
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([f'{t}s' for t in times])
        ax.grid(True, alpha=0.3)
        
        # Add computation time values on bars
        for bar, comp_time in zip(bars, comp_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{comp_time:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'trajectory_optimization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Trajectory optimization plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_performance_benchmarking(self):
        """Demonstrate performance benchmarking of trajectory planning algorithms."""
        logger.info("\n🎯 Demonstrating Performance Benchmarking...")
        
        # Get current planner performance stats
        initial_stats = self.planner.get_performance_stats()
        logger.info(f"Initial planner statistics: {initial_stats}")
        
        # Reset performance statistics for clean benchmarking
        self.planner.reset_performance_stats()
        
        # Benchmark different problem sizes
        test_cases = [
            {"N": 50, "joints": len(self.joint_limits), "name": "Small"},
            {"N": 200, "joints": len(self.joint_limits), "name": "Medium"},
            {"N": 500, "joints": len(self.joint_limits), "name": "Large"},
            {"N": 1000, "joints": len(self.joint_limits), "name": "Very Large"},
        ]
        
        logger.info("Running performance benchmarks...")
        
        benchmark_results = self.planner.benchmark_performance(test_cases)
        
        # Display results
        logger.info("\n📊 Benchmark Results:")
        for case_name, result in benchmark_results.items():
            logger.info(f"  {case_name}:")
            logger.info(f"    Problem size: {result['N']} × {result['joints']}")
            logger.info(f"    Total time: {result['total_time']:.4f}s")
            logger.info(f"    Used GPU: {result['used_gpu']}")
            logger.info(f"    Trajectory shape: {result['trajectory_shape']}")
            
            stats = result['stats']
            logger.info(f"    GPU calls: {stats['gpu_calls']}")
            logger.info(f"    CPU calls: {stats['cpu_calls']}")
            if stats['gpu_calls'] > 0:
                logger.info(f"    Avg GPU time: {stats['avg_gpu_time']:.4f}s")
            if stats['cpu_calls'] > 0:
                logger.info(f"    Avg CPU time: {stats['avg_cpu_time']:.4f}s")
        
        # Plot benchmark results
        self.plot_performance_benchmark(benchmark_results)
        
        # Test batch processing performance if GPU is available
        if check_cuda_availability():
            logger.info("\nTesting batch processing performance...")
            self.benchmark_batch_processing()
        
        logger.info("✅ Performance benchmarking demonstration complete")
    
    def plot_performance_benchmark(self, results):
        """Plot performance benchmark results."""
        case_names = list(results.keys())
        total_times = [results[name]['total_time'] for name in case_names]
        problem_sizes = [results[name]['N'] * results[name]['joints'] for name in case_names]
        gpu_usage = [results[name]['used_gpu'] for name in case_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Benchmarking Results', fontsize=16, fontweight='bold')
        
        # Execution time vs problem size
        ax = axes[0]
        colors = ['green' if gpu else 'blue' for gpu in gpu_usage]
        bars = ax.bar(case_names, total_times, color=colors, alpha=0.8)
        
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Execution Time by Problem Size', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add time values on bars and GPU/CPU labels
        for bar, time_val, gpu in zip(bars, total_times, gpu_usage):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.4f}s\n{"GPU" if gpu else "CPU"}', 
                   ha='center', va='bottom')
        
        # Throughput (trajectories per second)
        ax = axes[1]
        throughput = [1.0 / time_val for time_val in total_times]
        bars = ax.bar(case_names, throughput, color=colors, alpha=0.8)
        
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Throughput (trajectories/s)')
        ax.set_title('Throughput by Problem Size', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add throughput values on bars
        for bar, tp in zip(bars, throughput):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{tp:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'performance_benchmark.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Performance benchmark plot saved as '{save_path}'")
        plt.close()
    
    def benchmark_batch_processing(self):
        """Benchmark batch processing capabilities."""
        batch_sizes = [1, 5, 10, 20]
        num_joints = len(self.joint_limits)
        N = 100
        T_final = 2.0
        method = 5
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            # Generate random configurations
            np.random.seed(42)
            theta_start_batch = np.zeros((batch_size, num_joints))
            theta_end_batch = np.zeros((batch_size, num_joints))
            
            for i in range(batch_size):
                for j in range(num_joints):
                    min_pos = self.joint_limits[j, 0]
                    max_pos = self.joint_limits[j, 1]
                    range_size = (max_pos - min_pos) * 0.4
                    center = (min_pos + max_pos) / 2
                    
                    theta_start_batch[i, j] = center + np.random.uniform(-range_size/4, range_size/4)
                    theta_end_batch[i, j] = center + np.random.uniform(-range_size/2, range_size/2)
            
            # Time batch processing
            start_time = time.time()
            if batch_size == 1:
                # Single trajectory
                result = self.planner.joint_trajectory(
                    thetastart=theta_start_batch[0],
                    thetaend=theta_end_batch[0],
                    Tf=T_final,
                    N=N,
                    method=method
                )
            else:
                # Batch processing
                result = self.planner.batch_joint_trajectory(
                    thetastart_batch=theta_start_batch,
                    thetaend_batch=theta_end_batch,
                    Tf=T_final,
                    N=N,
                    method=method
                )
            
            batch_time = time.time() - start_time
            
            # Calculate throughput
            throughput = batch_size / batch_time
            
            batch_results[batch_size] = {
                'time': batch_time,
                'throughput': throughput,
                'time_per_trajectory': batch_time / batch_size
            }
            
            logger.info(f"    Time: {batch_time:.4f}s, Throughput: {throughput:.2f} traj/s")
        
        # Plot batch processing results
        self.plot_batch_benchmark(batch_results)
    
    def plot_batch_benchmark(self, results):
        """Plot batch processing benchmark results."""
        batch_sizes = list(results.keys())
        times = [results[bs]['time'] for bs in batch_sizes]
        throughputs = [results[bs]['throughput'] for bs in batch_sizes]
        time_per_traj = [results[bs]['time_per_trajectory'] for bs in batch_sizes]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Batch Processing Performance', fontsize=16, fontweight='bold')
        
        # Throughput vs batch size
        ax = axes[0]
        ax.plot(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (trajectories/s)')
        ax.set_title('Throughput vs Batch Size', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Time per trajectory
        ax = axes[1]
        ax.plot(batch_sizes, time_per_traj, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Time per Trajectory (s)')
        ax.set_title('Efficiency vs Batch Size', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (bs, tp, tpt) in enumerate(zip(batch_sizes, throughputs, time_per_traj)):
            axes[0].annotate(f'{tp:.1f}', (bs, tp), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            axes[1].annotate(f'{tpt:.3f}', (bs, tpt), textcoords="offset points", 
                           xytext=(0,10), ha='center')
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'batch_benchmark.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Batch benchmark plot saved as '{save_path}'")
        plt.close()
    
    def demonstrate_trajectory_smoothing(self):
        """Demonstrate trajectory smoothing and filtering techniques."""
        logger.info("\n🎯 Demonstrating Trajectory Smoothing...")
        
        num_joints = len(self.joint_limits)
        
        # Generate a trajectory with some noise
        theta_start = np.zeros(num_joints)
        theta_end = self.get_safe_waypoints(2)[1]
        
        # Generate base trajectory
        T_final = 3.0
        N = 200  # Higher resolution for smoothing demo
        
        trajectory = self.planner.joint_trajectory(
            thetastart=theta_start,
            thetaend=theta_end,
            Tf=T_final,
            N=N,
            method=5
        )
        
        # Add noise to simulate measurement errors
        np.random.seed(42)
        noise_level = 0.01  # 1% noise
        noisy_positions = trajectory['positions'] + noise_level * np.random.randn(*trajectory['positions'].shape)
        
        # Apply different smoothing techniques
        smoothed_results = self.apply_smoothing_techniques(noisy_positions, T_final)
        
        # Plot smoothing comparison
        self.plot_smoothing_comparison(trajectory['positions'], noisy_positions, smoothed_results, T_final)
        
        # Analyze smoothing effectiveness
        self.analyze_smoothing_effectiveness(trajectory['positions'], noisy_positions, smoothed_results)
        
        logger.info("✅ Trajectory smoothing demonstration complete")
    
    def apply_smoothing_techniques(self, noisy_positions, T_final):
        """Apply various smoothing techniques to trajectory data."""
        from scipy import signal
        from scipy.ndimage import gaussian_filter1d
        
        smoothed_results = {}
        
        # 1. Moving average filter
        window_size = 5
        smoothed_results['Moving Average'] = np.array([
            np.convolve(noisy_positions[:, j], np.ones(window_size)/window_size, mode='same')
            for j in range(noisy_positions.shape[1])
        ]).T
        
        # 2. Gaussian filter
        sigma = 2.0
        smoothed_results['Gaussian'] = np.array([
            gaussian_filter1d(noisy_positions[:, j], sigma=sigma)
            for j in range(noisy_positions.shape[1])
        ]).T
        
        # 3. Butterworth low-pass filter
        fs = noisy_positions.shape[0] / T_final  # Sampling frequency
        cutoff = fs / 10  # Cutoff frequency
        b, a = signal.butter(4, cutoff, btype='low', fs=fs)
        smoothed_results['Butterworth'] = np.array([
            signal.filtfilt(b, a, noisy_positions[:, j])
            for j in range(noisy_positions.shape[1])
        ]).T
        
        # 4. Savitzky-Golay filter
        window_length = min(21, noisy_positions.shape[0] // 2 * 2 + 1)  # Ensure odd number
        polyorder = 3
        smoothed_results['Savgol'] = np.array([
            signal.savgol_filter(noisy_positions[:, j], window_length, polyorder)
            for j in range(noisy_positions.shape[1])
        ]).T
        
        return smoothed_results
    
    def plot_smoothing_comparison(self, original, noisy, smoothed_results, T_final):
        """Plot comparison of different smoothing techniques."""
        N = original.shape[0]
        time_history = np.linspace(0, T_final, N)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trajectory Smoothing Comparison', fontsize=16, fontweight='bold')
        
        # Joint 1 comparison
        ax = axes[0, 0]
        ax.plot(time_history, original[:, 0], 'k-', linewidth=3, label='Original', alpha=0.8)
        ax.plot(time_history, noisy[:, 0], 'r:', linewidth=1, label='Noisy', alpha=0.6)
        
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (method, smoothed) in enumerate(smoothed_results.items()):
            ax.plot(time_history, smoothed[:, 0], color=colors[i], 
                   linewidth=2, label=method, alpha=0.8)
        
        ax.set_title('Joint 1 Position - Smoothing Comparison', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed view of a section
        ax = axes[0, 1]
        zoom_start, zoom_end = N//3, 2*N//3
        zoom_time = time_history[zoom_start:zoom_end]
        
        ax.plot(zoom_time, original[zoom_start:zoom_end, 0], 'k-', 
               linewidth=3, label='Original', alpha=0.8)
        ax.plot(zoom_time, noisy[zoom_start:zoom_end, 0], 'r:', 
               linewidth=1, label='Noisy', alpha=0.6)
        
        for i, (method, smoothed) in enumerate(smoothed_results.items()):
            ax.plot(zoom_time, smoothed[zoom_start:zoom_end, 0], color=colors[i], 
                   linewidth=2, label=method, alpha=0.8)
        
        ax.set_title('Zoomed View - Middle Section', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error comparison
        ax = axes[1, 0]
        for i, (method, smoothed) in enumerate(smoothed_results.items()):
            error = np.linalg.norm(smoothed - original, axis=1)
            ax.plot(time_history, error, color=colors[i], 
                   linewidth=2, label=f'{method}', alpha=0.8)
        
        # Noisy error for reference
        noisy_error = np.linalg.norm(noisy - original, axis=1)
        ax.plot(time_history, noisy_error, 'r:', linewidth=1, label='Noisy', alpha=0.6)
        
        ax.set_title('Reconstruction Error vs Time', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('L2 Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # RMS error comparison
        ax = axes[1, 1]
        methods = list(smoothed_results.keys()) + ['Noisy']
        rms_errors = []
        
        for method, smoothed in smoothed_results.items():
            rms_error = np.sqrt(np.mean((smoothed - original)**2))
            rms_errors.append(rms_error)
        
        # Add noisy RMS error
        noisy_rms = np.sqrt(np.mean((noisy - original)**2))
        rms_errors.append(noisy_rms)
        
        bars = ax.bar(methods, rms_errors, color=colors + ['red'], alpha=0.8)
        ax.set_title('RMS Error Comparison', fontweight='bold')
        ax.set_ylabel('RMS Error')
        ax.grid(True, alpha=0.3)
        
        # Add error values on bars
        for bar, error in zip(bars, rms_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{error:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.script_dir, 'trajectory_smoothing.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Trajectory smoothing plot saved as '{save_path}'")
        plt.close()
    
    def analyze_smoothing_effectiveness(self, original, noisy, smoothed_results):
        """Analyze the effectiveness of different smoothing techniques."""
        logger.info("\n📊 Smoothing Effectiveness Analysis:")
        
        # Calculate metrics for noisy data
        noisy_rms = np.sqrt(np.mean((noisy - original)**2))
        noisy_max = np.max(np.abs(noisy - original))
        
        logger.info(f"  Noisy data:")
        logger.info(f"    RMS error: {noisy_rms:.6f}")
        logger.info(f"    Max error: {noisy_max:.6f}")
        
        # Calculate metrics for each smoothing method
        for method, smoothed in smoothed_results.items():
            rms_error = np.sqrt(np.mean((smoothed - original)**2))
            max_error = np.max(np.abs(smoothed - original))
            improvement = (noisy_rms - rms_error) / noisy_rms * 100
            
            # Calculate smoothness (second derivative)
            dt = 1.0 / (smoothed.shape[0] - 1)
            second_deriv = np.diff(smoothed, n=2, axis=0) / (dt**2)
            smoothness = np.mean(np.var(second_deriv, axis=0))
            
            logger.info(f"  {method}:")
            logger.info(f"    RMS error: {rms_error:.6f}")
            logger.info(f"    Max error: {max_error:.6f}")
            logger.info(f"    Improvement: {improvement:.1f}%")
            logger.info(f"    Smoothness metric: {smoothness:.6f}")
    
    def run_complete_demonstration(self):
        """Run the complete intermediate trajectory planning demonstration."""
        logger.info("🚀 Starting Intermediate Trajectory Planning Demonstration")
        logger.info("=" * 70)
        
        try:
            # 1. Basic trajectory generation
            self.demonstrate_basic_trajectory_generation()
            
            # 2. Multi-segment trajectories
            self.demonstrate_multi_segment_trajectories()
            
            # 3. Batch processing
            self.demonstrate_batch_trajectory_generation()
            
            # 4. Cartesian space planning
            self.demonstrate_cartesian_trajectory_planning()
            
            # 5. Trajectory optimization
            self.demonstrate_trajectory_optimization()
            
            # 6. Performance benchmarking
            self.demonstrate_performance_benchmarking()
            
            # 7. Trajectory smoothing
            self.demonstrate_trajectory_smoothing()
            
            # Final summary
            self.print_demonstration_summary()
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {e}")
            raise
        
        logger.info("🎉 Intermediate Trajectory Planning Demonstration Complete!")
        logger.info("=" * 70)
    
    def print_demonstration_summary(self):
        """Print a summary of the demonstration results."""
        logger.info("\n📋 Demonstration Summary:")
        logger.info("  ✅ Basic trajectory generation (cubic & quintic)")
        logger.info("  ✅ Multi-segment trajectory through waypoints") 
        logger.info("  ✅ Batch trajectory processing")
        logger.info("  ✅ Cartesian space trajectory planning")
        logger.info("  ✅ Trajectory optimization analysis")
        logger.info("  ✅ Performance benchmarking")
        logger.info("  ✅ Trajectory smoothing and filtering")
        
        # Get final planner statistics
        final_stats = self.planner.get_performance_stats()
        logger.info(f"\n📊 Final Planner Statistics:")
        logger.info(f"  Total GPU calls: {final_stats['gpu_calls']}")
        logger.info(f"  Total CPU calls: {final_stats['cpu_calls']}")
        logger.info(f"  GPU usage: {final_stats['gpu_usage_percent']:.1f}%")
        logger.info(f"  Average GPU time: {final_stats['avg_gpu_time']:.4f}s")
        logger.info(f"  Average CPU time: {final_stats['avg_cpu_time']:.4f}s")
        
        logger.info(f"\n📁 Generated Files:")
        generated_files = [
            'trajectory_comparison.png',
            'multi_segment_trajectory.png', 
            'batch_trajectories.png',
            'cartesian_comparison.png',
            'trajectory_optimization.png',
            'performance_benchmark.png',
            'batch_benchmark.png',
            'trajectory_smoothing.png'
        ]
        
        for filename in generated_files:
            file_path = os.path.join(self.script_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"  ✅ {filename}")
            else:
                logger.info(f"  ❌ {filename} (not generated)")


def main():
    """Main function to run the intermediate trajectory planning demonstration."""
    print("🔧 Intermediate Trajectory Planning Demo - ManipulaPy")
    print("=" * 60)
    
    try:
        # Initialize the demonstration
        # Try XArm first, fall back to simple robot if needed
        demo = IntermediateTrajectoryDemo(use_simple_robot=False)
        
        # Run complete demonstration
        demo.run_complete_demonstration()
        
        # Cleanup
        demo.planner.cleanup_gpu_memory()
        
        print("\n🎯 Demo completed successfully!")
        print("Check the generated PNG files for visualization results.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Full error details:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())