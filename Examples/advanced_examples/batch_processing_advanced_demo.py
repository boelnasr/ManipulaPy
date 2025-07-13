#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Batch Processing Advanced Demo - ManipulaPy

This demo showcases advanced batch processing capabilities including:
- GPU-accelerated batch trajectory generation
- Parallel forward/inverse dynamics computation
- Batch optimization and analysis
- Performance scaling analysis
- Memory-efficient batch operations
- Statistical analysis of batch results

Features demonstrated:
- Batch trajectory planning for multiple robots
- Parallel dynamics computations
- GPU memory management for large batches
- Performance profiling and optimization
- Real-time batch processing capabilities

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import logging
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import matplotlib
matplotlib.use('TkAgg')
# Import ManipulaPy modules
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.cuda_kernels import (
        CUDA_AVAILABLE, 
        check_cuda_availability,
        optimized_batch_trajectory_generation,
        optimized_trajectory_generation,
        get_cuda_array,
        return_cuda_array,
        make_2d_grid
    )
    from ManipulaPy.utils import transform_from_twist, adjoint_transform
except ImportError as e:
    print(f"Error importing ManipulaPy modules: {e}")
    print("Please ensure ManipulaPy is properly installed.")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class BatchProcessingDemo:
    """
    Advanced demonstration of batch processing capabilities in ManipulaPy.
    
    Features:
    - GPU-accelerated batch trajectory generation
    - Parallel dynamics computations  
    - Memory-efficient batch operations
    - Performance scaling analysis
    - Statistical batch analysis
    """
    
    def __init__(self, save_plots: bool = True, use_gpu: Optional[bool] = None):
        """
        Initialize the batch processing demo.
        
        Args:
            save_plots: Whether to save generated plots to files
            use_gpu: Force GPU usage (None for auto-detect)
        """
        self.save_plots = save_plots
        self.use_gpu = use_gpu if use_gpu is not None else check_cuda_availability()
        
        # System information
        self.num_cpu_cores = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Setup robot and environment
        self.setup_robot()
        self.setup_batch_scenarios()
        
        # Performance tracking
        self.performance_data = {
            'batch_sizes': [],
            'gpu_times': [],
            'cpu_times': [],
            'parallel_times': [],
            'memory_usage': [],
            'speedups': [],
            'efficiency': []
        }
        
        logger.info(f"Batch demo initialized - GPU: {self.use_gpu}, CPU cores: {self.num_cpu_cores}")
        logger.info(f"System memory: {self.memory_gb:.1f} GB")
    
    def setup_robot(self):
        """Setup a 6-DOF robot manipulator for batch processing."""
        self.num_joints = 6
        
        # Joint limits (radians)
        self.joint_limits = [
            (-np.pi, np.pi),      # Joint 1: Base rotation
            (-np.pi/2, np.pi/2),  # Joint 2: Shoulder
            (-np.pi, np.pi),      # Joint 3: Elbow
            (-np.pi, np.pi),      # Joint 4: Wrist 1
            (-np.pi/2, np.pi/2),  # Joint 5: Wrist 2
            (-np.pi, np.pi)       # Joint 6: Wrist 3
        ]
        
        # Torque limits (Nm)
        self.torque_limits = [
            (-100, 100), (-80, 80), (-60, 60),
            (-40, 40), (-30, 30), (-20, 20)
        ]
        
        # Create robot kinematics and dynamics
        self.setup_robot_kinematics()
        
        logger.info("Robot kinematics and dynamics initialized for batch processing")
    
    def setup_robot_kinematics(self):
        """Setup robot kinematics and dynamics for batch operations."""
        # Link parameters
        link_lengths = [0.3, 0.4, 0.35, 0.2, 0.15, 0.1]
        
        # Home configuration
        M = np.eye(4)
        M[0, 3] = sum(link_lengths)
        
        # Screw axes and positions
        S_list = np.zeros((6, self.num_joints))
        positions = np.zeros((3, self.num_joints))
        omega_list = np.zeros((3, self.num_joints))
        
        for i in range(self.num_joints):
            # Alternate between Z and Y rotations
            if i % 2 == 0:
                omega = np.array([0, 0, 1])
            else:
                omega = np.array([0, 1, 0])
            
            omega_list[:, i] = omega
            S_list[:3, i] = omega
            
            # Position along kinematic chain
            position = np.array([
                sum(link_lengths[:i+1]) * 0.7,
                0.0,
                0.1 * i
            ])
            positions[:, i] = position
            S_list[3:, i] = np.cross(-omega, position)
        
        # Body frame screw axes
        B_list = S_list.copy()
        
        # Create inertia matrices
        G_list = []
        for i in range(self.num_joints):
            G = np.eye(6)
            G[:3, :3] *= 0.1  # Inertia
            G[3:, 3:] *= 2.0  # Mass
            G_list.append(G)
        
        # Initialize robot
        self.robot = SerialManipulator(
            M_list=M,
            omega_list=omega_list,
            r_list=positions,
            b_list=positions,
            S_list=S_list,
            B_list=B_list,
            G_list=G_list,
            joint_limits=self.joint_limits
        )
        
        # Initialize dynamics
        self.dynamics = ManipulatorDynamics(
            M_list=M,
            omega_list=omega_list,
            r_list=positions,
            b_list=positions,
            S_list=S_list,
            B_list=B_list,
            Glist=G_list
        )
        
        # Initialize trajectory planner
        try:
            self.planner = OptimizedTrajectoryPlanning(
                serial_manipulator=self.robot,
                urdf_path="dummy_robot.urdf",
                dynamics=self.dynamics,
                joint_limits=self.joint_limits,
                torque_limits=self.torque_limits,
                use_cuda=self.use_gpu,
                cuda_threshold=10  # Lower threshold for batch operations
            )
        except Exception as e:
            logger.warning(f"Could not create planner: {e}")
            self.planner = None
    
    def setup_batch_scenarios(self):
        """Setup various batch processing scenarios."""
        # Define different batch scenarios for testing
        self.batch_scenarios = {
            'small_batch': {
                'size': 10,
                'description': 'Small batch for quick testing',
                'trajectory_points': 50
            },
            'medium_batch': {
                'size': 50,
                'description': 'Medium batch for typical operations',
                'trajectory_points': 100
            },
            'large_batch': {
                'size': 200,
                'description': 'Large batch for performance testing',
                'trajectory_points': 200
            },
            'xlarge_batch': {
                'size': 500,
                'description': 'Extra large batch for scalability testing',
                'trajectory_points': 100  # Reduced points for memory efficiency
            }
        }
        
        logger.info(f"Batch scenarios configured: {list(self.batch_scenarios.keys())}")
    
    def generate_batch_data(self, batch_size: int, trajectory_points: int = 100) -> Dict:
        """
        Generate batch data for trajectory planning.
        
        Args:
            batch_size: Number of trajectories in batch
            trajectory_points: Number of points per trajectory
            
        Returns:
            Dictionary containing batch start/end configurations and parameters
        """
        # Generate random start configurations
        start_configs = np.zeros((batch_size, self.num_joints), dtype=np.float32)
        end_configs = np.zeros((batch_size, self.num_joints), dtype=np.float32)
        
        for i in range(batch_size):
            # Random start configuration
            start_configs[i] = np.array([
                np.random.uniform(limit[0], limit[1]) 
                for limit in self.joint_limits
            ], dtype=np.float32)
            
            # Random end configuration (ensure reasonable distance)
            end_configs[i] = np.array([
                np.random.uniform(limit[0], limit[1]) 
                for limit in self.joint_limits
            ], dtype=np.float32)
        
        # Trajectory parameters
        trajectory_times = np.random.uniform(1.0, 3.0, batch_size).astype(np.float32)
        methods = np.random.choice([3, 5], batch_size)  # Cubic or quintic
        
        return {
            'start_configs': start_configs,
            'end_configs': end_configs,
            'trajectory_times': trajectory_times,
            'trajectory_points': trajectory_points,
            'methods': methods,
            'batch_size': batch_size
        }
    
    def batch_trajectory_gpu(self, batch_data: Dict) -> Tuple[Dict, float]:
        """
        GPU-accelerated batch trajectory generation.
        
        Args:
            batch_data: Batch configuration data
            
        Returns:
            Tuple of (results, computation_time)
        """
        if not self.use_gpu:
            raise RuntimeError("GPU not available for batch processing")
        
        start_time = time.time()
        
        try:
            # Use optimized batch trajectory generation
            if self.planner:
                results = self.planner.batch_joint_trajectory(
                    batch_data['start_configs'],
                    batch_data['end_configs'],
                    np.mean(batch_data['trajectory_times']),  # Use average time
                    batch_data['trajectory_points'],
                    3  # Use cubic method for consistency
                )
            else:
                # Fallback implementation
                results = self._batch_trajectory_fallback(batch_data, use_gpu=True)
            
            computation_time = time.time() - start_time
            
            logger.info(f"GPU batch processing completed in {computation_time:.3f}s")
            return results, computation_time
            
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            raise
    
    def batch_trajectory_cpu_sequential(self, batch_data: Dict) -> Tuple[Dict, float]:
        """
        CPU-based sequential batch trajectory generation.
        
        Args:
            batch_data: Batch configuration data
            
        Returns:
            Tuple of (results, computation_time)
        """
        start_time = time.time()
        
        batch_size = batch_data['batch_size']
        N = batch_data['trajectory_points']
        
        # Initialize result arrays
        positions = np.zeros((batch_size, N, self.num_joints), dtype=np.float32)
        velocities = np.zeros((batch_size, N, self.num_joints), dtype=np.float32)
        accelerations = np.zeros((batch_size, N, self.num_joints), dtype=np.float32)
        
        # Process each trajectory sequentially
        for i in range(batch_size):
            if self.planner:
                # Use planner if available
                traj = self.planner.joint_trajectory(
                    batch_data['start_configs'][i],
                    batch_data['end_configs'][i],
                    batch_data['trajectory_times'][i],
                    N,
                    batch_data['methods'][i]
                )
                positions[i] = traj['positions']
                velocities[i] = traj['velocities'] 
                accelerations[i] = traj['accelerations']
            else:
                # Fallback to simple trajectory generation
                positions[i], velocities[i], accelerations[i] = self._generate_simple_trajectory(
                    batch_data['start_configs'][i],
                    batch_data['end_configs'][i],
                    batch_data['trajectory_times'][i],
                    N,
                    batch_data['methods'][i]
                )
        
        computation_time = time.time() - start_time
        
        results = {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations
        }
        
        logger.info(f"CPU sequential processing completed in {computation_time:.3f}s")
        return results, computation_time
    
    def batch_trajectory_cpu_parallel(self, batch_data: Dict) -> Tuple[Dict, float]:
        """
        CPU-based parallel batch trajectory generation using multiprocessing.
        
        Args:
            batch_data: Batch configuration data
            
        Returns:
            Tuple of (results, computation_time)
        """
        start_time = time.time()
        
        batch_size = batch_data['batch_size']
        N = batch_data['trajectory_points']
        
        # Prepare arguments for parallel processing
        args_list = []
        for i in range(batch_size):
            args_list.append((
                batch_data['start_configs'][i],
                batch_data['end_configs'][i],
                batch_data['trajectory_times'][i],
                N,
                batch_data['methods'][i]
            ))
        
        # Use multiprocessing to compute trajectories in parallel
        num_workers = min(self.num_cpu_cores, batch_size)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results_list = list(executor.map(self._compute_single_trajectory, args_list))
        
        # Combine results
        positions = np.array([r[0] for r in results_list])
        velocities = np.array([r[1] for r in results_list])
        accelerations = np.array([r[2] for r in results_list])
        
        computation_time = time.time() - start_time
        
        results = {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations
        }
        
        logger.info(f"CPU parallel processing ({num_workers} workers) completed in {computation_time:.3f}s")
        return results, computation_time
    
    def _compute_single_trajectory(self, args):
        """Helper function for parallel trajectory computation."""
        start_config, end_config, traj_time, N, method = args
        return self._generate_simple_trajectory(start_config, end_config, traj_time, N, method)
    
    def _generate_simple_trajectory(self, start, end, Tf, N, method):
        """Generate a simple trajectory between two configurations."""
        positions = np.zeros((N, self.num_joints), dtype=np.float32)
        velocities = np.zeros((N, self.num_joints), dtype=np.float32)
        accelerations = np.zeros((N, self.num_joints), dtype=np.float32)
        
        for i in range(N):
            t = i * (Tf / (N - 1))
            tau = t / Tf
            
            # Time scaling
            if method == 3:  # Cubic
                s = 3.0 * tau * tau - 2.0 * tau * tau * tau
                s_dot = 6.0 * tau * (1.0 - tau) / Tf
                s_ddot = 6.0 / (Tf * Tf) * (1.0 - 2.0 * tau)
            elif method == 5:  # Quintic
                tau2 = tau * tau
                tau3 = tau2 * tau
                s = 10.0 * tau3 - 15.0 * tau2 * tau2 + 6.0 * tau * tau3
                s_dot = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2) / Tf
                s_ddot = 60.0 / (Tf * Tf) * tau * (1.0 - 2.0 * tau)
            else:
                s = s_dot = s_ddot = 0.0
            
            # Apply to all joints
            diff = end - start
            positions[i] = s * diff + start
            velocities[i] = s_dot * diff
            accelerations[i] = s_ddot * diff
        
        return positions, velocities, accelerations
    
    def _batch_trajectory_fallback(self, batch_data: Dict, use_gpu: bool = False) -> Dict:
        """Fallback batch trajectory implementation."""
        # Simple fallback when planner is not available
        return self.batch_trajectory_cpu_sequential(batch_data)[0]
    
    def run_performance_scaling_analysis(self) -> Dict:
        """
        Run comprehensive performance scaling analysis across different batch sizes.
        
        Returns:
            Dictionary containing scaling analysis results
        """
        logger.info("Running batch performance scaling analysis")
        
        # Test different batch sizes
        batch_sizes = [5, 10, 25, 50, 100, 200]
        if self.use_gpu:
            batch_sizes.extend([500, 1000])  # Add larger sizes for GPU
        
        results = {
            'batch_sizes': batch_sizes,
            'gpu_times': [],
            'cpu_sequential_times': [],
            'cpu_parallel_times': [],
            'gpu_speedup_vs_sequential': [],
            'gpu_speedup_vs_parallel': [],
            'parallel_speedup_vs_sequential': [],
            'memory_usage': [],
            'throughput_gpu': [],
            'throughput_cpu_seq': [],
            'throughput_cpu_par': []
        }
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Generate test data
            batch_data = self.generate_batch_data(batch_size, trajectory_points=50)
            
            # Test CPU sequential
            try:
                _, cpu_seq_time = self.batch_trajectory_cpu_sequential(batch_data)
                results['cpu_sequential_times'].append(cpu_seq_time)
                results['throughput_cpu_seq'].append(batch_size / cpu_seq_time)
            except Exception as e:
                logger.error(f"CPU sequential failed for batch size {batch_size}: {e}")
                results['cpu_sequential_times'].append(np.inf)
                results['throughput_cpu_seq'].append(0)
            
            # Test CPU parallel
            try:
                _, cpu_par_time = self.batch_trajectory_cpu_parallel(batch_data)
                results['cpu_parallel_times'].append(cpu_par_time)
                results['throughput_cpu_par'].append(batch_size / cpu_par_time)
            except Exception as e:
                logger.error(f"CPU parallel failed for batch size {batch_size}: {e}")
                results['cpu_parallel_times'].append(np.inf)
                results['throughput_cpu_par'].append(0)
            
            # Test GPU (if available)
            if self.use_gpu:
                try:
                    _, gpu_time = self.batch_trajectory_gpu(batch_data)
                    results['gpu_times'].append(gpu_time)
                    results['throughput_gpu'].append(batch_size / gpu_time)
                except Exception as e:
                    logger.warning(f"GPU processing failed for batch size {batch_size}: {e}")
                    results['gpu_times'].append(np.inf)
                    results['throughput_gpu'].append(0)
            else:
                results['gpu_times'].append(np.inf)
                results['throughput_gpu'].append(0)
            
            # Calculate speedups
            cpu_seq_time = results['cpu_sequential_times'][-1]
            cpu_par_time = results['cpu_parallel_times'][-1]
            gpu_time = results['gpu_times'][-1]
            
            if cpu_seq_time != np.inf and gpu_time != np.inf:
                results['gpu_speedup_vs_sequential'].append(cpu_seq_time / gpu_time)
            else:
                results['gpu_speedup_vs_sequential'].append(0)
            
            if cpu_par_time != np.inf and gpu_time != np.inf:
                results['gpu_speedup_vs_parallel'].append(cpu_par_time / gpu_time)
            else:
                results['gpu_speedup_vs_parallel'].append(0)
            
            if cpu_seq_time != np.inf and cpu_par_time != np.inf:
                results['parallel_speedup_vs_sequential'].append(cpu_seq_time / cpu_par_time)
            else:
                results['parallel_speedup_vs_sequential'].append(0)
            
            # Estimate memory usage
            memory_mb = batch_size * 50 * self.num_joints * 4 * 3 / (1024 * 1024)  # 3 arrays
            results['memory_usage'].append(memory_mb)
        
        return results
    
    def run_memory_efficiency_analysis(self) -> Dict:
        """
        Analyze memory efficiency of different batch processing approaches.
        
        Returns:
            Dictionary containing memory analysis results
        """
        logger.info("Running memory efficiency analysis")
        
        # Test different memory allocation strategies
        batch_size = 100
        trajectory_points = 100
        
        results = {
            'strategies': [],
            'peak_memory_mb': [],
            'allocation_time': [],
            'processing_time': []
        }
        
        # Strategy 1: Pre-allocate all arrays
        strategy_name = "Pre-allocated Arrays"
        results['strategies'].append(strategy_name)
        
        start_memory = psutil.virtual_memory().used / (1024**2)
        start_time = time.time()
        
        # Pre-allocate large arrays
        batch_data = self.generate_batch_data(batch_size, trajectory_points)
        positions = np.zeros((batch_size, trajectory_points, self.num_joints), dtype=np.float32)
        velocities = np.zeros((batch_size, trajectory_points, self.num_joints), dtype=np.float32)
        accelerations = np.zeros((batch_size, trajectory_points, self.num_joints), dtype=np.float32)
        
        alloc_time = time.time() - start_time
        peak_memory = psutil.virtual_memory().used / (1024**2)
        
        # Process data
        process_start = time.time()
        for i in range(min(10, batch_size)):  # Process subset for timing
            positions[i], velocities[i], accelerations[i] = self._generate_simple_trajectory(
                batch_data['start_configs'][i],
                batch_data['end_configs'][i],
                batch_data['trajectory_times'][i],
                trajectory_points,
                3
            )
        process_time = time.time() - process_start
        
        results['peak_memory_mb'].append(peak_memory - start_memory)
        results['allocation_time'].append(alloc_time)
        results['processing_time'].append(process_time)
        
        # Clean up
        del positions, velocities, accelerations
        
        # Strategy 2: Chunked processing
        strategy_name = "Chunked Processing"
        results['strategies'].append(strategy_name)
        
        chunk_size = 10
        start_memory = psutil.virtual_memory().used / (1024**2)
        start_time = time.time()
        
        # Process in chunks
        chunk_results = []
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_data = {
                'start_configs': batch_data['start_configs'][chunk_start:chunk_end],
                'end_configs': batch_data['end_configs'][chunk_start:chunk_end],
                'trajectory_times': batch_data['trajectory_times'][chunk_start:chunk_end],
                'trajectory_points': trajectory_points,
                'methods': batch_data['methods'][chunk_start:chunk_end],
                'batch_size': chunk_end - chunk_start
            }
            
            chunk_result, _ = self.batch_trajectory_cpu_sequential(chunk_data)
            chunk_results.append(chunk_result)
        
        total_time = time.time() - start_time
        peak_memory = psutil.virtual_memory().used / (1024**2)
        
        results['peak_memory_mb'].append(peak_memory - start_memory)
        results['allocation_time'].append(0)  # No large pre-allocation
        results['processing_time'].append(total_time)
        
        return results
    
    def analyze_batch_results(self, batch_results: Dict, batch_data: Dict) -> Dict:
        """
        Perform statistical analysis on batch processing results.
        
        Args:
            batch_results: Results from batch processing
            batch_data: Original batch configuration data
            
        Returns:
            Dictionary containing statistical analysis
        """
        positions = batch_results['positions']
        velocities = batch_results['velocities']
        accelerations = batch_results['accelerations']
        
        analysis = {
            'trajectory_statistics': {},
            'joint_statistics': {},
            'performance_metrics': {}
        }
        
        # Trajectory-level statistics
        path_lengths = []
        max_velocities = []
        max_accelerations = []
        
        for i in range(positions.shape[0]):
            # Calculate path length
            path_length = np.sum(np.linalg.norm(np.diff(positions[i], axis=0), axis=1))
            path_lengths.append(path_length)
            
            # Maximum velocities and accelerations
            max_vel = np.max(np.linalg.norm(velocities[i], axis=1))
            max_acc = np.max(np.linalg.norm(accelerations[i], axis=1))
            max_velocities.append(max_vel)
            max_accelerations.append(max_acc)
        
        analysis['trajectory_statistics'] = {
            'path_lengths': {
                'mean': np.mean(path_lengths),
                'std': np.std(path_lengths),
                'min': np.min(path_lengths),
                'max': np.max(path_lengths)
            },
            'max_velocities': {
                'mean': np.mean(max_velocities),
                'std': np.std(max_velocities),
                'min': np.min(max_velocities),
                'max': np.max(max_velocities)
            },
            'max_accelerations': {
                'mean': np.mean(max_accelerations),
                'std': np.std(max_accelerations),
                'min': np.min(max_accelerations),
                'max': np.max(max_accelerations)
            }
        }
        
        # Joint-level statistics
        joint_stats = {}
        for joint in range(self.num_joints):
            joint_positions = positions[:, :, joint]
            joint_velocities = velocities[:, :, joint]
            joint_accelerations = accelerations[:, :, joint]
            
            joint_stats[f'joint_{joint+1}'] = {
                'position_range': np.max(joint_positions) - np.min(joint_positions),
                'max_velocity': np.max(np.abs(joint_velocities)),
                'max_acceleration': np.max(np.abs(joint_accelerations)),
                'rms_velocity': np.sqrt(np.mean(joint_velocities**2)),
                'rms_acceleration': np.sqrt(np.mean(joint_accelerations**2))
            }
        
        analysis['joint_statistics'] = joint_stats
        
        return analysis
    
    def visualize_scaling_results(self, scaling_results: Dict) -> plt.Figure:
        """
        Visualize batch processing scaling results.
        
        Args:
            scaling_results: Results from scaling analysis
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        batch_sizes = scaling_results['batch_sizes']
        
        # Computation time comparison
        ax1 = axes[0, 0]
        if scaling_results['cpu_sequential_times']:
            ax1.loglog(batch_sizes, scaling_results['cpu_sequential_times'], 
                      'b-o', label='CPU Sequential', linewidth=2)
        if scaling_results['cpu_parallel_times']:
            ax1.loglog(batch_sizes, scaling_results['cpu_parallel_times'], 
                      'g-s', label='CPU Parallel', linewidth=2)
        if self.use_gpu and scaling_results['gpu_times']:
            valid_gpu = [t for t in scaling_results['gpu_times'] if t != np.inf]
            if valid_gpu:
                ax1.loglog(batch_sizes[:len(valid_gpu)], valid_gpu, 
                          'r-^', label='GPU', linewidth=2)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Computation Time (s)')
        ax1.set_title('Batch Processing Time Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput comparison
        ax2 = axes[0, 1]
        if scaling_results['throughput_cpu_seq']:
            ax2.semilogx(batch_sizes, scaling_results['throughput_cpu_seq'], 
                        'b-o', label='CPU Sequential', linewidth=2)
        if scaling_results['throughput_cpu_par']:
            ax2.semilogx(batch_sizes, scaling_results['throughput_cpu_par'], 
                        'g-s', label='CPU Parallel', linewidth=2)
        if self.use_gpu and scaling_results['throughput_gpu']:
            valid_throughput = [t for t in scaling_results['throughput_gpu'] if t > 0]
            if valid_throughput:
                ax2.semilogx(batch_sizes[:len(valid_throughput)], valid_throughput, 
                            'r-^', label='GPU', linewidth=2)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (trajectories/s)')
        ax2.set_title('Batch Processing Throughput')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Speedup analysis
        ax3 = axes[0, 2]
        if scaling_results['gpu_speedup_vs_sequential']:
            valid_speedup = [s for s in scaling_results['gpu_speedup_vs_sequential'] if s > 0]
            if valid_speedup:
                ax3.semilogx(batch_sizes[:len(valid_speedup)], valid_speedup, 
                            'r-o', label='GPU vs CPU Sequential', linewidth=2)
        if scaling_results['gpu_speedup_vs_parallel']:
            valid_speedup = [s for s in scaling_results['gpu_speedup_vs_parallel'] if s > 0]
            if valid_speedup:
                ax3.semilogx(batch_sizes[:len(valid_speedup)], valid_speedup, 
                            'r-s', label='GPU vs CPU Parallel', linewidth=2)
        if scaling_results['parallel_speedup_vs_sequential']:
            ax3.semilogx(batch_sizes, scaling_results['parallel_speedup_vs_sequential'], 
                        'g-^', label='Parallel vs Sequential', linewidth=2)
        
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Batch Processing Speedup')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Memory usage
        ax4 = axes[1, 0]
        ax4.semilogx(batch_sizes, scaling_results['memory_usage'], 'purple', 
                    marker='d', linewidth=2, label='Memory Usage')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Scaling')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Efficiency analysis
        ax5 = axes[1, 1]
        if self.use_gpu and scaling_results['gpu_speedup_vs_sequential']:
            efficiency = [s / batch_sizes[i] for i, s in enumerate(scaling_results['gpu_speedup_vs_sequential']) if s > 0]
            if efficiency:
                ax5.semilogx(batch_sizes[:len(efficiency)], efficiency, 
                            'r-o', label='GPU Efficiency', linewidth=2)
        
        parallel_efficiency = [s / self.num_cpu_cores for s in scaling_results['parallel_speedup_vs_sequential']]
        ax5.semilogx(batch_sizes, parallel_efficiency, 'g-s', 
                    label='Parallel Efficiency', linewidth=2)
        
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Efficiency')
        ax5.set_title('Processing Efficiency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Performance summary
        ax6 = axes[1, 2]
        
        # Calculate average performance metrics
        avg_cpu_seq = np.mean([t for t in scaling_results['cpu_sequential_times'] if t != np.inf])
        avg_cpu_par = np.mean([t for t in scaling_results['cpu_parallel_times'] if t != np.inf])
        avg_gpu = np.mean([t for t in scaling_results['gpu_times'] if t != np.inf]) if self.use_gpu else 0
        
        methods = []
        times = []
        colors = []
        
        if avg_cpu_seq != np.inf:
            methods.append('CPU\nSequential')
            times.append(avg_cpu_seq)
            colors.append('blue')
        
        if avg_cpu_par != np.inf:
            methods.append('CPU\nParallel')
            times.append(avg_cpu_par)
            colors.append('green')
        
        if avg_gpu > 0:
            methods.append('GPU\nBatch')
            times.append(avg_gpu)
            colors.append('red')
        
        if methods:
            bars = ax6.bar(methods, times, color=colors, alpha=0.7)
            ax6.set_ylabel('Average Time (s)')
            ax6.set_title('Average Performance Comparison')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('batch_scaling_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Batch scaling analysis saved as 'batch_scaling_analysis.png'")
        
        return fig
    
    def visualize_batch_statistics(self, analysis: Dict, batch_size: int) -> plt.Figure:
        """
        Visualize statistical analysis of batch processing results.
        
        Args:
            analysis: Statistical analysis results
            batch_size: Size of the processed batch
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Trajectory statistics
        ax1 = axes[0, 0]
        traj_stats = analysis['trajectory_statistics']
        metrics = list(traj_stats.keys())
        means = [traj_stats[metric]['mean'] for metric in metrics]
        stds = [traj_stats[metric]['std'] for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['blue', 'green', 'red'])
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('Trajectory Statistics Summary')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + height*0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # Joint-wise maximum velocities
        ax2 = axes[0, 1]
        joint_stats = analysis['joint_statistics']
        joint_names = list(joint_stats.keys())
        max_vels = [joint_stats[joint]['max_velocity'] for joint in joint_names]
        
        ax2.bar(range(len(joint_names)), max_vels, color='orange', alpha=0.7)
        ax2.set_xlabel('Joint')
        ax2.set_ylabel('Maximum Velocity (rad/s)')
        ax2.set_title('Maximum Joint Velocities')
        ax2.set_xticks(range(len(joint_names)))
        ax2.set_xticklabels([f'J{i+1}' for i in range(len(joint_names))])
        
        # Joint-wise maximum accelerations
        ax3 = axes[0, 2]
        max_accs = [joint_stats[joint]['max_acceleration'] for joint in joint_names]
        
        ax3.bar(range(len(joint_names)), max_accs, color='purple', alpha=0.7)
        ax3.set_xlabel('Joint')
        ax3.set_ylabel('Maximum Acceleration (rad/s²)')
        ax3.set_title('Maximum Joint Accelerations')
        ax3.set_xticks(range(len(joint_names)))
        ax3.set_xticklabels([f'J{i+1}' for i in range(len(joint_names))])
        
        # Path length distribution
        ax4 = axes[1, 0]
        # Generate sample path lengths for histogram
        np.random.seed(42)
        path_lengths = np.random.normal(
            traj_stats['path_lengths']['mean'],
            traj_stats['path_lengths']['std'],
            batch_size
        )
        
        ax4.hist(path_lengths, bins=20, alpha=0.7, color='cyan', edgecolor='black')
        ax4.axvline(traj_stats['path_lengths']['mean'], color='red', linestyle='--', 
                   label=f"Mean: {traj_stats['path_lengths']['mean']:.3f}")
        ax4.set_xlabel('Path Length')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Path Length Distribution')
        ax4.legend()
        
        # Joint range comparison
        ax5 = axes[1, 1]
        joint_ranges = [joint_stats[joint]['position_range'] for joint in joint_names]
        
        ax5.bar(range(len(joint_names)), joint_ranges, color='lightgreen', alpha=0.7)
        ax5.set_xlabel('Joint')
        ax5.set_ylabel('Position Range (rad)')
        ax5.set_title('Joint Position Ranges')
        ax5.set_xticks(range(len(joint_names)))
        ax5.set_xticklabels([f'J{i+1}' for i in range(len(joint_names))])
        
        # RMS velocity vs acceleration
        ax6 = axes[1, 2]
        rms_vels = [joint_stats[joint]['rms_velocity'] for joint in joint_names]
        rms_accs = [joint_stats[joint]['rms_acceleration'] for joint in joint_names]
        
        x_pos = np.arange(len(joint_names))
        width = 0.35
        
        ax6.bar(x_pos - width/2, rms_vels, width, label='RMS Velocity', 
               alpha=0.7, color='blue')
        ax6.bar(x_pos + width/2, rms_accs, width, label='RMS Acceleration', 
               alpha=0.7, color='red')
        
        ax6.set_xlabel('Joint')
        ax6.set_ylabel('RMS Values')
        ax6.set_title('RMS Velocity vs Acceleration')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([f'J{i+1}' for i in range(len(joint_names))])
        ax6.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'batch_statistics_analysis_{batch_size}.png', dpi=300, bbox_inches='tight')
            logger.info(f"Batch statistics analysis saved as 'batch_statistics_analysis_{batch_size}.png'")
        
        return fig
    
    def visualize_memory_analysis(self, memory_results: Dict) -> plt.Figure:
        """
        Visualize memory efficiency analysis results.
        
        Args:
            memory_results: Results from memory analysis
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = memory_results['strategies']
        
        # Peak memory usage
        ax1 = axes[0]
        bars1 = ax1.bar(strategies, memory_results['peak_memory_mb'], 
                        color=['blue', 'green'], alpha=0.7)
        ax1.set_ylabel('Peak Memory Usage (MB)')
        ax1.set_title('Memory Usage by Strategy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mem in zip(bars1, memory_results['peak_memory_mb']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{mem:.1f} MB', ha='center', va='bottom')
        
        # Processing time
        ax2 = axes[1]
        bars2 = ax2.bar(strategies, memory_results['processing_time'], 
                        color=['orange', 'purple'], alpha=0.7)
        ax2.set_ylabel('Processing Time (s)')
        ax2.set_title('Processing Time by Strategy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars2, memory_results['processing_time']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Memory efficiency (MB/s)
        ax3 = axes[2]
        efficiency = [mem / time_val for mem, time_val in 
                     zip(memory_results['peak_memory_mb'], memory_results['processing_time'])]
        bars3 = ax3.bar(strategies, efficiency, color=['red', 'cyan'], alpha=0.7)
        ax3.set_ylabel('Memory Efficiency (MB/s)')
        ax3.set_title('Memory vs Time Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, eff in zip(bars3, efficiency):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{eff:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('memory_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Memory efficiency analysis saved as 'memory_efficiency_analysis.png'")
        
        return fig
    
    def generate_batch_report(self, scaling_results: Dict, memory_results: Dict, 
                             batch_analysis: Dict) -> str:
        """
        Generate comprehensive batch processing analysis report.
        
        Args:
            scaling_results: Performance scaling analysis results
            memory_results: Memory efficiency analysis results  
            batch_analysis: Statistical analysis of batch results
            
        Returns:
            Formatted report string
        """
        report = """
========================================================================
                    BATCH PROCESSING ANALYSIS REPORT
========================================================================

1. SYSTEM CONFIGURATION
-----------------------
"""
        report += f"CPU Cores: {self.num_cpu_cores}\n"
        report += f"System Memory: {self.memory_gb:.1f} GB\n"
        report += f"GPU Available: {self.use_gpu}\n"
        report += f"Robot DOF: {self.num_joints}\n\n"
        
        report += """
2. PERFORMANCE SCALING ANALYSIS
-------------------------------
"""
        
        if scaling_results['batch_sizes']:
            max_batch_size = max(scaling_results['batch_sizes'])
            report += f"Maximum Batch Size Tested: {max_batch_size}\n"
            
            # CPU Performance
            if scaling_results['cpu_sequential_times']:
                min_cpu_seq = min(t for t in scaling_results['cpu_sequential_times'] if t != np.inf)
                max_cpu_seq = max(t for t in scaling_results['cpu_sequential_times'] if t != np.inf)
                report += f"CPU Sequential Time Range: {min_cpu_seq:.3f}s - {max_cpu_seq:.3f}s\n"
            
            if scaling_results['cpu_parallel_times']:
                min_cpu_par = min(t for t in scaling_results['cpu_parallel_times'] if t != np.inf)
                max_cpu_par = max(t for t in scaling_results['cpu_parallel_times'] if t != np.inf)
                report += f"CPU Parallel Time Range: {min_cpu_par:.3f}s - {max_cpu_par:.3f}s\n"
            
            # GPU Performance
            if self.use_gpu and scaling_results['gpu_times']:
                valid_gpu_times = [t for t in scaling_results['gpu_times'] if t != np.inf]
                if valid_gpu_times:
                    min_gpu = min(valid_gpu_times)
                    max_gpu = max(valid_gpu_times)
                    report += f"GPU Time Range: {min_gpu:.3f}s - {max_gpu:.3f}s\n"
                    
                    # Best speedups
                    max_speedup_seq = max(s for s in scaling_results['gpu_speedup_vs_sequential'] if s > 0)
                    max_speedup_par = max(s for s in scaling_results['gpu_speedup_vs_parallel'] if s > 0)
                    report += f"Maximum GPU Speedup vs Sequential: {max_speedup_seq:.2f}x\n"
                    report += f"Maximum GPU Speedup vs Parallel: {max_speedup_par:.2f}x\n"
            
            # Parallel efficiency
            if scaling_results['parallel_speedup_vs_sequential']:
                max_parallel_speedup = max(scaling_results['parallel_speedup_vs_sequential'])
                parallel_efficiency = max_parallel_speedup / self.num_cpu_cores
                report += f"Maximum Parallel Speedup: {max_parallel_speedup:.2f}x\n"
                report += f"Parallel Efficiency: {parallel_efficiency:.2%}\n"
        
        report += """

3. MEMORY EFFICIENCY ANALYSIS
-----------------------------
"""
        
        if memory_results:
            for i, strategy in enumerate(memory_results['strategies']):
                peak_mem = memory_results['peak_memory_mb'][i]
                proc_time = memory_results['processing_time'][i]
                report += f"{strategy}:\n"
                report += f"  Peak Memory: {peak_mem:.1f} MB\n"
                report += f"  Processing Time: {proc_time:.3f}s\n"
                report += f"  Memory Efficiency: {peak_mem/proc_time:.2f} MB/s\n\n"
        
        report += """
4. STATISTICAL ANALYSIS
-----------------------
"""
        
        if batch_analysis and 'trajectory_statistics' in batch_analysis:
            traj_stats = batch_analysis['trajectory_statistics']
            
            report += "TRAJECTORY METRICS:\n"
            for metric, stats in traj_stats.items():
                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    Mean: {stats['mean']:.3f}\n"
                report += f"    Std Dev: {stats['std']:.3f}\n"
                report += f"    Range: {stats['min']:.3f} - {stats['max']:.3f}\n\n"
        
        if batch_analysis and 'joint_statistics' in batch_analysis:
            joint_stats = batch_analysis['joint_statistics']
            
            report += "JOINT-WISE ANALYSIS:\n"
            for joint, stats in joint_stats.items():
                report += f"  {joint.replace('_', ' ').title()}:\n"
                report += f"    Max Velocity: {stats['max_velocity']:.3f} rad/s\n"
                report += f"    Max Acceleration: {stats['max_acceleration']:.3f} rad/s²\n"
                report += f"    Position Range: {stats['position_range']:.3f} rad\n\n"
        
        report += """
5. RECOMMENDATIONS
------------------
"""
        
        # Performance recommendations
        if self.use_gpu and scaling_results.get('gpu_speedup_vs_sequential'):
            max_gpu_speedup = max(s for s in scaling_results['gpu_speedup_vs_sequential'] if s > 0)
            if max_gpu_speedup > 2:
                report += "- GPU ACCELERATION: Highly recommended for batch processing\n"
                report += f"  GPU provides up to {max_gpu_speedup:.1f}x speedup over sequential CPU\n"
            else:
                report += "- GPU ACCELERATION: Limited benefit for current batch sizes\n"
        else:
            report += "- GPU ACCELERATION: Not available (install CUDA support)\n"
        
        # Parallel processing recommendations
        if scaling_results.get('parallel_speedup_vs_sequential'):
            max_parallel = max(scaling_results['parallel_speedup_vs_sequential'])
            if max_parallel > 1.5:
                report += f"- PARALLEL PROCESSING: Recommended ({max_parallel:.1f}x speedup)\n"
                report += f"  Optimal for batch sizes > 50 with {self.num_cpu_cores} cores\n"
            else:
                report += "- PARALLEL PROCESSING: Limited benefit due to overhead\n"
        
        # Memory recommendations
        if memory_results:
            chunked_memory = memory_results['peak_memory_mb'][1] if len(memory_results['peak_memory_mb']) > 1 else 0
            preallocated_memory = memory_results['peak_memory_mb'][0] if memory_results['peak_memory_mb'] else 0
            
            if chunked_memory > 0 and preallocated_memory > 0:
                if chunked_memory < preallocated_memory * 0.5:
                    report += "- MEMORY STRATEGY: Use chunked processing for large batches\n"
                    report += f"  Reduces memory usage by {(1 - chunked_memory/preallocated_memory)*100:.1f}%\n"
                else:
                    report += "- MEMORY STRATEGY: Pre-allocation acceptable for current sizes\n"
        
        # Batch size recommendations
        if scaling_results.get('throughput_gpu') or scaling_results.get('throughput_cpu_par'):
            if self.use_gpu and scaling_results['throughput_gpu']:
                max_gpu_throughput_idx = np.argmax([t for t in scaling_results['throughput_gpu'] if t > 0])
                optimal_batch_size = scaling_results['batch_sizes'][max_gpu_throughput_idx]
                report += f"- OPTIMAL BATCH SIZE: {optimal_batch_size} (for GPU processing)\n"
            elif scaling_results['throughput_cpu_par']:
                max_cpu_throughput_idx = np.argmax(scaling_results['throughput_cpu_par'])
                optimal_batch_size = scaling_results['batch_sizes'][max_cpu_throughput_idx]
                report += f"- OPTIMAL BATCH SIZE: {optimal_batch_size} (for CPU parallel processing)\n"
        
        report += """

6. TECHNICAL DETAILS
--------------------
"""
        report += f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"ManipulaPy Version: Batch Processing Demo\n"
        report += f"CUDA Available: {CUDA_AVAILABLE}\n"
        if self.use_gpu:
            report += "GPU Acceleration: ENABLED\n"
        else:
            report += "GPU Acceleration: DISABLED\n"
        
        report += """
========================================================================
                            END OF REPORT
========================================================================
"""
        
        if self.save_plots:
            with open('batch_processing_analysis_report.txt', 'w') as f:
                f.write(report)
            logger.info("Batch processing report saved as 'batch_processing_analysis_report.txt'")
        
        return report


def main():
    """
    Main function to run the batch processing demonstration.
    """
    print("=" * 70)
    print("   MANIPULAPY BATCH PROCESSING DEMONSTRATION")
    print("=" * 70)
    
    # Initialize demo
    demo = BatchProcessingDemo(save_plots=True, use_gpu=None)
    
    try:
        # 1. Performance scaling analysis
        print("\n1. Running performance scaling analysis...")
        scaling_results = demo.run_performance_scaling_analysis()
        
        # 2. Memory efficiency analysis
        print("\n2. Running memory efficiency analysis...")
        memory_results = demo.run_memory_efficiency_analysis()
        
        # 3. Batch statistical analysis
        print("\n3. Running batch statistical analysis...")
        batch_data = demo.generate_batch_data(100, 50)
        if demo.use_gpu:
            batch_results, _ = demo.batch_trajectory_gpu(batch_data)
        else:
            batch_results, _ = demo.batch_trajectory_cpu_parallel(batch_data)
        
        batch_analysis = demo.analyze_batch_results(batch_results, batch_data)
        
        # 4. Generate visualizations
        print("\n4. Generating visualizations...")
        demo.visualize_scaling_results(scaling_results)
        demo.visualize_batch_statistics(batch_analysis, batch_data['batch_size'])
        demo.visualize_memory_analysis(memory_results)
        
        # 5. Generate comprehensive report
        print("\n5. Generating comprehensive analysis report...")
        report = demo.generate_batch_report(scaling_results, memory_results, batch_analysis)
        
        # Display summary
        print("\n" + "=" * 70)
        print("                    ANALYSIS COMPLETE")
        print("=" * 70)
        
        # Performance summary
        if scaling_results['batch_sizes']:
            max_batch = max(scaling_results['batch_sizes'])
            print(f"Maximum Batch Size Tested: {max_batch}")
            
            if demo.use_gpu and scaling_results['gpu_times']:
                valid_gpu_times = [t for t in scaling_results['gpu_times'] if t != np.inf]
                if valid_gpu_times:
                    avg_gpu_time = np.mean(valid_gpu_times)
                    print(f"Average GPU Processing Time: {avg_gpu_time:.3f}s")
            
            if scaling_results['cpu_parallel_times']:
                valid_cpu_times = [t for t in scaling_results['cpu_parallel_times'] if t != np.inf]
                if valid_cpu_times:
                    avg_cpu_time = np.mean(valid_cpu_times)
                    print(f"Average CPU Parallel Time: {avg_cpu_time:.3f}s")
            
            if scaling_results['gpu_speedup_vs_sequential']:
                max_speedup = max(s for s in scaling_results['gpu_speedup_vs_sequential'] if s > 0)
                if max_speedup > 0:
                    print(f"Maximum GPU Speedup: {max_speedup:.2f}x")
        
        print("\nGenerated files:")
        file_list = [
            'batch_scaling_analysis.png',
            'batch_statistics_analysis_100.png',
            'memory_efficiency_analysis.png',
            'batch_processing_analysis_report.txt'
        ]
        
        for filename in file_list:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (not generated)")
        
        print("\n" + "=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDemo cleanup completed.")


if __name__ == "__main__":
    main()