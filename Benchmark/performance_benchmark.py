#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Performance Benchmark Suite for ManipulaPy Path Planning

This comprehensive benchmark compares CPU vs GPU performance for various path planning
tasks including joint trajectory generation, inverse dynamics, forward dynamics,
Cartesian trajectory planning, and batch processing.

Features:
- Systematic benchmarking across different problem sizes
- Statistical analysis with confidence intervals
- Performance profiling and memory usage tracking
- Visual performance comparison charts
- Scalability analysis
- Hardware capability assessment

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import gc
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import matplotlib
matplotlib.use("Agg")
# ManipulaPy imports
try:
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning, compare_implementations
    from ManipulaPy.cuda_kernels import (
        CUDA_AVAILABLE, check_cuda_availability, get_gpu_properties,
        benchmark_kernel_performance, profile_start, profile_stop
    )
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy import utils
except ImportError as e:
    print(f"Error importing ManipulaPy: {e}")
    print("Please ensure ManipulaPy is properly installed")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters"""
    test_name: str
    trajectory_points: List[int]
    joint_counts: List[int]
    batch_sizes: List[int]
    num_runs: int
    timeout_seconds: float
    enable_profiling: bool
    save_results: bool
    output_dir: str

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_name: str
    implementation: str  # 'cpu' or 'gpu'
    n_points: int
    n_joints: int
    batch_size: int
    execution_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None

class PathPlanningBenchmark:
    """
    Comprehensive benchmark suite for ManipulaPy path planning performance.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark suite.
        
        Args:
            config: Benchmark configuration parameters
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
        
        # Create output directory in the same directory as the script
        script_dir = Path(__file__).parent if hasattr(Path(__file__), 'parent') else Path.cwd()
        self.output_path = script_dir / config.output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized data storage
        self.plots_dir = self.output_path / "plots"
        self.data_dir = self.output_path / "data" 
        self.reports_dir = self.output_path / "reports"
        self.logs_dir = self.output_path / "logs"
        
        for directory in [self.plots_dir, self.data_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Setup file logging in addition to console logging
        log_file = self.logs_dir / f"benchmark_{config.test_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Initialized benchmark suite - CUDA Available: {CUDA_AVAILABLE}")
        logger.info(f"Output directory: {self.output_path}")
        logger.info(f"Created subdirectories: plots, data, reports, logs")
        
        if CUDA_AVAILABLE:
            gpu_props = get_gpu_properties()
            if gpu_props:
                logger.info(f"GPU: {gpu_props.get('multiprocessor_count', 'Unknown')} SMs")
        
        # Save system info immediately
        self._save_system_info()
    
    def _save_system_info(self) -> None:
        """Save detailed system information to file"""
        system_info_extended = self.system_info.copy()
        
        # Add more detailed system information
        import platform
        import sys
        
        system_info_extended.update({
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmark_config': {
                'test_name': self.config.test_name,
                'trajectory_points': self.config.trajectory_points,
                'joint_counts': self.config.joint_counts,
                'batch_sizes': self.config.batch_sizes,
                'num_runs': self.config.num_runs,
                'timeout_seconds': self.config.timeout_seconds,
                'enable_profiling': self.config.enable_profiling
            }
        })
        
        # Try to get more GPU info if available
        if CUDA_AVAILABLE:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                system_info_extended.update({
                    'gpu_name': gpu_name,
                    'gpu_memory_total_gb': memory_info.total / (1024**3),
                    'gpu_memory_free_gb': memory_info.free / (1024**3),
                })
            except:
                pass  # pynvml not available or other GPU info gathering failed
        
        # Save to JSON file
        system_info_file = self.data_dir / 'system_info.json'
        with open(system_info_file, 'w') as f:
            json.dump(system_info_extended, f, indent=2, default=str)
        
        logger.info(f"System information saved to {system_info_file}")
    
    def _get_system_info(self) -> Dict:
        """Collect system information for benchmark context"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': CUDA_AVAILABLE,
        }
        
        if CUDA_AVAILABLE:
            gpu_props = get_gpu_properties()
            if gpu_props:
                info.update({
                    'gpu_multiprocessors': gpu_props.get('multiprocessor_count', 0),
                    'gpu_max_threads_per_block': gpu_props.get('max_threads_per_block', 0),
                    'gpu_max_shared_memory': gpu_props.get('max_shared_memory_per_block', 0),
                })
        
        return info
    
    def _create_test_robot(self, n_joints: int) -> Tuple[SerialManipulator, ManipulatorDynamics, List]:
        """
        Create a synthetic robot model for testing.
        
        Args:
            n_joints: Number of joints for the robot
            
        Returns:
            Tuple of (SerialManipulator, ManipulatorDynamics, joint_limits)
        """
        # Create synthetic robot parameters
        M_list = np.eye(4)
        M_list[:3, 3] = [0, 0, 1]  # End-effector at (0,0,1)
        
        # Create screw axes (alternating revolute joints)
        S_list = np.zeros((6, n_joints))
        for i in range(n_joints):
            if i % 2 == 0:  # Z-axis rotation
                S_list[2, i] = 1.0  # omega_z = 1
                S_list[4, i] = 0.1 * i  # v_y related to position
            else:  # Y-axis rotation
                S_list[1, i] = 1.0  # omega_y = 1
                S_list[5, i] = 0.1 * i  # v_z related to position
        
        # Extract components
        omega_list = S_list[:3, :]
        r_list = utils.extract_r_list(S_list)
        B_list = np.random.randn(6, n_joints) * 0.1  # Body frame screws
        
        # Create inertia matrices
        G_list = []
        for i in range(n_joints):
            G = np.eye(6) * (1.0 + 0.1 * i)  # Varying inertias
            G_list.append(G)
        
        # Joint limits
        joint_limits = [(-np.pi, np.pi) for _ in range(n_joints)]
        
        # Create manipulator
        robot = SerialManipulator(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            S_list=S_list,
            B_list=B_list,
            G_list=G_list,
            joint_limits=joint_limits
        )
        
        # Create dynamics
        dynamics = ManipulatorDynamics(
            M_list=M_list,
            omega_list=omega_list,
            r_list=r_list,
            b_list=None,
            S_list=S_list,
            B_list=B_list,
            Glist=G_list
        )
        
        return robot, dynamics, joint_limits
    
    def _measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, bool, Optional[str]]:
        """
        Measure execution time of a function with timeout protection.
        
        Returns:
            Tuple of (execution_time, success, error_message)
        """
        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            if execution_time > self.config.timeout_seconds:
                return execution_time, False, f"Timeout exceeded ({execution_time:.2f}s)"
            
            return execution_time, True, None
            
        except Exception as e:
            return 0.0, False, str(e)
    
    def _create_safe_planner(
        self, 
        robot, 
        dynamics, 
        joint_limits, 
        use_cuda: bool
    ) -> OptimizedTrajectoryPlanning:
        """
        Create a planner instance with safe parameters and error handling.

        Args:
            robot: SerialManipulator instance
            dynamics: ManipulatorDynamics instance
            joint_limits: Joint limits
            use_cuda: Whether to use CUDA acceleration

        Returns:
            OptimizedTrajectoryPlanning instance
        """
        try:
            cuda_threshold = 0 if use_cuda else 999999  # Large number instead of infinity

            # Bypass __init__, build object manually
            planner = OptimizedTrajectoryPlanning.__new__(OptimizedTrajectoryPlanning)

            # Core attributes
            planner.serial_manipulator = robot
            planner.dynamics = dynamics
            planner.joint_limits = np.asarray(joint_limits, dtype=np.float32)
            planner.torque_limits = np.array(
                [[-np.inf, np.inf]] * len(joint_limits),
                dtype=np.float32
            )

            # CUDA settings
            detected_cuda = check_cuda_availability()
            if use_cuda and not detected_cuda:
                raise RuntimeError("use_cuda=True requested but CUDA is not available.")
            planner.cuda_available = use_cuda and detected_cuda
            planner.gpu_properties = (
                get_gpu_properties() if planner.cuda_available else None
            )
            planner.cpu_threshold = cuda_threshold

            # Disable collision checking
            planner.collision_checker = None
            planner.potential_field = None

            # Performance bookkeeping
            planner._gpu_arrays = {}
            planner.enable_profiling = self.config.enable_profiling
            planner.performance_stats = {
                "gpu_calls": 0,
                "cpu_calls": 0,
                "total_gpu_time": 0.0,
                "total_cpu_time": 0.0,
                "memory_transfers": 0,
                "kernel_launches": 0,
            }

            # Start profiling if desired
            if planner.enable_profiling and planner.cuda_available:
                profile_start()

            return planner

        except Exception as e:
            logger.error(f"Failed to create safe planner: {e}")
            raise
       
    def _measure_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    
    def benchmark_joint_trajectory(self) -> None:
        """Benchmark joint trajectory generation"""
        logger.info("Benchmarking joint trajectory generation...")
        
        for n_joints in self.config.joint_counts:
            for n_points in self.config.trajectory_points:
                logger.info(f"Testing {n_joints} joints, {n_points} points")
                
                # Create test robot
                robot, dynamics, joint_limits = self._create_test_robot(n_joints)
                
                # Generate test data
                thetastart = np.random.uniform(-1, 1, n_joints).astype(np.float32)
                thetaend = np.random.uniform(-1, 1, n_joints).astype(np.float32)
                Tf = 2.0
                method = 5  # Cubic
                
                # Test both implementations
                implementations = ['cpu', 'gpu'] if CUDA_AVAILABLE else ['cpu']
                
                for impl in implementations:
                    # Create planner with specific implementation
                    use_cuda = (impl == 'gpu')
                    planner = self._create_safe_planner(robot, dynamics, joint_limits, use_cuda)
                    
                    # Run multiple tests for statistical accuracy
                    times = []
                    successes = []
                    
                    for run in range(self.config.num_runs):
                        gc.collect()  # Clean memory before measurement
                        
                        start_memory = self._measure_memory_usage()
                        
                        exec_time, success, error = self._measure_execution_time(
                            planner.joint_trajectory,
                            thetastart, thetaend, Tf, n_points, method
                        )
                        
                        end_memory = self._measure_memory_usage()
                        memory_diff = end_memory - start_memory
                        
                        times.append(exec_time)
                        successes.append(success)
                        
                        if not success:
                            logger.warning(f"Failed run {run+1}/{self.config.num_runs}: {error}")
                    
                    # Store result
                    if any(successes):
                        avg_time = np.mean([t for t, s in zip(times, successes) if s])
                        result = BenchmarkResult(
                            test_name="joint_trajectory",
                            implementation=impl,
                            n_points=n_points,
                            n_joints=n_joints,
                            batch_size=1,
                            execution_time=avg_time,
                            memory_usage_mb=memory_diff,
                            success=True
                        )
                    else:
                        result = BenchmarkResult(
                            test_name="joint_trajectory",
                            implementation=impl,
                            n_points=n_points,
                            n_joints=n_joints,
                            batch_size=1,
                            execution_time=0.0,
                            memory_usage_mb=0.0,
                            success=False,
                            error_message="All runs failed"
                        )
                    
                    self.results.append(result)
                    
                    # Cleanup
                    planner.cleanup_gpu_memory()
                    del planner
    
    def benchmark_inverse_dynamics(self) -> None:
        """Benchmark inverse dynamics computation"""
        logger.info("Benchmarking inverse dynamics...")
        
        for n_joints in self.config.joint_counts:
            for n_points in self.config.trajectory_points:
                logger.info(f"Testing inverse dynamics: {n_joints} joints, {n_points} points")
                
                # Create test robot
                robot, dynamics, joint_limits = self._create_test_robot(n_joints)
                
                # Generate test trajectory data
                thetalist_traj = np.random.uniform(-1, 1, (n_points, n_joints)).astype(np.float32)
                dthetalist_traj = np.random.uniform(-0.5, 0.5, (n_points, n_joints)).astype(np.float32)
                ddthetalist_traj = np.random.uniform(-0.1, 0.1, (n_points, n_joints)).astype(np.float32)
                
                # Test both implementations
                implementations = ['cpu', 'gpu'] if CUDA_AVAILABLE else ['cpu']
                
                for impl in implementations:
                    use_cuda = (impl == 'gpu')
                    planner = self._create_safe_planner(robot, dynamics, joint_limits, use_cuda)
                    
                    # Multiple runs for accuracy
                    times = []
                    successes = []
                    
                    for run in range(self.config.num_runs):
                        gc.collect()
                        start_memory = self._measure_memory_usage()
                        
                        exec_time, success, error = self._measure_execution_time(
                            planner.inverse_dynamics_trajectory,
                            thetalist_traj, dthetalist_traj, ddthetalist_traj
                        )
                        
                        end_memory = self._measure_memory_usage()
                        memory_diff = end_memory - start_memory
                        
                        times.append(exec_time)
                        successes.append(success)
                    
                    # Store result
                    if any(successes):
                        avg_time = np.mean([t for t, s in zip(times, successes) if s])
                        result = BenchmarkResult(
                            test_name="inverse_dynamics",
                            implementation=impl,
                            n_points=n_points,
                            n_joints=n_joints,
                            batch_size=1,
                            execution_time=avg_time,
                            memory_usage_mb=memory_diff,
                            success=True
                        )
                    else:
                        result = BenchmarkResult(
                            test_name="inverse_dynamics",
                            implementation=impl,
                            n_points=n_points,
                            n_joints=n_joints,
                            batch_size=1,
                            execution_time=0.0,
                            memory_usage_mb=0.0,
                            success=False,
                            error_message="All runs failed"
                        )
                    
                    self.results.append(result)
                    planner.cleanup_gpu_memory()
                    del planner
    
    def benchmark_batch_processing(self) -> None:
        """Benchmark batch trajectory processing"""
        if not CUDA_AVAILABLE:
            logger.info("Skipping batch processing benchmark (CUDA not available)")
            return
        
        logger.info("Benchmarking batch processing...")
        
        for n_joints in self.config.joint_counts:
            for batch_size in self.config.batch_sizes:
                n_points = 500  # Fixed points for batch testing
                
                logger.info(f"Testing batch: {batch_size} trajectories, {n_joints} joints")
                
                # Create test robot
                robot, dynamics, joint_limits = self._create_test_robot(n_joints)
                
                # Generate batch test data
                thetastart_batch = np.random.uniform(-1, 1, (batch_size, n_joints)).astype(np.float32)
                thetaend_batch = np.random.uniform(-1, 1, (batch_size, n_joints)).astype(np.float32)
                Tf = 2.0
                method = 5
                
                # Create planner with specific implementation
                planner = self._create_safe_planner(robot, dynamics, joint_limits, True)
                
                times = []
                successes = []
                
                for run in range(self.config.num_runs):
                    gc.collect()
                    start_memory = self._measure_memory_usage()
                    
                    exec_time, success, error = self._measure_execution_time(
                        planner.batch_joint_trajectory,
                        thetastart_batch, thetaend_batch, Tf, n_points, method
                    )
                    
                    end_memory = self._measure_memory_usage()
                    memory_diff = end_memory - start_memory
                    
                    times.append(exec_time)
                    successes.append(success)
                
                # Store result
                if any(successes):
                    avg_time = np.mean([t for t, s in zip(times, successes) if s])
                    result = BenchmarkResult(
                        test_name="batch_processing",
                        implementation="gpu",
                        n_points=n_points,
                        n_joints=n_joints,
                        batch_size=batch_size,
                        execution_time=avg_time,
                        memory_usage_mb=memory_diff,
                        success=True
                    )
                else:
                    result = BenchmarkResult(
                        test_name="batch_processing",
                        implementation="gpu",
                        n_points=n_points,
                        n_joints=n_joints,
                        batch_size=batch_size,
                        execution_time=0.0,
                        memory_usage_mb=0.0,
                        success=False,
                        error_message="All runs failed"
                    )
                
                self.results.append(result)
                planner.cleanup_gpu_memory()
                del planner
    
    def benchmark_cartesian_trajectory(self) -> None:
        """Benchmark Cartesian trajectory generation"""
        logger.info("Benchmarking Cartesian trajectory generation...")
        
        for n_points in self.config.trajectory_points:
            logger.info(f"Testing Cartesian trajectory: {n_points} points")
            
            # Create test robot (use 6 DOF for Cartesian space)
            robot, dynamics, joint_limits = self._create_test_robot(6)
            
            # Generate test Cartesian data
            Xstart = np.eye(4)
            Xstart[:3, 3] = [0, 0, 0.5]
            
            Xend = np.eye(4)
            Xend[:3, 3] = [0.3, 0.3, 0.8]
            
            Tf = 2.0
            method = 5
            
            # Test both implementations
            implementations = ['cpu', 'gpu'] if CUDA_AVAILABLE else ['cpu']
            
            for impl in implementations:
                use_cuda = (impl == 'gpu')
                planner = self._create_safe_planner(robot, dynamics, joint_limits, use_cuda)
                
                times = []
                successes = []
                
                for run in range(self.config.num_runs):
                    gc.collect()
                    start_memory = self._measure_memory_usage()
                    
                    exec_time, success, error = self._measure_execution_time(
                        planner.cartesian_trajectory,
                        Xstart, Xend, Tf, n_points, method
                    )
                    
                    end_memory = self._measure_memory_usage()
                    memory_diff = end_memory - start_memory
                    
                    times.append(exec_time)
                    successes.append(success)
                
                # Store result
                if any(successes):
                    avg_time = np.mean([t for t, s in zip(times, successes) if s])
                    result = BenchmarkResult(
                        test_name="cartesian_trajectory",
                        implementation=impl,
                        n_points=n_points,
                        n_joints=6,  # Fixed for Cartesian
                        batch_size=1,
                        execution_time=avg_time,
                        memory_usage_mb=memory_diff,
                        success=True
                    )
                else:
                    result = BenchmarkResult(
                        test_name="cartesian_trajectory",
                        implementation=impl,
                        n_points=n_points,
                        n_joints=6,
                        batch_size=1,
                        execution_time=0.0,
                        memory_usage_mb=0.0,
                        success=False,
                        error_message="All runs failed"
                    )
                
                self.results.append(result)
                planner.cleanup_gpu_memory()
                del planner
    
    def run_all_benchmarks(self) -> None:
        """Execute all benchmark tests"""
        logger.info("Starting comprehensive benchmark suite...")
        
        if self.config.enable_profiling and CUDA_AVAILABLE:
            profile_start()
        
        try:
            self.benchmark_joint_trajectory()
            self.benchmark_inverse_dynamics()
            self.benchmark_cartesian_trajectory()
            self.benchmark_batch_processing()
        finally:
            if self.config.enable_profiling and CUDA_AVAILABLE:
                profile_stop()
        
        logger.info(f"Completed {len(self.results)} benchmark tests")
    
    def analyze_results(self) -> Dict:
        """Analyze benchmark results and compute statistics"""
        if not self.results:
            logger.warning("No results to analyze")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'test_name': r.test_name,
                'implementation': r.implementation,
                'n_points': r.n_points,
                'n_joints': r.n_joints,
                'batch_size': r.batch_size,
                'execution_time': r.execution_time,
                'memory_usage_mb': r.memory_usage_mb,
                'success': r.success
            }
            for r in self.results if r.success
        ])
        
        if df.empty:
            logger.warning("No successful benchmark results")
            return {}
        
        # Compute speedup statistics
        analysis = {}
        
        for test_name in df['test_name'].unique():
            test_df = df[df['test_name'] == test_name]
            
            if 'gpu' in test_df['implementation'].values and 'cpu' in test_df['implementation'].values:
                cpu_times = test_df[test_df['implementation'] == 'cpu']['execution_time']
                gpu_times = test_df[test_df['implementation'] == 'gpu']['execution_time']
                
                if len(cpu_times) > 0 and len(gpu_times) > 0:
                    # Match corresponding problem sizes
                    speedups = []
                    for _, cpu_row in test_df[test_df['implementation'] == 'cpu'].iterrows():
                        matching_gpu = test_df[
                            (test_df['implementation'] == 'gpu') &
                            (test_df['n_points'] == cpu_row['n_points']) &
                            (test_df['n_joints'] == cpu_row['n_joints']) &
                            (test_df['batch_size'] == cpu_row['batch_size'])
                        ]
                        
                        if not matching_gpu.empty:
                            speedup = cpu_row['execution_time'] / matching_gpu.iloc[0]['execution_time']
                            speedups.append(speedup)
                    
                    if speedups:
                        analysis[test_name] = {
                            'mean_speedup': np.mean(speedups),
                            'max_speedup': np.max(speedups),
                            'min_speedup': np.min(speedups),
                            'std_speedup': np.std(speedups),
                            'speedup_samples': len(speedups)
                        }
        
        # Overall statistics
        if 'gpu' in df['implementation'].values and 'cpu' in df['implementation'].values:
            cpu_df = df[df['implementation'] == 'cpu']
            gpu_df = df[df['implementation'] == 'gpu']
            
            analysis['overall'] = {
                'cpu_mean_time': cpu_df['execution_time'].mean(),
                'gpu_mean_time': gpu_df['execution_time'].mean(),
                'cpu_total_tests': len(cpu_df),
                'gpu_total_tests': len(gpu_df),
                'success_rate_cpu': cpu_df['success'].mean(),
                'success_rate_gpu': gpu_df['success'].mean()
            }
        
        return analysis
    
    def plot_results(self) -> None:
        """Generate comprehensive visualization plots"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'test_name': r.test_name,
                'implementation': r.implementation,
                'n_points': r.n_points,
                'n_joints': r.n_joints,
                'batch_size': r.batch_size,
                'execution_time': r.execution_time,
                'memory_usage_mb': r.memory_usage_mb,
                'success': r.success
            }
            for r in self.results if r.success
        ])
        
        if df.empty:
            logger.warning("No successful results to plot")
            return
        
        # Create comprehensive plot grid
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Execution Time Comparison by Test Type
        plt.subplot(2, 3, 1)
        if 'gpu' in df['implementation'].values and 'cpu' in df['implementation'].values:
            sns.boxplot(data=df, x='test_name', y='execution_time', hue='implementation')
            plt.yscale('log')
            plt.title('Execution Time by Test Type')
            plt.ylabel('Time (seconds, log scale)')
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=df, x='test_name', y='execution_time')
            plt.yscale('log')
            plt.title('Execution Time by Test Type')
            plt.ylabel('Time (seconds, log scale)')
            plt.xticks(rotation=45)
        
        # 2. Scalability with Number of Points
        plt.subplot(2, 3, 2)
        joint_traj_df = df[df['test_name'] == 'joint_trajectory']
        if not joint_traj_df.empty:
            for impl in joint_traj_df['implementation'].unique():
                impl_df = joint_traj_df[joint_traj_df['implementation'] == impl]
                grouped = impl_df.groupby('n_points')['execution_time'].mean()
                plt.plot(grouped.index, grouped.values, 'o-', label=f'{impl.upper()}', linewidth=2)
            plt.xlabel('Number of Trajectory Points')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Scalability: Trajectory Points')
            plt.legend()
            plt.yscale('log')
            plt.xscale('log')
        
        # 3. Scalability with Number of Joints
        plt.subplot(2, 3, 3)
        if not joint_traj_df.empty:
            for impl in joint_traj_df['implementation'].unique():
                impl_df = joint_traj_df[joint_traj_df['implementation'] == impl]
                grouped = impl_df.groupby('n_joints')['execution_time'].mean()
                plt.plot(grouped.index, grouped.values, 's-', label=f'{impl.upper()}', linewidth=2)
            plt.xlabel('Number of Joints')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Scalability: Joint Count')
            plt.legend()
            plt.yscale('log')
        
        # 4. Memory Usage Comparison
        plt.subplot(2, 3, 4)
        if 'gpu' in df['implementation'].values and 'cpu' in df['implementation'].values:
            sns.boxplot(data=df, x='test_name', y='memory_usage_mb', hue='implementation')
            plt.title('Memory Usage by Test Type')
            plt.ylabel('Memory (MB)')
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=df, x='test_name', y='memory_usage_mb')
            plt.title('Memory Usage by Test Type')
            plt.ylabel('Memory (MB)')
            plt.xticks(rotation=45)
        
        # 5. Speedup Analysis
        plt.subplot(2, 3, 5)
        if 'gpu' in df['implementation'].values and 'cpu' in df['implementation'].values:
            speedups = []
            test_labels = []
            
            for test_name in df['test_name'].unique():
                test_df = df[df['test_name'] == test_name]
                for _, cpu_row in test_df[test_df['implementation'] == 'cpu'].iterrows():
                    matching_gpu = test_df[
                        (test_df['implementation'] == 'gpu') &
                        (test_df['n_points'] == cpu_row['n_points']) &
                        (test_df['n_joints'] == cpu_row['n_joints'])
                    ]
                    
                    if not matching_gpu.empty:
                        speedup = cpu_row['execution_time'] / matching_gpu.iloc[0]['execution_time']
                        speedups.append(speedup)
                        test_labels.append(test_name)
            
            if speedups:
                speedup_df = pd.DataFrame({'speedup': speedups, 'test_name': test_labels})
                sns.boxplot(data=speedup_df, x='test_name', y='speedup')
                plt.axhline(y=1, color='red', linestyle='--', label='No speedup')
                plt.title('GPU Speedup by Test Type')
                plt.ylabel('Speedup Factor')
                plt.xticks(rotation=45)
                plt.legend()
        
        # 6. Batch Processing Performance
        plt.subplot(2, 3, 6)
        batch_df = df[df['test_name'] == 'batch_processing']
        if not batch_df.empty:
            grouped = batch_df.groupby('batch_size')['execution_time'].mean()
            plt.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Batch Size')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Batch Processing Scalability')
            plt.yscale('log')
            plt.xscale('log')
        else:
            plt.text(0.5, 0.5, 'No batch processing data', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Batch Processing Scalability')
        
        plt.tight_layout()
        
        # Save the plot in plots directory
        plot_path = self.plots_dir / 'benchmark_results_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance plots to {plot_path}")
        
        # Also save as PDF for publication quality
        plot_path_pdf = self.plots_dir / 'benchmark_results_overview.pdf'
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        
        # Close to save memory
        plt.close()
        
        # Create additional detailed plots
        self._plot_detailed_analysis(df)
    
    def _plot_detailed_analysis(self, df: pd.DataFrame) -> None:
        """Create detailed analysis plots"""
        
        # Performance heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap of execution times
        if not df.empty:
            # Joint trajectory heatmap
            joint_df = df[df['test_name'] == 'joint_trajectory']
            if not joint_df.empty and 'gpu' in joint_df['implementation'].values:
                gpu_joint_df = joint_df[joint_df['implementation'] == 'gpu']
                pivot_data = gpu_joint_df.pivot_table(
                    values='execution_time', 
                    index='n_joints', 
                    columns='n_points', 
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                               ax=axes[0,0], cbar_kws={'label': 'Time (s)'})
                    axes[0,0].set_title('GPU Joint Trajectory: Execution Time Heatmap')
            
            # Speedup heatmap
            if 'gpu' in df['implementation'].values and 'cpu' in df['implementation'].values:
                speedup_data = []
                for test_name in df['test_name'].unique():
                    test_df = df[df['test_name'] == test_name]
                    for _, cpu_row in test_df[test_df['implementation'] == 'cpu'].iterrows():
                        matching_gpu = test_df[
                            (test_df['implementation'] == 'gpu') &
                            (test_df['n_points'] == cpu_row['n_points']) &
                            (test_df['n_joints'] == cpu_row['n_joints'])
                        ]
                        
                        if not matching_gpu.empty:
                            speedup = cpu_row['execution_time'] / matching_gpu.iloc[0]['execution_time']
                            speedup_data.append({
                                'test_name': test_name,
                                'n_points': cpu_row['n_points'],
                                'n_joints': cpu_row['n_joints'],
                                'speedup': speedup
                            })
                
                if speedup_data:
                    speedup_df = pd.DataFrame(speedup_data)
                    joint_speedup = speedup_df[speedup_df['test_name'] == 'joint_trajectory']
                    
                    if not joint_speedup.empty:
                        speedup_pivot = joint_speedup.pivot_table(
                            values='speedup', 
                            index='n_joints', 
                            columns='n_points', 
                            aggfunc='mean'
                        )
                        
                        if not speedup_pivot.empty:
                            sns.heatmap(speedup_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                                       center=1, ax=axes[0,1], cbar_kws={'label': 'Speedup Factor'})
                            axes[0,1].set_title('GPU vs CPU Speedup Heatmap')
        
        # Efficiency analysis - Points processed per second per joint
        if not df.empty:
            efficiency_data = []
            for _, row in df.iterrows():
                if row['execution_time'] > 0:
                    # Calculate efficiency as (points × joints) per second
                    efficiency = (row['n_points'] * row['n_joints']) / row['execution_time']
                    efficiency_data.append({
                        'implementation': row['implementation'],
                        'test_name': row['test_name'],
                        'efficiency': efficiency,
                        'problem_size': row['n_points'] * row['n_joints']
                    })
            
            if efficiency_data:
                eff_df = pd.DataFrame(efficiency_data)
                
                # Plot efficiency by implementation
                if 'gpu' in eff_df['implementation'].values and 'cpu' in eff_df['implementation'].values:
                    sns.boxplot(data=eff_df, x='implementation', y='efficiency', ax=axes[1,0])
                    axes[1,0].set_ylabel('Efficiency (Points×Joints/sec)')
                    axes[1,0].set_title('Computational Efficiency by Implementation')
                    axes[1,0].set_yscale('log')
                else:
                    # Single implementation
                    sns.boxplot(data=eff_df, x='test_name', y='efficiency', ax=axes[1,0])
                    axes[1,0].set_ylabel('Efficiency (Points×Joints/sec)')
                    axes[1,0].set_title('Computational Efficiency by Test Type')
                    axes[1,0].set_yscale('log')
                    axes[1,0].tick_params(axis='x', rotation=45)
            else:
                axes[1,0].text(0.5, 0.5, 'No efficiency data available', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
        else:
            axes[1,0].text(0.5, 0.5, 'No data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
        
        if not hasattr(axes[1,0], 'get_ylabel') or not axes[1,0].get_ylabel():
            axes[1,0].set_title('Computational Efficiency')
        
        # Problem size scaling
        if not df.empty:
            joint_df = df[df['test_name'] == 'joint_trajectory']
            if not joint_df.empty:
                for impl in joint_df['implementation'].unique():
                    impl_df = joint_df[joint_df['implementation'] == impl]
                    # Calculate throughput (points per second)
                    impl_df = impl_df.copy()
                    impl_df['throughput'] = impl_df['n_points'] / impl_df['execution_time']
                    
                    grouped = impl_df.groupby('n_points')['throughput'].mean()
                    axes[1,1].plot(grouped.index, grouped.values, 'o-', 
                                  label=f'{impl.upper()}', linewidth=2)
                
                axes[1,1].set_xlabel('Number of Trajectory Points')
                axes[1,1].set_ylabel('Throughput (points/second)')
                axes[1,1].set_title('Processing Throughput Comparison')
                axes[1,1].legend()
                axes[1,1].set_xscale('log')
                axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save detailed analysis in plots directory
        detailed_plot_path = self.plots_dir / 'detailed_analysis.png'
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        
        # Also save as PDF
        detailed_plot_path_pdf = self.plots_dir / 'detailed_analysis.pdf'
        plt.savefig(detailed_plot_path_pdf, dpi=300, bbox_inches='tight')
        
        logger.info(f"Saved detailed analysis to {detailed_plot_path}")
        plt.close()
        
        # Save individual performance metric plots
        self._save_individual_plots(df)
    
    def _save_individual_plots(self, df: pd.DataFrame) -> None:
        """Save individual plots for each metric and test type"""
        if df.empty:
            return
        
        # Plot execution time trends for each test type
        for test_name in df['test_name'].unique():
            test_df = df[df['test_name'] == test_name]
            
            # Create subplots for different views
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{test_name.replace("_", " ").title()} - Detailed Analysis', fontsize=16)
            
            # 1. Execution time vs problem size
            if 'gpu' in test_df['implementation'].values and 'cpu' in test_df['implementation'].values:
                for impl in test_df['implementation'].unique():
                    impl_df = test_df[test_df['implementation'] == impl]
                    grouped = impl_df.groupby('n_points')['execution_time'].agg(['mean', 'std'])
                    
                    axes[0,0].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                                      label=f'{impl.upper()}', marker='o', capsize=5)
                
                axes[0,0].set_xlabel('Number of Points')
                axes[0,0].set_ylabel('Execution Time (s)')
                axes[0,0].set_title('Performance vs Problem Size')
                axes[0,0].legend()
                axes[0,0].set_yscale('log')
                axes[0,0].grid(True, alpha=0.3)
            
            # 2. Memory usage comparison
            if 'memory_usage_mb' in test_df.columns:
                sns.boxplot(data=test_df, x='implementation', y='memory_usage_mb', ax=axes[0,1])
                axes[0,1].set_title('Memory Usage Distribution')
                axes[0,1].set_ylabel('Memory Usage (MB)')
            
            # 3. Speedup analysis (if both CPU and GPU data available)
            if 'gpu' in test_df['implementation'].values and 'cpu' in test_df['implementation'].values:
                speedups = []
                problem_sizes = []
                
                for _, cpu_row in test_df[test_df['implementation'] == 'cpu'].iterrows():
                    matching_gpu = test_df[
                        (test_df['implementation'] == 'gpu') &
                        (test_df['n_points'] == cpu_row['n_points']) &
                        (test_df['n_joints'] == cpu_row['n_joints'])
                    ]
                    
                    if not matching_gpu.empty:
                        speedup = cpu_row['execution_time'] / matching_gpu.iloc[0]['execution_time']
                        speedups.append(speedup)
                        problem_sizes.append(cpu_row['n_points'] * cpu_row['n_joints'])
                
                if speedups:
                    axes[1,0].scatter(problem_sizes, speedups, alpha=0.7)
                    axes[1,0].axhline(y=1, color='red', linestyle='--', label='No speedup')
                    axes[1,0].set_xlabel('Problem Size (Points × Joints)')
                    axes[1,0].set_ylabel('Speedup Factor')
                    axes[1,0].set_title('GPU Speedup vs Problem Size')
                    axes[1,0].set_xscale('log')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)
            
            # 4. Success rate analysis
            success_rate = test_df.groupby('implementation')['success'].mean()
            axes[1,1].bar(success_rate.index, success_rate.values)
            axes[1,1].set_title('Success Rate by Implementation')
            axes[1,1].set_ylabel('Success Rate')
            axes[1,1].set_ylim(0, 1.1)
            
            # Add success rate labels on bars
            for i, v in enumerate(success_rate.values):
                axes[1,1].text(i, v + 0.02, f'{v:.1%}', ha='center')
            
            plt.tight_layout()
            
            # Save individual test plot
            individual_plot_path = self.plots_dir / f'{test_name}_analysis.png'
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            
            individual_plot_path_pdf = self.plots_dir / f'{test_name}_analysis.pdf'
            plt.savefig(individual_plot_path_pdf, dpi=300, bbox_inches='tight')
            
            plt.close()  # Close to save memory
            
            logger.info(f"Saved individual analysis for {test_name}")
    
    def save_results(self) -> None:
        """Save comprehensive benchmark results to multiple file formats"""
        if not self.config.save_results:
            return
        
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save raw results as JSON
        results_data = {
            'metadata': {
                'benchmark_version': '1.0.0',
                'timestamp': timestamp,
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r.success),
                'failed_tests': sum(1 for r in self.results if not r.success)
            },
            'system_info': self.system_info,
            'config': {
                'test_name': self.config.test_name,
                'trajectory_points': self.config.trajectory_points,
                'joint_counts': self.config.joint_counts,
                'batch_sizes': self.config.batch_sizes,
                'num_runs': self.config.num_runs,
                'timeout_seconds': self.config.timeout_seconds,
                'enable_profiling': self.config.enable_profiling
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'implementation': r.implementation,
                    'n_points': r.n_points,
                    'n_joints': r.n_joints,
                    'batch_size': r.batch_size,
                    'execution_time': r.execution_time,
                    'memory_usage_mb': r.memory_usage_mb,
                    'success': r.success,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }
        
        # Save complete results as JSON
        json_path = self.data_dir / f'benchmark_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Saved complete results to {json_path}")
        
        # Save latest results (without timestamp for easy access)
        json_path_latest = self.data_dir / 'benchmark_results_latest.json'
        with open(json_path_latest, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV for easy analysis (successful results only)
        if self.results:
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                df = pd.DataFrame([
                    {
                        'test_name': r.test_name,
                        'implementation': r.implementation,
                        'n_points': r.n_points,
                        'n_joints': r.n_joints,
                        'batch_size': r.batch_size,
                        'execution_time': r.execution_time,
                        'memory_usage_mb': r.memory_usage_mb,
                        'success': r.success,
                        'error_message': r.error_message
                    }
                    for r in successful_results
                ])
                
                csv_path = self.data_dir / f'benchmark_results_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                
                csv_path_latest = self.data_dir / 'benchmark_results_latest.csv'
                df.to_csv(csv_path_latest, index=False)
                
                logger.info(f"Saved CSV data to {csv_path}")
                
                # Save summary statistics
                self._save_summary_statistics(df, timestamp)
        
        # Save failed results separately if any
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            failed_data = {
                'timestamp': timestamp,
                'failed_count': len(failed_results),
                'failed_tests': [
                    {
                        'test_name': r.test_name,
                        'implementation': r.implementation,
                        'n_points': r.n_points,
                        'n_joints': r.n_joints,
                        'batch_size': r.batch_size,
                        'error_message': r.error_message
                    }
                    for r in failed_results
                ]
            }
            
            failed_path = self.data_dir / f'failed_tests_{timestamp}.json'
            with open(failed_path, 'w') as f:
                json.dump(failed_data, f, indent=2)
            logger.info(f"Saved failed test data to {failed_path}")
    
    def _save_summary_statistics(self, df: pd.DataFrame, timestamp: str) -> None:
        """Save statistical summary of benchmark results"""
        summary_stats = {
            'timestamp': timestamp,
            'overview': {
                'total_successful_tests': len(df),
                'test_types': df['test_name'].unique().tolist(),
                'implementations': df['implementation'].unique().tolist(),
                'joint_counts_tested': sorted(df['n_joints'].unique().tolist()),
                'point_counts_tested': sorted(df['n_points'].unique().tolist())
            },
            'performance_summary': {},
            'timing_statistics': {
                'overall_min_time': float(df['execution_time'].min()),
                'overall_max_time': float(df['execution_time'].max()),
                'overall_mean_time': float(df['execution_time'].mean()),
                'overall_std_time': float(df['execution_time'].std())
            }
        }
        
        # Calculate per-test-type statistics
        for test_name in df['test_name'].unique():
            test_df = df[df['test_name'] == test_name]
            
            test_stats = {
                'count': len(test_df),
                'mean_time': float(test_df['execution_time'].mean()),
                'std_time': float(test_df['execution_time'].std()),
                'min_time': float(test_df['execution_time'].min()),
                'max_time': float(test_df['execution_time'].max()),
                'mean_memory': float(test_df['memory_usage_mb'].mean()),
                'implementations': test_df['implementation'].unique().tolist()
            }
            
            # Calculate speedup if both CPU and GPU results exist
            if 'cpu' in test_df['implementation'].values and 'gpu' in test_df['implementation'].values:
                cpu_mean = test_df[test_df['implementation'] == 'cpu']['execution_time'].mean()
                gpu_mean = test_df[test_df['implementation'] == 'gpu']['execution_time'].mean()
                test_stats['average_speedup'] = float(cpu_mean / gpu_mean) if gpu_mean > 0 else 0.0
            
            summary_stats['performance_summary'][test_name] = test_stats
        
        # Save summary statistics
        summary_path = self.data_dir / f'summary_statistics_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        summary_path_latest = self.data_dir / 'summary_statistics_latest.json'
        with open(summary_path_latest, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Saved summary statistics to {summary_path}")

    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        analysis = self.analyze_results()

        # ── Header ───────────────────────────────────────────────────────────
        report = f"""
# ManipulaPy Path Planning Performance Benchmark Report

## System Information
- CPU Cores: {self.system_info.get('cpu_count', 'Unknown')}
- Memory: {self.system_info.get('memory_gb', 'Unknown'):.1f} GB
- CUDA Available: {self.system_info.get('cuda_available', False)}
"""
        if self.system_info.get('cuda_available', False):
            report += (
                f"- GPU Multiprocessors: "
                f"{self.system_info.get('gpu_multiprocessors', 'Unknown')}\n"
                f"- Max Threads per Block: "
                f"{self.system_info.get('gpu_max_threads_per_block', 'Unknown')}\n"
                f"- Max Shared Memory: "
                f"{self.system_info.get('gpu_max_shared_memory', 'Unknown')} bytes\n"
            )

        # ── Benchmark config ────────────────────────────────────────────────
        report += f"""
## Benchmark Configuration
- Test Name: {self.config.test_name}
- Trajectory Points: {self.config.trajectory_points}
- Joint Counts: {self.config.joint_counts}
- Batch Sizes: {self.config.batch_sizes}
- Runs per Test: {self.config.num_runs}
- Timeout: {self.config.timeout_seconds}s

## Results Summary
- Total Tests: {len(self.results)}
- Successful Tests: {sum(1 for r in self.results if r.success)}
- Failed Tests: {sum(1 for r in self.results if not r.success)}
"""

        # ── Per-test analysis ───────────────────────────────────────────────
        if analysis:
            report += "\n## Performance Analysis\n"
            for test_name, stats in analysis.items():
                if test_name == "overall":
                    continue
                report += (
                    f"""
### {test_name.replace('_', ' ').title()}
- Mean Speedup: {stats.get('mean_speedup', 0):.2f}x
- Max Speedup: {stats.get('max_speedup', 0):.2f}x
- Min Speedup: {stats.get('min_speedup', 0):.2f}x
- Standard Deviation: {stats.get('std_speedup', 0):.2f}
- Sample Size: {stats.get('speedup_samples', 0)}
"""
                )

            # ── Overall stats ───────────────────────────────────────────────
            if "overall" in analysis:
                overall = analysis["overall"]
                cpu_mean_time = overall.get("cpu_mean_time", 0)
                gpu_mean_time = overall.get("gpu_mean_time", 0)
                # Safe division (∞ when GPU time is zero / absent)
                performance_factor = (
                    cpu_mean_time / gpu_mean_time if gpu_mean_time else float("inf")
                )
                overall["performance_factor"] = performance_factor  # Optional: store

                report += f"""
## Overall Statistics
- CPU Mean Time: {cpu_mean_time:.4f}s
- GPU Mean Time: {gpu_mean_time:.4f}s
- Performance Factor: {performance_factor:.2f}×
- CPU Success Rate: {overall.get('success_rate_cpu', 0):.1%}
- GPU Success Rate: {overall.get('success_rate_gpu', 0):.1%}
- Total CPU Tests: {overall.get('cpu_total_tests', 0)}
- Total GPU Tests: {overall.get('gpu_total_tests', 0)}
"""

        # ── Recommendations ────────────────────────────────────────────────
        report += """
## Recommendations

### When to Use GPU Acceleration:
- Large trajectory point counts (>1000 points)
- Multiple joint configurations (>6 joints)
- Batch processing multiple trajectories
- Real-time applications requiring low latency

### When to Use CPU:
- Small problem sizes (<500 points, <6 joints)
- Single trajectory computations
- Limited GPU memory scenarios
- Development and debugging

### Optimization Tips:
1. Use batch processing for multiple trajectories
2. Consider GPU memory limitations for very large problems
3. Profile your specific use case for optimal performance
4. Utilize adaptive thresholding in OptimizedTrajectoryPlanning
"""

        return report


def create_default_config() -> BenchmarkConfig:
    """Create a default benchmark configuration"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return BenchmarkConfig(
        test_name="ManipulaPy_Performance_Benchmark",
        trajectory_points = [int(p * 100) for p in [100, 500, 1000]],
        joint_counts=[3, 6],
        batch_sizes=[1, 2, 3],
        num_runs=3,
        timeout_seconds=60.0,
        enable_profiling=True,
        save_results=True,
        output_dir=f"benchmark_results_{timestamp}"  # Timestamped folder
    )


def main():
    """Main benchmark execution function"""
    print("=" * 60)
    print("ManipulaPy Path Planning Performance Benchmark")
    print("=" * 60)
    
    # Check CUDA availability
    if CUDA_AVAILABLE:
        print("✅ CUDA acceleration available")
        gpu_props = get_gpu_properties()
        if gpu_props:
            print(f"📊 GPU: {gpu_props.get('multiprocessor_count', 'Unknown')} SMs, "
                  f"{gpu_props.get('max_threads_per_block', 'Unknown')} max threads/block")
    else:
        print("⚠️  CUDA acceleration not available - CPU-only benchmarks")
    
    # Create configuration
    config = create_default_config()
    
    # Allow user customization
    print("\nBenchmark Configuration:")
    print(f"- Trajectory points: {config.trajectory_points}")
    print(f"- Joint counts: {config.joint_counts}")
    print(f"- Batch sizes: {config.batch_sizes}")
    print(f"- Runs per test: {config.num_runs}")
    
    # Run benchmark
    benchmark = PathPlanningBenchmark(config)
    
    try:
        benchmark.run_all_benchmarks()
        
        # Analyze and visualize results
        analysis = benchmark.analyze_results()
        benchmark.plot_results()
        
        # Generate and save report
        report = benchmark.generate_report()
        report_path = benchmark.reports_dir / f'benchmark_report_{time.strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Also save latest report (without timestamp for easy access)
        report_path_latest = benchmark.reports_dir / 'benchmark_report_latest.md'
        with open(report_path_latest, 'w') as f:
            f.write(report)
        
        print(f"\n📝 Full report saved to: {report_path}")
        print(f"📝 Latest report available at: {report_path_latest}")
        
        # Save results
        benchmark.save_results()
        
        # Create a summary file in the main directory
        summary_file = benchmark.output_path / 'README.md'
        with open(summary_file, 'w') as f:
            f.write(f"""# ManipulaPy Benchmark Results

## Directory Structure

- **`data/`** - Raw benchmark data in JSON and CSV formats
  - `benchmark_results_latest.json` - Complete results in JSON format
  - `benchmark_results_latest.csv` - Tabular data for analysis
  - `summary_statistics_latest.json` - Statistical summary
  - `system_info.json` - System configuration and hardware info
  - `failed_tests_*.json` - Information about any failed tests

- **`plots/`** - Visualization and charts
  - `benchmark_results_overview.*` - Main performance comparison charts
  - `detailed_analysis.*` - In-depth analysis plots
  - `*_analysis.*` - Individual test type analysis

- **`reports/`** - Written reports and documentation
  - `benchmark_report_latest.md` - Comprehensive benchmark report
  - Timestamped reports for historical comparison

- **`logs/`** - Execution logs and debugging information
  - `benchmark_*.log` - Detailed execution logs

## Quick Access Files

- 📊 **Latest Results**: `data/benchmark_results_latest.csv`
- 📈 **Main Charts**: `plots/benchmark_results_overview.png`
- 📋 **Full Report**: `reports/benchmark_report_latest.md`
- ⚙️ **System Info**: `data/system_info.json`

## Benchmark Configuration

- **Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Trajectory Points**: {benchmark.config.trajectory_points}
- **Joint Counts**: {benchmark.config.joint_counts}
- **Batch Sizes**: {benchmark.config.batch_sizes}
- **Runs per Test**: {benchmark.config.num_runs}
- **CUDA Available**: {CUDA_AVAILABLE}

Generated by ManipulaPy Performance Benchmark Suite
""")
        
        print(f"📁 All results saved to: {benchmark.output_path}")
        print(f"📖 Directory overview: {summary_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if analysis:
            for test_name, stats in analysis.items():
                if test_name != 'overall' and 'mean_speedup' in stats:
                    print(f"{test_name}: {stats['mean_speedup']:.2f}x average speedup")
        
        print(f"\nResults saved to: {benchmark.output_path}")
        print("✅ Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        logger.exception("Benchmark failed with exception")
    finally:
        # Cleanup
        gc.collect()


if __name__ == "__main__":
    main()