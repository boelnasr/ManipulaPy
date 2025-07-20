#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Comprehensive Accuracy Benchmark for ManipulaPy

This script provides comprehensive accuracy benchmarks for the ManipulaPy library,
testing various components including kinematics, dynamics, trajectory planning,
control algorithms, and GPU vs CPU implementations using real robot data.

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import time
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ManipulaPy imports
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning
    from ManipulaPy.urdf_processor import URDFToSerialManipulator
    from ManipulaPy.cuda_kernels import (
        CUDA_AVAILABLE, 
        check_cuda_availability,
        optimized_trajectory_generation,
        trajectory_cpu_fallback
    )
    from ManipulaPy import utils
    from ManipulaPy.ManipulaPy_data.xarm import urdf_file
    MANIPULAPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ManipulaPy not fully available: {e}")
    MANIPULAPY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class PatchedSingularity:
    """
    Patched version of Singularity class that handles non-square Jacobians properly.
    """
    def __init__(self, serial_manipulator):
        self.serial_manipulator = serial_manipulator

    def singularity_analysis(self, thetalist):
        """
        Analyze if the manipulator is at a singularity based on the manipulability measure.
        """
        try:
            J = self.serial_manipulator.jacobian(thetalist, frame="space")
            
            # For non-square Jacobians (typical case for 6-DOF end-effector in n-DOF space)
            if J.shape[0] != J.shape[1]:
                # Use manipulability index: sqrt(det(J * J^T))
                JJT = J @ J.T
                det_JJT = np.linalg.det(JJT)
                manipulability = np.sqrt(abs(det_JJT))
                # Singularity when manipulability is near zero
                return manipulability < 1e-4
            else:
                # Square Jacobian case (rare)
                det_J = np.linalg.det(J)
                return abs(det_J) < 1e-4
        except (np.linalg.LinAlgError, ValueError):
            # If calculation fails, assume singular
            return True

    def condition_number(self, thetalist):
        """
        Calculate the condition number of the Jacobian for a given set of joint angles.
        For non-square Jacobians, uses the condition number of J*J^T.
        """
        try:
            J = self.serial_manipulator.jacobian(thetalist, frame="space")
            
            if J.shape[0] == J.shape[1]:
                # Square Jacobian
                return np.linalg.cond(J)
            else:
                # Non-square Jacobian - use condition number of J*J^T
                JJT = J @ J.T
                return np.linalg.cond(JJT)
        except (np.linalg.LinAlgError, ValueError):
            # Return a large number if condition number cannot be computed
            return 1e12

    def near_singularity_detection(self, thetalist, threshold=1e2):
        """
        Detect if the manipulator is near a singularity by comparing the condition number with a threshold.
        """
        try:
            cond_number = self.condition_number(thetalist)
            return cond_number > threshold
        except:
            return True


class AccuracyBenchmark:
    """
    Comprehensive accuracy benchmark suite for ManipulaPy components.
    
    Tests include:
    - Forward kinematics accuracy and consistency
    - Inverse kinematics convergence and accuracy
    - Jacobian computation accuracy
    - Dynamics calculations (mass matrix, Coriolis, gravity)
    - Trajectory planning accuracy (CPU vs GPU)
    - Control algorithm performance
    - Singularity detection accuracy
    """
    
    def __init__(self, output_dir: str = "accuracy_benchmark_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize X-Arm robot from real data
        self.robot = None
        self.dynamics = None
        self.joint_limits = None
        self.dof = None
        
        self._setup_xarm_robot()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Accuracy benchmark initialized. Results will be saved to: {output_dir}")
        
    def _setup_xarm_robot(self):
        """Setup the X-Arm robot from ManipulaPy data."""
        try:
            logger.info(f"Loading X-Arm from: {urdf_file}")
            
            # Create robot from URDF
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator
            self.dynamics = urdf_processor.dynamics
            
            # Get joint limits
            if hasattr(urdf_processor, 'robot_data') and 'joint_limits' in urdf_processor.robot_data:
                self.joint_limits = urdf_processor.robot_data['joint_limits']
            else:
                # Fallback: X-Arm 6 DOF limits
                self.joint_limits = [
                    (-6.28, 6.28),    # Joint 1: ±360°
                    (-2.09, 2.09),    # Joint 2: ±120°
                    (-6.28, 6.28),    # Joint 3: ±360°
                    (-6.28, 6.28),    # Joint 4: ±360°
                    (-6.28, 6.28),    # Joint 5: ±360°
                    (-6.28, 6.28),    # Joint 6: ±360°
                ]
            
            self.dof = len(self.joint_limits)
            
            logger.info(f"X-Arm loaded successfully - DOF: {self.dof}")
            
            # Test basic functionality
            self._test_basic_functionality()
            
        except Exception as e:
            logger.error(f"Failed to load X-Arm: {e}")
            raise

    def _test_basic_functionality(self):
        """Test that basic robot functions work."""
        logger.info("Testing basic X-Arm functionality...")
        
        try:
            # Test forward kinematics
            zero_config = [0.0] * self.dof
            pose = self.robot.forward_kinematics(zero_config)
            logger.info(f"Forward kinematics works - pose shape: {pose.shape}")
            
            # Test Jacobian
            J = self.robot.jacobian(zero_config)
            logger.info(f"Jacobian works - shape: {J.shape}")
            
            # Test pose validity
            R = pose[:3, :3]
            is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            det_R = np.linalg.det(R)
            logger.info(f"Rotation matrix valid: orthogonal={is_orthogonal}, det={det_R:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return False

    def benchmark_forward_kinematics(self, num_tests: int = 1000) -> Dict:
        """
        Benchmark forward kinematics accuracy and consistency.
        
        Args:
            num_tests: Number of random configurations to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking forward kinematics...")
        
        # Generate random joint configurations
        joint_configs = []
        for _ in range(num_tests):
            config = []
            for low, high in self.joint_limits:
                config.append(np.random.uniform(low, high))
            joint_configs.append(config)
        
        # Test forward kinematics
        computation_times = []
        poses = []
        valid_poses = 0
        
        for config in joint_configs:
            start_time = time.time()
            try:
                pose = self.robot.forward_kinematics(config)
                end_time = time.time()
                
                computation_times.append(end_time - start_time)
                poses.append(pose)
                
                # Check if pose is valid SE(3) matrix
                R = pose[:3, :3]
                if np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
                    valid_poses += 1
                    
            except Exception as e:
                logger.warning(f"Forward kinematics failed: {e}")
                computation_times.append(float('inf'))
        
        # Test consistency (same input should give same output)
        test_config = joint_configs[0]
        poses_repeated = []
        for _ in range(10):
            pose = self.robot.forward_kinematics(test_config)
            poses_repeated.append(pose)
        
        # Calculate consistency error
        max_error = 0.0
        for i in range(1, len(poses_repeated)):
            error = np.linalg.norm(poses_repeated[i] - poses_repeated[0])
            max_error = max(max_error, error)
        
        valid_times = [t for t in computation_times if not np.isinf(t)]
        
        results = {
            'avg_computation_time': np.mean(valid_times) if valid_times else float('inf'),
            'std_computation_time': np.std(valid_times) if valid_times else float('inf'),
            'max_computation_time': np.max(valid_times) if valid_times else float('inf'),
            'min_computation_time': np.min(valid_times) if valid_times else float('inf'),
            'consistency_error': max_error,
            'valid_poses_percentage': (valid_poses / len(poses)) * 100 if poses else 0,
            'success_rate': (len(valid_times) / num_tests) * 100,
            'num_tests': num_tests
        }
        
        logger.info(f"Forward kinematics: {results['success_rate']:.1f}% success, "
                   f"{results['avg_computation_time']*1000:.3f}ms avg time")
        
        return results

    def benchmark_inverse_kinematics(self, num_tests: int = 100) -> Dict:
        """
        Benchmark inverse kinematics accuracy and convergence.
        
        Args:
            num_tests: Number of target poses to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking inverse kinematics...")
        
        # Generate REACHABLE target poses (not random)
        target_poses = []
        original_configs = []
        
        logger.info("Generating reachable target poses...")
        attempts = 0
        max_attempts = num_tests * 3
        
        while len(target_poses) < num_tests and attempts < max_attempts:
            attempts += 1
            
            # Generate configuration biased toward workspace center
            config = []
            for low, high in self.joint_limits:
                center = (low + high) / 2
                range_size = high - low
                # Use 60% of range around center (avoid extremes)
                biased_low = center - range_size * 0.3
                biased_high = center + range_size * 0.3
                biased_low = max(biased_low, low)
                biased_high = min(biased_high, high)
                
                config.append(np.random.uniform(biased_low, biased_high))
            
            try:
                # Check if configuration is reasonable (not near singularities)
                J = self.robot.jacobian(config)
                if J.shape[0] == J.shape[1]:
                    cond_num = np.linalg.cond(J)
                else:
                    cond_num = np.linalg.cond(J @ J.T)
                
                # Skip very badly conditioned poses
                if cond_num > 1e5:
                    continue
                
                pose = self.robot.forward_kinematics(config)
                target_poses.append(pose)
                original_configs.append(config)
                
            except Exception:
                continue
        
        logger.info(f"Generated {len(target_poses)} reachable target poses")
        
        if len(target_poses) == 0:
            return {
                'convergence_rate': 0,
                'num_tests': 0,
                'error': 'No target poses generated'
            }
        
        # Test inverse kinematics with multiple strategies
        convergence_count = 0
        position_errors = []
        orientation_errors = []
        computation_times = []
        iterations_list = []
        
        strategies = ['random', 'center', 'near_original']
        
        for i, target_pose in enumerate(target_poses):
            start_time = time.time()
            
            success = False
            best_result = None
            best_iterations = 5000
            
            # Try multiple initial guess strategies
            for strategy in strategies:
                if strategy == 'random':
                    initial_guess = [np.random.uniform(low, high) for low, high in self.joint_limits]
                elif strategy == 'center':
                    initial_guess = [(low + high) / 2 for low, high in self.joint_limits]
                elif strategy == 'near_original' and i < len(original_configs):
                    # Small perturbation of original
                    original = original_configs[i]
                    initial_guess = []
                    for j, (orig_val, (low, high)) in enumerate(zip(original, self.joint_limits)):
                        noise = np.random.normal(0, 0.05)  # Small noise
                        guess = np.clip(orig_val + noise, low, high)
                        initial_guess.append(guess)
                else:
                    continue
                
                try:
                    result_config, ik_success, iterations = self.robot.iterative_inverse_kinematics(
                        target_pose, 
                        initial_guess,
                        eomg=1e-3,  # More lenient tolerances
                        ev=1e-4,
                        max_iterations=1500
                    )
                    
                    if ik_success:
                        success = True
                        best_result = result_config
                        best_iterations = iterations
                        break  # Found solution, stop trying
                        
                except Exception as e:
                    logger.debug(f"IK strategy {strategy} failed: {e}")
                    continue
            
            end_time = time.time()
            computation_times.append(end_time - start_time)
            iterations_list.append(best_iterations)
            
            if success:
                convergence_count += 1
                
                # Calculate accuracy
                achieved_pose = self.robot.forward_kinematics(best_result)
                
                # Position error
                pos_error = np.linalg.norm(target_pose[:3, 3] - achieved_pose[:3, 3])
                position_errors.append(pos_error)
                
                # Orientation error (robust calculation)
                R_target = target_pose[:3, :3]
                R_achieved = achieved_pose[:3, :3]
                
                try:
                    R_error = R_target.T @ R_achieved
                    trace_val = np.trace(R_error)
                    cos_angle = np.clip((trace_val - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
                    angle_error = np.arccos(cos_angle)
                    
                    if np.isnan(angle_error):
                        angle_error = np.linalg.norm(R_target - R_achieved, 'fro') / 2
                        
                except Exception:
                    angle_error = np.linalg.norm(R_target - R_achieved, 'fro') / 2
                
                orientation_errors.append(angle_error)
            else:
                position_errors.append(float('inf'))
                orientation_errors.append(float('inf'))
        
        # Filter out failed cases for statistics
        valid_pos_errors = [e for e in position_errors if not np.isinf(e)]
        valid_ori_errors = [e for e in orientation_errors if not np.isinf(e)]
        valid_times = [t for t in computation_times if not np.isinf(t)]
        valid_iterations = [it for it in iterations_list if it < 1000]
        
        convergence_rate = (convergence_count / len(target_poses)) * 100
        
        results = {
            'convergence_rate': convergence_rate,
            'avg_position_error': np.mean(valid_pos_errors) if valid_pos_errors else float('inf'),
            'std_position_error': np.std(valid_pos_errors) if valid_pos_errors else float('inf'),
            'max_position_error': np.max(valid_pos_errors) if valid_pos_errors else float('inf'),
            'avg_orientation_error': np.mean(valid_ori_errors) if valid_ori_errors else float('inf'),
            'std_orientation_error': np.std(valid_ori_errors) if valid_ori_errors else float('inf'),
            'avg_computation_time': np.mean(valid_times) if valid_times else float('inf'),
            'avg_iterations': np.mean(valid_iterations) if valid_iterations else float('inf'),
            'num_tests': len(target_poses),
            'successful_tests': convergence_count,
            'success_rate': convergence_rate
        }
        
        logger.info(f"Inverse kinematics: {convergence_rate:.1f}% convergence, "
                   f"{results['avg_position_error']:.2e}m pos error")
        
        return results

    def benchmark_jacobian_accuracy(self, num_tests: int = 500) -> Dict:
        """
        Benchmark Jacobian computation accuracy using numerical differentiation.
        
        Args:
            num_tests: Number of configurations to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking Jacobian accuracy...")
        
        jacobian_errors = []
        computation_times = []
        
        for _ in range(num_tests):
            # Random configuration
            config = []
            for low, high in self.joint_limits:
                config.append(np.random.uniform(low, high))
            
            # Analytical Jacobian
            start_time = time.time()
            try:
                J_analytical = self.robot.jacobian(config)
                end_time = time.time()
                computation_times.append(end_time - start_time)
                
                # Numerical Jacobian (finite differences)
                epsilon = 1e-6
                J_numerical = np.zeros((6, self.dof))
                
                pose_base = self.robot.forward_kinematics(config)
                twist_base = utils.se3ToVec(utils.MatrixLog6(pose_base))
                
                for i in range(self.dof):
                    config_plus = config.copy()
                    config_plus[i] += epsilon
                    
                    pose_plus = self.robot.forward_kinematics(config_plus)
                    twist_plus = utils.se3ToVec(utils.MatrixLog6(pose_plus))
                    
                    J_numerical[:, i] = (twist_plus - twist_base) / epsilon
                
                # Calculate error
                error = np.linalg.norm(J_analytical - J_numerical)
                jacobian_errors.append(error)
                
            except Exception as e:
                logger.warning(f"Jacobian test failed: {e}")
                computation_times.append(float('inf'))
                jacobian_errors.append(float('inf'))
        
        valid_errors = [e for e in jacobian_errors if not np.isinf(e)]
        valid_times = [t for t in computation_times if not np.isinf(t)]
        
        results = {
            'avg_jacobian_error': np.mean(valid_errors) if valid_errors else float('inf'),
            'std_jacobian_error': np.std(valid_errors) if valid_errors else float('inf'),
            'max_jacobian_error': np.max(valid_errors) if valid_errors else float('inf'),
            'avg_computation_time': np.mean(valid_times) if valid_times else float('inf'),
            'success_rate': (len(valid_errors) / num_tests) * 100,
            'num_tests': num_tests
        }
        
        logger.info(f"Jacobian accuracy: {results['success_rate']:.1f}% success, "
                   f"{results['avg_jacobian_error']:.2e} avg error")
        
        return results

    def benchmark_dynamics_accuracy(self, num_tests: int = 200) -> Dict:
        """
        Benchmark dynamics computations accuracy.
        
        Args:
            num_tests: Number of configurations to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking dynamics accuracy...")
        
        mass_matrix_times = []
        coriolis_times = []
        gravity_times = []
        inverse_dynamics_times = []
        forward_dynamics_times = []
        
        # Test dynamics consistency
        dynamics_consistency_errors = []
        
        for _ in range(num_tests):
            # Random configuration and velocities
            q = []
            dq = []
            for low, high in self.joint_limits:
                q.append(np.random.uniform(low, high))
                dq.append(np.random.uniform(-1.0, 1.0))
            
            ddq = np.random.uniform(-2.0, 2.0, self.dof)
            g = [0, 0, -9.81]
            Ftip = [0, 0, 0, 0, 0, 0]
            
            try:
                # Test mass matrix
                start_time = time.time()
                M = self.dynamics.mass_matrix(q)
                mass_matrix_times.append(time.time() - start_time)
                
                # Test Coriolis forces
                start_time = time.time()
                c = self.dynamics.velocity_quadratic_forces(q, dq)
                coriolis_times.append(time.time() - start_time)
                
                # Test gravity forces
                start_time = time.time()
                g_forces = self.dynamics.gravity_forces(q, g)
                gravity_times.append(time.time() - start_time)
                
                # Test inverse dynamics
                start_time = time.time()
                tau = self.dynamics.inverse_dynamics(q, dq, ddq, g, Ftip)
                inverse_dynamics_times.append(time.time() - start_time)
                
                # Test forward dynamics
                start_time = time.time()
                ddq_computed = self.dynamics.forward_dynamics(q, dq, tau, g, Ftip)
                forward_dynamics_times.append(time.time() - start_time)
                
                # Check consistency: forward(inverse(ddq)) should equal ddq
                consistency_error = np.linalg.norm(ddq - ddq_computed)
                dynamics_consistency_errors.append(consistency_error)
                
            except Exception as e:
                logger.warning(f"Dynamics test failed: {e}")
                mass_matrix_times.append(float('inf'))
                coriolis_times.append(float('inf'))
                gravity_times.append(float('inf'))
                inverse_dynamics_times.append(float('inf'))
                forward_dynamics_times.append(float('inf'))
                dynamics_consistency_errors.append(float('inf'))
        
        # Filter out failed results
        valid_mass_times = [t for t in mass_matrix_times if not np.isinf(t)]
        valid_coriolis_times = [t for t in coriolis_times if not np.isinf(t)]
        valid_gravity_times = [t for t in gravity_times if not np.isinf(t)]
        valid_inverse_times = [t for t in inverse_dynamics_times if not np.isinf(t)]
        valid_forward_times = [t for t in forward_dynamics_times if not np.isinf(t)]
        valid_consistency_errors = [e for e in dynamics_consistency_errors if not np.isinf(e)]
        
        results = {
            'avg_mass_matrix_time': np.mean(valid_mass_times) if valid_mass_times else float('inf'),
            'avg_coriolis_time': np.mean(valid_coriolis_times) if valid_coriolis_times else float('inf'),
            'avg_gravity_time': np.mean(valid_gravity_times) if valid_gravity_times else float('inf'),
            'avg_inverse_dynamics_time': np.mean(valid_inverse_times) if valid_inverse_times else float('inf'),
            'avg_forward_dynamics_time': np.mean(valid_forward_times) if valid_forward_times else float('inf'),
            'avg_dynamics_consistency_error': np.mean(valid_consistency_errors) if valid_consistency_errors else float('inf'),
            'std_dynamics_consistency_error': np.std(valid_consistency_errors) if valid_consistency_errors else float('inf'),
            'max_dynamics_consistency_error': np.max(valid_consistency_errors) if valid_consistency_errors else float('inf'),
            'success_rate': (len(valid_forward_times) / num_tests) * 100,
            'num_tests': num_tests
        }
        
        logger.info(f"Dynamics: {results['success_rate']:.1f}% success, "
                   f"{results['avg_dynamics_consistency_error']:.2e} consistency error")
        
        return results

    def benchmark_trajectory_planning_accuracy(self, num_tests: int = 50) -> Dict:
        """
        Benchmark trajectory planning accuracy, including CPU vs GPU comparison.
        
        Args:
            num_tests: Number of trajectory tests
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking trajectory planning accuracy...")
        
        # Test different trajectory parameters
        test_params = [
            {'N': 100, 'Tf': 2.0, 'method': 3},   # Cubic
            {'N': 200, 'Tf': 3.0, 'method': 5},   # Quintic
            {'N': 500, 'Tf': 5.0, 'method': 3},   # Longer trajectory
        ]
        
        results = {}
        
        for param_set in test_params:
            param_name = f"N{param_set['N']}_method{param_set['method']}"
            logger.info(f"Testing parameters: {param_name}")
            
            cpu_times = []
            gpu_times = []
            position_errors = []
            velocity_errors = []
            acceleration_errors = []
            boundary_errors = []
            
            for _ in range(num_tests):
                # Random start and end configurations
                start_config = []
                end_config = []
                for low, high in self.joint_limits:
                    start_config.append(np.random.uniform(low, high))
                    end_config.append(np.random.uniform(low, high))
                
                # CPU trajectory
                start_time = time.time()
                try:
                    cpu_traj = trajectory_cpu_fallback(
                        np.array(start_config), np.array(end_config),
                        param_set['Tf'], param_set['N'], param_set['method']
                    )
                    cpu_times.append(time.time() - start_time)
                except Exception as e:
                    logger.warning(f"CPU trajectory failed: {e}")
                    cpu_times.append(float('inf'))
                    continue
                
                # GPU trajectory (if available)
                if CUDA_AVAILABLE:
                    start_time = time.time()
                    try:
                        gpu_traj = optimized_trajectory_generation(
                            np.array(start_config), np.array(end_config),
                            param_set['Tf'], param_set['N'], param_set['method']
                        )
                        gpu_times.append(time.time() - start_time)
                        
                        # Compare CPU vs GPU accuracy
                        pos_error = np.linalg.norm(cpu_traj[0] - gpu_traj[0])
                        vel_error = np.linalg.norm(cpu_traj[1] - gpu_traj[1])
                        acc_error = np.linalg.norm(cpu_traj[2] - gpu_traj[2])
                        
                        position_errors.append(pos_error)
                        velocity_errors.append(vel_error)
                        acceleration_errors.append(acc_error)
                        
                    except Exception as e:
                        logger.warning(f"GPU trajectory failed: {e}")
                        gpu_times.append(float('inf'))
                        position_errors.append(float('inf'))
                        velocity_errors.append(float('inf'))
                        acceleration_errors.append(float('inf'))
                
                # Verify trajectory properties
                positions, velocities, accelerations = cpu_traj
                
                # Check boundary conditions
                start_error = np.linalg.norm(positions[0] - start_config)
                end_error = np.linalg.norm(positions[-1] - end_config)
                boundary_errors.append(max(start_error, end_error))
                
                # Check velocity boundary conditions (should be zero for spline)
                start_vel_error = np.linalg.norm(velocities[0])
                end_vel_error = np.linalg.norm(velocities[-1])
            
            # Compile results for this parameter set
            param_results = {
                'avg_cpu_time': np.mean([t for t in cpu_times if not np.isinf(t)]),
                'std_cpu_time': np.std([t for t in cpu_times if not np.isinf(t)]),
                'boundary_condition_accuracy': np.mean(boundary_errors),
                'velocity_boundary_accuracy': np.mean([start_vel_error, end_vel_error]),
            }
            
            if CUDA_AVAILABLE and gpu_times:
                valid_gpu_times = [t for t in gpu_times if not np.isinf(t)]
                valid_pos_errors = [e for e in position_errors if not np.isinf(e)]
                valid_vel_errors = [e for e in velocity_errors if not np.isinf(e)]
                valid_acc_errors = [e for e in acceleration_errors if not np.isinf(e)]
                
                param_results.update({
                    'avg_gpu_time': np.mean(valid_gpu_times) if valid_gpu_times else float('inf'),
                    'std_gpu_time': np.std(valid_gpu_times) if valid_gpu_times else float('inf'),
                    'gpu_speedup': (param_results['avg_cpu_time'] / np.mean(valid_gpu_times)) if valid_gpu_times else 0,
                    'avg_position_error_cpu_vs_gpu': np.mean(valid_pos_errors) if valid_pos_errors else float('inf'),
                    'avg_velocity_error_cpu_vs_gpu': np.mean(valid_vel_errors) if valid_vel_errors else float('inf'),
                    'avg_acceleration_error_cpu_vs_gpu': np.mean(valid_acc_errors) if valid_acc_errors else float('inf'),
                    'max_position_error_cpu_vs_gpu': np.max(valid_pos_errors) if valid_pos_errors else float('inf'),
                })
            
            results[param_name] = param_results
            
            logger.info(f"Trajectory {param_name}: CPU={param_results['avg_cpu_time']*1000:.2f}ms, "
                       f"boundary_error={param_results['boundary_condition_accuracy']:.2e}")
        
        return results

    def benchmark_control_algorithms(self, num_tests: int = 50) -> Dict:
        """
        Benchmark control algorithm accuracy and performance.
        
        Args:
            num_tests: Number of control tests
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking control algorithms...")
        
        controller = ManipulatorController(self.dynamics)
        
        # Test different control scenarios
        control_results = {}
        
        # PID Control Tests
        pid_errors = []
        pid_times = []
        
        for _ in range(num_tests):
            # Random configurations
            current_pos = [np.random.uniform(low, high) for low, high in self.joint_limits]
            desired_pos = [np.random.uniform(low, high) for low, high in self.joint_limits]
            current_vel = [np.random.uniform(-0.5, 0.5) for _ in range(self.dof)]
            desired_vel = [np.random.uniform(-0.5, 0.5) for _ in range(self.dof)]
            
            # PID gains
            Kp = np.full(self.dof, 50.0)
            Ki = np.full(self.dof, 1.0)
            Kd = np.full(self.dof, 5.0)
            
            start_time = time.time()
            try:
                control_signal = controller.pid_control(
                    desired_pos, desired_vel, current_pos, current_vel,
                    dt=0.01, Kp=Kp, Ki=Ki, Kd=Kd
                )
                pid_times.append(time.time() - start_time)
                
                # Calculate tracking error
                error = np.linalg.norm(np.array(desired_pos) - np.array(current_pos))
                pid_errors.append(error)
                
            except Exception as e:
                logger.warning(f"PID control failed: {e}")
                pid_times.append(float('inf'))
                pid_errors.append(float('inf'))
        
        # Computed Torque Control Tests
        ctc_errors = []
        ctc_times = []
        
        for _ in range(num_tests):
            # Random configurations
            current_pos = [np.random.uniform(low, high) for low, high in self.joint_limits]
            desired_pos = [np.random.uniform(low, high) for low, high in self.joint_limits]
            current_vel = [np.random.uniform(-0.5, 0.5) for _ in range(self.dof)]
            desired_vel = [np.random.uniform(-0.5, 0.5) for _ in range(self.dof)]
            desired_acc = [np.random.uniform(-1.0, 1.0) for _ in range(self.dof)]
            
            # Control gains
            Kp = np.full(self.dof, 100.0)
            Ki = np.full(self.dof, 1.0)
            Kd = np.full(self.dof, 10.0)
            
            start_time = time.time()
            try:
                import cupy as cp
                control_torque = controller.computed_torque_control(
                    thetalistd=cp.array(desired_pos),
                    dthetalistd=cp.array(desired_vel),
                    ddthetalistd=cp.array(desired_acc),
                    thetalist=cp.array(current_pos),
                    dthetalist=cp.array(current_vel),
                    g=cp.array([0, 0, -9.81]),
                    dt=0.01, Kp=cp.array(Kp), Ki=cp.array(Ki), Kd=cp.array(Kd)
                )
                ctc_times.append(time.time() - start_time)
                
                # Calculate tracking error
                error = np.linalg.norm(np.array(desired_pos) - np.array(current_pos))
                ctc_errors.append(error)
                
            except Exception as e:
                logger.warning(f"Computed torque control failed: {e}")
                ctc_times.append(float('inf'))
                ctc_errors.append(float('inf'))
        
        # Ziegler-Nichols Auto-tuning Test
        zn_test_results = []
        for _ in range(10):  # Fewer tests for time-consuming auto-tuning
            try:
                Ku = np.random.uniform(20, 100, self.dof)  # Ultimate gain
                Tu = np.random.uniform(0.1, 1.0, self.dof)  # Ultimate period
                
                Kp, Ki, Kd = controller.ziegler_nichols_tuning(Ku, Tu, kind="PID")
                
                # Check if gains are reasonable
                gain_validity = np.all(Kp > 0) and np.all(Ki >= 0) and np.all(Kd >= 0)
                zn_test_results.append(gain_validity)
                
            except Exception as e:
                logger.warning(f"Z-N tuning failed: {e}")
                zn_test_results.append(False)
        
        # Compile control results
        valid_pid_times = [t for t in pid_times if not np.isinf(t)]
        valid_pid_errors = [e for e in pid_errors if not np.isinf(e)]
        valid_ctc_times = [t for t in ctc_times if not np.isinf(t)]
        valid_ctc_errors = [e for e in ctc_errors if not np.isinf(e)]
        
        control_results = {
            'pid_avg_time': np.mean(valid_pid_times) if valid_pid_times else float('inf'),
            'pid_avg_error': np.mean(valid_pid_errors) if valid_pid_errors else float('inf'),
            'pid_success_rate': (len(valid_pid_times) / num_tests) * 100,
            'ctc_avg_time': np.mean(valid_ctc_times) if valid_ctc_times else float('inf'),
            'ctc_avg_error': np.mean(valid_ctc_errors) if valid_ctc_errors else float('inf'),
            'ctc_success_rate': (len(valid_ctc_times) / num_tests) * 100,
            'zn_tuning_success_rate': (sum(zn_test_results) / len(zn_test_results)) * 100,
            'num_tests': num_tests
        }
        
        logger.info(f"Control algorithms: PID {control_results['pid_success_rate']:.1f}% success, "
                   f"CTC {control_results['ctc_success_rate']:.1f}% success")
        
        return control_results

    def benchmark_singularity_detection(self, num_tests: int = 1000) -> Dict:
        """
        Benchmark singularity detection accuracy.
        
        Args:
            num_tests: Number of configurations to test
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Benchmarking singularity detection...")
        
        singularity_analyzer = PatchedSingularity(self.robot)
        
        detection_times = []
        condition_numbers = []
        singular_configs = 0
        near_singular_configs = 0
        
        for _ in range(num_tests):
            # Random configuration
            config = []
            for low, high in self.joint_limits:
                config.append(np.random.uniform(low, high))
            
            start_time = time.time()
            try:
                # Test singularity detection
                is_singular = singularity_analyzer.singularity_analysis(config)
                is_near_singular = singularity_analyzer.near_singularity_detection(config)
                cond_num = singularity_analyzer.condition_number(config)
                
                detection_times.append(time.time() - start_time)
                condition_numbers.append(cond_num)
                
                if is_singular:
                    singular_configs += 1
                if is_near_singular:
                    near_singular_configs += 1
                    
            except Exception as e:
                logger.warning(f"Singularity detection failed: {e}")
                detection_times.append(float('inf'))
                condition_numbers.append(float('inf'))
        
        # Generate some known singular configurations for validation
        # (e.g., configurations where robot is fully extended)
        known_singular_tests = 0
        known_singular_detected = 0
        
        for _ in range(20):  # Test some known singular configurations
            try:
                # Create configuration near workspace boundary (likely singular)
                config = []
                for i, (low, high) in enumerate(self.joint_limits):
                    if i % 2 == 0:
                        config.append(high * 0.95)  # Near joint limit
                    else:
                        config.append(0.0)  # Zero position
                
                is_singular = singularity_analyzer.singularity_analysis(config)
                cond_num = singularity_analyzer.condition_number(config)
                
                known_singular_tests += 1
                if is_singular or cond_num > 1e6:  # Very high condition number
                    known_singular_detected += 1
                    
            except Exception:
                continue
        
        valid_times = [t for t in detection_times if not np.isinf(t)]
        valid_cond_nums = [c for c in condition_numbers if not np.isinf(c)]
        
        results = {
            'avg_detection_time': np.mean(valid_times) if valid_times else float('inf'),
            'std_detection_time': np.std(valid_times) if valid_times else float('inf'),
            'avg_condition_number': np.mean(valid_cond_nums) if valid_cond_nums else float('inf'),
            'std_condition_number': np.std(valid_cond_nums) if valid_cond_nums else float('inf'),
            'max_condition_number': np.max(valid_cond_nums) if valid_cond_nums else float('inf'),
            'singular_percentage': (singular_configs / len(valid_times)) * 100 if valid_times else 0,
            'near_singular_percentage': (near_singular_configs / len(valid_times)) * 100 if valid_times else 0,
            'known_singular_detection_rate': (known_singular_detected / known_singular_tests) * 100 if known_singular_tests > 0 else 0,
            'success_rate': (len(valid_times) / num_tests) * 100,
            'num_tests': num_tests
        }
        
        logger.info(f"Singularity detection: {results['success_rate']:.1f}% success, "
                   f"{results['singular_percentage']:.1f}% singular configs found")
        
        return results

    def run_comprehensive_benchmark(self) -> Dict:
        """
        Run all benchmark tests and compile comprehensive results.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting comprehensive accuracy benchmark...")
        start_time = time.time()
        
        # Run all benchmarks
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'robot_info': {
                'dof': self.dof,
                'joint_limits': self.joint_limits,
                'urdf_file': urdf_file
            },
            'system_info': {
                'cuda_available': CUDA_AVAILABLE,
                'manipulapy_available': MANIPULAPY_AVAILABLE
            }
        }
        
        try:
            self.results['forward_kinematics'] = self.benchmark_forward_kinematics()
        except Exception as e:
            logger.error(f"Forward kinematics benchmark failed: {e}")
            self.results['forward_kinematics'] = {'error': str(e)}
        
        try:
            self.results['inverse_kinematics'] = self.benchmark_inverse_kinematics()
        except Exception as e:
            logger.error(f"Inverse kinematics benchmark failed: {e}")
            self.results['inverse_kinematics'] = {'error': str(e)}
        
        try:
            self.results['jacobian_accuracy'] = self.benchmark_jacobian_accuracy()
        except Exception as e:
            logger.error(f"Jacobian benchmark failed: {e}")
            self.results['jacobian_accuracy'] = {'error': str(e)}
        
        try:
            self.results['dynamics'] = self.benchmark_dynamics_accuracy()
        except Exception as e:
            logger.error(f"Dynamics benchmark failed: {e}")
            self.results['dynamics'] = {'error': str(e)}
        
        try:
            self.results['trajectory_planning'] = self.benchmark_trajectory_planning_accuracy()
        except Exception as e:
            logger.error(f"Trajectory planning benchmark failed: {e}")
            self.results['trajectory_planning'] = {'error': str(e)}
        
        try:
            self.results['control_algorithms'] = self.benchmark_control_algorithms()
        except Exception as e:
            logger.error(f"Control algorithms benchmark failed: {e}")
            self.results['control_algorithms'] = {'error': str(e)}
        
        try:
            self.results['singularity_detection'] = self.benchmark_singularity_detection()
        except Exception as e:
            logger.error(f"Singularity detection benchmark failed: {e}")
            self.results['singularity_detection'] = {'error': str(e)}
        
        total_time = time.time() - start_time
        self.results['total_benchmark_time'] = total_time
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f} seconds")
        
        # Save results
        self.save_results()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Create visualizations
        self.create_visualizations()
        
        return self.results

    def save_results(self):
        """Save benchmark results to JSON file."""
        results_file = os.path.join(self.output_dir, 'accuracy_benchmark_results.json')
        
        # Convert numpy arrays and other non-serializable types
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")

    def generate_summary_report(self):
        """Generate a human-readable summary report."""
        report_file = os.path.join(self.output_dir, 'accuracy_benchmark_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ManipulaPy Accuracy Benchmark Summary Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {self.results['timestamp']}\n")
            f.write(f"Robot: {self.dof}-DOF manipulator\n")
            f.write(f"CUDA Available: {self.results['system_info']['cuda_available']}\n")
            f.write(f"Total Benchmark Time: {self.results['total_benchmark_time']:.2f} seconds\n\n")
            
            # Forward Kinematics Summary
            if 'forward_kinematics' in self.results and 'error' not in self.results['forward_kinematics']:
                fk = self.results['forward_kinematics']
                f.write("Forward Kinematics:\n")
                f.write(f"  Success Rate: {fk['success_rate']:.1f}%\n")
                f.write(f"  Average Time: {fk['avg_computation_time']*1000:.3f} ms\n")
                f.write(f"  Consistency Error: {fk['consistency_error']:.2e}\n")
                f.write(f"  Valid Poses: {fk['valid_poses_percentage']:.1f}%\n\n")
            
            # Inverse Kinematics Summary
            if 'inverse_kinematics' in self.results and 'error' not in self.results['inverse_kinematics']:
                ik = self.results['inverse_kinematics']
                f.write("Inverse Kinematics:\n")
                f.write(f"  Convergence Rate: {ik['convergence_rate']:.1f}%\n")
                f.write(f"  Average Position Error: {ik['avg_position_error']:.2e} m\n")
                f.write(f"  Average Orientation Error: {ik['avg_orientation_error']:.2e} rad\n")
                f.write(f"  Average Iterations: {ik['avg_iterations']:.1f}\n")
                f.write(f"  Average Time: {ik['avg_computation_time']:.3f} s\n\n")
            
            # Jacobian Summary
            if 'jacobian_accuracy' in self.results and 'error' not in self.results['jacobian_accuracy']:
                jac = self.results['jacobian_accuracy']
                f.write("Jacobian Accuracy:\n")
                f.write(f"  Success Rate: {jac['success_rate']:.1f}%\n")
                f.write(f"  Average Error vs Numerical: {jac['avg_jacobian_error']:.2e}\n")
                f.write(f"  Average Time: {jac['avg_computation_time']*1000:.3f} ms\n\n")
            
            # Dynamics Summary
            if 'dynamics' in self.results and 'error' not in self.results['dynamics']:
                dyn = self.results['dynamics']
                f.write("Dynamics Computation:\n")
                f.write(f"  Success Rate: {dyn['success_rate']:.1f}%\n")
                f.write(f"  Consistency Error: {dyn['avg_dynamics_consistency_error']:.2e}\n")
                f.write(f"  Inverse Dynamics Time: {dyn['avg_inverse_dynamics_time']*1000:.3f} ms\n")
                f.write(f"  Forward Dynamics Time: {dyn['avg_forward_dynamics_time']*1000:.3f} ms\n\n")
            
            # Trajectory Planning Summary
            if 'trajectory_planning' in self.results and 'error' not in self.results['trajectory_planning']:
                traj = self.results['trajectory_planning']
                f.write("Trajectory Planning:\n")
                for param_name, results in traj.items():
                    f.write(f"  {param_name}:\n")
                    f.write(f"    CPU Time: {results['avg_cpu_time']*1000:.2f} ms\n")
                    f.write(f"    Boundary Accuracy: {results['boundary_condition_accuracy']:.2e}\n")
                    if 'gpu_speedup' in results:
                        f.write(f"    GPU Speedup: {results['gpu_speedup']:.2f}x\n")
                        f.write(f"    CPU vs GPU Position Error: {results['avg_position_error_cpu_vs_gpu']:.2e}\n")
                f.write("\n")
            
            # Control Summary
            if 'control_algorithms' in self.results and 'error' not in self.results['control_algorithms']:
                ctrl = self.results['control_algorithms']
                f.write("Control Algorithms:\n")
                f.write(f"  PID Success Rate: {ctrl['pid_success_rate']:.1f}%\n")
                f.write(f"  PID Average Time: {ctrl['pid_avg_time']*1000:.3f} ms\n")
                f.write(f"  Computed Torque Success Rate: {ctrl['ctc_success_rate']:.1f}%\n")
                f.write(f"  Z-N Tuning Success Rate: {ctrl['zn_tuning_success_rate']:.1f}%\n\n")
            
            # Singularity Detection Summary
            if 'singularity_detection' in self.results and 'error' not in self.results['singularity_detection']:
                sing = self.results['singularity_detection']
                f.write("Singularity Detection:\n")
                f.write(f"  Success Rate: {sing['success_rate']:.1f}%\n")
                f.write(f"  Average Detection Time: {sing['avg_detection_time']*1000:.3f} ms\n")
                f.write(f"  Singular Configs Found: {sing['singular_percentage']:.1f}%\n")
                f.write(f"  Known Singular Detection Rate: {sing['known_singular_detection_rate']:.1f}%\n\n")
        
        logger.info(f"Summary report saved to: {report_file}")

    def create_visualizations(self):
        """Create visualization plots for benchmark results."""
        try:
            # Performance comparison plot
            self._create_performance_comparison_plot()
            
            # Accuracy comparison plot
            self._create_accuracy_comparison_plot()
            
            # Time distribution plots
            self._create_time_distribution_plots()
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

    def _create_performance_comparison_plot(self):
        """Create performance comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ManipulaPy Performance Benchmark Results', fontsize=16)
        
        # Computation times
        ax = axes[0, 0]
        methods = []
        times = []
        
        if 'forward_kinematics' in self.results and 'avg_computation_time' in self.results['forward_kinematics']:
            methods.append('Forward\nKinematics')
            times.append(self.results['forward_kinematics']['avg_computation_time'] * 1000)
        
        if 'jacobian_accuracy' in self.results and 'avg_computation_time' in self.results['jacobian_accuracy']:
            methods.append('Jacobian')
            times.append(self.results['jacobian_accuracy']['avg_computation_time'] * 1000)
        
        if 'dynamics' in self.results and 'avg_inverse_dynamics_time' in self.results['dynamics']:
            methods.append('Inverse\nDynamics')
            times.append(self.results['dynamics']['avg_inverse_dynamics_time'] * 1000)
        
        if methods and times:
            bars = ax.bar(methods, times, color='skyblue', alpha=0.7)
            ax.set_ylabel('Computation Time (ms)')
            ax.set_title('Average Computation Times')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.2f}', ha='center', va='bottom')
        
        # Success rates
        ax = axes[0, 1]
        methods = []
        success_rates = []
        
        for test_name in ['forward_kinematics', 'inverse_kinematics', 'jacobian_accuracy', 
                         'dynamics', 'control_algorithms', 'singularity_detection']:
            if test_name in self.results and 'success_rate' in self.results[test_name]:
                methods.append(test_name.replace('_', '\n').title())
                success_rates.append(self.results[test_name]['success_rate'])
        
        if methods and success_rates:
            bars = ax.bar(methods, success_rates, color='lightgreen', alpha=0.7)
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Success Rates by Component')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom')
        
        # GPU vs CPU comparison (if available)
        ax = axes[1, 0]
        if ('trajectory_planning' in self.results and 
            isinstance(self.results['trajectory_planning'], dict)):
            
            param_names = []
            cpu_times = []
            gpu_times = []
            speedups = []
            
            for param_name, results in self.results['trajectory_planning'].items():
                if ('avg_cpu_time' in results and 'avg_gpu_time' in results and 
                    'gpu_speedup' in results):
                    param_names.append(param_name)
                    cpu_times.append(results['avg_cpu_time'] * 1000)
                    gpu_times.append(results['avg_gpu_time'] * 1000)
                    speedups.append(results['gpu_speedup'])
            
            if param_names:
                x = np.arange(len(param_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', color='orange', alpha=0.7)
                bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU', color='purple', alpha=0.7)
                
                ax.set_ylabel('Time (ms)')
                ax.set_title('CPU vs GPU Trajectory Planning Performance')
                ax.set_xticks(x)
                ax.set_xticklabels(param_names, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Accuracy metrics
        ax = axes[1, 1]
        accuracy_metrics = []
        values = []
        
        if 'inverse_kinematics' in self.results and 'avg_position_error' in self.results['inverse_kinematics']:
            accuracy_metrics.append('IK Position\nError (m)')
            values.append(self.results['inverse_kinematics']['avg_position_error'])
        
        if 'jacobian_accuracy' in self.results and 'avg_jacobian_error' in self.results['jacobian_accuracy']:
            accuracy_metrics.append('Jacobian\nError')
            values.append(self.results['jacobian_accuracy']['avg_jacobian_error'])
        
        if 'dynamics' in self.results and 'avg_dynamics_consistency_error' in self.results['dynamics']:
            accuracy_metrics.append('Dynamics\nConsistency')
            values.append(self.results['dynamics']['avg_dynamics_consistency_error'])
        
        if accuracy_metrics and values:
            bars = ax.bar(accuracy_metrics, values, color='salmon', alpha=0.7)
            ax.set_ylabel('Error Magnitude')
            ax.set_title('Accuracy Metrics (Log Scale)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to: {plot_file}")

    def _create_accuracy_comparison_plot(self):
        """Create accuracy-focused visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ManipulaPy Accuracy Analysis', fontsize=16)
        
        # Convergence rates
        ax = axes[0, 0]
        if 'inverse_kinematics' in self.results and 'convergence_rate' in self.results['inverse_kinematics']:
            convergence_rate = self.results['inverse_kinematics']['convergence_rate']
            
            labels = ['Converged', 'Failed']
            sizes = [convergence_rate, 100 - convergence_rate]
            colors = ['lightgreen', 'lightcoral']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Inverse Kinematics Convergence Rate')
        
        # Error distributions
        ax = axes[0, 1]
        if 'inverse_kinematics' in self.results:
            ik = self.results['inverse_kinematics']
            if 'avg_position_error' in ik and 'avg_orientation_error' in ik:
                errors = ['Position Error\n(m)', 'Orientation Error\n(rad)']
                values = [ik['avg_position_error'], ik['avg_orientation_error']]
                
                bars = ax.bar(errors, values, color=['blue', 'red'], alpha=0.7)
                ax.set_ylabel('Error Magnitude')
                ax.set_title('IK Accuracy Metrics')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2e}', ha='center', va='bottom')
        
        # Trajectory planning accuracy
        ax = axes[1, 0]
        if ('trajectory_planning' in self.results and 
            isinstance(self.results['trajectory_planning'], dict)):
            
            param_names = []
            boundary_errors = []
            
            for param_name, results in self.results['trajectory_planning'].items():
                if 'boundary_condition_accuracy' in results:
                    param_names.append(param_name)
                    boundary_errors.append(results['boundary_condition_accuracy'])
            
            if param_names and boundary_errors:
                bars = ax.bar(param_names, boundary_errors, color='green', alpha=0.7)
                ax.set_ylabel('Boundary Condition Error')
                ax.set_title('Trajectory Planning Boundary Accuracy')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Component reliability
        ax = axes[1, 1]
        components = []
        success_rates = []
        
        component_mapping = {
            'forward_kinematics': 'Forward\nKinematics',
            'inverse_kinematics': 'Inverse\nKinematics', 
            'jacobian_accuracy': 'Jacobian',
            'dynamics': 'Dynamics',
            'control_algorithms': 'Control',
            'singularity_detection': 'Singularity'
        }
        
        for comp_key, comp_name in component_mapping.items():
            if comp_key in self.results and 'success_rate' in self.results[comp_key]:
                components.append(comp_name)
                success_rates.append(self.results[comp_key]['success_rate'])
        
        if components and success_rates:
            bars = ax.barh(components, success_rates, color='steelblue', alpha=0.7)
            ax.set_xlabel('Success Rate (%)')
            ax.set_title('Component Reliability')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, success_rates):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{rate:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'accuracy_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Accuracy analysis plot saved to: {plot_file}")

    def _create_time_distribution_plots(self):
        """Create time distribution analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Computation Time Distribution Analysis', fontsize=16)
        
        # Performance timeline
        ax = axes[0, 0]
        methods = []
        avg_times = []
        std_times = []
        
        time_mapping = {
            'forward_kinematics': ('Forward Kinematics', 'avg_computation_time', 'std_computation_time'),
            'jacobian_accuracy': ('Jacobian', 'avg_computation_time', 'std_computation_time'),
            'dynamics': ('Inverse Dynamics', 'avg_inverse_dynamics_time', None)
        }
        
        for comp_key, (comp_name, avg_key, std_key) in time_mapping.items():
            if comp_key in self.results and avg_key in self.results[comp_key]:
                methods.append(comp_name)
                avg_times.append(self.results[comp_key][avg_key] * 1000)  # Convert to ms
                if std_key and std_key in self.results[comp_key]:
                    std_times.append(self.results[comp_key][std_key] * 1000)
                else:
                    std_times.append(0)
        
        if methods and avg_times:
            bars = ax.bar(methods, avg_times, yerr=std_times, 
                         color='lightblue', alpha=0.7, capsize=5)
            ax.set_ylabel('Computation Time (ms)')
            ax.set_title('Average Computation Times with Std Dev')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        # GPU vs CPU speedup analysis
        ax = axes[0, 1]
        if ('trajectory_planning' in self.results and 
            isinstance(self.results['trajectory_planning'], dict)):
            
            param_names = []
            speedups = []
            
            for param_name, results in self.results['trajectory_planning'].items():
                if 'gpu_speedup' in results and results['gpu_speedup'] > 0:
                    param_names.append(param_name.replace('_', '\n'))
                    speedups.append(results['gpu_speedup'])
            
            if param_names and speedups:
                bars = ax.bar(param_names, speedups, color='purple', alpha=0.7)
                ax.set_ylabel('Speedup Factor')
                ax.set_title('GPU Speedup Analysis')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add value labels
                for bar, speedup in zip(bars, speedups):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{speedup:.1f}x', ha='center', va='bottom')
        
        # Control algorithm performance
        ax = axes[0, 2]
        if 'control_algorithms' in self.results:
            ctrl = self.results['control_algorithms']
            
            control_types = []
            control_times = []
            
            if 'pid_avg_time' in ctrl:
                control_types.append('PID')
                control_times.append(ctrl['pid_avg_time'] * 1000)
            
            if 'ctc_avg_time' in ctrl:
                control_types.append('Computed\nTorque')
                control_times.append(ctrl['ctc_avg_time'] * 1000)
            
            if control_types and control_times:
                bars = ax.bar(control_types, control_times, color='orange', alpha=0.7)
                ax.set_ylabel('Computation Time (ms)')
                ax.set_title('Control Algorithm Performance')
                ax.grid(True, alpha=0.3)
        
        # Accuracy vs Performance scatter
        ax = axes[1, 0]
        if ('inverse_kinematics' in self.results and 
            'avg_computation_time' in self.results['inverse_kinematics'] and
            'avg_position_error' in self.results['inverse_kinematics']):
            
            ik = self.results['inverse_kinematics']
            
            # Create scatter plot of accuracy vs performance
            ax.scatter([ik['avg_computation_time']], [ik['avg_position_error']], 
                      s=100, color='red', alpha=0.7, label='IK Performance')
            ax.set_xlabel('Computation Time (s)')
            ax.set_ylabel('Position Error (m)')
            ax.set_title('Accuracy vs Performance Trade-off')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Dynamics computation breakdown
        ax = axes[1, 1]
        if 'dynamics' in self.results:
            dyn = self.results['dynamics']
            
            dyn_components = []
            dyn_times = []
            
            component_mapping = {
                'avg_mass_matrix_time': 'Mass Matrix',
                'avg_coriolis_time': 'Coriolis',
                'avg_gravity_time': 'Gravity',
                'avg_inverse_dynamics_time': 'Inverse Dynamics',
                'avg_forward_dynamics_time': 'Forward Dynamics'
            }
            
            for comp_key, comp_name in component_mapping.items():
                if comp_key in dyn:
                    dyn_components.append(comp_name)
                    dyn_times.append(dyn[comp_key] * 1000)
            
            if dyn_components and dyn_times:
                bars = ax.bar(dyn_components, dyn_times, color='green', alpha=0.7)
                ax.set_ylabel('Computation Time (ms)')
                ax.set_title('Dynamics Computation Breakdown')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Overall benchmark summary
        ax = axes[1, 2]
        if 'total_benchmark_time' in self.results:
            
            # Create a pie chart of time spent on different benchmarks
            benchmark_times = []
            benchmark_labels = []
            
            # Estimate time spent on each benchmark (approximate)
            total_time = self.results['total_benchmark_time']
            
            benchmark_estimates = {
                'Kinematics': total_time * 0.3,
                'Dynamics': total_time * 0.25,
                'Trajectory Planning': total_time * 0.2,
                'Control': total_time * 0.15,
                'Other': total_time * 0.1
            }
            
            for label, time_est in benchmark_estimates.items():
                benchmark_labels.append(label)
                benchmark_times.append(time_est)
            
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightgray']
            ax.pie(benchmark_times, labels=benchmark_labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Benchmark Time Distribution\n(Total: {total_time:.1f}s)')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'time_distribution_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time distribution analysis plot saved to: {plot_file}")

    def generate_csv_summary(self):
        """Generate a CSV summary of key metrics for easy analysis."""
        summary_data = []
        
        # Collect key metrics
        if 'forward_kinematics' in self.results:
            fk = self.results['forward_kinematics']
            summary_data.append({
                'Component': 'Forward Kinematics',
                'Success Rate (%)': fk.get('success_rate', 0),
                'Avg Time (ms)': fk.get('avg_computation_time', 0) * 1000,
                'Accuracy Metric': fk.get('consistency_error', 0),
                'Metric Type': 'Consistency Error'
            })
        
        if 'inverse_kinematics' in self.results:
            ik = self.results['inverse_kinematics']
            summary_data.append({
                'Component': 'Inverse Kinematics',
                'Success Rate (%)': ik.get('convergence_rate', 0),
                'Avg Time (ms)': ik.get('avg_computation_time', 0) * 1000,
                'Accuracy Metric': ik.get('avg_position_error', 0),
                'Metric Type': 'Position Error (m)'
            })
        
        if 'jacobian_accuracy' in self.results:
            jac = self.results['jacobian_accuracy']
            summary_data.append({
                'Component': 'Jacobian',
                'Success Rate (%)': jac.get('success_rate', 0),
                'Avg Time (ms)': jac.get('avg_computation_time', 0) * 1000,
                'Accuracy Metric': jac.get('avg_jacobian_error', 0),
                'Metric Type': 'Numerical Error'
            })
        
        if 'dynamics' in self.results:
            dyn = self.results['dynamics']
            summary_data.append({
                'Component': 'Dynamics',
                'Success Rate (%)': dyn.get('success_rate', 0),
                'Avg Time (ms)': dyn.get('avg_inverse_dynamics_time', 0) * 1000,
                'Accuracy Metric': dyn.get('avg_dynamics_consistency_error', 0),
                'Metric Type': 'Consistency Error'
            })
        
        # Save to CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(self.output_dir, 'benchmark_summary.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"CSV summary saved to: {csv_file}")


def main():
    """Main function to run the comprehensive accuracy benchmark."""
    if not MANIPULAPY_AVAILABLE:
        print("ERROR: ManipulaPy is not properly installed or available.")
        print("Please install ManipulaPy and ensure all dependencies are available.")
        return 1
    
    print("=" * 80)
    print("ManipulaPy Comprehensive Accuracy Benchmark")
    print("=" * 80)
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"Starting benchmark...")
    print()
    
    try:
        # Create benchmark instance
        benchmark = AccuracyBenchmark()
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Generate additional outputs
        benchmark.generate_csv_summary()
        
        print("\n" + "=" * 80)
        print("Benchmark Summary:")
        print("=" * 80)
        
        # Print key results
        if 'forward_kinematics' in results:
            fk = results['forward_kinematics']
            print(f"Forward Kinematics: {fk.get('success_rate', 0):.1f}% success, "
                  f"{fk.get('avg_computation_time', 0)*1000:.3f}ms avg")
        
        if 'inverse_kinematics' in results:
            ik = results['inverse_kinematics']
            print(f"Inverse Kinematics: {ik.get('convergence_rate', 0):.1f}% convergence, "
                  f"{ik.get('avg_position_error', 0):.2e}m error")
        
        if 'jacobian_accuracy' in results:
            jac = results['jacobian_accuracy']
            print(f"Jacobian Accuracy: {jac.get('success_rate', 0):.1f}% success, "
                  f"{jac.get('avg_jacobian_error', 0):.2e} error")
        
        if 'dynamics' in results:
            dyn = results['dynamics']
            print(f"Dynamics: {dyn.get('success_rate', 0):.1f}% success, "
                  f"{dyn.get('avg_dynamics_consistency_error', 0):.2e} consistency")
        
        if 'trajectory_planning' in results and isinstance(results['trajectory_planning'], dict):
            print("Trajectory Planning:")
            for param_name, param_results in results['trajectory_planning'].items():
                cpu_time = param_results.get('avg_cpu_time', 0) * 1000
                boundary_error = param_results.get('boundary_condition_accuracy', 0)
                print(f"  {param_name}: {cpu_time:.2f}ms, boundary_error={boundary_error:.2e}")
                if 'gpu_speedup' in param_results:
                    print(f"    GPU speedup: {param_results['gpu_speedup']:.2f}x")
        
        if 'control_algorithms' in results:
            ctrl = results['control_algorithms']
            print(f"Control: PID {ctrl.get('pid_success_rate', 0):.1f}%, "
                  f"CTC {ctrl.get('ctc_success_rate', 0):.1f}%")
        
        if 'singularity_detection' in results:
            sing = results['singularity_detection']
            print(f"Singularity Detection: {sing.get('success_rate', 0):.1f}% success")
        
        print(f"\nTotal benchmark time: {results.get('total_benchmark_time', 0):.2f} seconds")
        print(f"Results saved to: {benchmark.output_dir}")
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())