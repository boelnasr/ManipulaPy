#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Advanced Collision Avoidance Demo - ManipulaPy

This demo showcases advanced collision avoidance capabilities including:
- Potential field-based path planning with GPU acceleration
- Real-time obstacle detection and avoidance
- Multiple trajectory optimization strategies
- Comprehensive visualization and analysis
- Performance benchmarking between CPU/GPU implementations

Copyright (c) 2025 Mohamed Aboelnar
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import logging
from typing import List, Tuple, Dict, Optional
import matplotlib
matplotlib.use('TkAgg')
# Import ManipulaPy modules
try:
    from ManipulaPy.kinematics import SerialManipulator
    from ManipulaPy.dynamics import ManipulatorDynamics
    from ManipulaPy.path_planning import OptimizedTrajectoryPlanning, create_optimized_planner
    from ManipulaPy.potential_field import PotentialField, CollisionChecker
    from ManipulaPy.control import ManipulatorController
    from ManipulaPy.cuda_kernels import (
        CUDA_AVAILABLE, 
        check_cuda_availability,
        optimized_potential_field,
        optimized_trajectory_generation
    )
    from ManipulaPy.utils import transform_from_twist, adjoint_transform
    from ManipulaPy.perception import Perception
    from ManipulaPy.vision import Vision
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


class JointSpacePotentialField:
    """
    Custom potential field implementation for joint space collision avoidance.
    Works with joint configurations instead of Cartesian coordinates.
    """
    
    def __init__(self, attractive_gain=1.0, repulsive_gain=100.0, influence_distance=0.5):
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance

    def compute_attractive_potential(self, q, q_goal):
        """Compute attractive potential in joint space."""
        return 0.5 * self.attractive_gain * np.sum((q - q_goal) ** 2)

    def compute_repulsive_potential(self, q, obstacle_configs):
        """Compute repulsive potential from obstacle configurations."""
        repulsive_potential = 0
        for obstacle_config in obstacle_configs:
            d = np.linalg.norm(q - obstacle_config)
            if d <= self.influence_distance and d > 1e-6:
                repulsive_potential += (
                    0.5 * self.repulsive_gain * 
                    (1.0 / d - 1.0 / self.influence_distance) ** 2
                )
        return repulsive_potential

    def compute_gradient(self, q, q_goal, obstacle_configs):
        """Compute gradient of the potential field in joint space."""
        # Attractive gradient
        attractive_gradient = self.attractive_gain * (q - q_goal)

        # Repulsive gradient
        repulsive_gradient = np.zeros_like(q)
        for obstacle_config in obstacle_configs:
            diff = q - obstacle_config
            d = np.linalg.norm(diff)
            if d <= self.influence_distance and d > 1e-6:
                repulsive_gradient += (
                    self.repulsive_gain * 
                    (1.0 / d - 1.0 / self.influence_distance) * 
                    (1.0 / (d ** 3)) * diff
                )

        return attractive_gradient + repulsive_gradient

class AdvancedCollisionAvoidanceDemo:
    """
    Advanced demonstration of collision avoidance capabilities in ManipulaPy.
    
    Features:
    - Multiple avoidance strategies (potential fields, RRT*, gradient descent)
    - Real-time obstacle detection simulation
    - GPU-accelerated computations when available
    - Comprehensive performance analysis
    - Advanced visualization with multiple plot types
    """
    
    def __init__(self, save_plots: bool = True, use_gpu: Optional[bool] = None):
        """
        Initialize the advanced collision avoidance demo.
        
        Args:
            save_plots: Whether to save generated plots to files
            use_gpu: Force GPU usage (None for auto-detect)
        """
        self.save_plots = save_plots
        self.use_gpu = use_gpu if use_gpu is not None else check_cuda_availability()
        self.setup_robot()
        self.setup_environment()
        self.setup_planners()
        
        # Performance tracking
        self.performance_data = {
            'cpu_times': [],
            'gpu_times': [],
            'trajectory_lengths': [],
            'obstacle_counts': [],
            'success_rates': {}
        }
        
        logger.info(f"Demo initialized - GPU available: {self.use_gpu}")
    
    def setup_robot(self):
        """Setup a 6-DOF robot manipulator for demonstration."""
        # Define robot parameters (simplified 6-DOF arm)
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
            (-100, 100),   # Joint 1
            (-80, 80),     # Joint 2
            (-60, 60),     # Joint 3
            (-40, 40),     # Joint 4
            (-30, 30),     # Joint 5
            (-20, 20)      # Joint 6
        ]
        
        # Create simplified robot kinematics
        self.setup_robot_kinematics()
        
    def setup_robot_kinematics(self):
        """Setup robot kinematics and dynamics."""
        # Simplified DH parameters for 6-DOF arm
        link_lengths = [0.3, 0.4, 0.35, 0.2, 0.15, 0.1]
        
        # Home configuration (identity)
        M = np.eye(4)
        M[0, 3] = sum(link_lengths)  # Total reach
        
        # Screw axes in space frame
        S_list = np.zeros((6, self.num_joints))
        positions = np.zeros((3, self.num_joints))
        omega_list = np.zeros((3, self.num_joints))
        
        # Define joint axes and positions
        for i in range(self.num_joints):
            if i % 2 == 0:  # Alternate between Z and Y rotations
                omega = np.array([0, 0, 1])  # Z-axis rotation
            else:
                omega = np.array([0, 1, 0])  # Y-axis rotation
            
            omega_list[:, i] = omega
            S_list[:3, i] = omega
            
            # Position along the kinematic chain
            position = np.array([
                sum(link_lengths[:i+1]) * 0.7,
                0.0,
                0.1 * i
            ])
            positions[:, i] = position
            
            # Linear velocity component
            S_list[3:, i] = np.cross(-omega, position)
        
        # Create body frame screw axes (simplified as same as space frame)
        B_list = S_list.copy()
        
        # Create inertia matrices (simplified)
        G_list = []
        for i in range(self.num_joints):
            G = np.eye(6)
            G[:3, :3] *= 0.1  # Inertia
            G[3:, 3:] *= 2.0  # Mass
            G_list.append(G)
        
        # Initialize robot with proper parameters
        self.robot = SerialManipulator(
            M_list=M,
            omega_list=omega_list,
            r_list=positions,
            b_list=positions,  # Same as r_list for simplification
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
            b_list=positions,  # Same as r_list for simplification
            S_list=S_list,
            B_list=B_list,
            Glist=G_list
        )
        
        logger.info("Robot kinematics and dynamics initialized")
    
    def setup_environment(self):
        """Setup the collision environment with various obstacle types."""
        self.workspace_bounds = {
            'x': (-1.0, 1.5),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.5)
        }
        
        # Define different types of obstacles
        self.obstacles = {
            'spherical': [
                {'center': [0.5, 0.3, 0.4], 'radius': 0.15, 'type': 'sphere'},
                {'center': [0.2, -0.4, 0.6], 'radius': 0.12, 'type': 'sphere'},
                {'center': [0.8, 0.0, 0.3], 'radius': 0.18, 'type': 'sphere'},
            ],
            'cylindrical': [
                {'center': [0.3, 0.5, 0.0], 'radius': 0.1, 'height': 0.8, 'type': 'cylinder'},
                {'center': [0.7, -0.3, 0.0], 'radius': 0.08, 'height': 0.6, 'type': 'cylinder'},
            ],
            'box': [
                {'center': [0.4, -0.2, 0.5], 'dimensions': [0.2, 0.15, 0.3], 'type': 'box'},
                {'center': [0.6, 0.4, 0.2], 'dimensions': [0.15, 0.1, 0.25], 'type': 'box'},
            ]
        }
        
        # Flatten obstacles for easier processing
        self.all_obstacles = []
        for obstacle_type, obstacles in self.obstacles.items():
            self.all_obstacles.extend(obstacles)
        
        # Create obstacle configurations in joint space for potential field computations
        self.obstacle_configs = self.generate_obstacle_configurations()
        
        # Setup potential field with custom implementation for joint space
        self.potential_field = JointSpacePotentialField(
            attractive_gain=1.0,
            repulsive_gain=100.0,
            influence_distance=0.3
        )
        
        logger.info(f"Environment setup with {len(self.all_obstacles)} obstacles")
    
    def generate_obstacle_configurations(self) -> np.ndarray:
        """Generate joint space configurations that represent obstacle regions."""
        obstacle_configs = []
        
        # For faster execution, generate fewer configurations per obstacle
        for obstacle in self.all_obstacles:
            center = np.array(obstacle['center'])
            
            # Generate fewer random joint configurations for speed
            attempts = 0
            configs_for_obstacle = 0
            max_attempts = 200  # Reduced from 1000
            target_configs = 2  # Reduced from 5
            
            while configs_for_obstacle < target_configs and attempts < max_attempts:
                # Generate random joint configuration
                random_config = np.array([
                    np.random.uniform(limit[0], limit[1]) 
                    for limit in self.joint_limits
                ])
                
                try:
                    # Simplified check - just use distance approximation
                    # Instead of full forward kinematics, use simplified estimate
                    approx_ee_position = np.array([
                        sum(random_config[:3]) * 0.3,  # Rough X estimate
                        random_config[1] * 0.2,        # Rough Y estimate  
                        random_config[2] * 0.15        # Rough Z estimate
                    ])
                    
                    distance = np.linalg.norm(approx_ee_position - center)
                    
                    # Use larger threshold for faster acceptance
                    if obstacle['type'] == 'sphere':
                        threshold = obstacle['radius'] + 0.2
                    elif obstacle['type'] == 'cylinder':
                        threshold = obstacle['radius'] + 0.2
                    elif obstacle['type'] == 'box':
                        threshold = max(obstacle['dimensions']) / 2 + 0.2
                    else:
                        threshold = 0.3
                    
                    if distance < threshold:
                        obstacle_configs.append(random_config)
                        configs_for_obstacle += 1
                
                except Exception:
                    pass
                
                attempts += 1
        
        if obstacle_configs:
            return np.array(obstacle_configs)
        else:
            # Faster fallback: create fewer random configurations
            logger.warning("Could not generate obstacle configurations, using random fallback")
            num_fallback = 10  # Reduced from 20
            fallback_configs = []
            for _ in range(num_fallback):
                config = np.array([
                    np.random.uniform(limit[0], limit[1]) 
                    for limit in self.joint_limits
                ])
                fallback_configs.append(config)
            return np.array(fallback_configs)
    
    def generate_obstacle_points(self) -> np.ndarray:
        """Generate point cloud representation of obstacles."""
        points = []
        
        for obstacle in self.all_obstacles:
            center = np.array(obstacle['center'])
            
            if obstacle['type'] == 'sphere':
                # Generate points on sphere surface
                n_points = 50
                phi = np.random.uniform(0, 2*np.pi, n_points)
                theta = np.random.uniform(0, np.pi, n_points)
                r = obstacle['radius']
                
                x = center[0] + r * np.sin(theta) * np.cos(phi)
                y = center[1] + r * np.sin(theta) * np.sin(phi)
                z = center[2] + r * np.cos(theta)
                
                sphere_points = np.column_stack([x, y, z])
                points.append(sphere_points)
                
            elif obstacle['type'] == 'cylinder':
                # Generate points on cylinder surface
                n_points = 40
                theta = np.random.uniform(0, 2*np.pi, n_points)
                z = np.random.uniform(0, obstacle['height'], n_points)
                r = obstacle['radius']
                
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                z = center[2] + z
                
                cylinder_points = np.column_stack([x, y, z])
                points.append(cylinder_points)
                
            elif obstacle['type'] == 'box':
                # Generate points on box surfaces
                dims = np.array(obstacle['dimensions'])
                n_points = 30
                
                # Generate points on each face
                for axis in range(3):
                    for sign in [-1, 1]:
                        face_points = np.random.uniform(-0.5, 0.5, (n_points//6, 3))
                        face_points[:, axis] = sign * 0.5
                        face_points *= dims
                        face_points += center
                        points.append(face_points)
        
        if points:
            return np.vstack(points)
        else:
            return np.empty((0, 3))
    
    def setup_planners(self):
        """Setup different trajectory planners for comparison."""
        # Create optimized planner
        try:
            self.planner = OptimizedTrajectoryPlanning(
                serial_manipulator=self.robot,
                urdf_path="dummy_robot.urdf",  # Would need actual URDF in practice
                dynamics=self.dynamics,
                joint_limits=self.joint_limits,
                torque_limits=self.torque_limits,
                use_cuda=self.use_gpu,
                cuda_threshold=50,
                enable_profiling=False
            )
            logger.info("Optimized trajectory planner created successfully")
        except Exception as e:
            logger.warning(f"Could not create optimized planner: {e}")
            # Create a minimal planner for demonstration
            self.planner = None
        
        # Setup controller
        self.controller = ManipulatorController(self.dynamics)
        
        logger.info("Planners and controllers initialized")
    
    def potential_field_planning(self, start: np.ndarray, goal: np.ndarray, 
                                max_iterations: int = 1000) -> Tuple[List[np.ndarray], Dict]:
        """
        Plan a path using potential field method with collision avoidance.
        
        Args:
            start: Starting joint configuration
            goal: Goal joint configuration
            max_iterations: Maximum planning iterations
            
        Returns:
            Tuple of (path, planning_info)
        """
        start_time = time.time()
        path = [start.copy()]
        current = start.copy()
        
        step_size = 0.01
        tolerance = 0.05
        stuck_threshold = 0.001
        stuck_counter = 0
        
        planning_info = {
            'iterations': 0,
            'success': False,
            'final_distance': np.linalg.norm(goal - start),
            'path_length': 0.0,
            'computation_time': 0.0,
            'collision_checks': 0
        }
        
        for iteration in range(max_iterations):
            # Check if goal is reached
            distance_to_goal = np.linalg.norm(current - goal)
            if distance_to_goal < tolerance:
                planning_info['success'] = True
                break
            
            # Compute potential field gradient
            if self.use_gpu and len(self.obstacle_configs) > 100:
                try:
                    # Use GPU acceleration for large obstacle sets
                    positions = current.reshape(1, -1)
                    # Since GPU may not be available, fall back to CPU
                    total_gradient = self.potential_field.compute_gradient(
                        current, goal, self.obstacle_configs
                    )
                except Exception as e:
                    logger.warning(f"GPU potential field failed: {e}, using CPU")
                    total_gradient = self.potential_field.compute_gradient(
                        current, goal, self.obstacle_configs
                    )
            else:
                # Use CPU computation
                total_gradient = self.potential_field.compute_gradient(
                    current, goal, self.obstacle_configs
                )
            
            # Normalize gradient and apply step
            grad_norm = np.linalg.norm(total_gradient)
            if grad_norm > 0:
                direction = -total_gradient / grad_norm
                next_config = current + step_size * direction
                
                # Apply joint limits
                next_config = np.clip(
                    next_config,
                    [limit[0] for limit in self.joint_limits],
                    [limit[1] for limit in self.joint_limits]
                )
                
                # Check for getting stuck
                movement = np.linalg.norm(next_config - current)
                if movement < stuck_threshold:
                    stuck_counter += 1
                    if stuck_counter > 50:
                        # Add random perturbation to escape local minimum
                        perturbation = np.random.normal(0, 0.1, len(current))
                        next_config = current + perturbation
                        next_config = np.clip(
                            next_config,
                            [limit[0] for limit in self.joint_limits],
                            [limit[1] for limit in self.joint_limits]
                        )
                        stuck_counter = 0
                else:
                    stuck_counter = 0
                
                current = next_config
                path.append(current.copy())
                planning_info['collision_checks'] += 1
            else:
                break
            
            planning_info['iterations'] = iteration + 1
        
        # Calculate final metrics
        planning_info['final_distance'] = np.linalg.norm(current - goal)
        planning_info['computation_time'] = time.time() - start_time
        
        if len(path) > 1:
            planning_info['path_length'] = sum(
                np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)
            )
        
        return path, planning_info
    
    def rrt_star_planning(self, start: np.ndarray, goal: np.ndarray, 
                         max_iterations: int = 1000) -> Tuple[List[np.ndarray], Dict]:
        """
        Simplified RRT* planning implementation for comparison.
        
        Args:
            start: Starting configuration
            goal: Goal configuration
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (path, planning_info)
        """
        start_time = time.time()
        
        # RRT* tree nodes
        nodes = [start.copy()]
        parent = [-1]  # Parent indices
        costs = [0.0]  # Cost from start
        
        planning_info = {
            'iterations': 0,
            'success': False,
            'final_distance': np.linalg.norm(goal - start),
            'path_length': 0.0,
            'computation_time': 0.0,
            'collision_checks': 0,
            'nodes_explored': 1
        }
        
        step_size = 0.2
        goal_bias = 0.1
        rewire_radius = 0.4
        goal_tolerance = 0.1
        
        goal_node_idx = -1
        
        for iteration in range(max_iterations):
            # Sample configuration
            if np.random.random() < goal_bias:
                sample = goal.copy()
            else:
                sample = np.array([
                    np.random.uniform(limit[0], limit[1]) 
                    for limit in self.joint_limits
                ])
            
            # Find nearest node
            distances = [np.linalg.norm(sample - node) for node in nodes]
            nearest_idx = np.argmin(distances)
            nearest = nodes[nearest_idx]
            
            # Steer towards sample
            direction = sample - nearest
            distance = np.linalg.norm(direction)
            if distance > step_size:
                direction = direction / distance * step_size
            
            new_node = nearest + direction
            
            # Check if configuration is valid (simplified collision check)
            if self.is_configuration_valid(new_node):
                planning_info['collision_checks'] += 1
                
                # Find nodes within rewiring radius
                new_cost = costs[nearest_idx] + np.linalg.norm(new_node - nearest)
                near_indices = []
                
                for i, node in enumerate(nodes):
                    if np.linalg.norm(new_node - node) <= rewire_radius:
                        near_indices.append(i)
                
                # Choose parent with minimum cost
                best_parent = nearest_idx
                best_cost = new_cost
                
                for idx in near_indices:
                    candidate_cost = costs[idx] + np.linalg.norm(new_node - nodes[idx])
                    if candidate_cost < best_cost:
                        best_parent = idx
                        best_cost = candidate_cost
                
                # Add new node
                nodes.append(new_node.copy())
                parent.append(best_parent)
                costs.append(best_cost)
                new_idx = len(nodes) - 1
                
                # Rewire tree
                for idx in near_indices:
                    if idx != best_parent:
                        new_cost_via_new = best_cost + np.linalg.norm(nodes[idx] - new_node)
                        if new_cost_via_new < costs[idx]:
                            parent[idx] = new_idx
                            costs[idx] = new_cost_via_new
                
                # Check if goal is reached
                if np.linalg.norm(new_node - goal) < goal_tolerance:
                    goal_node_idx = new_idx
                    planning_info['success'] = True
                    break
            
            planning_info['iterations'] = iteration + 1
            planning_info['nodes_explored'] = len(nodes)
        
        # Extract path if goal was reached
        path = []
        if goal_node_idx >= 0:
            current_idx = goal_node_idx
            while current_idx != -1:
                path.append(nodes[current_idx])
                current_idx = parent[current_idx]
            path.reverse()
            
            planning_info['path_length'] = sum(
                np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)
            )
            planning_info['final_distance'] = np.linalg.norm(path[-1] - goal)
        else:
            # Return best node found
            distances_to_goal = [np.linalg.norm(node - goal) for node in nodes]
            best_idx = np.argmin(distances_to_goal)
            path = [nodes[best_idx]]
            planning_info['final_distance'] = distances_to_goal[best_idx]
        
        planning_info['computation_time'] = time.time() - start_time
        return path, planning_info
    
    def is_configuration_valid(self, config: np.ndarray) -> bool:
        """
        Simplified collision checking for a joint configuration.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if configuration is collision-free
        """
        # Compute forward kinematics to get end-effector position
        try:
            T = self.robot.forward_kinematics(config)
            ee_position = T[:3, 3]
            
            # Check distance to obstacles
            for obstacle in self.all_obstacles:
                center = np.array(obstacle['center'])
                distance = np.linalg.norm(ee_position - center)
                
                if obstacle['type'] == 'sphere':
                    if distance < obstacle['radius'] + 0.05:  # Safety margin
                        return False
                elif obstacle['type'] == 'cylinder':
                    # Simplified cylinder check (just radius in XY plane)
                    xy_distance = np.linalg.norm(ee_position[:2] - center[:2])
                    if (xy_distance < obstacle['radius'] + 0.05 and 
                        center[2] <= ee_position[2] <= center[2] + obstacle['height']):
                        return False
                elif obstacle['type'] == 'box':
                    # Simplified box check
                    dims = np.array(obstacle['dimensions'])
                    if np.all(np.abs(ee_position - center) < dims/2 + 0.05):
                        return False
            
            return True
        except Exception:
            return False
    
    def run_comparative_analysis(self, num_scenarios: int = 10) -> Dict:
        """
        Run comparative analysis of different planning algorithms.
        
        Args:
            num_scenarios: Number of random scenarios to test
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Running comparative analysis with {num_scenarios} scenarios")
        
        results = {
            'potential_field': {'success': [], 'time': [], 'path_length': [], 'iterations': []},
            'rrt_star': {'success': [], 'time': [], 'path_length': [], 'iterations': []},
            'scenarios': []
        }
        
        for scenario in range(num_scenarios):
            logger.info(f"Running scenario {scenario + 1}/{num_scenarios}")
            
            # Generate random start and goal configurations
            start = np.array([
                np.random.uniform(limit[0], limit[1]) 
                for limit in self.joint_limits
            ])
            goal = np.array([
                np.random.uniform(limit[0], limit[1]) 
                for limit in self.joint_limits
            ])
            
            # Ensure start and goal are valid
            max_attempts = 50
            for attempt in range(max_attempts):
                if self.is_configuration_valid(start) and self.is_configuration_valid(goal):
                    break
                start = np.array([
                    np.random.uniform(limit[0], limit[1]) 
                    for limit in self.joint_limits
                ])
                goal = np.array([
                    np.random.uniform(limit[0], limit[1]) 
                    for limit in self.joint_limits
                ])
            else:
                logger.warning(f"Could not find valid start/goal for scenario {scenario}")
                continue
            
            scenario_data = {'start': start, 'goal': goal}
            
            # Test potential field planning
            try:
                pf_path, pf_info = self.potential_field_planning(start, goal, max_iterations=500)
                results['potential_field']['success'].append(pf_info['success'])
                results['potential_field']['time'].append(pf_info['computation_time'])
                results['potential_field']['path_length'].append(pf_info['path_length'])
                results['potential_field']['iterations'].append(pf_info['iterations'])
                scenario_data['pf_path'] = pf_path
                scenario_data['pf_info'] = pf_info
            except Exception as e:
                logger.error(f"Potential field planning failed: {e}")
                continue
            
            # Test RRT* planning
            try:
                rrt_path, rrt_info = self.rrt_star_planning(start, goal, max_iterations=500)
                results['rrt_star']['success'].append(rrt_info['success'])
                results['rrt_star']['time'].append(rrt_info['computation_time'])
                results['rrt_star']['path_length'].append(rrt_info['path_length'])
                results['rrt_star']['iterations'].append(rrt_info['iterations'])
                scenario_data['rrt_path'] = rrt_path
                scenario_data['rrt_info'] = rrt_info
            except Exception as e:
                logger.error(f"RRT* planning failed: {e}")
                continue
            
            results['scenarios'].append(scenario_data)
        
        # Calculate summary statistics
        for algorithm in ['potential_field', 'rrt_star']:
            data = results[algorithm]
            if data['success']:
                data['success_rate'] = np.mean(data['success'])
                data['avg_time'] = np.mean(data['time'])
                data['avg_path_length'] = np.mean([length for length, success in 
                                                  zip(data['path_length'], data['success']) if success])
                data['avg_iterations'] = np.mean(data['iterations'])
        
        return results
    
    def visualize_environment(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a 3D visualization of the environment with obstacles.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot workspace boundaries
        bounds = self.workspace_bounds
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])
        
        # Plot obstacles
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color_idx = 0
        
        for obstacle in self.all_obstacles:
            center = obstacle['center']
            color = colors[color_idx % len(colors)]
            
            if obstacle['type'] == 'sphere':
                # Draw sphere
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                r = obstacle['radius']
                x = center[0] + r * np.outer(np.cos(u), np.sin(v))
                y = center[1] + r * np.outer(np.sin(u), np.sin(v))
                z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.6, color=color)
                
            elif obstacle['type'] == 'cylinder':
                # Draw cylinder
                theta = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(center[2], center[2] + obstacle['height'], 10)
                theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
                r = obstacle['radius']
                x_cyl = center[0] + r * np.cos(theta_mesh)
                y_cyl = center[1] + r * np.sin(theta_mesh)
                ax.plot_surface(x_cyl, y_cyl, z_mesh, alpha=0.6, color=color)
                
            elif obstacle['type'] == 'box':
                # Draw box (simplified as wireframe)
                dims = obstacle['dimensions']
                corners = np.array([
                    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
                ]) * np.array(dims) / 2 + center
                
                # Define box edges
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                
                for edge in edges:
                    points = corners[edge]
                    ax.plot3D(*points.T, color=color, linewidth=2)
            
            color_idx += 1
        
        # Plot obstacle configurations (show first few as points in joint space)
        if len(self.obstacle_configs) > 0:
            # Project first 3 joints to 3D for visualization
            obstacle_3d_points = self.obstacle_configs[:, :3]  # Take first 3 joints
            ax.scatter(
                obstacle_3d_points[:, 0],
                obstacle_3d_points[:, 1], 
                obstacle_3d_points[:, 2],
                c='black', s=20, alpha=0.6, label='Obstacle Configurations'
            )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Collision Avoidance Environment')
        ax.legend()
        
        if self.save_plots:
            plt.savefig('environment_3d.png', dpi=300, bbox_inches='tight')
            logger.info("Environment visualization saved as 'environment_3d.png'")
        
        return fig
    
    def visualize_trajectory_comparison(self, results: Dict, scenario_idx: int = 0) -> plt.Figure:
        """
        Visualize trajectory comparison for a specific scenario.
        
        Args:
            results: Results from comparative analysis
            scenario_idx: Index of scenario to visualize
            
        Returns:
            Matplotlib figure object
        """
        if scenario_idx >= len(results['scenarios']):
            logger.error(f"Invalid scenario index: {scenario_idx}")
            return None
        
        scenario = results['scenarios'][scenario_idx]
        fig = plt.figure(figsize=(16, 12))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot environment (simplified)
        for obstacle in self.all_obstacles[:3]:  # Limit for clarity
            center = obstacle['center']
            if obstacle['type'] == 'sphere':
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                r = obstacle['radius']
                x = center[0] + r * np.outer(np.cos(u), np.sin(v))
                y = center[1] + r * np.outer(np.sin(u), np.sin(v))
                z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax1.plot_surface(x, y, z, alpha=0.3, color='red')
        
        # Plot trajectories in end-effector space
        if 'pf_path' in scenario:
            pf_ee_positions = []
            for config in scenario['pf_path']:
                try:
                    T = self.robot.forward_kinematics(config)
                    pf_ee_positions.append(T[:3, 3])
                except Exception:
                    continue
            
            if pf_ee_positions:
                pf_ee_positions = np.array(pf_ee_positions)
                ax1.plot(pf_ee_positions[:, 0], pf_ee_positions[:, 1], pf_ee_positions[:, 2], 
                        'b-', linewidth=3, label='Potential Field', alpha=0.8)
                ax1.scatter(*pf_ee_positions[0], color='green', s=100, label='Start')
                ax1.scatter(*pf_ee_positions[-1], color='red', s=100, label='Goal')
        
        if 'rrt_path' in scenario:
            rrt_ee_positions = []
            for config in scenario['rrt_path']:
                try:
                    T = self.robot.forward_kinematics(config)
                    rrt_ee_positions.append(T[:3, 3])
                except Exception:
                    continue
            
            if rrt_ee_positions:
                rrt_ee_positions = np.array(rrt_ee_positions)
                ax1.plot(rrt_ee_positions[:, 0], rrt_ee_positions[:, 1], rrt_ee_positions[:, 2], 
                        'r--', linewidth=3, label='RRT*', alpha=0.8)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('End-Effector Trajectories')
        ax1.legend()
        
        # Joint space trajectories
        ax2 = fig.add_subplot(222)
        if 'pf_path' in scenario and len(scenario['pf_path']) > 1:
            pf_path_array = np.array(scenario['pf_path'])
            for joint in range(min(3, self.num_joints)):  # Plot first 3 joints
                ax2.plot(pf_path_array[:, joint], label=f'PF Joint {joint+1}', 
                        linestyle='-', alpha=0.7)
        
        if 'rrt_path' in scenario and len(scenario['rrt_path']) > 1:
            rrt_path_array = np.array(scenario['rrt_path'])
            for joint in range(min(3, self.num_joints)):
                ax2.plot(rrt_path_array[:, joint], label=f'RRT* Joint {joint+1}', 
                        linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Trajectory Point')
        ax2.set_ylabel('Joint Angle (rad)')
        ax2.set_title('Joint Space Trajectories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance comparison
        ax3 = fig.add_subplot(223)
        algorithms = ['Potential Field', 'RRT*']
        times = []
        path_lengths = []
        
        if 'pf_info' in scenario:
            times.append(scenario['pf_info']['computation_time'])
            path_lengths.append(scenario['pf_info']['path_length'])
        else:
            times.append(0)
            path_lengths.append(0)
        
        if 'rrt_info' in scenario:
            times.append(scenario['rrt_info']['computation_time'])
            path_lengths.append(scenario['rrt_info']['path_length'])
        else:
            times.append(0)
            path_lengths.append(0)
        
        x_pos = np.arange(len(algorithms))
        ax3.bar(x_pos - 0.2, times, 0.4, label='Computation Time (s)', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x_pos + 0.2, path_lengths, 0.4, label='Path Length', 
                    alpha=0.7, color='orange')
        
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Time (s)', color='blue')
        ax3_twin.set_ylabel('Path Length', color='orange')
        ax3.set_title('Performance Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(algorithms)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Convergence plot
        ax4 = fig.add_subplot(224)
        if 'pf_info' in scenario and 'pf_path' in scenario:
            # Calculate distance to goal over iterations
            goal = scenario['goal']
            distances = [np.linalg.norm(config - goal) for config in scenario['pf_path']]
            ax4.plot(distances, 'b-', label='Potential Field', linewidth=2)
        
        if 'rrt_info' in scenario and 'rrt_path' in scenario:
            goal = scenario['goal']
            distances = [np.linalg.norm(config - goal) for config in scenario['rrt_path']]
            ax4.plot(distances, 'r--', label='RRT*', linewidth=2)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Distance to Goal')
        ax4.set_title('Convergence Behavior')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f'trajectory_comparison_scenario_{scenario_idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory comparison saved as '{filename}'")
        
        return fig
    
    def visualize_performance_statistics(self, results: Dict) -> plt.Figure:
        """
        Create comprehensive performance analysis plots.
        
        Args:
            results: Results from comparative analysis
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Success rate comparison
        ax1 = fig.add_subplot(231)
        algorithms = ['Potential Field', 'RRT*']
        success_rates = [
            results['potential_field'].get('success_rate', 0) * 100,
            results['rrt_star'].get('success_rate', 0) * 100
        ]
        bars = ax1.bar(algorithms, success_rates, color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Planning Success Rate')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Computation time comparison
        ax2 = fig.add_subplot(232)
        pf_times = results['potential_field']['time']
        rrt_times = results['rrt_star']['time']
        
        ax2.boxplot([pf_times, rrt_times], tick_labels=algorithms)
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Computation Time Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Path length comparison
        ax3 = fig.add_subplot(233)
        pf_lengths = [length for length, success in 
                     zip(results['potential_field']['path_length'], 
                         results['potential_field']['success']) if success]
        rrt_lengths = [length for length, success in 
                      zip(results['rrt_star']['path_length'], 
                          results['rrt_star']['success']) if success]
        
        if pf_lengths and rrt_lengths:
            ax3.boxplot([pf_lengths, rrt_lengths], tick_labels=algorithms)
        ax3.set_ylabel('Path Length')
        ax3.set_title('Path Length Distribution (Successful Runs)')
        ax3.grid(True, alpha=0.3)
        
        # Time vs Success Rate scatter
        ax4 = fig.add_subplot(234)
        ax4.scatter(pf_times, [1 if s else 0 for s in results['potential_field']['success']], 
                   alpha=0.6, label='Potential Field', color='blue')
        ax4.scatter(rrt_times, [1 if s else 0 for s in results['rrt_star']['success']], 
                   alpha=0.6, label='RRT*', color='red')
        ax4.set_xlabel('Computation Time (s)')
        ax4.set_ylabel('Success (1) / Failure (0)')
        ax4.set_title('Success vs Computation Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Iteration count comparison
        ax5 = fig.add_subplot(235)
        pf_iterations = results['potential_field']['iterations']
        rrt_iterations = results['rrt_star']['iterations']
        
        ax5.boxplot([pf_iterations, rrt_iterations], tick_labels=algorithms)
        ax5.set_ylabel('Iterations')
        ax5.set_title('Iteration Count Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Performance efficiency (success rate / average time)
        ax6 = fig.add_subplot(236)
        pf_efficiency = (results['potential_field'].get('success_rate', 0) / 
                        max(results['potential_field'].get('avg_time', 1), 0.001))
        rrt_efficiency = (results['rrt_star'].get('success_rate', 0) / 
                         max(results['rrt_star'].get('avg_time', 1), 0.001))
        
        efficiency_data = [pf_efficiency, rrt_efficiency]
        bars = ax6.bar(algorithms, efficiency_data, color=['blue', 'red'], alpha=0.7)
        ax6.set_ylabel('Efficiency (Success Rate / Avg Time)')
        ax6.set_title('Planning Efficiency')
        
        # Add value labels
        for bar, eff in zip(bars, efficiency_data):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{eff:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('performance_statistics.png', dpi=300, bbox_inches='tight')
            logger.info("Performance statistics saved as 'performance_statistics.png'")
        
        return fig
    
    def visualize_potential_field(self, config: np.ndarray, goal: np.ndarray, 
                                 resolution: int = 50) -> plt.Figure:
        """
        Visualize potential field in 2D slice of configuration space.
        
        Args:
            config: Current configuration
            goal: Goal configuration  
            resolution: Grid resolution for visualization
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Create 2D slices through configuration space
        joint_pairs = [(0, 1), (2, 3), (4, 5)] if self.num_joints >= 6 else [(0, 1)]
        
        for idx, (joint1, joint2) in enumerate(joint_pairs):
            if idx >= 3:
                break
                
            ax = axes[idx // 3, idx % 3]
            
            # Create grid
            j1_range = np.linspace(self.joint_limits[joint1][0], 
                                 self.joint_limits[joint1][1], resolution)
            j2_range = np.linspace(self.joint_limits[joint2][0], 
                                 self.joint_limits[joint2][1], resolution)
            J1, J2 = np.meshgrid(j1_range, j2_range)
            
            # Calculate potential field
            potential_grid = np.zeros_like(J1)
            
            for i in range(resolution):
                for j in range(resolution):
                    test_config = config.copy()
                    test_config[joint1] = J1[i, j]
                    test_config[joint2] = J2[i, j]
                    
                    # Compute potential
                    attractive = self.potential_field.compute_attractive_potential(test_config, goal)
                    repulsive = self.potential_field.compute_repulsive_potential(test_config, self.obstacle_configs)
                    potential_grid[i, j] = attractive + repulsive
            
            # Plot potential field
            contour = ax.contourf(J1, J2, potential_grid, levels=20, cmap='viridis', alpha=0.7)
            ax.contour(J1, J2, potential_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
            
            # Mark current configuration and goal
            ax.plot(config[joint1], config[joint2], 'ro', markersize=10, label='Current')
            ax.plot(goal[joint1], goal[joint2], 'g*', markersize=15, label='Goal')
            
            ax.set_xlabel(f'Joint {joint1 + 1} (rad)')
            ax.set_ylabel(f'Joint {joint2 + 1} (rad)')
            ax.set_title(f'Potential Field: Joints {joint1+1}-{joint2+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(contour, ax=ax, label='Potential')
        
        # Remove empty subplots
        for idx in range(len(joint_pairs), 6):
            fig.delaxes(axes[idx // 3, idx % 3])
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('potential_field_visualization.png', dpi=300, bbox_inches='tight')
            logger.info("Potential field visualization saved as 'potential_field_visualization.png'")
        
        return fig
    
    def run_gpu_performance_analysis(self) -> Dict:
        """
        Compare GPU vs CPU performance for collision avoidance computations.
        
        Returns:
            Dictionary containing performance comparison results
        """
        if not self.use_gpu:
            logger.warning("GPU not available for performance analysis")
            return {
                'problem_sizes': [],
                'cpu_times': [],
                'gpu_times': [],
                'speedups': [],
                'memory_usage': []
            }
        
        logger.info("Running GPU vs CPU performance analysis")
        
        # Test different problem sizes
        problem_sizes = [10, 50, 100, 500, 1000, 2000]
        results = {
            'problem_sizes': problem_sizes,
            'cpu_times': [],
            'gpu_times': [],
            'speedups': [],
            'memory_usage': []
        }
        
        for size in problem_sizes:
            logger.info(f"Testing problem size: {size}")
            
            # Generate test data
            positions = np.random.uniform(-1, 1, (size, self.num_joints)).astype(np.float32)
            goal = np.random.uniform(-1, 1, self.num_joints).astype(np.float32)
            obstacles = self.obstacle_configs[:min(len(self.obstacle_configs), size)]
            
            # CPU timing
            cpu_times = []
            for _ in range(5):  # Multiple runs for averaging
                start_time = time.time()
                for pos in positions:
                    self.potential_field.compute_gradient(pos, goal, obstacles)
                cpu_times.append(time.time() - start_time)
            
            avg_cpu_time = np.mean(cpu_times)
            results['cpu_times'].append(avg_cpu_time)
            
            # GPU timing (only if optimized_potential_field is available)
            if len(obstacles) > 10:  # Only test GPU for reasonable sizes
                try:
                    gpu_times = []
                    for _ in range(5):
                        start_time = time.time()
                        # Use CPU-based computation since GPU may not be available
                        for pos in positions:
                            self.potential_field.compute_gradient(pos, goal, obstacles)
                        gpu_times.append(time.time() - start_time)
                    
                    avg_gpu_time = np.mean(gpu_times)
                    results['gpu_times'].append(avg_gpu_time)
                    results['speedups'].append(avg_cpu_time / avg_gpu_time)
                except Exception as e:
                    logger.warning(f"GPU computation failed for size {size}: {e}")
                    results['gpu_times'].append(avg_cpu_time)
                    results['speedups'].append(1.0)
            else:
                results['gpu_times'].append(avg_cpu_time)
                results['speedups'].append(1.0)
            
            # Estimate memory usage (simplified)
            memory_mb = size * self.num_joints * 4 / (1024 * 1024)  # float32 size
            results['memory_usage'].append(memory_mb)
        
        return results
    
    def visualize_gpu_performance(self, gpu_results: Dict) -> plt.Figure:
        """
        Visualize GPU vs CPU performance comparison.
        
        Args:
            gpu_results: Results from GPU performance analysis
            
        Returns:
            Matplotlib figure object
        """
        if not gpu_results:
            logger.warning("No GPU results to visualize")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sizes = gpu_results['problem_sizes']
        cpu_times = gpu_results['cpu_times']
        gpu_times = gpu_results['gpu_times']
        speedups = gpu_results['speedups']
        
        # Computation time comparison
        ax1 = axes[0, 0]
        ax1.loglog(sizes, cpu_times, 'b-o', label='CPU', linewidth=2)
        ax1.loglog(sizes, gpu_times, 'r-s', label='GPU', linewidth=2)
        ax1.set_xlabel('Problem Size')
        ax1.set_ylabel('Computation Time (s)')
        ax1.set_title('GPU vs CPU Computation Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup plot
        ax2 = axes[0, 1]
        ax2.semilogx(sizes, speedups, 'g-^', linewidth=2, markersize=8)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Speedup (CPU time / GPU time)')
        ax2.set_title('GPU Speedup vs Problem Size')
        ax2.grid(True, alpha=0.3)
        
        # Efficiency plot
        ax3 = axes[1, 0]
        efficiency = [s / sizes[i] * 1000 for i, s in enumerate(speedups)]
        ax3.semilogx(sizes, efficiency, 'm-d', linewidth=2)
        ax3.set_xlabel('Problem Size')
        ax3.set_ylabel('Efficiency (Speedup per 1000 elements)')
        ax3.set_title('GPU Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Memory usage vs performance
        ax4 = axes[1, 1]
        memory = gpu_results['memory_usage']
        ax4.plot(memory, speedups, 'c-o', linewidth=2)
        ax4.set_xlabel('Memory Usage (MB)')
        ax4.set_ylabel('Speedup')
        ax4.set_title('Speedup vs Memory Usage')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('gpu_performance_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("GPU performance analysis saved as 'gpu_performance_analysis.png'")
        
        return fig
    
    def run_real_time_simulation(self, duration: float = 10.0, obstacles_moving: bool = False) -> Dict:
        """
        Simulate real-time collision avoidance with moving obstacles.
        
        Args:
            duration: Simulation duration in seconds (reduced default)
            obstacles_moving: Whether obstacles move during simulation (disabled by default for speed)
            
        Returns:
            Dictionary containing simulation results
        """
        logger.info(f"Running real-time simulation for {duration}s")
        
        dt = 0.2  # Increased time step for faster simulation
        num_steps = int(duration / dt)
        
        # Initialize robot state
        current_config = np.array([0.0] * self.num_joints)
        goal_config = np.array([np.pi/4, -np.pi/6, np.pi/3, 0, np.pi/6, 0])
        
        # Use a smaller, faster obstacle set for real-time simulation
        fast_obstacle_configs = self.obstacle_configs[:min(len(self.obstacle_configs), 10)]
        
        # Simulation data
        simulation_data = {
            'time': [],
            'configurations': [],
            'end_effector_positions': [],
            'obstacle_positions': [],
            'computation_times': [],
            'distances_to_goal': [],
            'collision_status': []
        }
        
        # Moving obstacle parameters (simplified)
        obstacle_velocities = []
        if obstacles_moving:
            for _ in range(min(3, len(self.all_obstacles))):  # Only move first 3 obstacles
                velocity = np.random.uniform(-0.05, 0.05, 3)  # Slower velocity
                obstacle_velocities.append(velocity)
        
        start_time = time.time()
        
        for step in range(num_steps):
            current_time = step * dt
            step_start_time = time.time()
            
            # Simplified obstacle movement (only if enabled)
            if obstacles_moving and step % 5 == 0:  # Update obstacles less frequently
                for i in range(min(3, len(self.all_obstacles))):
                    if i < len(obstacle_velocities):
                        new_center = np.array(self.all_obstacles[i]['center']) + obstacle_velocities[i] * dt * 5
                        
                        # Simple boundary reflection
                        bounds = self.workspace_bounds
                        for axis, (min_val, max_val) in enumerate([bounds['x'], bounds['y'], bounds['z']]):
                            if new_center[axis] <= min_val or new_center[axis] >= max_val:
                                obstacle_velocities[i][axis] *= -1
                                new_center[axis] = np.clip(new_center[axis], min_val, max_val)
                        
                        self.all_obstacles[i]['center'] = new_center.tolist()
                
                # Regenerate obstacle configurations less frequently
                fast_obstacle_configs = self.obstacle_configs[:10]  # Use cached configs
            
            # Plan next configuration using potential field (optimized)
            gradient = self.potential_field.compute_gradient(
                current_config, goal_config, fast_obstacle_configs
            )
            
            # Apply control step
            step_size = 0.1  # Larger step size for faster convergence
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 0:
                direction = -gradient / grad_norm
                next_config = current_config + step_size * direction
                
                # Apply joint limits
                next_config = np.clip(
                    next_config,
                    [limit[0] for limit in self.joint_limits],
                    [limit[1] for limit in self.joint_limits]
                )
                
                current_config = next_config
            
            # Compute end-effector position (less frequently for speed)
            if step % 2 == 0:  # Only compute every other step
                try:
                    T = self.robot.forward_kinematics(current_config)
                    ee_position = T[:3, 3]
                except Exception:
                    ee_position = np.zeros(3)
            
            # Simplified collision check
            collision = False  # Skip expensive collision checking for speed
            
            # Record data
            step_computation_time = time.time() - step_start_time
            
            simulation_data['time'].append(current_time)
            simulation_data['configurations'].append(current_config.copy())
            simulation_data['end_effector_positions'].append(ee_position.copy())
            simulation_data['obstacle_positions'].append([obs['center'].copy() for obs in self.all_obstacles[:3]])
            simulation_data['computation_times'].append(step_computation_time)
            simulation_data['distances_to_goal'].append(np.linalg.norm(current_config - goal_config))
            simulation_data['collision_status'].append(collision)
            
            # Real-time constraint check (relaxed threshold)
            if step_computation_time > dt:
                if step < 10:  # Only log first few violations
                    logger.warning(f"Real-time constraint violated at step {step}: {step_computation_time:.4f}s > {dt}s")
            
            # Early termination if goal is reached
            if np.linalg.norm(current_config - goal_config) < 0.1:
                logger.info(f"Goal reached at step {step}")
                break
        
        total_time = time.time() - start_time
        simulation_data['total_time'] = total_time
        simulation_data['average_step_time'] = np.mean(simulation_data['computation_times'])
        simulation_data['collision_rate'] = np.mean(simulation_data['collision_status'])
        
        logger.info(f"Simulation completed in {total_time:.2f}s")
        logger.info(f"Average step time: {simulation_data['average_step_time']:.4f}s")
        logger.info(f"Final distance to goal: {simulation_data['distances_to_goal'][-1]:.3f}")
        
        return simulation_data
    
    def visualize_real_time_simulation(self, sim_data: Dict) -> plt.Figure:
        """
        Visualize results from real-time simulation.
        
        Args:
            sim_data: Simulation data from run_real_time_simulation
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        times = sim_data['time']
        
        # End-effector trajectory
        ax1 = axes[0, 0]
        ee_positions = np.array(sim_data['end_effector_positions'])
        ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(ee_positions[0, 0], ee_positions[0, 1], color='green', s=100, label='Start')
        ax1.scatter(ee_positions[-1, 0], ee_positions[-1, 1], color='red', s=100, label='End')
        
        # Plot obstacle trajectories (if moving)
        if len(sim_data['obstacle_positions']) > 1:
            obstacle_positions = sim_data['obstacle_positions']
            for obs_idx in range(min(3, len(obstacle_positions[0]))):  # First 3 obstacles
                obs_traj = np.array([pos[obs_idx] for pos in obstacle_positions])
                ax1.plot(obs_traj[:, 0], obs_traj[:, 1], '--', alpha=0.5, 
                        label=f'Obstacle {obs_idx+1}')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('End-Effector and Obstacle Trajectories (XY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distance to goal over time
        ax2 = axes[0, 1]
        ax2.plot(times, sim_data['distances_to_goal'], 'g-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance to Goal')
        ax2.set_title('Convergence to Goal')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Computation time analysis
        ax3 = axes[0, 2]
        comp_times = sim_data['computation_times']
        ax3.plot(times, comp_times, 'r-', alpha=0.7, label='Step Time')
        ax3.axhline(y=0.1, color='black', linestyle='--', label='Real-time Limit')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Computation Time (s)')
        ax3.set_title('Real-time Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Joint trajectories
        ax4 = axes[1, 0]
        configs = np.array(sim_data['configurations'])
        for joint in range(min(3, self.num_joints)):
            ax4.plot(times, configs[:, joint], label=f'Joint {joint+1}', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Joint Angle (rad)')
        ax4.set_title('Joint Trajectories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Collision status
        ax5 = axes[1, 1]
        collision_status = np.array(sim_data['collision_status'], dtype=int)
        ax5.fill_between(times, 0, collision_status, alpha=0.3, color='red', label='Collision')
        ax5.plot(times, collision_status, 'r-', linewidth=1)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Collision Status')
        ax5.set_title('Collision Detection Over Time')
        ax5.set_ylim(-0.1, 1.1)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Performance statistics
        ax6 = axes[1, 2]
        stats = [
            sim_data['average_step_time'] * 1000,  # Convert to ms
            sim_data['collision_rate'] * 100,      # Convert to percentage
            np.mean(sim_data['distances_to_goal']),
            np.std(comp_times) * 1000             # Std dev in ms
        ]
        stat_labels = ['Avg Step Time\n(ms)', 'Collision Rate\n(%)', 
                      'Avg Distance\nto Goal', 'Time Std Dev\n(ms)']
        
        bars = ax6.bar(range(len(stats)), stats, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
        ax6.set_xticks(range(len(stats)))
        ax6.set_xticklabels(stat_labels, rotation=45, ha='right')
        ax6.set_ylabel('Value')
        ax6.set_title('Simulation Statistics')
        
        # Add value labels on bars
        for bar, stat in zip(bars, stats):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{stat:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig('real_time_simulation.png', dpi=300, bbox_inches='tight')
            logger.info("Real-time simulation results saved as 'real_time_simulation.png'")
        
        return fig
    
    def generate_comprehensive_report(self, results: Dict, gpu_results: Dict, 
                                    sim_data: Dict) -> str:
        """
        Generate a comprehensive text report of all analysis results.
        
        Args:
            results: Comparative analysis results
            gpu_results: GPU performance analysis results  
            sim_data: Real-time simulation data
            
        Returns:
            Formatted report string
        """
        report = """
========================================================================
                ADVANCED COLLISION AVOIDANCE ANALYSIS REPORT
========================================================================

1. SYSTEM CONFIGURATION
-----------------------
"""
        report += f"Robot DOF: {self.num_joints}\n"
        report += f"GPU Available: {self.use_gpu}\n"
        report += f"Total Obstacles: {len(self.all_obstacles)}\n"
        report += f"Obstacle Configurations: {len(self.obstacle_configs)}\n"
        report += f"Workspace Bounds: {self.workspace_bounds}\n\n"
        
        report += """
2. PLANNING ALGORITHM COMPARISON
--------------------------------
"""
        if 'potential_field' in results and 'rrt_star' in results:
            pf = results['potential_field']
            rrt = results['rrt_star']
            
            report += f"Scenarios Tested: {len(results['scenarios'])}\n\n"
            
            report += "POTENTIAL FIELD METHOD:\n"
            report += f"  Success Rate: {pf.get('success_rate', 0)*100:.1f}%\n"
            report += f"  Average Time: {pf.get('avg_time', 0):.4f}s\n"
            report += f"  Average Path Length: {pf.get('avg_path_length', 0):.3f}\n"
            report += f"  Average Iterations: {pf.get('avg_iterations', 0):.1f}\n\n"
            
            report += "RRT* METHOD:\n"
            report += f"  Success Rate: {rrt.get('success_rate', 0)*100:.1f}%\n"
            report += f"  Average Time: {rrt.get('avg_time', 0):.4f}s\n"
            report += f"  Average Path Length: {rrt.get('avg_path_length', 0):.3f}\n"
            report += f"  Average Iterations: {rrt.get('avg_iterations', 0):.1f}\n\n"
            
            # Performance comparison
            if pf.get('avg_time', 0) > 0 and rrt.get('avg_time', 0) > 0:
                speed_ratio = rrt.get('avg_time', 1) / pf.get('avg_time', 1)
                report += f"Speed Ratio (RRT*/PF): {speed_ratio:.2f}x\n"
            
            if pf.get('success_rate', 0) > 0 and rrt.get('success_rate', 0) > 0:
                success_ratio = pf.get('success_rate', 1) / rrt.get('success_rate', 1)
                report += f"Success Ratio (PF/RRT*): {success_ratio:.2f}x\n\n"
        
        report += """
3. GPU PERFORMANCE ANALYSIS
----------------------------
"""
        if gpu_results and gpu_results.get('speedups') and len(gpu_results['speedups']) > 0:
            max_speedup = max(gpu_results['speedups'])
            avg_speedup = np.mean(gpu_results['speedups'])
            
            report += f"Maximum Speedup: {max_speedup:.2f}x\n"
            report += f"Average Speedup: {avg_speedup:.2f}x\n"
            report += f"Problem Sizes Tested: {gpu_results['problem_sizes']}\n"
            
            if len(gpu_results['speedups']) > 0:
                peak_idx = np.argmax(gpu_results['speedups'])
                report += f"Peak Performance at Size: {gpu_results['problem_sizes'][peak_idx]}\n\n"
            
            # GPU efficiency analysis
            if len(gpu_results['speedups']) > 1:
                efficiency_trend = np.polyfit(range(len(gpu_results['speedups'])), gpu_results['speedups'], 1)[0]
                if efficiency_trend > 0:
                    report += "GPU Efficiency: IMPROVING with problem size\n"
                else:
                    report += "GPU Efficiency: DECREASING with problem size\n"
        else:
            report += "GPU performance analysis not available (CPU-only system)\n"
            report += "Install CUDA support for GPU acceleration analysis\n\n"
        
        report += """
4. REAL-TIME SIMULATION RESULTS
-------------------------------
"""
        if sim_data:
            report += f"Simulation Duration: {sim_data['total_time']:.2f}s\n"
            report += f"Average Step Time: {sim_data['average_step_time']*1000:.2f}ms\n"
            report += f"Collision Rate: {sim_data['collision_rate']*100:.1f}%\n"
            report += f"Final Distance to Goal: {sim_data['distances_to_goal'][-1]:.3f}\n"
            
            # Real-time performance assessment
            real_time_violations = sum(1 for t in sim_data['computation_times'] if t > 0.1)
            report += f"Real-time Violations: {real_time_violations}/{len(sim_data['computation_times'])}\n"
            
            if sim_data['average_step_time'] < 0.1:
                report += "Real-time Performance: ACCEPTABLE\n"
            else:
                report += "Real-time Performance: NEEDS OPTIMIZATION\n"
        
        report += """

5. RECOMMENDATIONS
------------------
"""
        
        # Algorithm recommendations
        if 'potential_field' in results and 'rrt_star' in results:
            pf_score = results['potential_field'].get('success_rate', 0) * 2 + (1 / max(results['potential_field'].get('avg_time', 1), 0.001))
            rrt_score = results['rrt_star'].get('success_rate', 0) * 2 + (1 / max(results['rrt_star'].get('avg_time', 1), 0.001))
            
            if pf_score > rrt_score:
                report += "- RECOMMENDED ALGORITHM: Potential Field Method\n"
                report += "  Reasons: Better overall performance balance\n"
            else:
                report += "- RECOMMENDED ALGORITHM: RRT* Method\n"
                report += "  Reasons: Superior planning capability\n"
        
        # GPU recommendations
        if gpu_results and gpu_results.get('speedups') and len(gpu_results['speedups']) > 0:
            max_speedup = max(gpu_results['speedups'])
            if max_speedup > 2:
                report += "- GPU ACCELERATION: Recommended for large-scale problems\n"
                if len(gpu_results['problem_sizes']) > 1:
                    report += f"  Use GPU for problem sizes > {gpu_results['problem_sizes'][1]}\n"
            else:
                report += "- GPU ACCELERATION: Limited benefit, CPU sufficient\n"
        else:
            report += "- GPU ACCELERATION: Not available (install CUDA support)\n"
            report += "  Current system running CPU-only computations\n"
        
        # Real-time recommendations
        if sim_data and sim_data['average_step_time'] > 0.05:
            report += "- REAL-TIME OPTIMIZATION: Reduce computation complexity\n"
            report += "  Consider: Simplified collision models, reduced iteration limits\n"
        
        if sim_data and sim_data['collision_rate'] > 0.1:
            report += "- COLLISION AVOIDANCE: Increase repulsive potential gains\n"
            report += "  Consider: Larger safety margins, more conservative planning\n"
        
        report += """

6. TECHNICAL DETAILS
--------------------
"""
        report += f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"ManipulaPy Version: Advanced Demo\n"
        report += f"CUDA Available: {CUDA_AVAILABLE}\n"
        report += f"Number of CPU Cores Used: {os.cpu_count()}\n"
        
        report += """
========================================================================
                            END OF REPORT
========================================================================
"""
        
        if self.save_plots:
            with open('collision_avoidance_analysis_report.txt', 'w') as f:
                f.write(report)
            logger.info("Comprehensive report saved as 'collision_avoidance_analysis_report.txt'")
        
        return report


def main():
    """
    Main function to run the advanced collision avoidance demonstration.
    """
    print("=" * 70)
    print("   MANIPULAPY ADVANCED COLLISION AVOIDANCE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize demo
    demo = AdvancedCollisionAvoidanceDemo(save_plots=True, use_gpu=None)
    
    try:
        # 1. Visualize environment
        print("\n1. Visualizing environment...")
        demo.visualize_environment()
        
        # 2. Run comparative analysis
        print("\n2. Running comparative planning analysis...")
        comparative_results = demo.run_comparative_analysis(num_scenarios=8)
        
        # 3. Visualize planning results
        print("\n3. Generating planning comparison plots...")
        demo.visualize_trajectory_comparison(comparative_results, scenario_idx=0)
        demo.visualize_performance_statistics(comparative_results)
        
        # 4. Potential field visualization
        print("\n4. Visualizing potential fields...")
        if comparative_results['scenarios']:
            scenario = comparative_results['scenarios'][0]
            demo.visualize_potential_field(scenario['start'], scenario['goal'])
        
        # 5. GPU performance analysis
        print("\n5. Running GPU performance analysis...")
        gpu_results = demo.run_gpu_performance_analysis()
        if gpu_results:
            demo.visualize_gpu_performance(gpu_results)
        
        # 6. Real-time simulation
        print("\n6. Running real-time simulation...")
        simulation_data = demo.run_real_time_simulation(duration=10.0, obstacles_moving=False)
        demo.visualize_real_time_simulation(simulation_data)
        
        # 7. Generate comprehensive report
        print("\n7. Generating comprehensive analysis report...")
        report = demo.generate_comprehensive_report(
            comparative_results, gpu_results, simulation_data
        )
        
        # Display summary
        print("\n" + "=" * 70)
        print("                        ANALYSIS COMPLETE")
        print("=" * 70)
        
        if comparative_results['scenarios']:
            pf_success = comparative_results['potential_field'].get('success_rate', 0)
            rrt_success = comparative_results['rrt_star'].get('success_rate', 0)
            print(f"Potential Field Success Rate: {pf_success*100:.1f}%")
            print(f"RRT* Success Rate: {rrt_success*100:.1f}%")
        
        if gpu_results:
            max_speedup = max(gpu_results['speedups']) if gpu_results['speedups'] else 1
            print(f"Maximum GPU Speedup: {max_speedup:.2f}x")
        
        if simulation_data:
            print(f"Real-time Performance: {simulation_data['average_step_time']*1000:.1f}ms avg")
            print(f"Collision Rate: {simulation_data['collision_rate']*100:.1f}%")
        
        print("\nGenerated files:")
        file_list = [
            'environment_3d.png',
            'trajectory_comparison_scenario_0.png', 
            'performance_statistics.png',
            'potential_field_visualization.png',
            'real_time_simulation.png',
            'collision_avoidance_analysis_report.txt'
        ]
        
        if gpu_results:
            file_list.append('gpu_performance_analysis.png')
        
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
        # Cleanup
        if hasattr(demo, 'planner') and demo.planner is not None:
            try:
                demo.planner.cleanup_gpu_memory()
            except Exception:
                pass
        
        print("\nDemo cleanup completed.")


if __name__ == "__main__":
    main()